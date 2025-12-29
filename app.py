import io
import numpy as _np
from pathlib import Path
import pickle
from typing import Tuple, Any, Dict, List, Optional
import warnings
import datetime
import traceback
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, precision_score, confusion_matrix, average_precision_score, matthews_corrcoef
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
import math
import json
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import re
import os

pool_view = None

# ------------------------------------------------------------------
# Safe badge helper (used by pool_view before BestOf block defines it)
# ------------------------------------------------------------------
def get_match_character_badge(row: pd.Series) -> str:
    try:
        if bool(row.get("Danger_Flag", False)):
            return "DANGER"
        if bool(row.get("Trap_Flag", False)):
            return "TRAP"
        arch = str(row.get("Archetype", "") or "")
        if "Under" in arch or "LOW" in arch:
            return "COLD"
        if "Over" in arch or "HIGH" in arch:
            return "HOT"
        return "OK"
    except Exception:
        return ""

# ------------------------------------------------------------------
# CANONICAL DEFENSE HELPERS (single source of truth)
# ------------------------------------------------------------------
def _ensure_unique_index(df):
    import pandas as pd

    if df is None or not hasattr(df, "reset_index"):
        return pd.DataFrame()
    # her kritik pipeline başında tek standart
    return df.reset_index(drop=True)

def _align_mask(mask, df):
    import numpy as np
    import pandas as pd

    if df is None:
        return pd.Series([], dtype=bool)
    if len(df) == 0:
        return pd.Series([], index=df.index, dtype=bool)

    if isinstance(mask, pd.Series):
        return mask.reindex(df.index).fillna(False).astype(bool)

    arr = np.asarray(mask, dtype=bool)
    if arr.shape[0] != len(df):
        arr = np.resize(arr, len(df))
    return pd.Series(arr, index=df.index, dtype=bool)

def _dbg_to_array(val, n):
    import numpy as np
    import pandas as pd

    if n <= 0:
        return np.asarray([])

    if isinstance(val, pd.Series):
        arr = val.to_numpy()
    elif np.isscalar(val):
        arr = np.full(n, val)
    else:
        arr = np.asarray(val)

    if arr.shape[0] != n:
        arr = np.resize(arr, n)
    return np.asarray(arr).reshape(-1)

# ------------------------------------------------------------------
# JOURNAL DIRECTORY (single source of truth)
# ------------------------------------------------------------------
def _journal_dir() -> str:
    """Return absolute path to the journal folder (created if missing).
    Uses the directory of this script as base to avoid CWD surprises."""
#     try:  # AUTO-COMMENTED (illegal global try)
#         base = os.path.dirname(os.path.abspath(__file__))
#     except Exception:
#         base = os.path.abspath(os.getcwd())
#     path = os.path.join(base, "journal")
#     try:  # AUTO-COMMENTED (illegal global try)
#         os.makedirs(path, exist_ok=True)
#     except Exception:
#         pass
    # Robust: derive journal folder relative to app file if available
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    path = str(base / "journal")
    return path


# Global default to avoid NameError in conditional paths
quick_mode = False

# Global default to avoid NameError for WF odd-grid in conditional paths
odd_grid = ((None, None),)

# Global default to avoid NameError for open odds toggle in conditional paths
use_open_odds = True


# -----------------------------------------------------------------------------
# Robust datetime parsing (prevents losing early years due to locale formats)
# -----------------------------------------------------------------------------

def _robust_to_datetime(s: pd.Series) -> pd.Series:
    """Parse date-like series robustly.
    Tries common formats and both dayfirst True/False. Returns pandas datetime64[ns]."""
    if s is None:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce")
    # Convert to string where appropriate, keep NaN
    ss = s.copy()
    # normalize separators
#     try:  # AUTO-COMMENTED (illegal global try)
#         ss = ss.astype(str)
#         ss = ss.str.replace(r"\s+", " ", regex=True).str.strip()
#         ss = ss.str.replace(".", "-", regex=False)
#         ss = ss.str.replace("/", "-", regex=False)
#     except Exception:
#         pass
    # First pass: dayfirst=True (TR/EU)
    dt = pd.to_datetime(ss, errors="coerce", dayfirst=True)
    # Second pass: fill remaining with dayfirst=False
    if dt.isna().any():
        dt2 = pd.to_datetime(ss, errors="coerce", dayfirst=False)
        dt = dt.fillna(dt2)
    return dt


# --- UI helpers (AutoMod badges & styling) ---

# =============================================================
# AMQS helpers (MUST be defined before UI uses them)
# =============================================================

# -----------------------------------------------------------------------------
# Similarity config sync: ensure UI settings drive prediction blending (cache-safe)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Deterministic sorting helpers (prevents flickering when scores are near-tied)
# -----------------------------------------------------------------------------

def _ensure_tie_key(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a deterministic tie-breaker key column '_TIE' exists."""
    if df is None or df.empty:
        return df
    if "_TIE" in df.columns:
        return df
    if "MatchID_Unique" in df.columns:
        df["_TIE"] = df["MatchID_Unique"].astype(str)
        return df
    key_cols = [
    c for c in [
        "Date",
        "League",
        "HomeTeam",
        "AwayTeam",
         "Seçim"] if c in df.columns]
    if key_cols:
        s = df[key_cols].astype(str).fillna("")
        df["_TIE"] = s.agg("|".join, axis=1)
    else:
        df["_TIE"] = df.index.astype(str)
    return df


def _top_n_with_ties(
    df: pd.DataFrame,
    score_col: str,
    n: int,
     eps: float = 1e-4) -> pd.DataFrame:
    """Return top-n rows plus any rows tied within eps to the nth score (assumes df is already sorted DESC by score_col)."""
    import math
    if df is None or df.empty or n <= 0 or score_col not in df.columns:
        return df.head(max(n, 0)) if df is not None else df
    if len(df) <= n:
        return df
#     try:  # AUTO-COMMENTED (illegal global try)
#         nth_score = float(pd.to_numeric(
#             df.iloc[n - 1][score_col], errors="coerce"))
#     except Exception:
#         return df.head(n)
    nth_score = float(pd.to_numeric(df.iloc[n - 1][score_col], errors="coerce"))
    if not math.isfinite(nth_score):
        return df.head(n)
    thr = nth_score - float(eps)
    return df.loc[pd.to_numeric(df[score_col], errors="coerce") >= thr]


def _get_current_sim_cfg_from_session_state() -> dict:
    return {
        'alpha_max': float(st.session_state.get('SIM_ALPHA_MAX', 0.60)),
        'k': int(st.session_state.get('SIM_K', 20)),
        'use_cutoff': bool(st.session_state.get('SIM_USE_CUTOFF', True)),
        'min_similarity': float(st.session_state.get('SIM_MIN_SIM', 0.55)),
        'min_neighbors': int(st.session_state.get('SIM_MIN_NEIGHBORS', 10)),
        'same_league': bool(st.session_state.get('SIM_SAME_LEAGUE', True)),
        'max_pool': int(st.session_state.get('SIM_MAX_POOL', 60000)),
        'years_back': int(st.session_state.get('SIM_YEARS_BACK', 0)),
        'preset': str(st.session_state.get('SIM_PRESET', 'Auto (mevcut kolonlar)')),
        'k_ref': float(st.session_state.get('SIM_K_REF', 15.0)),
    }


def _sim_cfg_hash(cfg: dict) -> str:
    import json, hashlib
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def maybe_refresh_similarity_on_session_predictions() -> None:
    """Re-apply similarity blending when UI settings changed, even if disk-cache loaded."""
    if not st.session_state.get('processed', False):
        return
    if 'final_df' not in st.session_state or st.session_state.final_df is None:
        return
    if 'train_core' not in st.session_state or st.session_state.train_core is None:
        return
    cfg = _get_current_sim_cfg_from_session_state()
    h = _sim_cfg_hash(cfg)
    if st.session_state.get('SIM_CFG_HASH_USED') == h:
        return
#     try:  # AUTO-COMMENTED (illegal global try)
#         st.session_state.final_df = apply_similarity_anchor_to_predictions(
#             st.session_state.final_df,
#             st.session_state.train_core,
#             alpha_max=cfg['alpha_max'],
#             k=cfg['k'],
#             use_cutoff=cfg['use_cutoff'],
#             min_similarity=cfg['min_similarity'],
#             min_neighbors=cfg['min_neighbors'],
#             same_league=cfg['same_league'],
#             max_pool=cfg['max_pool'],
#             years_back=cfg['years_back'],
#             preset=cfg['preset'],
#             k_ref=cfg['k_ref'],
#         )
#         st.session_state['SIM_CFG_HASH_USED'] = h
#         st.session_state['SIM_CFG_USED'] = cfg
#     except Exception as _e:
#         # fail-safe: never crash app because of similarity refresh
#         st.session_state['SIM_CFG_HASH_USED'] = h
#         st.session_state['SIM_CFG_USED'] = cfg
#         return


# -----------------------------------------------------------------------------
# Market Derivation Layer (MDL) - lightweight profile + controlled score boost
# - No market switching here. Only produces PROFILE_CONF / MDL_BOOST and
#   GoldenScore_MDL (for ranking), keeping base predictions intact.
# -----------------------------------------------------------------------------
def _poisson_over_probs(lmb: float):
    """Return (P_O15, P_O25, P_O35) using Poisson(lambda=lmb) on total goals.
    O15 = goals >=2, O25 = goals >=3, O35 = goals >=4.
    """
#     try:  # AUTO-COMMENTED (illegal global try)
#         l = float(lmb)
#     except Exception:
#         return (None, None, None)
    try:
        l = float(lmb)
    except Exception:
        l = float("nan")
    if not np.isfinite(l) or l <= 0:
        return (None, None, None)
    # P(X<=k) = e^-l * sum_{i=0..k} l^i/i!
    e = float(np.exp(-l))
    p_le_1 = e * (1.0 + l)
    p_le_2 = e * (1.0 + l + (l * l) / 2.0)
    p_le_3 = e * (1.0 + l + (l * l) / 2.0 + (l * l * l) / 6.0)
    p_o15 = float(1.0 - p_le_1)
    p_o25 = float(1.0 - p_le_2)
    p_o35 = float(1.0 - p_le_3)
    return (
    np.clip(
        p_o15, 0.0, 1.0), np.clip(
            p_o25, 0.0, 1.0), np.clip(
                p_o35, 0.0, 1.0))


def _poisson_quantile(lam: float, q: float, max_k: int = 15) -> float:
    """Approximate Poisson quantile for small k (stable, no scipy).
    Returns the smallest k such that P(X <= k) >= q.
    """
#     try:  # AUTO-COMMENTED (illegal global try)
#         lam = float(lam)
#     except Exception:
#         return float("nan")
    try:
        lam = float(lam)
    except Exception:
        return float("nan")
    if not np.isfinite(lam) or lam < 0:
        return float("nan")
    q = float(q)
    if q <= 0:
        return 0.0
    if q >= 1:
        return float(max_k)
    # PMF(0)
    p = np.exp(-lam)
    cdf = p
    if cdf >= q:
        return 0.0
    for k in range(1, max_k + 1):
        # recursive pmf
        p = p * lam / k
        cdf += p
        if cdf >= q:
            return float(k)
    return float(max_k)


def add_mdl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MDL numeric profile columns + controlled GoldenScore boost.
    Safe: if required base cols missing, produces neutral defaults.
    Columns added:
      - P_1X, P_2X
      - P_O15, P_O25, P_O35 (from xG total Poisson, if xG columns exist)
      - PROFILE_CONF (0..1)
      - MDL_BOOST (0..0.10)
      - GoldenScore_MDL (0..1) = GoldenScore * (1 + MDL_BOOST)
    """
    if df is None or getattr(df, "empty", True):
        return df
    d = df.copy()

    # defaults for new MDL+KNN profile columns (filled later if possible)
    if "DRAW_PRESSURE" not in d.columns:
        d["DRAW_PRESSURE"] = np.nan

    # --- 1X / 2X from similarity probabilities (if present)
    p1 = pd.to_numeric(d.get("SIM_P1", np.nan), errors="coerce")
    px = pd.to_numeric(d.get("SIM_PX", np.nan), errors="coerce")
    p2 = pd.to_numeric(d.get("SIM_P2", np.nan), errors="coerce")
    d["P_1X"] = (p1.fillna(0) + px.fillna(0)).clip(0.0, 1.0)
    d["P_2X"] = (p2.fillna(0) + px.fillna(0)).clip(0.0, 1.0)

    # --- Goal tail probabilities from xG total (best effort)
    xg_h = None
    xg_a = None
    for c in ["xG_Home", "xGHome", "FT_xG_Home", "XG_Home"]:
        if c in d.columns:
            xg_h = c
            break
    for c in ["xG_Away", "xGAway", "FT_xG_Away", "XG_Away"]:
        if c in d.columns:
            xg_a = c
            break

    d["P_O15"] = np.nan
    d["P_O25"] = np.nan
    d["P_O35"] = np.nan

    if xg_h and xg_a:
        if "TG_Q75" not in d.columns:
            d["TG_Q75"] = np.nan
        if "TG_Q90" not in d.columns:
            d["TG_Q90"] = np.nan
        lam = pd.to_numeric(d[xg_h], errors="coerce").fillna(
            0) + pd.to_numeric(d[xg_a], errors="coerce").fillna(0)
        lam = lam.clip(lower=0.05)
        probs = lam.apply(_poisson_over_probs)
        d["P_O15"] = probs.apply(
    lambda t: t[0] if isinstance(
        t, tuple) else np.nan)
        d["P_O25"] = probs.apply(
    lambda t: t[1] if isinstance(
        t, tuple) else np.nan)
        d["P_O35"] = probs.apply(
    lambda t: t[2] if isinstance(
        t, tuple) else np.nan)

        # --- Poisson-based total-goals quantiles (proxy for tail / variance)
        d["TG_Q75"] = lam.apply(lambda lam: _poisson_quantile(lam, 0.75))
        d["TG_Q90"] = lam.apply(lambda lam: _poisson_quantile(lam, 0.90))
    else:
        try:
            st.session_state.setdefault("_dbg_pool_stages", {})
            st.session_state["_dbg_pool_stages"]["tg_compute_skipped_reason"] = "missing_xg_inputs"
        except Exception:
            pass

    # --- PROFILE_CONF: single numeric profile quality (0..1)
    primary = d.get("Seçim", "")
    # normalize primary market labels
    prim = primary.astype(str).str.upper()

    sim_p1 = p1.fillna(0).clip(0, 1)
    sim_p2 = p2.fillna(0).clip(0, 1)
    sim_px = px.fillna(0).clip(0, 1)

    # --- Draw pressure: how strong draw is relative to winner side (0..inf, we clip later)
    denom = pd.concat([sim_p1, sim_p2], axis=1).max(axis=1).replace(0, np.nan)
    d["DRAW_PRESSURE"] = (sim_px /
    (denom.fillna(1e-6))).replace([np.inf, -
    np.inf], np.nan).fillna(0.0).clip(0.0, 1.5)

    p1x = d["P_1X"].fillna(0).clip(0, 1)
    p2x = d["P_2X"].fillna(0).clip(0, 1)
    po25 = pd.to_numeric(
    d.get(
        "P_O25",
        np.nan),
        errors="coerce").fillna(0).clip(
            0,
             1)
    po35 = pd.to_numeric(
    d.get(
        "P_O35",
        np.nan),
        errors="coerce").fillna(0).clip(
            0,
             1)
    po15 = pd.to_numeric(
    d.get(
        "P_O15",
        np.nan),
        errors="coerce").fillna(0).clip(
            0,
             1)

    profile_conf = pd.Series(0.5, index=d.index, dtype=float)

    # MS markets
    is_ms1 = prim.isin(["MS1", "1", "HOME", "1H", "H"]) | prim.str.contains(
        r"\bMS\s*1\b") | prim.str.contains("MS 1") | prim.str.contains("EV SAH")
    is_ms2 = prim.isin(["MS2", "2", "AWAY", "A"]) | prim.str.contains(
        r"\bMS\s*2\b") | prim.str.contains("MS 2") | prim.str.contains("DEP")
    is_msx = prim.isin(["MSX", "X", "DRAW", "D"]) | prim.str.contains(
        r"\bMS\s*X\b") | prim.str.contains("MS X") | prim.str.contains("BERABERE")
    profile_conf.loc[is_ms1] = (0.6 * sim_p1 + 0.4 * p1x).loc[is_ms1]
    profile_conf.loc[is_ms2] = (0.6 * sim_p2 + 0.4 * p2x).loc[is_ms2]
    profile_conf.loc[is_msx] = (
        0.7 * sim_px + 0.3 * (1.0 - (sim_p1 - sim_p2).abs().clip(0, 1))).loc[is_msx]

    # OU markets
    is_ou_over = (prim.str.contains("O25") | prim.str.contains("OU25") | prim.str.contains(
        "OVER") | prim.str.contains("ÜST") | prim.str.contains("UST"))
    is_ou_under = (prim.str.contains("U25") | prim.str.contains(
        "ALT") | prim.str.contains("UNDER"))
    profile_conf.loc[is_ou_over] = (0.6 * po25 + 0.4 * po35).loc[is_ou_over]
    profile_conf.loc[is_ou_under] = (
        0.6 * (1.0 - po25) + 0.4 * (1.0 - po15)).loc[is_ou_under]

    # BTTS markets (KG)
    is_btts = prim.str.contains("KG") | prim.str.contains("BTTS")
    # robust BTTS probability column (if present); else fallback to OU25 proxy
    _btts_candidates = [
    c for c in d.columns if str(c).upper() in [
        "SIM_PB",
        "SIM_PBTS",
        "SIM_PBTTS",
        "SIM_PBTT",
        "SIM_PBTTS_YES",
        "SIM_PBTTSYES",
        "SIM_PBTTSYES_PROB",
        "SIM_PB_Y",
         "SIM_PBYES"]]
    if _btts_candidates:
        btts_p = pd.to_numeric(d[_btts_candidates[0]],
                               errors="coerce").fillna(0).clip(0, 1)
    else:
        btts_p = po25
    profile_conf.loc[is_btts] = (0.7 * btts_p + 0.3 * po25).loc[is_btts]

    d["PROFILE_CONF"] = profile_conf.clip(0.0, 1.0)

    # --- Controlled boost (hard-capped)
    boost = pd.Series(0.0, index=d.index, dtype=float)

    # MS boost using double-chance bias (max +8%)
    ms1_boost = ((p1x - 0.78) / 0.10).clip(0, 1) * 0.08
    ms2_boost = ((p2x - 0.78) / 0.10).clip(0, 1) * 0.08
    boost.loc[is_ms1] = ms1_boost.loc[is_ms1]
    boost.loc[is_ms2] = ms2_boost.loc[is_ms2]

    # OU over boost using 3.5 tail (max +10%)
    ou_boost = ((po35 - 0.40) / 0.20).clip(0, 1) * 0.10
    boost.loc[is_ou_over] = np.maximum(
    boost.loc[is_ou_over],
     ou_boost.loc[is_ou_over])

    # --- Quality gate: do not let low-quality similarity distort ranking
    sim_q = pd.to_numeric(d.get("SIM_QUALITY", np.nan),
                          errors="coerce").fillna(0.0)
    eff_n = pd.to_numeric(d.get("EFFECTIVE_N", np.nan),
                          errors="coerce").fillna(0.0)
    q_gate = (((sim_q - 0.32) / 0.10).clip(0, 1) *
              ((eff_n - 15) / 10).clip(0, 1)).astype(float)

    # --- Modifiers (small, controlled): draw pressure & goal tail
    dp = pd.to_numeric(
    d.get(
        "DRAW_PRESSURE",
        np.nan),
        errors="coerce").fillna(0.0).clip(
            0.0,
             1.0)
    # MS picks: if draw pressure high, prefer safer profile -> slight boost
    # (still capped)
    boost.loc[is_ms1 | is_ms2] = boost.loc[is_ms1 |
        is_ms2] + (dp.loc[is_ms1 | is_ms2] * 0.02)
    # OU Over picks: if 90th percentile goals high, gollü tail -> slight boost
    tg_q90 = pd.to_numeric(d.get("TG_Q90", np.nan), errors="coerce")
    tail_bonus = (tg_q90 >= 5.0).astype(float) * 0.02
    boost.loc[is_ou_over] = boost.loc[is_ou_over] + \
        tail_bonus.loc[is_ou_over].fillna(0.0)

    # apply gate + cap
    d["MDL_BOOST"] = (boost * q_gate).clip(0.0, 0.10)

    # --- Apply boost to GoldenScore (keep base GoldenScore unchanged)
    if "GoldenScore" in d.columns:
        gs = pd.to_numeric(d["GoldenScore"], errors="coerce").fillna(0.0)
        d["GoldenScore_MDL"] = (gs * (1.0 + d["MDL_BOOST"])).clip(0.0, 1.0)
    else:
        d["GoldenScore_MDL"] = np.nan

    return d
# 📌 (B) Helper function for Z-Score calculation


def _safe_z(series: pd.Series) -> pd.Series:
    """Sabit / tekdüze serilerde patlamayan z-score hesaplayıcı."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.std(ddof=0) == 0 or s.std(ddof=0) == 0.0 or s.count() < 2:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

safe_z_score = _safe_z


def ensure_bestofrank(df: pd.DataFrame) -> pd.DataFrame:
    """BestOfRank'i **tek sefer** (yüklenen veri evreni üzerinde) üret.

    Önemli: Bu fonksiyon filtrelenmiş alt-kümelerde tekrar çalıştırılırsa z-score ve percentile'lar değişir.
    Bu yüzden 'pool_bets' üzerinde 1 kez çalıştırıp, daha sonra sadece filtrelemek gerekir.
    """
    if df is None or len(df) == 0:
        return df
    has_bestofrank = "BestOfRank" in df.columns

    _df = df.copy()

    # Gerekli kolonlar yoksa default değerlerle tamamla
    init_cols = [
        ("GoldenScore", 0.0),
        ("GoldenScore_MDL", 0.0),
        ("MDL_BOOST", 0.0),
        ("PROFILE_CONF", 0.5),
        ("DRAW_PRESSURE", 0.0),
        ("TG_Q75", np.nan),
        ("TG_Q90", np.nan),
        ("P_1X", 0.0),
        ("P_2X", 0.0),
        ("P_O15", 0.0),
        ("P_O25", 0.0),
        ("P_O35", 0.0),
        ("League_Conf", 0.5),
        ("Market_Conf_Score", 0.5),
        ("Score", 0.0),
        ("Prob", 0.0),
    ]
    for c, d in init_cols:
        if c not in _df.columns:
            _df[c] = d

    frames = []
    for market_name, sub in _df.groupby("Seçim", dropna=False):
        # Market -> weight anahtar eşlemesi (geriye dönük uyumlu)
        if market_name in ["MS 1", "MS X", "MS 2"]:
            w = RANK_WEIGHTS_BY_MARKET.get(
                "1X2", RANK_WEIGHTS_BY_MARKET.get("MS", {}))
        elif market_name in ["2.5 Üst", "2.5 Alt"]:
            w = RANK_WEIGHTS_BY_MARKET.get("OU", {})
        else:
            # KG / BTTS
            w = RANK_WEIGHTS_BY_MARKET.get(
                "KG", RANK_WEIGHTS_BY_MARKET.get("BTTS", {}))

        sub["_z_score"] = _safe_z(sub["Score"])
        sub["_z_golden"] = _safe_z(sub["GoldenScore"])
        sub["_z_prob"] = _safe_z(sub["Prob"])

        league_term = (
    pd.to_numeric(
        sub["League_Conf"],
         errors="coerce").fillna(0.5).astype(float) - 0.5) * 2.0
        market_term = (
    pd.to_numeric(
        sub["Market_Conf_Score"],
         errors="coerce").fillna(0.5).astype(float) - 0.5) * 2.0

        sub["BestOfRank"] = (
            float(w.get("score", 0.0)) * sub["_z_score"] +
            float(w.get("golden", 0.0)) * sub["_z_golden"] +
            float(w.get("prob", 0.0)) * sub["_z_prob"] +
            float(w.get("league", 0.0)) * league_term +
            float(w.get("market", 0.0)) * market_term
        )
        sub.drop(
    columns=[
        "_z_score",
        "_z_golden",
        "_z_prob"],
        inplace=True,
         errors="ignore")
        frames.append(sub)

    return pd.concat(frames, axis=0) if frames else _df


# -----------------------------------------------------------------------------#
# Helper: ensure implied/model-vs-market/trap columns exist on any view dataframe
# -----------------------------------------------------------------------------#
def _ensure_trap_cols(_df: pd.DataFrame) -> pd.DataFrame:
    if _df is None:
        return pd.DataFrame()
    if len(_df) == 0:
        return _df
    # Robust Prob -> Prob_dec
    if "Prob_dec" not in _df.columns:
        pr = _df.get("Prob", np.nan)
        pr_num = pd.to_numeric(
            pr.astype(str).str.replace(
    "%", "", regex=False).str.replace(
        ",", ".", regex=False).str.strip(),
            errors="coerce"
        )
        _df["Prob_dec"] = np.where(pr_num > 1.0, pr_num / 100.0, pr_num)
    # Robust Odd -> Odd_num
    if "Odd_num" not in _df.columns:
        od = _df.get("Odd", np.nan)
        od_num = pd.to_numeric(
            od.astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
        _df["Odd_num"] = od_num
    if "Implied_Prob" not in _df.columns or _df["Implied_Prob"].isna().all():
        od_num = pd.to_numeric(
    _df.get(
        "Odd_num",
        _df.get(
            "Odd",
            np.nan)),
             errors="coerce")
        _df["Implied_Prob"] = np.where(od_num > 0, 1.0 / od_num, np.nan)
    if "Model_vs_Market" not in _df.columns or _df["Model_vs_Market"].isna(
    ).all():
        _df["Model_vs_Market"] = pd.to_numeric(
            _df["Prob_dec"], errors="coerce") - pd.to_numeric(_df["Implied_Prob"], errors="coerce")

    # Trap logic (works for MS / OU / KG)
    _imp = pd.to_numeric(_df["Implied_Prob"], errors="coerce").fillna(0.0)
    _mvm = pd.to_numeric(_df["Model_vs_Market"], errors="coerce").fillna(0.0)
    _sel = _df.get("Seçim", "").astype(str)

    base_trap = (_imp >= 0.60) & (_mvm <= -0.06)
    ms_trap = (_sel.isin(["MS 1", "MS 2"])) & (_imp >= 0.62) & (_mvm <= -0.05)

    conf_hi = _df.get("CONF_Status", "").astype(
        str).str.upper().isin(["HIGH", "GREEN"])
    am_hi = _df.get("AutoMod_Status", "").astype(
        str).str.upper().isin(["HIGH", "GREEN"])
    stars_hi = pd.to_numeric(
    _df.get(
        "Star_Rating",
        0),
         errors="coerce").fillna(0) >= 3
    _df["Trap_Flag"] = (base_trap | ms_trap)
    _df["Trap_Type"] = np.where(
    ms_trap, "TRAP_FAV_MS", np.where(
        base_trap, "TRAP_MARKET_OVERCONF", ""))
    _df["TRAP_ICON"] = np.where(_df["Trap_Flag"].astype(bool), "⚠️", "")

    # EV_pure_pct (market-agnostic, derived from Prob & Odd)
    if "EV_pure_pct" not in _df.columns or _df["EV_pure_pct"].isna().all():
        _p = pd.to_numeric(_df.get("Prob_dec", np.nan), errors="coerce")
        _o = pd.to_numeric(
    _df.get(
        "Odd_num",
        _df.get(
            "Odd",
            np.nan)),
             errors="coerce")
        _df["EV_pure_pct"] = (_p * _o - 1.0) * 100.0

    # Dangerous label: "en tehlikeli maçlar" (strong signal + trap / market
    # disagreement)
    _conf_pct = pd.to_numeric(
    _df.get(
        "CONF_percentile",
        np.nan),
         errors="coerce")
    _amqs_pct = pd.to_numeric(
    _df.get(
        "AMQS_percentile",
        np.nan),
         errors="coerce")
    _stars = pd.to_numeric(_df.get("Star_Rating", 0),
                           errors="coerce").fillna(0)

    # Dangerous label: "en tehlikeli maçlar"
    # Goal: NOT the same as Trap. Danger is a *subset* of Trap where the favorite is
    # both (a) aggressively priced by the market and (b) our model is meaningfully BELOW market,
    # and (c) confidence looks high (a typical "looks too easy" profile).
    _conf_pct = pd.to_numeric(
    _df.get(
        "CONF_percentile",
        np.nan),
         errors="coerce").fillna(0)
    _amqs_pct = pd.to_numeric(
    _df.get(
        "AMQS_percentile",
        np.nan),
         errors="coerce").fillna(0)
    _stars = pd.to_numeric(_df.get("Star_Rating", 0),
                           errors="coerce").fillna(0)

    _imp2 = pd.to_numeric(_df.get("Implied_Prob", 0),
                          errors="coerce").fillna(0.0)
    _mvm2d = pd.to_numeric(
    _df.get(
        "Model_vs_Market",
        0),
         errors="coerce").fillna(0.0)
    _odd2 = pd.to_numeric(
    _df.get(
        "Odd_num",
        _df.get(
            "Odd",
            np.nan)),
            errors="coerce").fillna(
                np.nan)
    _evp2 = pd.to_numeric(_df.get("EV_pure_pct", 0),
                          errors="coerce").fillna(0.0)

    # Market-type sensitivity (favorites / totals / btts behave differently)
    _is_1x2 = _sel.str.contains("^MS", regex=True)
    _is_ou = _sel.str.contains(r"2\.5|3\.5|1\.5|4\.5", regex=True)
    _is_kg = _sel.str.contains("KG", case=False)

    # Conservative thresholds:
    # - implied is high (market says "banko")
    # - our model is notably below market (negative model-vs-market)
    # - odds are short (book is confident)
    # - still EV_pure not great (usually negative / thin)
    _danger_core = (
        (_imp2 >= 0.62)
        & (_mvm2d <= -0.07)
        & (_odd2 <= np.where(_is_1x2, 1.70, 1.85))
        & (_evp2 <= -2.0)
    )

    # Require "looks great" layer: high conf OR high AMQS OR high stars
    _looks_easy = ((_conf_pct >= 90) | (_amqs_pct >= 90) | (_stars >= 4))

    _danger = _df["Trap_Flag"].astype(bool) & _danger_core & _looks_easy

    _df["Danger_Flag"] = _danger
    _df["DANGER_ICON"] = np.where(_danger, "☠️", "")

    # Archetype (fallback for upcoming-only data): make it non-flat &
    # market-agnostic
    if ("Archetype" not in _df.columns) or (
        _df["Archetype"].astype(str).str.upper().eq("NONE").all()):
        _sel = _df.get("Seçim", "").astype(str).fillna("")
        _prefix = np.where(_sel.str.contains("KG", case=False), "BTTS_",
                           np.where(_sel.str.contains(r"2\.5|3\.5|1\.5|4\.5", regex=True), "OU_",
                           np.where(_sel.str.contains("^MS", regex=True), "1X2_", "MK_")))

        _mvm2 = pd.to_numeric(
    _df.get(
        "Model_vs_Market",
        0),
         errors="coerce").fillna(0)
        _evp = pd.to_numeric(_df.get("EV_pure_pct", 0),
                             errors="coerce").fillna(0)

        _arch = np.full(len(_df), "NONE", dtype=object)
        _arch = np.where(_df["Trap_Flag"].astype(bool), "TRAP", _arch)
        _arch = np.where(
    (~_df["Trap_Flag"].astype(bool)) & (
        _mvm2 >= 0.06) & (
            _evp >= 5),
            "MODEL_EDGE",
             _arch)
        _arch = np.where(
    (~_df["Trap_Flag"].astype(bool)) & (
        _mvm2 <= -
        0.06) & (
            _evp <= -
            5),
            "MARKET_OVERPRICE",
             _arch)
        _arch = np.where(
    (_arch == "NONE") & (
        np.abs(_mvm2) <= 0.02) & (
            np.abs(_evp) <= 2),
            "NO_EDGE",
             _arch)
        _arch = np.where((_arch == "NONE") & (_danger), "DANGER_TRAP", _arch)

        _df["Archetype"] = (_prefix.astype(object) + _arch.astype(object))
    # Archetype icon (quick scan without opening "gelişmiş")
    # Examples: OU_NO_EDGE -> 🟦 , 1X2_TRAP_* -> 🪤 , *_MODEL_EDGE -> ✅
    #     try:  # AUTO-COMMENTED (illegal global try)
    #         _archs = _df.get("Archetype", "").astype(str).fillna("")
    _archs = _df.get("Archetype", "").astype(str).fillna("")

    _df["ARCH_ICON"] = np.select(
        [
            _archs.str.contains("TRAP", case=False, regex=False),
            _archs.str.contains("DANGER", case=False, regex=False),
            _archs.str.contains("MODEL_EDGE", case=False, regex=False),
            _archs.str.contains("MARKET_OVERPRICE", case=False, regex=False),
            _archs.str.contains("NO_EDGE", case=False, regex=False),
        ],
        ["🪤", "☠️", "✅", "💚", "🟦"],
        default=""
    )
    # -----------------------------------------------------------
    # MDL Rank / Quantile (snapshot inside pool universe)
    # -----------------------------------------------------------

    _mdl = pd.to_numeric(
        _df.get("GoldenScore_MDL", np.nan),
        errors="coerce"
    )

    grp_cols = [c for c in ["League", "Seçim"] if c in _df.columns]

    if grp_cols:
        _g = _df.groupby(grp_cols, dropna=False)

        _df["MDL_quantile"] = _g["GoldenScore_MDL"].transform(
            lambda s: pd.to_numeric(s, errors="coerce")
            .rank(pct=True, method="average")
        )

        _df["MDL_rank"] = _g["GoldenScore_MDL"].transform(
            lambda s: pd.to_numeric(s, errors="coerce")
            .rank(ascending=False, method="min")
        )
    else:
        _df["MDL_quantile"] = _mdl.rank(pct=True, method="average")
        _df["MDL_rank"] = _mdl.rank(ascending=False, method="min")

    return _df if isinstance(_df, pd.DataFrame) else pd.DataFrame()


REQUIRED_VIEW_COLS = [
    "Match_ID", "Date", "League", "HomeTeam", "AwayTeam", "Seçim", "Odd", "Prob",
    "EV", "PROFILE_CONF", "DRAW_PRESSURE", "TG_Q75", "TG_Q90",
    "SIM_ANCHOR_STRENGTH", "EFFECTIVE_N", "_EN_PEN", "CONF_ICON",
    "AUTOMOD_ICON", "Karakter",
]

# ------------------------------------------------------------------
# Column audit + lineage helpers (debug-only)
# ------------------------------------------------------------------
DBG_COL_AUDIT_KEY = "DEBUG_COL_AUDIT"


def _dbg_col_audit_enabled() -> bool:
    return bool(st.session_state.get(DBG_COL_AUDIT_KEY, False))


ADV_COLS_TRACKED = [
    # SIM
    "SIM_ANCHOR_STRENGTH", "SIM_ANCHOR_GROUP", "SIM_ALPHA", "SIM_QUALITY",
    "SIM_POver", "SIM_PBTTS", "SIM_MS_STRENGTH", "SIM_OU_STRENGTH",
    "SIM_BTTS_STRENGTH",
    # TG
    "TG_Q75", "TG_Q90",
    # Model probs
    "P_Home_Model", "P_Draw_Model", "P_Away_Model", "P_Over_Model",
    "P_BTTS_Model",
    # Blend/gates
    "BLEND_MODE_MS", "LEAGUE_OK_MS", "BLEND_W_LEAGUE_MS",
    "BLEND_MODE_OB", "LEAGUE_OK_OB", "BLEND_W_LEAGUE_OB",
    # Core
    "Prob", "EV", "Score", "GoldenScore", "Final_Confidence", "AMQS",
    "CONF_percentile", "AMQS_percentile", "AutoMod_Status",
    # Hygiene
    "EFFECTIVE_N",
    # Ranking
    "BestOfRank",
]

NECESSITY_BUCKETS = {
    "MUST-HAVE": {
        "Prob": "Min prob gate + ranking inputs.",
        "EV": "Min EV gate + EV top-k selection.",
        "Score": "Min score gate + ranking inputs.",
        "GoldenScore": "Ranking input for BestOfRank.",
        "SIM_ANCHOR_STRENGTH": "BestOf eligibility gate + strength labeling.",
        "SIM_QUALITY": "KNN gate + reliability filters.",
        "EFFECTIVE_N": "KNN gate + reliability filters.",
        "BestOfRank": "BestOf selection/ranking backbone.",
    },
    "SHOULD-HAVE": {
        "Final_Confidence": "Used in dedup/sort + display.",
        "AMQS": "AutoMod quality core; used for percentiles.",
        "CONF_percentile": "UI filters / gating sliders.",
        "AMQS_percentile": "UI filters / gating sliders.",
        "AutoMod_Status": "UI filters + badge logic.",
    },
    "EXPLAINABILITY": {
        "SIM_ANCHOR_GROUP": "Diagnostic grouping for similarity.",
        "SIM_ALPHA": "Similarity blending diagnostic.",
        "SIM_POver": "Character badge context (OU).",
        "SIM_PBTTS": "Character badge context (BTTS).",
        "SIM_MS_STRENGTH": "UI/diagnostic strength signal.",
        "SIM_OU_STRENGTH": "UI/diagnostic strength signal.",
        "SIM_BTTS_STRENGTH": "UI/diagnostic strength signal.",
        "TG_Q75": "Profile context (goal tail).",
        "TG_Q90": "Profile context (goal tail).",
        "P_Home_Model": "Model component visibility.",
        "P_Draw_Model": "Model component visibility.",
        "P_Away_Model": "Model component visibility.",
        "P_Over_Model": "Model component visibility.",
        "P_BTTS_Model": "Model component visibility.",
        "BLEND_MODE_MS": "Blend diagnostics.",
        "LEAGUE_OK_MS": "Blend diagnostics.",
        "BLEND_W_LEAGUE_MS": "Blend diagnostics.",
        "BLEND_MODE_OB": "Blend diagnostics.",
        "LEAGUE_OK_OB": "Blend diagnostics.",
        "BLEND_W_LEAGUE_OB": "Blend diagnostics.",
    },
}


def _col_audit(df: pd.DataFrame, name: str, cols: list) -> dict:
    """Return a safe, compact column audit summary for df."""
    out = {"name": name, "rows": 0, "cols_total": 0, "cols": {}}
    if df is None or not hasattr(df, "columns"):
        return out
    try:
        out["rows"] = int(len(df))
        out["cols_total"] = int(len(df.columns))
        for c in cols:
            if c not in df.columns:
                out["cols"][c] = {"present": False}
                continue
            s = df[c]
            null_ratio = float(s.isna().mean()) if len(s) else 1.0
            nunique = int(s.nunique(dropna=True)) if len(s) else 0
            sample = None
            try:
                idx = s.first_valid_index()
                if idx is not None:
                    sample = s.loc[idx]
            except Exception:
                sample = None
            if sample is not None:
                sample = str(sample)
                if len(sample) > 120:
                    sample = sample[:120] + "..."
            out["cols"][c] = {
                "present": True,
                "null_ratio": round(null_ratio, 4),
                "nunique": nunique,
                "sample": sample,
            }
    except Exception:
        return out
    return out


def _record_col_lineage(df: pd.DataFrame, stage: str, obj_name: str) -> None:
    """Track column deltas + all-NaN transitions for tracked columns."""
    if df is None or not hasattr(df, "columns"):
        return
    cols = set(df.columns)
    if "_dbg_col_lineage" not in st.session_state:
        st.session_state["_dbg_col_lineage"] = []
    state = st.session_state.get("_dbg_col_lineage_state", {})
    prev_cols = set(state.get(obj_name, {}).get("cols", []))
    prev_nonnull = state.get(obj_name, {}).get("nonnull", {})

    added = sorted(cols - prev_cols)
    removed = sorted(prev_cols - cols)

    all_nan = []
    became_all_nan = []
    disappeared = []
    nonnull = {}
    for c in ADV_COLS_TRACKED:
        if c not in df.columns:
            if c in prev_cols:
                disappeared.append(c)
            continue
        s = df[c]
        is_all_nan = bool(s.isna().all())
        all_nan.append(c) if is_all_nan else None
        nonnull[c] = not is_all_nan
        if prev_nonnull.get(c, None) is True and is_all_nan:
            became_all_nan.append(c)

    entry = {
        "object": obj_name,
        "stage": stage,
        "rows": int(len(df)),
        "cols_total": int(len(df.columns)),
        "added": added,
        "removed": removed,
        "all_nan_tracked": sorted(all_nan),
        "became_all_nan": sorted(became_all_nan),
        "disappeared": sorted(disappeared),
    }
    st.session_state["_dbg_col_lineage"].append(entry)
    state[obj_name] = {"cols": sorted(cols), "nonnull": nonnull}
    st.session_state["_dbg_col_lineage_state"] = state


def _store_col_audit(df: pd.DataFrame, name: str) -> None:
    if not _dbg_col_audit_enabled():
        return
    if "_dbg_col_audits" not in st.session_state:
        st.session_state["_dbg_col_audits"] = {}
    st.session_state["_dbg_col_audits"][name] = _col_audit(
        df, name, ADV_COLS_TRACKED)


def _maybe_record_col_lineage(df: pd.DataFrame, stage: str, obj_name: str) -> None:
    if not _dbg_col_audit_enabled():
        return
    try:
        _record_col_lineage(df, stage, obj_name)
    except Exception:
        pass


def _track_view_error(where: str) -> None:
    st.session_state["_last_view_error"] = traceback.format_exc()
    st.session_state["_last_view_error_where"] = where


def _safe_view_cols(df: pd.DataFrame, required_cols: list, tag: str = "generic") -> list:
    if df is None:
        return []
    safe_cols = [c for c in required_cols if c in df.columns]
    missing = sorted(set(required_cols) - set(safe_cols))
    if missing:
        key = f"_dbg_missing_cols_{tag}"
        st.session_state[key] = sorted(set(missing))
    return safe_cols if safe_cols else list(df.columns)


def _ensure_view_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols] if cols else out


def _style_df_for_view(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if getattr(df, "empty", True):
        return df
    return _ensure_view_cols(df, required_cols)


def ensure_amqs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee AMQS columns exist and provide a non-flat fallback when history-join is empty.
    SAFE: never raises, never hard-filters.
    """
    if df is None or getattr(df, "empty", True):
        return df

    # Ensure required columns exist
    if "AMQS" not in df.columns:
        df["AMQS"] = 0.50
    if "AMQS_percentile" not in df.columns:
        df["AMQS_percentile"] = 0.50

    # Detect flat AMQS (common when amqs_df join is missing/empty)
#     try:  # AUTO-COMMENTED (illegal global try)
#         amqs_num = pd.to_numeric(df["AMQS"], errors="coerce")
#         flat = amqs_num.dropna().round(6).nunique() <= 1
#     except Exception:
#         flat = True
    # guard: AMQS may not exist in some builds/exports
    try:
        amqs_series = df.get("AMQS", pd.Series(dtype=float))
        amqs_num = pd.to_numeric(amqs_series, errors="coerce")
        flat = (amqs_num.dropna().round(6).nunique() <= 1) if len(amqs_num) else True
    except Exception:
        flat = True

    if flat:
        # Derive from confidence columns if available (soft signal)
        if "League_Conf" in df.columns:
            lc = pd.to_numeric(
    df["League_Conf"],
    errors="coerce").rank(
        pct=True).fillna(0.50)
        else:
            lc = pd.Series([0.50] * len(df), index=df.index)

        if "Market_Conf_Score" in df.columns:
            mc = pd.to_numeric(
    df["Market_Conf_Score"],
    errors="coerce").rank(
        pct=True).fillna(0.50)
        else:
            mc = pd.Series([0.50] * len(df), index=df.index)

        df["AMQS"] = (0.55 * lc + 0.45 * mc).clip(0.0, 1.0)

        # Percentile within market (Seçim) if present
        if "Seçim" in df.columns:
            df["AMQS_percentile"] = df.groupby("Seçim")["AMQS"].transform(
                lambda x: x.rank(pct=True)).fillna(0.50)
        else:
            df["AMQS_percentile"] = df["AMQS"].rank(pct=True).fillna(0.50)

    p = pd.to_numeric(df["AMQS_percentile"], errors="coerce").fillna(0.50)
    # Compatibility: Walk-Forward policy engine expects 0-100 scale
    df["AutoMod_percentile"] = (p * 100.0).clip(0.0, 100.0)
    df["AutoMod_Status"] = np.select([p >= 0.80, p >= 0.60], [
                                     "High", "Medium"], default="Low")
    df["AutoMod"] = df["AutoMod_Status"].map(
        {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}).fillna("🟡 Medium")

    return df


def _automod_cell_style(val) -> str:
    if not isinstance(val, str):
        return ""
    v = val.lower()
    if "high" in v:
        return "background-color: rgba(34,197,94,0.20); font-weight: 700;"
    if "medium" in v:
        return "background-color: rgba(245,158,11,0.20); font-weight: 700;"
    if "low" in v:
        return "background-color: rgba(239,68,68,0.20); font-weight: 700;"
    return ""
# =============================================================
# END AMQS helpers
# =============================================================


# ================= FINAL CONFIDENCE (NUMERIC) & ICON HELPERS =================
def _safe_norm(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series([0.5] * len(series), index=series.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series([0.5] * len(series), index=series.index)
    return ((s - mn) / (mx - mn)).fillna(0.5)


def compute_final_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite confidence score for UI (NOT a hard filter).
    Uses: Score, GoldenScore, League_Conf, Market_Conf_Score, and EV downside penalty.
    Returns Final_Confidence in [0,1].
    """
    if df is None or df.empty:
        return df
    d = df.copy()

    for col, default in [
        ("Score", 0.5),
        ("League_Conf", 0.5),
        ("Market_Conf_Score", 0.5),
        ("EV", 0.0),
    ]:
        if col not in d.columns:
            d[col] = default

    # IMPORTANT: Do NOT materialize a constant GoldenScore column here.
    # If GoldenScore is missing, use a temporary neutral series so later
    # stages can still compute the real GoldenScore.
    _gold_series = d["GoldenScore"] if "GoldenScore" in d.columns else pd.Series(
        0.5, index=d.index)

    n_score = _safe_norm(d["Score"])
    n_gold = _safe_norm(_gold_series)
    n_lconf = _safe_norm(d["League_Conf"])
    n_mconf = _safe_norm(d["Market_Conf_Score"])

    ev = pd.to_numeric(d["EV"], errors="coerce").fillna(0.0)
    ev_pen = ev.where(ev < 0, 0.0).abs()
    n_evpen = _safe_norm(ev_pen)

    d["Final_Confidence"] = (
        0.40 * n_score
        + 0.25 * n_gold
        + 0.20 * n_lconf
        + 0.15 * n_mconf
        - 0.10 * n_evpen
    ).clip(0.0, 1.0)

    return d


def ensure_conf_percentile_columns(
    df: pd.DataFrame,
    by_market: bool = True,
    high_cut: float = 80.0,
    medium_cut: float = 50.0,
) -> pd.DataFrame:
    """
    Creates numeric confidence percentile columns similar to AutoMod percentile.

    Adds:
      - CONF_percentile: 0..100
      - CONF_Status: Low / Medium / High
      - CONF_ICON: 🔴 / 🟡 / 🟢

    If by_market=True, percentile is computed separately per 'Seçim' (so MS1/MS2/OU/KG are comparable within type).
    """
    if df is None or df.empty:
        return df
    d = df.copy()

    if "Final_Confidence" not in d.columns:
        d = compute_final_confidence(d)

    fc = pd.to_numeric(d["Final_Confidence"], errors="coerce")
    # Default percentile if everything is missing
    if fc.isna().all():
        d["CONF_percentile"] = 50.0
    else:
        if by_market and ("Seçim" in d.columns):
            # percentile within each bet type
            d["CONF_percentile"] = (
                d.groupby("Seçim")["Final_Confidence"]
                 .rank(pct=True, method="average")
                 .mul(100.0)
            )
        else:
            d["CONF_percentile"] = d["Final_Confidence"].rank(
                pct=True, method="average").mul(100.0)

        d["CONF_percentile"] = pd.to_numeric(
    d["CONF_percentile"],
    errors="coerce").fillna(50.0).clip(
        0.0,
         100.0)

    def _status(p):
        p = float(p)
#         except Exception:  # AUTO-COMMENTED (orphan except)
#             return "Medium"
        if p >= float(high_cut):
            return "High"
        if p >= float(medium_cut):
            return "Medium"
        return "Low"

    d["CONF_Status"] = d["CONF_percentile"].apply(_status)

    icon_map = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    d["CONF_ICON"] = d["CONF_Status"].map(icon_map).fillna("🟡")

    return d

# Backward-compat: existing code calls assign_final_conf_icon(...)


def assign_final_conf_icon(
    df: pd.DataFrame,
    top_pct: float = 0.30,
     bottom_pct: float = 0.20) -> pd.DataFrame:
    """
    Compatibility wrapper:
    Previously used quantiles (top_pct/bottom_pct). Now we expose numeric percentiles like AutoMod.
    The UI filters should rely on CONF_percentile / CONF_Status / CONF_ICON.
    """
    return ensure_conf_percentile_columns(df, by_market=True)

# ===================================================================


def _automod_badge(status: str) -> str:
    status = (status or "").strip().lower()
    if status == "high":
        return "🟢 High"
    if status == "low":
        return "🔴 Low"
    if status == "medium":
        return "🟡 Medium"
    return "⚪ N/A"


def _automod_css(status: str) -> str:
    status = (status or "").strip().lower()
    if status == "high":
        return "background-color: rgba(0, 200, 0, 0.18); font-weight: 700;"
    if status == "low":
        return "background-color: rgba(220, 0, 0, 0.16); font-weight: 700;"
    if status == "medium":
        return "background-color: rgba(255, 170, 0, 0.18); font-weight: 700;"
    return "color: #888;"


# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ===================== Anchor Decision Discipline (ADD) =====================
def apply_anchor_decision_discipline(df):
    """
    Anchor Decision Discipline (ADD)
    - Hard-gate: Keep only anchor-family candidates per match (secondary is handled upstream).
    - Soft-lock direction:
        * BTTS: SIM_PBTTS >= 0.55 => YES(KG Var) preferred; <=0.45 => NO(KG Yok) preferred
        * OU  : SIM_POver >= 0.55 => Over(Üst) preferred; <=0.45 => Under(Alt) preferred
    Works with Turkish labels (KG Var/Yok, 2.5 Üst/Alt) and English (BTTS Yes/No, Over/Under).
    """
    if df is None or len(df) == 0:
        return df

    import pandas as pd
    import numpy as np

    cols = {c.lower(): c for c in df.columns}

    def _get(*names):
        for n in names:
            c = cols.get(n.lower())
            if c is not None:
                return c
        return None

    # Required columns
    anchor_col = _get("SIM_ANCHOR_GROUP", "sim_anchor_group")
    if anchor_col is None:
        return df

    # Market/selection column (your app uses 'Seçim')
    market_col = _get(
    "Seçim",
    "secim",
    "Selection",
    "selection",
    "Market",
    "market",
    "bet_type",
    "BetType",
    "Pick",
     "pick")
    if market_col is None:
        return df

    p_over = _get("SIM_POver", "sim_pover")
    p_btts = _get("SIM_PBTTS", "sim_pbtts")

    # -------------------------
    # Similarity Gate (hard)
    # If similarity evidence is weak or direction is uncertain, do NOT force a pick.
    # -------------------------
    simq_col = _get("SIM_QUALITY", "sim_quality")
    strength_col = _get(
    "SIM_ANCHOR_STRENGTH",
    "sim_anchor_strength",
    "AnchorStrength",
     "anchor_strength")

    gate_enabled = bool(st.session_state.get("SIM_GATE_ENABLED", True))
    min_simq = float(st.session_state.get("SIM_GATE_MIN_QUALITY", 0.45))
    min_strength = float(
    st.session_state.get(
        "SIM_GATE_MIN_ANCHOR_STRENGTH",
         0.50))

    # Direction certainty bands (grey zone => PASS)
    btts_yes_cut = float(
    st.session_state.get(
        "SIM_BTTS_YES_CUT",
         0.60))  # >= => KG Var
    btts_no_cut = float(
    st.session_state.get(
        "SIM_BTTS_NO_CUT",
         0.45))   # <= => KG Yok
    ou_over_cut = float(
    st.session_state.get(
        "SIM_OU_OVER_CUT",
         0.60))   # >= => Üst
    ou_under_cut = float(
    st.session_state.get(
        "SIM_OU_UNDER_CUT",
         0.45))  # <= => Alt

    # Build a robust match key if no explicit id exists
    id_col = _get("match_id", "Match_ID", "fixture_id", "Fixture_ID")
    if id_col is None:
        # Use canonical tuple key
        kcols = [c for c in [_get("Date", "date"), _get("League", "league"), _get("HomeTeam", "hometeam"),
                             _get("AwayTeam", "awayteam")] if c is not None]
        if not kcols:
            # last resort: keep as-is
            return df
        tmp = df.copy()
        tmp["_ADD_KEY"] = tmp[kcols].astype(str).agg("||".join, axis=1)
        id_col = "_ADD_KEY"
        df2 = tmp
    else:
        df2 = df

    def is_btts(s):
        s = str(s).lower()
        return ("kg" in s) or s.startswith("btts")

    def is_ou(s):
        s = str(s).lower()
        return (
    "2.5" in s) or (
        "üst" in s) or (
            "alt" in s) or (
                "over" in s) or (
                    "under" in s)

    def is_ms(s):
        s = str(s).lower().strip()
        return s.startswith("ms") or s in {
    "1", "x", "2", "ms1", "msx", "ms2"} or (
        "1x2" in s)

    def is_yes_btts(s):
        s = str(s).lower()
        return ("var" in s) or ("yes" in s)

    def is_no_btts(s):
        s = str(s).lower()
        return ("yok" in s) or ("no" in s)

    def is_over(s):
        s = str(s).lower()
        return ("üst" in s) or ("over" in s)

    def is_under(s):
        s = str(s).lower()
        return ("alt" in s) or ("under" in s)

    out_parts = []
    for _, g in df2.groupby(id_col, dropna=False):
        anchor = str(g.iloc[0][anchor_col]).upper().strip()
        gg = g.copy()

        # Hard gate: similarity evidence too weak -> PASS (no rows for this
        # match)
        if gate_enabled:
            if simq_col is not None:
                _sq = pd.to_numeric(
    g.iloc[0].get(
        simq_col,
        np.nan),
     errors="coerce")
                if pd.notna(_sq) and float(_sq) < float(min_simq):
                    continue
            if strength_col is not None:
                _st = pd.to_numeric(
    g.iloc[0].get(
        strength_col,
        np.nan),
     errors="coerce")
                if pd.notna(_st) and float(_st) < float(min_strength):
                    continue

        if anchor == "BTTS":
            fam = gg[gg[market_col].map(is_btts)]
            if fam.empty:
                continue  # hard-gate

            # Direction certainty gate: grey zone => PASS (do not force KG
            # Var/Yok)
            if gate_enabled and (p_btts is not None):
                pb = pd.to_numeric(
    g.iloc[0].get(
        p_btts,
        np.nan),
     errors="coerce")
                if pd.notna(pb) and (
    float(pb) < float(btts_yes_cut)) and (
        float(pb) > float(btts_no_cut)):
                    continue
            # soft-lock direction if we have a usable SIM_PBTTS
            if p_btts is not None and pd.notna(g.iloc[0].get(p_btts, np.nan)):
                pb = float(g.iloc[0][p_btts])
                if pb >= 0.55:
                    fam = fam[fam[market_col].map(is_yes_btts)]
                elif pb <= 0.45:
                    fam = fam[fam[market_col].map(is_no_btts)]
                # if grey zone, keep both
            gg = fam

        elif anchor == "OU":
            fam = gg[gg[market_col].map(is_ou)]
            if fam.empty:
                continue

            # Direction certainty gate: grey zone => PASS (do not force
            # Üst/Alt)
            if gate_enabled and (p_over is not None):
                po = pd.to_numeric(
    g.iloc[0].get(
        p_over,
        np.nan),
     errors="coerce")
                if pd.notna(po) and (
    float(po) < float(ou_over_cut)) and (
        float(po) > float(ou_under_cut)):
                    continue

            if p_over is not None and pd.notna(g.iloc[0].get(p_over, np.nan)):
                po = float(g.iloc[0][p_over])
                if po >= 0.55:
                    fam = fam[fam[market_col].map(is_over)]
                elif po <= 0.45:
                    fam = fam[fam[market_col].map(is_under)]
            gg = fam

        else:  # MS default
            fam = gg[gg[market_col].map(is_ms)]
            if fam.empty:
                continue
            gg = fam

        out_parts.append(gg)

    if not out_parts:
        return df.iloc[0:0].copy()

    out = pd.concat(out_parts, ignore_index=True)

    # Clean helper column if added
    if "_ADD_KEY" in out.columns:
        out = out.drop(columns=["_ADD_KEY"])

    return out
# ===================== END ADD =====================


# -----------------------------
# Disk cache helpers (reduces Full Mode startup by reusing prior training outputs)
# -----------------------------
CACHE_DIR = Path.home() / ".kmquant_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAJOR_CACHE_VERSION = "v1"

# -----------------------------
# Daily coupon journal (persistent)
# -----------------------------
JOURNAL_DIR = Path.cwd() / "kmquant_journal"
JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_FILE = JOURNAL_DIR / "daily_coupons.parquet"
DEFAULTS_FILE = JOURNAL_DIR / "ui_defaults.json"

# -----------------------------
# Daily prediction log (persistent)
# -----------------------------
# Stores a snapshot of the "Geniş Aday" universe so that:
# - Walk-forward filter optimization can use *fixed* daily candidates
# - BestOfRank / CONF_percentile / AMQS_percentile do not drift within the day
#
# One row = one pick candidate (match + market) at snapshot time.
PRED_LOG_FILE = JOURNAL_DIR / "prediction_log.parquet"
# ===================== Rule Engine (Quality -> Anchor -> Direction -> EV)
# Locked config (per Kerem's spec)
RULE_ENGINE_CFG = {
    "Q_MIN": 0.45,
    "S_MIN": 0.50,
    "GAP_MIN": 0.05,
    "OU_OVER": 0.60,
    "OU_UNDER": 0.40,
    "BTTS_YES": 0.60,
    "BTTS_NO": 0.40,
    "MS_PMAX": 0.50,
    "MS_MARGIN": 0.10,
    "USE_EV": False,
    "EV_MIN": 0.00,
}


def _rule_engine_match_key_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in ["Date", "League", "HomeTeam", "AwayTeam"]:
        if c in df.columns:
            cols.append(c)
    # minimum viable: Date+Home+Away (League optional)
    if ("Date" in cols) and ("HomeTeam" in cols) and ("AwayTeam" in cols):
        return cols
    return []


def _rule_engine_make_key(df: pd.DataFrame) -> pd.Series:
    key_cols = _rule_engine_match_key_cols(df)
    if not key_cols:
        # fallback to index (still deterministic within a dataframe)
        return df.index.astype(str)
    # stringify robustly
    parts = []
    for c in key_cols:
        if c == "Date":
            parts.append(pd.to_datetime(df[c], errors="coerce").dt.strftime(
                "%Y-%m-%d").fillna(df[c].astype(str)))
        else:
            parts.append(df[c].astype(str).fillna(""))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def decide_pick_rule_engine(
    row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tiered rule-engine decision (match-level).
    Returns a dict with:
      - DECISION_TIER: PRIMARY / SECONDARY / PASS
      - DECISION_REASON
      - ANCHOR_GROUP, ANCHOR_STRENGTH, ANCHOR_GAP
      - DIRECTION_DECISION (target selection label, if any)
      - PICK_ALLOWED (back-compat: True only when PRIMARY)
      - NO_PICK_REASON (back-compat: mirrors DECISION_REASON when not PRIMARY)
    NOTE: This function NEVER hard-filters candidates; it only decides the preferred direction and tier.
    """
    # --- Gate 0: similarity quality ---
    simq = float(
    pd.to_numeric(
        row.get(
            "SIM_QUALITY",
            np.nan),
             errors="coerce"))
    if (not np.isfinite(simq)) or (simq < float(cfg.get("Q_MIN", 0.45))):
        return dict(
            PICK_ALLOWED=False,
            NO_PICK_REASON="LOW_SIM_QUALITY",
            DECISION_TIER="PASS",
            DECISION_REASON="LOW_SIM_QUALITY",
            ANCHOR_GROUP=None,
            ANCHOR_STRENGTH=np.nan,
            ANCHOR_GAP=np.nan,
            DIRECTION_DECISION=None,
        )

    # strengths (prefer precomputed)
    ms_s = pd.to_numeric(row.get("SIM_MS_STRENGTH", np.nan), errors="coerce")
    ou_s = pd.to_numeric(row.get("SIM_OU_STRENGTH", np.nan), errors="coerce")
    btts_s = pd.to_numeric(
    row.get(
        "SIM_BTTS_STRENGTH",
        np.nan),
         errors="coerce")

    # compute strengths if missing
    if not (np.isfinite(ms_s) and np.isfinite(ou_s) and np.isfinite(btts_s)):
        sp1 = pd.to_numeric(row.get("SIM_P1", np.nan), errors="coerce")
        spx = pd.to_numeric(row.get("SIM_PX", np.nan), errors="coerce")
        sp2 = pd.to_numeric(row.get("SIM_P2", np.nan), errors="coerce")
        spo = pd.to_numeric(row.get("SIM_POver", np.nan), errors="coerce")
        spb = pd.to_numeric(row.get("SIM_PBTTS", np.nan), errors="coerce")

        u = 1.0 / 3.0
        ms_tv = 0.5 * (abs(sp1 - u) + abs(spx - u) + abs(sp2 - u)) if np.isfinite(
            sp1) and np.isfinite(spx) and np.isfinite(sp2) else np.nan
        ms_s = (ms_tv / (2.0 / 3.0)) if np.isfinite(ms_tv) else np.nan
        ou_s = (abs(spo - 0.5) * 2.0) if np.isfinite(spo) else np.nan
        btts_s = (abs(spb - 0.5) * 2.0) if np.isfinite(spb) else np.nan

    strengths = {
        "MS": float(ms_s) if np.isfinite(ms_s) else 0.0,
        "OU": float(ou_s) if np.isfinite(ou_s) else 0.0,
        "BTTS": float(btts_s) if np.isfinite(btts_s) else 0.0,
    }

    # --- Gate 1: anchor selection ---
    anchor = max(strengths, key=strengths.get)
    s1 = float(strengths[anchor])
    s_sorted = sorted(strengths.values(), reverse=True)
    s2 = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0
    gap = float(s1 - s2)

    if s1 < float(cfg.get("S_MIN", 0.50)):
        return dict(
            PICK_ALLOWED=False,
            NO_PICK_REASON="WEAK_ANCHOR",
            DECISION_TIER="PASS",
            DECISION_REASON="WEAK_ANCHOR",
            ANCHOR_GROUP=anchor,
            ANCHOR_STRENGTH=s1,
            ANCHOR_GAP=gap,
            DIRECTION_DECISION=None,
        )

    if gap < float(cfg.get("GAP_MIN", 0.05)):
        return dict(
            PICK_ALLOWED=False,
            NO_PICK_REASON="GAP_TOO_SMALL",
            DECISION_TIER="PASS",
            DECISION_REASON="GAP_TOO_SMALL",
            ANCHOR_GROUP=anchor,
            ANCHOR_STRENGTH=s1,
            ANCHOR_GAP=gap,
            DIRECTION_DECISION=None,
        )

    # --- Gate 2: direction (tiered; never kills the match) ---
    decision = None
    tier = "SECONDARY"
    reason = f"SECONDARY_{anchor}"

    if anchor == "OU":
        p = pd.to_numeric(row.get("SIM_POver", np.nan), errors="coerce")
        if np.isfinite(p) and (float(p) >= float(cfg.get("OU_OVER", 0.60))):
            decision = "2.5 Üst"
            tier = "PRIMARY"
            reason = "PRIMARY_DOMINANT_OU_OVER"
        elif np.isfinite(p) and (float(p) <= float(cfg.get("OU_UNDER", 0.40))):
            decision = "2.5 Alt"
            tier = "PRIMARY"
            reason = "PRIMARY_DOMINANT_OU_UNDER"
        else:
            # Grey zone => no primary, but allow a secondary fallback later
            decision = None
            tier = "SECONDARY"
            reason = "SECONDARY_OU_DIRECTION_UNCERTAIN"

    elif anchor == "BTTS":
        p = pd.to_numeric(row.get("SIM_PBTTS", np.nan), errors="coerce")
        if np.isfinite(p) and (float(p) >= float(cfg.get("BTTS_YES", 0.60))):
            decision = "KG Var"
            tier = "PRIMARY"
            reason = "PRIMARY_DOMINANT_BTTS_YES"
        elif np.isfinite(p) and (float(p) <= float(cfg.get("BTTS_NO", 0.40))):
            decision = "KG Yok"
            tier = "PRIMARY"
            reason = "PRIMARY_DOMINANT_BTTS_NO"
        else:
            decision = None
            tier = "SECONDARY"
            reason = "SECONDARY_BTTS_DIRECTION_UNCERTAIN"

    else:  # MS
        p1 = pd.to_numeric(row.get("SIM_P1", np.nan), errors="coerce")
        px = pd.to_numeric(row.get("SIM_PX", np.nan), errors="coerce")
        p2 = pd.to_numeric(row.get("SIM_P2", np.nan), errors="coerce")
        probs = {
            "MS 1": float(p1) if np.isfinite(p1) else -1.0,
            "MS X": float(px) if np.isfinite(px) else -1.0,
            "MS 2": float(p2) if np.isfinite(p2) else -1.0,
        }
        best = max(probs, key=probs.get)
        pmax = float(probs[best])
        pvals = sorted(probs.values(), reverse=True)
        p2nd = float(pvals[1]) if len(pvals) > 1 else -1.0
        margin = float(pmax - p2nd)

        if (pmax >= float(cfg.get("MS_PMAX", 0.50))) and (
            margin >= float(cfg.get("MS_MARGIN", 0.10))):
            decision = best
            tier = "PRIMARY"
            reason = "PRIMARY_DOMINANT_MS"
        else:
            decision = None
            tier = "SECONDARY"
            reason = "SECONDARY_MS_DIRECTION_UNCERTAIN"

    return dict(
        PICK_ALLOWED=(tier == "PRIMARY"),
        NO_PICK_REASON=None if (tier == "PRIMARY") else reason,
        DECISION_TIER=tier,
        DECISION_REASON=reason,
        ANCHOR_GROUP=anchor,
        ANCHOR_STRENGTH=s1,
        ANCHOR_GAP=gap,
        DIRECTION_DECISION=decision,
    )


def apply_rule_engine_to_candidates(
    df: pd.DataFrame, cfg: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Apply tiered decision tree to a *candidate* dataframe (one row per bet).

    What it does:
      - Adds debug + decision columns per row:
          PICK_ALLOWED (PRIMARY only), NO_PICK_REASON, ANCHOR_GROUP, ANCHOR_STRENGTH, ANCHOR_GAP, DIRECTION_DECISION,
          DECISION_TIER, DECISION_REASON, RULE_PRIMARY_PICK
      - NEVER hard-filters (does not drop rows). This is key to avoid "122 maç günü -> 0 pick".
      - If a PRIMARY direction exists but that candidate row is missing, it falls back to the best available row as SECONDARY.

    Back-compat:
      - Existing code that filters by PICK_ALLOWED will still behave like "primary only".
    """
    if df is None or getattr(df, "empty", True):
        return df
    cfg = cfg or RULE_ENGINE_CFG

    d = df.copy()

    # ensure Seçim exists (candidate pool contract)
    if "Seçim" not in d.columns:
        if "Selection" in d.columns:
            d["Seçim"] = d["Selection"]
        elif "Market" in d.columns:
            d["Seçim"] = d["Market"]
        else:
            return d

    d["_MATCH_KEY_RE"] = _rule_engine_make_key(d)

    # initialize decision/debug columns
    init_cols = [
        ("PICK_ALLOWED", False),
        ("NO_PICK_REASON", ""),
        ("ANCHOR_GROUP", ""),
        ("ANCHOR_STRENGTH", np.nan),
        ("ANCHOR_GAP", np.nan),
        ("DIRECTION_DECISION", ""),
        ("DECISION_TIER", "PASS"),
        ("DECISION_REASON", "PASS"),
        ("RULE_PRIMARY_PICK", False),
    ]
    for c, default in init_cols:
        if c not in d.columns:
            d[c] = default
        else:
            # reset to default to avoid stale values when re-running
            d[c] = default

    def _best_row_idx(gg: pd.DataFrame) -> int:
        """Pick a stable 'best available' row for fallback / secondary."""
        if gg is None or gg.empty:
            return -1
        # Prefer Score, then Final_Confidence, then GoldenScore
        s = pd.to_numeric(gg.get("Score", np.nan), errors="coerce")
        fc = pd.to_numeric(gg.get("Final_Confidence", np.nan), errors="coerce")
        gs = pd.to_numeric(gg.get("GoldenScore", np.nan), errors="coerce")
        # Build a composite ranking score (robust to NaNs)
        comp = (
            s.fillna(-1e9) * 1000.0
            + fc.fillna(-1e9) * 10.0
            + gs.fillna(-1e9) * 1.0
        )
        return int(comp.idxmax())

    for mk, g in d.groupby("_MATCH_KEY_RE", sort=False):
        r0 = g.iloc[0]
        dec = decide_pick_rule_engine(r0, cfg)

        # write match-level info on all rows in group
        for k, v in dec.items():
            if k in d.columns:
                d.loc[g.index, k] = v

        anchor = dec.get("ANCHOR_GROUP", "")
        s1 = dec.get("ANCHOR_STRENGTH", np.nan)
        gap = dec.get("ANCHOR_GAP", np.nan)
        tier = dec.get("DECISION_TIER", "PASS")
        reason = dec.get("DECISION_REASON", "PASS")
        pick = dec.get("DIRECTION_DECISION", None)

        # Default: group rows are PASS; we'll elevate one row if PRIMARY or
        # SECONDARY fallback chosen
        d.loc[g.index, "DECISION_TIER"] = "PASS"
        d.loc[g.index, "DECISION_REASON"] = reason
        d.loc[g.index, "PICK_ALLOWED"] = False
        d.loc[g.index, "RULE_PRIMARY_PICK"] = False

        if tier == "PRIMARY" and pick:
            gg = g[g["Seçim"].astype(str) == str(pick)]
            if not gg.empty:
                # mark that row(s) as PRIMARY
                d.loc[gg.index, "DECISION_TIER"] = "PRIMARY"
                d.loc[gg.index, "DECISION_REASON"] = reason
                d.loc[gg.index, "PICK_ALLOWED"] = True
                d.loc[gg.index, "RULE_PRIMARY_PICK"] = True
                # everything else becomes SECONDARY (kept for broad list, but
                # not BestOf)
                rest = g.index.difference(gg.index)
                if len(rest) > 0:
                    d.loc[rest, "DECISION_TIER"] = "SECONDARY"
                    d.loc[rest,
     "DECISION_REASON"] = f"SECONDARY_NON_PRIMARY_{anchor}"
            else:
                # PRIMARY direction exists but candidate missing -> fallback
                bi = _best_row_idx(g)
                if bi != -1:
                    d.loc[bi, "DECISION_TIER"] = "SECONDARY"
                    d.loc[bi, "DECISION_REASON"] = "FALLBACK_MISSING_CANDIDATE"
                    d.loc[g.index, "NO_PICK_REASON"] = "MISSING_CANDIDATE"
        else:
            # No primary -> choose a single secondary fallback (best
            # available), keep others PASS
            bi = _best_row_idx(g)
            if bi != -1:
                d.loc[bi, "DECISION_TIER"] = "SECONDARY"
                d.loc[bi, "DECISION_REASON"] = reason

        # Ensure anchor columns are present for group (even when PASS)
        d.loc[g.index, "ANCHOR_GROUP"] = anchor
        d.loc[g.index, "ANCHOR_STRENGTH"] = s1
        d.loc[g.index, "ANCHOR_GAP"] = gap

    out = d.copy()
    # (MDL) add profile + controlled GoldenScore boost for UI/ranking
#     try:  # AUTO-COMMENTED (illegal global try)
#         out = add_mdl_features(out)
#     except Exception:
#         pass

    if "_MATCH_KEY_RE" in out.columns:
        out.drop(columns=["_MATCH_KEY_RE"], inplace=True, errors="ignore")
    return out


def _make_match_id_row(r: pd.Series) -> str:
    """Stable match id for snapshots / joins. Safe: never raises."""
#     try:  # AUTO-COMMENTED (illegal global try)
#         d = str(r.get("Date", "")).strip()
#         lg = str(r.get("League", "")).strip()
#         ht = str(r.get("HomeTeam", "")).strip()
#         at = str(r.get("AwayTeam", "")).strip()
#         return f"{lg}|{d}|{ht}|{at}"
#     except Exception:
#         return ""


def load_prediction_log() -> pd.DataFrame:
    """Load prediction log (parquet). Safe: returns empty df on errors."""
#     try:  # AUTO-COMMENTED (illegal global try)
#         if PRED_LOG_FILE.exists():
#             df = pd.read_parquet(PRED_LOG_FILE)
#             # Normalize timestamps
#             for c in ["Snapshot_Date", "Date", "Logged_At"]:
#                 if c in df.columns:
#                     df[c] = pd.to_datetime(df[c], errors="coerce")
#             return df
#     except Exception:
#         pass
    return pd.DataFrame()


# ===========================================================
# JOURNAL / PICK LOG (v1) — Tahmin → Pick Log → Sonuç Bağla → Performans
# ===========================================================
def _canon_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def _col_series(df, col, default=""):
    if col in df.columns:
        return df[col]
    return pd.Series([default]*len(df), index=df.index)


def canonicalize_picklog_df(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize Date/Snapshot_Date and identifiers to avoid split-brain formats.
    - Date, Snapshot_Date -> 'YYYY-MM-DD' (string)
    - League/HomeTeam/AwayTeam -> lower/strip
    - Match_ID -> 'league|YYYY-MM-DD|home|away'
    - Ensures Pick_ID exists as 'Match_ID|Seçim' (date-neutral, deterministic)
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out = out.reset_index(drop=True)

    # column aliases
    if "HomeTeam" not in out.columns and "Home_Team" in out.columns:
        out["HomeTeam"] = out["Home_Team"]
    if "AwayTeam" not in out.columns and "Away_Team" in out.columns:
        out["AwayTeam"] = out["Away_Team"]
    if "League" not in out.columns and "Lig" in out.columns:
        out["League"] = out["Lig"]
    if "Date" not in out.columns and "MatchDate" in out.columns:
        out["Date"] = out["MatchDate"]

    # Date-like columns
    if "Date" in out.columns:
        out["Date"] = _canon_date_series(out["Date"])
    if "Snapshot_Date" in out.columns:
        out["Snapshot_Date"] = _canon_date_series(out["Snapshot_Date"])
    else:
        out["Snapshot_Date"] = _col_series(out, "Date", pd.NaT).astype(str)

    # normalize strings - ensure required base cols exist
    d = _col_series(out, "Date", "").astype(str)
    l = _col_series(out, "League", "").astype(str).str.strip().str.lower()
    h = _col_series(out, "HomeTeam", "").astype(str).str.strip().str.lower()
    a = _col_series(out, "AwayTeam", "").astype(str).str.strip().str.lower()
    s = _col_series(out, "Seçim", "").astype(str)

    # Update the dataframe with normalized values
    out["Date"] = d
    out["League"] = l
    out["HomeTeam"] = h
    out["AwayTeam"] = a
    out["Seçim"] = s

    # rebuild Match_ID ALWAYS (this is the key fix)
    out["Match_ID"] = l + "|" + d + "|" + h + "|" + a

    # deterministic Pick_ID
    sel = (
        out["Seçim"] if "Seçim" in out.columns else
        out["Secim"] if "Secim" in out.columns else
        out.get("SeA\x02im", pd.Series("", index=out.index))
    )

    if not isinstance(sel, pd.Series):
        sel = pd.Series("", index=out.index)

    out["Pick_ID"] = (
        out.get("Match_ID", pd.Series("", index=out.index)).astype(str)
        + "|"
        + sel.astype(str)
    )

    # ensure no dup columns
#     try:  # AUTO-COMMENTED (illegal global try)
#         out = out.loc[:, ~out.columns.duplicated()].copy()
#     except Exception:
#         pass

    return out


def ensure_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical Match_ID (league|YYYY-MM-DD|home|away) and canonical dates.
    NOTE: We intentionally rebuild Match_ID even if it already exists to avoid mixed formats
    (e.g. '2025-12-20 00:00:00' vs '2025-12-20') that break filtering/dedup.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty:
        return df
    # If Match_ID is missing or all null/empty, canonicalize
    if "Match_ID" not in df.columns or df["Match_ID"].isna().all() or (df["Match_ID"].astype(str).str.strip() == "").all():
        df = canonicalize_picklog_df(df)
    else:
        # Even if Match_ID exists, ensure it's canonical by re-canonicalizing
        df = canonicalize_picklog_df(df)
    return df


def _pick_log_path() -> str:
    return os.path.join(_journal_dir(), "pick_log.parquet")


def _debug_journal_info() -> dict:
    """Return paths used for persistence (for on-screen debugging)."""
#     try:  # AUTO-COMMENTED (illegal global try)
#         app_dir = Path(__file__).resolve().parent
#     except Exception:
#         app_dir = Path.cwd()
    app_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    return {
    "cwd": str(
        Path.cwd()), "app_dir": str(app_dir), "journal_dir": str(
            Path(
                _journal_dir())), "pick_log_path": str(
                    Path(
                        _pick_log_path())), "pick_log_exists": bool(
                            os.path.exists(
                                _pick_log_path())), "pick_log_csv_exists": bool(
                                    os.path.exists(
                                        _pick_log_path().replace(
                                            ".parquet", ".csv"))), }


def _find_legacy_pick_logs() -> list:
    """Heuristic search for legacy pick_log files in common locations."""
    cand = []
#     try:  # AUTO-COMMENTED (illegal global try)
#         app_dir = Path(__file__).resolve().parent
#     except Exception:
#         app_dir = Path.cwd()
#     cwd = Path.cwd()
    app_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    bases = [app_dir, app_dir / "data", Path.cwd(), Path.home() / "Downloads"]
    bases = [b for b in bases if b.exists()]
    for b in bases:
        for name in ["pick_log.parquet", "pick_log.csv"]:
            p = b / name
            if p.exists():
                cand.append(str(p))
    # De-dup preserve order
    out = []
    for p in cand:
        if p not in out:
            out.append(p)
    return out


def migrate_pick_log_if_needed() -> None:
    """If current pick log is empty but legacy logs exist, merge them into current store."""
#     try:  # AUTO-COMMENTED (illegal global try)
#         cur = load_pick_log()
#         cur_n = 0 if cur is None else int(len(cur))
#     except Exception:
#         cur_n = 0
#         cur = pd.DataFrame()
    try:
        cur = load_pick_log()
        cur_n = 0 if cur is None else int(len(cur))
    except Exception:
        cur_n = 0
        cur = pd.DataFrame()

    legacy = _find_legacy_pick_logs()
    if cur_n > 0:
        return

    merged = []
    for lp in legacy:
        if lp.endswith(".parquet"):
            df = pd.read_parquet(lp)
        else:
            df = pd.read_csv(lp)
        if df is not None and not df.empty:
            merged.append(df)

    if not merged:
        return

    out = pd.concat(merged, ignore_index=True, sort=False)
    # Drop duplicated column labels defensively
#     try:  # AUTO-COMMENTED (illegal global try)
#         out = out.loc[:, ~out.columns.duplicated()].copy()
#     except Exception:
#         pass
    # Dedup by Pick_ID if available
    if "Pick_ID" in out.columns:
        out["Pick_ID"] = out["Pick_ID"].astype(str)
        out = out.drop_duplicates(subset=["Pick_ID"], keep="last")
    save_pick_log(out)


# ------------------------------------------------------------------
# Journal safety: migrate legacy pick logs into current store (once)
# ------------------------------------------------------------------
migrate_pick_log_if_needed()


def load_pick_log() -> pd.DataFrame:
    """Safe loader: choose newest between parquet and csv, then canonicalize."""
    p = _pick_log_path()
    csvp = p.replace(".parquet", ".csv")

    p_exists = os.path.exists(p)
    c_exists = os.path.exists(csvp)

    if not p_exists and not c_exists:
        return pd.DataFrame()

    def _mtime(x: str) -> float:
        return os.path.getmtime(x)

    # choose newest
    use_parquet = p_exists and (_mtime(p) >= _mtime(csvp))

    df = None
    if use_parquet and p_exists:
        df = pd.read_parquet(p)
    if df is None and c_exists:
        df = pd.read_csv(csvp)
    if df is None and p_exists:
        df = pd.read_parquet(p)

    df = canonicalize_picklog_df(df)
    return df


def save_pick_log(df: pd.DataFrame) -> None:
    """Save pick log in both parquet and csv (best-effort), canonicalized."""
    if df is None:
        return
    df = canonicalize_picklog_df(df)

    p = _pick_log_path()
    csvp = p.replace(".parquet", ".csv")

    # write csv first (more robust under OneDrive locks)
#     try:  # AUTO-COMMENTED (illegal global try)
#         df.to_csv(csvp, index=False, encoding="utf-8-sig")
#     except Exception:
#         pass

    # parquet is optional; if it fails, we prefer csv as source of truth
#     try:  # AUTO-COMMENTED (illegal global try)
#         df.to_parquet(p, index=False)
#     except Exception:
#         # if parquet exists but is stale, remove so loader won't prefer it
#         try:
#             if os.path.exists(p):
#                 os.remove(p)
#         except Exception:
#             pass


def append_to_pick_log(new_rows: pd.DataFrame) -> int:
    """Append rows with robust canonicalization + dedup on Pick_ID."""
    if new_rows is None or new_rows.empty:
        return 0

    plog = load_pick_log()
    nr = canonicalize_picklog_df(new_rows.copy())
    plog = canonicalize_picklog_df(plog)

    # ensure Snapshot_Date exists (string YYYY-MM-DD)
    if "Snapshot_Date" not in nr.columns or nr["Snapshot_Date"].isna().all():
        nr["Snapshot_Date"] = nr.get(
    "Date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    nr["Snapshot_Date"] = pd.to_datetime(
    nr["Snapshot_Date"],
     errors="coerce").dt.strftime("%Y-%m-%d")

    # Pick_ID always deterministic
    nr["Pick_ID"] = nr["Match_ID"].astype(
        str) + "|" + nr.get("Seçim", "").astype(str)

    # dedup against existing
    if plog is not None and not plog.empty:
        if "Pick_ID" in plog.columns:
            plog["Pick_ID"] = plog["Pick_ID"].astype(str)
            nr = nr[~nr["Pick_ID"].astype(str).isin(
                set(plog["Pick_ID"].astype(str)))]
        else:
            # if old log missing Pick_ID, create then dedup
            plog["Pick_ID"] = plog.get("Match_ID", "").astype(
                str) + "|" + plog.get("Seçim", "").astype(str)
            nr = nr[~nr["Pick_ID"].astype(str).isin(
                set(plog["Pick_ID"].astype(str)))]

    if nr.empty:
        save_pick_log(plog)
        return 0

#     try:  # AUTO-COMMENTED (illegal global try)
#         plog = plog.loc[:, ~plog.columns.duplicated()].copy()
#     except Exception:
#         pass
#     try:
#         nr = nr.loc[:, ~nr.columns.duplicated()].copy()
#     except Exception:
#         pass

    out = pd.concat([plog.reset_index(drop=True), nr.reset_index(
        drop=True)], ignore_index=True, sort=False) if plog is not None else nr
    save_pick_log(out)
    return int(nr.shape[0])


def _infer_market_from_selection(sel: str) -> str:
    s = str(sel).lower()
    if "ms 1" in s or s.strip() in ["ms1", "1"]:
        return "MS1"
    if "ms x" in s or s.strip() in ["msx", "x", "draw"]: return "MSX"
    if "ms 2" in s or s.strip() in ["ms2", "2"]:
        return "MS2"
    if "2.5" in s and ("üst" in s or "over" in s): return "OU_O25"
    if "2.5" in s and ("alt" in s or "under" in s):
        return "OU_U25"
    if "kg var" in s or "btts" in s and ("yes" in s or "var" in s): return "BTTS_Y"
    if "kg yok" in s or "btts" in s and ("no" in s or "yok" in s):
        return "BTTS_N"
    return "OTHER"


def _result_from_scores(sel: str, ft_h: float, ft_a: float):
    """Return (hit: bool|nan, void: bool) for known markets."""
#     try:  # AUTO-COMMENTED (illegal global try)
#         if pd.isna(ft_h) or pd.isna(ft_a):
#             return (np.nan, False)
#         ft_h = int(ft_h)
#         ft_a = int(ft_a)
#     except Exception:
#         return (np.nan, False)

    m = _infer_market_from_selection(sel)
    tg = ft_h + ft_a

    if m == "MS1":
        return (ft_h > ft_a, False)
    if m == "MSX": return (ft_h == ft_a, False)
    if m == "MS2":
        return (ft_h < ft_a, False)
    if m == "OU_O25": return (tg >= 3, False)
    if m == "OU_U25":
        return (tg <= 2, False)
    if m == "BTTS_Y": return ((ft_h > 0 and ft_a > 0), False)
    if m == "BTTS_N":
        return ((ft_h == 0 or ft_a == 0), False)
    return (np.nan, False)


def settle_from_past(
    pick_log: pd.DataFrame,
    past_df: pd.DataFrame,
    *,
     force: bool = False) -> pd.DataFrame:
    """
    Join pick_log with past results (even if past has no Match_ID).
    - Primary: STRICT join on normalized (League, Date, HomeTeam, AwayTeam)
    - Secondary: DATE_SHIFT join with ±1 day tolerance
    Writes audit columns:
      RESULT_MATCH_METHOD: STRICT / DATE_SHIFT / NO_MATCH / MANUAL
      RESULT_MATCH_CONFIDENCE: 1.00 / 0.80 / 0.00 / 1.00
      RESULT_SOURCE_ROW_ID: integer row id from past_df when matched
    Manual overrides are preserved (rows with RESULT_MATCH_METHOD == 'MANUAL' or RESULT_OVERRIDE True).
    """
    if pick_log is None or pick_log.empty:
        return pick_log
    if past_df is None or past_df.empty:
        return pick_log

    pl = pick_log.copy()
    past = past_df.copy()

    # --- normalize column aliases (teams) ---
    if "HomeTeam" not in pl.columns and "Home_Team" in pl.columns:
        pl["HomeTeam"] = pl["Home_Team"]
    if "AwayTeam" not in pl.columns and "Away_Team" in pl.columns:
        pl["AwayTeam"] = pl["Away_Team"]
    if "HomeTeam" not in past.columns and "Home_Team" in past.columns:
        past["HomeTeam"] = past["Home_Team"]
    if "AwayTeam" not in past.columns and "Away_Team" in past.columns:
        past["AwayTeam"] = past["Away_Team"]

    # --- parse dates ---
    def _to_date_str(x):
        return pd.to_datetime(x, errors="coerce").strftime("%Y-%m-%d")
    pl["__date"] = pl.get("Date", "").apply(_to_date_str)
    past["__date"] = past.get("Date", "").apply(_to_date_str)

    # --- simple normalization for join keys ---
    def _norm(s):
        s = "" if s is None else str(s)
        s = s.strip().lower()
        # keep letters/numbers, drop punctuation; collapse spaces
        s = re.sub(r"[\u00A0]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9 ]+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _k(df):
        lg = df.get("League", "")
        ht = df.get("HomeTeam", "")
        at = df.get("AwayTeam", "")
        return (
            df["__date"].astype(str).map(_norm) + "|" +
            lg.astype(str).map(_norm) + "|" +
            ht.astype(str).map(_norm) + "|" +
            at.astype(str).map(_norm)
        )

    # Prepare keys
    pl["__k"] = _k(pl)
    past["__k"] = _k(past)

    # Identify score columns in past
    def _score_cols(df):
        pairs = [
            ("FT_Score_Home", "FT_Score_Away"),
            ("FT_Home", "FT_Away"),
            ("FT_ScoreHome", "FT_ScoreAway"),
            ("HomeFullTimeScore", "AwayFullTimeScore"),
            ("FTHG", "FTAG"),
        ]
        for h, a in pairs:
            if h in df.columns and a in df.columns:
                return h, a
        return None, None

    hcol, acol = _score_cols(past)
    if hcol is None:
        # no score columns, nothing to settle
        return pl.drop(
    columns=[
        c for c in [
            "__date",
            "__k"] if c in pl.columns],
             errors="ignore")

    # Add result audit columns if missing
    if "RESULT_MATCH_METHOD" not in pl.columns:
        pl["RESULT_MATCH_METHOD"] = ""
    if "RESULT_MATCH_CONFIDENCE" not in pl.columns:
        pl["RESULT_MATCH_CONFIDENCE"] = np.nan
    if "RESULT_SOURCE_ROW_ID" not in pl.columns:
        pl["RESULT_SOURCE_ROW_ID"] = np.nan
    if "RESULT_OVERRIDE" not in pl.columns:
        pl["RESULT_OVERRIDE"] = False

    # Respect manual overrides unless force=True
    manual_mask = (
    pl.get("RESULT_MATCH_METHOD").astype(str) == "MANUAL") | (
        pl.get("RESULT_OVERRIDE").astype(bool))
    if not force and manual_mask.any():
        pass

    # Determine which rows are open / to update
    open_mask = pl["Hit"].isna() if "Hit" in pl.columns else pd.Series([
                               True] * len(pl))
    if not force:
        open_mask = open_mask & (~manual_mask)

    # Source row id for debug
    past = past.reset_index(drop=True)
    past["__src_row"] = past.index.astype(int)

    # STRICT join
    strict = pl.loc[open_mask, ["__k"]].merge(
        past[["__k", "__src_row", hcol, acol]],
        on="__k", how="left"
    )
    # Map back
    pl.loc[open_mask, "__src_row"] = strict["__src_row"].values
    pl.loc[open_mask, "__h"] = strict[hcol].values
    pl.loc[open_mask, "__a"] = strict[acol].values

    matched_strict = pl.loc[open_mask, "__src_row"].notna()

    # DATE_SHIFT join for remaining
    remaining_idx = pl.loc[open_mask & (~matched_strict)].index
    if len(remaining_idx):
        # build past shifted keys
        past_shift = past.copy()
        dts = pd.to_datetime(past_shift["__date"], errors="coerce")
        for delta in [-1, 1]:
            ps = past_shift.copy()
            ps["__date"] = (
    dts +
    pd.to_timedelta(
        delta,
         unit="D")).dt.strftime("%Y-%m-%d")
            ps["__k_shift"] = (
                ps["__date"].astype(str).map(_norm) + "|" +
                ps["League"].astype(str).map(_norm) + "|" +
                ps["HomeTeam"].astype(str).map(_norm) + "|" +
                ps["AwayTeam"].astype(str).map(_norm)
            )
            # join pl key to shifted key
            tmp = pl.loc[remaining_idx, ["__k"]].rename(columns={"__k": "__k_shift"}).merge(
                ps[["__k_shift", "__src_row", hcol, acol]], on="__k_shift", how="left")
            # fill only where still empty
            fill_mask = pl.loc[remaining_idx,
     "__src_row"].isna() & tmp["__src_row"].notna()
            if fill_mask.any():
                fill_idx = remaining_idx[fill_mask]
                pl.loc[fill_idx,
    "__src_row"] = tmp.loc[fill_mask,
     "__src_row"].values
                pl.loc[fill_idx, "__h"] = tmp.loc[fill_mask, hcol].values
                pl.loc[fill_idx, "__a"] = tmp.loc[fill_mask, acol].values

    # Final match status
    has_match = pl.loc[open_mask, "__src_row"].notna()
    # write audit columns (only for rows we attempted)
    pl.loc[open_mask & has_match,
    "RESULT_SOURCE_ROW_ID"] = pl.loc[open_mask & has_match,
     "__src_row"].astype(float)
    # method/confidence
    is_shift = (pl.loc[open_mask & has_match, "RESULT_SOURCE_ROW_ID"].notna()) & (
        ~matched_strict.loc[open_mask & has_match].values)
    # Set STRICT first
    pl.loc[open_mask & has_match, "RESULT_MATCH_METHOD"] = "STRICT"
    pl.loc[open_mask & has_match, "RESULT_MATCH_CONFIDENCE"] = 1.00
    # Override shift
    if len(remaining_idx):
        shift_rows = open_mask & pl["__src_row"].notna() & (
            ~matched_strict.reindex(pl.index, fill_value=False))
        pl.loc[shift_rows, "RESULT_MATCH_METHOD"] = "DATE_SHIFT"
        pl.loc[shift_rows, "RESULT_MATCH_CONFIDENCE"] = 0.80

    # No match rows
    no_match_rows = open_mask & pl["__src_row"].isna()
    pl.loc[no_match_rows, "RESULT_MATCH_METHOD"] = "NO_MATCH"
    pl.loc[no_match_rows, "RESULT_MATCH_CONFIDENCE"] = 0.00

    # Write scores
    pl.loc[open_mask & has_match, "FT_Home"] = pd.to_numeric(
        pl.loc[open_mask & has_match, "__h"], errors="coerce")
    pl.loc[open_mask & has_match, "FT_Away"] = pd.to_numeric(
        pl.loc[open_mask & has_match, "__a"], errors="coerce")
    pl.loc[open_mask & has_match, "Total_Goals"] = pl.loc[open_mask &
        has_match, "FT_Home"] + pl.loc[open_mask & has_match, "FT_Away"]

    # Compute Hit + Profit for simple markets
    def _calc_hit(row):
        sel = str(row.get("Seçim", "")).strip()
        h = float(row.get("FT_Home", np.nan))
        a = float(row.get("FT_Away", np.nan))
        if not (h == h and a == a):
            return np.nan
        if sel in ["MS 1", "MS1", "1", "Home", "Ev",
            "MS 1 "] or sel.startswith("MS 1"):
            return 1.0 if h > a else 0.0
        if sel in ["MS 2", "MS2", "2", "Away", "Dep",
            "MS 2 "] or sel.startswith("MS 2"):
            return 1.0 if a > h else 0.0
        if sel in ["MS X", "MSX", "X", "Draw",
            "Ber"] or sel.startswith("MS X"):
            return 1.0 if h == a else 0.0
        # OU 2.5
        if "2.5 Üst" in sel or "Over 2.5" in sel or sel.strip() == "2.5Üst":
            return 1.0 if (h + a) > 2.5 else 0.0
        if "2.5 Alt" in sel or "Under 2.5" in sel or sel.strip() == "2.5Alt":
            return 1.0 if (h + a) < 2.5 else 0.0
        # BTTS
        if "KG Var" in sel or "BTTS" in sel or "Both Teams" in sel:
            return 1.0 if (h > 0 and a > 0) else 0.0
        if "KG Yok" in sel:
            return 1.0 if (h == 0 or a == 0) else 0.0
        return np.nan

    # Only compute for rows where we have scores
    calc_rows = open_mask & has_match
    if "Hit" not in pl.columns:
        pl["Hit"] = np.nan
    pl.loc[calc_rows, "Hit"] = pl.loc[calc_rows].apply(_calc_hit, axis=1)

    # Profit (1u): if Hit==1 -> odd-1 else -1
    if "Profit" not in pl.columns:
        pl["Profit"] = np.nan
    if "Odd" in pl.columns:
        odds = pd.to_numeric(pl.get("Odd"), errors="coerce")
        hitv = pd.to_numeric(pl.get("Hit"), errors="coerce")
        pl.loc[calc_rows, "Profit"] = np.where(
            hitv.loc[calc_rows] == 1.0, odds.loc[calc_rows] - 1.0, -1.0)

    # Cleanup
    pl = pl.drop(
    columns=[
        c for c in [
            "__date",
            "__k",
            "__src_row",
            "__h",
            "__a"] if c in pl.columns],
             errors="ignore")
    return pl


def _prediction_log_key_cols(df: pd.DataFrame) -> List[str]:
    # Use stable identifiers to dedupe (prefer Match_ID when available)
    cols = ["Snapshot_Date", "Match_ID", "Seçim"]
    cols_fallback = [
    "Snapshot_Date",
    "Date",
    "League",
    "HomeTeam",
    "AwayTeam",
     "Seçim"]
    k = [c for c in cols if c in df.columns]
    return k if len(k) >= 3 else [c for c in cols_fallback if c in df.columns]


def append_prediction_snapshot(df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    *,
    meta: dict) -> Tuple[bool,
     str]:
    """Append a daily snapshot to prediction log.

    Dedup rule: unique by (Snapshot_Date, Date, League, HomeTeam, AwayTeam, Seçim).
    Returns: (ok, message)
    """
    if df is None or getattr(df, "empty", True):
        return False, "Snapshot boş (df empty)."

    snap = df.copy()

    # Normalize canonical match id + opening odds
    snap["Match_ID"] = snap.apply(_make_match_id_row, axis=1)
    if "Odd_Open" not in snap.columns:
        # If you already have Opening odds in a dedicated column, map it
        # upstream.
        snap["Odd_Open"] = snap.get("Odd", np.nan)

    # Ensure Snapshot_Date
    snapshot_date = pd.to_datetime(snapshot_date, errors="coerce")
    if pd.isna(snapshot_date):
        snapshot_date = pd.Timestamp.utcnow().normalize()
    snap["Snapshot_Date"] = snapshot_date.normalize()

    # Ensure mandatory columns exist (soft defaults)
    for c, dflt in [
        ("CONF_ICON", "🟡"),
        ("AutoMod_Status", "Medium"),
        ("AMQS", 0.50),
        ("AMQS_percentile", 0.50),
        ("Final_Confidence", 0.50),
        ("CONF_percentile", 50.0),
        ("CONF_Status", "Medium"),
        ("BestOfRank", 0.0),
        ("Star_Rating", 1),
    ]:
        if c not in snap.columns:
            snap[c] = dflt

    # UX aliases
    if "AUTOMOD_ICON" not in snap.columns:
        snap["AUTOMOD_ICON"] = pd.Series(["🟡"] * len(snap), index=snap.index)
        if "AutoMod_Status" in snap.columns:
            snap["AUTOMOD_ICON"] = snap["AutoMod_Status"].map(
                {"High": "🟢", "Medium": "🟡", "Low": "🔴"}).fillna("🟡")
        elif "AMQS_percentile" in snap.columns:
            _p = pd.to_numeric(
    snap["AMQS_percentile"],
     errors="coerce").fillna(0.50)
            snap["AUTOMOD_ICON"] = np.select(
                [_p >= 0.80, _p >= 0.60], ["🟢", "🟡"], default="🔴")

    # Metadata
    snap["Logged_At"] = pd.Timestamp.utcnow()
    for k, v in (meta or {}).items():
        col = f"Meta_{k}"
        snap[col] = v

    # Keep only relevant columns (plus any extras user may want later)
    preferred_cols = [
        "Snapshot_Date",
        "Match_ID",
        "Date", "League",
        "CONF_ICON", "AUTOMOD_ICON",
        "HomeTeam", "AwayTeam",
        "Seçim", "Odd_Open", "Odd", "Prob", "EV",
        "Star_Rating", "BestOfRank", "MDL_rank", "MDL_quantile", "STRENGTH_ICON", "StrengthCategory", "AnchorStrengthCategory", "Score", "GoldenScore",
        "SIM_ANCHOR_STRENGTH", "SIM_QUALITY", "EFFECTIVE_N",

        "League_Conf", "Market_Conf_Score", "Final_Confidence",
        "CONF_percentile", "CONF_Status",
        "AMQS", "AMQS_percentile", "AutoMod", "AutoMod_Status",
        "Logged_At",
    ]
    cols_out = [c for c in preferred_cols if c in snap.columns]
    # Keep all Meta_* columns
    cols_out += [c for c in snap.columns if c.startswith(
        "Meta_") and c not in cols_out]
    snap = snap[cols_out].copy()

    # Dedup with existing
    existing = load_prediction_log()
    if existing is None or existing.empty:
        out = snap
    else:
        out = pd.concat([existing, snap], ignore_index=True)
        kcols = _prediction_log_key_cols(out)
        if kcols:
            out = out.drop_duplicates(subset=kcols, keep="last")

    out.to_parquet(PRED_LOG_FILE, index=False)
    return True, (
        f"Prediction log yazıldı: {len(snap)} satır "
        f"(dosya: {PRED_LOG_FILE.name})."
    )



def backfill_prediction_log_from_hist_df(hist_df: pd.DataFrame,
                                         start_date: Optional[pd.Timestamp] = None,
                                         end_date: Optional[pd.Timestamp] = None,
                                         *,
                                         meta: Optional[dict] = None,
                                         max_days: Optional[int] = None) -> Tuple[int, int, List[str]]:
    """
    Retro-create daily snapshots from existing model outputs (hist_df).

    IMPORTANT:
    - This does NOT retrain models historically. It only snapshots whatever rows already exist in hist_df.
    - It is still extremely useful to:
        * Stabilize WF Lab optimization (fixed candidates per day)
        * Extend Prediction Log coverage within the available hist_df period
    Returns: (days_appended, days_skipped, messages)
    """
    msgs: List[str] = []
    if hist_df is None or hist_df.empty or "Date" not in hist_df.columns:
        return 0, 0, ["hist_df boş veya Date kolonu yok."]

    df = hist_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    if start_date is not None:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["Date"] <= pd.to_datetime(end_date)]
    if df.empty:
        return 0, 0, ["Seçilen tarih aralığında hist_df satırı yok."]

    # group by match date (not snapshot date). Snapshot_Date = Date (same day)
    # for retro fill.
    unique_days = sorted(df["Date"].dt.normalize().unique())
    if max_days is not None:
        unique_days = unique_days[: int(max_days)]

    days_appended = 0
    days_skipped = 0
    for d in unique_days:
        day_df = df[df["Date"].dt.normalize() == pd.Timestamp(d)].copy()
        if day_df.empty:
            days_skipped += 1
            continue

        ok, msg = append_prediction_snapshot(
            day_df,
            pd.Timestamp(d),
            meta=(meta or {"mode": "retro_from_hist_df"})
        )
        if ok:
            days_appended += 1
        else:
            days_skipped += 1
        # keep only a few messages to avoid flooding UI
        if msg and len(msgs) < 10:
            msgs.append(f"{pd.Timestamp(d).date()}: {msg}")

    return days_appended, days_skipped, msgs


def load_ui_defaults() -> dict:
#     try:  # AUTO-COMMENTED (illegal global try)
#         if DEFAULTS_FILE.exists():
#             return json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
#     except Exception:
#         pass
    return {}


def save_ui_defaults(d: dict) -> None:
    DEFAULTS_FILE.write_text(
        json.dumps(
            d,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )



# -----------------------------
# Walk-Forward Filter Lab (Policy Finder)
# -----------------------------
def _detect_ft_cols(df: pd.DataFrame):
    """Try to detect full-time score columns for home/away goals."""
    candidates = [
        ("FT_Score_Home", "FT_Score_Away"),
        ("HomeFullTimeScore", "AwayFullTimeScore"),
        ("FTHG", "FTAG"),
        ("HG", "AG"),
    ]
    for h, a in candidates:
        if h in df.columns and a in df.columns:
            return h, a
    return None, None


def _calc_hit_profit(selection: str, odd: float, hg: float, ag: float):
    """Return (hit, profit_1u) for supported markets."""
    if selection is None or selection == "":
        return (np.nan, np.nan)
#     try:  # AUTO-COMMENTED (illegal global try)
#         odd = float(odd)
#     except Exception:
#         return (np.nan, np.nan)
#     try:
#         hg = float(hg)
#         ag = float(ag)
#     except Exception:
        return (np.nan, np.nan)
    if np.isnan(hg) or np.isnan(ag) or np.isnan(odd):
        return (np.nan, np.nan)
    total = hg + ag
    sel = str(selection).strip()

    hit = None
    if sel == "MS 1":
        hit = hg > ag
    elif sel in ("MS X", "MSX", "X"):
        hit = hg == ag
    elif sel == "MS 2":
        hit = hg < ag
    elif sel in ("2.5 Üst", "2.5 Ust", "Over 2.5", "O2.5"):
        hit = total >= 3
    elif sel in ("2.5 Alt", "Under 2.5", "U2.5"):
        hit = total <= 2
    elif sel in ("KG Var", "BTTS Yes", "BTTS-Yes", "KGV", "BTTS"):
        hit = (hg > 0) and (ag > 0)
    elif sel in ("KG Yok", "BTTS No", "BTTS-No", "KGY"):
        hit = not ((hg > 0) and (ag > 0))
    else:
        # Unknown market
        return (np.nan, np.nan)

    profit = (odd - 1.0) if hit else -1.0
    return (1 if hit else 0, profit)


def _apply_policy(df: pd.DataFrame,
                  min_bestof: float,
                  min_conf_pct: float,
                  min_ev: float,
                  min_prob: float,
                  min_automod_pct: float,
                  # --- YENİ PARAMETRELER ---
                  min_anchor: float = 0.0,
                  min_sim_quality: float = 0.0,
                  min_eff_n: int = 0,
                  # -------------------------
                  allowed_markets: tuple | list | None = None,
                  odd_min: float = None,
                  odd_max: float = None,
                  use_open_odds: bool = True) -> pd.DataFrame:
    """Filter + dedup: Yeni metrikler (Anchor, SimQ, EffN) eklendi."""
    if df.empty:
        return df

        # Required columns check + ROBUST alias mapping
    odd_col = "Odd_Open" if (
    use_open_odds and (
        "Odd_Open" in df.columns)) else (
            "Odd" if "Odd" in df.columns else None)

    _df = df.copy()

    def _first_existing(cands):
        for c in cands:
            if c in _df.columns:
                return c
        return None

    # --- Ensure core metric columns exist (create from aliases if needed) ---
    if "BestOfRank" not in _df.columns:
        alt = _first_existing(
            ["BestOf_Rank", "BESTOF_RANK", "BestOf", "BestOfScore", "BestOf_Score"])
        if alt is not None:
            _df["BestOfRank"] = _df[alt]

    if "CONF_percentile" not in _df.columns:
        alt = _first_existing(
            ["CONF_pct", "CONF_PCT", "CONF", "Conf", "CONF_SCORE", "CONF_Score"])
        if alt is not None:
            _df["CONF_percentile"] = _df[alt]

    if "AutoMod_percentile" not in _df.columns:
        alt = _first_existing(["AUTOMOD_percentile",
    "AMQS_percentile",
    "AUTOMOD",
    "AutoMod",
    "AMQS",
     "AutoModScore"])
        if alt is not None:
            _df["AutoMod_percentile"] = _df[alt]

    if "Prob" not in _df.columns:
        alt = _first_existing(
            ["PROB", "prob", "P", "P1", "p1", "Prob_1", "P_Home", "P_home"])
        if alt is not None:
            _df["Prob"] = _df[alt]

    if "EV" not in _df.columns:
        alt = _first_existing(
            ["EV_Value", "Value", "Edge", "EDGE", "ExpectedValue", "expected_value"])
        if alt is not None:
            _df["EV"] = _df[alt]

    # Optional metrics
    if "Anchor_Strength" not in _df.columns:
        alt = _first_existing(
            ["ANCHOR_STRENGTH", "ANCHOR", "Anchor", "anchor_strength"])
        if alt is not None:
            _df["Anchor_Strength"] = _df[alt]
    if "Sim_Quality" not in _df.columns:
        alt = _first_existing(
            ["SIM_QUALITY", "SIMQ", "SimQuality", "sim_quality"])
        if alt is not None:
            _df["Sim_Quality"] = _df[alt]
    if "Effective_N" not in _df.columns:
        alt = _first_existing(["EFFECTIVE_N", "EffN", "eff_n", "N_effective"])
        if alt is not None:
            _df["Effective_N"] = _df[alt]

    # Fallback defaults (so we don't return empty just because logs are
    # partial)
    for col, default in [
        ("BestOfRank", 0.0),
        ("CONF_percentile", 0.0),
        ("AutoMod_percentile", 0.0),
        ("Prob", 0.5),
        ("EV", -999.0),
        ("Anchor_Strength", 0.0),
        ("Sim_Quality", 0.0),
        ("Effective_N", 0),
    ]:
        if col not in _df.columns:
            _df[col] = default

    # Minimal required columns (odd_col may be missing in some logs; if so,
    # skip odd filtering)
    must = ["Date", "League", "HomeTeam", "AwayTeam", "Seçim"]
    for c in must:
        if c not in _df.columns:
            return _df.iloc[0:0].copy()
    # Ensure numeric
    for c in [
    "BestOfRank",
    "CONF_percentile",
    "EV",
    "Prob",
    "AutoMod_percentile",
     odd_col]:
        _df[c] = pd.to_numeric(_df[c], errors="coerce")

    # --- YENİ METRİKLERİ HAZIRLA (Eksikse varsayılan değer ata) ---
    if "SIM_ANCHOR_STRENGTH" in _df.columns:
        _df["SIM_ANCHOR_STRENGTH"] = pd.to_numeric(
    _df["SIM_ANCHOR_STRENGTH"], errors="coerce").fillna(0.0)
    else:
        _df["SIM_ANCHOR_STRENGTH"] = 1.0  # Kolon yoksa eleme yapma

    if "SIM_QUALITY" in _df.columns:
        _df["SIM_QUALITY"] = pd.to_numeric(
    _df["SIM_QUALITY"], errors="coerce").fillna(0.0)
    else:
        _df["SIM_QUALITY"] = 1.0  # Kolon yoksa eleme yapma

    if "EFFECTIVE_N" in _df.columns:
        _df["EFFECTIVE_N"] = pd.to_numeric(
    _df["EFFECTIVE_N"], errors="coerce").fillna(0.0)
    else:
        _df["EFFECTIVE_N"] = 100.0  # Kolon yoksa eleme yapma

    mask = (
        (_df["BestOfRank"] >= min_bestof) &
        (_df["CONF_percentile"] >= min_conf_pct) &
        (_df["EV"] >= min_ev) &
        (_df["Prob"] >= min_prob) &
        (_df["AutoMod_percentile"] >= min_automod_pct) &
        # --- YENİ FİLTRELER ---
        (_df["SIM_ANCHOR_STRENGTH"] >= min_anchor) &
        (_df["SIM_QUALITY"] >= min_sim_quality) &
        (_df["EFFECTIVE_N"] >= min_eff_n)
    )

    # Optional market filter
    if allowed_markets:
        def _norm_pick(x):
            s = str(x).strip().upper().replace(" ", "")
            if s in ["MS1", "1"]:
                return "MS1"
            if s in ["MSX", "X", "DRAW"]:
                return "MSX"
            if s in ["MS2", "2"]:
                return "MS2"
            if s in [
    "2.5ÜST",
    "25ÜST",
    "2.5UST",
    "25UST",
    "O25",
    "OVER25",
    "OVER2.5",
    "2.5OVER",
     "2.5O"]:
                return "O25"
            if s in [
    "2.5ALT",
    "25ALT",
    "2.5UNDER",
    "U25",
    "UNDER25",
    "UNDER2.5",
     "2.5U"]:
                return "U25"
            if s in ["KGVAR", "BTTSYES", "BTTSY", "GG", "BOTHSCORING"]:
                return "BTTS_Y"
            if s in ["KGYOK", "BTTSNO", "BTTSN", "NG", "BOTHNOSCORE"]:
                return "BTTS_N"
            return s

        allowed_set = set(_norm_pick(x) for x in allowed_markets)
        mask = mask & _df["Seçim"].apply(_norm_pick).isin(allowed_set)

    # Optional odds band filter
    if odd_min is not None:
        mask = mask & (_df[odd_col] >= float(odd_min))
    if odd_max is not None:
        mask = mask & (_df[odd_col] <= float(odd_max))

    _df = _df.loc[mask].copy()
    if _df.empty:
        return _df

    # Create match_id and pick best per match (Dedup)
    if "Match_ID" in _df.columns:
        _df["__match_id__"] = _df["Match_ID"].astype(str)
    else:
        _df["__match_id__"] = (
            _df["Date"].astype(str) + "|" +
            _df["League"].astype(str) + "|" +
            _df["HomeTeam"].astype(str) + "|" +
            _df["AwayTeam"].astype(str)
        )

    # Tie-break score
    _df["__dedup_score__"] = pd.to_numeric(_df["CONF_percentile"], errors="coerce").fillna(
        -1) * 1000.0 + pd.to_numeric(_df["BestOfRank"], errors="coerce").fillna(-1)
    idx = _df.groupby("__match_id__")["__dedup_score__"].idxmax()
    out = _df.loc[idx].drop(
    columns=[
        "__match_id__",
        "__dedup_score__"],
         errors="ignore").copy()
    return out


def _calc_hit_profit_proxy(
    selection: str,
    hg: float,
    ag: float,
    win_profit: float = 1.0,
     lose_profit: float = -1.0):
    """Odds yoksa (signal-only WF) için: (hit, proxy_profit_1u).
    Varsayılan olarak 1 unit stake ve 'even odds' proxy kullanır (+1 / -1).
    """
    if selection is None or selection == "":
        return (np.nan, np.nan)
#     try:  # AUTO-COMMENTED (illegal global try)
#         hg = float(hg)
#         ag = float(ag)
#     except Exception:
#         return (np.nan, np.nan)
    if np.isnan(hg) or np.isnan(ag):
        return (np.nan, np.nan)

    total = hg + ag
    sel = str(selection).strip()

    hit = None
    if sel == "MS 1":
        hit = hg > ag
    elif sel in ("MS X", "MSX", "X"):
        hit = hg == ag
    elif sel == "MS 2":
        hit = hg < ag
    elif sel in ("2.5 Üst", "O2.5", "Over 2.5", "OVER"):
        hit = total > 2.5
    elif sel in ("2.5 Alt", "U2.5", "Under 2.5", "UNDER"):
        hit = total < 2.5
    elif sel in ("KG Var", "BTTS Yes", "BTTSY", "BTTS"):
        hit = (hg > 0) and (ag > 0)
    elif sel in ("KG Yok", "BTTS No", "BTTSN"):
        hit = not ((hg > 0) and (ag > 0))
    else:
        return (np.nan, np.nan)

    profit = win_profit if bool(hit) else lose_profit
    return (float(bool(hit)), float(profit))


# -------------------------------------------------------
# WF canonical flags (safe defaults – avoid NameError)
# -------------------------------------------------------
exploration_mode = bool(locals().get('exploration_mode', False))
hard_sanity_mode = bool(locals().get('hard_sanity_mode', False))
signal_only = bool(locals().get('signal_only', False))

# -------------------------------------------------------
# WF canonical numeric params (safe defaults – avoid NameError)
# -------------------------------------------------------
_wf_topk = locals().get(
    'candidate_topk_per_day',
    locals().get(
        'topk_candidates',
        locals().get(
            'topk_per_day',
             60)))
_wf_per_match = locals().get(
    'candidate_topk_per_match_per_day',
    locals().get(
        'per_match_cap',
        locals().get(
            'per_match_max',
             1)))
_wf_topk = int(_wf_topk) if _wf_topk is not None else 60
_wf_per_match = int(_wf_per_match) if _wf_per_match is not None else 1


@st.cache_data(show_spinner=False)
def _run_walk_forward_policy_search(

    hist_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_days: int = 90,
    test_days: int = 14,
    bestof_grid=(0.45, 0.50, 0.55, 0.60),
    conf_grid=(60, 70, 80),
    ev_grid=(0, 2, 4),
    prob_grid=(0.45, 0.50, 0.55),
    automod_grid=(40, 60, 75),
    # --- YENİ GRIDLER ---
    anchor_grid=(0.0,),
    simq_grid=(0.0,),
    eff_n_grid=(0,),
    # --------------------
    market_filter=None,
    odd_grid=((None, None), (1.40, 2.20), (1.50, 2.50), (1.60, 3.00)),
    use_open_odds: bool = True,
    min_picks_per_day: float = 2.0,
    min_total_picks_train: int = 200,
    signal_only: bool = False, candidate_pool_mode: bool = False,
    candidate_topk_per_day: int = 50,
    candidate_topk_per_match_per_day: int = 1,
    # --- compatibility aliases (UI wiring) ---
    topk_candidates: int | None = None,
    per_match_cap: int | None = None,
    hard_sanity_mode: bool = False,
    hard_sanity: bool | None = None,  # alias (old keyword)
    exploration_mode: bool = False,
    quick_mode: bool = False,
    **kwargs,
):
    # Backward-compat: accept old keyword 'hard_sanity'
    if hard_sanity is not None:
        hard_sanity_mode = bool(hard_sanity)

    # -------------------------------------------------------
    # WF DEBUG: belirli market(ler)i izole test etmek için
    # -------------------------------------------------------
    ONLY_MARKET = None  # örn: "MS1" / "MS2" / "MSX"  | None = kapalı

    if ONLY_MARKET:
        def _norm_pick(x):
            s = str(x).strip().upper().replace(" ", "")
            if s in ["MS1", "1", "HOME", "H"]:
                return "MS1"
            if s in ["MS2", "2", "AWAY", "A"]:
                return "MS2"
            if s in ["MSX", "X", "DRAW", "D"]:
                return "MSX"
            return s

        if "Seçim" in hist_df.columns:
            _tmp = hist_df.copy()
            _tmp["__PICK_N__"] = _tmp["Seçim"].apply(_norm_pick)
            target = _norm_pick(ONLY_MARKET)
            hist_df = _tmp[_tmp["__PICK_N__"] == target].drop(
                columns=["__PICK_N__"], errors="ignore").copy()

    """Walk-forward: choose best policy on train window, apply to next test window."""
    # -------------------------------------------------------
    # Compatibility mapping: accept legacy/wired param names
    # -------------------------------------------------------
    if topk_candidates is not None:
        candidate_topk_per_day = int(topk_candidates)
    if per_match_cap is not None:
        candidate_topk_per_match_per_day = int(per_match_cap)

    if hist_df is None or hist_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = hist_df.copy()
    # Normalize Date to date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------------------------------------------------------------------
    # Candidate Pool Mode (WF-friendly)
    # Many pipelines feed WF with an already narrowed list (often 1 pick/day).
    # Here we explicitly create a wider candidate pool per day (and optionally per match),
    # then let the policy thresholds do the filtering.
    # ---------------------------------------------------------------------
    def _wf_expand_markets_to_candidates(
    _df: pd.DataFrame,
     use_open_odds: bool = False) -> pd.DataFrame:
        if _df is None or _df.empty:
            return _df

        def _col_first(cands):
            for c in cands:
                if c in _df.columns:
                    return c
            return None

        # Prob columns (support multiple naming conventions)
        p_home = _col_first(
            ["P_Home_Final", "P_Home", "Prob_Home", "Prob_MS1"])
        p_draw = _col_first(
            ["P_Draw_Final", "P_Draw", "Prob_Draw", "Prob_MSX", "Prob_X"])
        p_away = _col_first(
            ["P_Away_Final", "P_Away", "Prob_Away", "Prob_MS2"])
        p_over = _col_first(["P_Over_Final", "P_Over", "Prob_Over"])
        p_btts = _col_first(["P_BTTS_Final", "P_BTTS", "Prob_BTTS"])

        # Odds columns (open vs closing)
        if bool(use_open_odds):
            o_home = _col_first(
                ["Odds_Open_Home", "OpenHomeOdd", "Open_Home_Odd"])
            o_draw = _col_first(
                ["Odds_Open_Draw", "OpenDrawOdd", "Open_Draw_Odd"])
            o_away = _col_first(
                ["Odds_Open_Away", "OpenAwayOdd", "Open_Away_Odd"])
            o_o25 = _col_first(["Odds_Open_Over25",
    "Odds_Open_Over2.5",
    "OpenO25",
    "OpenOver25",
     "Open_Over25"])
            o_u25 = _col_first(["Odds_Open_Under25",
    "Odds_Open_Under2.5",
    "OpenU25",
    "OpenUnder25",
     "Open_Under25"])
            o_by = _col_first(
                ["Odds_Open_BTTS_Yes", "OpenBTTSY", "Open_BTTSY"])
            o_bn = _col_first(["Odds_Open_BTTS_No", "OpenBTTSN", "Open_BTTSN"])
        else:
            o_home = _col_first(["ClosingHomeOdd", "HomeOdd", "Odd_Home"])
            o_draw = _col_first(["ClosingDrawOdd", "DrawOdd", "Odd_Draw"])
            o_away = _col_first(["ClosingAwayOdd", "AwayOdd", "Odd_Away"])
            o_o25 = _col_first(["ClosingO25", "O25", "Odd_Over25", "Odd_O25"])
            o_u25 = _col_first(["ClosingU25", "U25", "Odd_Under25", "Odd_U25"])
            o_by = _col_first(["ClosingBTTSY", "BTTSY", "Odd_BTTSY"])
            o_bn = _col_first(["ClosingBTTSN", "BTTSN", "Odd_BTTSN"])

        # If we don't have multi-market probabilities, do nothing
        has_multi = any([p_home, p_draw, p_away, p_over, p_btts])
        if not has_multi:
            return _df

        def _num(s):
            return pd.to_numeric(s, errors="coerce")

        # -----------------------------------------------------------------
        # Anchor-guided candidate generation
        # We generate candidates primarily from the anchor market family for each match,
        # and optionally from ONE secondary family (chosen by conviction among the remaining families).
        # This prevents situations like "BTTS strong but only OU survives" due to global scoring.
        # -----------------------------------------------------------------
        anchor_col = "SIM_ANCHOR_GROUP" if "SIM_ANCHOR_GROUP" in _df.columns else None

        # Decide secondary family per row (vectorized) using simple conviction proxies
        #   MS conviction  : gap between top1 and top2 among (P1,PX,P2)
        #   OU conviction  : |POver - 0.5|
        #   BTTS conviction: |PBTTS - 0.5|
        # If a required prob column is missing, conviction becomes 0 for that
        # family.
        p1s = _num(
    _df[p_home]) if p_home in _df.columns else pd.Series(
        0.0, index=_df.index)
        pxs = _num(
    _df[p_draw]) if p_draw in _df.columns else pd.Series(
        0.0, index=_df.index)
        p2s = _num(
    _df[p_away]) if p_away in _df.columns else pd.Series(
        0.0, index=_df.index)
        ms_sorted = pd.concat([p1s, pxs, p2s], axis=1).apply(
            lambda r: sorted(r.values, reverse=True), axis=1)
        ms_conv = ms_sorted.apply(lambda v: float(
            v[0] - v[1]) if len(v) >= 2 else 0.0)

        povs = _num(
    _df[p_over]) if p_over in _df.columns else pd.Series(
        0.0, index=_df.index)
        ou_conv = (povs - 0.5).abs()

        pbtts = _num(
    _df[p_btts]) if p_btts in _df.columns else pd.Series(
        0.0, index=_df.index)
        btts_conv = (pbtts - 0.5).abs()

        if anchor_col:
            anc = _df[anchor_col].astype(str).str.upper().fillna("")
            # Normalize to MS/OU/BTTS
            anc = anc.replace({"1X2": "MS", "RESULT": "MS",
                              "OU": "OU", "O/U": "OU", "BTTS": "BTTS"})
        else:
            anc = pd.Series("", index=_df.index)

        # Choose one secondary family among the two non-anchor families
        # If anchor is missing/unknown, allow all families (backwards
        # compatible).
        def _choose_secondary(a, ms, ou, bt):
            fams = {"MS": ms, "OU": ou, "BTTS": bt}
            keys = ["MS", "OU", "BTTS"]
            if a in keys:
                keys = [k for k in keys if k != a]
            # pick max conviction among remaining
            best = max(keys, key=lambda k: fams.get(k, 0.0))
            return best

        secondary = pd.Series([_choose_secondary(a, ms, ou, bt) for a, ms, ou, bt in zip(
            anc, ms_conv, ou_conv, btts_conv)], index=_df.index)

        # Policy: allow anchor family always; allow ONE secondary family (secondary) as "backup".
        # If anchor unknown, allow all families.
        def _allow_family(fam):
            if not anchor_col:
                return pd.Series(True, index=_df.index)
            fam = fam.upper()
            return (anc == fam) | ((anc != "") & (secondary == fam))

        rows = []
        for mkt, pcol, ocol, sel, inv_prob, fam in [
            ("MS 1", p_home, o_home, "MS 1", False, "MS"),
            ("MS X", p_draw, o_draw, "MS X", False, "MS"),
            ("MS 2", p_away, o_away, "MS 2", False, "MS"),
            ("2.5 Üst", p_over, o_o25, "2.5 Üst", False, "OU"),
            ("2.5 Alt", p_over, o_u25, "2.5 Alt", True, "OU"),
            ("KG Var", p_btts, o_by, "KG Var", False, "BTTS"),
            ("KG Yok", p_btts, o_bn, "KG Yok", True, "BTTS"),
        ]:
            if pcol is None or ocol is None:
                continue
            # Anchor-guided filtering disabled: keep full multi-market pool so
            # rule engine can choose.
            base = _df.copy()
            tmp = base
            prob = _num(tmp[pcol])
            if inv_prob:
                prob = 1.0 - prob
            odd = _num(tmp[ocol])
            tmp["Market"] = mkt
            tmp["Seçim"] = sel
            tmp["Selection"] = tmp.get("Selection", tmp["Seçim"])
            tmp["Prob"] = prob
            tmp["Odd"] = odd
            tmp["EV"] = (tmp["Prob"] * tmp["Odd"]) - 1.0

            # Tag anchor/secondary for scoring
            if anchor_col:
                tmp["IS_ANCHOR_MARKET"] = (
                    anc.loc[tmp.index] == fam).astype(int)
                tmp["IS_SECONDARY_MARKET"] = ((anc.loc[tmp.index] != "") & (
                    secondary.loc[tmp.index] == fam) & (anc.loc[tmp.index] != fam)).astype(int)
            else:
                tmp["IS_ANCHOR_MARKET"] = 0
                tmp["IS_SECONDARY_MARKET"] = 0

            rows.append(tmp)

        cand = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        out = cand
        out = out.dropna(subset=["Prob", "Odd"]).copy()
        out = out[(out["Odd"] > 1.0) & (out["Prob"] >= 0.0)
                   & (out["Prob"] <= 1.0)].copy()
        out = apply_rule_engine_to_candidates(out)
        return out

    def _wf_candidate_pool(_df: pd.DataFrame) -> pd.DataFrame:
        if _df is None or _df.empty:
            return _df
        cand = _df.copy()
        cand = _wf_expand_markets_to_candidates(cand)

        # Score used only for ranking candidates (not a hard filter)
        score = 0.0
        if "EV" in cand.columns:
            score = score + \
                pd.to_numeric(cand["EV"], errors="coerce").fillna(0.0) * 1.0
        if "Prob" in cand.columns:
            score = score + \
                pd.to_numeric(cand["Prob"], errors="coerce").fillna(0.0) * 0.20
        if "AutoMod_percentile" in cand.columns:
            score = score + \
                pd.to_numeric(
    cand["AutoMod_percentile"],
     errors="coerce").fillna(0.0) * 0.01
        if "CONF_percentile" in cand.columns:
            score = score + \
                pd.to_numeric(
    cand["CONF_percentile"],
     errors="coerce").fillna(0.0) * 0.01

        # Anchor-guided ranking tweak (safe, small effect)
        if "IS_ANCHOR_MARKET" in cand.columns:
            score = score * \
                (1.0 + 0.10 *
                 pd.to_numeric(cand["IS_ANCHOR_MARKET"], errors="coerce").fillna(0.0))
        if "IS_SECONDARY_MARKET" in cand.columns:
            score = score * \
                (1.0 - 0.05 *
                 pd.to_numeric(cand["IS_SECONDARY_MARKET"], errors="coerce").fillna(0.0))
        cand["__WF_SCORE__"] = score

        # Ensure date
        cand["Date"] = pd.to_datetime(cand["Date"], errors="coerce")
        cand = cand.dropna(subset=["Date"]).copy()

        # Optional: limit per match per day (prevents same match flooding)
        team_cols = [
    c for c in [
        "HomeTeam",
        "AwayTeam",
        "home_team",
         "away_team"] if c in cand.columns]
        if candidate_topk_per_match_per_day and candidate_topk_per_match_per_day > 0 and len(
            team_cols) >= 2:
            def _topk_match(g):
                return g.sort_values(
    "__WF_SCORE__", ascending=False).head(
        int(candidate_topk_per_match_per_day))
            cand = cand.groupby([cand["Date"].dt.date,
    team_cols[0],
    team_cols[1]],
    dropna=False,
     group_keys=False).apply(_topk_match)

        # Top-K per day
        k = int(candidate_topk_per_day) if candidate_topk_per_day else 0
        if k > 0:
            cand = cand.groupby(
    cand["Date"].dt.date,
    dropna=False,
    group_keys=False).apply(
        lambda g: g.sort_values(
            "__WF_SCORE__",
             ascending=False).head(k))

        return cand.drop(columns=["__WF_SCORE__"], errors="ignore")

    # detect FT cols and compute profit
    hcol, acol = _detect_ft_cols(df)
    if hcol is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df[hcol] = pd.to_numeric(df[hcol], errors="coerce")
    df[acol] = pd.to_numeric(df[acol], errors="coerce")
    df["Odd"] = pd.to_numeric(df.get("Odd"), errors="coerce")
    if bool(signal_only):
        df["Hit"], df["Profit_1u"] = zip(
            *df.apply(lambda r: _calc_hit_profit_proxy(r.get("Seçim"), r.get(hcol), r.get(acol)), axis=1))
    else:
        df["Hit"], df["Profit_1u"] = zip(*df.apply(lambda r: _calc_hit_profit(
            r.get("Seçim"), r.get("Odd"), r.get(hcol), r.get(acol)), axis=1))
    df["Hit"] = pd.to_numeric(df["Hit"], errors="coerce")
    df["Profit_1u"] = pd.to_numeric(df["Profit_1u"], errors="coerce")
    df_valid = df.dropna(subset=["Profit_1u"]).copy()
    if df_valid.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # dates
    min_d = df_valid["Date"].min().normalize()
    max_d = df_valid["Date"].max().normalize()
    # align
    start = max(start_date.normalize(), min_d + pd.Timedelta(days=train_days))
    end = min(end_date.normalize(), max_d)
    block_starts = []
    cur = start
    while cur + pd.Timedelta(days=test_days - 1) <= end:
        block_starts.append(cur)
        cur = cur + pd.Timedelta(days=test_days)

    results = []
    chosen = []

    for b0 in block_starts:
        train_start = b0 - pd.Timedelta(days=train_days)
        train_end = b0 - pd.Timedelta(days=1)
        test_start = b0
        test_end = b0 + pd.Timedelta(days=test_days - 1)

        train_df = df_valid[(df_valid["Date"] >= train_start) & (
            df_valid["Date"] <= train_end)].copy()
        test_df = df_valid[(df_valid["Date"] >= test_start)
                            & (df_valid["Date"] <= test_end)].copy()
        if train_df.empty or test_df.empty:
            continue

        # Candidate Pool Mode: widen the universe BEFORE policy grid search
        if bool(candidate_pool_mode):
            train_df = _wf_candidate_pool(train_df)
            test_df = _wf_candidate_pool(test_df)
            if (train_df is None) or (
                test_df is None) or train_df.empty or test_df.empty:
                continue

        train_days_count = max(1, train_df["Date"].dt.normalize().nunique())

        best = None
        best_score = -1e18
        best_stats = None

        # grid search on train
        for mb in bestof_grid:
            for mc in conf_grid:
                for me in ev_grid:
                    for mp in prob_grid:
                        for ma in automod_grid:
                            # --- YENİ DÖNGÜ KATMANLARI ---
                            for m_anc in anchor_grid:
                                for m_simq in simq_grid:
                                    for m_effn in eff_n_grid:
                                        # -----------------------------
                                        for (
    omin, omax) in (
        odd_grid or [
            (None, None)]):
                                            if bool(signal_only):
                                                omin, omax = (None, None)

                                            sel_train = _apply_policy(
                                                train_df,
                                                mb, mc, me, mp, ma,
                                                min_anchor=m_anc,
                                                min_sim_quality=m_simq,
                                                min_eff_n=m_effn,
                                                odd_min=omin,
                                                odd_max=omax,
                                                allowed_markets=market_filter,
                                                use_open_odds=use_open_odds,
                                            )
                                            if sel_train.empty:
                                                continue
                                            n = len(sel_train)
                                            low_sample = bool(
                                                n < int(min_total_picks_train))
                                            # In exploration/Hard Sanity mode we do NOT hard-reject low-sample policies.
                                            # Exploration-friendly: only
                                            # hard-reject low-sample when Hard
                                            # Sanity is ON
                                            if low_sample and bool(
                                                hard_sanity_mode):
                                                continue

                                            # train yoğunluk (bilgi amaçlı)
                                            avg_ppd = n / train_days_count

                                            profit = float(
                                                sel_train["Profit_1u"].sum())
                                            roi = profit / max(1, n)

                                            # penalize volatility lightly: std
                                            # of daily pnl
                                            daily = sel_train.groupby(
    sel_train["Date"].dt.normalize())["Profit_1u"].sum()
                                            vol = float(daily.std()) if len(
                                                daily) > 1 else 0.0

                                            # sample-size penalty (keeps tiny
                                            # policies from dominating when
                                            # hard_sanity_mode is ON)
                                            if low_sample and int(
                                                min_total_picks_train) > 0:
                                                sample_pen = 0.10 * \
                                                    (int(min_total_picks_train) - n) / \
                                                     float(
                                                         int(min_total_picks_train))
                                            else:
                                                sample_pen = 0.0

                                            score = roi - 0.02 * vol - sample_pen

                                            if score > best_score:
                                                best_score = score
                                                best = (
    mb, mc, me, mp, ma, m_anc, m_simq, m_effn, omin, omax)
                                                best_stats = (
                                                    roi, vol, n, avg_ppd)

        if best is None:
            continue

        mb, mc, me, mp, ma, m_anc, m_simq, m_effn, omin, omax = best
        sel_test = _apply_policy(
            test_df,
            mb, mc, me, mp, ma,
            min_anchor=m_anc,
            min_sim_quality=m_simq,
            min_eff_n=m_effn,
            odd_min=omin,
            odd_max=omax,
            allowed_markets=market_filter,
            use_open_odds=use_open_odds,
        )
        test_days_count = max(1, test_df["Date"].dt.normalize().nunique())
        if sel_test.empty:
            test_roi = np.nan
            test_n = 0
            test_profit = 0.0
            test_ppd = 0.0
            test_unstable = True
        else:
            test_n = len(sel_test)
            test_profit = float(sel_test["Profit_1u"].sum())
            test_roi = test_profit / test_n if test_n else np.nan
            test_ppd = float(test_n) / float(test_days_count)
            test_unstable = bool(test_ppd < float(min_picks_per_day))

        # Flags (exploration-friendly): never block, just label
        train_n = int(best_stats[2]) if best_stats is not None else 0
        train_low_sample = bool(
    (int(min_total_picks_train) > 0) and (
        train_n < int(min_total_picks_train)))
        test_low_ppd = bool(
    (float(min_picks_per_day) > 0) and (
        float(test_ppd) < float(min_picks_per_day)))

        _flags = []
        if train_low_sample:
            _flags.append("LOW_SAMPLE_TRAIN")
        if test_low_ppd:
            _flags.append("LOW_PPD_TEST")
        flag_str = ",".join(_flags) if _flags else "OK"

        results.append({
            "TestWindowStart": test_start.date(),
            "TestWindowEnd": test_end.date(),
            "TrainStart": train_start.date(),
            "TrainEnd": train_end.date(),
            "Best_min_BestOfRank": mb,
            "Best_min_CONF_pct": mc,
            "Best_min_EV": me,
            "Best_min_Prob": mp,
            "Best_min_AutoMod_pct": ma,
            "Best_min_Anchor": m_anc,
            "Best_min_SimQ": m_simq,
            "Best_min_EffN": m_effn,
            "Market_Filter": (None if not market_filter else list(market_filter)),
            "Best_Odd_Min": (None if omin is None else float(omin)),
            "Best_Odd_Max": (None if omax is None else float(omax)),
            "Train_ROI": best_stats[0],
            "Train_Vol": best_stats[1],
            "Train_Picks": best_stats[2],
            "Train_PicksPerDay": best_stats[3],
            "Test_ROI": test_roi,
            "Test_Profit": test_profit,
            "Test_Picks": test_n,
            "Test_PicksPerDay": test_ppd,
            "Test_Unstable": test_unstable,
            "Train_LowSample": train_low_sample,
            "Test_LowPPD": test_low_ppd,
            "Flags": flag_str
        })
        chosen.append({
            "min_BestOfRank": mb,
            "min_CONF_pct": mc,
            "min_EV": me,
            "min_Prob": mp,
            "min_AutoMod_pct": ma,
            "min_Anchor": m_anc,
            "min_SimQ": m_simq,
            "min_EffN": m_effn,
            "market_filter": (None if not market_filter else list(market_filter)),
            "odd_min": (None if omin is None else float(omin)),
            "odd_max": (None if omax is None else float(omax)),
        })

    res_df = pd.DataFrame(results)
    chosen_df = pd.DataFrame(chosen)
    if chosen_df.empty:
        freq_df = pd.DataFrame()
    else:
        freq_df = chosen_df.value_counts().reset_index(
    name="Count").sort_values(
        "Count", ascending=False)

    # summary
    if res_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame([{
            "Blocks": len(res_df),
            "Total_Test_Picks": int(res_df["Test_Picks"].sum()),
            "Total_Test_Profit": float(res_df["Test_Profit"].sum()),
            "Avg_Test_ROI": float(res_df["Test_ROI"].mean()),
            "Median_Test_ROI": float(res_df["Test_ROI"].median()),
            "Neg_Test_ROI_Block_Ratio": float((res_df["Test_ROI"] < 0).mean()),
            "Blocks_With_No_Test_Picks_Ratio": float((res_df["Test_Picks"] <= 0).mean()),
            "Train_LowSample_Block_Ratio": float(res_df["Train_LowSample"].astype(bool).mean()) if "Train_LowSample" in res_df.columns else 0.0,
            "Test_LowSample_Block_Ratio": float(res_df["Test_LowSample"].astype(bool).mean()) if "Test_LowSample" in res_df.columns else 0.0,
        }])

    return res_df, freq_df, summary_df


def _wf_expand_markets_to_candidates(
    _df: pd.DataFrame,
     _use_open: bool) -> pd.DataFrame:
    if _df is None or _df.empty:
        return _df

    def _col_first(cands):
        for c in cands:
            if c in _df.columns:
                return c
        return None

    # Prob columns (support multiple naming conventions)
    p_home = _col_first(["P_Home_Final", "P_Home", "Prob_Home", "Prob_MS1"])
    p_draw = _col_first(
        ["P_Draw_Final", "P_Draw", "Prob_Draw", "Prob_MSX", "Prob_X"])
    p_away = _col_first(["P_Away_Final", "P_Away", "Prob_Away", "Prob_MS2"])
    p_over = _col_first(["P_Over_Final", "P_Over", "Prob_Over"])
    p_btts = _col_first(["P_BTTS_Final", "P_BTTS", "Prob_BTTS"])

    # Odds columns (open vs closing)
    if _use_open:
        o_home = _col_first(["Odds_Open_Home", "OpenHomeOdd", "Open_Home_Odd"])
        o_draw = _col_first(["Odds_Open_Draw", "OpenDrawOdd", "Open_Draw_Odd"])
        o_away = _col_first(["Odds_Open_Away", "OpenAwayOdd", "Open_Away_Odd"])
        o_o25 = _col_first(["Odds_Open_Over25",
    "Odds_Open_Over2.5",
    "OpenO25",
    "OpenOver25",
     "Open_Over25"])
        o_u25 = _col_first(["Odds_Open_Under25",
    "Odds_Open_Under2.5",
    "OpenU25",
    "OpenUnder25",
     "Open_Under25"])
        o_by = _col_first(["Odds_Open_BTTS_Yes", "OpenBTTSY", "Open_BTTSY"])
        o_bn = _col_first(["Odds_Open_BTTS_No", "OpenBTTSN", "Open_BTTSN"])
    else:
        o_home = _col_first(["ClosingHomeOdd", "HomeOdd", "Odd_Home"])
        o_draw = _col_first(["ClosingDrawOdd", "DrawOdd", "Odd_Draw"])
        o_away = _col_first(["ClosingAwayOdd", "AwayOdd", "Odd_Away"])
        o_o25 = _col_first(["ClosingO25", "O25", "Odd_Over25", "Odd_O25"])
        o_u25 = _col_first(["ClosingU25", "U25", "Odd_Under25", "Odd_U25"])
        o_by = _col_first(["ClosingBTTSY", "BTTSY", "Odd_BTTSY"])
        o_bn = _col_first(["ClosingBTTSN", "BTTSN", "Odd_BTTSN"])

    # If we don't have multi-market probabilities, do nothing
    has_multi = any([p_home, p_draw, p_away, p_over, p_btts])
    if not has_multi:
        return _df

    base_cols = list(_df.columns)

    rows = []
    # pre-coerce

    def _num(s):
        return pd.to_numeric(s, errors="coerce")

    for mkt, pcol, ocol, sel, inv_prob in [
        ("MS 1", p_home, o_home, "MS 1", False),
        ("MS X", p_draw, o_draw, "MS X", False),
        ("MS 2", p_away, o_away, "MS 2", False),
        ("2.5 Üst", p_over, o_o25, "2.5 Üst", False),
        ("2.5 Alt", p_over, o_u25, "2.5 Alt", True),
        ("KG Var", p_btts, o_by, "KG Var", False),
        ("KG Yok", p_btts, o_bn, "KG Yok", True),
    ]:
        if pcol is None or ocol is None:
            continue
        tmp = _df.copy()
        prob = _num(tmp[pcol])
        if inv_prob:
            prob = 1.0 - prob
        odd = _num(tmp[ocol])
        tmp["Market"] = mkt
        tmp["Seçim"] = sel
        tmp["Prob"] = prob
        tmp["Odd"] = odd
        tmp["EV"] = (prob * odd) - 1.0
        # Some pipelines expect these names
        tmp["Selection"] = tmp.get("Selection", tmp["Seçim"])
        rows.append(tmp)

    if not rows:
        return _df

    out = pd.concat(rows, ignore_index=True)
    # Clean
    out = out.dropna(subset=["Prob", "Odd"]).copy()
    out = out[(out["Odd"] > 1.0) & (out["Prob"] >= 0.0)
               & (out["Prob"] <= 1.0)].copy()
    return out
# ---------------------------------------------------------------------
    # Candidate Pool Mode (WF-friendly):
    # Many pipelines feed WF with an already "best-of" narrowed list (often 1 pick/day).
    # Here we explicitly create a wider candidate pool per day (and optionally per match),
    # then let the policy thresholds do the filtering.
    # ---------------------------------------------------------------------
    if candidate_pool_mode and (not df.empty):
        _cand = df.copy()
        # Expand into multi-market candidates if possible (prevents 1-pick/day
        # bottleneck)
        _cand = _wf_expand_markets_to_candidates(_cand, use_open_odds)
        # Build a robust score using whatever columns exist
        score_parts = []
        if "Final_Confidence" in _cand.columns:
            score_parts.append(
    pd.to_numeric(
        _cand["Final_Confidence"],
         errors="coerce").fillna(0.0) * 1.0)
        if "CONF_percentile" in _cand.columns:
            score_parts.append(
    pd.to_numeric(
        _cand["CONF_percentile"],
         errors="coerce").fillna(0.0) * 0.8)
        if "AutoMod_percentile" in _cand.columns:
            score_parts.append(
    pd.to_numeric(
        _cand["AutoMod_percentile"],
         errors="coerce").fillna(0.0) * 0.4)
        if "EV" in _cand.columns:
            score_parts.append(
    pd.to_numeric(
        _cand["EV"],
        errors="coerce").fillna(0.0) *
         0.25)
        if "Prob" in _cand.columns:
            score_parts.append(
    pd.to_numeric(
        _cand["Prob"],
        errors="coerce").fillna(0.0) *
         0.25)

        if score_parts:
            _cand["__cand_score__"] = score_parts[0]
            for _p in score_parts[1:]:
                _cand["__cand_score__"] = _cand["__cand_score__"] + _p
        else:
            _cand["__cand_score__"] = 0.0

        # Normalize Date to day for grouping
        _cand["__day__"] = _cand["Date"].dt.floor("D")

        # Optional: limit candidates per match per day (prevents same match
        # dominating)
        if candidate_topk_per_match_per_day is not None and int(
            candidate_topk_per_match_per_day) > 0:
            if "Match_ID" in _cand.columns:
                _cand["__mid__"] = _cand["Match_ID"].astype(str)
            else:
                # fallback
                _cand["__mid__"] = (
                    _cand["__day__"].astype(str) + "|" +
                    _cand.get("League", "").astype(str) + "|" +
                    _cand.get("HomeTeam", "").astype(str) + "|" +
                    _cand.get("AwayTeam", "").astype(str)
                )
            # If markets were expanded, treat each market as a separate
            # candidate bucket per match/day
            if "Market" in _cand.columns:
                _cand["__mid__"] = _cand["__mid__"].astype(
                    str) + "|" + _cand["Market"].astype(str)
            _cand = _cand.sort_values(
                ["__day__", "__mid__", "__cand_score__"], ascending=[True, True, False])
            _cand = _cand.groupby(["__day__", "__mid__"], as_index=False).head(
                int(candidate_topk_per_match_per_day))

        # Finally: keep Top-K candidates per day
        _cand = _cand.sort_values(
            ["__day__", "__cand_score__"], ascending=[True, False])
        _cand = _cand.groupby(
    "__day__", as_index=False).head(
        int(candidate_topk_per_day))
        df = _cand.drop(
    columns=[
        "__cand_score__",
        "__day__",
        "__mid__"],
         errors="ignore").copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # detect FT cols and compute profit
    hcol, acol = _detect_ft_cols(df)
    if hcol is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    df[hcol] = pd.to_numeric(df[hcol], errors="coerce")
    df[acol] = pd.to_numeric(df[acol], errors="coerce")
    df["Odd"] = pd.to_numeric(df.get("Odd"), errors="coerce")
    if bool(signal_only):
        df["Hit"], df["Profit_1u"] = zip(
            *df.apply(lambda r: _calc_hit_profit_proxy(r.get("Seçim"), r.get(hcol), r.get(acol)), axis=1))
    else:
        df["Hit"], df["Profit_1u"] = zip(*df.apply(lambda r: _calc_hit_profit(
            r.get("Seçim"), r.get("Odd"), r.get(hcol), r.get(acol)), axis=1))
    df["Hit"] = pd.to_numeric(df["Hit"], errors="coerce")
    df["Profit_1u"] = pd.to_numeric(df["Profit_1u"], errors="coerce")
    df_valid = df.dropna(subset=["Profit_1u"]).copy()
    if df_valid.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # dates
    min_d = df_valid["Date"].min().normalize()
    max_d = df_valid["Date"].max().normalize()
    # align
    start = max(start_date.normalize(), min_d + pd.Timedelta(days=train_days))
    end = min(end_date.normalize(), max_d)
    block_starts = []
    cur = start
    while cur + pd.Timedelta(days=test_days - 1) <= end:
        block_starts.append(cur)
        cur = cur + pd.Timedelta(days=test_days)

    results = []
    chosen = []

    for b0 in block_starts:
        train_start = b0 - pd.Timedelta(days=train_days)
        train_end = b0 - pd.Timedelta(days=1)
        test_start = b0
        test_end = b0 + pd.Timedelta(days=test_days - 1)

        train_df = df_valid[(df_valid["Date"] >= train_start) & (
            df_valid["Date"] <= train_end)].copy()
        test_df = df_valid[(df_valid["Date"] >= test_start)
                            & (df_valid["Date"] <= test_end)].copy()
        if train_df.empty or test_df.empty:
            continue

        train_days_count = max(1, train_df["Date"].dt.normalize().nunique())

        best = None
        best_score = -1e18
        best_stats = None

        # grid search on train
        for mb in bestof_grid:
            for mc in conf_grid:
                for me in ev_grid:
                    for mp in prob_grid:
                        for ma in automod_grid:
                            # --- YENİ DÖNGÜ KATMANLARI ---
                            for m_anc in anchor_grid:
                                for m_simq in simq_grid:
                                    for m_effn in eff_n_grid:
                                        # -----------------------------
                                        for (
    omin, omax) in (
        odd_grid or [
            (None, None)]):
                                            if bool(signal_only):
                                                omin, omax = (None, None)

                                            sel_train = _apply_policy(
                                                train_df,
                                                mb, mc, me, mp, ma,
                                                min_anchor=m_anc,
                                                min_sim_quality=m_simq,
                                                min_eff_n=m_effn,
                                                odd_min=omin,
                                                odd_max=omax,
                                                use_open_odds=use_open_odds,
                                            )
                                            if sel_train.empty:
                                                continue
                                            n = len(sel_train)
                                            low_sample = bool(
                                                n < int(min_total_picks_train))
                                            # In exploration/Hard Sanity mode we do NOT hard-reject low-sample policies.
                                            # Exploration-friendly: only
                                            # hard-reject low-sample when Hard
                                            # Sanity is ON
                                            if low_sample and bool(
                                                hard_sanity_mode):
                                                continue

                                            # train yoğunluk (bilgi amaçlı)
                                            avg_ppd = n / train_days_count

                                            profit = float(
                                                sel_train["Profit_1u"].sum())
                                            roi = profit / max(1, n)

                                            # penalize volatility lightly: std
                                            # of daily pnl
                                            daily = sel_train.groupby(
    sel_train["Date"].dt.normalize())["Profit_1u"].sum()
                                            vol = float(daily.std()) if len(
                                                daily) > 1 else 0.0

                                            # sample-size penalty (keeps tiny
                                            # policies from dominating when
                                            # hard_sanity_mode is ON)
                                            if low_sample and int(
                                                min_total_picks_train) > 0:
                                                sample_pen = 0.10 * \
                                                    (int(min_total_picks_train) - n) / \
                                                     float(
                                                         int(min_total_picks_train))
                                            else:
                                                sample_pen = 0.0

                                            score = roi - 0.02 * vol - sample_pen

                                            if score > best_score:
                                                best_score = score
                                                best = (
    mb, mc, me, mp, ma, m_anc, m_simq, m_effn, omin, omax)
                                                best_stats = (
                                                    roi, vol, n, avg_ppd)

        if best is None:
            continue

        mb, mc, me, mp, ma, m_anc, m_simq, m_effn, omin, omax = best
        sel_test = _apply_policy(
    test_df,
    mb,
    mc,
    me,
    mp,
    ma,
    omin,
    omax,
     use_open_odds=use_open_odds)
        test_days_count = max(1, test_df["Date"].dt.normalize().nunique())
        if sel_test.empty:
            test_roi = np.nan
            test_n = 0
            test_profit = 0.0
            test_ppd = 0.0
            test_unstable = True
        else:
            test_n = len(sel_test)
            test_profit = float(sel_test["Profit_1u"].sum())
            test_roi = test_profit / test_n if test_n else np.nan
            test_ppd = float(test_n) / float(test_days_count)
            test_unstable = bool(test_ppd < float(min_picks_per_day))

        # Flags (exploration-friendly): never block, just label
        train_n = int(best_stats[2]) if best_stats is not None else 0
        train_low_sample = bool(
    (int(min_total_picks_train) > 0) and (
        train_n < int(min_total_picks_train)))
        test_low_ppd = bool(
    (float(min_picks_per_day) > 0) and (
        float(test_ppd) < float(min_picks_per_day)))

        _flags = []
        if train_low_sample:
            _flags.append("LOW_SAMPLE_TRAIN")
        if test_low_ppd:
            _flags.append("LOW_PPD_TEST")
        flag_str = ",".join(_flags) if _flags else "OK"

        results.append({
            "TestWindowStart": test_start.date(),
            "TestWindowEnd": test_end.date(),
            "TrainStart": train_start.date(),
            "TrainEnd": train_end.date(),
            "Best_min_BestOfRank": mb,
            "Best_min_CONF_pct": mc,
            "Best_min_EV": me,
            "Best_min_Prob": mp,
            "Best_min_AutoMod_pct": ma,
            "Best_Odd_Min": (None if omin is None else float(omin)),
            "Best_Odd_Max": (None if omax is None else float(omax)),
            "Train_ROI": best_stats[0],
            "Train_Vol": best_stats[1],
            "Train_Picks": best_stats[2],
            "Train_PicksPerDay": best_stats[3],
            "Test_ROI": test_roi,
            "Test_Profit": test_profit,
            "Test_Picks": test_n,
            "Test_PicksPerDay": test_ppd,
            "Test_Unstable": test_unstable,
            "Train_LowSample": train_low_sample,
            "Test_LowPPD": test_low_ppd,
            "Flags": flag_str
        })
        chosen.append({
            "min_BestOfRank": mb,
            "min_CONF_pct": mc,
            "min_EV": me,
            "min_Prob": mp,
            "min_AutoMod_pct": ma,
            "odd_min": (None if omin is None else float(omin)),
            "odd_max": (None if omax is None else float(omax)),
        })

    res_df = pd.DataFrame(results)
    chosen_df = pd.DataFrame(chosen)
    if chosen_df.empty:
        freq_df = pd.DataFrame()
    else:
        freq_df = chosen_df.value_counts().reset_index(
    name="Count").sort_values(
        "Count", ascending=False)

    # summary
    if res_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame([{
            "Blocks": len(res_df),
            "Total_Test_Picks": int(res_df["Test_Picks"].sum()),
            "Total_Test_Profit": float(res_df["Test_Profit"].sum()),
            "Avg_Test_ROI": float(res_df["Test_ROI"].mean()),
            "Median_Test_ROI": float(res_df["Test_ROI"].median()),
            "Neg_Test_ROI_Block_Ratio": float((res_df["Test_ROI"] < 0).mean()),
            "Blocks_With_No_Test_Picks_Ratio": float((res_df["Test_Picks"] <= 0).mean()),
            "Train_LowSample_Block_Ratio": float(res_df["Train_LowSample"].astype(bool).mean()) if "Train_LowSample" in res_df.columns else 0.0,
            "Test_LowSample_Block_Ratio": float(res_df["Test_LowSample"].astype(bool).mean()) if "Test_LowSample" in res_df.columns else 0.0,
        }])

    return res_df, freq_df, summary_df


def load_daily_coupons() -> pd.DataFrame:
    """Persistent store of saved daily coupon rows + user-entered scores."""
#     try:  # AUTO-COMMENTED (illegal global try)
#         if JOURNAL_FILE.exists():
#             df = pd.read_parquet(JOURNAL_FILE)
#             if "Date" in df.columns:
#                 df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#             return df
#     except Exception:
#         pass
    return pd.DataFrame()


def save_daily_coupons(df: pd.DataFrame) -> None:
    df.to_parquet(JOURNAL_FILE, index=False)


def eval_pick_hit(pick: str, ft_h, ft_a):
    """Return (hit: bool|None, score_str: str). None if score missing."""
    if ft_h is None or ft_a is None or (pd.isna(ft_h) or pd.isna(ft_a)):
        return None, ""
#     try:  # AUTO-COMMENTED (illegal global try)
#         ft_h = int(ft_h)
#         ft_a = int(ft_a)
#     except Exception:
#         return None, ""
    if pick == "MS 1":
        return (ft_h > ft_a), f"{ft_h}-{ft_a}"
    if pick == "MS X":
        return (ft_h == ft_a), f"{ft_h}-{ft_a}"
    if pick == "MS 2":
        return (ft_h < ft_a), f"{ft_h}-{ft_a}"
    if pick == "2.5 Üst":
        return ((ft_h + ft_a) >= 3), f"{ft_h}-{ft_a}"
    if pick == "2.5 Alt":
        return ((ft_h + ft_a) <= 2), f"{ft_h}-{ft_a}"
    if pick == "KG Var":
        return ((ft_h > 0) and (ft_a > 0)), f"{ft_h}-{ft_a}"
    if pick == "KG Yok":
        return ((ft_h == 0) or (ft_a == 0)), f"{ft_h}-{ft_a}"
    return None, f"{ft_h}-{ft_a}"


def compute_profit(hit, odd):
    """1 unit stake per pick. Returns + (odd-1) if hit, else -1."""
    if hit is None or pd.isna(odd):
        return 0.0
#     try:  # AUTO-COMMENTED (illegal global try)
#         odd = float(odd)
#     except Exception:
#         return 0.0
    return (odd - 1.0) if hit else -1.0


def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest() if b else "NONE"


def build_stable_cache_key(
    past_md5: str,
    future_md5: str,
    mode: str,
     critical_flags: dict) -> str:
    """Stable cache key that does NOT change with code edits.
    It depends only on input data signatures and a small set of model-critical flags.
    """
    payload = {
        "mode": mode,
        "past": past_md5[:16],
        "future": future_md5[:16],
        "flags": critical_flags or {},
        "version": MAJOR_CACHE_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def load_disk_cache(key: str):
    p = CACHE_DIR / f"{key}.pkl"
    if not p.exists():
        return None
#     try:  # AUTO-COMMENTED (illegal global try)
#         with p.open("rb") as f:
#             return pickle.load(f)
#     except Exception:
#         return None


def save_disk_cache(key: str, payload: dict):
    p = CACHE_DIR / f"{key}.pkl"
#     try:  # AUTO-COMMENTED (illegal global try)
#         with p.open("wb") as f:
#             pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
#     except Exception:
#         pass


# -----------------------------
# AUTO MODE helper: combine FULL (1X2) with FAST (OU/KG)
# -----------------------------
AUTO_USE_FULL_FOR = {"MS 1", "MS X", "MS 2"}  # 1X2 -> FULL
AUTO_USE_FAST_FOR = {"2.5 Üst", "2.5 Alt", "KG Var", "KG Yok"}  # OU/KG -> FAST


def combine_fast_full_predictions(
    df_full: pd.DataFrame,
     df_fast: pd.DataFrame) -> pd.DataFrame:
    """Return a composite df that keeps 1X2 probs from FULL and overwrites OU/KG probs from FAST."""
    if df_full is None or getattr(df_full, "empty", True):
        return df_fast
    if df_fast is None or getattr(df_fast, "empty", True):
        return df_full

    key_cols = [c for c in ["Date", "League", "HomeTeam", "AwayTeam"]
        if c in df_full.columns and c in df_fast.columns]
    if len(key_cols) < 2:
        return df_full

    cols_to_take = [c for c in ["P_Over_Final", "P_BTTS_Final"]
        if c in df_fast.columns and c in df_full.columns]
    if not cols_to_take:
        return df_full

    fast_slice = df_fast[key_cols + cols_to_take].copy()
    merged = df_full.merge(
    fast_slice,
    on=key_cols,
    how="left",
    suffixes=(
        "",
         "_FAST"))

    for c in cols_to_take:
        cf = f"{c}_FAST"
        merged[c] = np.where(merged[cf].notna(), merged[cf], merged[c])
        merged.drop(columns=[cf], inplace=True, errors="ignore")

    return merged


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    # Cached CSV parsing (fast on reruns)
    return pd.read_csv(io.BytesIO(file_bytes))

# -----------------------------------------------------------------------------#
# SAFE ASOF MERGE HELPERS (prevents null-key and sorting errors)


def _safe_merge_asof_by(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    by: str,
     direction: str = "backward") -> pd.DataFrame:
    """Robust merge_asof wrapper for the common 'by + on' case.

    Why this exists:
    - pandas.merge_asof requires the *left* and *right* frames to be sorted by the 'on' key.
      In some pandas versions, sorting by [by, on] can still fail the internal monotonicity check
      (because 'on' is not globally monotonic when grouped by 'by').
    - This implementation performs merge_asof *per group* of `by` to satisfy sorting requirements
      deterministically and avoid:
        * ValueError: left keys must be sorted
        * ValueError: Merge keys contain null values on left side

    Behavior:
    - Keeps all left rows (rows with missing keys simply get NaNs for joined columns).
    - Preserves original row order.
    """
    if left is None or getattr(left, "empty", True):
        return left
    if right is None or getattr(right, "empty", True):
        return left

    l = left.copy()
    r = right.copy()

    # Normalize and clean keys
    l[on] = pd.to_datetime(l[on], errors="coerce")
    r[on] = pd.to_datetime(r[on], errors="coerce")

    # Ensure string-like team keys without turning NaN into literal 'nan'
    l[by] = l[by].astype("string").str.strip()
    r[by] = r[by].astype("string").str.strip()

    # Row id to restore original order after groupwise merges
    rid_col = "__row_id__"
    if rid_col in l.columns:
        # Avoid collisions
        rid_col = "__row_id_left__"
    l[rid_col] = np.arange(len(l), dtype=np.int64)

    # Identify join columns from right (exclude keys)
    right_cols = [c for c in r.columns if c not in {on, by}]
    # If there is nothing to join, return original left
    if not right_cols:
        return left

    # Pre-filter right to valid keys only (merge_asof forbids null keys)
    r_valid = r.dropna(subset=[on, by]).copy()
    if r_valid.empty:
        # nothing to merge
        out = l.sort_values(rid_col).drop(columns=[rid_col], errors="ignore")
        return out

    # Sort right within each group by `on` once
    r_valid = r_valid.sort_values([by, on], kind="mergesort")

    # Split left into valid/invalid key subsets
    l_valid = l.dropna(subset=[on, by]).copy()
    l_invalid = l[l[on].isna() | l[by].isna()].copy()

    merged_parts = []

    if not l_valid.empty:
        # Groupwise merge_asof to satisfy pandas' sorted-key requirement
        for key, g in l_valid.groupby(by, sort=False):
            g = g.sort_values(on, kind="mergesort")
            rg = r_valid[r_valid[by] == key]
            if rg.empty:
                # No history for this key; keep g with NaNs for right cols
                for c in right_cols:
                    if c not in g.columns:
                        g[c] = np.nan
                merged_parts.append(g)
                continue

            rg = rg.sort_values(on, kind="mergesort")
            mg = pd.merge_asof(g, rg[[on] + right_cols],
                               on=on, direction=direction)
            merged_parts.append(mg)

    # Combine all parts and restore full left (including invalid-key rows)
    if merged_parts:
        out_valid = pd.concat(merged_parts, ignore_index=False)
    else:
        out_valid = l_valid

    out = pd.concat([out_valid, l_invalid], ignore_index=False, sort=False)
    out = out.sort_values(
    rid_col,
    kind="mergesort").drop(
        columns=[rid_col],
         errors="ignore")

    return out


# -----------------------------------------------------------------------------#
# 1. SETTINGS AND CONSTANTS (V43.00 - DRAW SPECIALIST & OU/KG SPLIT)
# -----------------------------------------------------------------------------#
CURRENT_VERSION = "V43.00 (Platinum - Final Architect)"

st.set_page_config(
    page_title=f"Futbol Analisti (Platinum {CURRENT_VERSION})",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Visible build/version banner ---
st.sidebar.markdown(f"**Build:** `{CURRENT_VERSION}`")


# Track version without auto-clearing caches
if 'version' not in st.session_state:
    st.session_state.version = CURRENT_VERSION
elif st.session_state.version != CURRENT_VERSION:
    st.session_state.version = CURRENT_VERSION

if st.sidebar.button("🧹 Clear cache"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.session_state["_cache_hit_flags"] = {"cache_cleared": True}

# ✅ PATCH #2: Global Probability Thresholds (Historical ROI based)
# 🩹 PATCH 1 Update: Symmetrize OU thresholds
# 🩹 PATCH 5 Update: Relaxed thresholds to populate list better
PROB_THRESHOLDS_HARD = {
    "MS 1": 0.53,      # Relaxed from 0.55
    "MS X": 0.32,      # Relaxed from 0.34
    "MS 2": 0.43,      # Relaxed from 0.45
    "2.5 Üst": 0.56,   # Relaxed from 0.57
    "2.5 Alt": 0.56,   # Relaxed from 0.57
    "KG Var": 0.54,    # Relaxed from 0.55
    "KG Yok": 0.51,    # Relaxed from 0.52
}

# ✅ PATCH #4: Updated Market Limits for Best Of (Global Config)
# 🩹 PATCH 6 Update: Increased quotas to prevent empty lists due to distribution
BESTOF_QUOTA_MAP = {
    "MS 1": 8,         # Increased from 4
    "MS X": 4,         # Increased from 2
    "MS 2": 4,         # Increased from 2
    "2.5 Üst": 5,      # Increased from 2
    "2.5 Alt": 5,      # Increased from 2
    "KG Var": 5,       # Increased from 2
    "KG Yok": 5,       # Increased from 2
}

MARKET_LIMITS_HARD = {
    "MS 1": 8,
    "MS X": 4,
    "MS 2": 4,
    "2.5 Üst": 5,
    "2.5 Alt": 5,
    "KG Var": 5,
    "KG Yok": 5,
}

# 📌 3.2. Global Config for Market Specific Min Prob/EV
BESTOF_MARKET_MIN_CONF_EV = {
    "MS 1": (0.58, 3.0),
    "MS X": (0.35, 2.0),
    "MS 2": (0.45, 2.5),
    "2.5 Üst": (0.54, 1.0),
    "2.5 Alt": (0.54, 1.0),
    "KG Var": (0.52, 1.0),
    "KG Yok": (0.52, 1.0),
}

# 📦 Paket 2 Config: GoldenScore Market Weights & Noise Penalty
GOLDEN_MARKET_WEIGHTS = {
    "MS 1": 1.20,
    "MS X": 1.00,
    "MS 2": 1.15,
    "2.5 Üst": 0.90,
    "2.5 Alt": 0.90,
    "KG Var": 0.90,
    "KG Yok": 0.90,
}

GOLDEN_NOISE_PENALTY = {
    "MS 1": 0.0,
    "MS X": 0.05,
    "MS 2": 0.0,
    "2.5 Üst": 0.10,
    "2.5 Alt": 0.10,
    "KG Var": 0.12,
    "KG Yok": 0.12,
}

# 📌 3.3. Market Score Weights (Patch 5 Dependency)
MARKET_SCORE_WEIGHTS = {
    "MS 1": 1.05,
    "MS X": 0.95,
    "MS 2": 1.00,
    "2.5 Üst": 0.98,
    "2.5 Alt": 0.98,
    "KG Var": 0.98,
    "KG Yok": 0.98,
}

# 📌 3.4. Pool Max Limits (Patch 2 Dependency)
POOL_MAX_PER_MARKET = {
    "MS 1": 50,
    "MS X": 30,
    "MS 2": 30,
    "2.5 Üst": 30,
    "2.5 Alt": 30,
    "KG Var": 30,
    "KG Yok": 30,
}

# ------------------------------------------------------------------
# (A) Best Of: Pazara Özel Sıralama Ağırlıkları (GURME MODU)
# ------------------------------------------------------------------
# Mantık: Her pazarın dinamiği farklıdır.
# 1X2 -> Lig güveni ve Model skoru önemlidir.
# OU  -> Doğru fiyatlama (Golden/EV) ve Pazar karnesi önemlidir.
# KG  -> Pazar karnesi ve Fiyatlama önemlidir.


# -----------------------------------------------------------
# GÜNCELLENMİŞ AĞIRLIKLAR (Probability First, EV Second)
# -----------------------------------------------------------
# -----------------------------------------------------------
# GÜNCELLENMİŞ AĞIRLIKLAR (Probability First, EV Second)
# -----------------------------------------------------------
RANK_WEIGHTS_BY_MARKET = {
    # Taraf Bahsi (MS): Güven %60, EV %15
    "MS": {
        "score": 0.40,      # Model Puanı
        "conf": 0.20,       # Güven Endeksi
        "golden": 0.15,     # EV (Eskiden 0.30 idi)
        "amqs": 0.15,       # Lig Başarısı
        "market": 0.10      # Piyasa Onayı
    },

    # Gol Bahsi (OU): Güven %65, EV %10
    # Lyon gibi maçlarda Alt seçmesin diye EV'yi iyice kıstık.
    "OU": {
        "score": 0.45,      # Model Puanı (En önemlisi)
        "conf": 0.20,       # Güven Endeksi
        "golden": 0.10,     # EV (Eskiden 0.40 idi -> %10'a düşürüldü)
        "amqs": 0.15,
        "market": 0.10
    },

    # KG Bahsi (BTTS): Dengeli
    "BTTS": {
        "score": 0.40,
        "conf": 0.20,
        "golden": 0.15,     # EV Düşürüldü
        "amqs": 0.15,
        "market": 0.10
    },
    # Alias: 1X2 (MS) geriye dönük uyumluluk
    "1X2": {
        "score": 0.40,
        "conf": 0.20,
        "golden": 0.15,
        "amqs": 0.15,
        "market": 0.10
    },

    # Alias: KG (BTTS) geriye dönük uyumluluk
    "KG": {
        "score": 0.40,
        "conf": 0.20,
        "golden": 0.15,
        "amqs": 0.15,
        "market": 0.10
    }
}

# -----------------------------------------------------------------------------#
# BestOfRank helpers (single source of truth)
# -----------------------------------------------------------------------------#


def compute_bestofrank_for_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Compute BestOfRank for any dataframe containing at least:
    Seçim, Score, GoldenScore, Prob, League_Conf, Market_Conf_Score (missing confs default to 0.5).
    Works for both Pool and BestOf lists.
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    # Defaults for missing confidence columns
    if "League_Conf" not in df.columns:
        df["League_Conf"] = 0.5
    if "Market_Conf_Score" not in df.columns:
        df["Market_Conf_Score"] = 0.5

    # Ensure numeric
    for c in [
    "Score",
    "GoldenScore",
    "Prob",
    "League_Conf",
     "Market_Conf_Score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    frames = []
    for market_name, sub in df.groupby("Seçim"):
        if sub.empty:
            continue

        # Market group weights
        if market_name in ["MS 1", "MS X", "MS 2"]:
            w = RANK_WEIGHTS_BY_MARKET.get("1X2", {})
        elif market_name in ["2.5 Üst", "2.5 Alt"]:
            w = RANK_WEIGHTS_BY_MARKET.get("OU", {})
        else:
            w = RANK_WEIGHTS_BY_MARKET.get("KG", {})

        # Use safe z-score helper if available, else fallback
        if "safe_z_score" in globals():
            sub["_z_score"] = safe_z_score(
    sub.get(
        "Score",
        pd.Series(
            index=sub.index,
             data=0.0)))
            sub["_z_golden"] = safe_z_score(
    sub.get(
        "GoldenScore",
        pd.Series(
            index=sub.index,
             data=0.0)))
            sub["_z_prob"] = safe_z_score(
    sub.get(
        "Prob",
        pd.Series(
            index=sub.index,
             data=0.0)))
        else:
            def _z(s):
                # Accept either a pandas Series or a scalar; always return a
                # Series aligned to sub.index
                if not isinstance(s, pd.Series):
                    s = pd.Series([s] * len(sub), index=sub.index)
                s = pd.to_numeric(s, errors="coerce").fillna(0.0)
                return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

            sub["_z_score"] = _z(
    sub.get(
        "Score",
        pd.Series(
            index=sub.index,
             data=0.0)))
            sub["_z_golden"] = _z(
    sub.get(
        "GoldenScore",
        pd.Series(
            index=sub.index,
             data=0.0)))
            sub["_z_prob"] = _z(
    sub.get(
        "Prob",
        pd.Series(
            index=sub.index,
             data=0.0)))

        league_term = (sub["League_Conf"].fillna(0.5) - 0.5) * 2.0
        market_term = (sub["Market_Conf_Score"].fillna(0.5) - 0.5) * 2.0

        sub["BestOfRank"] = (
            float(w.get("score", 0.0)) * sub["_z_score"] +
            float(w.get("golden", 0.0)) * sub["_z_golden"] +
            float(w.get("prob", 0.0)) * sub["_z_prob"] +
            float(w.get("league", 0.0)) * league_term +
            float(w.get("market", 0.0)) * market_term
        )

        frames.append(sub)

    if not frames:
        df["BestOfRank"] = 0.0
        return df

    out = pd.concat(frames, ignore_index=True)
    out.drop(
    columns=[
        "_z_score",
        "_z_golden",
        "_z_prob"],
        errors="ignore",
         inplace=True)
    return out


def add_star_rating_by_bestofrank(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1–5 Star_Rating based on BestOfRank percentiles within df."""
    if df is None or df.empty or "BestOfRank" not in df.columns:
        return df
    df = df.copy()
    ranks = df["BestOfRank"].rank(method="dense", ascending=False)
    pct = ranks / max(ranks.max(), 1)

    df["Star_Rating"] = 1
    df.loc[pct <= 0.10, "Star_Rating"] = 5
    df.loc[(pct > 0.10) & (pct <= 0.25), "Star_Rating"] = 4
    df.loc[(pct > 0.25) & (pct <= 0.50), "Star_Rating"] = 3
    df.loc[(pct > 0.50) & (pct <= 0.75), "Star_Rating"] = 2
    df.loc[pct > 0.75, "Star_Rating"] = 1
    return df


# 2. FEATURE SET DEFINITIONS
FEATURE_SETS = {
    "common": [
        'HomeLeaguePosition', 'AwayLeaguePosition',
        'League_Code', 'League_Size', 'SeasonProgress', 'SupremacyShift',
        'Home_DomesticLeague_Strength', 'Away_DomesticLeague_Strength',
        'League_Strength_Index',
        'Is_Euro',       # 🔴 NEW: European cup flag
        'Euro_Tier'      # 🔴 NEW: CL/EL level difference
    ],
    "draw": [
        "Elo_Home", "Elo_Away", "Elo_Gap",
        "Form_Home", "Form_Away", "Form_Gap",
        "xG_Home", "xG_Away", "xG_Gap",
        "Def_Var_Home", "Def_Var_Away",
        "Total_xG_Exp", "Tempo_Gap", "Gap_Composite",
        # 📌 PACKAGE 1: DRAW SPECIALIST FEATURES
        "Tiny_Elo_Gap_Flag", "PreMatch_Total_xG", "xG_Variance_10", "League_Draw_Rate_5Y", "Draw_Odds_Compression"
    ],
    "ou_base": [
        'Roll_xG_5_Home', 'Roll_xG_5_Away', 'Roll_xGA_5_Home', 'Roll_xGA_5_Away',
        'Roll_GF_5_Home', 'Roll_GF_5_Away', 'Roll_GA_5_Home', 'Roll_GA_5_Away',
        'Roll_xGA_Std_Home', 'Roll_xGA_Std_Away', 'Tempo_Index_z', 'Collapse_Index',
        'TBxG', 'OU_Chaos_Index', 'OU_Att_Sum', 'OU_Def_Sum', 'OU_AttDef_Diff', 'L_Over_Factor',
        'Book_Exp_Total_Goals', 'Imp_Over25_Close', 'Imp_Under25_Close', 'Over25_Prob_Delta', 'Under25_Prob_Delta',
        'League_Tempo_Level', 'League_Tempo_z',
        'GF_Std_10_Home', 'GF_Std_10_Away', 'GA_Std_10_Home', 'GA_Std_10_Away',
        'Goals_Total_Std_10_Home', 'Goals_Total_Std_10_Away',
        'xGD_last5_Home', 'xGD_last5_Away'
    ],
    "totals": [
        'League_Goal_Avg', 'League_O25_Avg', 'Tempo_Index',
        'xG_For_5_Home_Adj', 'xG_Against_5_Home_Adj',
        'xG_For_5_Away_Adj', 'xG_Against_5_Away_Adj',
        'Goals_For_5_Home_Adj', 'Goals_Against_5_Home_Adj',
        'Goals_For_5_Away_Adj', 'Goals_Against_5_Away_Adj',
        'O25_Rate_5_Home_Adj', 'O25_Rate_5_Away_Adj',
        'Poisson_Home_Lambda', 'Poisson_Away_Lambda',
        'P_O25_Poisson',
        'Imp_Over25_Close', 'Imp_Under25_Close', 'Over25_Prob_Delta', 'Under25_Prob_Delta',
        # PACKAGE 2 BOOST
        "Attacking_Imbalance", "Defensive_Imbalance", "Total_Tempo", "Lambda_Home_Adj", "Lambda_Away_Adj"
    ],
    "btts": [
        'League_KG_Avg', 'Tempo_Index',
        'P_KG_Poisson',
        'xG_For_5_Home_Adj', 'xG_Against_5_Home_Adj',
        'xG_For_5_Away_Adj', 'xG_Against_5_Away_Adj',
        'Goals_For_5_Home_Adj', 'Goals_Against_5_Home_Adj',
        'Goals_For_5_Away_Adj', 'Goals_Against_5_Away_Adj',
        'KG_Rate_5_Home_Adj', 'KG_Rate_5_Away_Adj',
        'Imp_BTTSY_Close', 'Imp_BTTSN_Close',
        # PACKAGE 2 BOOST
        "Attacking_Imbalance", "Total_Tempo", "Lambda_Home_Adj", "Lambda_Away_Adj"
    ]
}

FEATURE_SETS["home"] = FEATURE_SETS["common"] + ['Att_Home',
    'Def_Home',
    'Att_Away',
    'Def_Away',
    'P_Home_Pois',
    'Tempo_Index_z',
    'Collapse_Index',
    'TBxG',
    'OU_Chaos_Index',
    'Market_Bias_Home',
    'Mgr_Exp_Home',
    'Mgr_Exp_Away',
    'Mgr_Exp_Diff',
    'Form_Power_Home',
    'Form_Power_Away',
    'Form_TMB_Diff',
    'Form_THBH_Diff',
    'Pois_Home_Ext',
    'Pois_Home_Diff',
    'Supremacy_Calc_z',
    'Odds_Compression_1X2',
    'Imp_Home_Open',
    'Imp_Home_Close',
    'Home_Prob_Delta',
    'Imp_Draw_Close',
    'Imp_Away_Close',
    'Book_xG_Home',
    'LogOdds_Home_Delta',
    'LogOdds_Draw_Delta',
    'LogOdds_Away_Delta',
    'League_Tempo_z',
    'xGD_last5_Home',
     'xGD_last5_Diff']

FEATURE_SETS["away"] = FEATURE_SETS["common"] + ['Att_Away',
    'Def_Away',
    'Elo_Diff',
    'P_Away_Pois',
    'Tempo_Index_z',
    'Collapse_Index',
    'TBxG',
    'Market_Bias_Away',
    'Mgr_Exp_Home',
    'Mgr_Exp_Away',
    'Mgr_Exp_Diff',
    'Form_Power_Home',
    'Form_Power_Away',
    'Form_TMB_Diff',
    'Form_THBH_Diff',
    'Pois_Away_Ext',
    'Pois_Away_Diff',
    'Supremacy_Calc_z',
    'Odds_Compression_1X2',
    'Imp_Away_Open',
    'Imp_Away_Close',
    'Away_Prob_Delta',
    'Imp_Draw_Close',
    'Imp_Home_Close',
    'Book_xG_Away',
    'LogOdds_Home_Delta',
    'LogOdds_Draw_Delta',
    'LogOdds_Away_Delta',
    'League_Tempo_z',
    'xGD_last5_Away',
     'xGD_last5_Diff']

FEATURE_SETS["ou"] = FEATURE_SETS["ou_base"]

# -----------------------------------------------------------------------------#
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------#


def get_available_features(
    df: pd.DataFrame,
     feature_list: List[str]) -> List[str]:
    """Filters the feature list to include only columns present in the dataframe."""
    return [c for c in feature_list if c in df.columns]


def sanitize_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans up odds data, replacing extreme values with NaN."""
    df = df.copy()
    odd_cols = [
        "HomeOdd", "DrawOdd", "AwayOdd",
        "O25", "U25", "BTTSY", "BTTSN",
        "Odds_Open_Home", "Odds_Open_Draw", "Odds_Open_Away",
        "Odds_Open_Over25", "Odds_Open_Under25",
        "Odds_Open_BTTS_Yes", "Odds_Open_BTTS_No",
        "ClosingHomeOdd", "ClosingDrawOdd", "ClosingAwayOdd",
        "ClosingO25", "ClosingU25", "ClosingBTTSY", "ClosingBTTSN"
    ]
    for c in odd_cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # Odds outside [1.01, 50] are considered noise/error
        df.loc[(df[c] < 1.01) | (df[c] > 50), c] = np.nan
    return df


def calculate_ece_clean(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
        # ECE = sum(|E[P] - E[Y]| * P_k)
        ece += np.abs(y_prob[mask].mean() -
                      y_true[mask].mean()) * (mask.sum() / N)
    return ece

# ✅ PATCH #1 & UPDATE B: Logic Fix with Dynamic Slider Support


def should_create_pick(
    market,
    prob,
    ev,
    prob_thresholds,
    ev_threshold,
     slider_min_prob=0.50):
    """
    Determines if a pick should be created based on hard market-specific thresholds.
    Filters candidates BEFORE they enter the pool.

    UPDATE B: Supports 'slider_min_prob' to relax thresholds if user lowers the slider.
    """
    # 1) Market based probability threshold
    base_req = prob_thresholds.get(market, 0.00)

    # Allow relaxing down to base - 0.07 if slider is lower
    # Example: If MS1 Base is 0.55, but user sets slider to 0.40,
    # we relax requirement to max(0.40, 0.55-0.07) = 0.48. Not all the way to
    # 0.40, but safer.
    required_prob = max(min(slider_min_prob, base_req),
                        max(0.05, base_req - 0.20))

    if prob < required_prob:
        return False

    # 2) Global EV threshold (can be overridden)
    if ev < ev_threshold:
        return False

    return True

# 📦 Paket 1.1: Oran Segment Fonksiyonu


def odds_bucket(odd: float) -> str:
    if pd.isna(odd):
        return "unknown"
    if odd < 1.40:
        return "low"
    elif odd < 2.10:
        return "mid"
    elif odd < 3.00:
        return "high"
    else:
        return "very_high"

# 📦 Paket 1.3: Bucket Calibrator Trainer


def train_bucket_calibrators(df, prob_col, target_col, bucket_col):
    models = {}
    for bucket, seg in df.groupby(bucket_col):
        if len(seg) < 50:  # Minimum sample to train isotonic
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(seg[prob_col], seg[target_col])
        models[bucket] = iso
    return models

# 📦 Paket 1.4: Apply Bucket Calibration


def apply_bucket_calibration(row, models, prob, bucket_col):
    bucket = row.get(bucket_col)
    if bucket in models:
        # Isotonic transform expects array-like
        return models[bucket].transform([prob])[0]
    return prob

# ✅ MISSING HELPERS RE-ADDED BELOW


def compute_expected_intensity(df):
    df = df.copy()
    # Simple proxy for match intensity (total expected goals)
    if 'Roll_xG_5_Home' in df.columns and 'Roll_xG_5_Away' in df.columns:
        df['expected_intensity'] = df['Roll_xG_5_Home'] + df['Roll_xG_5_Away']
    else:
        df['expected_intensity'] = 2.5  # Default
    return df


def intensity_segment(val):
    if val < 2.2:
        return "low"
    if val < 3.0: return "mid"
    return "high"


def train_segment_calibrators(df, prob_col, target_col):
    models = {}
    if 'intensity_segment' not in df.columns:
        return models

    for seg, group in df.groupby('intensity_segment'):
        if len(group) > 50:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(group[prob_col], group[target_col])
            models[seg] = iso
    return models


def apply_segment_calibration(row, models, prob):
    seg = row.get('intensity_segment', 'mid')
    if seg in models:
        return models[seg].transform([prob])[0]
    return prob


def train_ou_aux_model(df, weights):
    # Auxiliary model for Over 2.5 using simple features
    # Helps capture specific league/team tendencies not caught by meta stacking
    cols = ["Roll_xG_5_Home", "Roll_xG_5_Away", "Elo_Diff", "League_Goal_Avg"]
    X = df[get_available_features(df, cols)].fillna(0)
    y = df['Target_Over']

    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=100, random_state=42
    )
    model.fit(X, y, sample_weight=weights)
    return model


def train_btts_aux_model(df, weights):
    # Auxiliary model for BTTS
    cols = ["Roll_xG_5_Home", "Roll_xG_5_Away", "Elo_Diff", "League_Goal_Avg"]
    X = df[get_available_features(df, cols)].fillna(0)
    y = df['Target_BTTS']

    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=100, random_state=42
    )
    model.fit(X, y, sample_weight=weights)
    return model

# 📦 Paket 3.1: MS2 Aux Model Trainer


def train_ms2_aux_model(df: pd.DataFrame):
    # Sadece deplasman tarafının ciddi olduğu maçları al
    # close_away_odds -> ClosingAwayOdd, close_home_odds -> ClosingHomeOdd
    if 'ClosingAwayOdd' not in df.columns or 'ClosingHomeOdd' not in df.columns:
        return None

    mask = (df["ClosingAwayOdd"] < 3.20) & (df["ClosingHomeOdd"] > 1.40)
    aux_df = df[mask].copy()

    if len(aux_df) < 500:  # Slightly relaxed for small datasets
        return None

    # Calculate necessary diff features on the fly
    if "League_Pos_Diff" not in aux_df.columns:
        aux_df["League_Pos_Diff"] = aux_df["HomeLeaguePosition"] - \
            aux_df["AwayLeaguePosition"]
    if "xG_Diff" not in aux_df.columns:
        aux_df["xG_Diff"] = aux_df["Roll_xG_5_Home"] - aux_df["Roll_xG_5_Away"]

    # Feature set
    feats = [
    "Elo_Diff",
    "League_Pos_Diff",
    "Form_TMB_Diff",
    "xG_Diff",
     "League_Strength_Index"]
    X = aux_df[get_available_features(aux_df, feats)].fillna(0)

    # Target: 0 is Away Win in our encoding
    y = (aux_df["Target_Result"] == 0).astype(int)

    model = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=50,
        random_state=42
    )
    model.fit(X, y)
    return model

# 📌 2️⃣ CACHED & OPTIMIZED: League Confidence


@st.cache_data(show_spinner=False)
def calculate_league_confidence(hist_df):
    """Calculates a confidence score for each league based on past performance."""
    if hist_df.empty:
        return {}

    # Optimized for speed
    if len(hist_df) > 40000:
        hist_df = hist_df.sample(40000, random_state=42)

    league_scores = {}
    for lig, grp in hist_df.groupby('League'):
        if len(grp) < 5:
            league_scores[lig] = 0.4
            continue
        # Focus on Home Win market performance (Proxy for 1X2 quality)
        y_true = (grp['Target_Result'] == 2).astype(int)
        y_prob = grp['P_Home_Final']
        brier = brier_score_loss(y_true, y_prob)
        roi = grp['Actual_ROI'].mean()
        # Normalize Brier (closer to 0 is better, baseline is 0.25)
        s_brier = np.clip((0.25 - brier) / 0.10, 0, 1)
        # Normalize ROI (positive ROI is better)
        s_roi = np.clip((roi + 0.05) * 5, 0, 1)
        # Weighted average
        final_score = 0.7 * s_brier + 0.3 * s_roi
        league_scores[lig] = float(np.clip(final_score, 0.1, 1.0))
    return league_scores

# 📌 2️⃣ CACHED & OPTIMIZED: Market Threshold Stats (Patch #3)

# =============================================================
# AUTO-MOD QUALITY SCORE (AMQS) – Integrated (v5)
# =============================================================


@st.cache_data(show_spinner=False)
def calculate_amqs_df(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Auto-Mod Quality Score (AMQS) per (League, Market) from history.
    Uses best-effort proxies and never raises.
    Returns columns: ['League','Seçim','AMQS','AMQS_percentile','AutoMod_Status']
    """
    if hist_df is None or hist_df.empty:
        return pd.DataFrame(
    columns=[
        "League",
        "Seçim",
        "AMQS",
        "AMQS_percentile",
         "AutoMod_Status"])

    dfh = hist_df.copy()

    # Ensure required grouping columns exist
    if "League" not in dfh.columns or "Seçim" not in dfh.columns:
        return pd.DataFrame(
    columns=[
        "League",
        "Seçim",
        "AMQS",
        "AMQS_percentile",
         "AutoMod_Status"])

    # ROI proxy column
    if "Actual_ROI" not in dfh.columns:
        # if P/L available, approximate ROI as profit per bet in %
        if "Profit" in dfh.columns:
            dfh["Actual_ROI"] = dfh["Profit"].astype(float)
        else:
            dfh["Actual_ROI"] = 0.0

    # Brier proxy column
    if "Brier" not in dfh.columns:
        dfh["Brier"] = 0.25

    # EV proxy column
    if "EV" not in dfh.columns:
        dfh["EV"] = 0.0

    rows = []
    for (lig, mkt), g in dfh.groupby(["League", "Seçim"]):
        n = int(len(g))
        if n < 25:
            continue

        roi = float(np.nanmean(g["Actual_ROI"].astype(float)))
        roi_std = float(np.nanstd(g["Actual_ROI"].astype(float)))
        brier = float(np.nanmean(g["Brier"].astype(float)))
        ev = g["EV"].astype(float)

        roi_norm = float(np.clip((roi + 5.0) / 25.0, 0.0, 1.0))
        stability = float(1.0 / (1.0 + roi_std))
        calib = float(np.clip((0.25 - brier) / 0.10, 0.0, 1.0))
        edge = float(np.nanmean(ev > 0))
        density = float(np.clip(n / 300.0, 0.0, 1.0))

        amqs = (
            0.35 * roi_norm +
            0.20 * stability +
            0.20 * calib +
            0.15 * edge +
            0.10 * density
        )
        rows.append({"League": lig, "Seçim": mkt,
                    "AMQS": float(np.clip(amqs, 0.0, 1.0))})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
    columns=[
        "League",
        "Seçim",
        "AMQS",
        "AMQS_percentile",
         "AutoMod_Status"])

    out["AMQS_percentile"] = out["AMQS"].rank(pct=True)
    out["AutoMod_Status"] = pd.cut(
        out["AMQS_percentile"],
        bins=[-0.01, 0.60, 0.85, 1.01],
        labels=["Low", "Medium", "High"]
    ).astype(str)

    return out


def attach_amqs_to_picks(
    picks_df: pd.DataFrame,
     amqs_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join AMQS metrics to picks df by ['League','Seçim'].
    Always returns a df; fills missing with neutral defaults.
    """
    if picks_df is None or getattr(picks_df, "empty", True):
        return picks_df
    if amqs_df is None or getattr(amqs_df, "empty", True):
        df = picks_df.copy()
        df["AMQS"] = 0.50
        df["AMQS_percentile"] = 0.50
        df["AutoMod_Status"] = "Medium"
        return df

    df = picks_df.merge(amqs_df, how="left", on=["League", "Seçim"])
    df["AMQS"] = df["AMQS"].fillna(0.50)
    df["AMQS_percentile"] = df["AMQS_percentile"].fillna(0.50)
    df["AutoMod_Status"] = df["AutoMod_Status"].fillna("Medium")
    return df

# =============================================================
# END AMQS (v5)
# =============================================================


@st.cache_data(show_spinner=False)
def compute_market_threshold_stats(hist_df):
    """Finds the optimal probability threshold for maximizing ROI (and score) for key markets."""
    if hist_df.empty:
        return pd.DataFrame()

    # Optimized for speed: Sample if too large
    if len(hist_df) > 40000:
        hist_df = hist_df.sample(40000, random_state=42)

    # 📌 UPDATED: Using *_Final probabilities instead of legacy Prob_*
    markets = {
        "MS 1": {"y": (hist_df['Target_Result'] == 2).astype(int),
                 "p": hist_df['P_Home_Final'],
                 "odd": hist_df['ClosingHomeOdd']},
        "MS X": {"y": (hist_df['Target_Result'] == 1).astype(int),
                 "p": hist_df['P_Draw_Final'],
                 "odd": hist_df['ClosingDrawOdd']},
        "MS 2": {"y": (hist_df['Target_Result'] == 0).astype(int),
                 "p": hist_df['P_Away_Final'],
                 "odd": hist_df['ClosingAwayOdd']},
    }

    # Check for Final OU columns
    if 'P_Over_Final' in hist_df.columns:
        markets["2.5 Üst"] = {
            "y": hist_df['Target_Over'].astype(int),
            "p": hist_df['P_Over_Final'],
            "odd": hist_df['ClosingO25']
        }
        markets["2.5 Alt"] = {
            "y": 1 - hist_df['Target_Over'].astype(int),
            "p": 1 - hist_df['P_Over_Final'],
            "odd": hist_df['ClosingU25']
        }
    elif 'Prob_Over' in hist_df.columns:  # Fallback
        markets["2.5 Üst"] = {
            "y": hist_df['Target_Over'].astype(int),
            "p": hist_df['Prob_Over'],
            "odd": hist_df['ClosingO25']
        }
        markets["2.5 Alt"] = {
            "y": 1 - hist_df['Target_Over'].astype(int),
            "p": 1 - hist_df['Prob_Over'],
            "odd": hist_df['ClosingU25']
        }

    # Check for Final BTTS columns
    if 'P_BTTS_Final' in hist_df.columns:
        markets["KG Var"] = {
            "y": hist_df['Target_BTTS'].astype(int),
            "p": hist_df['P_BTTS_Final'],
            "odd": hist_df['ClosingBTTSY']
        }
        markets["KG Yok"] = {
            "y": 1 - hist_df['Target_BTTS'].astype(int),
            "p": 1 - hist_df['P_BTTS_Final'],
            "odd": hist_df['ClosingBTTSN']
        }
    elif 'Prob_BTTS' in hist_df.columns:  # Fallback
        markets["KG Var"] = {
            "y": hist_df['Target_BTTS'].astype(int),
            "p": hist_df['Prob_BTTS'],
            "odd": hist_df['ClosingBTTSY']
        }
        markets["KG Yok"] = {
            "y": 1 - hist_df['Target_BTTS'].astype(int),
            "p": 1 - hist_df['Prob_BTTS'],
            "odd": hist_df['ClosingBTTSN']
        }

    rows = []
    for name, d in markets.items():
        y = d["y"].copy()
        p = pd.to_numeric(d["p"], errors="coerce")
        odd = pd.to_numeric(d["odd"], errors="coerce")

        valid = (~p.isna()) & (~odd.isna()) & (odd >= 1.01)
        y, p, odd = y[valid], p[valid], odd[valid]

        # ✅ PATCH 2: OU & KG için daha yüksek minimum sample
        is_ou_or_kg = name in ["2.5 Üst", "2.5 Alt", "KG Var", "KG Yok"]
        min_total_samples = 150 if is_ou_or_kg else 40
        min_thr_samples = 120 if is_ou_or_kg else 20

        if len(y) < min_total_samples:
            rows.append({
                "Pazar": name,
                "En_iyi_Prob_Eşiği": np.nan,
                "ROI_%": np.nan,
                "Bahis_Sayısı": int(len(y))
            })
            continue

        # 📌 PATCH 1: Pazar bazlı tarama aralığı
        base_thr = PROB_THRESHOLDS_HARD.get(name, 0.45)

        if name == "MS X":
            scan_min = max(base_thr - 0.05, 0.25)
            scan_max = min(base_thr + 0.10, 0.70)
        elif name in ["2.5 Üst", "2.5 Alt", "KG Var", "KG Yok"]:
            # OU / KG için daha sıkı eşik aralığı
            scan_min = max(base_thr - 0.03, 0.52)
            scan_max = min(base_thr + 0.07, 0.80)
        else:
            scan_min = max(base_thr - 0.05, 0.45)
            scan_max = min(base_thr + 0.10, 0.80)

        best_score, best_roi, best_thr, best_count = -np.inf, -np.inf, np.nan, 0

        # Çok düşük veya çok geniş eşikleri engellemek için:
        max_share = 0.30 if name in [
    "2.5 Üst", "2.5 Alt", "KG Var", "KG Yok"] else 0.40

        for thr in np.linspace(scan_min, scan_max, 15):
            m = p >= thr
            # Çok az veya aşırı fazla maçı ele
            if m.sum() < min_thr_samples:
                continue
            if m.sum() > len(y) * max_share:
                continue

            profit = np.where(y[m] == 1, odd[m] - 1, -1)
            roi = np.nanmean(profit) * 100

            # ✅ PATCH #3: EV-Driven Threshold Optimization
            # precision (win rate) calculation
            precision = y[m].mean()

            if name in ["2.5 Üst", "2.5 Alt"]:
                # EV domine etmeli (roi), precision 2. planda
                # scale precision to % for consistency
                combined_score = 0.8 * roi + 0.2 * (precision * 100)
            elif name in ["KG Var", "KG Yok"]:
                combined_score = 0.9 * roi + 0.1 * (precision * 100)
            else:
                # 1X2 Balanced
                combined_score = 0.6 * roi + 0.4 * (precision * 100)

            if combined_score > best_score:
                best_score = combined_score
                best_roi = roi
                best_thr = thr
                best_count = int(m.sum())

        if not np.isfinite(best_roi):
            best_roi = np.nan

        rows.append({
            "Pazar": name,
            "En_iyi_Prob_Eşiği": best_thr,
            "ROI_%": best_roi,
            "Bahis_Sayısı": best_count
        })

    return pd.DataFrame(rows)

# 📌 2️⃣ CACHED & OPTIMIZED: EV Analysis


@st.cache_data(show_spinner=False)
def analyze_ev_buckets_dynamic(
    hist_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    EV bucket'larından küçük örnek sayıları nedeniyle oluşan aşırı pozitif/negatif
    boost'ları filtreler. Ayrıca performans için max 30k satıra sample eder.
    """
    if hist_df.empty:
        return {}

    # ✅ Performans için sample (full mode'da bile tüm geçmişe bakma)
    df = hist_df.copy()
    if len(df) > 30000:
        df = df.sample(30000, random_state=42)

    results: Dict[str, Dict[str, float]] = {}

    bins = [-np.inf, -10, 0, 10, 20, 30, 40, np.inf]
    labels = ['<-10', '-10to0', '0to10', '10to20', '20to30', '30to40', '40+']
    min_bin_size = 50  # her EV bucket'ı için minimum maç sayısı

    # 1. Global / MS 1 Analysis
    if 'P_Home_Final' in df.columns:
        df['EV_MS1'] = (df['P_Home_Final'] * df['ClosingHomeOdd'] - 1) * 100
        df['EV_Bin_MS1'] = pd.cut(df['EV_MS1'], bins=bins, labels=labels)
        df['Profit_MS1'] = np.where(
    df['Target_Result'] == 2, df['ClosingHomeOdd'] - 1, -1)

        roi_map = df.groupby('EV_Bin_MS1', observed=False)[
                             'Profit_MS1'].agg(['mean', 'count'])
        boost_map: Dict[str, float] = {}
        for bin_label, row in roi_map.iterrows():
            cnt = row['count']
            if cnt < min_bin_size:
                boost_map[bin_label] = 0.0
            else:
                roi = row['mean']
                boost_map[bin_label] = float(np.clip(roi, -0.2, 0.2))

        results["MS 1"] = boost_map
        results["Global"] = boost_map

    # 2. MS X Analysis
    if 'P_Draw_Final' in df.columns:
        df['EV_MSX'] = (df['P_Draw_Final'] * df['ClosingDrawOdd'] - 1) * 100
        df['EV_Bin_MSX'] = pd.cut(df['EV_MSX'], bins=bins, labels=labels)
        df['Profit_MSX'] = np.where(
    df['Target_Result'] == 1, df['ClosingDrawOdd'] - 1, -1)

        roi_map = df.groupby('EV_Bin_MSX', observed=False)[
                             'Profit_MSX'].agg(['mean', 'count'])
        boost_map: Dict[str, float] = {}
        for bin_label, row in roi_map.iterrows():
            cnt = row['count']
            if cnt < min_bin_size:
                boost_map[bin_label] = 0.0
            else:
                roi = row['mean']
                boost_map[bin_label] = float(np.clip(roi, -0.2, 0.2))
        results["MS X"] = boost_map

    # 3. MS 2 Analysis
    if 'P_Away_Final' in df.columns:
        df['EV_MS2'] = (df['P_Away_Final'] * df['ClosingAwayOdd'] - 1) * 100
        df['EV_Bin_MS2'] = pd.cut(df['EV_MS2'], bins=bins, labels=labels)
        df['Profit_MS2'] = np.where(
    df['Target_Result'] == 0, df['ClosingAwayOdd'] - 1, -1)

        roi_map = df.groupby('EV_Bin_MS2', observed=False)[
                             'Profit_MS2'].agg(['mean', 'count'])
        boost_map: Dict[str, float] = {}
        for bin_label, row in roi_map.iterrows():
            cnt = row['count']
            if cnt < min_bin_size:
                boost_map[bin_label] = 0.0
            else:
                roi = row['mean']
                boost_map[bin_label] = float(np.clip(roi, -0.2, 0.2))
        results["MS 2"] = boost_map

    # 4. OU Analysis
    p_over_col = 'P_Over_Final' if 'P_Over_Final' in df.columns else 'Prob_Over'
    if {p_over_col, 'ClosingO25', 'Target_Over'}.issubset(df.columns):
        df['EV_OU'] = (df[p_over_col] * df['ClosingO25'] - 1) * 100
        df['EV_Bin_OU'] = pd.cut(df['EV_OU'], bins=bins, labels=labels)
        df['Profit_OU'] = np.where(
    df['Target_Over'] == 1, df['ClosingO25'] - 1, -1)

        roi_map_ou = df.groupby('EV_Bin_OU', observed=False)[
                                'Profit_OU'].agg(['mean', 'count'])
        boost_map_ou: Dict[str, float] = {}
        for bin_label, row in roi_map_ou.iterrows():
            cnt = row['count']
            if cnt < min_bin_size:
                boost_map_ou[bin_label] = 0.0
            else:
                roi = row['mean']
                boost_map_ou[bin_label] = float(np.clip(roi, -0.2, 0.2))
        results["OU"] = boost_map_ou

    # 5. BTTS Analysis
    p_btts_col = 'P_BTTS_Final' if 'P_BTTS_Final' in df.columns else 'Prob_BTTS'
    if {p_btts_col, 'ClosingBTTSY', 'Target_BTTS'}.issubset(df.columns):
        df['EV_BTTS'] = (df[p_btts_col] * df['ClosingBTTSY'] - 1) * 100
        df['EV_Bin_BTTS'] = pd.cut(df['EV_BTTS'], bins=bins, labels=labels)
        df['Profit_BTTS'] = np.where(
    df['Target_BTTS'] == 1, df['ClosingBTTSY'] - 1, -1)

        roi_map_bt = df.groupby('EV_Bin_BTTS', observed=False)[
                                'Profit_BTTS'].agg(['mean', 'count'])
        boost_map_bt: Dict[str, float] = {}
        for bin_label, row in roi_map_bt.iterrows():
            cnt = row['count']
            if cnt < min_bin_size:
                boost_map_bt[bin_label] = 0.0
            else:
                roi = row['mean']
                boost_map_bt[bin_label] = float(np.clip(roi, -0.2, 0.2))
        results["BTTS"] = boost_map_bt

    return results


def get_ev_boost(ev_val, boost_map):
    """Retrieves the EV adjustment factor based on the EV value and historical boost map."""
    if not boost_map:
        return 0.0
    if ev_val < -10: return boost_map.get('<-10', 0)
    elif ev_val < 0:
        return boost_map.get('-10to0', 0)
    elif ev_val < 10: return boost_map.get('0to10', 0)
    elif ev_val < 20:
        return boost_map.get('10to20', 0)
    elif ev_val < 30: return boost_map.get('20to30', 0)
    elif ev_val < 40:
        return boost_map.get('30to40', 0)
    else: return boost_map.get('40+', 0)


def compute_feature_importance(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates permutation feature importance for the Home Win prediction model."""
    df = train_df.copy()
    DECAY_TAU_DAYS = 365 * 1.5
    # Apply time-decay weights if not present
    if 'Weight' not in df.columns:
        days_diff = (df['Date'].max() - df['Date']).dt.days
        df['Weight'] = np.exp(-days_diff / DECAY_TAU_DAYS).values
    weights = df['Weight'].values
    home_feats = get_available_features(df, FEATURE_SETS['home'])
    X = df[home_feats].fillna(0)
    y = df['Target_Home'].astype(int)

    # Use a relatively shallow, fast HGBoost model for feature importance
    # calculation
    clf = HistGradientBoostingClassifier(
    max_depth=6,
    max_iter=400,
    learning_rate=0.06,
     random_state=42)
    clf.fit(X, y, sample_weight=weights)

    # Use neg_log_loss for scoring as it focuses on probability prediction
    # quality
    result = permutation_importance(
    clf,
    X,
    y,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
     scoring="neg_log_loss")
    fi_df = pd.DataFrame(
    {
        "Feature": home_feats,
        "Importance": result.importances_mean}).sort_values(
            "Importance",
             ascending=False)
    return fi_df


def _get_effective_odd(row, close_col, main_col, open_col, default=np.nan):
    """Retrieves the best available odd from closing, main, or opening columns."""
    val = row.get(close_col, np.nan)
    if pd.isna(val) or val < 1.01:
        val = row.get(main_col, np.nan)
    if pd.isna(val) or val < 1.01:
        val = row.get(open_col, np.nan)
    if pd.isna(val) or val < 1.01:
        return default
    return val

# 📌 UPDATED: Advanced League Scoring & Weak Filter using Final models + Patch #3 Logic


def build_league_metrics(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates comprehensive performance metrics for each league."""
    if hist_df.empty:
        return pd.DataFrame()
    rows = []

    for lig, grp in hist_df.groupby("League"):
        if len(grp) < 40:
            continue  # Minimum match count
        row = {"League": lig, "Matches": len(grp)}

        def compute_auc(y, p):
            p = pd.to_numeric(p, errors="coerce")
            valid = (~p.isna())
            yv, pv = y[valid], p[valid]
            if len(yv) < 30 or len(np.unique(yv)) < 2:
                return 0.5
            return float(roc_auc_score(yv, pv))

        def compute_roi(y, p, odd, market_name, min_bets=25):
            p = pd.to_numeric(p, errors="coerce")
            odd = pd.to_numeric(odd, errors="coerce")
            valid = (~p.isna()) & (~odd.isna()) & (odd >= 1.01)
            yv, pv, ov = y[valid], p[valid], odd[valid]
            if len(yv) < min_bets:
                return 0.0, 0

            # ✅ PATCH #3 & 2: Use specific thresholds from PROB_THRESHOLDS_HARD
            thr = PROB_THRESHOLDS_HARD.get(market_name, 0.50)

            m = pv >= thr
            if m.sum() < min_bets:
                return 0.0, int(m.sum())
            profit = np.where(yv[m] == 1, ov[m] - 1, -1)
            roi = float(np.nanmean(profit) * 100)
            return roi, int(m.sum())

        market_stats = {}

        # 1X2
        if {'Target_Result', 'P_Home_Final',
            'ClosingHomeOdd'}.issubset(grp.columns):
            y = (grp['Target_Result'] == 2).astype(int)
            p = grp['P_Home_Final']; o = grp['ClosingHomeOdd']
            row["AUC_MS1"] = compute_auc(y, p)
            row["ROI_MS1_%"], row["Bets_MS1"] = compute_roi(y, p, o, "MS 1")
            market_stats["MS 1"] = {
    "auc": row["AUC_MS1"],
    "roi": row["ROI_MS1_%"],
     "bets": row["Bets_MS1"]}
        else:
            row["AUC_MS1"] = 0.5; row["ROI_MS1_%"] = 0.0; row["Bets_MS1"] = 0

        if {'Target_Result', 'P_Draw_Final',
            'ClosingDrawOdd'}.issubset(grp.columns):
            y = (grp['Target_Result'] == 1).astype(int)
            p = grp['P_Draw_Final']; o = grp['ClosingDrawOdd']
            row["AUC_MSX"] = compute_auc(y, p)
            row["ROI_MSX_%"], row["Bets_MSX"] = compute_roi(y, p, o, "MS X")
            market_stats["MS X"] = {
    "auc": row["AUC_MSX"],
    "roi": row["ROI_MSX_%"],
     "bets": row["Bets_MSX"]}
        else:
            row["AUC_MSX"] = 0.5; row["ROI_MSX_%"] = 0.0; row["Bets_MSX"] = 0

        if {'Target_Result', 'P_Away_Final',
            'ClosingAwayOdd'}.issubset(grp.columns):
            y = (grp['Target_Result'] == 0).astype(int)
            p = grp['P_Away_Final']; o = grp['ClosingAwayOdd']
            row["AUC_MS2"] = compute_auc(y, p)
            row["ROI_MS2_%"], row["Bets_MS2"] = compute_roi(y, p, o, "MS 2")
            market_stats["MS 2"] = {
    "auc": row["AUC_MS2"],
    "roi": row["ROI_MS2_%"],
     "bets": row["Bets_MS2"]}
        else:
            row["AUC_MS2"] = 0.5; row["ROI_MS2_%"] = 0.0; row["Bets_MS2"] = 0

        # OU (Using P_Over_Final if available)
        p_over_col = 'P_Over_Final' if 'P_Over_Final' in grp.columns else 'Prob_Over'
        if {'Target_Over', p_over_col, 'ClosingO25',
            'ClosingU25'}.issubset(grp.columns):
            y_o = grp['Target_Over'].astype(int)
            p_o = grp[p_over_col]; o_o = grp['ClosingO25']
            row["AUC_O25"] = compute_auc(y_o, p_o)
            row["ROI_O25_%"], row["Bets_O25"] = compute_roi(
                y_o, p_o, o_o, "2.5 Üst")
            market_stats["2.5 Üst"] = {
    "auc": row["AUC_O25"],
    "roi": row["ROI_O25_%"],
     "bets": row["Bets_O25"]}

            y_u = 1 - grp['Target_Over'].astype(int)
            p_u = 1 - grp[p_over_col]; o_u = grp['ClosingU25']
            row["AUC_U25"] = compute_auc(y_u, p_u)
            row["ROI_U25_%"], row["Bets_U25"] = compute_roi(
                y_u, p_u, o_u, "2.5 Alt")
            market_stats["2.5 Alt"] = {
    "auc": row["AUC_U25"],
    "roi": row["ROI_U25_%"],
     "bets": row["Bets_U25"]}
            row["AUC_OU"] = max(row["AUC_O25"], row["AUC_U25"])
        else:
            row["AUC_O25"] = 0.5
            row["ROI_O25_%"] = 0.0; row["Bets_O25"] = 0
            row["AUC_U25"] = 0.5
            row["ROI_U25_%"] = 0.0; row["Bets_U25"] = 0
            row["AUC_OU"] = 0.5

        # KG (Using P_BTTS_Final if available)
        p_btts_col = 'P_BTTS_Final' if 'P_BTTS_Final' in grp.columns else 'Prob_BTTS'
        if {'Target_BTTS', p_btts_col, 'ClosingBTTSY',
            'ClosingBTTSN'}.issubset(grp.columns):
            y_y = grp['Target_BTTS'].astype(int)
            p_y = grp[p_btts_col]; o_y = grp['ClosingBTTSY']
            row["AUC_KGY"] = compute_auc(y_y, p_y)
            row["ROI_KGY_%"], row["Bets_KGY"] = compute_roi(
                y_y, p_y, o_y, "KG Var")
            market_stats["KG Var"] = {
    "auc": row["AUC_KGY"],
    "roi": row["ROI_KGY_%"],
     "bets": row["Bets_KGY"]}

            y_n = 1 - grp['Target_BTTS'].astype(int)
            p_n = 1 - grp[p_btts_col]; o_n = grp['ClosingBTTSN']
            row["AUC_KGN"] = compute_auc(y_n, p_n)
            row["ROI_KGN_%"], row["Bets_KGN"] = compute_roi(
                y_n, p_n, o_n, "KG Yok")
            market_stats["KG Yok"] = {
    "auc": row["AUC_KGN"],
    "roi": row["ROI_KGN_%"],
     "bets": row["Bets_KGN"]}
            row["AUC_BTTS"] = max(row["AUC_KGY"], row["AUC_KGN"])
        else:
            row["AUC_KGY"] = 0.5
            row["ROI_KGY_%"] = 0.0; row["Bets_KGY"] = 0
            row["AUC_KGN"] = 0.5
            row["ROI_KGN_%"] = 0.0; row["Bets_KGN"] = 0
            row["AUC_BTTS"] = 0.5

        # Home ROI Legacy
        sel = grp[grp["P_Home_Final"] > 0.5].copy()
        sel["Eff_Home_Odd"] = sel.apply(lambda r: _get_effective_odd(
            r, "ClosingHomeOdd", "HomeOdd", "Odds_Open_Home"), axis=1)
        sel = sel[sel["Eff_Home_Odd"].notna()].copy()
        if len(sel) > 0:
            profit = np.where(
sel["Target_Result"] == 2, sel["Eff_Home_Odd"] - 1, -1)
            row["Home_ROI_%"] = float(np.nanmean(profit) * 100)
            row["Home_Bets"] = int(len(sel))
        else:
            row["Home_ROI_%"] = 0.0; row["Home_Bets"] = 0

        # Best Market Determination
        best_market = None
        best_score = -999
        for m_name, s in market_stats.items():
            if s["bets"] < 20:
                continue
            score_m = (s["auc"] - 0.5) / 0.2 + (s["roi"] / 15.0)
            if score_m > best_score:
                best_score = score_m
                best_market = (m_name, s["auc"], s["roi"])

        if best_market:
            row["Best_Market"] = best_market[0]
            row["Best_Market_AUC"] = float(best_market[1])
            row["Best_Market_ROI_%"] = float(best_market[2])
        else:
            row["Best_Market"] = None
            row["Best_Market_AUC"] = 0.5; row["Best_Market_ROI_%"] = 0.0

        # --- LEAGUE SCORE CALCULATION ---
        auc_ms1 = row["AUC_MS1"]
        best_auc_all = max([s["auc"] for s in market_stats.values()] + [0.5])
        all_rois = [s["roi"]
            for s in market_stats.values() if not np.isnan(s["roi"])]
        best_roi_all = max(all_rois) if all_rois else -5.0
        matches = row["Matches"]

        auc_ms1_norm = np.clip((auc_ms1 - 0.55) / 0.15, 0.0, 1.0)
        auc_best_norm = np.clip((best_auc_all - 0.60) / 0.15, 0.0, 1.0)
        roi_norm = np.clip((best_roi_all + 5.0) / 25.0, 0.0, 1.0)
        vol_norm = np.clip((matches - 80) / 220.0, 0.0, 1.0)

        raw_score = 0.35 * auc_ms1_norm + 0.35 * \
            auc_best_norm + 0.20 * roi_norm + 0.10 * vol_norm
        row["League_Score"] = float(0.8 + raw_score * 0.6)
        row["Weak_League"] = False

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df_league = pd.DataFrame(rows)

    q25 = df_league["League_Score"].quantile(0.25)

    mask_low_vol = (
        (df_league["Matches"] < 120)
        & (df_league["League_Score"] < 1.0)
    )

    mask_bottom = (
        (df_league["League_Score"] <= q25)
        & (df_league["Matches"] >= 120)
    )

    df_league.loc[mask_low_vol | mask_bottom, "Weak_League"] = True

    return df_league.sort_values("AUC_MS1", ascending=False)



def build_market_quality(hist_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates the 'E-Score' (normalized AUC) for each market."""
    quality = {}
    if not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
        st.session_state["_market_quality_missing_cols"] = ["hist_df"]
        st.session_state["_market_quality_cols"] = (
            list(hist_df.columns)[:200] if isinstance(hist_df, pd.DataFrame) else []
        )
        return quality
    st.session_state["_market_quality_cols"] = list(hist_df.columns)[:200]
    missing_cols = []

    def get_score(y, p):
        if len(np.unique(y)) > 1:
            auc = roc_auc_score(y, p)
            return (auc - 0.5) / 0.2
        return 0.0
    try:
        target_series = None
        for col in ("Target_Result", "RESULT", "FT_Result", "Result"):
            if col in hist_df.columns:
                target_series = hist_df[col]
                break
        if target_series is None and all(
                c in hist_df.columns for c in ("HomeWin", "Draw", "AwayWin")):
            target_series = np.select(
                [
                    hist_df["HomeWin"].astype(bool),
                    hist_df["Draw"].astype(bool),
                    hist_df["AwayWin"].astype(bool),
                ],
                [2, 1, 0],
                default=np.nan,
            )
        if target_series is None:
            missing_cols.append("Target_Result")
            st.session_state["_market_quality_missing_cols"] = missing_cols
            return quality

        if "P_Home_Final" in hist_df.columns:
            quality["MS 1"] = get_score(
                (pd.to_numeric(target_series, errors="coerce") == 2).astype(int),
                hist_df["P_Home_Final"],
            )
        else:
            missing_cols.append("P_Home_Final")

        if "P_Draw_Final" in hist_df.columns:
            quality["MS X"] = get_score(
                (pd.to_numeric(target_series, errors="coerce") == 1).astype(int),
                hist_df["P_Draw_Final"],
            )
        else:
            missing_cols.append("P_Draw_Final")

        if "P_Away_Final" in hist_df.columns:
            quality["MS 2"] = get_score(
                (pd.to_numeric(target_series, errors="coerce") == 0).astype(int),
                hist_df["P_Away_Final"],
            )
        else:
            missing_cols.append("P_Away_Final")
    except Exception:
        st.session_state["_last_market_quality_error"] = traceback.format_exc()
        st.session_state["_market_quality_missing_cols"] = missing_cols
        return {}

    # 📌 UPDATED: Uses P_Over_Final / P_BTTS_Final
    p_over_col = 'P_Over_Final' if 'P_Over_Final' in hist_df.columns else 'Prob_Over'
    if "Target_Over" in hist_df.columns and p_over_col in hist_df.columns:
        s = get_score(hist_df["Target_Over"].astype(int), hist_df[p_over_col])
        quality["2.5 Üst"] = s
        quality["2.5 Alt"] = s

    p_btts_col = 'P_BTTS_Final' if 'P_BTTS_Final' in hist_df.columns else 'Prob_BTTS'
    if "Target_BTTS" in hist_df.columns and p_btts_col in hist_df.columns:
        s = get_score(hist_df["Target_BTTS"].astype(int), hist_df[p_btts_col])
        quality["KG Var"] = s
        quality["KG Yok"] = s

    for k, v in quality.items():
        quality[k] = float(np.clip(v, 0.0, 1.0))
    st.session_state["_market_quality_missing_cols"] = missing_cols
    return quality


def optimize_betting_strategy(df):
    """Searches for the optimal combination of Min EV and Min Confidence to maximize a risk-adjusted metric."""
    df = df.copy()
    p_col = 'P_Home_Final' if 'P_Home_Final' in df.columns else 'P_Home'
    req_cols = [p_col, 'ClosingHomeOdd', 'Target_Result']
    if not all(col in df.columns for col in req_cols):
        return {'min_ev': 0, 'min_conf': 0.50, 'roi': 0, 'bets': 0}
    df['Target_Home'] = (df['Target_Result'] == 2).astype(int)
    df['EV_Calc'] = (df[p_col] * df['ClosingHomeOdd'] - 1) * 100

    best_metric = -100
    best_params = {'min_ev': 0, 'min_conf': 0.50, 'roi': 0, 'bets': 0}

    for ev_min in np.linspace(-5, 5, 5):
        for conf_min in np.linspace(0.40, 0.60, 5):
            subset = df[(df['EV_Calc'] >= ev_min) & (df[p_col] >= conf_min)]
            if len(subset) < 20:
                continue
            profit_series = np.where(
    subset['Target_Home'] == 1, subset['ClosingHomeOdd'] - 1, -1)
            roi = profit_series.mean()

            cum_pnl = np.cumsum(profit_series)
            peak = np.maximum.accumulate(cum_pnl)
            drawdown = peak - cum_pnl
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

            metric_val = (roi * 100) - (0.3 * max_dd)

            if metric_val > best_metric:
                best_metric = metric_val
                best_params = {
    'min_ev': float(ev_min),
    'min_conf': float(conf_min),
    'roi': float(
        roi * 100),
        'bets': int(
            len(subset))}
    return best_params

# -----------------------------------------------------------------------------#
# 7. META-LAYER: ARCHETYPE RECOGNITION
# -----------------------------------------------------------------------------#


ARCHETYPE_MULTIPLIERS = {
    "LOW_TEMP_CLOSE": {"MS": 1.05, "OTH": 1.00},
    "FAVORITE_HOLD": {"MS": 1.10, "OTH": 1.03},
    "CHAOTIC_HIGH_TEMPO": {"MS": 1.12, "OTH": 1.05},
    "UPSET_SIGNATURE": {"MS": 1.15, "OTH": 1.00},
    "TRAP_FAVORITE": {"MS": 0.92, "OTH": 1.00},
    "NONE": {"MS": 1.00, "OTH": 1.00},
}


def classify_archetype(row):
    """Classifies a match into a dynamic archetype based on pre-match features."""
    pre_xg_home = row.get("Roll_xG_5_Home", 1.3)
    pre_xg_away = row.get("Roll_xG_5_Away", 1.1)
    form_diff_z = row.get("Form_TMB_Diff", 0.0)
    raw_tempo = row.get("Tempo_Index", 2.5)
    tempo_index = np.clip((raw_tempo - 1.5) / 2.5, 0, 1)

    close_home_odds = _get_effective_odd(
    row, "ClosingHomeOdd", "HomeOdd", "Odds_Open_Home")
    close_away_odds = _get_effective_odd(
    row, "ClosingAwayOdd", "AwayOdd", "Odds_Open_Away")

    if pd.isna(close_home_odds) or close_home_odds < 1.01:
        return "NONE"

    prob_home = row.get("P_Home_Final", 0.33)
    impl_home_prob = 1.0 / close_home_odds
    # Trap favorite: market is more confident than the model on a short-priced
    # favorite
#     try:  # AUTO-COMMENTED (illegal global try)
#         if close_home_odds <= 1.60 and (prob_home - impl_home_prob) <= -0.06:
#             return "TRAP_FAVORITE"
#     except Exception:
#         pass

    exp_goal_diff = abs(pre_xg_home - pre_xg_away)

    if (exp_goal_diff <= 0.30 and abs(form_diff_z) <= 0.6 and tempo_index <=
        0.40 and close_home_odds >= 1.70 and close_away_odds >= 2.80):
        return "LOW_TEMP_CLOSE"
    if ((pre_xg_home - pre_xg_away) >= 0.4 and form_diff_z >=
        0.5 and close_home_odds <= 1.70 and prob_home >= 0.60):
        return "FAVORITE_HOLD"
    if (tempo_index >= 0.60 and pre_xg_home >=
        1.3 and pre_xg_away >= 1.1 and abs(form_diff_z) <= 1.5):
        return "CHAOTIC_HIGH_TEMPO"
    if (close_home_odds <= 1.85 and prob_home <= (
        impl_home_prob - 0.05) and form_diff_z <= 0.2):
        return "UPSET_SIGNATURE"
    return "NONE"


def apply_archetype_layer(df):
    """Applies archetype-specific multipliers to the GoldenScore."""
    if df.empty:
        return df
    df = df.copy()
    df["Archetype"] = df.apply(classify_archetype, axis=1)

    def get_mult(row):
        arch = row.get("Archetype", "NONE")
        sel = row.get('Seçim', '')
        base_mult = 1.0

        # 🩹 PATCH 2 Update: Soften under boost
        if arch == "LOW_TEMP_CLOSE":
            if sel in ["2.5 Alt", "KG Yok", "MS X"]:
                base_mult = 1.05   # 1.10 -> 1.05
            elif sel in ["2.5 Üst", "KG Var"]:
                base_mult = 0.95   # 0.90 -> 0.95
        elif arch == "CHAOTIC_HIGH_TEMPO":
            if sel in ["2.5 Üst", "KG Var"]:
                base_mult = 1.10
            elif sel in ["2.5 Alt", "KG Yok"]: base_mult = 0.90
        elif arch == "FAVORITE_HOLD":
            if sel == "MS 1":
                base_mult = 1.05

        mtype = 'MS' if sel in ['MS 1', 'MS X', 'MS 2'] else 'OTH'
        return base_mult * \
            ARCHETYPE_MULTIPLIERS.get(
    arch, ARCHETYPE_MULTIPLIERS['NONE'])[mtype]

    df["Archetype_Multiplier"] = df.apply(get_mult, axis=1)
    # Note: Multiplier application moved to compute_golden_score
    return df

# -----------------------------------------------------------------------------#
# NEW HELPERS: GOLDEN SCORE & BEST OF LIST
# -----------------------------------------------------------------------------#


def validate_odds_row(row):
    """Checks if the odds for a selection are valid."""
    odd = row.get('Odd')
    if pd.isna(odd) or odd < 1.10:
        return False
    return True

# 📌 UPDATED: Generalized resolve_odd logic and Early League Mult


def compute_ev_scores(bets_df, df, market_quality):
    """Calculates Expected Value (EV) and applies market/league quality adjustments."""
    ev_boost_maps = st.session_state.get('ev_boost_map', {})
    if isinstance(bets_df, pd.DataFrame):
        _sel_aliases = ["Seçim", "Secim", "Selection", "Pick", "Market", "BetType", "MarketType"]
        _sel_src = next((c for c in _sel_aliases if c in bets_df.columns), None)
        if _sel_src is None:
            bets_df["Seçim"] = ""
            st.session_state["_dbg_missing_selection_col"] = True
        else:
            bets_df["Seçim"] = bets_df[_sel_src]
            st.session_state["_dbg_missing_selection_col"] = False
        bets_df["Seçim"] = bets_df["Seçim"].astype(str)
    else:
        st.session_state["_dbg_missing_selection_col"] = True
    _missing_date_cols = []
    _date_source = "missing"

    def _ensure_date_col(_df):
        nonlocal _date_source
        if not isinstance(_df, pd.DataFrame):
            return _df
        if "Date" in _df.columns and "Time" in _df.columns:
            _df["Date"] = pd.to_datetime(
                _df["Date"].astype(str) + " " + _df["Time"].astype(str),
                errors="coerce"
            ).dt.date
            _date_source = "Date+Time"
            return _df
        if "Date" in _df.columns:
            _df["Date"] = pd.to_datetime(_df["Date"], errors="coerce").dt.date
            _date_source = "Date"
            return _df
        for c in ("Date Time", "Datetime", "DateTime"):
            if c in _df.columns:
                _df["Date"] = pd.to_datetime(_df[c], errors="coerce").dt.date
                _date_source = "Date Time"
                return _df
        _df["Date"] = pd.NaT
        return _df

    def _ensure_team_cols(_df):
        if not isinstance(_df, pd.DataFrame):
            return _df
        if "HomeTeam" not in _df.columns:
            for c in ("Home_Team", "Home", "Home Team"):
                if c in _df.columns:
                    _df["HomeTeam"] = _df[c]
                    break
        if "HomeTeam" not in _df.columns:
            _df["HomeTeam"] = ""
        if "AwayTeam" not in _df.columns:
            for c in ("Away_Team", "Away", "Away Team"):
                if c in _df.columns:
                    _df["AwayTeam"] = _df[c]
                    break
        if "AwayTeam" not in _df.columns:
            _df["AwayTeam"] = ""
        return _df

    df = _ensure_date_col(df)
    bets_df = _ensure_date_col(bets_df)
    df = _ensure_team_cols(df)
    bets_df = _ensure_team_cols(bets_df)
    if isinstance(df, pd.DataFrame) and df["Date"].isna().all():
        _missing_date_cols = [c for c in ("Date", "Date Time", "Datetime", "DateTime", "Time")]
    st.session_state["_dbg_date_source"] = _date_source
    st.session_state["_dbg_missing_date_cols"] = _missing_date_cols

    if 'Smart_EV' not in df.columns:
        df['Smart_EV'] = (df.get('P_Home_Final', 0.5) *
                          df.get('HomeOdd', 2.0) - 1) * 100

    _id_col = "Match_ID" if "Match_ID" in df.columns else (
        "MatchID" if "MatchID" in df.columns else None)
    if _id_col:
        df['MatchID_Unique'] = df[_id_col].astype(str)
        bets_df['MatchID_Unique'] = bets_df.get(
            _id_col, pd.Series("", index=bets_df.index)).astype(str)
    else:
        df['MatchID_Unique'] = df['Date'].astype(
            str) + "_" + df['HomeTeam'] + "_" + df['AwayTeam']
        bets_df['MatchID_Unique'] = bets_df['Date'].astype(
            str) + "_" + bets_df['HomeTeam'] + "_" + bets_df['AwayTeam']
    smart_ev_map = df.set_index('MatchID_Unique')['Smart_EV'].to_dict()
    bets_df['Smart_EV_Mapped'] = bets_df['MatchID_Unique'].map(smart_ev_map)

    if isinstance(bets_df, pd.DataFrame):
        _league_aliases = ["League", "Lig", "League_Name", "LeagueName"]
        _league_src = next((c for c in _league_aliases if c in bets_df.columns), None)
        if _league_src is None:
            bets_df["League"] = ""
            st.session_state["_dbg_missing_league_col"] = True
        else:
            bets_df["League"] = bets_df[_league_src]
            st.session_state["_dbg_missing_league_col"] = False
    else:
        st.session_state["_dbg_missing_league_col"] = True
    league_metrics = st.session_state.get('league_metrics_df', pd.DataFrame())
    if not league_metrics.empty and "League" in bets_df.columns:
        l_score_map = league_metrics.set_index(
            "League")["League_Score"].to_dict()
        bets_df["League_Mult"] = bets_df["League"].map(l_score_map).fillna(1.0)
    else:
        bets_df["League_Mult"] = 1.0

    # 📌 6️⃣ Generalized Odds Fallback
    def resolve_odd(row):
        m = row['Seçim']
        if m == "MS 1":
            val = _get_effective_odd(
    row,
    "ClosingHomeOdd",
    "HomeOdd",
    "Odds_Open_Home",
     default=np.nan)
        elif m == "MS X":
            val = _get_effective_odd(
    row,
    "ClosingDrawOdd",
    "DrawOdd",
    "Odds_Open_Draw",
     default=np.nan)
        elif m == "MS 2":
            val = _get_effective_odd(
    row,
    "ClosingAwayOdd",
    "AwayOdd",
    "Odds_Open_Away",
     default=np.nan)
        elif m == "2.5 Üst":
            val = _get_effective_odd(
    row,
    "ClosingO25",
    "O25",
    "Odds_Open_Over25",
     default=np.nan)
        elif m == "2.5 Alt":
            val = _get_effective_odd(
    row,
    "ClosingU25",
    "U25",
    "Odds_Open_Under25",
     default=np.nan)
        elif m == "KG Var":
            val = _get_effective_odd(
    row,
    "ClosingBTTSY",
    "BTTSY",
    "Odds_Open_BTTS_Yes",
     default=np.nan)
            if pd.isna(val):
                val = _get_effective_odd(
    row,
    "ClosingBTTS_Y",
    "BTTS_Y",
    "Odds_Open_BTTS_Yes",
     default=np.nan)
        elif m == "KG Yok":
            val = _get_effective_odd(
    row,
    "ClosingBTTSN",
    "BTTSN",
    "Odds_Open_BTTS_No",
     default=np.nan)
            if pd.isna(val):
                val = _get_effective_odd(
    row,
    "ClosingBTTS_N",
    "BTTS_N",
    "Odds_Open_BTTS_No",
     default=np.nan)
        else:
            val = np.nan
        if isinstance(val, (pd.Series, pd.DataFrame)):
            st.session_state["_dbg_resolve_odd_scalar"] = {
                "selection": m,
                "type": str(type(val)),
            }
            return np.nan
        return val

    # Recalculate Odd column with robust fallback
    _odd_res = bets_df.apply(resolve_odd, axis=1)
    if isinstance(_odd_res, pd.DataFrame):
        st.session_state["_dbg_resolve_odd_multi"] = list(_odd_res.columns)
        st.session_state["_dbg_resolve_odd_bad_rows"] = list(_odd_res.index[:5])
        _sel = bets_df.get("Seçim", pd.Series("", index=bets_df.index)).astype(str)
        _map = {
            "MS 1": ["HomeOdd"],
            "MS X": ["DrawOdd"],
            "MS 2": ["AwayOdd"],
            "2.5 Üst": ["O25"],
            "2.5 Alt": ["U25"],
            "KG Var": ["BTTSY", "BTTS_Y"],
            "KG Yok": ["BTTSN", "BTTS_N"],
        }
        _odd_series = pd.Series(np.nan, index=_odd_res.index, dtype=float)
        _fallback_rows = []
        for _mkt, _cols in _map.items():
            _mask = _sel.eq(_mkt)
            if not _mask.any():
                continue
            _col = next((c for c in _cols if c in _odd_res.columns), None)
            if _col is None:
                _fallback_rows.extend(_odd_res.index[_mask].tolist())
                continue
            _odd_series.loc[_mask] = pd.to_numeric(_odd_res.loc[_mask, _col], errors="coerce")
        _missing_mask = _odd_series.isna()
        if _missing_mask.any():
            miss_pos = np.flatnonzero(_missing_mask.to_numpy())
            _odd_series.iloc[miss_pos] = pd.to_numeric(_odd_res.iloc[miss_pos, 0], errors="coerce")
            _fallback_rows.extend(_odd_res.index[miss_pos].tolist())
        if _fallback_rows:
            st.session_state["_dbg_resolve_odd_fallback_rows"] = _fallback_rows[:5]
        bets_df["Odd"] = _odd_series
    else:
        bets_df["Odd"] = pd.to_numeric(_odd_res, errors="coerce")

    if not isinstance(bets_df, pd.DataFrame):
        st.session_state["_dbg_prob_source_cols"] = {}
        st.session_state["_dbg_prob_missing_count"] = 0
        return bets_df

    def resolve_prob(row):
        m = row['Seçim']
        prob_map = {
            "MS 1": ["P_Home_Final", "P_Home", "Prob_Home", "Poisson_Home_Pct"],
            "MS X": ["P_Draw_Final", "P_Draw", "Prob_Draw", "Poisson_Draw_Pct"],
            "MS 2": ["P_Away_Final", "P_Away", "Prob_Away", "Poisson_Away_Pct"],
            "2.5 Üst": ["P_O25_Final", "P_Over25_Final", "P_Over25", "Prob_O25",
                        "Poisson_O25_Pct", "Poisson_Over25_Pct"],
            "2.5 Alt": ["P_U25_Final", "P_Under25_Final", "P_Under25", "Prob_U25",
                        "Poisson_U25_Pct", "Poisson_Under25_Pct"],
            "KG Var": ["P_BTTSY_Final", "P_BTTS_Yes_Final", "P_BTTS_Yes",
                       "Prob_BTTSY", "Poisson_BTTS_Yes_Pct"],
            "KG Yok": ["P_BTTSN_Final", "P_BTTS_No_Final", "P_BTTS_No",
                       "Prob_BTTSN", "Poisson_BTTS_No_Pct"],
        }
        cols = prob_map.get(m, [])
        for c in cols:
            if c in row.index:
                val = row.get(c)
                if isinstance(val, str):
                    val = val.replace("%", "").replace(",", ".").strip()
                val = pd.to_numeric(val, errors="coerce")
                if pd.notna(val):
                    return val
        return np.nan

    _prob_src = {}
    for _m, _cols in {
        "MS 1": ["P_Home_Final", "P_Home", "Prob_Home", "Poisson_Home_Pct"],
        "MS X": ["P_Draw_Final", "P_Draw", "Prob_Draw", "Poisson_Draw_Pct"],
        "MS 2": ["P_Away_Final", "P_Away", "Prob_Away", "Poisson_Away_Pct"],
        "2.5 Üst": ["P_O25_Final", "P_Over25_Final", "P_Over25", "Prob_O25",
                    "Poisson_O25_Pct", "Poisson_Over25_Pct"],
        "2.5 Alt": ["P_U25_Final", "P_Under25_Final", "P_Under25", "Prob_U25",
                    "Poisson_U25_Pct", "Poisson_Under25_Pct"],
        "KG Var": ["P_BTTSY_Final", "P_BTTS_Yes_Final", "P_BTTS_Yes",
                   "Prob_BTTSY", "Poisson_BTTS_Yes_Pct"],
        "KG Yok": ["P_BTTSN_Final", "P_BTTS_No_Final", "P_BTTS_No",
                   "Prob_BTTSN", "Poisson_BTTS_No_Pct"],
    }.items():
        _src = next((c for c in _cols if c in bets_df.columns), None)
        _prob_src[_m] = f"first_available:{_src}" if _src else "missing"
    st.session_state["_dbg_prob_source_cols"] = _prob_src

    bets_df["Prob"] = bets_df.apply(resolve_prob, axis=1)
    bets_df["Prob"] = pd.to_numeric(bets_df["Prob"], errors="coerce")
    st.session_state["_dbg_prob_missing_count"] = int(bets_df["Prob"].isna().sum())
    bets_df["Prob_dec"] = np.where(bets_df["Prob"] > 1.0, bets_df["Prob"] / 100.0, bets_df["Prob"])

    def raw_ev_row(row):
        m = row['Seçim']
        if pd.isna(row['Odd']) or row['Odd'] < 1.01:
            return -10.0

        # --- Standardize Prob to decimal (0-1)
        prob_dec = row.get("Prob_dec", row.get("Prob", np.nan))
        if pd.isna(prob_dec):
            return -10.0
        # Pure EV (percent) computed from standardized prob
        base_val = (prob_dec * row['Odd'] - 1) * 100

        # 📌 2️⃣ Apply League Multiplier EARLY
        l_mult = row.get("League_Mult", 1.0)
        base_val = base_val * l_mult

        base_val = max(min(base_val, 25), -20)

        boost = 0
        # 📌 7️⃣ Use specific EV boost maps if available
        if m == "MS 1":
            boost = get_ev_boost(
    base_val, ev_boost_maps.get(
        "MS 1", ev_boost_maps.get(
            "Global", {})))
        elif m == "MS X": boost = get_ev_boost(base_val, ev_boost_maps.get("MS X", ev_boost_maps.get("Global", {})))
        elif m == "MS 2":
            boost = get_ev_boost(
    base_val, ev_boost_maps.get(
        "MS 2", ev_boost_maps.get(
            "Global", {})))
        elif m in ["2.5 Üst", "2.5 Alt"]: boost = get_ev_boost(base_val, ev_boost_maps.get("OU", {}))
        elif m in ["KG Var", "KG Yok"]:
            boost = get_ev_boost(base_val, ev_boost_maps.get("BTTS", {}))

        # Blend EV for 1X2 with Smart EV (with Pure-EV guard)
        pure_val = base_val
        if m in ['MS 1', 'MS X', 'MS 2']:
            smart_val = row.get('Smart_EV_Mapped', 0)
            blended = 0.7 * smart_val + 0.3 * base_val
            # Guardrail: if the bet is overpriced vs market (pure EV < 0),
            # don't let blending flip it strongly positive
            if pure_val < 0 and blended > 0:
                # Keep the sign consistent with "pure" edge. If model<market (pure<0) but smart makes it positive,
                # fall back to a conservative pure-based EV (keeps negative,
                # avoids 0.00 illusion).
                blended = 0.30 * pure_val
            base_val = blended

        final_val = base_val + (boost * 5.0)
        # 📌 3️⃣ Removed 'final_val *= 1.12' for OU/KG

        if isinstance(final_val, (pd.Series, pd.DataFrame)):
            st.session_state["_dbg_raw_ev_scalar"] = {
                "selection": m,
                "type": str(type(final_val)),
            }
            return np.nan
        return final_val

    _ev_res = bets_df.apply(raw_ev_row, axis=1)
    if isinstance(_ev_res, pd.DataFrame):
        st.session_state["_dbg_raw_ev_multi_cols"] = list(_ev_res.columns)
        st.session_state["_dbg_raw_ev_bad_rows"] = list(_ev_res.index[:5])
        st.session_state["_dbg_raw_ev_fallback"] = True
        bets_df["EV_raw"] = pd.to_numeric(_ev_res.iloc[:, 0], errors="coerce")
    else:
        st.session_state["_dbg_raw_ev_fallback"] = False
        bets_df["EV_raw"] = pd.to_numeric(_ev_res, errors="coerce")

    EV_SCALE = {
        "MS 1": 1.00, "MS 2": 1.00, "MS X": 0.95,
        "2.5 Üst": 0.90, "2.5 Alt": 0.90, "KG Var": 0.90, "KG Yok": 0.90,
    }
    EV_CLIP_LOW = {
    "MS 1": -10,
    "MS 2": -8,
    "MS X": -12,
    "2.5 Üst": -8,
    "2.5 Alt": -8,
    "KG Var": -8,
     "KG Yok": -8}
    EV_CLIP_HIGH = {
    "MS 1": 20,
    "MS 2": 20,
    "MS X": 22,
    "2.5 Üst": 20,
    "2.5 Alt": 20,
    "KG Var": 20,
     "KG Yok": 20}

    ev_quality_mult = {}
    for m in [
    'MS 1',
    'MS X',
    'MS 2',
    '2.5 Üst',
    '2.5 Alt',
    'KG Var',
     'KG Yok']:
        q = market_quality.get(m, 0.5)
        if m in ["2.5 Üst", "2.5 Alt", "KG Var", "KG Yok"]:
            ev_quality_mult[m] = 0.7 + 0.5 * q
        else:
            ev_quality_mult[m] = 0.5 + 0.8 * q

    def adjust_ev(row):
        m = row['Seçim']
        scale = EV_SCALE.get(m, 0.8)
        low = EV_CLIP_LOW.get(m, -10)
        high = EV_CLIP_HIGH.get(m, 18)
        q_mult = ev_quality_mult.get(m, 0.9)
        # League mult already applied in raw_ev_row, removed from here
        ev_val = row['EV_raw'] * scale * q_mult
        val = np.clip(ev_val, low, high)
        if isinstance(val, (pd.Series, pd.DataFrame)):
            st.session_state["_dbg_adjust_ev_scalar"] = {
                "selection": m,
                "type": str(type(val)),
            }
            return np.nan
        return val

    _ev_adj_res = bets_df.apply(adjust_ev, axis=1)
    if isinstance(_ev_adj_res, pd.DataFrame):
        st.session_state["_dbg_adjust_ev_multi_cols"] = list(_ev_adj_res.columns)
        st.session_state["_dbg_adjust_ev_bad_rows"] = list(_ev_adj_res.index[:5])
        st.session_state["_dbg_adjust_ev_fallback"] = True
        bets_df["EV"] = pd.to_numeric(_ev_adj_res.iloc[:, 0], errors="coerce")
    else:
        st.session_state["_dbg_adjust_ev_fallback"] = False
        bets_df["EV"] = pd.to_numeric(_ev_adj_res, errors="coerce")

    # ---------------------------------------------------------------------
    # EV / Market sanity columns (used for Trap detection & debugging)
    # ---------------------------------------------------------------------
    # Robust numeric parsing for Odd (handles '2,10', etc.)
    # ---------------------------------------------------------------------
    odd_raw = bets_df.get("Odd", np.nan)
    if isinstance(odd_raw, pd.Series):
        odd_str = odd_raw.astype(str)
    else:
        odd_str = pd.Series(str(odd_raw), index=bets_df.index)
    odd_num = pd.to_numeric(
        odd_str
        .str.replace(",", ".", regex=False)
              .str.strip(),
        errors="coerce"
    )
    bets_df["Odd_num"] = odd_num
    bets_df["Implied_Prob"] = np.where(odd_num > 0, 1.0 / odd_num, np.nan)

    lmult = bets_df["League_Mult"] if "League_Mult" in bets_df.columns else 1.0

    bets_df["EV_pure_pct"] = (
        pd.to_numeric(bets_df["Prob_dec"], errors="coerce")
        * pd.to_numeric(bets_df.get("Odd_num", bets_df["Odd"]), errors="coerce")
        - 1.0
    ) * 100.0 * lmult

    bets_df["EV_pure_pct"] = bets_df["EV_pure_pct"].clip(-50, 50)


    bets_df["Model_vs_Market"] = bets_df["Prob_dec"] - bets_df["Implied_Prob"]

    # Trap Favorite flag: short odds, but model is materially less confident than market
       # Trap flag (generic): market implies high probability, but model is materially less confident
    # Works for MS / OU / KG markets. We use implied probability rather than a
    # hard odds cutoff.
    _imp = pd.to_numeric(bets_df["Implied_Prob"], errors="coerce").fillna(0.0)
    _mvm = pd.to_numeric(bets_df["Model_vs_Market"], errors="coerce").fillna(0.0)
    _sel = bets_df["Seçim"].astype(str)

    # Baseline trigger: market >= 60% implied, model at least 6pp lower
    base_trap = (_imp >= 0.60) & (_mvm <= -0.06)

    # Slightly looser for 1X2 short favorites (common 'banko' trap zone)
    ms_trap = (_sel.isin(["MS 1", "MS 2"])) & (
        _imp >= 0.62) & (_mvm <= -0.05)

    # Extra guardrail: only surface when we are otherwise "confident"
    conf_hi = bets_df.get(
        "CONF_Status", pd.Series("", index=bets_df.index)).astype(
            str).str.upper().isin(["HIGH", "GREEN"])
    am_hi = bets_df.get(
        "AutoMod_Status", pd.Series("", index=bets_df.index)).astype(
            str).str.upper().isin(["HIGH", "GREEN"])
    stars_hi = pd.to_numeric(
bets_df.get(
    "Star_Rating",
    pd.Series(0, index=bets_df.index)),
     errors="coerce").fillna(0) >= 3

    bets_df["Trap_Flag"] = (
base_trap | ms_trap) & (
    conf_hi | am_hi | stars_hi)

    # Optional: categorize for UI/debug
    bets_df["Trap_Type"] = np.where(
    ms_trap, "TRAP_FAV_MS", np.where(
        base_trap, "TRAP_MARKET_OVERCONF", ""))
#     except Exception:  # AUTO-COMMENTED (orphan except)
#         bets_df["Trap_Flag"] = False
#         bets_df["Trap_Type"] = ""

    # 📌 3️⃣ Soft Dynamic Adjustment for OU/KG instead of hard 0.75 clamp
    for m in ["2.5 Üst", "2.5 Alt", "KG Var", "KG Yok"]:
        q = market_quality.get(m, 0.5)
        # 0.70 to 1.0 range multiplier
        mult = 0.7 + 0.3 * q
        mask = bets_df['Seçim'] == m
        bets_df.loc[mask, 'EV'] *= mult

    def dampen_noise(ev):
        if -2 < ev < 5:
            return ev * 0.5
        return ev
    bets_df['EV'] = bets_df['EV'].apply(dampen_noise)

    # ---------------------------------------------------------------------
    # Trap Favorite penalty: short-priced favorites where model < market
    # ---------------------------------------------------------------------
    if "Trap_Flag" in bets_df.columns:
        mask_trap = bets_df["Trap_Flag"].fillna(False)
        # reduce EV impact to avoid 'banko illusion' in BestOf
        bets_df.loc[mask_trap, "EV"] *= 0.60
        # tag archetype for UI/debugging (doesn't break if column absent)
        if "Archetype" in bets_df.columns:
            bets_df.loc[mask_trap, "Archetype"] = "TRAP_FAVORITE"
        else:
            bets_df["Archetype"] = np.where(mask_trap, "TRAP_FAVORITE", "NONE")
    mask_low_odd = (
    (pd.to_numeric(
        bets_df.get(
            'Prob_dec',
            bets_df.get(
                'Prob',
                np.nan)),
                errors='coerce') > 0.68) & (
                    pd.to_numeric(
                        bets_df.get(
                            'Odd_num',
                            bets_df.get(
                                'Odd',
                                np.nan)),
                                 errors='coerce') < 1.35))
    bets_df.loc[mask_low_odd, 'EV'] *= 0.4
    return bets_df

# 📌 5️⃣ Integrate Archetype directly


def compute_golden_score(df: pd.DataFrame,
    market_quality: Dict[str,
    float] = None,
     use_archetype: bool = True) -> pd.DataFrame:
    """Combines Prob, Score, and EV into a single 'GoldenScore' (0-1).

    Notes:
    - Uses per-market z-scores to keep scales comparable.
    - Robust to string % columns (e.g. '58.7%') and missing values.
    - Exposes GoldenScore_raw (unbounded) for debugging.
    """
    df = df.copy()
    if df is None or df.empty:
        return df

    if "Score" not in df.columns:
        df["Score"] = 0.0

    # Prefer decimal probability if available
    prob_col = "Prob_dec" if "Prob_dec" in df.columns else "Prob"

    # Ensure numeric types (handles '58.7%' -> 58.7, and then we map to 0-1 if
    # needed)
    def _to_num(s):
        if isinstance(s, pd.Series):
            s = s.astype(str).str.replace(
    '%', '', regex=False).str.replace(
        ',', '.', regex=False)
            return pd.to_numeric(s, errors="coerce")
        return pd.to_numeric(pd.Series([s]), errors="coerce")

    for c in [prob_col, "Score", "EV"]:
        if c in df.columns:
            df[c] = _to_num(df[c])

    # Map Prob to 0-1 if it looks like percent (e.g., 58.7)
    if prob_col in df.columns:
        p = df[prob_col].copy()
        # If median > 1.5 assume it's percent
        if pd.notna(p.median()) and p.median() > 1.5:
            p = p / 100.0
        df[prob_col] = p.clip(0.0, 1.0)

    if market_quality is None:
        market_quality = {}

    parts = []
    for market, g in df.groupby("Seçim", dropna=False):
        g = g.copy()

        # Safe z-score (ddof=0 keeps small groups stable)
        def _z(col):
            if col not in g.columns:
                return pd.Series(0.0, index=g.index)
            s = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(0.0, index=g.index)
            return (s - s.mean()) / sd

        z_prob = _z(prob_col)
        z_score = _z("Score")
        z_ev = _z("EV")

        # Market weights
        if market in ["MS 1", "MS X", "MS 2"]:
            w_prob, w_score, w_ev = 0.35, 0.25, 0.40
        else:
            w_prob, w_score, w_ev = 0.55, 0.25, 0.20

        base = (w_prob * z_prob + w_score * z_score + w_ev * z_ev)

        q = float(market_quality.get(market, 0.5))
        l_mult = pd.to_numeric(
    g.get(
        "League_Mult",
        1.0),
         errors="coerce").fillna(1.0)

        arch_mult = 1.0
        if use_archetype and "Archetype_Multiplier" in g.columns:
            arch_mult = pd.to_numeric(
    g["Archetype_Multiplier"],
     errors="coerce").fillna(1.0)

        raw = (0.6 + 0.8 * q) * l_mult * base * arch_mult
        g["GoldenScore_raw"] = pd.to_numeric(raw, errors="coerce").fillna(0.0)

        # Normalize to 0-1 for UI stability
        g["GoldenScore"] = (
            1.0 / (1.0 + np.exp(-g["GoldenScore_raw"].clip(-8, 8)))).clip(0.0, 1.0)

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


# 📌 4️⃣ Market Specific Min Prob/EV +
# -----------------------------------------------------------------------------
# (B2) Style Profile Pack (Banko / Zeki / Value / Güvenli / Sürpriz)
# -----------------------------------------------------------------------------
def _apply_style_profile_pack(
    df: pd.DataFrame,
    style_mode: str,
     scope: str = "bestof") -> pd.DataFrame:
    """Apply profile-specific hard gates + soft bonuses using available columns.

    - scope="bestof": stricter, list should be tighter and cleaner.
    - scope="pool": looser, radar view; still removes obvious noise for some modes.
    """
    if df is None:
        return pd.DataFrame()
    if getattr(df, "empty", True):
        return df

    _style = (style_mode or "").upper()

    out = df.copy()

    # Ensure numeric safety
    for col in [
    "MDL_quantile",
    "CONF_percentile",
    "AutoMod_percentile",
    "EV",
    "Prob",
    "SIM_QUALITY",
    "EFFECTIVE_N",
     "SIM_ANCHOR_STRENGTH"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Helper getters
    mdlq = out["MDL_quantile"] if "MDL_quantile" in out.columns else pd.Series(
        np.nan, index=out.index)
    confp = out["CONF_percentile"] if "CONF_percentile" in out.columns else pd.Series(
        np.nan, index=out.index)
    amp = out["AutoMod_percentile"] if "AutoMod_percentile" in out.columns else pd.Series(
        np.nan, index=out.index)
    ev = out["EV"] if "EV" in out.columns else pd.Series(
        np.nan, index=out.index)
    simq = out["SIM_QUALITY"] if "SIM_QUALITY" in out.columns else pd.Series(
        np.nan, index=out.index)
    effn = out["EFFECTIVE_N"] if "EFFECTIVE_N" in out.columns else (
        out["SIM_N"] if "SIM_N" in out.columns else pd.Series(
            np.nan, index=out.index)
    )

    # Icons (may not exist)
    trap = out["TRAP_ICON"].astype(
        str) if "TRAP_ICON" in out.columns else pd.Series("", index=out.index)
    danger = out["DANGER_ICON"].astype(
        str) if "DANGER_ICON" in out.columns else pd.Series("", index=out.index)

    def _is_bad_icon(s: pd.Series) -> pd.Series:
        # treat non-empty red/danger emojis/text as "bad" conservatively
        u = s.str.upper()
        return u.str.contains("🟥|RED|DANGER|RISK|X", regex=True, na=False)

    trap_bad = _is_bad_icon(trap)
    danger_bad = _is_bad_icon(danger)

    # Defaults per style (hard gates)
    if "BANKO" in _style:
        mdl_min = 0.75 if scope == "bestof" else 0.65
        conf_min = 50.0 if scope == "bestof" else 40.0
        am_min = 50.0 if scope == "bestof" else 40.0
        allow_trap = False
        allow_danger = False
        # bonus small, mostly tie-breaker
        bonus_scale = 0.03
    elif "GÜVEN" in _style or "GÜVEN" in _style:
        mdl_min = 0.70 if scope == "bestof" else 0.60
        conf_min = 40.0 if scope == "bestof" else 30.0
        am_min = 40.0 if scope == "bestof" else 30.0
        allow_trap = False
        allow_danger = False
        bonus_scale = 0.04
    elif "ZEK" in _style:
        mdl_min = 0.60 if scope == "bestof" else 0.55
        conf_min = 30.0 if scope == "bestof" else 20.0
        am_min = 30.0 if scope == "bestof" else 20.0
        allow_trap = True
        allow_danger = False
        bonus_scale = 0.06
    elif "VALUE" in _style or "VAL" in _style:
        # Value: don't over-gate by MDL; use it as noise guard / confirmation
        mdl_min = 0.55 if scope == "bestof" else 0.45
        conf_min = 20.0 if scope == "bestof" else 0.0
        am_min = 20.0 if scope == "bestof" else 0.0
        allow_trap = True
        allow_danger = False if scope == "bestof" else True
        bonus_scale = 0.08
    else:  # SÜRPRİZ / default
        mdl_min = 0.0
        conf_min = 0.0
        am_min = 0.0
        allow_trap = True
        allow_danger = True
        bonus_scale = 0.00

    # Apply hard gates (only if columns exist; otherwise do nothing)
    mask = pd.Series(True, index=out.index)

    if "MDL_quantile" in out.columns and mdl_min > 0:
        mask &= mdlq.fillna(-1.0) >= float(mdl_min)

    if "CONF_percentile" in out.columns and conf_min > 0:
        mask &= confp.fillna(-1.0) >= float(conf_min)

    if "AutoMod_percentile" in out.columns and am_min > 0:
        mask &= amp.fillna(-1.0) >= float(am_min)

    if (not allow_trap) and ("TRAP_ICON" in out.columns):
        mask &= ~trap_bad

    if (not allow_danger) and ("DANGER_ICON" in out.columns):
        mask &= ~danger_bad

    # Value-specific "fake value" guard: EV high but weak structure + weak MDL
    # -> drop
    if ("VALUE" in _style or "VAL" in _style) and ("EV" in out.columns):
        _fake = (ev.fillna(-999) >= 1.5)
        if "MDL_quantile" in out.columns:
            _fake &= (mdlq.fillna(-1.0) < 0.55)
        if "SIM_QUALITY" in out.columns:
            _fake &= (simq.fillna(-1.0) < 0.25)
        if effn is not None:
            _fake &= (pd.to_numeric(effn, errors="coerce").fillna(-1.0) < 10)
        mask &= ~_fake

    out = out[mask].copy()

    # Soft bonus: push higher-MDL items up without turning MDL into a gas pedal
    # (does not change gates; only affects tie-break sorting if used downstream)
    if bonus_scale > 0 and ("MDL_quantile" in out.columns):
        _bonus = (out["MDL_quantile"].fillna(0.0) * float(bonus_scale))
        if "GoldenScore_MDL" in out.columns:
            out["GoldenScore_MDL"] = pd.to_numeric(
    out["GoldenScore_MDL"], errors="coerce").fillna(0.0) + _bonus
        elif "GoldenScore" in out.columns:
            out["GoldenScore"] = pd.to_numeric(
    out["GoldenScore"], errors="coerce").fillna(0.0) + _bonus
        out["STYLE_BONUS"] = _bonus

    return out if isinstance(out, pd.DataFrame) else pd.DataFrame()


# (C) BestOfRank Logic
def build_best_of_list(
    pool_bets: pd.DataFrame,
    min_prob: float,
    min_score: float,
    min_ev: float,
    max_per_market: int = 6,
    max_total: int = 30,
    max_per_market_map: Optional[Dict[str, int]] = None,
    market_quality: Dict[str, float] = None,
    market_min_conf_ev: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """Generate Best Of list from pool.
    Core principle (Auto Mod): no hard-gating by sliders; instead rank by FinalScore and
    keep market representation close to pool distribution (with soft strength tilt).
    """
    if pool_bets is None or getattr(pool_bets, "empty", True):
        return pd.DataFrame()

    df = pool_bets.copy()

    # Basic hygiene guards (not manual thresholds; just prevent NaN/inf
    # explosions)
    for col in ["Prob", "EV", "Odd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
    subset=[
        c for c in [
            "Seçim",
            "Prob",
            "EV",
            "Odd"] if c in df.columns],
             how="any")
    if df.empty:
        return pd.DataFrame()

    # Ensure GoldenScore exists before ranking (best-effort, no crash)
#     try:  # AUTO-COMMENTED (illegal global try)
#         if "Archetype" not in df.columns:
#             df = apply_archetype_layer(df)
#     except Exception:
#         pass
#     try:
        if "GoldenScore" not in df.columns:
            df = compute_golden_score(df, market_quality, use_archetype=True)
#     except Exception:  # AUTO-COMMENTED (orphan except)
#         if "GoldenScore" not in df.columns:
#             df["GoldenScore"] = 0.0

    # ------------------------------------------------------------------
    # (MDL) Add profile + controlled GoldenScore boost (no market switching)
    # ------------------------------------------------------------------
#     try:  # AUTO-COMMENTED (illegal global try)
#         df = add_mdl_features(df)
#     except Exception:
#         pass

    # Ensure confidence columns exist
    if "League_Conf" not in df.columns:
        df["League_Conf"] = 0.5
    if "Market_Conf_Score" not in df.columns:
        df["Market_Conf_Score"] = 0.5

    # ------------------------------------------------------------------
    # (1) Compute BestOfRank (market-local z-score blend) if missing
    # ------------------------------------------------------------------
    if "BestOfRank" not in df.columns:
        df["BestOfRank"] = 0.0
        frames = []
        for market_name, sub in df.groupby("Seçim"):
            if sub.empty:
                continue
            if market_name in ["MS 1", "MS X", "MS 2"]:
                w = RANK_WEIGHTS_BY_MARKET["1X2"]
            elif market_name in ["2.5 Üst", "2.5 Alt"]:
                w = RANK_WEIGHTS_BY_MARKET["OU"]
            else:
                w = RANK_WEIGHTS_BY_MARKET["KG"]

            sub["_z_score"] = _safe_z(
    sub["Score"]) if "Score" in sub.columns else 0.0
            sub["_z_golden"] = _safe_z(
    sub["GoldenScore"]) if "GoldenScore" in sub.columns else 0.0
            sub["_z_prob"] = _safe_z(
    sub["Prob"]) if "Prob" in sub.columns else 0.0

            league_term = (sub["League_Conf"].fillna(
                0.5).astype(float) - 0.5) * 2.0
            market_term = (sub["Market_Conf_Score"].fillna(
                0.5).astype(float) - 0.5) * 2.0

            # Similarity anchor market term (prefer bets matching similarity's
            # strongest market for this match)
            sim_term = 0.0
            if "SIM_ANCHOR_GROUP" in sub.columns:
                def _sel_group(_s: str) -> str:
                    s = str(_s)
                    if s.startswith("MS"):
                        return "MS"
                    if "2.5" in s:
                        return "OU"
                    if s.startswith("KG"):
                        return "BTTS"
                    return ""
                sel_group = sub["Seçim"].map(_sel_group).astype(str)
                anchor_g = sub["SIM_ANCHOR_GROUP"].astype(str)
                strength = pd.to_numeric(
    sub.get(
        "SIM_ANCHOR_STRENGTH",
        0.0),
    errors="coerce").fillna(0.0).clip(
        0.0,
         1.0)
                sim_term = np.where(
    sel_group == anchor_g, 1.0, -1.0) * strength

            sub["BestOfRank"] = (
                w["score"] * sub["_z_score"] +
                w["golden"] * sub["_z_golden"] +
                w["prob"] * sub["_z_prob"] +
                w["league"] * league_term +
                w["market"] * market_term +
                w.get("sim_market", 0.0) * sim_term
            )
            frames.append(sub)
        df = pd.concat(frames, ignore_index=True) if frames else df
        df.drop(
    columns=[
        "_z_score",
        "_z_golden",
        "_z_prob"],
        errors="ignore",
         inplace=True)

        # ------------------------------------------------------------------
    # (1.5) PHASE-3: Composite RankScore (new regime)
    #   Goal: make ranking align with Similarity Anchor + Confidence + AutoMod + EV.
    #   - Keeps legacy BestOfRank for reference (BestOfRank_legacy)
    #   - Builds RankScore (0..1-ish) and uses it as BestOfRank going forward
    # ------------------------------------------------------------------
#     try:  # AUTO-COMMENTED (illegal global try)
#         # Preserve legacy BestOfRank (z-score blend) for debugging
#         df["BestOfRank_legacy"] = pd.to_numeric(
#             df.get("BestOfRank", 0.0), errors="coerce").fillna(0.0)

        # Core signals (0..1)
        _a = pd.to_numeric(
    df.get(
        "SIM_ANCHOR_STRENGTH",
        np.nan),
         errors="coerce")
        _a = _a.fillna(0.50).clip(0.0, 1.0)

        _conf = pd.to_numeric(
    df.get(
        "CONF_percentile",
        df.get(
            "CONF_PCT",
            np.nan)),
             errors="coerce")
        _conf = _conf.fillna(0.50).clip(0.0, 1.0)

        _amqs = pd.to_numeric(
    df.get(
        "AMQS_percentile",
        df.get(
            "AutoMod_percentile",
            np.nan)),
             errors="coerce")
        _amqs = _amqs.fillna(0.50).clip(0.0, 1.0)

        _mconf = pd.to_numeric(
    df.get(
        "Market_Conf_Score",
        np.nan),
        errors="coerce").fillna(0.50).clip(
            0.0,
             1.0)
        _lconf = pd.to_numeric(
    df.get(
        "League_Conf",
        np.nan),
        errors="coerce").fillna(0.50).clip(
            0.0,
             1.0)

        # EV component: market-local z-score -> squash to 0..1
        if "EV" in df.columns:
            df["_ev_z"] = df.groupby("Seçim")["EV"].transform(
                lambda s: _safe_z(pd.to_numeric(s, errors="coerce").fillna(0.0)))
            _ev01 = (1.0 / (1.0 + np.exp(-pd.to_numeric(
                df["_ev_z"], errors="coerce").fillna(0.0).clip(-6, 6)))).clip(0.0, 1.0)
        else:
            _ev01 = pd.Series(
    [0.50] * len(df),
    index=df.index,
     dtype="float64")

        def _center01(x):
            return (x.astype(float) - 0.5) * 2.0

        a_c = _center01(_a)
        conf_c = _center01(_conf)
        amqs_c = _center01(_amqs)
        mc_c = _center01(_mconf)
        lc_c = _center01(_lconf)
        ev_c = _center01(_ev01)

        # Weights: anchor dominates; confidence & automod next; EV and confs as
        # support
        wA, wC, wM, wMC, wLC, wEV = 0.45, 0.20, 0.15, 0.08, 0.07, 0.05
        rank_centered = (
    wA *
    a_c +
    wC *
    conf_c +
    wM *
    amqs_c +
    wMC *
    mc_c +
    wLC *
    lc_c +
    wEV *
     ev_c)

        df["RankScore"] = (0.50 + 0.50 * rank_centered).clip(0.0, 1.0)

        # Use RankScore as the primary rank for selection
        df["BestOfRank"] = df["RankScore"].astype(float)

    df.drop(columns=["_ev_z"], errors="ignore", inplace=True)

# ------------------------------------------------------------------
    # (2) Auto Mod soft strengths (no hard gate)
    # ------------------------------------------------------------------
    # Map confidence to multipliers (0.7..1.3) and (0.8..1.2)
    league_mult = 0.7 + 0.6 * \
        df["League_Conf"].fillna(0.5).clip(0, 1).astype(float)
    market_mult = 0.8 + 0.4 * \
        df["Market_Conf_Score"].fillna(0.5).clip(0, 1).astype(float)

    # Mode strength (if available); default 1.0
    mode_mult = 1.0
    current_mode = "FULL" if globals().get("FULL_MODE", False) else "FAST"
    # NOTE: MODE_MARKET_STRENGTH is expected to be a dict/map (per-market multipliers)
    mode_strength_map = globals().get("MODE_MARKET_STRENGTH", {}).get(current_mode, {}) or {}
    # FIX: encoding/merge corruption can break "Seçim" column name
    if "Seçim" in df.columns:
        sel_col = "Seçim"
    elif "Secim" in df.columns:
        sel_col = "Secim"
    else:
        sel_col = None
#     try:  # AUTO-COMMENTED (illegal global try)
#         current_mode = "FULL" if globals().get("FULL_MODE", False) else "FAST"
#         mode_strength = globals().get(
#     "MODE_MARKET_STRENGTH", {}).get(
#         current_mode, {})
    if isinstance(mode_strength_map, dict) and sel_col in df.columns:
        mode_mult = df[sel_col].map(
lambda m: float(
    mode_strength_map.get(
        m, 1.0))).astype(float)
    else:
        mode_mult = 1.0

    # FinalScore: core rank * suitability
    # --- AMQS multiplier (robust: df.get default must be a Series, not scalar) ---
    if "AMQS_percentile" in df.columns:
        _amqs_pct = pd.to_numeric(df["AMQS_percentile"], errors="coerce")
    else:
        _amqs_pct = pd.Series(
    [0.50] * len(df),
    index=df.index,
     dtype="float64")
    _amqs_pct = _amqs_pct.fillna(0.50).clip(0.0, 1.0).astype(float)
    amqs_mult = 0.85 + 0.30 * _amqs_pct
    df["Suitability"] = league_mult * market_mult * mode_mult * amqs_mult
    df["FinalScore"] = df["BestOfRank"].astype(
        float) * df["Suitability"].astype(float)

    # ------------------------------------------------------------------
    # (3) Build quotas: pool distribution + soft strength tilt
    # ------------------------------------------------------------------
    if max_per_market_map is None:
        # hard caps; fallback to max_per_market
        max_per_market_map = MARKET_LIMITS_HARD.copy()

    # Slider-driven knobs (if present) - safe defaults
    balance_pct = 80
    quality_pct = 70
#     try:  # AUTO-COMMENTED (illegal global try)
#         balance_pct = int(st.session_state.get("AUTO_DENGE", 80))
#         quality_pct = int(st.session_state.get("AUTO_KALITE", 70))
#     except Exception:
#         pass
#     balance = np.clip(balance_pct / 100.0, 0.0, 1.0)  # 1 => follow pool
    # 1 => trust BestOfRank more (less tilt)
#     quality = np.clip(quality_pct / 100.0, 0.0, 1.0)
    balance = np.clip(balance_pct / 100.0, 0.0, 1.0)  # 1 => follow pool
    quality = np.clip(quality_pct / 100.0, 0.0, 1.0)

    counts = df[sel_col].value_counts(dropna=True) if sel_col else pd.Series(dtype=int)
    total = float(counts.sum()) if counts is not None else 0.0
    if total <= 0:
        return pd.DataFrame()

    pool_share = (counts / total).to_dict()

    # strength by market: use market_quality dict if provided else average
    # Suitability per market
    strength = {}
    if isinstance(market_quality, dict) and market_quality:
        for mkt in counts.index:
            strength[mkt] = float(market_quality.get(mkt, 0.5))
    else:
        strength = df.groupby("Seçim")["Suitability"].mean().to_dict()
        # normalize to 0..1
        if strength:
            mn = min(strength.values())
            mx = max(strength.values())
            denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
            for k in list(strength.keys()):
                strength[k] = (strength[k] - mn) / denom

    # Convert strength to a share distribution
    strength_sum = sum(max(0.0, v) for v in strength.values()) or 1.0
    strength_share = {
    k: max(
        0.0,
        float(v)) /
        strength_sum for k,
         v in strength.items()}

    # Mix: follow pool vs follow strength (strength influence increases as
    # balance decreases)
    mixed_share = {}
    for mkt in counts.index:
        ps = float(pool_share.get(mkt, 0.0))
        ss = float(strength_share.get(mkt, 0.0))
        # quality dampens strength influence (quality=1 => mostly pool;
        # quality=0 => allow more strength tilt)
        tilt = (1.0 - quality)
        mixed_share[mkt] = balance * ps + \
            (1.0 - balance) * ((1.0 - tilt) * ps + tilt * ss)

    # Normalize mixed shares
    ssum = sum(mixed_share.values()) or 1.0
    for k in mixed_share:
        mixed_share[k] /= ssum

    # Allocate quotas via largest remainder with caps
    quotas = {}
    rema = {}
    allocated = 0
    for mkt, share in mixed_share.items():
        ideal = share * int(max_total)
        q = int(math.floor(ideal))
        cap = int(max_per_market_map.get(mkt, max_per_market))
        q = max(0, min(q, cap))
        quotas[mkt] = q
        rema[mkt] = ideal - q
        allocated += q

    remaining = int(max_total) - allocated
    if remaining > 0:
        for mkt, _ in sorted(rema.items(), key=lambda kv: kv[1], reverse=True):
            if remaining <= 0:
                break
            cap = int(max_per_market_map.get(mkt, max_per_market))
            if quotas.get(mkt, 0) < cap:
                quotas[mkt] += 1
                remaining -= 1

    # Ensure at least 1 from markets that exist in pool when max_total is
    # large enough
    if int(max_total) >= 8:
        for mkt in counts.index:
            if counts.get(mkt, 0) > 0 and quotas.get(mkt, 0) == 0:
                # borrow a slot from the currently largest quota market
                donor = max(
    quotas.keys(), key=lambda k: quotas.get(
        k, 0), default=None)
                if donor and quotas.get(donor, 0) > 1:
                    quotas[donor] -= 1
                    quotas[mkt] = 1

    # ------------------------------------------------------------------
    # (4) Select by market top FinalScore, then match-dedup, then fill to max_total
    # ------------------------------------------------------------------
    selected = []
    for mkt, sub in df.groupby("Seçim"):
        take_n = int(quotas.get(mkt, 0))
        if take_n <= 0:
            continue
        sort_cols = ["FinalScore", "GoldenScore_MDL", "GoldenScore", "Score"]
        sort_cols = [c for c in sort_cols if c in sub.columns]

        # Deterministic sorting (stable) + meaningful tie-breakers (no random
        # final pick)
        sub = _ensure_tie_key(sub)
        primary_score = sort_cols[0] if sort_cols else None

        # Prefer higher similarity quality when scores are near-tied
        if "SIM_QUALITY" in sub.columns and "SIM_QUALITY" not in sort_cols:
            sort_cols2 = sort_cols + ["SIM_QUALITY"]
            asc2 = [False] * len(sort_cols) + [False]
        else:
            sort_cols2 = sort_cols[:]
            asc2 = [False] * len(sort_cols2)

        # Penalize overly-wide effective neighborhoods (prefer mid-band)
        if "EFFECTIVE_N" in sub.columns:
            sub["_EN_PEN"] = (
    pd.to_numeric(
        sub["EFFECTIVE_N"],
        errors="coerce") -
         40.0).abs()
            sort_cols2 += ["_EN_PEN"]
            asc2 += [True]

        # Optional confidence / rating tie-breakers
        for c in ["CONF_percentile", "AMQS_percentile", "Star_Rating"]:
            if c in sub.columns:
                sort_cols2.append(c)
                asc2.append(False)

        # Odds & time as very-late tie-breakers
        for c, a in [("Odd", True), ("Date", True),
                      ("HomeTeam", True), ("AwayTeam", True)]:
            if c in sub.columns:
                sort_cols2.append(c)
                asc2.append(a)

        sort_cols2.append("_TIE")
        asc2.append(True)

        sub2 = sub.sort_values(by=sort_cols2, ascending=asc2, kind="mergesort")

        # Top-N + ties: do not force a single random pick when scores are
        # indistinguishable
        eps = float(st.session_state.get("GR_TIE_EPS", 1e-4))
        if primary_score:
            selected.append(
    _top_n_with_ties(
        sub2,
        primary_score,
        take_n,
         eps=eps))
        else:
            selected.append(sub2.head(take_n))

    best_of = pd.concat(
    selected,
     ignore_index=True) if selected else pd.DataFrame()

    # Match-level dedup: keep best signal per match
    key_cols = [
    c for c in [
        "Date",
        "HomeTeam",
         "AwayTeam"] if c in best_of.columns]
    if len(key_cols) == 3 and not best_of.empty:
        # Match-level dedup: keep best signal per match (prefer higher
        # CONF_percentile if available)
        _dedup_cols = [
    c for c in [
        "CONF_percentile",
        "FinalScore",
        "GoldenScore",
         "Score"] if c in best_of.columns]
        if not _dedup_cols:
            _dedup_cols = [
                "FinalScore"] if "FinalScore" in best_of.columns else key_cols
        best_of = (
            best_of.sort_values(
    by=_dedup_cols,
    ascending=[False] *
     len(_dedup_cols))
            .groupby(key_cols, as_index=False)
            .first()
        )

    # If still short, fill from global top not already included
    if best_of is None or best_of.empty:
        # hard fallback: return top FinalScore overall
        best_of = df.sort_values(
    by=["FinalScore"],
    ascending=[False]).head(
        int(max_total)).copy()
    else:
        remaining = int(max_total) - len(best_of)
        if remaining > 0:
            if len(key_cols) == 3:
                existing_keys = set(
    tuple(x) for x in best_of[key_cols].itertuples(
        index=False, name=None))
                df2 = df.copy()
                df2["_k"] = list(
                    zip(df2[key_cols[0]], df2[key_cols[1]], df2[key_cols[2]]))
                df2 = df2[~df2["_k"].isin(existing_keys)].copy()
                df2.drop(columns=["_k"], errors="ignore", inplace=True)
            else:
                df2 = df.copy()
            _fill_cols = [
    c for c in [
        "CONF_percentile",
        "FinalScore",
        "GoldenScore",
         "Score"] if c in df2.columns]
            if not _fill_cols:
                _fill_cols = [
                    "FinalScore"] if "FinalScore" in df2.columns else key_cols
            filler = df2.sort_values(
    by=_fill_cols,
    ascending=[False] *
     len(_fill_cols)).head(remaining)
            best_of = pd.concat([best_of, filler], ignore_index=True)

    # Final sort and trim
    _final_sort_cols = [
    c for c in [
        "CONF_percentile",
        "FinalScore",
        "GoldenScore_MDL",
        "GoldenScore",
         "Score"] if c in best_of.columns]
    if not _final_sort_cols:
        _final_sort_cols = [
    "FinalScore",
    "GoldenScore_MDL",
    "GoldenScore",
     "Score"]
    best_of = best_of.sort_values(
    by=_final_sort_cols,
    ascending=[False] *
     len(_final_sort_cols))
    best_of = best_of.head(int(max_total)).reset_index(drop=True)

    return best_of


def plot_profit_curve(df, prob_col, odd_col, target_col):
    if df.empty or prob_col not in df.columns:
        return go.Figure()
    thresholds = np.linspace(0.40, 0.85, 15)
    profits, counts = [], []
    for t in thresholds:
        subset = df[df[prob_col] >= t]
        if len(subset) == 0:
            profits.append(0); counts.append(0); continue
        profit = np.sum(
    np.where(
        subset[target_col] == 1, subset[odd_col] - 1, -1))
        profits.append(profit)
        counts.append(len(subset))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
    go.Scatter(
        x=thresholds,
        y=profits,
        mode='lines+markers',
        name='Toplam Kâr (Unit)',
        line=dict(
            color='#00E676',
            width=3)),
             secondary_y=False)
    fig.add_trace(
    go.Bar(
        x=thresholds,
        y=counts,
        name='Bahis Sayısı',
        marker=dict(
            color='rgba(255, 255, 255, 0.2)')),
             secondary_y=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
    fig.update_layout(
    title="Kâr Eğrisi (Profit Curve)",
    xaxis_title="Güven Eşiği",
    yaxis_title="Kâr",
    template="plotly_dark",
    height=400,
    legend=dict(
        orientation="h",
         y=1.1))
    return fig


def plot_calibration_curve_plotly(y_true, y_prob, title="Calibration Plot"):
    if len(y_true) == 0:
        return go.Figure()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig = go.Figure()
    fig.add_trace(
    go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode='markers+lines',
        name='Model',
        line=dict(
            color='#4CAF50')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[
                  0, 1], mode='lines', name='Mükemmel', line=dict(dash='dash', color='white')))
    fig.update_layout(
    title=title,
    xaxis_title="Tahmin Edilen Olasılık",
    yaxis_title="Gerçekleşen Oran",
    template="plotly_dark",
     height=400)
    return fig


def plot_gain_chart(y_true, y_prob, title="Cumulative Gain Chart"):
    if len(y_true) == 0:
        return go.Figure()
    df_gain = pd.DataFrame({'y': y_true, 'p': y_prob}
                           ).sort_values('p', ascending=False)
    total_pos = df_gain['y'].sum()
    if total_pos == 0:
        return go.Figure()
    df_gain['cum_pos'] = df_gain['y'].cumsum()
    df_gain['cum_pos_rate'] = df_gain['cum_pos'] / total_pos
    df_gain['population_rate'] = np.arange(1, len(df_gain) + 1) / len(df_gain)
    baseline_x = [0, 1]
    baseline_y = [0, 1]
    fig = go.Figure()
    fig.add_trace(
    go.Scatter(
        x=df_gain['population_rate'],
        y=df_gain['cum_pos_rate'],
        mode='lines',
        name='Model',
        line=dict(
            color='#00E676')))
    fig.add_trace(
    go.Scatter(
        x=baseline_x,
        y=baseline_y,
        mode='lines',
        name='Random',
        line=dict(
            dash='dash',
             color='white')))
    fig.update_layout(
    title=title,
    xaxis_title="% of Data Tested",
    yaxis_title="% of Positives Captured",
    template="plotly_dark",
     height=400)
    return fig


def plot_ev_analysis(
    df,
    prob_col,
    odd_col,
    target_col,
     title="EV Calibration"):
    if df.empty or prob_col not in df.columns:
        return go.Figure()
    df_calc = df.copy()
    df_calc['EV_Calc'] = (df_calc[prob_col] * df_calc[odd_col] - 1) * 100
    bins = np.linspace(-10, 40, 6)
    df_calc['EV_Bin'] = pd.cut(df_calc['EV_Calc'], bins=bins).astype(str)
    df_calc['Profit'] = np.where(
    df_calc[target_col] == 1, df_calc[odd_col] - 1, -1)
    grouped = df_calc.groupby(
    'EV_Bin', observed=False).agg(
        ROI=(
            'Profit', 'mean'), Count=(
                'Profit', 'count'), Avg_EV=(
                    'EV_Calc', 'mean')).reset_index()
    grouped['ROI_Percent'] = grouped['ROI'] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ['#ef5350' if x < 0 else '#66bb6a' for x in grouped['ROI_Percent']]
    fig.add_trace(
    go.Bar(
        x=grouped['EV_Bin'],
        y=grouped['ROI_Percent'],
        marker_color=colors,
        name='Gerçekleşen ROI'),
         secondary_y=False)
    fig.add_trace(
    go.Bar(
        x=grouped['EV_Bin'],
        y=grouped['Count'],
        name='Hacim',
        marker=dict(
            color='rgba(255, 255, 255, 0.1)'),
            opacity=0.3),
             secondary_y=True)
    fig.update_layout(
    title=title,
    xaxis_title="Beklenen Değer (EV) Aralığı",
    yaxis_title="Gerçek ROI (%)",
    template="plotly_dark",
     height=450)
    return fig


def normalize_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    xgf_col = None
    xga_col = None
    if 'FT_xG_Home' in df.columns:
        xgf_col = 'FT_xG_Home'
    elif 'xG_Home_Actual' in df.columns: xgf_col = 'xG_Home_Actual'
    if 'FT_xG_Away' in df.columns:
        xga_col = 'FT_xG_Away'
    elif 'xG_Away_Actual' in df.columns: xga_col = 'xG_Away_Actual'
    gf_col = 'FT_Score_Home' if 'FT_Score_Home' in df.columns else None
    ga_col = 'FT_Score_Away' if 'FT_Score_Away' in df.columns else None

    def find_col(df, keywords):
        kws = [k.lower() for k in keywords]
        for c in df.columns:
            if all(k in c.lower() for k in kws):
                return c
        return None

    if xgf_col is None:
        xgf_col = (
    find_col(
        df, [
            'xg', 'result', 'home']) or find_col(
                df, [
                    'ft', 'xg', 'home']) or find_col(
                        df, [
                            'xg', 'h']))
    if xga_col is None: xga_col = (find_col(df, ['xg', 'result', 'away']) or find_col(df, ['ft', 'xg', 'away']) or find_col(df, ['xg', 'a']))
    if gf_col is None:
        gf_col = (
    find_col(
        df, [
            'ft', 'score', 'home']) or find_col(
                df, [
                    'goals', 'home']) or find_col(
                        df, ['fthg']))
    if ga_col is None: ga_col = (find_col(df, ['ft', 'score', 'away']) or find_col(df, ['goals', 'away']) or find_col(df, ['ftag']))

    rename_map = {}
    if xgf_col and 'xGF' not in df.columns:
        rename_map[xgf_col] = 'xGF'
    if xga_col and 'xGA' not in df.columns: rename_map[xga_col] = 'xGA'
    if gf_col and 'GF' not in df.columns:
        rename_map[gf_col] = 'GF'
    if ga_col and 'GA' not in df.columns: rename_map[ga_col] = 'GA'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def train_base_models(X_train, y_train, sample_weight=None):
    models = {}
    hgb = HistGradientBoostingClassifier(
    max_depth=6, learning_rate=0.12, random_state=42)
    hgb.fit(X_train, y_train, sample_weight=sample_weight)
    models['hgb'] = hgb
    lr = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight='balanced',
     random_state=42)
    lr.fit(X_train, y_train, sample_weight=sample_weight)
    models['lr'] = lr
    rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=10,
     random_state=42)
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    models['rf'] = rf
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_train, y_train)
    models['knn'] = knn
    return models


def get_stacking_features(models, X):
    X = X.fillna(0)
    return pd.DataFrame({
        "p_hgb": models['hgb'].predict_proba(X)[:, 1],
        "p_lr": models['lr'].predict_proba(X)[:, 1],
        "p_rf": models['rf'].predict_proba(X)[:, 1],
        "p_knn": models['knn'].predict_proba(X)[:, 1],
    })


def train_stacking_meta(X_stack, y, sample_weight=None):
    meta = LogisticRegression(max_iter=3000, random_state=42)
    meta.fit(X_stack, y, sample_weight=sample_weight)
    return meta

# -----------------------------------------------------------------------------#
# 4. DATA PREP & FEATURE ENGINEERING
# -----------------------------------------------------------------------------#


@st.cache_data(show_spinner=False)
def prepare_base_data(
    past_df: pd.DataFrame,
     future_df: pd.DataFrame) -> pd.DataFrame:
    past_df = past_df.copy()
    future_df = future_df.copy()
    past_df["Dataset"] = "past"
    future_df["Dataset"] = "future"
    past_df.columns = past_df.columns.str.strip()
    future_df.columns = future_df.columns.str.strip()

    for df in [past_df, future_df]:
        if 'HomeTeam' in df.columns:
            df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip()
        if 'AwayTeam' in df.columns: df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip()

    from_map = {
        "Date": "Date", "Season": "Season", "Home_Team": "HomeTeam", "Away_Team": "AwayTeam", "League": "League",
        "League_Pos_Home": "HomeLeaguePosition", "League_Pos_Away": "AwayLeaguePosition",
        "Home Overall Elo Rank": "Elo_Home", "Away Overall Elo Rank": "Elo_Away",
        "FT_Score_Home": "HomeFullTimeScore", "FT_Score_Away": "AwayFullTimeScore",
        "xG_Home": "xGHome", "xG_Away": "xGAway",
        "Odds_Close_Home": "ClosingHomeOdd", "Odds_Close_Draw": "ClosingDrawOdd", "Odds_Close_Away": "ClosingAwayOdd",
        "Odds_Close_Over25": "ClosingO25", "Odds_Close_Under25": "ClosingU25", "Odds_Close_BTTS_Yes": "ClosingBTTSY", "Odds_Close_BTTS_No": "ClosingBTTSN",
        "close_home_odds": "ClosingHomeOdd", "close_draw_odds": "ClosingDrawOdd", "close_away_odds": "ClosingAwayOdd",
        "close_o25_odds": "ClosingO25", "close_u25_odds": "ClosingU25", "close_bttsy_odds": "ClosingBTTSY", "close_bttsn_odds": "ClosingBTTSN",
        "open_home_odds": "Odds_Open_Home", "open_draw_odds": "Odds_Open_Draw", "open_away_odds": "Odds_Open_Away",
        "open_o25_odds": "Odds_Open_Over25", "open_u25_odds": "Odds_Open_Under25", "open_bttsy_odds": "Odds_Open_BTTS_Yes", "open_bttsn_odds": "Odds_Open_BTTS_No",
    }

    frames = []
    for df in (past_df, future_df):
        df = df.rename(columns=from_map)
        df = df.loc[:, ~df.columns.duplicated()]

        odd_cols_map = [
    ("HomeOdd",
    "ClosingHomeOdd",
    "Odds_Open_Home"),
    ("DrawOdd",
    "ClosingDrawOdd",
    "Odds_Open_Draw"),
    ("AwayOdd",
    "ClosingAwayOdd",
    "Odds_Open_Away"),
    ("O25",
    "ClosingO25",
    "Odds_Open_Over25"),
    ("U25",
    "ClosingU25",
    "Odds_Open_Under25"),
    ("BTTSY",
    "ClosingBTTSY",
    "Odds_Open_BTTS_Yes"),
    ("BTTSN",
    "ClosingBTTSN",
    "Odds_Open_BTTS_No"),
     ]
        # Always build robust 'main' odd columns (HomeOdd/DrawOdd/...) as: prefer closing -> fallback to open.
        # Even if the target column exists (but is empty/None), we still refill
        # it.
        for target, close_col, open_col in odd_cols_map:
            s_close = df[close_col] if close_col in df.columns else pd.Series(
                np.nan, index=df.index)
            s_open = df[open_col] if open_col in df.columns else pd.Series(
                np.nan, index=df.index)
            if target in ("HomeOdd", "DrawOdd", "AwayOdd"):
                df[target] = pd.to_numeric(
                    s_close,
                    errors='coerce').fillna(
                    pd.to_numeric(
                        s_open,
                        errors='coerce'))
            else:
                s_main = df[target] if target in df.columns else pd.Series(
                    np.nan, index=df.index)
                df[target] = pd.to_numeric(
                    s_main,
                    errors='coerce').fillna(
                    pd.to_numeric(
                        s_close,
                        errors='coerce')).fillna(
                        pd.to_numeric(
                            s_open,
                            errors='coerce'))

        num_cols = ['HomeFullTimeScore', 'AwayFullTimeScore', 'xGHome', 'xGAway', 'HomeOdd', 'DrawOdd', 'AwayOdd', 'O25', 'U25', 'BTTSY', 'BTTSN',
                    'Odds_Open_Home', 'Odds_Open_Draw', 'Odds_Open_Away', 'Odds_Open_Over25', 'Odds_Open_Under25', 'Odds_Open_BTTS_Yes', 'Odds_Open_BTTS_No',
                     'ClosingHomeOdd', 'ClosingDrawOdd', 'ClosingAwayOdd', 'ClosingO25', 'ClosingU25', 'ClosingBTTSY', 'ClosingBTTSN']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True).sort_values('Date')
    all_data = normalize_match_columns(all_data)
    all_data = sanitize_odds(all_data)

    all_data['Season'] = all_data['Season'].astype(str)
    all_data['Is_Euro'] = all_data['Season'].str.contains(
    'Europe -',
    na=False) | all_data['League'].str.contains(
        'Europe',
        case=False,
         na=False)

    all_data['Euro_Tier'] = 0.0
    mask_ucl = all_data['League'].str.contains(
        'Champions League', case=False, na=False)
    mask_uel = all_data['League'].str.contains(
        'Europa League', case=False, na=False)
    mask_conf = all_data['League'].str.contains(
        'Conference League', case=False, na=False)

    all_data.loc[mask_ucl, 'Euro_Tier'] = 2.0
    all_data.loc[mask_uel, 'Euro_Tier'] = 1.5
    all_data.loc[mask_conf, 'Euro_Tier'] = 1.2

    domestic_mask = ~all_data['Is_Euro']
    home_dom_league = all_data[domestic_mask].groupby('HomeTeam')['League'].agg(
        lambda s: s.value_counts().index[0] if not s.empty else np.nan)
    away_dom_league = all_data[domestic_mask].groupby('AwayTeam')['League'].agg(
        lambda s: s.value_counts().index[0] if not s.empty else np.nan)
    domestic_league_map = home_dom_league.combine_first(away_dom_league)
    all_data['Home_DomesticLeague'] = all_data['HomeTeam'].map(
        domestic_league_map).fillna('Unknown')
    all_data['Away_DomesticLeague'] = all_data['AwayTeam'].map(
        domestic_league_map).fillna('Unknown')

    return all_data


@st.cache_data(show_spinner=False)
def engineer_features(all_data: pd.DataFrame) -> pd.DataFrame:
    all_data = all_data.copy()
    n0 = len(all_data)

    home_elos = np.zeros(n0)
    away_elos = np.zeros(n0)
    team_elos = {}
    base_elo = 1500.0

    for idx, row in enumerate(all_data.itertuples(index=False)):
        h = getattr(row, 'HomeTeam')
        a = getattr(row, 'AwayTeam')
        eh = team_elos.get(h, base_elo)
        ea = team_elos.get(a, base_elo)
        home_elos[idx] = eh
        away_elos[idx] = ea

        hs = getattr(row, 'HomeFullTimeScore', np.nan)
        as_ = getattr(row, 'AwayFullTimeScore', np.nan)
        if not np.isnan(hs) and not np.isnan(as_):
            S_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            E_h = 1 / (1 + 10 ** ((ea - eh) / 400))
            team_elos[h] = eh + 20 * (S_h - E_h)
            team_elos[a] = ea + 20 * ((1 - S_h) - (1 - E_h))

    all_data['HomeELO_Final'] = home_elos
    all_data['AwayELO_Final'] = away_elos
    all_data['Elo_Diff'] = (all_data['HomeELO_Final'] -
                            all_data['AwayELO_Final']) / 400.0
    all_data['Elo_Gap_Abs'] = all_data['Elo_Diff'].abs()

    all_data['Mean_Elo'] = (all_data['HomeELO_Final'] +
                            all_data['AwayELO_Final']) / 2
    league_strength = all_data.groupby('League')['Mean_Elo'].transform('mean')
    mu = league_strength.mean()
    sigma = league_strength.std() if league_strength.std() != 0 else 1.0
    all_data['League_Strength_Index'] = (league_strength - mu) / sigma

    all_data['Tiny_Elo_Gap_Flag'] = (
    all_data['Elo_Diff'].abs() < 0.05).astype(int)

    league_str_map = all_data.groupby('League')['League_Strength_Index'].mean()
    all_data['Home_DomesticLeague_Strength'] = all_data['Home_DomesticLeague'].map(
        league_str_map).fillna(all_data['League_Strength_Index'])
    all_data['Away_DomesticLeague_Strength'] = all_data['Away_DomesticLeague'].map(
        league_str_map).fillna(all_data['League_Strength_Index'])

    all_data['SupremacyShift'] = (
    all_data['HomeELO_Final'] - all_data['AwayELO_Final']) / (
        all_data['HomeELO_Final'] + all_data['AwayELO_Final'] + 1e-6)

    all_data['match_day'] = all_data.groupby(['League', 'Season'])[
                                             'Date'].rank("dense")
    total_rounds = all_data.groupby(['League', 'Season'])[
                                    'match_day'].transform("max")
    all_data['SeasonProgress'] = all_data['match_day'] / (total_rounds + 1e-6)

    cols = [
    'xGHome',
    'xGAway',
    'HomeFullTimeScore',
    'AwayFullTimeScore',
    'Is_Euro',
    'League',
     'Season']
    sel_cols = [c for c in cols if c in all_data.columns]
    temp = all_data[['Date', 'HomeTeam', 'AwayTeam'] + sel_cols].copy()

    h_df = temp.rename(
    columns={
        'HomeTeam': 'Team',
        'xGHome': 'xGF',
        'xGAway': 'xGA',
        'HomeFullTimeScore': 'GF',
         'AwayFullTimeScore': 'GA'})
    a_df = temp.rename(
    columns={
        'AwayTeam': 'Team',
        'xGAway': 'xGF',
        'xGHome': 'xGA',
        'AwayFullTimeScore': 'GF',
         'HomeFullTimeScore': 'GA'})

    team_df = pd.concat([h_df[['Date', 'Team', 'xGF', 'xGA', 'GF', 'GA', 'Is_Euro', 'League', 'Season']],
                         a_df[['Date', 'Team', 'xGF', 'xGA', 'GF', 'GA', 'Is_Euro', 'League', 'Season']]],
                         ignore_index=True).sort_values(['Team', 'Date'])

    team_df['KG_Flag'] = (
    (team_df['GF'] > 0) & (
        team_df['GA'] > 0)).astype(int)
    team_df['O25_Flag'] = ((team_df['GF'] + team_df['GA']) > 2.5).astype(int)

    grp = team_df.groupby('Team', group_keys=False)

    def exp_weighted_rolling(series, window=5):
        return series.shift(1).ewm(
    span=window,
    min_periods=1,
     adjust=False).mean()

    team_df['Roll_xG_5'] = grp['xGF'].apply(
        lambda s: exp_weighted_rolling(s, 5))
    team_df['Roll_xGA_5'] = grp['xGA'].apply(
        lambda s: exp_weighted_rolling(s, 5))
    team_df['Roll_GF_5'] = grp['GF'].apply(
        lambda s: exp_weighted_rolling(s, 5))
    team_df['Roll_GA_5'] = grp['GA'].apply(
        lambda s: exp_weighted_rolling(s, 5))

    team_df['xG_For_5'] = grp['xGF'].apply(
    lambda s: s.shift(1).rolling(
        5, min_periods=1).mean())
    team_df['xG_Against_5'] = grp['xGA'].apply(
    lambda s: s.shift(1).rolling(
        5, min_periods=1).mean())
    team_df['Goals_For_5'] = grp['GF'].apply(
    lambda s: s.shift(1).rolling(
        5, min_periods=1).mean())
    team_df['Goals_Against_5'] = grp['GA'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_df['KG_Rate_5'] = grp['KG_Flag'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    team_df['O25_Rate_5'] = grp['O25_Flag'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean())

    team_df['Roll_xGA_Std'] = grp['xGA'].apply(
    lambda x: x.shift(1).rolling(
        150, min_periods=10).std()).fillna(0.5)
    team_df['GF_Std_10'] = grp['GF'].apply(
    lambda x: x.shift(1).rolling(
        10, min_periods=3).std())
    team_df['GA_Std_10'] = grp['GA'].apply(
    lambda x: x.shift(1).rolling(
        10, min_periods=3).std())
    team_df['Goals_Total_Std_10'] = grp[['GF', 'GA']].apply(lambda df_: (
        df_['GF'] + df_['GA']).shift(1).rolling(10, min_periods=3).std())

    dom_df = team_df[~team_df['Is_Euro']].copy()
    grp_dom = dom_df.groupby('Team', group_keys=False)
    dom_df['Dom_xG_For_5'] = grp_dom['xGF'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    dom_df['Dom_xG_Against_5'] = grp_dom['xGA'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    dom_df['Dom_Goals_For_5'] = grp_dom['GF'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    dom_df['Dom_Goals_Against_5'] = grp_dom['GA'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    dom_df['Dom_O25_Rate_5'] = grp_dom['O25_Flag'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    dom_df['Dom_KG_Rate_5'] = grp_dom['KG_Flag'].apply(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    home_dom = dom_df[['Team',
    'Date',
    'Dom_xG_For_5',
    'Dom_xG_Against_5',
    'Dom_Goals_For_5',
    'Dom_Goals_Against_5',
    'Dom_O25_Rate_5',
     'Dom_KG_Rate_5']].copy()
    home_dom = home_dom.rename(
    columns={
        c: c +
        '_Home' for c in home_dom.columns if c not in [
            'Team',
             'Date']})
    home_dom = home_dom.rename(
        columns={'Team': 'HomeTeam'}).sort_values('Date')

    away_dom = dom_df[['Team',
    'Date',
    'Dom_xG_For_5',
    'Dom_xG_Against_5',
    'Dom_Goals_For_5',
    'Dom_Goals_Against_5',
    'Dom_O25_Rate_5',
     'Dom_KG_Rate_5']].copy()
    away_dom = away_dom.rename(
    columns={
        c: c +
        '_Away' for c in away_dom.columns if c not in [
            'Team',
             'Date']})
    away_dom = away_dom.rename(
        columns={'Team': 'AwayTeam'}).sort_values('Date')

    h_cols = {
    c: c +
    '_Home' for c in team_df.columns if c not in [
        'Date',
        'Team',
        'Is_Euro',
        'League',
         'Season']}
    a_cols = {
    c: c +
    '_Away' for c in team_df.columns if c not in [
        'Date',
        'Team',
        'Is_Euro',
        'League',
         'Season']}

    # ------------------------------------------------------------------
    # SAFETY: prevent many-to-many merges (can explode RAM via cartesian joins)
    # We enforce one row per (Date, Team) in team_df before merging.
    # ------------------------------------------------------------------
    team_df = team_df.copy()
    all_data = all_data.copy()
    team_df['Date'] = pd.to_datetime(
    team_df['Date'], errors='coerce').dt.normalize()
    all_data['Date'] = pd.to_datetime(
    all_data['Date'], errors='coerce').dt.normalize()

    team_df = team_df.sort_values(['Team', 'Date']).drop_duplicates([
                                  'Team', 'Date'], keep='last')

    h_rename_map = h_cols.copy()
    h_rename_map['Team'] = 'HomeTeam'
    _home_feats = team_df.rename(columns=h_rename_map)[
                                 ['Date', 'HomeTeam'] + list(h_cols.values())]
    _home_feats = _home_feats.sort_values(['HomeTeam', 'Date']).drop_duplicates([
                                          'HomeTeam', 'Date'], keep='last')
    all_data = pd.merge(
        all_data, _home_feats,
        on=['Date', 'HomeTeam'],
        how='left',
        validate='m:1'
    )

    a_rename_map = a_cols.copy()
    a_rename_map['Team'] = 'AwayTeam'
    _away_feats = team_df.rename(columns=a_rename_map)[
                                 ['Date', 'AwayTeam'] + list(a_cols.values())]
    _away_feats = _away_feats.sort_values(['AwayTeam', 'Date']).drop_duplicates([
                                          'AwayTeam', 'Date'], keep='last')
    all_data = pd.merge(
        all_data, _away_feats,
        on=['Date', 'AwayTeam'],
        how='left',
        validate='m:1'
    )

    all_data = all_data.sort_values('Date')
    all_data = _safe_merge_asof_by(
    all_data,
    home_dom,
    on='Date',
    by='HomeTeam',
     direction='backward')
    all_data = _safe_merge_asof_by(
    all_data,
    away_dom,
    on='Date',
    by='AwayTeam',
     direction='backward')

    euro_mask = all_data['Is_Euro']

    def create_adj_feature(df, base_col, dom_col):
        adj_col = df[base_col].copy()
        if dom_col in df.columns:
            blended = 0.7 * df.loc[euro_mask,
    dom_col].fillna(df.loc[euro_mask,
    base_col]) + 0.3 * df.loc[euro_mask,
     base_col]
            adj_col.loc[euro_mask] = blended
        return adj_col

    for side in ['Home', 'Away']:
        all_data[f'xG_For_5_{side}_Adj'] = create_adj_feature(
            all_data, f'xG_For_5_{side}', f'Dom_xG_For_5_{side}')
        all_data[f'xG_Against_5_{side}_Adj'] = create_adj_feature(
            all_data, f'xG_Against_5_{side}', f'Dom_xG_Against_5_{side}')
        all_data[f'Goals_For_5_{side}_Adj'] = create_adj_feature(
            all_data, f'Goals_For_5_{side}', f'Dom_Goals_For_5_{side}')
        all_data[f'Goals_Against_5_{side}_Adj'] = create_adj_feature(
            all_data, f'Goals_Against_5_{side}', f'Dom_Goals_Against_5_{side}')
        all_data[f'O25_Rate_5_{side}_Adj'] = create_adj_feature(
            all_data, f'O25_Rate_5_{side}', f'Dom_O25_Rate_5_{side}')
        all_data[f'KG_Rate_5_{side}_Adj'] = create_adj_feature(
            all_data, f'KG_Rate_5_{side}', f'Dom_KG_Rate_5_{side}')

    def _fill(col, fallback):
        if col in all_data.columns:
            if isinstance(fallback, str) and fallback in all_data.columns:
                all_data[col] = all_data[col].fillna(all_data[fallback])
            else:
                all_data[col] = all_data[col].fillna(fallback)

    _fill('Roll_xG_5_Home', all_data.get('xGHome', 1.3))
    _fill('Roll_xG_5_Away', all_data.get('xGAway', 1.1))
    _fill('Roll_GF_5_Home', all_data.get('HomeFullTimeScore', 1.2))
    _fill('Roll_GF_5_Away', all_data.get('AwayFullTimeScore', 1.1))
    _fill('Roll_GA_5_Home', 1.2)
    _fill('Roll_GA_5_Away', 1.1)
    _fill('Roll_xGA_Std_Home', 0.5)
    _fill('Roll_xGA_Std_Away', 0.5)

    _fill('GF_Std_10_Home', 0.8)
    _fill('GF_Std_10_Away', 0.8)
    _fill('GA_Std_10_Home', 0.8)
    _fill('GA_Std_10_Away', 0.8)
    _fill('Goals_Total_Std_10_Home', 1.0)
    _fill('Goals_Total_Std_10_Away', 1.0)

    _fill('xG_For_5_Home', 1.3)
    _fill('xG_For_5_Away', 1.1)
    _fill('xG_Against_5_Home', 1.1)
    _fill('xG_Against_5_Away', 1.3)
    _fill('Goals_For_5_Home', 1.4)
    _fill('Goals_For_5_Away', 1.0)
    _fill('Goals_Against_5_Home', 1.0)
    _fill('Goals_Against_5_Away', 1.4)
    _fill('KG_Rate_5_Home', 0.5)
    _fill('KG_Rate_5_Away', 0.5)
    _fill('O25_Rate_5_Home', 0.5)
    _fill('O25_Rate_5_Away', 0.5)

    all_data['xGD_last5_Home'] = all_data['Roll_xG_5_Home'] - \
        all_data['Roll_xGA_5_Home']
    all_data['xGD_last5_Away'] = all_data['Roll_xG_5_Away'] - \
        all_data['Roll_xGA_5_Away']
    all_data['xGD_last5_Diff'] = all_data['xGD_last5_Home'] - \
        all_data['xGD_last5_Away']

    league_avg_gf = all_data.groupby('League')['Roll_GF_5_Home'].transform('mean').fillna(
        1.2) + all_data.groupby('League')['Roll_GF_5_Away'].transform('mean').fillna(1.0)
    league_avg_gf /= 2.0

    league_avg_ga = all_data.groupby('League')['Roll_GA_5_Home'].transform('mean').fillna(
        1.2) + all_data.groupby('League')['Roll_GA_5_Away'].transform('mean').fillna(1.0)
    league_avg_ga /= 2.0

    league_total_goals = (
    all_data['Roll_GF_5_Home'] +
    all_data['Roll_GF_5_Away']).groupby(
        all_data['League']).transform('mean')
    all_data['League_Tempo_Level'] = league_total_goals

    league_mean = all_data.groupby(
        'League')['League_Tempo_Level'].transform('mean')
    league_std = all_data.groupby(
        'League')['League_Tempo_Level'].transform('std').replace(0, 1.0)
    all_data['League_Tempo_z'] = (
        (all_data['League_Tempo_Level'] - league_mean) / league_std).fillna(0.0).clip(-3, 3)

    all_data['PreMatch_Total_xG'] = all_data['Roll_xG_5_Home'] + \
        all_data['Roll_xG_5_Away']

    all_data['xG_Variance_10'] = (
    all_data['Goals_Total_Std_10_Home'] +
     all_data['Goals_Total_Std_10_Away']).fillna(0)

    league_draw_map = (
        all_data[all_data['Dataset'] == 'past']
        .groupby('League')
        .apply(lambda g: np.mean(g['HomeFullTimeScore'] == g['AwayFullTimeScore']))
    ).to_dict()
    all_data['League_Draw_Rate_5Y'] = all_data['League'].map(
        league_draw_map).fillna(0.28)

    all_data['AttStrength_Home'] = all_data['Roll_GF_5_Home'] / \
        (league_avg_gf + 0.01)
    all_data['AttStrength_Away'] = all_data['Roll_GF_5_Away'] / \
        (league_avg_gf + 0.01)

    all_data['DefWeakness_Home'] = all_data['Roll_GA_5_Home'] / \
        (league_avg_ga + 0.01)
    all_data['DefWeakness_Away'] = all_data['Roll_GA_5_Away'] / \
        (league_avg_ga + 0.01)

    HOME_ADV_FACTOR = 1.12
    lam_h = (all_data['Roll_xG_5_Home'].clip(0.2, 4.0) *
    all_data['AttStrength_Home'].clip(0.5, 2.0) *
    all_data['DefWeakness_Away'].clip(0.5, 2.0) *
     HOME_ADV_FACTOR).fillna(1.3).to_numpy()
    lam_a = (
    all_data['Roll_xG_5_Away'].clip(
        0.2,
        4.0) *
        all_data['AttStrength_Away'].clip(
            0.5,
            2.0) *
            all_data['DefWeakness_Home'].clip(
                0.5,
                 2.0)).fillna(1.1).to_numpy()

    n2 = len(all_data)
    p_home = np.zeros(n2)
    p_away = np.zeros(n2); p_draw = np.zeros(n2); p_btts = np.zeros(n2)
    p_over25 = np.zeros(n2)
    max_g = 8
    g = np.arange(max_g + 1)

    for idx in range(n2):
        ph = poisson.pmf(g, lam_h[idx])
        pa = poisson.pmf(g, lam_a[idx])
        joint_matrix = np.outer(ph, pa)

        p_home[idx] = np.sum(np.tril(joint_matrix, -1))
        p_away[idx] = np.sum(np.triu(joint_matrix, 1))
        p_draw[idx] = np.sum(np.diag(joint_matrix))

        p_btts[idx] = np.sum(joint_matrix[1:, 1:])

        total_goals_probs = np.convolve(ph, pa)
        if len(total_goals_probs) > 3:
            p_over25[idx] = total_goals_probs[3:].sum()
        else:
            p_over25[idx] = 0.0

    all_data['P_Home_Pois'] = p_home
    all_data['P_Away_Pois'] = p_away; all_data['P_Draw_Pois'] = p_draw; all_data['P_BTTS_Pois'] = p_btts
    all_data['P_Over25_Pois'] = p_over25

    future_mask = (all_data['Dataset'] == 'future')

    all_data['Imp_Home_Open'] = 1 / \
        all_data['Odds_Open_Home'].replace(0, np.nan).fillna(100)
    all_data['Imp_Draw_Open'] = 1 / \
        all_data['Odds_Open_Draw'].replace(0, np.nan).fillna(100)
    all_data['Imp_Away_Open'] = 1 / \
        all_data['Odds_Open_Away'].replace(0, np.nan).fillna(100)

    all_data['Imp_Home_Close'] = 1 / \
        all_data['ClosingHomeOdd'].replace(0, np.nan)
    all_data.loc[future_mask,
    'Imp_Home_Close'] = all_data.loc[future_mask,
     'Imp_Home_Open']
    all_data['Imp_Home_Close'] = all_data['Imp_Home_Close'].fillna(0)

    all_data['Imp_Draw_Close'] = 1 / \
        all_data['ClosingDrawOdd'].replace(0, np.nan)
    all_data.loc[future_mask,
    'Imp_Draw_Close'] = all_data.loc[future_mask,
     'Imp_Draw_Open']
    all_data['Imp_Draw_Close'] = all_data['Imp_Draw_Close'].fillna(0)

    all_data['Imp_Away_Close'] = 1 / \
        all_data['ClosingAwayOdd'].replace(0, np.nan)
    all_data.loc[future_mask,
    'Imp_Away_Close'] = all_data.loc[future_mask,
     'Imp_Away_Open']
    all_data['Imp_Away_Close'] = all_data['Imp_Away_Close'].fillna(0)

    all_data['Home_Prob_Delta'] = (
    all_data['Imp_Home_Close'] -
     all_data['Imp_Home_Open']).fillna(0)
    all_data['Draw_Prob_Delta'] = (
    all_data['Imp_Draw_Close'] -
     all_data['Imp_Draw_Open']).fillna(0)
    all_data['Away_Prob_Delta'] = (
    all_data['Imp_Away_Close'] -
     all_data['Imp_Away_Open']).fillna(0)

    all_data['LogOdds_Home_Open'] = - \
        np.log(all_data['Odds_Open_Home'].clip(lower=1.01))
    all_data['LogOdds_Draw_Open'] = - \
        np.log(all_data['Odds_Open_Draw'].clip(lower=1.01))
    all_data['LogOdds_Away_Open'] = - \
        np.log(all_data['Odds_Open_Away'].clip(lower=1.01))

    all_data['LogOdds_Home_Close'] = - \
        np.log(all_data['ClosingHomeOdd'].clip(lower=1.01))
    all_data['LogOdds_Draw_Close'] = - \
        np.log(all_data['ClosingDrawOdd'].clip(lower=1.01))
    all_data['LogOdds_Away_Close'] = - \
        np.log(all_data['ClosingAwayOdd'].clip(lower=1.01))

    all_data['LogOdds_Home_Delta'] = (
    all_data['LogOdds_Home_Close'] -
     all_data['LogOdds_Home_Open']).fillna(0)
    all_data['LogOdds_Draw_Delta'] = (
    all_data['LogOdds_Draw_Close'] -
     all_data['LogOdds_Draw_Open']).fillna(0)
    all_data['LogOdds_Away_Delta'] = (
    all_data['LogOdds_Away_Close'] -
     all_data['LogOdds_Away_Open']).fillna(0)

    if "LogOdds_Draw_Delta" in all_data.columns:
        all_data['Draw_Odds_Compression'] = all_data['LogOdds_Draw_Delta'].abs()
    else:
        all_data['Draw_Odds_Compression'] = 0.0

    all_data['Imp_Over25_Open'] = 1 / \
        all_data['Odds_Open_Over25'].replace(0, np.nan).fillna(100)
    all_data['Imp_Under25_Open'] = 1 / \
        all_data['Odds_Open_Under25'].replace(0, np.nan).fillna(100)

    all_data['Imp_Over25_Close'] = 1 / \
        all_data['ClosingO25'].replace(0, np.nan)
    all_data.loc[future_mask,
    'Imp_Over25_Close'] = all_data.loc[future_mask,
     'Imp_Over25_Open']
    all_data['Imp_Over25_Close'] = all_data['Imp_Over25_Close'].fillna(0)

    all_data['Imp_Under25_Close'] = 1 / \
        all_data['ClosingU25'].replace(0, np.nan)
    all_data.loc[future_mask,
    'Imp_Under25_Close'] = all_data.loc[future_mask,
     'Imp_Under25_Open']
    all_data['Imp_Under25_Close'] = all_data['Imp_Under25_Close'].fillna(0)

    all_data['Over25_Prob_Delta'] = (
    all_data['Imp_Over25_Close'] -
     all_data['Imp_Over25_Open']).fillna(0)
    all_data['Under25_Prob_Delta'] = (
    all_data['Imp_Under25_Close'] -
     all_data['Imp_Under25_Open']).fillna(0)

    all_data['Imp_BTTSY_Open'] = 1 / \
        all_data['Odds_Open_BTTS_Yes'].replace(0, np.nan).fillna(100)
    all_data['Imp_BTTSN_Open'] = 1 / \
        all_data['Odds_Open_BTTS_No'].replace(0, np.nan).fillna(100)

    all_data['Imp_BTTSY_Close'] = 1 / \
        all_data['ClosingBTTSY'].replace(0, np.nan)
    all_data.loc[future_mask,
    'Imp_BTTSY_Close'] = all_data.loc[future_mask,
     'Imp_BTTSY_Open']
    all_data['Imp_BTTSY_Close'] = all_data['Imp_BTTSY_Close'].fillna(0)

    all_data['Imp_BTTSN_Close'] = 1 / \
        all_data['ClosingBTTSN'].replace(0, np.nan)
    all_data.loc[future_mask,
    'Imp_BTTSN_Close'] = all_data.loc[future_mask,
     'Imp_BTTSN_Open']
    all_data['Imp_BTTSN_Close'] = all_data['Imp_BTTSN_Close'].fillna(0)

    all_data['BTTSY_Prob_Delta'] = (
    all_data['Imp_BTTSY_Close'] -
     all_data['Imp_BTTSY_Open']).fillna(0)

    all_data['Book_Exp_Total_Goals'] = 2.5 + \
        (all_data['Imp_Over25_Close'] - 0.5) * 2.5

    all_data['Tempo_Index'] = (all_data['Roll_xG_5_Home'] + all_data['Roll_xG_5_Away'] +
                               all_data['Roll_GF_5_Home'] + all_data['Roll_GF_5_Away']) / 4.0
    tempo_mean = all_data['Tempo_Index'].expanding().mean()
    tempo_std = all_data['Tempo_Index'].expanding().std().fillna(1.0)
    all_data['Tempo_Index_z'] = (
        all_data['Tempo_Index'] - tempo_mean) / (tempo_std + 1e-6)

    all_data['Attacking_Imbalance'] = (
    all_data['xG_For_5_Home_Adj'] -
     all_data['xG_Against_5_Away_Adj'])
    all_data['Defensive_Imbalance'] = (
    all_data['xG_Against_5_Home_Adj'] -
     all_data['xG_For_5_Away_Adj'])

    all_data['Total_Tempo'] = (
    all_data['Roll_xG_5_Home'] +
    all_data['Roll_xG_5_Away'] +
    all_data['Roll_GF_5_Home'] +
     all_data['Roll_GF_5_Away'])

    all_data['Lambda_Home_Adj'] = (
    1.0 *
    all_data['xG_For_5_Home_Adj'] +
    0.75 *
    all_data['Goals_For_5_Home_Adj'] +
    0.40 *
     all_data['Roll_xG_5_Home'])
    all_data['Lambda_Away_Adj'] = (
    1.0 *
    all_data['xG_For_5_Away_Adj'] +
    0.75 *
    all_data['Goals_For_5_Away_Adj'] +
    0.40 *
     all_data['Roll_xG_5_Away'])

    all_data['Collapse_Index'] = (
    all_data['Roll_xGA_5_Home'] *
    all_data['Roll_xGA_Std_Home'] +
    all_data['Roll_xGA_5_Away'] *
     all_data['Roll_xGA_Std_Away'])
    all_data['TBxG'] = (
    (all_data['Roll_xG_5_Home'] +
    all_data['Roll_xG_5_Away']) *
    all_data['Tempo_Index'].clip(
        0.8,
         1.2))
    all_data['OU_Att_Sum'] = all_data['Roll_GF_5_Home'] + \
        all_data['Roll_GF_5_Away']
    all_data['OU_Def_Sum'] = all_data['Roll_GA_5_Home'] + \
        all_data['Roll_GA_5_Away']
    all_data['OU_AttDef_Diff'] = all_data['OU_Att_Sum'] - \
        all_data['OU_Def_Sum']
    all_data['OU_Chaos_Index'] = (
    all_data['Roll_xGA_Std_Home'] +
     all_data['Roll_xGA_Std_Away']).fillna(0.5)
    all_data['L_Over_Factor'] = all_data['Tempo_Index_z'] + \
        all_data['TBxG'].fillna(2.5)
    all_data['L_BTTS_Factor'] = all_data['OU_Att_Sum'] - all_data['OU_Def_Sum']
    all_data['BTTS_Att_Sum'] = all_data['OU_Att_Sum']
    all_data['BTTS_Def_Weak'] = all_data['OU_Def_Sum']

    all_data['Mgr_Exp_Home'] = np.log1p(all_data.get('Manager_Games_Home', 0))
    all_data['Mgr_Exp_Away'] = np.log1p(all_data.get('Manager_Games_Away', 0))
    all_data['Mgr_Exp_Diff'] = (
        all_data['Mgr_Exp_Home'] - all_data['Mgr_Exp_Away']).clip(-3, 3)

    all_data['Form_Power_Home'] = all_data['Roll_xG_5_Home'] - \
        all_data['Roll_xGA_5_Home']
    all_data['Form_Power_Away'] = all_data['Roll_xG_5_Away'] - \
        all_data['Roll_xGA_5_Away']
    all_data['Form_TMB_Diff'] = all_data['Form_Power_Home'] - \
        all_data['Form_Power_Away']
    all_data['Form_THBH_Diff'] = (
    (all_data['Roll_GF_5_Home'] - all_data['Roll_GA_5_Home']) - (
        all_data['Roll_GF_5_Away'] - all_data['Roll_GA_5_Away']))

    if 'Att_Home' not in all_data.columns:
        all_data['Att_Home'] = all_data['Roll_xG_5_Home']
    if 'Def_Home' not in all_data.columns: all_data['Def_Home'] = -all_data['Roll_xGA_5_Home']
    if 'Att_Away' not in all_data.columns:
        all_data['Att_Away'] = all_data['Roll_xG_5_Away']
    if 'Def_Away' not in all_data.columns: all_data['Def_Away'] = -all_data['Roll_xGA_5_Away']

    all_data['Elo_Diff'] *= 1.4
    all_data['SupremacyShift'] *= 1.4
    for col in ['Att_Home', 'Att_Away', 'Def_Home', 'Def_Away']:
        if col in all_data.columns:
            all_data[col] *= 1.3
    if 'HomeLeaguePosition' in all_data.columns: all_data['HomeLeaguePosition'] *= 1.2
    if 'AwayLeaguePosition' in all_data.columns:
        all_data['AwayLeaguePosition'] *= 1.2
    roll_cols = [c for c in all_data.columns if 'Roll_' in c]
    for c in roll_cols:
        all_data[c] *= 0.8
    if 'League_Size' in all_data.columns: all_data['League_Size'] *= 0.5

    all_data['Imp_Home'] = 1 / all_data['HomeOdd'].replace(0, np.nan)
    all_data['Imp_Draw'] = 1 / all_data['DrawOdd'].replace(
        0, np.nan); all_data['Imp_Away'] = 1 / all_data['AwayOdd'].replace(0, np.nan)
    all_data[['Imp_Home', 'Imp_Draw', 'Imp_Away']] = all_data[[
        'Imp_Home', 'Imp_Draw', 'Imp_Away']].fillna(0)

    all_data['Market_Bias_Home'] = all_data['Imp_Home'] - \
        (1 / (1 + 10 ** (-all_data['Elo_Diff'])))
    all_data['Market_Bias_Away'] = all_data['Imp_Away'] + \
        (1 / (1 + 10 ** (-all_data['Elo_Diff'])))
    if 'Supremacy_Calc' not in all_data.columns:
        all_data['Supremacy_Calc'] = all_data['Elo_Diff'] * 0.8
    all_data['Supremacy_Calc_z'] = np.tanh(all_data['Supremacy_Calc'])
    all_data['Total_Goals'] = (
    all_data['HomeFullTimeScore'].fillna(0) +
     all_data['AwayFullTimeScore'].fillna(0))

    ratio_raw = np.exp(all_data['Supremacy_Calc_z']).clip(0.6, 1.4)
    all_data['Book_xG_Home'] = all_data['Book_Exp_Total_Goals'] * \
        ratio_raw / (1 + ratio_raw)
    all_data['Book_xG_Away'] = all_data['Book_Exp_Total_Goals'] - \
        all_data['Book_xG_Home']

    if 'Odds_Open_Home' in all_data.columns and 'ClosingHomeOdd' in all_data.columns:
        diff_h = (
    all_data['Odds_Open_Home'] -
     all_data['ClosingHomeOdd']).abs()
        diff_d = (
    all_data['Odds_Open_Draw'] -
     all_data['ClosingDrawOdd']).abs()
        diff_a = (
    all_data['Odds_Open_Away'] -
     all_data['ClosingAwayOdd']).abs()
        all_data['Odds_Compression_1X2'] = diff_h + diff_d + diff_a
    else:
        all_data['Odds_Compression_1X2'] = 0.0

    all_data['Elo_Home'] = all_data['HomeELO_Final']
    all_data['Elo_Away'] = all_data['AwayELO_Final']
    all_data['Elo_Gap'] = all_data['Elo_Gap_Abs']

    all_data['Form_Home'] = all_data['Form_Power_Home']
    all_data['Form_Away'] = all_data['Form_Power_Away']
    all_data['Form_Gap'] = all_data['Form_TMB_Diff'].abs()

    all_data['xG_Home'] = all_data['Roll_xG_5_Home']
    all_data['xG_Away'] = all_data['Roll_xG_5_Away']
    all_data['xG_Gap'] = all_data['xGD_last5_Diff'].abs()

    all_data['Def_Var_Home'] = all_data['Roll_xGA_Std_Home']
    all_data['Def_Var_Away'] = all_data['Roll_xGA_Std_Away']
    all_data['Total_xG_Exp'] = all_data['Book_Exp_Total_Goals']

    all_data['Gap_Composite'] = (
        all_data['Elo_Gap'] * 0.4 +
        all_data['Form_Gap'] * 0.3 +
        all_data['xG_Gap'] * 0.3
    )

    league_avg_xg_rolling = all_data.groupby(
        'League')[['Roll_xG_5_Home', 'Roll_xG_5_Away']].transform('mean').sum(axis=1)
    all_data['Tempo_Gap'] = abs(
    (all_data['xG_Home'] +
    all_data['xG_Away']) -
     league_avg_xg_rolling)

    all_data['KG_Flag'] = (
    (all_data['HomeFullTimeScore'] > 0) & (
        all_data['AwayFullTimeScore'] > 0)).astype(int)
    all_data['O25_Flag'] = (all_data['Total_Goals'] > 2.5).astype(int)

    all_data['League_Goal_Avg'] = all_data.groupby("League")['Total_Goals'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(2.5)
    all_data['League_KG_Avg'] = all_data.groupby("League")['KG_Flag'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.5)
    all_data['League_O25_Avg'] = all_data.groupby("League")['O25_Flag'].transform(
        lambda x: x.shift(1).expanding().mean()).fillna(0.5)

    # -------------------------------------------------------------------------
    # GÜNCELLEME: Rest_Days (Fikstür) & Home_Away_Contrast (Karakter) Eklentisi
    # -------------------------------------------------------------------------
#     try:  # AUTO-COMMENTED (illegal global try)
#         _date_col = "Date"
#         if _date_col in all_data.columns:
#             _tmp = all_data[[_date_col, "HomeTeam", "AwayTeam"]].copy()
#             _tmp[_date_col] = pd.to_datetime(_tmp[_date_col], errors="coerce")

    all_data["Rest_Days"] = 7

#     try:  # AUTO-COMMENTED (illegal global try)
#         # Home/Away Contrast (Placeholder - veri sızıntısı riski olmadan basit
#         # default)
#         all_data["Home_Away_Contrast"] = 0.0
#         # (İleri versiyonda buraya detaylı hesaplama eklenebilir, şimdilik sütun bulunsun yeter)
#     except Exception:
#         all_data["Home_Away_Contrast"] = 0.0

    return all_data

# -----------------------------------------------------------------------------#
# 5. MAIN ORCHESTRATOR & BLIND TEST PROTOCOL
# -----------------------------------------------------------------------------#


def train_goals_poisson_model(train_df, predict_df, weights):
    feats = FEATURE_SETS['ou_base'] + ['AttStrength_Home',
        'AttStrength_Away', 'DefWeakness_Home', 'DefWeakness_Away']
    feats = get_available_features(train_df, feats)

    X_train = train_df[feats].fillna(0)
    y_home = train_df['HomeFullTimeScore'].fillna(1.0)
    y_away = train_df['AwayFullTimeScore'].fillna(1.0)

    reg_home = HistGradientBoostingRegressor(
    max_depth=5, learning_rate=0.08, random_state=42)
    reg_home.fit(X_train, y_home, sample_weight=weights)

    reg_away = HistGradientBoostingRegressor(
    max_depth=5, learning_rate=0.08, random_state=42)
    reg_away.fit(X_train, y_away, sample_weight=weights)

    X_pred = predict_df[feats].fillna(0)
    lam_h = reg_home.predict(X_pred).clip(0.1, 5.0)
    lam_a = reg_away.predict(X_pred).clip(0.1, 5.0)

    n = len(lam_h)
    p_o25 = np.zeros(n)
    p_kg = np.zeros(n)

    for i in range(n):
        lh, la = lam_h[i], lam_a[i]
        p_matrix = np.outer(
    poisson.pmf(
        np.arange(6), lh), poisson.pmf(
            np.arange(6), la))
        mask_o25 = np.add.outer(np.arange(6), np.arange(6)) > 2.5
        p_o25[i] = np.sum(p_matrix * mask_o25)
        p_kg[i] = np.sum(p_matrix[1:, 1:])

    return lam_h, lam_a, p_o25, p_kg


def build_oof_predictions(model_class, params, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros(
    (len(X), 3)) if len(
        np.unique(y)) > 2 else np.zeros(
            (len(X), 2))

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]

        mdl = model_class(**params)
        mdl.fit(X_tr, y_tr)

        probs = mdl.predict_proba(X_val)
        if probs.shape[1] == oof_probs.shape[1]:
            oof_probs[val_idx, :] = probs
        else:
            # Handle binary case where only 1 col might be returned (rare but
            # safe)
            oof_probs[val_idx, 1] = probs[:, 1]
            oof_probs[val_idx, 0] = probs[:, 0]

    return oof_probs


def run_training_pipeline(
    train_data,
    predict_data,
    fast_mode: bool = False,
     weights_col: str = 'Weight'):
    if fast_mode:
        return run_training_pipeline_fast(
    train_data, predict_data, weights_col=weights_col)
    else:
        return run_training_pipeline_full(
    train_data, predict_data, weights_col=weights_col)

# 📌 1️⃣ UPDATED: Full OU & KG Stacking + Meta Pipeline


def run_training_pipeline_full(train_data, predict_data, weights_col='Weight'):

    if weights_col not in train_data.columns:
        days_diff = (train_data['Date'].max() - train_data['Date']).dt.days
        train_data[weights_col] = np.exp(-days_diff / (365 * 2)).values

    train_data = train_data.sort_values('Date')
    n_train = len(train_data)
    inner_idx = int(n_train * 0.8)
    train_in = train_data.iloc[:inner_idx].copy()  # Base training set
    calib_in = train_data.iloc[inner_idx:].copy()  # Calibration/Validation set
    weights_in = train_in[weights_col].values

    # -------------------------------------------------------------------------
    # 1) 1X2 STACKING PIPELINE (Existing)
    # -------------------------------------------------------------------------
    oneX2_feats = sorted(
        list(set(FEATURE_SETS['home']) | set(FEATURE_SETS['away'])))
    oneX2_feats = get_available_features(train_data, oneX2_feats)

    oof_probs_train_in = build_oof_predictions(
        HistGradientBoostingClassifier,
        {'max_depth': 6, 'learning_rate': 0.08,
            'max_iter': 400, 'random_state': 42},
        train_in[oneX2_feats].fillna(0),
        train_in['Target_Result']
    )

    clf_base = HistGradientBoostingClassifier(
    max_depth=6, learning_rate=0.08, max_iter=400, random_state=42)
    clf_base.fit(
    train_in[oneX2_feats].fillna(0),
    train_in['Target_Result'],
     sample_weight=weights_in)

    # Specialist Draw Model
    mask_draw = (train_in['Elo_Gap'].abs() < 80) & (
        train_in['Supremacy_Calc_z'].abs() < 0.6)
    train_draw_subset = train_in[mask_draw]
    has_draw_spec = False
    clf_draw_spec = None
    if len(train_draw_subset) > 50:
        has_draw_spec = True
        draw_feats = get_available_features(train_in, FEATURE_SETS['draw'])
        y_draw_bin = (train_draw_subset['Target_Result'] == 1).astype(int)
        clf_draw_spec = HistGradientBoostingClassifier(
    max_depth=4, learning_rate=0.05, random_state=42)
        clf_draw_spec.fit(train_draw_subset[draw_feats].fillna(0), y_draw_bin)

    def get_meta_features_1x2(df_input, is_training_set=False):
        meta = pd.DataFrame(index=df_input.index)
        if is_training_set:
            meta['P_Home_raw'] = oof_probs_train_in[:, 2]
            meta['P_Draw_raw'] = oof_probs_train_in[:, 1]
            meta['P_Away_raw'] = oof_probs_train_in[:, 0]
        else:
            p_base = clf_base.predict_proba(df_input[oneX2_feats].fillna(0))
            meta['P_Home_raw'] = p_base[:, 2]
            meta['P_Draw_raw'] = p_base[:, 1]
            meta['P_Away_raw'] = p_base[:, 0]

        if has_draw_spec and clf_draw_spec is not None:
            draw_feats = get_available_features(df_input, FEATURE_SETS['draw'])
            p_draw_spec = clf_draw_spec.predict_proba(
                df_input[draw_feats].fillna(0))[:, 1]
            meta['P_Draw_Special'] = p_draw_spec
        else:
            meta['P_Draw_Special'] = 0.25
        return meta

    meta_train_X = get_meta_features_1x2(train_in, is_training_set=True)
    meta_lr = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
     max_iter=1000)
    meta_lr.fit(
    meta_train_X,
    train_in['Target_Result'],
     sample_weight=weights_in)

    # 1X2 Calibration
    meta_calib_X = get_meta_features_1x2(calib_in, is_training_set=False)
    proba_cal_raw = meta_lr.predict_proba(meta_calib_X)

    # Preserve RAW (pre-calibration) 1X2 probabilities for "Raw metrics"
    calib_in['P_Away_Raw'] = proba_cal_raw[:, 0]
    calib_in['P_Draw_Raw'] = proba_cal_raw[:, 1]
    calib_in['P_Home_Raw'] = proba_cal_raw[:, 2]
    iso_home = IsotonicRegression(out_of_bounds='clip')
    iso_draw = IsotonicRegression(out_of_bounds='clip')
    iso_away = IsotonicRegression(out_of_bounds='clip')

    iso_home.fit(proba_cal_raw[:, 2],
                 (calib_in['Target_Result'] == 2).astype(int))
    iso_draw.fit(proba_cal_raw[:, 1],
                 (calib_in['Target_Result'] == 1).astype(int))
    iso_away.fit(proba_cal_raw[:, 0],
                 (calib_in['Target_Result'] == 0).astype(int))

    # 📦 Paket 3.1: Train MS2 Aux Model
    ms2_aux_model = train_ms2_aux_model(train_in)

    def apply_1x2_calibration(raw_probs, df_context=None):
        p_away = iso_away.transform(raw_probs[:, 0])
        p_draw = iso_draw.transform(raw_probs[:, 1])
        p_home = iso_home.transform(raw_probs[:, 2])

        # 📦 Paket 3.2: MS2 Aux Mixing
        if ms2_aux_model is not None and df_context is not None:
            # Recreate mask logic on the context df
            mask_ms2_zone = (
    df_context["ClosingAwayOdd"] < 3.20) & (
        df_context["ClosingHomeOdd"] > 1.40)

            # Need to align indices or work with arrays.
            # raw_probs aligns with df_context if passed correctly.
            # We iterate or use masking if shapes align.
            # Assuming df_context rows map 1-to-1 to raw_probs rows.

            if mask_ms2_zone.sum() > 0:
                # Prepare features for aux model
                aux_X_full = df_context.copy()
                if "League_Pos_Diff" not in aux_X_full.columns:
                    aux_X_full["League_Pos_Diff"] = aux_X_full["HomeLeaguePosition"] - \
                        aux_X_full["AwayLeaguePosition"]
                if "xG_Diff" not in aux_X_full.columns:
                    aux_X_full["xG_Diff"] = aux_X_full["Roll_xG_5_Home"] - \
                        aux_X_full["Roll_xG_5_Away"]

                feats_aux = [
    "Elo_Diff",
    "League_Pos_Diff",
    "Form_TMB_Diff",
    "xG_Diff",
     "League_Strength_Index"]
                X_pred_aux = aux_X_full.loc[mask_ms2_zone, get_available_features(
                    aux_X_full, feats_aux)].fillna(0)

                # Predict (1 = Away Win in Aux model target?)
                # Wait, train_ms2_aux_model uses y = (Target == 0). So
                # predict_proba[:, 1] is prob of Away Win.
                p_ms2_aux = ms2_aux_model.predict_proba(X_pred_aux)[:, 1]

                # Blend
                alpha = 0.30
                # Map mask to indices for p_away array
                idx_locs = np.where(mask_ms2_zone)[0]
                p_away[idx_locs] = (
    1 - alpha) * p_away[idx_locs] + alpha * p_ms2_aux

        probs = np.vstack([p_away, p_draw, p_home]).T
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / row_sums

    # Apply 1X2 Predictions
    if not predict_data.empty:
        predict_data = predict_data.copy()
        meta_pred_X = get_meta_features_1x2(
            predict_data, is_training_set=False)
        proba_pred_raw = meta_lr.predict_proba(meta_pred_X)
        # Preserve RAW (pre-calibration) 1X2 probabilities for "Raw metrics"
        predict_data['P_Away_Raw'] = proba_pred_raw[:, 0]
        predict_data['P_Draw_Raw'] = proba_pred_raw[:, 1]
        predict_data['P_Home_Raw'] = proba_pred_raw[:, 2]
        proba_pred_cal = apply_1x2_calibration(
    proba_pred_raw, df_context=predict_data)
        predict_data['P_Away_Final'] = proba_pred_cal[:, 0]
        predict_data['P_Draw_Final'] = proba_pred_cal[:, 1]
        predict_data['P_Home_Final'] = proba_pred_cal[:, 2]

    proba_cal_cal = apply_1x2_calibration(proba_cal_raw, df_context=calib_in)
    calib_in['P_Away_Final'] = proba_cal_cal[:, 0]
    calib_in['P_Draw_Final'] = proba_cal_cal[:, 1]
    calib_in['P_Home_Final'] = proba_cal_cal[:, 2]

    meta_train_viz = get_meta_features_1x2(train_in, is_training_set=False)
    proba_train_raw = meta_lr.predict_proba(meta_train_viz)
    proba_train_cal = apply_1x2_calibration(
        proba_train_raw, df_context=train_in)
    train_in['P_Away_Final'] = proba_train_cal[:, 0]
    train_in['P_Draw_Final'] = proba_train_cal[:, 1]
    train_in['P_Home_Final'] = proba_train_cal[:, 2]

    # -------------------------------------------------------------------------
    # 2) OU (2.5 Goals) STACKING PIPELINE (NEW)
    # -------------------------------------------------------------------------

    # Generate Poisson features first
    lam_h_tr, lam_a_tr, p_o25_tr, p_kg_tr = train_goals_poisson_model(
        train_in, train_in, weights_in)
    train_in['Poisson_Home_Lambda'] = lam_h_tr
    train_in['Poisson_Away_Lambda'] = lam_a_tr
    train_in['P_O25_Poisson'] = p_o25_tr
    train_in['P_KG_Poisson'] = p_kg_tr

    lam_h_cal, lam_a_cal, p_o25_cal, p_kg_cal = train_goals_poisson_model(
        train_in, calib_in, weights_in)
    calib_in['Poisson_Home_Lambda'] = lam_h_cal
    calib_in['Poisson_Away_Lambda'] = lam_a_cal
    calib_in['P_O25_Poisson'] = p_o25_cal
    calib_in['P_KG_Poisson'] = p_kg_cal

    if not predict_data.empty:
        lam_h_pr, lam_a_pr, p_o25_pr, p_kg_pr = train_goals_poisson_model(
            train_in, predict_data, weights_in)
        predict_data['Poisson_Home_Lambda'] = lam_h_pr
        predict_data['Poisson_Away_Lambda'] = lam_a_pr
        predict_data['P_O25_Poisson'] = p_o25_pr
        predict_data['P_KG_Poisson'] = p_kg_pr

    feats_ou = get_available_features(
    train_in,
    FEATURE_SETS['ou'] +
     FEATURE_SETS['totals'])

    # OU Base Models Training (OOF for meta)
    base_models_ou = train_base_models(
    train_in[feats_ou].fillna(0),
    train_in['Target_Over'],
     sample_weight=weights_in)
    # Note: Ideally real OOF, simplified here for perf
    oof_feats_ou = get_stacking_features(base_models_ou, train_in[feats_ou])

    # OU Meta Learner
    meta_ou = LogisticRegression(random_state=42)
    meta_ou.fit(
    oof_feats_ou,
    train_in['Target_Over'],
     sample_weight=weights_in)

    # ✅ PATCH #2: Train Aux Model (on Train)
    aux_model_ou = train_ou_aux_model(train_in, weights_in)

    # ✅ PATCH #1: Train Segment Calibrators (on Calib)
    calib_in = compute_expected_intensity(calib_in)
    calib_in["intensity_segment"] = calib_in["expected_intensity"].apply(
        intensity_segment)

    calib_feats_ou = get_stacking_features(base_models_ou, calib_in[feats_ou])
    calib_in['Prob_Over_Meta'] = meta_ou.predict_proba(calib_feats_ou)[:, 1]

    ou_segment_models = train_segment_calibrators(
        calib_in, 'Prob_Over_Meta', 'Target_Over')

    # 📦 Paket 1.2: Create Bucket Columns & Train Bucket Calibrators
    calib_in["OU25_Odds_Bucket"] = calib_in["ClosingO25"].apply(odds_bucket)

    # Apply segment calibration first to get a better base for bucket calibration
    # Or just use Meta prob. Let's use Meta prob for bucket calibration to
    # keep it distinct
    ou_bucket_models = train_bucket_calibrators(
    calib_in, 'Prob_Over_Meta', 'Target_Over', 'OU25_Odds_Bucket')

    # Apply OU Predictions (Meta -> Segment Calib -> Blend with Aux -> Bucket
    # Fix)
    def finalize_ou(df_target, feats):
        # 1. Base + Meta
        base_f = get_stacking_features(base_models_ou, df_target[feats])
        raw_p = meta_ou.predict_proba(base_f)[:, 1]

        # 2. Segment Calibration
        temp_df = df_target.copy()
        temp_df = compute_expected_intensity(temp_df)
        temp_df["intensity_segment"] = temp_df["expected_intensity"].apply(
            intensity_segment)

        calibrated_p = [apply_segment_calibration(row, ou_segment_models, p)
                        for (_, row), p in zip(temp_df.iterrows(), raw_p)]
        calibrated_p = np.array(calibrated_p)

        # 3. Aux Model Prediction
        aux_feats = [
    "Roll_xG_5_Home",
    "Roll_xG_5_Away",
    "Elo_Diff",
     "League_Goal_Avg"]
        aux_X = df_target[get_available_features(
            df_target, aux_feats)].fillna(0)
        aux_p = aux_model_ou.predict_proba(aux_X)[:, 1]

        # 4. Blend
        final_p = 0.7 * calibrated_p + 0.3 * aux_p

        # 📦 Paket 1.4: Bucket Calibration Adjustment
        # Create bucket column on temp_df
        temp_df["OU25_Odds_Bucket"] = temp_df["ClosingO25"].apply(odds_bucket)

        # Apply bucket calibration on top of the finalized prob (refinement)
        # Note: bucket calibrator was trained on Prob_Over_Meta.
        # Using it on final_p is slightly mismatched but acts as a sanity check for that bucket.
        # Alternatively, apply it on raw_p and blend. Let's apply on final_p as refinement.
        # Check if bucket exists
        bucket_p = []
        for (_, row), p in zip(temp_df.iterrows(), final_p):
            bucket = row["OU25_Odds_Bucket"]
            if bucket in ou_bucket_models:
                # Transform the *meta* probability logic usually, but here we transform the result
                # Actually, it's safer to just blend it or use apply_bucket_calibration logic
                # Let's use the helper: apply_bucket_calibration(row, models, p, col)
                # But helper expects row lookup.

                # Re-implement helper logic inline for speed in loop
                iso = ou_bucket_models[bucket]
                # Clip input to valid range
                p_clipped = np.clip(p, 0, 1)
                # Isotonic transform
                p_adj = iso.transform([p_clipped])[0]
                # Soft blend the adjustment (don't overwrite completely)
                bucket_p.append(0.8 * p + 0.2 * p_adj)
            else:
                bucket_p.append(p)

        return np.array(bucket_p)

    # Apply to all sets
    calib_in['P_Over_Final'] = finalize_ou(calib_in, feats_ou)
    calib_in['Prob_Over'] = calib_in['P_Over_Final']

    train_in['P_Over_Final'] = finalize_ou(train_in, feats_ou)
    train_in['Prob_Over'] = train_in['P_Over_Final']

    if not predict_data.empty:
        predict_data['P_Over_Final'] = finalize_ou(predict_data, feats_ou)
        predict_data['Prob_Over'] = predict_data['P_Over_Final']

    # -------------------------------------------------------------------------
    # 3) BTTS STACKING PIPELINE (NEW)
    # -------------------------------------------------------------------------
    feats_btts = get_available_features(train_in, FEATURE_SETS['btts'])

    base_models_btts = train_base_models(
    train_in[feats_btts].fillna(0),
    train_in['Target_BTTS'],
     sample_weight=weights_in)
    oof_feats_btts = get_stacking_features(
    base_models_btts, train_in[feats_btts])

    meta_btts = LogisticRegression(random_state=42)
    meta_btts.fit(
    oof_feats_btts,
    train_in['Target_BTTS'],
     sample_weight=weights_in)

    # ✅ PATCH #2: Train Aux Model (on Train)
    aux_model_btts = train_btts_aux_model(train_in, weights_in)

    # ✅ PATCH #1: Train Segment Calibrators (on Calib)
    calib_feats_btts = get_stacking_features(
        base_models_btts, calib_in[feats_btts])
    calib_in['Prob_BTTS_Meta'] = meta_btts.predict_proba(calib_feats_btts)[
                                                         :, 1]

    btts_segment_models = train_segment_calibrators(
        calib_in, 'Prob_BTTS_Meta', 'Target_BTTS')

    # 📦 Paket 1.2 & 1.3: BTTS Buckets
    calib_in["BTTSY_Odds_Bucket"] = calib_in["ClosingBTTSY"].apply(odds_bucket)
    btts_bucket_models = train_bucket_calibrators(
    calib_in, 'Prob_BTTS_Meta', 'Target_BTTS', 'BTTSY_Odds_Bucket')

    def finalize_btts(df_target, feats):
        # 1. Base + Meta
        base_f = get_stacking_features(base_models_btts, df_target[feats])
        raw_p = meta_btts.predict_proba(base_f)[:, 1]

        # 2. Segment Calibration
        temp_df = df_target.copy()
        temp_df = compute_expected_intensity(temp_df)
        temp_df["intensity_segment"] = temp_df["expected_intensity"].apply(
            intensity_segment)

        calibrated_p = [apply_segment_calibration(row, btts_segment_models, p)
                        for (_, row), p in zip(temp_df.iterrows(), raw_p)]
        calibrated_p = np.array(calibrated_p)

        # 3. Aux Model Prediction
        aux_feats = [
    "Roll_xG_5_Home",
    "Roll_xG_5_Away",
    "Elo_Diff",
     "League_Goal_Avg"]
        aux_X = df_target[get_available_features(
            df_target, aux_feats)].fillna(0)
        aux_p = aux_model_btts.predict_proba(aux_X)[:, 1]

        # 4. Final Blend
        final_p = 0.7 * calibrated_p + 0.3 * aux_p

        # 📦 Paket 1.4: Bucket Calibration
        temp_df["BTTSY_Odds_Bucket"] = temp_df["ClosingBTTSY"].apply(
            odds_bucket)
        bucket_p = []
        for (_, row), p in zip(temp_df.iterrows(), final_p):
            bucket = row["BTTSY_Odds_Bucket"]
            if bucket in btts_bucket_models:
                iso = btts_bucket_models[bucket]
                p_clipped = np.clip(p, 0, 1)
                p_adj = iso.transform([p_clipped])[0]
                bucket_p.append(0.8 * p + 0.2 * p_adj)
            else:
                bucket_p.append(p)

        return np.array(bucket_p)

    calib_in['P_BTTS_Final'] = finalize_btts(calib_in, feats_btts)
    calib_in['Prob_BTTS'] = calib_in['P_BTTS_Final']

    train_in['P_BTTS_Final'] = finalize_btts(train_in, feats_btts)
    train_in['Prob_BTTS'] = train_in['P_BTTS_Final']

    if not predict_data.empty:
        predict_data['P_BTTS_Final'] = finalize_btts(predict_data, feats_btts)
        predict_data['Prob_BTTS'] = predict_data['P_BTTS_Final']

        # EV & Smart EV Calc
        eff_odds = predict_data['ClosingHomeOdd'].fillna(
            predict_data['HomeOdd'])
        raw_ev = (predict_data['P_Home_Final'] * eff_odds - 1) * 100
        predict_data['Smart_EV'] = raw_ev.clip(-10, 20)
        predict_data['EV_Calc'] = predict_data['Smart_EV']

        return predict_data

    return pd.concat([train_in, calib_in], ignore_index=True)


def run_training_pipeline_fast(train_data, predict_data, weights_col='Weight'):
    """Simplified/Fast training pipeline."""
    train_data = train_data.copy()
    predict_data = predict_data.copy()
    if weights_col not in train_data.columns:
        train_data[weights_col] = 1.0

    feats = get_available_features(train_data, FEATURE_SETS['home'])
    model_h = HistGradientBoostingClassifier(
    max_depth=5).fit(
        train_data[feats].fillna(0),
         train_data['Target_Home'])
    predict_data['P_Home_Final'] = model_h.predict_proba(
        predict_data[feats].fillna(0))[:, 1]

    lam_h, lam_a, p_o25, p_kg = train_goals_poisson_model(
    train_data, predict_data, train_data[weights_col].values)
    predict_data['P_Over_Final'] = p_o25
    predict_data['Prob_Over'] = p_o25
    predict_data['P_BTTS_Final'] = p_kg
    predict_data['Prob_BTTS'] = p_kg

    predict_data['P_Draw_Final'] = (1.0 - predict_data['P_Home_Final']) * 0.35
    predict_data['P_Away_Final'] = (1.0 - predict_data['P_Home_Final']) * 0.65
    predict_data[['P_Away_Final',
    'P_Draw_Final',
    'P_Home_Final']] = predict_data[['P_Away_Final',
    'P_Draw_Final',
    'P_Home_Final']].div(predict_data[['P_Away_Final',
    'P_Draw_Final',
    'P_Home_Final']].sum(axis=1),
     axis=0)

    predict_data['Smart_EV'] = 0.0
    predict_data['EV_Calc'] = 0.0

    return predict_data


def run_walkforward_eval(train_core):
    full_df = train_core.sort_values("Date").copy()
    n = len(full_df)
    folds = [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
    results = []

    for i, (test_start_pct, test_end_pct) in enumerate(folds, start=1):
        train_end_idx = int(n * test_start_pct)
        test_end_idx = int(n * test_end_pct)

        train_subset = full_df.iloc[:train_end_idx].copy()
        val_subset = full_df.iloc[train_end_idx:test_end_idx].copy()

        if len(val_subset) < 50:
            continue

        val_pred = run_training_pipeline_full(train_subset, val_subset)

        cols_to_merge = [
    "Target_Result",
    "Target_Home",
    "Target_Over",
    "Target_BTTS",
     "ClosingHomeOdd"]
        for col in cols_to_merge:
            if col in val_subset.columns and col not in val_pred.columns:
                val_pred[col] = val_subset[col].values

        y_true = (val_pred["Target_Result"] == 2).astype(int)
        y_prob = val_pred["P_Home_Final"]
        auc_home = roc_auc_score(
    y_true, y_prob) if len(
        np.unique(y_true)) > 1 else 0.5

        sel = val_pred[val_pred["P_Home_Final"] > 0.5].copy()
        if len(sel) > 0:
            profit = np.where(sel["Target_Result"] == 2,
                              sel["ClosingHomeOdd"] - 1, -1)
            roi_home = profit.mean() * 100
            bets = len(sel)
        else:
            roi_home = 0.0
            bets = 0

        results.append({
            "Fold": i,
            "Train_Start": train_subset["Date"].min().date(),
            "Train_End": train_subset["Date"].max().date(),
            "Test_Start": val_subset["Date"].min().date(),
            "Test_End": val_subset["Date"].max().date(),
            "Home_AUC": float(auc_home),
            "Home_ROI_%": float(roi_home),
            "Home_Bets": int(bets),
            "Total_Matches": int(len(val_subset)),
        })
    return pd.DataFrame(results)


def process_and_train(past_df, future_df, fast_mode: bool = False, show_status: bool = True):
    class _NullStatus:
        def info(self, *a, **k): pass
        def success(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def write(self, *a, **k): pass
    status = st.empty() if show_status else _NullStatus()
    st.session_state["_dbg_runtime_marker"] = "PTJ_FIX_APPLIED_2025_12_23_A"
    def _status_info(msg: str) -> None:
        if status is not None:
            status.info(msg)
    if fast_mode:
        _status_info(
            "⚡ FAST MODE: Hızlı eğitim başlatılıyor (hafifletilmiş modeller, kısıtlı geçmiş)...")
    else:
        _status_info("🚀 V43.00 Motoru Başlatılıyor (Draw Spec + OU/KG Split)...")

    all_data = prepare_base_data(past_df, future_df)

    _status_info("🧬 Özellikler işleniyor (Expanding Window Anti-Leakage)...")
    all_data = engineer_features(all_data)

    all_data['Target_Result'] = np.select(
        [
            all_data['HomeFullTimeScore'] > all_data['AwayFullTimeScore'],
            all_data['HomeFullTimeScore'] == all_data['AwayFullTimeScore']
        ],
        [2, 1],  # 2: Home Win, 1: Draw, 0: Away Win
        default=0
    )
    all_data['Target_Home'] = (all_data['Target_Result'] == 2).astype(int)
    all_data['Target_Draw'] = (all_data['Target_Result'] == 1).astype(int)
    all_data['Target_Away'] = (all_data['Target_Result'] == 0).astype(int)
    all_data['Target_Over'] = (all_data['Total_Goals'] > 2.5).astype(int)
    all_data['Target_BTTS'] = (
    (all_data['HomeFullTimeScore'] > 0) & (
        all_data['AwayFullTimeScore'] > 0)).astype(int)

    train_core = all_data[all_data['Dataset'] == "past"].copy()
    predict_data = all_data[all_data['Dataset'] == "future"].copy()

    if train_core.empty:
        st.error("Eğitim verisi bulunamadı!")
        return predict_data, train_core, pd.DataFrame()

    if fast_mode and len(train_core) > 40000:
        train_core = train_core.sort_values("Date").iloc[-40000:].copy()

    train_core['Actual_ROI'] = np.where(
    train_core['Target_Result'] == 2, train_core['ClosingHomeOdd'] - 1, -1.0)

    if fast_mode:
        _status_info(
            "🛡️ FAST MODE: Geçmiş veriler 80/20 oranında 'Kör Test' için ayrılıyor...")
    else:
        _status_info(
            "🛡️ Güvenlik Protokolü: Geçmiş veriler 80/20 oranında 'Kör Test' için ayrılıyor...")

    split_ratio = 0.80
    train_core = train_core.sort_values('Date')
    split_idx = int(len(train_core) * split_ratio)

    train_subset = train_core.iloc[:split_idx].copy()
    validation_subset = train_core.iloc[split_idx:].copy()

    _status_info("📉 Validasyon modelleri eğitiliyor (Geçmiş performans)...")
    hist_df_results = run_training_pipeline(
    train_subset, validation_subset, fast_mode=fast_mode)

    st.session_state.market_threshold_stats = compute_market_threshold_stats(
        hist_df_results)

    if not fast_mode:
        _status_info("🔍 Feature Importance hesaplanıyor (sample ile)...")
        # ✅ Büyük datalarda hız için sample
        fi_data = train_subset
        if len(fi_data) > 25000:
            fi_data = fi_data.sample(25000, random_state=42)

        fi_df = compute_feature_importance(fi_data)
        st.session_state.feature_importance_df = fi_df
        st.session_state.feature_importance_df = pd.DataFrame()

    cols_to_merge = [
    'Target_Result',
    'Target_Over',
    'Target_BTTS',
    'Actual_ROI',
    'League',
    'ClosingHomeOdd',
    'ClosingDrawOdd',
    'ClosingAwayOdd',
    'ClosingO25',
    'ClosingU25',
    'ClosingBTTSY',
    'ClosingBTTSN',
     'Target_Home']
    for col in cols_to_merge:
        if col not in hist_df_results.columns and col in validation_subset.columns:
            hist_df_results[col] = validation_subset[col].values

    if fast_mode:
        _status_info(
            "🔮 FAST MODE: Üretim modelleri (gelecek tahminleri) hızlı pipeline ile eğitiliyor...")
    else:
        _status_info(
            "🔮 Production modelleri eğitiliyor (Gelecek tahmini - %100 veri)...")

    final_predictions = run_training_pipeline(
    train_core, predict_data, fast_mode=fast_mode)

    # --- Similarity Anchor (kNN prior) ---
    # Makes similarity influence probabilities (softly) while preserving model
    # outputs in *_Model columns.
#     try:  # AUTO-COMMENTED (illegal global try)
#         final_predictions = apply_similarity_anchor_to_predictions(
#             final_predictions,
#             train_core,
#             alpha_max=float(st.session_state.get("SIM_ALPHA_MAX", 0.60)),
#             k=int(st.session_state.get("SIM_K", 60)),
#             use_cutoff=bool(st.session_state.get("SIM_USE_CUTOFF", True)),
#             min_similarity=float(st.session_state.get("SIM_MIN_SIM", 0.55)),
#             min_neighbors=int(st.session_state.get("SIM_MIN_NEIGHBORS", 12)),
#             same_league=bool(st.session_state.get("SIM_SAME_LEAGUE", True)),
#             max_pool=int(st.session_state.get("SIM_MAX_POOL", 120000)),
#             years_back=int(st.session_state.get("SIM_YEARS_BACK", 4)),
#             preset=str(
#     st.session_state.get(
#         "SIM_PRESET",
#          "Auto (mevcut kolonlar)")),
#         )
#     except Exception:
#         pass

    opt_params = optimize_betting_strategy(hist_df_results)
    st.session_state.opt_params = opt_params

    if not fast_mode:
        st.session_state.global_penalty = 0.9
        ev_boost_map = analyze_ev_buckets_dynamic(hist_df_results)
        st.session_state.ev_boost_map = ev_boost_map
    else:
        st.session_state.global_penalty = 0.9
        st.session_state.ev_boost_map = {}

    def _status_success(msg: str) -> None:
        if status is not None:
            status.success(msg)
    if fast_mode:
        _status_success(
            "✅ FAST MODE tamamlandı: Metrikler hızlı kör test setinden üretildi.")
    else:
        _status_success(
            "✅ Analiz tamamlandı. Metrikler %100 bağımsız test verisinden üretildi.")

    return final_predictions, hist_df_results, train_core


@st.cache_data(show_spinner=False)
def process_and_train_cached(past_bytes: bytes, future_bytes: bytes, fast_mode: bool = False):
    try:
        past_df = read_csv_bytes(past_bytes)
        future_df = read_csv_bytes(future_bytes)
        preds, hist, _train_core = process_and_train(
            past_df, future_df, fast_mode=fast_mode, show_status=False)
        return preds, hist, _train_core
    except Exception:
        err = traceback.format_exc()
        st.session_state["_last_train_error"] = err
        st.session_state["_train_error"] = err
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
# -----------------------------------------------------------------------------#
# 5.X  SIMILARITY (kNN / Case-based)  -- LAZY (on-demand) to avoid startup stalls
# -----------------------------------------------------------------------------#


SIM_FEATURE_CANDIDATES = [
    "Elo_Diff",
    "League_Strength_Index",
    "Form_TMB_Diff",
    "xGD_last5_Diff",
    "Roll_xG_5_Home",
    "Roll_xG_5_Away",
    "Roll_xGA_Std_Home",
    "Roll_xGA_Std_Away",
    "Goals_Total_Std_10_Home",
    "Book_Exp_Total_Goals",
    "League_Goal_Avg",
    "League_Tempo_z",
    "Home_Prob_Delta",
    "SeasonProgress",
    "Rest_Days",
    "Home_Away_Contrast"
]


def _make_match_id(row: pd.Series) -> str:
    d = pd.to_datetime(row.get("Date", None), errors="coerce")
    ds = d.strftime("%Y-%m-%d") if pd.notna(d) else ""
    return f"{ds}|{
    row.get(
        'League',
        '')}|{
            row.get(
                'HomeTeam',
                '')}|{
                    row.get(
                        'AwayTeam',
                         '')}"


def _pick_similarity_features(
    upcoming_df: pd.DataFrame,
    train_core: pd.DataFrame,
     preset: str = "Auto (mevcut kolonlar)") -> list:
    """Pick similarity feature columns based on preset, restricted to columns available in BOTH dfs."""
    if upcoming_df is None or train_core is None or getattr(
    upcoming_df, "empty", True) or getattr(
        train_core, "empty", True):
        return []

    preset = (preset or "Auto (mevcut kolonlar)").strip()
    # Define lightweight presets (only a preference order; availability
    # filtering happens below).
    over_btts_pref = [
        "Roll_xG_5_Home",
        "Roll_xG_5_Away",
        "Book_Exp_Total_Goals",
        "Roll_xGA_Std_Home",
        "Goals_Total_Std_10_Home",
        "League_Tempo_z",
        "SeasonProgress",
        "Rest_Days",
        "Home_Away_Contrast"
    ]
    ms_pref = [
        "Elo_Diff",
        "Form_TMB_Diff",
        "xGD_last5_Diff",
        "Home_Prob_Delta",
        "League_Strength_Index",
        "SeasonProgress",
        "Book_Exp_Total_Goals",
        "Rest_Days",
        "Home_Away_Contrast"
    ]
    if ("OU" in preset) or ("Over/BTTS" in preset):
        # OU/Over: gol-temelli benzerlik
        cand = over_btts_pref + \
            [c for c in SIM_FEATURE_CANDIDATES if c not in over_btts_pref]
    elif "BTTS" in preset:
        # BTTS: karşılıklı gol-temelli benzerlik (mevcut kolonlar varsa onları
        # öne al)
        btts_pref = [
            "Roll_xG_5_Home",
            "Roll_xG_5_Away",
            "Roll_xGA_Std_Home",
            "Roll_xGA_Std_Away",
            "Goals_Total_Std_10_Home",
            "Goals_Total_Std_10_Away",
            "Home_Away_Contrast",
            "League_Tempo_z",
            "SeasonProgress",
            "Rest_Days",
        ]
        cand = btts_pref + \
            [c for c in SIM_FEATURE_CANDIDATES if c not in btts_pref]
    elif "MS" in preset:
        cand = ms_pref + \
            [c for c in SIM_FEATURE_CANDIDATES if c not in ms_pref]
    else:
        cand = list(SIM_FEATURE_CANDIDATES)

    feats = [
    c for c in cand if (
        c in upcoming_df.columns) and (
            c in train_core.columns)]
    return feats


# -----------------------------------------------------------
# Özellik Ağırlıklandırma Haritası
# Ana faktörler 1.0, Yan faktörler düşürülerek mesafeyi bozması engellenir.
# -----------------------------------------------------------
SIM_FEAT_WEIGHTS = {
    # --- CORE (Ana) Faktörler (1.0) ---
    "Elo_Diff": 1.0,
    "Form_TMB_Diff": 1.0,
    "Roll_xG_5_Home": 1.0, "Roll_xG_5_Away": 1.0,
    "Book_Exp_Total_Goals": 1.0,

    # --- SECONDARY (Orta) Faktörler (0.7) ---
    "League_Goal_Avg": 0.7,
    "xGD_last5_Diff": 0.7,
    "Roll_xGA_Std_Home": 0.7, "Roll_xGA_Std_Away": 0.7,

    # --- CONTEXT (Yan) Faktörler (0.4 - 0.5) ---
    # Bunlar benzerliği "bozmamalı", sadece ince ayar yapmalı
    "Rest_Days": 0.45,
    "Home_Away_Contrast": 0.45,
    "SeasonProgress": 0.40,
    "Home_Prob_Delta": 0.50,
    "League_Strength_Index": 0.50,
    "League_Tempo_z": 0.50,
    "Goals_Total_Std_10_Home": 0.50
}


def _safe_quality_from_dist(dist_arr, top_n: int = 30) -> float:
    """Return a 0..1 quality score from neighbor distances.

    We convert distance -> similarity via 1/(1+dist) and average the strongest neighbors.
    Designed to be robust to NaN/inf/empty inputs.
    """
    d = np.asarray(dist_arr if dist_arr is not None else [], dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    d.sort()
    d = d[: max(1, min(int(top_n), d.size))]
    sim = 1.0 / (1.0 + d)
    q = float(np.nanmean(sim)) if sim.size else 0.0
    if not np.isfinite(q):
        return 0.0
    return float(np.clip(q, 0.0, 1.0))


# =============================================================================
# V88 YAMA: Post-Scaling Weighting (Doğru Matematik)
# =============================================================================

def _similarity_neighbors_for_match(
    train_core: pd.DataFrame,
    upcoming_row: pd.Series,
    feats: list,
    k: int = 60,
    same_league: bool = True,
    max_pool: int = 60000,
    hard_cap_cross_league: int = 200000,
    use_cutoff: bool = True,
    min_similarity: float = 0.55,
    min_neighbors: int = 10,
    year_window: int = 0,
    # --- Sprint 5: Time-decay (regime awareness)
    use_time_decay: bool = False,
    half_life_days: int = 540,
    # --- Sprint 6: Gating thresholds (reliability brake)
    gate_min_simq: float = 0.30,
    gate_min_en: int = 6,
):
    if train_core is None or train_core.empty:
        return None, {"error": "train_core boş (past havuzu yok)."}
    if not feats or len(feats) < 2:
        return None, {
    "error": f"Benzerlik feature set'i yetersiz (bulunan: {
        len(feats)})."}

    h = train_core

    if year_window and year_window > 0 and ('Date' in h.columns):
        up_date = pd.to_datetime(
upcoming_row.get(
    'Date', None), errors='coerce')
        if pd.notna(up_date):
            min_date = up_date - pd.Timedelta(days=int(365 * year_window))
            hd = pd.to_datetime(h['Date'], errors='coerce')
            h = h.loc[hd >= min_date]

    # -----------------------------------------------------------
    # HYGIENE: Strict time filter (prevent future / same-day leakage)
    # -----------------------------------------------------------
    filtered_future_n = 0
    filtered_sameday_n = 0
    pool_max_date = None
    leak_ok = True
    dup_dropped_n = 0

    if 'Date' in h.columns:
        up_date = pd.to_datetime(
upcoming_row.get(
    'Date', None), errors='coerce')
        hd_all = pd.to_datetime(h['Date'], errors='coerce')
        pool_max_date = pd.to_datetime(
hd_all.max(), errors='coerce') if len(hd_all) else None
        if pd.notna(up_date):
            before = len(h)
            # Strictly earlier than match date (exclude same-day)
            h = h.loc[hd_all < up_date]
            filtered_future_n = int(before - len(h))
            # sanity
            hd_chk = pd.to_datetime(h['Date'], errors='coerce')
            if len(hd_chk) and pd.notna(
hd_chk.max()) and hd_chk.max() >= up_date:
                leak_ok = False

    # Optional: Deduplicate identical fixtures if columns exist
        dedup_cols = [
            c for c in [
                'League',
                'Date',
                'HomeTeam',
                'AwayTeam'] if c in h.columns]
        if len(dedup_cols) >= 3:
            before = len(h)
            h = h.drop_duplicates(subset=dedup_cols, keep='last')
            dup_dropped_n = int(before - len(h))
#     except Exception:  # AUTO-COMMENTED (orphan except)
#         pass

    if same_league and (
    "League" in h.columns) and pd.notna(
        upcoming_row.get(
            "League",
             None)):
        h = h[h["League"] == upcoming_row["League"]]

    if "Target_Result" not in h.columns:
        return None, {"error": "Target_Result train_core içinde yok."}

#     try:  # AUTO-COMMENTED (illegal global try)
#         x0 = pd.DataFrame([upcoming_row[feats]]).apply(
#             pd.to_numeric, errors="coerce")
#     except Exception:
#         return None, {
#     "error": "Seçilen maç satırında similarity feature erişimi başarısız."}

    x0 = pd.DataFrame([upcoming_row[feats]]).apply(
        pd.to_numeric, errors="coerce")

    if x0.isna().any(axis=1).iloc[0]:
        return None, {
    "error": "Seçilen maç satırında similarity feature'larında NaN var."}

    n_pool_raw = len(h)
    if not same_league:
        max_pool = min(int(max_pool), int(hard_cap_cross_league))
    else:
        max_pool = int(max_pool)

    if n_pool_raw > max_pool:
        h = h.sample(n=max_pool, random_state=42)

    X = h[feats].apply(pd.to_numeric, errors="coerce")
    ok = X.notna().all(axis=1)
    if ok.sum() < max(25, k):
        return None, {
            "error": f"Benzerlik için yeterli temiz satır yok (clean_n={int(ok.sum())})."}

    h = h.loc[ok]
    X = X.loc[ok]

    if len(X) < max(25, k):
        return None, {"error": f"Benzer maç havuzu çok küçük (n={len(X)})."}

    # 1. Önce Standartlaştırma (Scaling)
    sc = StandardScaler()
    Xs = sc.fit_transform(X.values).astype("float32", copy=False)
    x0s = sc.transform(x0.values).astype("float32", copy=False)

    # -----------------------------------------------------------
    # 2. FEATURE WEIGHTING (DÜZELTİLMİŞ - Scaling SONRASI)
    # -----------------------------------------------------------
    col_indices = {name: i for i, name in enumerate(X.columns)}
    for col, w in SIM_FEAT_WEIGHTS.items():
        if col in col_indices:
            idx = col_indices[col]
            Xs[:, idx] *= w
            x0s[:, idx] *= w
    # -----------------------------------------------------------

    nn = NearestNeighbors(
    n_neighbors=min(
        k,
        len(X)),
        metric="euclidean",
         algorithm="brute")
    nn.fit(Xs)
    dist, idx = nn.kneighbors(x0s)

    d = dist[0]
    ids = idx[0]

    neigh = h.iloc[ids].copy()

    neigh["_dist"] = d.astype("float32", copy=False)
    neigh["SimilarityScore"] = (
        1.0 / (1.0 + neigh["_dist"])).astype("float32", copy=False)

    cutoff_applied = False
    if use_cutoff:
        ms = float(min_similarity)
        ms = max(0.0, min(0.99, ms))
        strong = neigh[neigh["SimilarityScore"] >= ms]
        if len(strong) >= int(min_neighbors):
            neigh = strong
            cutoff_applied = True

    d_use = pd.to_numeric(
    neigh.get(
        "_dist",
        pd.Series(
            [],
            dtype="float32")),
             errors="coerce").values
    if d_use is None or len(d_use) == 0:
        return None, {"error": "Komşu listesi boş (cutoff sonrası)."}

    w = (1.0 / (d_use + 1e-6)).astype("float32", copy=False)
    neigh["_w"] = w

    # -----------------------------------------------------------
    # Sprint 5: Time-decay weighting (favor recent seasons)
    # -----------------------------------------------------------
    time_decay_used = False
    half_life_used = None
    age_days_mean = None
    if bool(use_time_decay) and int(half_life_days) > 0:
        up_date_td = pd.to_datetime(
upcoming_row.get(
    "Date", None), errors="coerce")
        neigh_dates = pd.to_datetime(
            neigh.get("Date", None), errors="coerce")
        if pd.notna(
            up_date_td) and neigh_dates is not None and neigh_dates.notna().any():
            age_days = (up_date_td - neigh_dates).dt.days.astype("float32")
            # clamp negatives (shouldn't happen if hygiene is correct)
            age_days = np.where(
np.isfinite(age_days), np.maximum(
    age_days, 0.0), np.nan).astype("float32")
            hl = float(max(1.0, float(half_life_days)))
            w_time = np.exp(-np.nan_to_num(age_days,
 nan=hl * 10.0) / hl).astype("float32")
            w = (w * w_time).astype("float32", copy=False)
            neigh["_w_time"] = w_time
            time_decay_used = True
            half_life_used = int(half_life_days)
            age_days_mean = float(np.nanmean(age_days))

    # re-attach potentially updated weights
    neigh["_w"] = w

    # -----------------------------------------------------------
    # RELIABILITY: Effective sample size (Kish effective N)
    # -----------------------------------------------------------
#     try:  # AUTO-COMMENTED (illegal global try)
#         wsum_eff = float(np.nansum(w))
#         w2sum_eff = float(np.nansum(np.square(w)))
#         effective_n_real = float(
#             (wsum_eff * wsum_eff) / (w2sum_eff + 1e-12)) if w2sum_eff > 0 else float(len(neigh))
#     except Exception:
#         effective_n_real = float(len(neigh))
    effective_n_real = float(len(neigh))

    tr = pd.to_numeric(neigh["Target_Result"],
     errors="coerce").fillna(-1).astype(int).values
    wsum = float(np.nansum(w))
    p1 = float(np.nansum(w[tr == 2]) / wsum) if wsum > 0 else float("nan")
    px = float(np.nansum(w[tr == 1]) / wsum) if wsum > 0 else float("nan")
    p2 = float(np.nansum(w[tr == 0]) / wsum) if wsum > 0 else float("nan")

    out = {
        "SIM_P1": p1,
        "SIM_PX": px,
        "SIM_P2": p2,
        "SIM_N": int(len(neigh)),
        "EFFECTIVE_N": int(round(effective_n_real)),
        "EFFECTIVE_N_REAL": float(effective_n_real),
        "LEAK_OK": bool(leak_ok),
        "FILTERED_FUTURE_N": int(filtered_future_n),
        "FILTERED_SAMEDAY_N": int(filtered_sameday_n),
        "DUP_DROPPED_N": int(dup_dropped_n),
        "CUTOFF_APPLIED": bool(cutoff_applied),
        "MIN_SIM": float(min_similarity) if use_cutoff else float("nan"),
        "POOL_RAW_N": int(n_pool_raw),
        "POOL_USED_N": int(len(h)),
        "POOL_MAX_DATE": str(pool_max_date) if pool_max_date is not None and pd.notna(pool_max_date) else "",
        "SIM_DIST_MED": float(pd.to_numeric(neigh["_dist"], errors="coerce").median()),
        "SIM_QUALITY": float(_safe_quality_from_dist(pd.to_numeric(neigh["_dist"], errors="coerce").values)),
    }
    # -----------------------------------------------------------
    # Sprint 5+6: expose time-decay diagnostics + reliability gate
    # -----------------------------------------------------------
#     try:  # AUTO-COMMENTED (illegal global try)
#         out["TIME_DECAY_USED"] = bool(time_decay_used)
#         out["HALF_LIFE_DAYS"] = int(
#             half_life_used) if half_life_used is not None else int(half_life_days)
#         out["AGE_DAYS_MEAN"] = float(
#             age_days_mean) if age_days_mean is not None else float("nan")
#     except Exception:
#         out["TIME_DECAY_USED"] = False

#     try:  # AUTO-COMMENTED (illegal global try)
#         simq_val = float(out.get("SIM_QUALITY", float("nan")))
#         en_val = float(out.get("EFFECTIVE_N_REAL", out.get("EFFECTIVE_N", 0)))
#         out["GATE_MIN_SIMQ"] = float(gate_min_simq)
#         out["GATE_MIN_EN"] = int(gate_min_en)
#         out["KNN_OK"] = bool(
#     (np.isfinite(simq_val) and simq_val >= float(gate_min_simq)) and (
#         np.isfinite(en_val) and en_val >= float(gate_min_en)))
#     except Exception:
#         out["KNN_OK"] = False

    if "Target_Over" in neigh.columns:
        ov = neigh["Target_Over"].astype(int).values
        out["SIM_POver"] = float(w[ov == 1].sum() /
     wsum) if wsum > 0 else float("nan")
    if "Target_BTTS" in neigh.columns:
        kg = neigh["Target_BTTS"].astype(int).values
        out["SIM_PBTTS"] = float(w[kg == 1].sum() /
     wsum) if wsum > 0 else float("nan")

    # -----------------------------------------------------------
    # Sprint 4: Distribution / Risk Shape from neighbors
    # -----------------------------------------------------------
    # NOTE: We intentionally keep these as simple, robust stats that
    # do not require new labels beyond what is already in the dataset.
#     try:  # AUTO-COMMENTED (illegal global try)
#         # Total goals (preferred) or derive from FT scores if present
#         if "Total_Goals" in neigh.columns:
#             tg = pd.to_numeric(neigh["Total_Goals"], errors="coerce").values
#         elif ("HomeFullTimeScore" in neigh.columns) and ("AwayFullTimeScore" in neigh.columns):
#             tg = (
#     pd.to_numeric(
#         neigh["HomeFullTimeScore"],
#         errors="coerce").fillna(0).values +
#         pd.to_numeric(
#             neigh["AwayFullTimeScore"],
#              errors="coerce").fillna(0).values)
#         else:
#             tg = None
        tg = None

#         if "Total_Goals" in neigh.columns:
#             tg = pd.to_numeric(neigh["Total_Goals"], errors="coerce").values
#         elif ("HomeFullTimeScore" in neigh.columns) and ("AwayFullTimeScore" in neigh.columns):
#             tg = (
#     pd.to_numeric(
#         neigh["HomeFullTimeScore"],
#         errors="coerce").fillna(0).values +
#         pd.to_numeric(
#             neigh["AwayFullTimeScore"],
#              errors="coerce").fillna(0).values)
#         else:
#             tg = None

        # OU tails / dispersion
        if tg is not None and wsum > 0:
            tg_num = pd.to_numeric(pd.Series(tg), errors="coerce").values
            mask = np.isfinite(tg_num) & np.isfinite(w)
            if mask.any():
                ww = w[mask]
                wwsum = float(ww.sum())
                if wwsum > 0:
                    out["SIM_TAIL_3PLUS"] = float(
                        ww[tg_num[mask] >= 3].sum() / wwsum)
                    # unweighted robust IQR/std (weighted quantiles add
                    # complexity; this is enough for gating/UI)
                    tg_clean = tg_num[mask]
                    out["SIM_TG_STD"] = float(np.nanstd(tg_clean))
                    q25, q75 = np.nanpercentile(tg_clean, [25, 75])
                    out["SIM_TG_IQR"] = float(q75 - q25)

        # Binary variance proxies (best-effort, from p where available)
        if "SIM_POver" in out and np.isfinite(out.get("SIM_POver", np.nan)):
            p = float(out["SIM_POver"])
            out["SIM_VAR_POver"] = float(max(0.0, min(0.25, p * (1.0 - p))))
        if "SIM_PBTTS" in out and np.isfinite(out.get("SIM_PBTTS", np.nan)):
            p = float(out["SIM_PBTTS"])
            out["SIM_VAR_PBTTS"] = float(max(0.0, min(0.25, p * (1.0 - p))))

        # BTTS tail: 0-0 share (requires FT scores)
        if ("HomeFullTimeScore" in neigh.columns) and (
            "AwayFullTimeScore" in neigh.columns) and wsum > 0:
            hs = pd.to_numeric(
    neigh["HomeFullTimeScore"],
    errors="coerce").fillna(
        np.nan).values
            as_ = pd.to_numeric(
    neigh["AwayFullTimeScore"],
    errors="coerce").fillna(
        np.nan).values
            mask = np.isfinite(hs) & np.isfinite(as_) & np.isfinite(w)
            if mask.any():
                ww = w[mask]
                wwsum = float(ww.sum())
                if wwsum > 0:
                    out["SIM_TAIL_00"] = float(
                        ww[(hs[mask] == 0) & (as_[mask] == 0)].sum() / wwsum)
#     except Exception:  # AUTO-COMMENTED (orphan except)
#         # Never fail similarity due to risk stats
#         pass

    if ("HomeFullTimeScore" in neigh.columns) and (
        "AwayFullTimeScore" in neigh.columns):
        hs_s = pd.to_numeric(
            neigh.get("HomeFullTimeScore", np.nan), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0).astype(int).astype(str)
        as_s = pd.to_numeric(
            neigh.get("AwayFullTimeScore", np.nan), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0).astype(int).astype(str)
        score_s = hs_s + "-" + as_s
        out["TOP_SCORES"] = score_s.value_counts().head(8).to_dict()

    neigh2 = neigh.sort_values("_dist").head(min(50, len(neigh)))
    front_cols = ["SimilarityScore", "_dist", "_w"]
    front_cols = [c for c in front_cols if c in neigh2.columns]
    if front_cols:
        cols = front_cols + [c for c in neigh2.columns if c not in front_cols]
        neigh2 = neigh2[cols]
    return neigh2, out


def _compute_similarity_priors_batch(
    upcoming_df: pd.DataFrame,
    train_core: pd.DataFrame,
    *,
    k: int = 60,
    use_cutoff: bool = True,
    min_similarity: float = 0.55,
    min_neighbors: int = 12,
    same_league: bool = True,
    years_back: int = 4,
    preset: str = "Auto (mevcut kolonlar)",
    hard_cap_cross_league: int = 120000,
) -> pd.DataFrame:
    if upcoming_df is None or getattr(upcoming_df, "empty", True):
        return pd.DataFrame(index=getattr(upcoming_df, "index", None))
    if train_core is None or getattr(train_core, "empty", True):
        return pd.DataFrame(index=upcoming_df.index)

    if bool(same_league) and (
    'League' in upcoming_df.columns) and (
        'League' in train_core.columns):
        parts = []
        for lg, upg in upcoming_df.groupby('League'):
            trg = train_core[train_core['League'] == lg]
            if trg is None or getattr(trg, 'empty', True):
                continue
            outg = _compute_similarity_priors_batch(
                upg, trg,
                k=int(k),
                use_cutoff=bool(use_cutoff),
                min_similarity=float(min_similarity),
                min_neighbors=int(min_neighbors),
                same_league=False,
                years_back=int(years_back),
                preset=str(preset),
                hard_cap_cross_league=int(hard_cap_cross_league),
            )
            if outg is not None and not outg.empty:
                parts.append(outg)
        if parts:
            return pd.concat(parts, axis=0).reindex(upcoming_df.index)
        return pd.DataFrame(index=upcoming_df.index)

    feats = _pick_similarity_features(
    upcoming_df, train_core, preset=str(preset))
    if not feats:
        return pd.DataFrame(index=upcoming_df.index)

    h = train_core.copy()
    if int(years_back) > 0 and (
    'Date' in h.columns) and (
        'Date' in upcoming_df.columns):
        _u_max = pd.to_datetime(upcoming_df['Date'], errors='coerce').max()
        if pd.notna(_u_max):
            _cut = _u_max - pd.DateOffset(years=int(years_back))
            h['Date'] = pd.to_datetime(h['Date'], errors='coerce')
            h = h[h['Date'] >= _cut].copy()
    for c in feats:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    X = h[feats]
    ok = X.notna().all(axis=1)
    h = h.loc[ok].copy()
    X = X.loc[ok].copy()
    if len(h) < max(25, int(k)):
        return pd.DataFrame(index=upcoming_df.index)

    if len(h) > int(hard_cap_cross_league):
        h = h.sample(n=int(hard_cap_cross_league), random_state=42)
        X = h[feats].copy()

    u = upcoming_df.copy()
    for c in feats:
        u[c] = pd.to_numeric(u[c], errors="coerce")
    Xu = u[feats]
    ok_u = Xu.notna().all(axis=1)

    out = pd.DataFrame(index=upcoming_df.index)
    for c in [
    "SIM_P1",
    "SIM_PX",
    "SIM_P2",
    "SIM_POver",
    "SIM_PBTTS",
    "SIM_N",
    "EFFECTIVE_N",
    "EFFECTIVE_N_REAL",
    "SIM_QUALITY",
    "CUTOFF_APPLIED",
     "MIN_SIM"]:
        out[c] = np.nan
    out.loc[:, "SIM_N"] = 0
    out.loc[:, "EFFECTIVE_N"] = 0
    out.loc[:, "EFFECTIVE_N_REAL"] = 0.0
    out.loc[:, "CUTOFF_APPLIED"] = False
    out.loc[:, "MIN_SIM"] = float(min_similarity) if use_cutoff else np.nan

    if ok_u.sum() == 0:
        return out

    # 1. Önce Standartlaştırma (Scaling)
    sc = StandardScaler()
    Xs = sc.fit_transform(X.values).astype("float32", copy=False)
    Xus = sc.transform(Xu.loc[ok_u].values).astype("float32", copy=False)

    # -----------------------------------------------------------
    # 2. FEATURE WEIGHTING (DÜZELTİLMİŞ - Scaling SONRASI)
    # -----------------------------------------------------------
    col_indices = {name: i for i, name in enumerate(X.columns)}
    for col, w in SIM_FEAT_WEIGHTS.items():
        if col in col_indices:
            idx = col_indices[col]
            Xs[:, idx] *= w
            Xus[:, idx] *= w
    # -----------------------------------------------------------

    nn = NearestNeighbors(
    n_neighbors=min(
        int(k),
        len(X)),
        metric="euclidean",
         algorithm="brute")
    nn.fit(Xs)
    dist, idx = nn.kneighbors(Xus)

    tr = pd.to_numeric(h["Target_Result"],
     errors="coerce").fillna(-1).astype(int).values
    has_ov = "Target_Over" in h.columns
    has_kg = "Target_BTTS" in h.columns
    ov = pd.to_numeric(h["Target_Over"], errors="coerce").fillna(
        0).astype(int).values if has_ov else None
    kg = pd.to_numeric(h["Target_BTTS"], errors="coerce").fillna(
        0).astype(int).values if has_kg else None

    u_idx = list(Xu.loc[ok_u].index)
    for rpos, ridx in enumerate(u_idx):
        d = dist[rpos].astype("float32", copy=False)
        ids = idx[rpos]
        sim = 1.0 / (1.0 + d)

        use_ids = ids
        use_d = d
        use_sim = sim
        cutoff_applied = False
        if use_cutoff:
            ms = float(min_similarity)
            ms = max(0.0, min(0.99, ms))
            mask = (sim >= ms)
            if int(mask.sum()) >= int(min_neighbors):
                use_ids = ids[mask]
                use_d = d[mask]
                use_sim = sim[mask]
                cutoff_applied = True

        # [FIX] RBF (Gaussian) distance weighting (scale-safe)
        d_use = np.asarray(use_d, dtype="float32").flatten()
        d_use = np.clip(d_use, 0.0, None)

        # Sigma (bandwidth) – robust to zeros / NaNs
        sigma = float(np.nanmedian(d_use))
        if sigma <= 1e-5:
            sigma = float(np.nanmean(d_use))
        if sigma <= 1e-5:
            sigma = 1.0

        sigma_mult = float(
st.session_state.get(
    "SIM_RBF_SIGMA_MULT", 0.70))
        sigma = max(1e-3, sigma * sigma_mult)
        sigma = max(1e-3, sigma * sigma_mult)

        power = float(st.session_state.get("SIM_RBF_POWER", 2.0))
        if not np.isfinite(power) or power <= 0:
            power = 2.0

        # w = exp( - (d/sigma)^power )
        w = np.exp(-np.power(d_use / (sigma + 1e-9), power)).astype("float32")

        # Safety fallback: if all weights are ~0, distribute uniformly
        if float(np.nansum(w)) <= 1e-9:
            w = np.ones_like(d_use, dtype="float32") / (len(d_use) + 1e-9)

        # Kish effective N (real)
        w_sum_eff = float(np.sum(w))
        w_sq_sum_eff = float(np.sum(w ** 2))
        if w_sq_sum_eff > 1e-12:
            effective_n_real = (w_sum_eff * w_sum_eff) / w_sq_sum_eff
        else:
            effective_n_real = float(len(w))
        effective_n_real = max(
    1.0, min(
        float(
            len(w)), float(effective_n_real)))

        wsum = float(np.nansum(w))
        if not (wsum > 0):
            continue

        tr_n = tr[use_ids]
        p1 = float(np.nansum(w[tr_n == 2]) / wsum)
        px = float(np.nansum(w[tr_n == 1]) / wsum)
        p2 = float(np.nansum(w[tr_n == 0]) / wsum)

        out.at[ridx, "SIM_P1"] = p1
        out.at[ridx, "SIM_PX"] = px
        out.at[ridx, "SIM_P2"] = p2
        out.at[ridx, "SIM_N"] = int(len(ids))
        out.at[ridx, "EFFECTIVE_N_REAL"] = float(effective_n_real)
        out.at[ridx, "EFFECTIVE_N"] = float(effective_n_real)
        out.at[ridx, "CUTOFF_APPLIED"] = bool(cutoff_applied)

        out.at[ridx, "SIM_QUALITY"] = float(_safe_quality_from_dist(use_d))

        if has_ov:
            ov_n = ov[use_ids]
            out.at[ridx, "SIM_POver"] = float(np.nansum(w[ov_n == 1]) / wsum)
        if has_kg:
            kg_n = kg[use_ids]
            out.at[ridx, "SIM_PBTTS"] = float(np.nansum(w[kg_n == 1]) / wsum)

    return out


def apply_similarity_anchor_to_predictions(
    pred_df: pd.DataFrame,
    train_core: pd.DataFrame,
    *,
    alpha_max: float = 0.60,
    k: int = 60,
    use_cutoff: bool = True,
    min_similarity: float = 0.55,
    min_neighbors: int = 12,
    same_league: bool = True,
    max_pool: int = 120000,
    years_back: int = 4,
    preset: str = "Auto (mevcut kolonlar)",
    k_ref: float = 30.0,
) -> pd.DataFrame:
    """Blend model probabilities with similarity priors (softly, market-agnostic).

    Keeps originals in *_Model columns and overwrites the *_Final columns (so the rest of the app
    automatically benefits without further refactors).
    """
    sim_df = None
    def _num0(x):
        if isinstance(x, pd.Series):
            return pd.to_numeric(x, errors="coerce").fillna(0.0)
        try:
            v = pd.to_numeric(x, errors="coerce")
        except Exception:
            return 0.0
        return 0.0 if pd.isna(v) else float(v)

    if pred_df is None or getattr(pred_df, "empty", True):
        return pred_df

    d = pred_df.copy()

    # v82: Dual (League-first + Global fallback) + Market-family presets
    if preset == "Dual (Lig+Global, MS/OB)":
        ms_preset = "MS odaklı (Elo+Form+xGD)"
        ou_preset = "OU odaklı (xG+Book)"
        btts_preset = "BTTS odaklı (xG+Tempo)"

        # --- MS (1X2 / taraf) ---
        ms_league = _compute_similarity_priors_batch(
    d,
    train_core,
    k=int(k),
    use_cutoff=bool(use_cutoff),
    min_similarity=float(min_similarity),
    min_neighbors=int(min_neighbors),
    same_league=True,
    years_back=int(years_back),
    preset=ms_preset,
    hard_cap_cross_league=int(max_pool),
     )
        ms_global = _compute_similarity_priors_batch(
    d,
    train_core,
    k=int(k),
    use_cutoff=bool(use_cutoff),
    min_similarity=float(min_similarity),
    min_neighbors=int(min_neighbors),
    same_league=False,
    years_back=int(years_back),
    preset=ms_preset,
    hard_cap_cross_league=int(max_pool),
     )

        # --- OU (Üst/Alt) ---
        ou_league = _compute_similarity_priors_batch(
    d,
    train_core,
    k=int(k),
    use_cutoff=bool(use_cutoff),
    min_similarity=float(min_similarity),
    min_neighbors=int(min_neighbors),
    same_league=True,
    years_back=int(years_back),
    preset=ou_preset,
    hard_cap_cross_league=int(max_pool),
     )
        ou_global = _compute_similarity_priors_batch(
    d,
    train_core,
    k=int(k),
    use_cutoff=bool(use_cutoff),
    min_similarity=float(min_similarity),
    min_neighbors=int(min_neighbors),
    same_league=False,
    years_back=int(years_back),
    preset=ou_preset,
    hard_cap_cross_league=int(max_pool),
     )

        # --- BTTS (KG Var/Yok) ---
        btts_league = _compute_similarity_priors_batch(
    d,
    train_core,
    k=int(k),
    use_cutoff=bool(use_cutoff),
    min_similarity=float(min_similarity),
    min_neighbors=int(min_neighbors),
    same_league=True,
    years_back=int(years_back),
    preset=btts_preset,
    hard_cap_cross_league=int(max_pool),
     )
        btts_global = _compute_similarity_priors_batch(
    d,
    train_core,
    k=int(k),
    use_cutoff=bool(use_cutoff),
    min_similarity=float(min_similarity),
    min_neighbors=int(min_neighbors),
    same_league=False,
    years_back=int(years_back),
    preset=btts_preset,
    hard_cap_cross_league=int(max_pool),
     )

        def _w(n, effn):
            n = pd.to_numeric(n, errors="coerce").fillna(0)
            effn = pd.to_numeric(effn, errors="coerce").fillna(0)
            ok = (n >= 20) & (effn >= 25)
            exists = (n > 0) & (effn > 0)
            wL = pd.Series(0.0, index=d.index)
            wG = pd.Series(1.0, index=d.index)
            mode = pd.Series("GLOBAL_ONLY", index=d.index, dtype="object")
            wL.loc[exists] = 0.35
            wG.loc[exists] = 0.65
            mode.loc[exists] = "GLOBAL_DOM"
            wL.loc[ok] = 0.75
            wG.loc[ok] = 0.25
            mode.loc[ok] = "LEAGUE_DOM"
            return ok, wL, wG, mode

        ok_ms, wL_ms, wG_ms, mode_ms = _w(
    ms_league.get(
        "SIM_N", 0), ms_league.get(
            "EFFECTIVE_N", 0))
        ok_ou, wL_ou, wG_ou, mode_ou = _w(
    ou_league.get(
        "SIM_N", 0), ou_league.get(
            "EFFECTIVE_N", 0))
        ok_btts, wL_btts, wG_btts, mode_btts = _w(
    btts_league.get(
        "SIM_N", 0), btts_league.get(
            "EFFECTIVE_N", 0))

        def _blend_col(col, a, b, wL, wG):
            aa = pd.to_numeric(a.get(col, 0), errors="coerce")
            if not isinstance(aa, pd.Series):
                aa = pd.Series(aa, index=wL.index)
            aa = aa.fillna(0)
            bb = pd.to_numeric(b.get(col, 0), errors="coerce")
            if not isinstance(bb, pd.Series):
                bb = pd.Series(bb, index=wL.index)
            bb = bb.fillna(0)
            return (wL * aa + wG * bb)

        # ---------------------------------------------------------------------
        # Market-aware priors (league/global blended)
        # ---------------------------------------------------------------------
        # MS priors
        d["SIM_P1"] = _blend_col("SIM_P1", ms_league, ms_global, wL_ms, wG_ms)
        d["SIM_PX"] = _blend_col("SIM_PX", ms_league, ms_global, wL_ms, wG_ms)
        d["SIM_P2"] = _blend_col("SIM_P2", ms_league, ms_global, wL_ms, wG_ms)

        # OU priors (only from OU neighbor set)
        d["SIM_POver"] = _blend_col(
    "SIM_POver", ou_league, ou_global, wL_ou, wG_ou)

        # BTTS priors (only from BTTS neighbor set)
        d["SIM_PBTTS"] = _blend_col(
    "SIM_PBTTS",
    btts_league,
    btts_global,
    wL_btts,
     wG_btts)

        # Market-aware qualities (keep both + a market-chosen SIM_QUALITY)
        q_ms = _blend_col("SIM_QUALITY", ms_league, ms_global, wL_ms, wG_ms)
        q_ou = _blend_col("SIM_QUALITY", ou_league, ou_global, wL_ou, wG_ou)
        q_btts = _blend_col(
    "SIM_QUALITY",
    btts_league,
    btts_global,
    wL_btts,
     wG_btts)

        # Alpha (use the strongest signal among the market priors; still
        # clipped later)
        a_ms = _blend_col("SIM_ALPHA", ms_league, ms_global, wL_ms, wG_ms)
        a_ou = _blend_col("SIM_ALPHA", ou_league, ou_global, wL_ou, wG_ou)
        a_btts = _blend_col(
    "SIM_ALPHA",
    btts_league,
    btts_global,
    wL_btts,
     wG_btts)

        d["SIM_QUALITY_MS"] = q_ms
        d["SIM_QUALITY_OU"] = q_ou
        d["SIM_QUALITY_BTTS"] = q_btts

        # -----------------------------------------------------------
        # Sprint 4: Market-aware risk / dispersion columns (best-effort)
        # -----------------------------------------------------------
        # OU risk stats
        d["SIM_TAIL_3PLUS_OU"] = _blend_col(
    "SIM_TAIL_3PLUS", ou_league, ou_global, wL_ou, wG_ou)
        d["SIM_TG_STD_OU"] = _blend_col(
    "SIM_TG_STD", ou_league, ou_global, wL_ou, wG_ou)
        d["SIM_TG_IQR_OU"] = _blend_col(
    "SIM_TG_IQR", ou_league, ou_global, wL_ou, wG_ou)
        d["SIM_VAR_POver_OU"] = _blend_col(
    "SIM_VAR_POver", ou_league, ou_global, wL_ou, wG_ou)

        # BTTS risk stats
        d["SIM_TAIL_00_BTTS"] = _blend_col(
    "SIM_TAIL_00", btts_league, btts_global, wL_btts, wG_btts)
        d["SIM_VAR_PBTTS_BTTS"] = _blend_col(
    "SIM_VAR_PBTTS", btts_league, btts_global, wL_btts, wG_btts)

        # Simple categorical risk (for UI/gating later)
        def _risk_class(sim_q, eff_n, var_p, tg_iqr):
            sq = float(pd.to_numeric(sim_q, errors="coerce"))
            en = float(pd.to_numeric(eff_n, errors="coerce"))
            vp = float(pd.to_numeric(var_p, errors="coerce"))
            iqr = float(pd.to_numeric(tg_iqr, errors="coerce"))
            # Conservative: low data or low quality -> WILD
            if (en > 0 and en < 6) or (sq > 0 and sq < 0.28):
                return "WILD"
            # High uncertainty in binary proxy or very wide goal IQR -> WILD
            if (vp > 0.235) or (iqr > 2.0):
                return "WILD"
            # Good quality + decent effective N + tight dispersion -> TIGHT
            if (sq >= 0.38) and (en >= 10) and (
                (vp > 0 and vp <= 0.21) or (iqr >= 0 and iqr <= 1.0)):
                return "TIGHT"
            return "NORMAL"

        def _col_series(df, col, default=np.nan):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
            return pd.Series([default] * len(df), index=df.index, dtype="float")

        # Market-specific risk classes
        d["SIM_RISK_OU"] = [
    _risk_class(
        q,
        n,
        v,
        i) for q,
        n,
        v,
        i in zip(
            _col_series(d, "SIM_QUALITY_OU"),
            _col_series(d, "EFFECTIVE_N_OU"),
            _col_series(d, "SIM_VAR_POver_OU"),
             _col_series(d, "SIM_TG_IQR_OU"))]
        d["SIM_RISK_BTTS"] = [
            _risk_class(q, n, v, 1.0)  # BTTS has no TG_IQR; pass neutral
            for q, n, v in zip(
                _col_series(d, "SIM_QUALITY_BTTS"),
                _col_series(d, "EFFECTIVE_N_BTTS"),
                _col_series(d, "SIM_VAR_PBTTS_BTTS"))
        ]

        # Pick SIM_QUALITY by selection market (so BANKO logic makes sense)
        sel = d.get("Seçim", "")
        sel_s = sel.astype(str).str.lower() if hasattr(
            sel, "astype") else pd.Series([""] * len(d))
        is_ou = sel_s.str.contains("alt|üst|over|under", regex=True)
        is_btts = sel_s.str.contains("kg|btts", regex=True)
        d["SIM_QUALITY"] = np.select(
            [is_btts, is_ou],
            [q_btts, q_ou],
            default=q_ms
        )

        d["SIM_ALPHA"] = np.clip(
            pd.concat([a_ms, a_ou, a_btts], axis=1).max(axis=1), 0.0, float(alpha_max))

        # Core neighbor counts (keep conservative max so UI/debug doesn't look
        # empty)
        d["SIM_N"] = pd.concat([
            pd.to_numeric(
    ms_global.get(
        "SIM_N",
        0),
         errors="coerce").fillna(0),
            pd.to_numeric(
    ou_global.get(
        "SIM_N",
        0),
         errors="coerce").fillna(0),
            pd.to_numeric(
    btts_global.get(
        "SIM_N",
        0),
         errors="coerce").fillna(0)
        ], axis=1).max(axis=1)

        d["EFFECTIVE_N"] = pd.concat([
            pd.to_numeric(
    ms_global.get(
        "EFFECTIVE_N",
        0),
         errors="coerce").fillna(0),
            pd.to_numeric(
    ou_global.get(
        "EFFECTIVE_N",
        0),
         errors="coerce").fillna(0),
            pd.to_numeric(
    btts_global.get(
        "EFFECTIVE_N",
        0),
         errors="coerce").fillna(0)
        ], axis=1).max(axis=1)

        # Market-specific effective N (used for risk & later gating)
        d["EFFECTIVE_N_MS"] = pd.to_numeric(ms_global.get(
            "EFFECTIVE_N", 0), errors="coerce").fillna(0)
        d["EFFECTIVE_N_OU"] = pd.to_numeric(ou_global.get(
            "EFFECTIVE_N", 0), errors="coerce").fillna(0)
        d["EFFECTIVE_N_BTTS"] = pd.to_numeric(btts_global.get(
            "EFFECTIVE_N", 0), errors="coerce").fillna(0)

        # Debug / proof columns

        d["LEAGUE_OK_MS"] = ok_ms
        d["LEAGUE_OK_OU"] = ok_ou
        d["LEAGUE_OK_BTTS"] = ok_btts
        # Backward-compatible OB flag (any of OU/BTTS okay)
        d["LEAGUE_OK_OB"] = (ok_ou | ok_btts)

        d["BLEND_MODE_MS"] = mode_ms
        d["BLEND_MODE_OU"] = mode_ou
        d["BLEND_MODE_BTTS"] = mode_btts
        # Backward-compatible OB mode
        d["BLEND_MODE_OB"] = mode_ou

        d["BLEND_W_LEAGUE_MS"] = wL_ms
        d["BLEND_W_GLOBAL_MS"] = wG_ms
        d["BLEND_W_LEAGUE_OU"] = wL_ou
        d["BLEND_W_GLOBAL_OU"] = wG_ou
        d["BLEND_W_LEAGUE_BTTS"] = wL_btts
        d["BLEND_W_GLOBAL_BTTS"] = wG_btts
        # Backward-compatible OB weights = OU weights
        d["BLEND_W_LEAGUE_OB"] = wL_ou
        d["BLEND_W_GLOBAL_OB"] = wG_ou
        val = pd.to_numeric(ms_league.get("SIM_P2", 0), errors="coerce")
        d["MS_LEAGUE_SIM_P2"] = 0.0 if pd.isna(val) else float(val)
        val = pd.to_numeric(ms_global.get("SIM_P2", 0), errors="coerce")
        d["MS_GLOBAL_SIM_P2"] = 0.0 if pd.isna(val) else float(val)
        val = pd.to_numeric(ou_league.get("SIM_POver", 0), errors="coerce")
        d["OB_LEAGUE_SIM_POver"] = 0.0 if pd.isna(val) else float(val)
        val = pd.to_numeric(ou_global.get("SIM_POver", 0), errors="coerce")
        d["OB_GLOBAL_SIM_POver"] = 0.0 if pd.isna(val) else float(val)

        # Continue with model-prior blending below using the now-correct SIM_*
        # priors.

    # Compute priors (fast, batch)
#     try:  # AUTO-COMMENTED (illegal global try)
#         sim_df = _compute_similarity_priors_batch(
#             d, train_core,
#             k=int(k),
#             use_cutoff=bool(use_cutoff),
#             min_similarity=float(min_similarity),
#             min_neighbors=int(min_neighbors),
#             same_league=bool(same_league),
#             hard_cap_cross_league=int(max_pool),
#             years_back=int(years_back),
#             preset=str(preset),
#         )
#     except Exception:
#         sim_df = pd.DataFrame(index=d.index)

    if sim_df is not None and not sim_df.empty:
        # Align indices to avoid assignment mismatches
        sim_df = sim_df.reindex(d.index)

        for c in sim_df.columns:
            # Critical columns: force overwrite (remove the K=60 stickiness)
            if c in [
    "EFFECTIVE_N",
    "EFFECTIVE_N_REAL",
    "SIM_QUALITY",
    "SIM_P1",
    "SIM_PX",
    "SIM_P2",
    "SIM_POver",
     "SIM_PBTTS"]:
                d[c] = sim_df[c]
            else:
                # Non-critical: keep old behavior (fill only if missing)
                if c not in d.columns:
                    d[c] = sim_df[c]
                else:
                    d[c] = d[c].fillna(sim_df[c])

    # -----------------------------
    # Similarity-derived ANCHOR MARKET inference (match-level)
    # Purpose: decide which market (MS / OU / BTTS) similarity signal is strongest in,
    # so downstream ranking can prefer that market for this match.
    # -----------------------------
#     try:  # AUTO-COMMENTED (illegal global try)
#         # Raw similarity priors (0..1)
#         sp1 = pd.to_numeric(d.get("SIM_P1", np.nan), errors="coerce")
#         spx = pd.to_numeric(d.get("SIM_PX", np.nan), errors="coerce")
#         sp2 = pd.to_numeric(d.get("SIM_P2", np.nan), errors="coerce")
#         spo = pd.to_numeric(d.get("SIM_POver", np.nan), errors="coerce")
#         spb = pd.to_numeric(d.get("SIM_PBTTS", np.nan), errors="coerce")
        sp1 = pd.to_numeric(d.get("SIM_P1", np.nan), errors="coerce")
        spx = pd.to_numeric(d.get("SIM_PX", np.nan), errors="coerce")
        sp2 = pd.to_numeric(d.get("SIM_P2", np.nan), errors="coerce")
        spo = pd.to_numeric(d.get("SIM_POver", np.nan), errors="coerce")
        spb = pd.to_numeric(d.get("SIM_PBTTS", np.nan), errors="coerce")

        # Strength definitions (0..1)
        # MS: distance from uniform baseline (1/3,1/3,1/3) using total-variation style L1 distance
        # This makes MS comparable to OU/BTTS (both measured as distance from a
        # neutral baseline).
        u = (1.0 / 3.0)
        ms_tv = 0.5 * ((sp1 - u).abs() + (spx - u).abs() +
                       (sp2 - u).abs())  # range: 0 .. 2/3
        ms_strength = (ms_tv / (2.0 / 3.0)).clip(0.0, 1.0)

        # OU / BTTS: distance from 0.5 (scaled to 0..1)
        ou_strength = ((spo - 0.5).abs() * 2.0).clip(0.0, 1.0)
        btts_strength = ((spb - 0.5).abs() * 2.0).clip(0.0, 1.0)

        d["SIM_MS_STRENGTH"] = ms_strength.fillna(0.0)
        d["SIM_OU_STRENGTH"] = ou_strength.fillna(0.0)
        d["SIM_BTTS_STRENGTH"] = btts_strength.fillna(0.0)

        # Pick anchor group by strongest signal
        strength_mat = pd.concat(
            [d["SIM_MS_STRENGTH"], d["SIM_OU_STRENGTH"], d["SIM_BTTS_STRENGTH"]],
            axis=1
        )
        strength_mat.columns = ["MS", "OU", "BTTS"]
        d["SIM_ANCHOR_GROUP"] = strength_mat.idxmax(axis=1).astype(str)
        d["SIM_ANCHOR_STRENGTH_RAW"] = strength_mat.max(
            axis=1).fillna(0.0).clip(0.0, 1.0)

#     except Exception:  # AUTO-COMMENTED (orphan except)
#         d["SIM_MS_STRENGTH"] = 0.0
#         d["SIM_OU_STRENGTH"] = 0.0
#         d["SIM_BTTS_STRENGTH"] = 0.0
#         d["SIM_ANCHOR_GROUP"] = "MS"
#         d["SIM_ANCHOR_STRENGTH_RAW"] = 0.0

# Determine anchor strength per row
    q = _num0(d.get("SIM_QUALITY", pd.Series(0.0, index=d.index)))
    n = _num0(d.get("EFFECTIVE_N", d.get(
        "SIM_N", pd.Series(0.0, index=d.index))))
    s = (q.clip(0, 1) * (np.log1p(n.clip(lower=0)) /
         np.log1p(max(1.0, float(k_ref))))).clip(0, 1)

    # Optionally modulate with existing confidence (if available)
    if "Final_Confidence" in d.columns:
        fc = pd.to_numeric(
    d["Final_Confidence"],
    errors="coerce").fillna(0.5).clip(
        0,
         1)
        s = (s * (0.60 + 0.40 * fc)).clip(0, 1)

    alpha = (float(alpha_max) * s).clip(0.0, float(alpha_max))
    d["SIM_ANCHOR_STRENGTH"] = s
    d["SIM_ALPHA"] = alpha

    # 1X2
    for col, sim_col in [("P_Home_Final", "SIM_P1"),
                          ("P_Draw_Final", "SIM_PX"), ("P_Away_Final", "SIM_P2")]:
        if col in d.columns and sim_col in d.columns:
            model_col = col.replace("_Final", "_Model")
            # Preserve original model output across reruns
            if model_col not in d.columns:
                d[model_col] = d[col]
            m = pd.to_numeric(d[model_col], errors="coerce")
            sp = pd.to_numeric(d[sim_col], errors="coerce")
            d[col] = ((1.0 - alpha) * m + alpha * sp).clip(0.0, 1.0)

    # OU (Over prob)
    if "P_Over_Final" in d.columns and "SIM_POver" in d.columns:
        # Preserve original model output across reruns
        if "P_Over_Model" not in d.columns:
            d["P_Over_Model"] = d["P_Over_Final"]
        m = pd.to_numeric(d["P_Over_Model"], errors="coerce")
        sp = pd.to_numeric(d["SIM_POver"], errors="coerce")
        d["P_Over_Final"] = ((1.0 - alpha) * m + alpha * sp).clip(0.0, 1.0)

    # BTTS
    if "P_BTTS_Final" in d.columns and "SIM_PBTTS" in d.columns:
        # Preserve original model output across reruns
        if "P_BTTS_Model" not in d.columns:
            d["P_BTTS_Model"] = d["P_BTTS_Final"]
        m = pd.to_numeric(d["P_BTTS_Model"], errors="coerce")
        sp = pd.to_numeric(d["SIM_PBTTS"], errors="coerce")
        d["P_BTTS_Final"] = ((1.0 - alpha) * m + alpha * sp).clip(0.0, 1.0)

    return d
    # Final anchor strength used by ranking: quality-weighted and
    # alpha-weighted (0..1)
#     try:  # AUTO-COMMENTED (illegal global try)
#         q = pd.to_numeric(
#     d.get(
#         "SIM_QUALITY",
#         0.0),
#         errors="coerce").fillna(0.0).clip(
#             0.0,
#              1.0)
#         raw = pd.to_numeric(
#     d.get(
#         "SIM_ANCHOR_STRENGTH_RAW",
#         0.0),
#         errors="coerce").fillna(0.0).clip(
#             0.0,
#              1.0)
#         a = pd.to_numeric(alpha, errors="coerce").fillna(0.0).clip(0.0, 1.0)
#         d["SIM_ANCHOR_STRENGTH"] = (
#             raw * q * (0.5 + 0.5 * (a / max(1e-9, float(alpha_max))))).clip(0.0, 1.0)
#     except Exception:
#         d["SIM_ANCHOR_STRENGTH"] = 0.0


# -----------------------------------------------------------------------------#
# 6. UI & LOGIC
# -----------------------------------------------------------------------------#
st.sidebar.header("1. Veri Yükleme")

mode = st.sidebar.radio("Eğitim Modu",
    ["🤖 Auto (Pazar Bazlı)",
    "⚡ Fast Mode (Hızlı Test)",
    "🧠 Full Mode (Detaylı Eğitim)"],
     index=0)
AUTO_MODE = mode.startswith("🤖")
FAST_MODE = mode.startswith("⚡")
st.session_state["AUTO_MODE"] = AUTO_MODE
st.session_state["FAST_MODE"] = FAST_MODE

uploaded_past = st.sidebar.file_uploader(
    "XGPast.csv", type=["csv"], key="past")
uploaded_future = st.sidebar.file_uploader(
    "XGfuture.csv", type=["csv"], key="future")

def _safe_uploader_bytes(upl, key: str):
    """
    upl: st.file_uploader return (UploadedFile or None)
    key: session_state bytes key (e.g., "_past_bytes", "_future_bytes")
    """
    prev = st.session_state.get(key, None)
    if upl is None:
        # IMPORTANT: do NOT overwrite with b""
        return prev
    try:
        data = upl.getvalue()
    except Exception:
        # fallback: keep previous if read fails
        return prev
    if data is None or len(data) == 0:
        # IMPORTANT: do NOT overwrite with empty
        return prev
    # accept new non-empty bytes
    st.session_state[key] = data
    return data

for _k in ["_dbg_missing_cols_pool", "_dbg_missing_cols_bestof"]:
    st.session_state[_k] = []
if "_dbg_missing_cols" in st.session_state:
    del st.session_state["_dbg_missing_cols"]
with st.sidebar.expander("🧯 View Build Errors", expanded=False):
    err = st.session_state.get("_last_view_error")
    where = st.session_state.get("_last_view_error_where")
    if err:
        st.code(f"{where}\n\n{err}" if where else err)
with st.sidebar.expander("🧪 View Debug", expanded=False):
    st.checkbox(
        "Enable column audit + lineage (slow)",
        value=bool(st.session_state.get(DBG_COL_AUDIT_KEY, False)),
        key=DBG_COL_AUDIT_KEY,
    )
    st.write({
        "pool_bets_rows": st.session_state.get("_dbg_pool_bets_rows", 0),
        "pool_view_rows_pre_style": st.session_state.get("_dbg_pool_view_rows_pre_style", 0),
        "pool_view_rows_post_style": st.session_state.get("_dbg_pool_view_rows_post_style", 0),
        "bestof_rows": st.session_state.get("_dbg_bestof_rows", 0),
        "bestof_view_rows_pre_style": st.session_state.get("_dbg_best_view_rows_pre_style", 0),
        "bestof_view_rows_post_style": st.session_state.get("_dbg_best_view_rows_post_style", 0),
        "uploaded_past_present": st.session_state.get("_dbg_uploaded_past_present", False),
        "uploaded_future_present": st.session_state.get("_dbg_uploaded_future_present", False),
        "past_bytes_len": st.session_state.get("_dbg_past_bytes_len", 0),
        "future_bytes_len": st.session_state.get("_dbg_future_bytes_len", 0),
        "past_df_rows": st.session_state.get("_dbg_past_df_rows", 0),
        "future_df_rows": st.session_state.get("_dbg_future_df_rows", 0),
        "pred_df_rows": st.session_state.get("_dbg_pred_df_rows", 0),
        "future_df_rows_ss": st.session_state.get("_dbg_future_df_rows_ss", 0),
        "bets_df_rows": st.session_state.get("_dbg_bets_df_rows", 0),
        "runtime_marker": st.session_state.get("_dbg_runtime_marker"),
    })
    st.write({
        "pool_view_cols_pre_style": st.session_state.get("_dbg_pool_view_cols_pre_style", []),
        "bestof_view_cols_pre_style": st.session_state.get("_dbg_best_view_cols_pre_style", []),
        "missing_cols_pool": st.session_state.get("_dbg_missing_cols_pool", []),
        "missing_cols_bestof": st.session_state.get("_dbg_missing_cols_bestof", []),
        "required_cols": REQUIRED_VIEW_COLS,
    })
    st.subheader("🧪 Column Audit Report")
    _audit = st.session_state.get("_dbg_col_audits", {})
    if _audit:
        st.json(_audit)
    else:
        st.caption("Column audit is empty. Enable it and rerun the pipeline.")
    st.subheader("🧪 Column Lineage (delta only)")
    _lineage = st.session_state.get("_dbg_col_lineage", [])
    if _lineage:
        _lineage_summary = []
        for _e in _lineage:
            _lineage_summary.append({
                "object": _e.get("object"),
                "stage": _e.get("stage"),
                "rows": _e.get("rows"),
                "cols": _e.get("cols_total"),
                "added": len(_e.get("added", [])),
                "removed": len(_e.get("removed", [])),
                "all_nan": _e.get("all_nan_tracked", []),
                "became_all_nan": _e.get("became_all_nan", []),
                "disappeared": _e.get("disappeared", []),
            })
        st.dataframe(pd.DataFrame(_lineage_summary))
        st.json(_lineage)
        _col_stage = {c: {"first_seen": None, "first_all_nan": None, "first_disappeared": None} for c in ADV_COLS_TRACKED}
        for _e in _lineage:
            _obj_stage = f"{_e.get('object')}::{_e.get('stage')}"
            for _c in _e.get("added", []):
                if _c in _col_stage and _col_stage[_c]["first_seen"] is None:
                    _col_stage[_c]["first_seen"] = _obj_stage
            for _c in _e.get("all_nan_tracked", []):
                if _c in _col_stage and _col_stage[_c]["first_all_nan"] is None:
                    _col_stage[_c]["first_all_nan"] = _obj_stage
            for _c in _e.get("disappeared", []):
                if _c in _col_stage and _col_stage[_c]["first_disappeared"] is None:
                    _col_stage[_c]["first_disappeared"] = _obj_stage
        st.subheader("🧪 Column Lineage Summary")
        st.dataframe(pd.DataFrame([
            {"column": k, **v} for k, v in _col_stage.items()
        ]))
    else:
        st.caption("Lineage is empty. Enable audit and rerun the pipeline.")
    st.subheader("🧪 Necessity Buckets (heuristic)")
    _bucket_rows = []
    for _bucket, _items in NECESSITY_BUCKETS.items():
        for _col, _why in _items.items():
            _bucket_rows.append({"bucket": _bucket, "column": _col, "reason": _why})
    if _bucket_rows:
        st.dataframe(pd.DataFrame(_bucket_rows))
    st.subheader("🧪 Pool Stage Debug")
    st.json(st.session_state.get("_dbg_pool_stages", {}))
    st.subheader("🧪 Pool Thresholds")
    st.json(st.session_state.get("_dbg_pool_thresholds", {}))
    st.subheader("🧪 Stage 11 Debug")
    st.json(st.session_state.get("_dbg_stage11", {}))
    st.json(st.session_state.get("_dbg_stage11_drop_counts", {}))
    st.json(st.session_state.get("_dbg_stage11_drop_sample", []))
with st.sidebar.expander("🧊 Cache Health", expanded=False):
    st.write({
        "cwd": os.getcwd(),
        "__file__": __file__ if "__file__" in globals() else "NONE",
    })
    st.json(st.session_state.get("_dbg_file_hashes", {}))
    st.json(st.session_state.get("_timings", {}))
    st.json(st.session_state.get("_cache_hit_flags", {}))
    _last_train_err = st.session_state.get("_last_train_error")
    if _last_train_err:
        st.code(_last_train_err)
    else:
        st.write("None / No error recorded")
    _train_err = st.session_state.get("_train_error")
    if _train_err:
        st.code(_train_err)

if uploaded_past and uploaded_future:
    use_disk_cache = st.sidebar.checkbox(
    "💾 Disk cache kullan (Full Mode açılışını ciddi hızlandırır)", value=True)
    if st.sidebar.button("Analizi Başlat"):
        # Clear prior run state
        for key in [
    'final_df',
    'hist_df',
    'train_core',
    'processed',
    'league_metrics_df',
    'league_confidence',
    'market_quality',
    'walkforward_df',
    'market_threshold_stats',
    'feature_importance_df',
    'opt_params',
     'ev_boost_map',
     '_last_train_error',
     '_train_error',
     '_last_view_error']:
            if key in st.session_state:
                del st.session_state[key]
        if _dbg_col_audit_enabled():
            st.session_state["_dbg_col_lineage"] = []
            st.session_state["_dbg_col_lineage_state"] = {}
            st.session_state["_dbg_col_audits"] = {}

        past_bytes = _safe_uploader_bytes(uploaded_past, "_past_bytes")
        future_bytes = _safe_uploader_bytes(uploaded_future, "_future_bytes")

        st.session_state["_dbg_uploaded_past_present"] = bool(past_bytes) and len(past_bytes) > 0
        st.session_state["_dbg_uploaded_future_present"] = bool(future_bytes) and len(future_bytes) > 0
        st.session_state["_dbg_past_bytes_len"] = int(len(past_bytes)) if past_bytes else 0
        st.session_state["_dbg_future_bytes_len"] = int(len(future_bytes)) if future_bytes else 0
        if not past_bytes or len(past_bytes) == 0 or not future_bytes or len(future_bytes) == 0:
            st.session_state["_last_view_error"] = "Upload bytes empty: skipping train/predict."
            st.sidebar.error("Upload bytes empty: skipping train/predict.")
            st.stop()
        _startup_status = st.status("Initializing pipeline...", expanded=True) if hasattr(st, "status") else None
        if _startup_status:
            _startup_status.write("?? Files received")
        else:
            st.info("? Initializing? please wait")

        def _startup_mark(msg: str) -> None:
            if _startup_status:
                _startup_status.write(msg)

        def _startup_done() -> None:
            if _startup_status:
                _startup_status.write("??? Building views")
                _startup_status.update(label="? Ready", state="complete")
            else:
                st.success("Ready")

        try:
            past_md5 = _md5_bytes(past_bytes)
            future_md5 = _md5_bytes(future_bytes)
            st.session_state["_dbg_file_hashes"] = {
                "past_sha1": _sha1_bytes(past_bytes),
                "future_sha1": _sha1_bytes(future_bytes),
                "past_len": len(past_bytes or b""),
                "future_len": len(future_bytes or b""),
                "fast_mode": bool(st.session_state.get("FAST_MODE", False)),
            }
            st.session_state.setdefault("_timings", {})
            st.session_state.setdefault("_cache_hit_flags", {})

            # Stable disk-cache key: depends on DATA + a small set of
            # model-critical flags (not code version)
            critical_flags = {
        "league_norm": bool(
            st.session_state.get(
                "LEAGUE_NORM", False)), "time_decay": bool(
                    st.session_state.get(
                        "TIME_DECAY", False)), "use_calibration": bool(
                            st.session_state.get(
                                "USE_CALIBRATION", True)), "use_archetype": bool(
                                    st.session_state.get(
                                        "USE_ARCHETYPE", True)), }

            mode_str = "FAST" if st.session_state["FAST_MODE"] else "FULL"
            cache_key = build_stable_cache_key(
        past_md5, future_md5, mode_str, critical_flags)
            cache_key_fast = build_stable_cache_key(
        past_md5, future_md5, "FAST", critical_flags)
            cache_key_full = build_stable_cache_key(
        past_md5, future_md5, "FULL", critical_flags)

            # -------------------------------
            # Disk cache kontrolü (UI)
            # -------------------------------
            if "use_disk" not in st.session_state:
                st.session_state["use_disk"] = True
            use_disk = st.sidebar.checkbox(
        "💾 Disk cache kullan",
         value=st.session_state["use_disk"])
            st.session_state["use_disk"] = use_disk

            cached = load_disk_cache(cache_key) if use_disk else None
            cached_fast = load_disk_cache(cache_key_fast) if use_disk else None
            cached_full = load_disk_cache(cache_key_full) if use_disk else None
            if st.session_state.get("AUTO_MODE", False):
                # AUTO: load both FAST & FULL if available, otherwise compute both
                # once
                if cached_full is not None and cached_fast is not None:
                    st.success(
                        "✅ AUTO: FAST + FULL disk cache yüklendi (eğitim çalıştırılmadan).")
                    full_final = cached_full.get("final_df")
                    fast_final = cached_fast.get("final_df")

                    st.session_state.final_df = combine_fast_full_predictions(
                        full_final, fast_final)
                    st.session_state.hist_df = cached_full.get(
                        "hist_df")  # baseline: FULL
                    st.session_state.train_core = cached_full.get("train_core")
                    st.session_state.processed = True
                    if _dbg_col_audit_enabled() and isinstance(st.session_state.train_core, pd.DataFrame):
                        st.session_state.setdefault("_dbg_pool_stages", {})
                        st.session_state["_dbg_pool_stages"]["train_core_rows"] = int(
                            len(st.session_state.train_core))
                        st.session_state["_dbg_pool_stages"]["train_core_cols"] = int(
                            len(st.session_state.train_core.columns))

                    st.session_state.market_threshold_stats = cached_full.get(
                        "market_threshold_stats", pd.DataFrame())
                    st.session_state.feature_importance_df = cached_full.get(
                        "feature_importance_df", pd.DataFrame())
                    st.session_state.opt_params = cached_full.get("opt_params", {})
                    st.session_state.ev_boost_map = cached_full.get(
                        "ev_boost_map", {})
                    st.session_state.league_metrics_df = cached_full.get(
                        "league_metrics_df", pd.DataFrame())
                    st.session_state.league_confidence = cached_full.get(
                        "league_confidence", {})
                else:
                    _startup_mark("📊 Reading CSV files")
                    df_past = read_csv_bytes(past_bytes)
                    df_future = read_csv_bytes(future_bytes)
                    df_past = ensure_match_id(df_past)
                    df_future = ensure_match_id(df_future)
                    st.session_state["_dbg_past_df_rows"] = int(len(df_past)) if df_past is not None else 0
                    st.session_state["_dbg_future_df_rows"] = int(len(df_future)) if df_future is not None else 0

                    # FULL first
                    _startup_mark("🧠 Training model (FULL)")
                    _t0 = time.perf_counter()
                    full_final_df, full_hist_df, full_train_core_df = process_and_train_cached(
                        past_bytes, future_bytes, fast_mode=False)
                    full_final_df = ensure_match_id(full_final_df)
                    _dt = time.perf_counter() - _t0
                    st.session_state["_timings"]["train_call_sec_full"] = _dt
                    st.session_state["_cache_hit_flags"]["train_cached_hit_full"] = (_dt < 3.0)
                    full_train_core_df = full_train_core_df if isinstance(
                        full_train_core_df, pd.DataFrame) else pd.DataFrame()
                    # FAST second
                    _startup_mark("⚡ Training model (FAST)")
                    _t0 = time.perf_counter()
                    fast_final_df, fast_hist_df, fast_train_core_df = process_and_train_cached(
                        past_bytes, future_bytes, fast_mode=True)
                    fast_final_df = ensure_match_id(fast_final_df)
                    _dt = time.perf_counter() - _t0
                    st.session_state["_timings"]["train_call_sec_fast"] = _dt
                    st.session_state["_cache_hit_flags"]["train_cached_hit_fast"] = (_dt < 3.0)
                    fast_train_core_df = fast_train_core_df if isinstance(
                        fast_train_core_df, pd.DataFrame) else pd.DataFrame()
                    _startup_mark("🧮 Post-processing results")

                    st.session_state.final_df = combine_fast_full_predictions(
                        full_final_df, fast_final_df)
                    st.session_state.hist_df = full_hist_df
                    st.session_state.train_core = full_train_core_df
                    if _dbg_col_audit_enabled():
                        st.session_state.setdefault("_dbg_pool_stages", {})
                        st.session_state["_dbg_pool_stages"]["train_core_rows"] = int(
                            len(st.session_state.train_core))
                        st.session_state["_dbg_pool_stages"]["train_core_cols"] = int(
                            len(st.session_state.train_core.columns))
                    st.session_state.processed = True
                    st.session_state["pred_df"] = st.session_state.final_df.copy()
                    st.session_state["future_df"] = df_future.copy()
                    st.session_state["_dbg_pred_df_rows"] = int(len(st.session_state["pred_df"]))
                    st.session_state["_dbg_future_df_rows_ss"] = int(len(st.session_state["future_df"]))

                    # Save caches
                    if use_disk:
                        payload_full = {
        "final_df": full_final_df,
        "hist_df": full_hist_df,
        "train_core": full_train_core_df,
        "market_threshold_stats": st.session_state.get(
            "market_threshold_stats",
            pd.DataFrame()),
            "feature_importance_df": st.session_state.get(
                "feature_importance_df",
                pd.DataFrame()),
                "opt_params": st.session_state.get(
                    "opt_params",
                    {}),
                    "ev_boost_map": st.session_state.get(
                        "ev_boost_map",
                        {}),
                        "league_metrics_df": st.session_state.get(
                            "league_metrics_df",
                            pd.DataFrame()),
                            "league_confidence": st.session_state.get(
                                "league_confidence",
                                {}),
                                 }
                        payload_fast = {
        "final_df": fast_final_df,
        "hist_df": fast_hist_df,
        "train_core": fast_train_core_df,
        "market_threshold_stats": st.session_state.get(
            "market_threshold_stats",
            pd.DataFrame()),
            "feature_importance_df": st.session_state.get(
                "feature_importance_df",
                pd.DataFrame()),
                "opt_params": st.session_state.get(
                    "opt_params",
                    {}),
                    "ev_boost_map": st.session_state.get(
                        "ev_boost_map",
                        {}),
                        "league_metrics_df": st.session_state.get(
                            "league_metrics_df",
                            pd.DataFrame()),
                            "league_confidence": st.session_state.get(
                                "league_confidence",
                                {}),
                                 }
                        save_disk_cache(cache_key_full, payload_full)
                        save_disk_cache(cache_key_fast, payload_fast)
                        st.info("💾 AUTO: FAST + FULL disk cache kaydedildi.")
            elif cached is not None:
                st.success(
                    "✅ Disk cache bulundu: eğitim tekrar çalıştırılmadan yüklendi.")
                st.session_state.final_df = cached.get("final_df")
                st.session_state.hist_df = cached.get("hist_df")
                st.session_state.train_core = cached.get("train_core")
                st.session_state.processed = True
                if _dbg_col_audit_enabled() and isinstance(st.session_state.train_core, pd.DataFrame):
                    st.session_state.setdefault("_dbg_pool_stages", {})
                    st.session_state["_dbg_pool_stages"]["train_core_rows"] = int(
                        len(st.session_state.train_core))
                    st.session_state["_dbg_pool_stages"]["train_core_cols"] = int(
                        len(st.session_state.train_core.columns))

                st.session_state.market_threshold_stats = cached.get(
                    "market_threshold_stats", pd.DataFrame())
                st.session_state.feature_importance_df = cached.get(
                    "feature_importance_df", pd.DataFrame())
                st.session_state.opt_params = cached.get("opt_params", {})
                st.session_state.ev_boost_map = cached.get("ev_boost_map", {})
                st.session_state.league_metrics_df = cached.get(
                    "league_metrics_df", pd.DataFrame())
                st.session_state.league_confidence = cached.get(
                    "league_confidence", {})
            else:
                # Parse CSV (cached) and run full pipeline
                _startup_mark("📊 Reading CSV files")
                df_past = read_csv_bytes(past_bytes)
                df_future = read_csv_bytes(future_bytes)
                df_past = ensure_match_id(df_past)
                df_future = ensure_match_id(df_future)
                st.session_state["_dbg_past_df_rows"] = int(len(df_past)) if df_past is not None else 0
                st.session_state["_dbg_future_df_rows"] = int(len(df_future)) if df_future is not None else 0

                _startup_mark("🧠 Training model (FULL / FAST)")
                _t0 = time.perf_counter()
                final_df, hist_df, train_core_df = process_and_train_cached(
                    past_bytes,
                    future_bytes,
                    fast_mode=st.session_state["FAST_MODE"]
                )
                final_df = ensure_match_id(final_df)
                _dt = time.perf_counter() - _t0
                st.session_state["_timings"]["train_call_sec_main"] = _dt
                st.session_state["_cache_hit_flags"]["train_cached_hit_main"] = (_dt < 3.0)
                train_core_df = train_core_df if isinstance(
                    train_core_df, pd.DataFrame) else pd.DataFrame()
                _startup_mark("🧮 Post-processing results")

                st.session_state.final_df = final_df
                st.session_state.hist_df = hist_df
                st.session_state.train_core = train_core_df
                if _dbg_col_audit_enabled():
                    st.session_state.setdefault("_dbg_pool_stages", {})
                    st.session_state["_dbg_pool_stages"]["train_core_rows"] = int(
                        len(st.session_state.train_core))
                    st.session_state["_dbg_pool_stages"]["train_core_cols"] = int(
                        len(st.session_state.train_core.columns))
                st.session_state.processed = True
                st.session_state["pred_df"] = st.session_state.final_df.copy()
                st.session_state["future_df"] = df_future.copy()
                st.session_state["_dbg_pred_df_rows"] = int(len(st.session_state["pred_df"]))
                st.session_state["_dbg_future_df_rows_ss"] = int(len(st.session_state["future_df"]))

                # These are used across tabs – compute once and keep
                st.session_state.league_metrics_df = build_league_metrics(hist_df)
                st.session_state.league_confidence = calculate_league_confidence(
                    hist_df)

                # Persist to disk cache for next run
                if use_disk_cache:
                    payload = {
                        "final_df": st.session_state.final_df,
                        "hist_df": st.session_state.hist_df,
                        "train_core": st.session_state.train_core,
                        "market_threshold_stats": st.session_state.get(
                            "market_threshold_stats",
                            pd.DataFrame()),
                        "feature_importance_df": st.session_state.get(
                            "feature_importance_df",
                            pd.DataFrame()),
                        "opt_params": st.session_state.get(
                            "opt_params",
                            {}),
                        "ev_boost_map": st.session_state.get(
                            "ev_boost_map",
                            {}),
                        "league_metrics_df": st.session_state.league_metrics_df,
                        "league_confidence": st.session_state.league_confidence,
                    }
                    save_disk_cache(cache_key, payload)
                    st.info(
                        "💾 Disk cache kaydedildi (bir sonraki açılış çok daha hızlı olacak).")


        finally:
            _startup_done()
if 'processed' in st.session_state and st.session_state.get(
    'final_df') is not None:
    df = st.session_state.final_df

    # --- Ensure Similarity Anchor columns exist even when loading from disk cache ---
    # If cache was created before similarity integration, re-apply the blend
    # on the fly.
    _train_core = st.session_state.get("train_core", None)
    _need_sim_cols = (
        (_train_core is not None and not getattr(_train_core, "empty", True)) and
        (("SIM_ALPHA" not in df.columns) or ("SIM_ANCHOR_GROUP" not in df.columns))
    )
    if _need_sim_cols:
        df = apply_similarity_anchor_to_predictions(
            df,
            _train_core,
            alpha_max=float(
st.session_state.get(
    "SIM_ALPHA_MAX", 0.60)),
            k=int(st.session_state.get("SIM_K", 60)),
            use_cutoff=bool(
st.session_state.get(
    "SIM_USE_CUTOFF", True)),
            min_similarity=float(
st.session_state.get(
    "SIM_MIN_SIM", 0.55)),
            min_neighbors=int(
st.session_state.get(
    "SIM_MIN_NEIGHBORS", 12)),
            same_league=bool(
st.session_state.get(
    "SIM_SAME_LEAGUE", True)),
            max_pool=int(st.session_state.get("SIM_MAX_POOL", 120000)),
            years_back=int(st.session_state.get("SIM_YEARS_BACK", 4)),
            preset=str(
st.session_state.get(
    "SIM_PRESET",
     "Auto (mevcut kolonlar)")),
            k_ref=float(st.session_state.get("SIM_K_REF", 30.0)),
        )
        st.session_state.final_df = df

    hist_df = st.session_state.hist_df

    # AMQS (Auto-Mod Quality Score) – compute once per run
    if "amqs_df" not in st.session_state:
        st.session_state.amqs_df = calculate_amqs_df(hist_df)
    if 'league_metrics_df' not in st.session_state:
        st.session_state.league_metrics_df = build_league_metrics(hist_df)
    league_metrics_df = st.session_state.league_metrics_df

    league_intel = {}
    if not league_metrics_df.empty:
        for _, row in league_metrics_df.iterrows():
            league_intel[row['League']] = {
                'MS': row.get('AUC_MS1', 0.5),
                'OU': row.get('AUC_OU', 0.5),
                'BTTS': row.get('AUC_BTTS', 0.5)
            }

    if 'league_confidence' not in st.session_state or st.session_state.league_confidence == {}:
        st.session_state.league_confidence = calculate_league_confidence(
            hist_df)

    market_quality = build_market_quality(hist_df)
    st.session_state.market_quality = market_quality

    ev_boost_map = st.session_state.get('ev_boost_map', {})

    all_market_types = [
    'MS 1',
    'MS X',
    'MS 2',
    '2.5 Üst',
    '2.5 Alt',
    'KG Var',
     'KG Yok']
    # ------------------------------
    # SOURCE SELECTION FOR bets_df
    # ------------------------------
    # We want bets_df to be built from FUTURE matches (future_df / pred_df),
    # not from historical/processed dataframes.
    # If df is already future-like and non-empty, keep it.
    # Otherwise pick the best future candidate from session_state.
    def _is_future_like(_d: pd.DataFrame) -> bool:
        if not isinstance(_d, pd.DataFrame) or len(_d) == 0:
            return False
        # future df usually has missing results/close odds
        future_signals = [
            # results (often missing for future)
            "FT_Score_Home", "FT_Score_Away", "Total_Goals",
            "HomeFullTimeScore", "AwayFullTimeScore", "FTHG", "FTAG",
            "HomeGoals", "AwayGoals", "FT_Home", "FT_Away",
            # close odds (often missing for future)
            "Odds_Close_Home", "Odds_Close_Draw", "Odds_Close_Away",
            "Odds_Close_Over25", "Odds_Close_Under25",
            "Odds_Close_BTTS_Yes", "Odds_Close_BTTS_No",
            "ClosingHomeOdd", "ClosingDrawOdd", "ClosingAwayOdd",
            "ClosingO25", "ClosingU25", "ClosingBTTSY", "ClosingBTTSN",
            "ClosingBTTS_Y", "ClosingBTTS_N",
        ]
        present = [c for c in future_signals if c in _d.columns]
        if not present:
            return False
        # if any of these exist and are mostly NaN -> likely future
        return any(
            pd.to_numeric(_d[c], errors="coerce").isna().mean() > 0.80
            for c in present
        )
    def _pick_best_future_df():
        # priority order: the objects that should represent upcoming matches
        priorities = [
            "future_df",
            "pred_df",
            "preds_df",
            "future_preds_df",
            "upcoming_df",
            "df_future",
        ]
        for name in priorities:
            cand = st.session_state.get(name, None)
            if isinstance(cand, pd.DataFrame) and len(cand) > 0:
                return name, cand
        # last resort: final_df if it looks future-like
        cand = st.session_state.get("final_df", None)
        if isinstance(cand, pd.DataFrame) and _is_future_like(cand):
            return "final_df", cand
        return None, None
    # --- hard fallback: choose df_for_bets ---
    df_for_bets = df if isinstance(df, pd.DataFrame) else None
    bets_source = st.session_state.get("_dbg_df_source_name", "df")
    def _get_df_candidate(name: str):
        if name in locals():
            cand = locals()[name]
            if isinstance(cand, pd.DataFrame) and len(cand) > 0:
                return cand, name
        cand = st.session_state.get(name, None)
        if isinstance(cand, pd.DataFrame) and len(cand) > 0:
            return cand, name
        return None, None
    if (not isinstance(df_for_bets, pd.DataFrame)) or (len(df_for_bets) == 0):
        cand, n = _get_df_candidate("pred_df")
        if cand is None:
            cand, n = _get_df_candidate("future_df")
        if isinstance(cand, pd.DataFrame) and len(cand) > 0:
            df_for_bets = cand.copy()
            bets_source = n
            st.session_state["_dbg_df_fallback_forced"] = {"from": "df", "to": n, "rows": int(len(df_for_bets))}
    st.session_state["_dbg_bets_source"] = bets_source
    # census for all sources (compare side-by-side)
    def _shape(x):
        return (int(len(x)), int(x.shape[1])) if isinstance(x, pd.DataFrame) else (-1, -1)
    st.session_state["_dbg_df_for_bets_census"] = {
        "df": _shape(df if "df" in locals() else None),
        "pred_df": _shape(locals().get("pred_df", st.session_state.get("pred_df", None))),
        "future_df": _shape(locals().get("future_df", st.session_state.get("future_df", None))),
        "df_for_bets": _shape(df_for_bets),
    }
    # IMPORTANT: build bets_df from df_for_bets, not df
    st.session_state["_dbg_all_market_types_len"] = int(len(all_market_types)) if isinstance(all_market_types, (list, tuple)) else -1
    st.session_state["_dbg_all_market_types_head"] = list(all_market_types)[:20] if isinstance(all_market_types, (list, tuple)) else []
    st.session_state["_dbg_df_for_bets_type"] = str(type(df_for_bets))
    st.session_state["_dbg_df_for_bets_rows"] = int(len(df_for_bets)) if isinstance(df_for_bets, pd.DataFrame) else -1
    st.session_state["_dbg_df_for_bets_cols"] = int(df_for_bets.shape[1]) if isinstance(df_for_bets, pd.DataFrame) else -1
    st.session_state["_dbg_df_for_bets_cols_head"] = list(df_for_bets.columns)[:50] if isinstance(df_for_bets, pd.DataFrame) else []
    if not isinstance(df_for_bets, pd.DataFrame) or len(df_for_bets) == 0:
        st.session_state["_dbg_blocked_bets_df_reason"] = "df_for_bets_empty_before_repeat"
        st.stop()
    if not isinstance(all_market_types, (list, tuple)) or len(all_market_types) == 0:
        st.session_state["_dbg_blocked_bets_df_reason"] = "all_market_types_empty"
        st.stop()
    # Ensure advanced diagnostics exist before expanding to bets_df
    _train_core = st.session_state.get("train_core", None)
    _adv_allowed = [
        "SIM_ANCHOR_STRENGTH", "SIM_QUALITY", "EFFECTIVE_N",
        "SIM_ANCHOR_GROUP", "SIM_ALPHA",
        "SIM_POver", "SIM_PBTTS", "SIM_MS_STRENGTH", "SIM_OU_STRENGTH",
        "SIM_BTTS_STRENGTH",
        "P_Home_Model", "P_Draw_Model", "P_Away_Model",
        "P_Over_Model", "P_BTTS_Model",
        "BLEND_MODE_MS", "LEAGUE_OK_MS", "BLEND_W_LEAGUE_MS",
        "BLEND_MODE_OB", "LEAGUE_OK_OB", "BLEND_W_LEAGUE_OB",
    ]
    if isinstance(df_for_bets, pd.DataFrame) and len(df_for_bets) > 0:
        _need_diag = any(c not in df_for_bets.columns for c in _adv_allowed)
        if _need_diag and isinstance(_train_core, pd.DataFrame) and not _train_core.empty:
            try:
                _diag = apply_similarity_anchor_to_predictions(
                    df_for_bets.copy(),
                    _train_core,
                    alpha_max=float(st.session_state.get("SIM_ALPHA_MAX", 0.60)),
                    k=int(st.session_state.get("SIM_K", 60)),
                    use_cutoff=bool(st.session_state.get("SIM_USE_CUTOFF", True)),
                    min_similarity=float(st.session_state.get("SIM_MIN_SIM", 0.55)),
                    min_neighbors=int(st.session_state.get("SIM_MIN_NEIGHBORS", 12)),
                    same_league=bool(st.session_state.get("SIM_SAME_LEAGUE", True)),
                    max_pool=int(st.session_state.get("SIM_MAX_POOL", 120000)),
                    years_back=int(st.session_state.get("SIM_YEARS_BACK", 4)),
                    preset="Dual (Lig+Global, MS/OB)",
                    k_ref=float(st.session_state.get("SIM_K_REF", 30.0)),
                )
                _cols = [c for c in _adv_allowed if c in _diag.columns]
                if "Match_ID" in df_for_bets.columns and "Match_ID" in _diag.columns:
                    _diag = _diag.drop_duplicates(subset=["Match_ID"], keep="last")
                    df_for_bets = df_for_bets.merge(
                        _diag[["Match_ID"] + _cols],
                        on="Match_ID",
                        how="left",
                    )
                else:
                    _diag = _diag.reindex(df_for_bets.index)
                    for _c in _cols:
                        df_for_bets[_c] = _diag[_c]
            except Exception:
                if _dbg_col_audit_enabled():
                    st.session_state.setdefault("_dbg_pool_stages", {})
                    st.session_state["_dbg_pool_stages"]["sim_diag_error"] = traceback.format_exc()
                    st.warning("Similarity diagnostics could not be computed.")
        _need_tg = any(c not in df_for_bets.columns for c in ["TG_Q75", "TG_Q90"])
        if _need_tg:
            try:
                df_for_bets = add_mdl_features(df_for_bets)
            except Exception:
                if _dbg_col_audit_enabled():
                    st.session_state.setdefault("_dbg_pool_stages", {})
                    st.session_state["_dbg_pool_stages"]["tg_diag_error"] = traceback.format_exc()
                    st.warning("TG_Q75/TG_Q90 could not be computed.")
    # create bets_df
    bets_df = df_for_bets.loc[df_for_bets.index.repeat(len(all_market_types))].copy()
    bets_df["Seçim"] = all_market_types * len(df_for_bets)
    if _dbg_col_audit_enabled():
        st.session_state.setdefault("_dbg_pool_stages", {})
        st.session_state["_dbg_pool_stages"]["bets_df_has_sim"] = all(
            c in bets_df.columns for c in ["SIM_ANCHOR_STRENGTH", "SIM_QUALITY", "EFFECTIVE_N"]
        )
    st.session_state["_dbg_bets_df_rows"] = int(len(bets_df))
    st.session_state["_dbg_bets_df_cols_count"] = int(bets_df.shape[1]) if isinstance(bets_df, pd.DataFrame) else -1
    st.session_state["_dbg_bets_df_cols_head"] = list(bets_df.columns)[:200] if isinstance(bets_df, pd.DataFrame) else []
    st.session_state["_bets_cols"] = list(bets_df.columns)[:200] if isinstance(bets_df, pd.DataFrame) else []
    if isinstance(bets_df, pd.DataFrame):
        _missing_odds = []
        _canon_map = {
            "HomeOdd": "Odds_Open_Home",
            "DrawOdd": "Odds_Open_Draw",
            "AwayOdd": "Odds_Open_Away",
            "O25": "Odds_Open_Over25",
            "U25": "Odds_Open_Under25",
            "BTTSY": "Odds_Open_BTTS_Yes",
            "BTTS_Y": "Odds_Open_BTTS_Yes",
            "BTTSN": "Odds_Open_BTTS_No",
            "BTTS_N": "Odds_Open_BTTS_No",
            "ClosingHomeOdd": "Odds_Close_Home",
            "ClosingDrawOdd": "Odds_Close_Draw",
            "ClosingAwayOdd": "Odds_Close_Away",
            "ClosingO25": "Odds_Close_Over25",
            "ClosingU25": "Odds_Close_Under25",
            "ClosingBTTSY": "Odds_Close_BTTS_Yes",
            "ClosingBTTS_Y": "Odds_Close_BTTS_Yes",
            "ClosingBTTSN": "Odds_Close_BTTS_No",
            "ClosingBTTS_N": "Odds_Close_BTTS_No",
        }
        for _dst, _src in _canon_map.items():
            if _dst in bets_df.columns:
                bets_df[_dst] = pd.to_numeric(bets_df[_dst], errors="coerce")
            elif _src in bets_df.columns:
                bets_df[_dst] = pd.to_numeric(bets_df[_src], errors="coerce")
            else:
                bets_df[_dst] = np.nan
            if bets_df[_dst].isna().all():
                _missing_odds.append(_dst)
        st.session_state["_dbg_missing_odds_cols"] = _missing_odds
    else:
        st.session_state["_dbg_missing_odds_cols"] = ["bets_df_not_dataframe"]
    _close_cols = ["ClosingHomeOdd", "ClosingDrawOdd", "ClosingAwayOdd", "Odds_Close_Home", "Odds_Close_Draw", "Odds_Close_Away"]
    _open_cols = ["Odds_Open_Home", "Odds_Open_Draw", "Odds_Open_Away"]
    _close_has = False
    _open_has = False
    for _c in _close_cols:
        if _c in bets_df.columns:
            _close_has = bool(pd.to_numeric(bets_df[_c], errors="coerce").notna().any())
            if _close_has:
                break
    if not _close_has:
        for _c in _open_cols:
            if _c in bets_df.columns:
                _open_has = bool(pd.to_numeric(bets_df[_c], errors="coerce").notna().any())
                if _open_has:
                    break
    st.session_state["_odds_source"] = "close" if _close_has else ("open" if _open_has else "nan")

    conditions = [
    bets_df['Seçim'] == 'MS 1',
    bets_df['Seçim'] == 'MS X',
    bets_df['Seçim'] == 'MS 2',
    bets_df['Seçim'] == '2.5 Üst',
    bets_df['Seçim'] == '2.5 Alt',
    bets_df['Seçim'] == 'KG Var',
     bets_df['Seçim'] == 'KG Yok']

    # 📌 USING FINAL PROBABILITIES
    probs = [
    bets_df.get(
        'P_Home_Final',
        0.5),
        bets_df.get(
            'P_Draw_Final',
            0.25),
            bets_df.get(
                'P_Away_Final',
                0.25),
                bets_df.get(
                    'P_Over_Final',
                    0.5),
                    1 -
                    bets_df.get(
                        'P_Over_Final',
                        0.5),
                        bets_df.get(
                            'P_BTTS_Final',
                            0.5),
                            1 -
                            bets_df.get(
                                'P_BTTS_Final',
                                 0.5)]

    odds = [
        bets_df['HomeOdd'], bets_df['DrawOdd'], bets_df['AwayOdd'],
        bets_df['O25'], bets_df['U25'],
        bets_df['BTTSY'], bets_df['BTTSN']
    ]

    bets_df['Prob'] = np.select(conditions, probs, default=0.5)
    bets_df['Odd'] = np.select(conditions, odds, default=np.nan)
    bets_df = bets_df[bets_df.apply(validate_odds_row, axis=1)].copy()

    bets_df = compute_ev_scores(bets_df, df_for_bets, market_quality)

    # De-dup: sometimes upcoming feeds contain repeated rows for the same
    # match/market.
    if 'MatchID_Unique' not in bets_df.columns:
        bets_df['MatchID_Unique'] = bets_df['Date'].astype(
            str) + '_' + bets_df['HomeTeam'].astype(str) + '_' + bets_df['AwayTeam'].astype(str)
    bets_df = bets_df.drop_duplicates(
    subset=[
        'MatchID_Unique',
        'Seçim'],
         keep='first').copy()

    bets_df['Market_Conf_Score'] = bets_df['Seçim'].map(
        market_quality).fillna(0.5)
    league_confidence_map = st.session_state.get('league_confidence', {})
    bets_df['League_Conf'] = bets_df['League'].map(
        league_confidence_map).fillna(0.5)

    def get_league_boost(row):
        lig_stats = league_intel.get(row['League'])
        if not lig_stats:
            return 0.0
        market = row['Seçim']
        auc = 0.5
        if 'MS' in market:
            auc = lig_stats.get('MS', 0.5)
        elif '2.5' in market: auc = lig_stats.get('OU', 0.5)
        elif 'KG' in market:
            auc = lig_stats.get('BTTS', 0.5)
        if auc > 0.70: return 0.25
        elif auc > 0.60:
            return 0.15
        elif auc < 0.50: return -0.15
        return 0.0

    bets_df['League_Boost'] = bets_df.apply(get_league_boost, axis=1)

    def get_ev_verification_boost(row):
        ev = row['EV']
        return get_ev_boost(ev, ev_boost_map.get("Global", {}))

    bets_df['EV_Boost'] = bets_df.apply(get_ev_verification_boost, axis=1)

    MARKET_TYPE_BASE_BONUS = {
        "MS 1": 0.03, "MS 2": 0.02, "MS X": -0.02,
        "2.5 Üst": 0.03, "2.5 Alt": 0.03, "KG Var": 0.02, "KG Yok": 0.02,
    }
    bets_df['Market_Type_Bonus'] = bets_df['Seçim'].map(
        MARKET_TYPE_BASE_BONUS).fillna(0.0)

    bets_df['Conf_Term'] = bets_df['Prob'].clip(0, 1)
    ev_clipped = bets_df['EV'].clip(-10, 20)
    bets_df['EV_Term'] = (ev_clipped + 10) / 30.0

    bets_df['MQ_Term'] = bets_df['Market_Conf_Score'].clip(0, 1)
    bets_df['LQ_Term'] = bets_df['League_Conf'].clip(0, 1)
    bets_df['Bonus_Term'] = (bets_df['League_Boost'] + bets_df['EV_Boost'] +
                             bets_df['Market_Type_Bonus']).clip(-0.30, 0.30)

    # ✅ PATCH 4: EV Normalization (Z-Score)
    if 'EV' in bets_df.columns and bets_df['EV'].std() > 0:
        bets_df['EV_Norm'] = (
            bets_df['EV'] - bets_df['EV'].mean()) / (bets_df['EV'].std() + 1e-9)
    else:
        bets_df['EV_Norm'] = 0.0

    # ✅ PATCH 5: Market-Weighted Score Calculation
    def calculate_weighted_score(row):
        m = row['Seçim']
        prob = row['Prob']
        ev_norm = row['EV_Norm']

        weight = MARKET_SCORE_WEIGHTS.get(m, 1.0)

        # Base Score: 60% Confidence + 40% EV Quality
        # Normalize EV z-score roughly to 0-1 scale
        base_score = 0.6 * prob + 0.4 * (0.5 + 0.1 * ev_norm)

        return weight * base_score

    bets_df['Score'] = bets_df.apply(calculate_weighted_score, axis=1)


# ===========================
# SIMILARITY HYBRID (OU/KG)
# ===========================


def _compute_sim_prior(row):
    sel = str(row.get('Seçim', ''))
    if sel in [
    '2.5 Üst',
    '2.5 Alt'] and _np.isfinite(
        row.get(
            'SIM_POver',
             _np.nan)):
        return row['SIM_POver'] if sel == '2.5 Üst' else (
            1.0 - row['SIM_POver'])
    if sel in [
    'KG Var',
    'KG Yok'] and _np.isfinite(
        row.get(
            'SIM_PBTTS',
             _np.nan)):
        return row['SIM_PBTTS'] if sel == 'KG Var' else (
            1.0 - row['SIM_PBTTS'])
    return _np.nan


def _compute_alpha(row):
    q = float(row.get('SIM_QUALITY', 0.0) or 0.0)
    n = float(row.get('EFFECTIVE_N', 0.0) or 0.0)
    alpha = 0.15 + 0.35 * max(0.0, min(1.0, q))
    if n < 15:
        alpha *= 0.5
    return float(max(0.10, min(0.45, alpha)))


if "bets_df" in locals():
    bets_df['SIM_PRIOR'] = bets_df.apply(_compute_sim_prior, axis=1)

    bets_df['Prob_HYBRID'] = bets_df['Prob']
    _mask = bets_df['SIM_PRIOR'].notna()
    _mask = _align_mask(_mask, bets_df)
    _alphas = bets_df.loc[_mask].apply(_compute_alpha, axis=1)
    bets_df.loc[_mask, 'Prob_HYBRID'] = (
        (1.0 - _alphas) * bets_df.loc[_mask, 'Prob']
        + _alphas * bets_df.loc[_mask, 'SIM_PRIOR']
    )

    bets_df['EV_HYBRID'] = (bets_df['Prob_HYBRID'] * bets_df['Odd']) - 1.0

    # ===========================
    # MDL INJECTION (WIDE POOL)
    # ===========================
    for _c in ['GoldenScore_MDL', 'MDL_BOOST']:
        if _c in bets_df.columns:
            bets_df[_c] = bets_df[_c].astype(float).fillna(0.0)

    # ===========================
    # SCORE OVERRIDE -> HYBRID
    # ===========================


    def _final_score(row):
        base = float(row.get('Score', 0.0) or 0.0)
        evh = float(row.get('EV_HYBRID', 0.0) or 0.0)
        mdl = float(row.get('MDL_BOOST', 0.0) or 0.0)
        return base * 0.4 + evh * 0.6 + mdl


    bets_df['Score'] = bets_df.apply(_final_score, axis=1)

    # -----------------------------------------------------------
    # KNN / Similarity FOUNDATION: inject historical-neighbor stats into candidate pool
    # (This makes EFFECTIVE_N and SIM_* available for ADD / rule engine.)
    # -----------------------------------------------------------

    if "Match_ID" not in bets_df.columns:
        bets_df["Match_ID"] = bets_df.apply(_make_match_id, axis=1)

        # Pick similarity features that exist in BOTH upcoming (bets_df) and
        # past (train_core)
        _sim_feats = _pick_similarity_features(
        bets_df, _train_core, preset="Auto (mevcut kolonlar)")
        if _sim_feats and (
            _train_core is not None) and (
            not getattr(
                _train_core,
                "empty",
                True)):
            _sim_payload = []
            for _mid, _g in bets_df.groupby("Match_ID", sort=False):
                _sel = _g.iloc[0]  # representative row for the match
                _neigh_df, _sim_out = _similarity_neighbors_for_match(
                        train_core=_train_core,
                        upcoming_row=_sel,
                        feats=_sim_feats,
                    k=50,
                    same_league=True,
                    use_cutoff=False,         # allow low-sim results; we'll let gates handle reliability
                    min_similarity=0.30,
                    min_neighbors=1,
                    year_window=0,
                    use_time_decay=False,
                    half_life_days=540,
                    gate_min_simq=0.0,
                    gate_min_en=0,
                )
                if isinstance(_sim_out, dict) and (not _sim_out.get("error")):
                    _sim_payload.append({
                            "Match_ID": _mid,
                            "SIM_QUALITY": float(_sim_out.get("SIM_QUALITY", np.nan)),
                            "EFFECTIVE_N": int(_sim_out.get("EFFECTIVE_N", _sim_out.get("SIM_N", 0)) or 0),
                            "SIM_DIST_MED": float(_sim_out.get("SIM_DIST_MED", np.nan)),
                            "SIM_P1": float(_sim_out.get("SIM_P1", np.nan)),
                            "SIM_PX": float(_sim_out.get("SIM_PX", np.nan)),
                            "SIM_P2": float(_sim_out.get("SIM_P2", np.nan)),
                            "SIM_POver": float(_sim_out.get("SIM_POver", np.nan)),
                            "SIM_PBTTS": float(_sim_out.get("SIM_PBTTS", np.nan)),
                            "SIM_ANCHOR_STRENGTH": float(_sim_out.get("SIM_ANCHOR_STRENGTH", np.nan)),
                            "POOL_USED_N": int(_sim_out.get("POOL_USED_N", 0) or 0),
                        })
            if _sim_payload:
                _sim_df = pd.DataFrame(_sim_payload)
                # ensure columns exist even if merge yields NaN
                bets_df = bets_df.merge(_sim_df, on="Match_ID", how="left")

        # Apply Anchor Decision Discipline (ADD) to candidate pool
        # (post-similarity)
    #     try:  # AUTO-COMMENTED (illegal global try)
    #         bets_df = apply_rule_engine_to_candidates(bets_df)
    #     except Exception as _e_add:
    #         # Fail-safe: never break the app due to ADD
    #         pass

    if "_dbg_pool_stages" not in st.session_state:
        st.session_state["_dbg_pool_stages"] = {}
    st.session_state["_dbg_pool_stages"]["00_start_bets_df"] = (
        0 if bets_df is None else int(len(bets_df)))
    pool_bets = _ensure_unique_index(bets_df.copy())
    st.session_state["_dbg_pool_stages"]["01_pool_copy"] = int(len(pool_bets))
    _maybe_record_col_lineage(pool_bets, "pool: start", "pool_bets")
    _store_col_audit(pool_bets, "pool_bets")

    # --- Similarity Anchor Diagnostics (quick sanity) ---
#     try:  # AUTO-COMMENTED (illegal global try)
#         if "SIM_ANCHOR_GROUP" in pool_bets.columns:
#             # match-level distribution (not per-bet) -> use unique by
#             # (Date,HomeTeam,AwayTeam) if possible
#             _mcols = [c for c in ["Date", "HomeTeam","AwayTeam"] if c in pool_bets.columns]
#             _tmp = pool_bets.copy()
#             if _mcols:
#                 _tmp = _tmp.drop_duplicates(subset=_mcols)
#             dist = _tmp["SIM_ANCHOR_GROUP"].value_counts(
#                 normalize=True).to_dict()
#             st.session_state["sim_anchor_dist"] = dist
#             st.session_state["sim_anchor_strength_mean"] = float(pd.to_numeric(_tmp.get("SIM_ANCHOR_STRENGTH", 0.0), errors="coerce").fillna(0.0).mean())
#     except Exception:
#         pass

    # Attach AMQS to pool bets for BestOf ranking & advanced columns
    pool_bets = attach_amqs_to_picks(
    pool_bets, st.session_state.get(
        "amqs_df", pd.DataFrame()))

    # --- STABLE METRICS (tek sefer) ---
    # AMQS / CONF / Final_Conf / BestOfRank gibi kolonlar alt-kümelerde tekrar
    # hesaplanırsa değerler oynar.
    pool_bets = ensure_amqs_columns(pool_bets)
    # Compute GoldenScore BEFORE Final_Confidence so it isn't stuck at a placeholder.
    # We recompute each run (cheap) to avoid carrying over any placeholder
    # values.
#     try:  # AUTO-COMMENTED (illegal global try)
#         if "Archetype" not in pool_bets.columns:
#             pool_bets = apply_archetype_layer(pool_bets)
#         pool_bets = compute_golden_score(
#     pool_bets, market_quality, use_archetype=True)
#     except Exception:
#         pass
    if "Final_Confidence" not in pool_bets.columns:
        pool_bets = compute_final_confidence(pool_bets)
    if "CONF_ICON" not in pool_bets.columns:
        pool_bets = assign_final_conf_icon(pool_bets)
    pool_bets = ensure_bestofrank(pool_bets)

    if "Star_Rating" not in pool_bets.columns:
        pool_bets = add_star_rating_by_bestofrank(pool_bets)

    # Keep an unfiltered snapshot in session (used by Prediction Log + WF Lab)
    # IMPORTANT: downstream UI may further filter `pool_bets` based on sliders.
    st.session_state["pool_bets_raw"] = pool_bets.copy()

    # -----------------------------
    # (Removed legacy Prediction Log autosnapshot block)

    # -------------------------------------------------
    # MAIN TABS (tek yerde tanımla)
    # -------------------------------------------------
    tab1, tab2, tab3, tab4, tab_qc, tab_wf, tab_saved = st.tabs([
        "Günün Best Of",
        "Derin Metrikler",
        "Lig Güven Karnesi",
        "Backtest",
        "QC Panel",
        "Walk-Forward Filter Lab",
        "Günün Maçları (Kayıt)",
    ])

    with tab1:
        st.subheader("🏆 Günün 'Best Of' Listesi & Golden Ratio")

        # ===========================================================
        # TEK ANA SLIDER (5 SEVİYE) → arka planda tüm Golden Ratio ayarlarını set eder
        # ===========================================================
        STYLE_OPTIONS = [
            "🔒 BANKO",
            "🧠 ZEKİ",
            "💎 VALUE",
            "✅ GÜVENLİ",
            "💣 SÜRPRİZ",
        ]

        # ===========================================================
        # PRESET PROFİLLERİ (REVİZE: AKIŞI AÇAN DENGELİ AYARLAR)
        # ===========================================================
        PRESET_PROFILES = {
            "🔒 BANKO": {
                # STRATEJİ: Olasılık barajını düşür (0.60 -> 0.53), ama Sim Quality ile sağlama al.
                "gr_min_prob": 0.53,            # BARAJ İNDİRİLDİ (Daha çok maç girer)
                "gr_min_score": 0.50,           # Score şartı gevşetildi
                "gr_min_ev": -3.0,              # Banko ararken EV'ye takılma
                "gr_max_total": 6,
                "gr_min_star_global": 3,
                "gr_show_adv_global": False,
                "gr_min_league_conf": 0.25,     # Lig güveni biraz gevşetildi
                "gr_min_market_conf": 0.25,
                "gr_hide_weak": True,
                # --- KALİTE KONTROLÜ (Giren maçlar sağlam olsun) ---
                "gr_min_anchor_strength": 0.40,
                "gr_min_effective_n": 20,       # 20 makul bir sayı
                "gr_use_knn_gate_bestof": True,
                "gr_min_simq_bestof": 0.33,     # QC Ortalamasının (0.38) bir tık altı, akışı kesmez.
                # ------------------------------
                "gr_amqs_pct_range": (50, 100),
                "gr_conf_pct_range": (50, 100),
                "gr_conf_statuses": ["High", "Medium"],
                "gr_automod_statuses": ["High", "Medium"],
                "gr_min_bestofrank": -0.50,     # Negatif rank'e izin ver (Score düşükse bile)
                "bo_inc_trap": False,
                "bo_inc_danger": False,
                "bo_inc_profile": False,
            },
            "🧠 ZEKİ": {
                # STRATEJİ: Orta risk, yüksek zeka.
                "gr_min_prob": 0.49,            # %50 altına izin ver (Sürpriz favoriler için)
                "gr_min_score": 0.48,
                "gr_min_ev": -1.0,
                "gr_max_total": 8,
                "gr_min_star_global": 2,        # 2 yıldıza izin ver
                "gr_show_adv_global": False,
                "gr_min_league_conf": 0.20,
                "gr_min_market_conf": 0.20,
                "gr_hide_weak": False,          # Zayıf ligleri de gör (Fırsat olabilir)
                # --- KALİTE KONTROLÜ ---
                "gr_min_anchor_strength": 0.40,
                "gr_min_effective_n": 15,
                "gr_use_knn_gate_bestof": True,
                "gr_min_simq_bestof": 0.32,     # Makul bir kalite sınırı
                # ------------------------------
                "gr_amqs_pct_range": (40, 100),
                "gr_conf_pct_range": (40, 100),
                "gr_conf_statuses": ["High", "Medium"],
                "gr_automod_statuses": ["High", "Medium"],
                "gr_min_bestofrank": -1.0,
                "bo_inc_trap": False,
                "bo_inc_danger": False,
                "bo_inc_profile": False,
            },
            "💎 VALUE": {
                # STRATEJİ: Sadece EV odaklı. Prob ve Quality düşük olabilir.
                "gr_min_prob": 0.42,            # Çok düşük olasılık kabul
                "gr_min_score": 0.42,
                "gr_min_ev": 1.5,               # Ama EV mutlaka pozitif olmalı
                "gr_max_total": 12,
                "gr_min_star_global": 1,
                "gr_show_adv_global": False,
                "gr_min_league_conf": 0.10,
                "gr_min_market_conf": 0.10,
                "gr_hide_weak": False,
                # --- KALİTE KONTROLÜ ---
                "gr_min_anchor_strength": 0.30,
                "gr_min_effective_n": 10,
                "gr_use_knn_gate_bestof": True,
                "gr_min_simq_bestof": 0.25,     # Esnek
                # ------------------------------
                "gr_amqs_pct_range": (20, 100),
                "gr_conf_pct_range": (20, 100),
                "gr_conf_statuses": ["High", "Medium", "Low"],
                "gr_automod_statuses": ["High", "Medium", "Low"],
                "gr_min_bestofrank": -2.0,
                "bo_inc_trap": True,            # Tuzaklı maçlar Value olabilir
                "bo_inc_danger": False,
                "bo_inc_profile": True,
            },
            "✅ GÜVENLİ": {
                # Banko'nun bir tık altı, Zeki'nin bir tık üstü.
                "gr_min_prob": 0.51,
                "gr_min_score": 0.49,
                "gr_min_ev": -2.0,
                "gr_max_total": 10,
                "gr_min_star_global": 2,
                "gr_show_adv_global": False,
                "gr_min_league_conf": 0.25,
                "gr_min_market_conf": 0.25,
                "gr_hide_weak": True,
                # --- KALİTE KONTROLÜ ---
                "gr_min_anchor_strength": 0.40,
                "gr_min_effective_n": 15,
                "gr_use_knn_gate_bestof": True,
                "gr_min_simq_bestof": 0.30,
                # ------------------------------
                "gr_amqs_pct_range": (35, 100),
                "gr_conf_pct_range": (35, 100),
                "gr_conf_statuses": ["High", "Medium"],
                "gr_automod_statuses": ["High", "Medium"],
                "gr_min_bestofrank": -1.0,
                "bo_inc_trap": False,
                "bo_inc_danger": False,
                "bo_inc_profile": True,
            },
            "💣 SÜRPRİZ": {
                # Kapıları sonuna kadar aç.
                "gr_min_prob": 0.30,
                "gr_min_score": 0.30,
                "gr_min_ev": -10.0,
                "gr_max_total": 20,
                "gr_min_star_global": 1,
                "gr_show_adv_global": False,
                "gr_min_league_conf": 0.0,
                "gr_min_market_conf": 0.0,
                "gr_hide_weak": False,
                # --- KALİTE KONTROLÜ ---
                "gr_min_anchor_strength": 0.15,
                "gr_min_effective_n": 5,
                "gr_use_knn_gate_bestof": False,  # Gate kapalı
                "gr_min_simq_bestof": 0.10,
                # ------------------------------
                "gr_amqs_pct_range": (0, 100),
                "gr_conf_pct_range": (0, 100),
                "gr_conf_statuses": ["High", "Medium", "Low"],
                "gr_automod_statuses": ["High", "Medium", "Low"],
                "gr_min_bestofrank": -5.0,
                "bo_inc_trap": True,
                "bo_inc_danger": True,
                "bo_inc_profile": True,
            },
        }

        def _apply_style_preset(style_name: str, force: bool = False):
            preset = PRESET_PROFILES.get(style_name, {})
            if not preset:
                return
            # Preset kilidi açıksa (veya force) session_state'i preset'e göre
            # set et
            preset_lock = bool(st.session_state.get("gr_preset_lock", True))
            if (not preset_lock) and (not force):
                return
            for k, v in preset.items():
                st.session_state[k] = v

        # --- Main slider / preset control ---
        c_style1, c_style2, c_style3 = st.columns([2.2, 1.2, 1.2])
        style_mode = c_style1.select_slider(
            "🎚️ Tek Filtre: Stil (Banko → Zeki → Value → Güvenli → Sürpriz)",
            options=STYLE_OPTIONS,
            value=st.session_state.get("gr_style_mode", "✅ GÜVENLİ"),
            key="gr_style_mode",
        )
        c_style2.checkbox("🔒 Preset kilidi", value=True, key="gr_preset_lock")
        st.checkbox(
    "🏷️ BestOf rozetini stile kilitle",
    value=True,
     key="gr_force_style_badge")
        if c_style3.button("↻ Preset uygula", use_container_width=True):
            _apply_style_preset(style_mode, force=True)

        # Auto-apply when style changes (or first run)
        _last = st.session_state.get("_gr_style_last", None)
        if (_last is None) or (style_mode != _last):
            _apply_style_preset(style_mode, force=True)
            st.session_state["_gr_style_last"] = style_mode

        st.caption("İpucu: Gelişmiş filtreleri açmadan sadece bu slider ile BestOf/Geniş Liste davranışını yöneteceksin. Gelişmiş filtreler sadece override içindir.")

        with st.expander("🔧 Gelişmiş Filtreler (override)", expanded=False):
            is_fast = st.session_state.get("FAST_MODE", False)
            st.checkbox(
    "🛟 Havuz küçükse otomatik gevşet (önerilir)",
    value=True,
     key="gr_auto_relax_small_pool")

            # Load persistent UI defaults (so you can keep sliders at your
            # preferred levels)
            _ui = load_ui_defaults()
            _dflt = (_ui or {}).get("golden_filters", {})

            thr_stats = st.session_state.get(
    "market_threshold_stats", pd.DataFrame())
            default_min_prob = float(_dflt.get("min_prob", 0.58))
            if not thr_stats.empty and "Pazar" in thr_stats.columns and "En_iyi_Prob_Eşiği" in thr_stats.columns:
                ms1_row = thr_stats[thr_stats["Pazar"] == "MS 1"]
                if not ms1_row.empty:
                    val = ms1_row["En_iyi_Prob_Eşiği"].iloc[0]
                    if not np.isnan(val) and "min_prob" not in _dflt:
                        default_min_prob = float(val)

            default_min_score = float(
    _dflt.get(
        "min_score",
         0.62 if not is_fast else 0.52))
            default_min_ev    = float(_dflt.get("min_ev", -1.0 if not is_fast else -5.0))
            default_max_total = int(_dflt.get("max_total", 10))
            default_min_league_conf = float(_dflt.get("min_league_conf", 0.30))
            default_min_market_conf = float(_dflt.get("min_market_conf", 0.30))
            default_hide_weak = bool(_dflt.get("hide_weak_leagues", True))

            # NEW (Similarity Anchor): Eligibility gates for BestOf
            default_min_anchor_strength = float(
                _dflt.get("min_anchor_strength", 0.55))
            default_min_effective_n = int(_dflt.get("min_effective_n", 20))

            # Extra filters (AutoMod / Conf)
            default_amqs_pct_range = tuple(
                _dflt.get("amqs_pct_range", (0, 100)))
            default_conf_pct_range = tuple(
                _dflt.get("conf_pct_range", (0, 100)))
            default_conf_statuses = list(
    _dflt.get(
        "conf_statuses", [
            "High", "Medium", "Low"]))
            default_automod_statuses = list(
    _dflt.get(
        "automod_statuses", [
            "High", "Medium", "Low"]))
            default_min_bestofrank = float(_dflt.get("min_bestofrank", -0.50))

            default_min_prob = min(0.85, max(0.40, float(default_min_prob)))
            default_min_score = min(0.85, max(0.10, float(default_min_score)))
            default_min_ev = min(10.0, max(-5.0, float(default_min_ev)))

            c_f1, c_f2, c_f3, c_f4 = st.columns(4)

            # --- SAFETY: clamp persisted slider state to widget bounds (prevents StreamlitValueBelowMinError) ---
            if "gr_min_prob" in st.session_state:
                st.session_state["gr_min_prob"] = float(
                    st.session_state["gr_min_prob"])
                st.session_state["gr_min_prob"] = min(
                    0.85, max(0.40, st.session_state["gr_min_prob"]))
            if "gr_min_score" in st.session_state:
                st.session_state["gr_min_score"] = float(
                    st.session_state["gr_min_score"])
                st.session_state["gr_min_score"] = min(
                    0.85, max(0.10, st.session_state["gr_min_score"]))
            if "gr_min_ev" in st.session_state:
                st.session_state["gr_min_ev"] = float(
                    st.session_state["gr_min_ev"])
                st.session_state["gr_min_ev"] = min(
                    10.0, max(-5.0, st.session_state["gr_min_ev"]))

            # Also clamp Similarity/Anchor related persisted state (prevents
            # errors when switching presets)
            if "gr_min_anchor_strength" in st.session_state:
                st.session_state["gr_min_anchor_strength"] = float(
                    st.session_state["gr_min_anchor_strength"])
                st.session_state["gr_min_anchor_strength"] = min(
                    0.85, max(0.30, st.session_state["gr_min_anchor_strength"]))
            if "gr_min_effective_n" in st.session_state:
                st.session_state["gr_min_effective_n"] = int(
                    float(st.session_state["gr_min_effective_n"]))
                st.session_state["gr_min_effective_n"] = min(
                    120, max(0, st.session_state["gr_min_effective_n"]))
            min_prob_val = c_f1.slider("Min Prob", 0.40, 0.85, default_min_prob, 0.01, key="gr_min_prob")
            min_score_val = c_f2.slider(
    "Min Score",
    0.10,
    0.85,
    default_min_score,
    0.01,
     key="gr_min_score")
            min_ev_val = c_f3.slider("Min EV", -5.0, 10.0, default_min_ev, 0.5, key="gr_min_ev")
            max_total_val = c_f4.slider(
    "Günlük Max Maç",
    4,
    30,
    default_max_total,
    1,
     key="gr_max_total")

            # Global display gates (applies to both BestOf and Havuz tables)
            default_min_star_global = int(_dflt.get("min_star", 1))
            default_show_adv_global = bool(_dflt.get("show_adv", False))
            c_star1, c_star2 = st.columns([1, 2])
            min_star_global = c_star1.slider(
    "Min Yıldız (⭐)",
    1,
    5,
    default_min_star_global,
    1,
     key="gr_min_star_global")
            show_adv_global = c_star2.checkbox(
    "Gelişmiş kolonları göster (Score / GoldenScore / Conf)",
    value=default_show_adv_global,
     key="gr_show_adv_global")

            c_f5, c_f6, c_f7 = st.columns(3)
            min_league_conf = c_f5.slider(
    "Min Lig Güven",
    0.00,
    1.00,
    default_min_league_conf,
    0.01,
     key="gr_min_league_conf")
            min_market_conf = c_f6.slider(
    "Min Pazar Kalitesi",
    0.00,
    1.00,
    default_min_market_conf,
    0.01,
     key="gr_min_market_conf")
            hide_weak_leagues = c_f7.checkbox(
    "Zayıf Ligleri Gizle",
    value=default_hide_weak,
     key="gr_hide_weak")

            # NEW: Similarity Anchor eligibility sliders
            c_f11, c_f12 = st.columns(2)
            min_anchor_strength_val = c_f11.slider("Min Anchor Strength", 0.30, 0.85, float(
                default_min_anchor_strength), 0.01, key="gr_min_anchor_strength")
            min_effective_n_val = c_f12.slider("Min Effective_N", 0, 120, int(
                default_min_effective_n), 1, key="gr_min_effective_n")

            # NEW: KNN reliability gate (BestOf) - uses SIM_QUALITY +
            # Effective_N
            c_f13, c_f14 = st.columns(2)
            use_knn_gate_bestof = c_f13.checkbox(
    "KNN Gate (BestOf)", value=True, key="gr_use_knn_gate_bestof")
            min_simq_bestof_val = c_f14.slider(
    "Min SIM_QUALITY", 0.0, 0.90, 0.30, 0.05, key="gr_min_simq_bestof")
            c_f8, c_f9, c_f10 = st.columns(3)
            amqs_pct_range = c_f8.slider(
    "AutoMod Percentile Aralığı",
    0,
    100,
    default_amqs_pct_range,
    1,
     key="gr_amqs_pct_range")
            conf_pct_range = c_f9.slider(
    "CONF Percentile Aralığı",
    0,
    100,
    default_conf_pct_range,
    1,
     key="gr_conf_pct_range")
            conf_status_sel = c_f10.multiselect(
    "CONF Seviye Filtre", [
        "High", "Medium", "Low"], default=default_conf_statuses, key="gr_conf_statuses")

            c_f11, c_f12 = st.columns(2)
            automod_status_sel = c_f11.multiselect(
    "AutoMod Seviye Filtre", [
        "High", "Medium", "Low"], default=default_automod_statuses, key="gr_automod_statuses")
            min_bestofrank_val = c_f12.slider(
    "Min BestOfRank", -5.0, 3.0, float(default_min_bestofrank), 0.05, key="gr_min_bestofrank")

            # Save these slider values as your persistent default
            if st.button("💾 Bu filtreleri varsayılan yap"):
                save_ui_defaults({
                    "golden_filters": {
                        "min_prob": float(min_prob_val),
                        "min_score": float(min_score_val),
                        "min_ev": float(min_ev_val),
                        "max_total": int(max_total_val),
                        "min_star": int(min_star_global),
                        "show_adv": bool(show_adv_global),
                        "min_league_conf": float(min_league_conf),
                        "min_market_conf": float(min_market_conf),
                        "hide_weak_leagues": bool(hide_weak_leagues),
                        "min_anchor_strength": float(min_anchor_strength_val),
                        "min_effective_n": int(min_effective_n_val),
                        "amqs_pct_range": tuple(amqs_pct_range),
                        "conf_pct_range": tuple(conf_pct_range),
                        "conf_statuses": list(conf_status_sel),
                        "automod_statuses": list(automod_status_sel),
                        "min_bestofrank": float(min_bestofrank_val),
                    }
                })
                st.success(
                    "Kaydedildi: Bundan sonra uygulama açıldığında bu filtreler varsayılan gelir.")

            st.markdown("#### Görünüm / Risk Etiketleri (tabloyu filtreler)")
            cR1, cR2, cR3 = st.columns(3)
            with cR1:
                bo_inc_trap = st.checkbox(
    "⚠️ Trap (dahil et + göster)",
    value=True,
     key="bo_inc_trap")
            with cR2:
                bo_inc_danger = st.checkbox(
    "☠️ En tehlikeli (dahil et + göster)",
    value=True,
     key="bo_inc_danger")
            with cR3:
                bo_inc_profile = st.checkbox(
    "🧩 Profil (archetype) (dahil et + göster)",
    value=True,
     key="bo_inc_profile")

            with st.expander("🧩 Profil ikonları ne demek?", expanded=False):
                st.caption(
    "Bu ikonlar, maçın hangi 'profil/archetype' sınıfına girdiğini gösterir. "
    "NONE = özel bir profil yok (nötr).")
                st.markdown(
                    """- 🪤 **1X2_TRAP**: Favori tarafında 'trap' imzası (model ≫ market ama fiyat çok sıkışık / ters risk)
- ✅ **OU_NO_EDGE / BTTS_NO_EDGE**: Sinyal var ama edge zayıf (ince buz)
- 🟦 **OU_NONE / BTTS_NONE / 1X2_NONE**: O markette anlamlı sinyal yok
- (boş): **NONE**""")
        # Weak leagues (optional): older builds may not have the Weak_League
        # flag in league_metrics_df
        weak_leagues = []
        if isinstance(
    league_metrics_df,
     pd.DataFrame) and not league_metrics_df.empty:
            if "Weak_League" in league_metrics_df.columns and "League" in league_metrics_df.columns:
                weak_leagues = (
                    league_metrics_df.loc[league_metrics_df["Weak_League"], "League"]
                    .dropna().astype(str).unique().tolist()
                )
            # Fallback: if only ROI exists, treat negative-ROI leagues as
            # weak (conservative)
            elif "ROI" in league_metrics_df.columns and "League" in league_metrics_df.columns:
                weak_leagues = (
                    league_metrics_df.loc[league_metrics_df["ROI"] < 0, "League"]
                    .dropna().astype(str).unique().tolist()
                )

        if hide_weak_leagues and weak_leagues and "League" in pool_bets.columns:
            pool_bets = pool_bets[~pool_bets["League"].astype(
                str).isin(set(weak_leagues))]

        # --- DEDUP (upcoming data can contain repeated rows / joins can duplicate) ---
        # Keep the best row per (match, market) so the pool doesn't show the
        # same pick 3-5 times.
        _dedup_key = [c for c in ["Date", "League","HomeTeam","AwayTeam","Seçim","Odd"] if c in pool_bets.columns]
        if _dedup_key:
            pool_bets = pool_bets.sort_values(
                [c for c in ["Final_Confidence", "EV","Prob"] if c in pool_bets.columns],
                ascending=False
            )
            pool_bets = pool_bets.drop_duplicates(
                subset=_dedup_key, keep="first")

        # Ensure derived columns exist BEFORE applying extra UI filters
        # (Some runs build CONF/AutoMod fields later; without this, filters look "ignored".)
        pool_bets = ensure_amqs_columns(pool_bets)
        pool_bets = ensure_conf_percentile_columns(
            pool_bets, by_market=True)
        st.session_state["_dbg_pool_stages"]["02_after_numeric_cast"] = int(
            len(pool_bets))
        _maybe_record_col_lineage(
            pool_bets, "pool: after_numeric_cast", "pool_bets")

# --- Extra slider filters (AutoMod percentile + CONF/AutoMod levels) ---
        # NOTE: We apply these on pool_bets so both "Best Of" and "Havuz"
        # respect them.
        # AutoMod percentile range (supports AMQS_percentile as 0-1, 0-100,
        # or '25%')
        if "AMQS_percentile" in pool_bets.columns:
            _p = pool_bets["AMQS_percentile"]
            if _p.dtype == object:
                _p = _p.astype(str).str.replace("%", "", regex=False)
            _p = pd.to_numeric(_p, errors="coerce")
            # normalize to 0-100 for filtering
            _p100 = np.where(_p <= 1.0, _p * 100.0, _p)
            lo_pct, hi_pct = float(
amqs_pct_range[0]), float(
    amqs_pct_range[1])
            pool_bets = pool_bets[(_p100 >= lo_pct) & (_p100 <= hi_pct)]
        # CONF percentile range + CONF level filter
        if "CONF_percentile" in pool_bets.columns:
            lo_c, hi_c = float(conf_pct_range[0]), float(conf_pct_range[1])
            _c = pd.to_numeric(
pool_bets["CONF_percentile"],
 errors="coerce").fillna(50.0)
            pool_bets = pool_bets[(_c >= lo_c) & (_c <= hi_c)]
        if "CONF_Status" in pool_bets.columns and conf_status_sel:
            pool_bets = pool_bets[pool_bets["CONF_Status"].isin(
                conf_status_sel)]
        # AutoMod status filter
        if "AutoMod_Status" in pool_bets.columns and automod_status_sel:
            pool_bets = pool_bets[pool_bets["AutoMod_Status"].isin(
                automod_status_sel)]
        # --- DÜZELTME BAŞLANGICI: Geniş Liste için Gevşetilmiş Filtreler ---

        # Best Of çok sıkı olsa bile, Geniş Liste (Havuz) daha kapsayıcı olmalı.
        # Bu yüzden slider değerlerinden 'gevşetme payı' düşüyoruz.

        # Prob: Slider ne olursa olsun en az 0.12 puan altını da havuza al (ama
        # 0.40'ın altına inme)
        pool_min_prob = max(0.40, float(min_prob_val) - 0.12)

        # Score: Slider'dan 0.10 puan daha esnek ol (ama 0.35'in altına inme)
        pool_min_score = max(0.35, float(min_score_val) - 0.10)

        # EV: Slider pozitifse havuz için 0.0'a çek; slider negatifse bir tık
        # daha gevşet (-1.0)
        pool_min_ev = (0.0 if float(min_ev_val) >
                       0 else float(min_ev_val) - 1.0)
        st.session_state["_dbg_pool_thresholds"] = {
            "min_prob_val": float(min_prob_val),
            "min_score_val": float(min_score_val),
            "min_ev_val": float(min_ev_val),
            "pool_min_prob": float(pool_min_prob),
            "pool_min_score": float(pool_min_score),
            "pool_min_ev": float(pool_min_ev),
            "min_league_conf": float(min_league_conf),
            "min_market_conf": float(min_market_conf),
            "min_anchor_strength": float(
                st.session_state.get("gr_min_anchor_strength", 0.0)),
            "min_effective_n": int(
                st.session_state.get("gr_min_effective_n", 0)),
            "min_sim_quality": st.session_state.get(
                "knn_gate_pool_min_simq", None),
            "odds_min": st.session_state.get("odd_min", None),
            "odds_max": st.session_state.get("odd_max", None),
            "market_selection": None,
        }

        # Dynamic offsets (mevcut koddan aynen korundu)
        prob_offset = np.where(
            pool_bets['Seçim'] == 'MS X', -0.03, np.where(pool_bets['Seçim'] == 'MS 2', -0.02, 0.0))
        score_offset = np.where(
            pool_bets['Seçim'] == 'MS X', -0.02, np.where(pool_bets['Seçim'] == 'MS 2', -0.01, 0.0))

        ev_offset_map = {
            "MS 1": -0.3, "MS X": -0.1, "MS 2": -0.3,
            "2.5 Üst": +0.5, "2.5 Alt": +0.5,
            "KG Var": +0.7, "KG Yok": +0.7,
        }
        ev_offset_series = pool_bets['Seçim'].map(ev_offset_map).fillna(0.0)

        # EV Threshold series (gevşetilmiş pool_min_ev kullanılıyor)
        ev_threshold_series = pool_min_ev + ev_offset_series

        # Ensure clean index to avoid duplicate index issues in filtering stages
        pool_bets = _ensure_unique_index(pool_bets)

        q_factor = pool_bets["Seçim"].map(
            market_quality).fillna(0.5).clip(0.0, 1.0)
        prob_adj_dynamic = 0.03 * (q_factor - 0.5)
        score_adj_dynamic = 0.04 * (q_factor - 0.5)
        ev_adj_dynamic = 0.50 * (q_factor - 0.5)

        _prob_raw = pool_bets.get("Prob", np.nan)
        _prob_num = pd.to_numeric(
            _prob_raw.astype(str).str.replace("%", "", regex=False).str.replace(
                ",", ".", regex=False).str.strip(),
            errors="coerce",
        )
        _prob_dec = pd.to_numeric(pool_bets.get("Prob_dec", np.nan), errors="coerce")
        _p = _prob_dec.where(_prob_dec.notna(), _prob_num)
        _s = pd.to_numeric(pool_bets.get("Score", np.nan), errors="coerce")
        _ev = pd.to_numeric(pool_bets.get("EV", np.nan), errors="coerce")

        if pool_min_ev <= -4.0:
            ev_cond = _ev >= -20.0
        else:
            ev_cond = _ev >= (ev_threshold_series - ev_adj_dynamic)


        # GAoNCELLENM???z MASK (Gev?Yetilmi?Y de?Yerler ile)
        mask_min_prob = (_p >= (pool_min_prob + prob_offset - prob_adj_dynamic))
        mask_min_prob = _align_mask(mask_min_prob, pool_bets)

        # Auto-relax min_prob if 0 rows
        relaxed_min_prob = pool_min_prob
        prob_relax_applied = False
        if mask_min_prob.sum() == 0:
            prob_relax_steps = [pool_min_prob, 0.48, 0.45, 0.42, 0.40, 0.38]
            for p in prob_relax_steps:
                mask_min_prob = (_p >= (p + prob_offset - prob_adj_dynamic))
                if mask_min_prob.sum() >= 10:
                    relaxed_min_prob = p
                    prob_relax_applied = p < pool_min_prob
                    break
        if _dbg_col_audit_enabled():
            _mp = _align_mask(mask_min_prob, pool_bets)
            _maybe_record_col_lineage(
                pool_bets[_mp].copy(), "pool: after_min_prob", "pool_bets")

        mask_min_score = (_s >= (pool_min_score + score_offset - score_adj_dynamic))
        mask_min_ev = pd.Series(True, index=pool_bets.index, dtype=bool)
        mask_league_conf = (
            pool_bets['League_Conf'] >= max(0.10, float(min_league_conf) - 0.10))
        mask_market_conf = (
            pool_bets['Market_Conf_Score'] >= max(0.10, float(min_market_conf) - 0.10))
        _pass_prob = int(mask_min_prob.sum())
        _pass_score = int(mask_min_score.sum())
        _pass_ev = int(ev_cond.sum()) if isinstance(ev_cond, pd.Series) else 0
        _pass_prob_score = int((mask_min_prob & mask_min_score).sum())
        st.session_state["_dbg_gate_stats"] = {
            "p": {
                "count_nonnull": int(_p.notna().sum()),
                "min": float(_p.min()) if _p.notna().any() else None,
                "p05": float(_p.quantile(0.05)) if _p.notna().any() else None,
                "p50": float(_p.quantile(0.50)) if _p.notna().any() else None,
                "p95": float(_p.quantile(0.95)) if _p.notna().any() else None,
                "max": float(_p.max()) if _p.notna().any() else None,
            },
            "s": {
                "count_nonnull": int(_s.notna().sum()),
                "min": float(_s.min()) if _s.notna().any() else None,
                "p05": float(_s.quantile(0.05)) if _s.notna().any() else None,
                "p50": float(_s.quantile(0.50)) if _s.notna().any() else None,
                "p95": float(_s.quantile(0.95)) if _s.notna().any() else None,
                "max": float(_s.max()) if _s.notna().any() else None,
            },
            "ev": {
                "count_nonnull": int(_ev.notna().sum()),
                "min": float(_ev.min()) if _ev.notna().any() else None,
                "p05": float(_ev.quantile(0.05)) if _ev.notna().any() else None,
                "p50": float(_ev.quantile(0.50)) if _ev.notna().any() else None,
                "p95": float(_ev.quantile(0.95)) if _ev.notna().any() else None,
                "max": float(_ev.max()) if _ev.notna().any() else None,
            },
            "pass_prob": _pass_prob,
            "pass_score": _pass_score,
            "pass_ev": _pass_ev,
            "pass_prob_and_score": _pass_prob_score,
        }
        _th_prob = pool_min_prob + prob_offset - prob_adj_dynamic
        _th_score = pool_min_score + score_offset - score_adj_dynamic
        _min_margin = np.minimum(_p - _th_prob, _s - _th_score)
        _sample_cols = [c for c in [
            "Match_ID", "Seçim", "Prob", "Prob_dec", "Score", "EV"
        ] if c in pool_bets.columns]
        _sample_df = _ensure_unique_index(pool_bets.copy())


        _n_dbg = len(_sample_df)
        _sample_df["_min_margin"] = _dbg_to_array(_min_margin, _n_dbg)
        _sample_df["_p"] = _dbg_to_array(_p, _n_dbg)
        _sample_df["_s"] = _dbg_to_array(_s, _n_dbg)
        _sample_df["_ev"] = _dbg_to_array(_ev, _n_dbg)
        st.session_state["_dbg_gate_sample"] = _sample_df.sort_values(
            "_min_margin", ascending=False
        ).head(10)[_sample_cols + ["_min_margin"]].to_dict(orient="records")

        # EV FILTER MODE SWITCH (immediately after min_prob, before score)
        rows_after_prob = int(mask_min_prob.sum())
        original_pass_min_prob = rows_after_prob  # Store original count before dataframe replacement
        ev_mode = "hard_filter"
        ev_topk_k = None
        if rows_after_prob < 80:
            ev_mode = "topk_rank"
            ev_topk_k = 25
            # Safe EV topK: use numpy array to avoid index reindexing issues
            # Debug assertion: ensure mask aligns with pool_bets
            assert isinstance(mask_min_prob, pd.Series) and mask_min_prob.index.equals(pool_bets.index), \
                f"mask_min_prob misaligned: mask_idx={mask_min_prob.index[:5]} df_idx={pool_bets.index[:5]}"
            mask_min_prob = _align_mask(mask_min_prob, pool_bets)
            df_ev_in = pool_bets.loc[mask_min_prob].copy()
            rows_before_ev = len(df_ev_in)
            ev = pd.to_numeric(df_ev_in.get("EV", np.nan), errors="coerce").fillna(-9999.0).to_numpy()
            k = min(ev_topk_k, len(df_ev_in))
            top_pos = np.argsort(-ev)[:k]  # positions of top k
            df_ev_out = df_ev_in.iloc[top_pos].copy()
            rows_after_ev = len(df_ev_out)
            # Replace working df directly - no merge back
            pool_bets = df_ev_out
            if _dbg_col_audit_enabled():
                _maybe_record_col_lineage(
                    pool_bets, "pool: after_ev_pass1", "pool_bets")
            # Recompute masks on new pool_bets
            mask_min_prob = pd.Series(True, index=pool_bets.index, dtype=bool)  # already passed
            mask_ev = pd.Series(True, index=pool_bets.index, dtype=bool)  # all passed EV topK
        else:
            mask_ev = _ev >= (ev_threshold_series - ev_adj_dynamic)
            rows_before_ev = rows_after_prob
            rows_after_ev = int(mask_ev.sum())

        # Invariants: ensure EV stage doesn't expand the dataframe
        assert rows_after_ev <= rows_before_ev, f"EV stage expanded df: {rows_before_ev} -> {rows_after_ev}"
        if ev_mode == "topk_rank":
            assert rows_after_ev <= ev_topk_k, f"EV topK returned {rows_after_ev} > {ev_topk_k}"

        # Dynamic score relaxation for small pools
        relaxed_score_threshold = pool_min_score
        if ev_mode == "topk_rank" or rows_after_ev < 20:  # MIN_POOL_TARGET * 2
            relaxed_score_threshold = max(0.25, pool_min_score - 0.10)

        mask_min_score = (_s >= (relaxed_score_threshold + score_offset - score_adj_dynamic))

        # Combine based on EV mode
        if ev_mode == "topk_rank":
            _mask_prob_score = mask_ev & mask_min_score
        else:
            _mask_prob_score = mask_min_prob & mask_min_score

        st.session_state["_dbg_auto_relax_applied"] = False
        if _pass_prob_score == 0:
            _mask_prob_score = (mask_min_prob | mask_min_score)
            st.session_state["_dbg_auto_relax_applied"] = True
        if _dbg_col_audit_enabled():
            _mps = _align_mask(_mask_prob_score, pool_bets)
            _maybe_record_col_lineage(
                pool_bets[_mps].copy(), "pool: after_min_score", "pool_bets")

        # EV FILTER MODE SWITCH (min_prob sonrası, min_ev uygulanmadan önce)
        rows_after_prob_score = int(_mask_prob_score.sum())
        ev_mode = "hard_filter"
        ev_topk_k = None
        if rows_after_prob_score < 80:
            ev_mode = "topk_rank"
            ev_topk_k = 25
            # Safe EV topK: use numpy array to avoid index reindexing issues
            _mask_prob_score = _align_mask(_mask_prob_score, pool_bets)

            df_ev_in = pool_bets.loc[_mask_prob_score].copy()
            rows_before_ev_second = len(df_ev_in)
            ev = pd.to_numeric(df_ev_in.get("EV", np.nan), errors="coerce").fillna(-9999.0).to_numpy()
            k = min(ev_topk_k, len(df_ev_in))
            top_pos = np.argsort(-ev)[:k]  # positions of top k
            df_ev_out = df_ev_in.iloc[top_pos].copy()
            rows_after_ev_second = len(df_ev_out)
            # Replace working df directly - no merge back
            pool_bets = df_ev_out
            if _dbg_col_audit_enabled():
                _maybe_record_col_lineage(
                    pool_bets, "pool: after_ev_pass2", "pool_bets")
            # Recompute masks on new pool_bets
            _mask_prob_score = pd.Series(True, index=pool_bets.index, dtype=bool)  # already passed
            mask_ev = pd.Series(True, index=pool_bets.index, dtype=bool)  # all passed EV topK
            # Invariants
            assert rows_after_ev_second <= rows_before_ev_second, f"EV stage 2 expanded df: {rows_before_ev_second} -> {rows_after_ev_second}"
            assert rows_after_ev_second <= ev_topk_k, f"EV topK 2 returned {rows_after_ev_second} > {ev_topk_k}"
        else:
            mask_ev = _ev >= (ev_threshold_series - ev_adj_dynamic)

        mask_quality = (
            _mask_prob_score & mask_ev & mask_league_conf & mask_market_conf)

        # Small-pool auto relax score
        rows_after_ev = int(mask_quality.sum())
        if ev_mode == "topk_rank" and rows_after_ev < 10:  # MIN_POOL_TARGET approx
            # reduce pool_min_score to 0.25 for this branch
            pool_min_score_relaxed = 0.25
            # recalculate mask_min_score with relaxed threshold
            _th_score_relaxed = pool_min_score_relaxed + score_offset - score_adj_dynamic
            mask_min_score_relaxed = (_s >= _th_score_relaxed)
            if ev_mode == "topk_rank":
                _mask_prob_score_relaxed = mask_ev & mask_min_score_relaxed
            else:
                _mask_prob_score_relaxed = mask_min_prob & mask_min_score_relaxed
            if _pass_prob_score == 0:
                _mask_prob_score_relaxed = (mask_min_prob | mask_min_score_relaxed)
            mask_quality = (
                _mask_prob_score_relaxed & mask_ev & mask_league_conf & mask_market_conf)
            rows_after_ev = int(mask_quality.sum())
        if _dbg_col_audit_enabled():
            _mq = _align_mask(mask_quality, pool_bets)
            _maybe_record_col_lineage(
                pool_bets[_mq].copy(), "pool: after_quality_gate", "pool_bets")

        st.session_state["_dbg_gate_stats"]["pass_prob_and_score"] = int(_mask_prob_score.sum())
        st.session_state["_dbg_gate_stats"]["pass_all"] = int(mask_quality.sum())
        st.session_state["_dbg_pool_stages"]["04a_pass_min_prob"] = original_pass_min_prob
        st.session_state["_dbg_pool_stages"]["04b_pass_min_score"] = int(
            mask_min_score.sum())
        st.session_state["_dbg_pool_stages"]["04c_pass_min_ev"] = int(
            mask_ev.sum())
        st.session_state["_dbg_pool_stages"]["04d_pass_league_conf"] = int(
            mask_league_conf.sum())
        st.session_state["_dbg_pool_stages"]["04e_pass_market_conf"] = int(
            mask_market_conf.sum())
        st.session_state["_dbg_pool_stages"]["05_pass_prob_and_score"] = int(
            _mask_prob_score.sum())
        st.session_state["_dbg_pool_stages"]["06_pass_prob_score_league"] = int(
            (_mask_prob_score & mask_league_conf).sum())
        st.session_state["_dbg_pool_stages"]["07_pass_quality_all"] = int(
            mask_quality.sum())
        st.session_state["_dbg_pool_stages"]["__PATCH_MARKER__"] = "STAGES_SPLIT_V1"
        st.session_state["_dbg_pool_stages"]["04_after_min_prob"] = rows_after_prob
        st.session_state["_dbg_pool_stages"]["05_after_ev"] = int(mask_ev.sum())
        st.session_state["_dbg_pool_stages"]["06_after_min_score"] = int(_mask_prob_score.sum())
        st.session_state["_dbg_pool_stages"]["10_after_quality_gate"] = int(
            mask_quality.sum())
        # Add new debug
        st.session_state["_dbg_pool_stages"]["ev_mode"] = ev_mode
        st.session_state["_dbg_pool_stages"]["ev_topk_k"] = ev_topk_k
        st.session_state["_dbg_pool_stages"]["rows_before_ev"] = rows_before_ev
        st.session_state["_dbg_pool_stages"]["rows_after_ev"] = rows_after_ev
        st.session_state["_dbg_pool_stages"]["relaxed_score_threshold"] = relaxed_score_threshold
        st.session_state["_dbg_pool_stages"]["rows_before_min_score"] = int(mask_ev.sum())
        st.session_state["_dbg_pool_stages"]["rows_after_min_score"] = int(_mask_prob_score.sum())
        st.session_state["_dbg_pool_stages"]["prob_relax_applied"] = prob_relax_applied
        st.session_state["_dbg_pool_stages"]["final_min_prob_used"] = relaxed_min_prob
        st.session_state["_dbg_pool_stages"]["rows_after_prob_relax"] = rows_after_prob
        pool_mask = mask_quality

        # --- DÜZELTME BİTİŞİ ---

        coupon_mode = st.radio(
            "Kupon Tipi:",
            ["🎯 Sadece 1X2", "🎛 Karışık Kupon (1X2 + OU + KG)"],
            index=1,
            horizontal=True
        )

        if coupon_mode.startswith("🎯"):
            mask_market = pool_bets['Seçim'].isin(["MS 1", "MS X", "MS 2"])
        else:
            mask_market = pd.Series(True, index=pool_bets.index)
        st.session_state["_dbg_pool_stages"]["03_after_market_filter"] = int(
            (pool_mask & mask_market).sum())
        pool_mask = pool_mask & mask_market

        pool_mask = _align_mask(pool_mask, pool_bets)
        pool_bets = pool_bets[pool_mask].copy()

        # -----------------------------------------------------------
        # Auto-Relax (Small Pool) - broad, controlled relaxation
        # If pool is tiny/empty, relax a few core gates progressively.
        # -----------------------------------------------------------
        _auto_relax_on = bool(
    st.session_state.get(
        'gr_autorelax_small_pool', True))
        if _auto_relax_on:
            _style = str(st.session_state.get('gr_style_mode', '✅ GÜVENLİ'))
            _min_target_map = {
    '🔒 BANKO': 4,
    '🧠 ZEKİ': 5,
    '💎 VALUE': 7,
    '✅ GÜVENLİ': 8,
     '💣 SÜRPRİZ': 10}
            _min_target = int(_min_target_map.get(_style, 6))
            if pool_bets is None or getattr(
    pool_bets, 'empty', True) or (
        len(pool_bets) < _min_target):
                _base = pool_bets_base.copy() if 'pool_bets_base' in locals(
                ) and pool_bets_base is not None else None
                if _base is not None and not getattr(_base, 'empty', True):
                    def _simple_relax(
    df_in: pd.DataFrame,
    dprob: float,
    dscore: float,
    danchor: float,
     dn: int) -> pd.DataFrame:
                        df = df_in.copy()
                        if 'Prob' in df.columns:
                            df['Prob'] = pd.to_numeric(
                                df['Prob'], errors='coerce')
                            df = df[df['Prob'].fillna(0.0) >= max(
                                float(min_prob_val) - dprob, 0.45)]
                        if 'Score' in df.columns:
                            df['Score'] = pd.to_numeric(
                                df['Score'], errors='coerce')
                            df = df[df['Score'].fillna(-999.0)
                                                       >= (float(min_score_val) - dscore)]
                        if 'EV' in df.columns:
                            df['EV'] = pd.to_numeric(df['EV'], errors='coerce')
                            _ev_floor = float(min_ev_val)
                            if _ev_floor >= 0:
                                _ev_floor = 0.0
                            df = df[df['EV'].fillna(-999.0) >= _ev_floor]
                        if 'SIM_ANCHOR_STRENGTH' in df.columns:
                            df['SIM_ANCHOR_STRENGTH'] = pd.to_numeric(
                                df['SIM_ANCHOR_STRENGTH'], errors='coerce')
                            df = df[df['SIM_ANCHOR_STRENGTH'].fillna(0.0) >= max(
                                float(min_anchor_strength_val) - danchor, 0.0)]
                        if 'EFFECTIVE_N' in df.columns:
                            df['EFFECTIVE_N'] = pd.to_numeric(
                                df['EFFECTIVE_N'], errors='coerce')
                            df = df[df['EFFECTIVE_N'].fillna(0.0) >= max(
                                int(min_effective_n_val) - int(dn), 0)]
                        return df

                    _steps = [
                        {'dprob': 0.00, 'dscore': 0.00, 'danchor': 0.00, 'dn': 0},
                        {'dprob': 0.02, 'dscore': 0.02, 'danchor': 0.05, 'dn': 5},
                        {'dprob': 0.04, 'dscore': 0.03, 'danchor': 0.08, 'dn': 10},
                        {'dprob': 0.06, 'dscore': 0.05, 'danchor': 0.12, 'dn': 15},
                        {'dprob': 0.09, 'dscore': 0.07, 'danchor': 0.16, 'dn': 20},
                        {'dprob': 0.12, 'dscore': 0.10, 'danchor': 0.20, 'dn': 25},
                    ]
                    _picked = None
                    for _stp in _steps:
                        _cand = _simple_relax(_base, **_stp)
                        if _cand is not None and not getattr(
    _cand,
    'empty',
    True) and (
        len(_cand) >= min(
            _min_target,
             3)):
                            _picked = _cand
                            break
                    if _picked is not None:
                        pool_bets = _picked.copy()
                        st.session_state['__GR_AUTO_RELAX_APPLIED__'] = True


        def is_good_draw(
    row,
    min_prob_slider: float,
     min_ev_slider: float) -> bool:
            """Extra sanity filter for MS X. Must loosen when sliders are loose."""
            if row.get('Seçim') != 'MS X':
                return True

            p = float(row.get('Prob', 0.0) or 0.0)

            # Base thresholds (loosened by sliders)
            base_prob_x = 0.27
            # allow down to ~0.17 when slider is low
            req_prob = max(min_prob_slider, base_prob_x - 0.10)
            if p < req_prob:
                return False

            p_h = float(row.get('P_Home_Final', 0.0) or 0.0)
            p_x = float(row.get('P_Draw_Final', 0.0) or 0.0)
            p_a = float(row.get('P_Away_Final', 0.0) or 0.0)
            max_p = max(p_h, p_a)

            # If slider is strict, keep the draw close to best side. If loose,
            # allow more gap.
            gap_allow = 0.06 + max(0.0,
     0.45 - min_prob_slider) * 0.20  # up to ~0.15
            if p_x < (max_p - gap_allow):
                return False

            # Elo gap filter (loosened)
            elo_cap = 0.6 + max(0.0,
     0.45 - min_prob_slider) * 0.8  # up to ~1.0
            if float(row.get('Elo_Gap_Abs', 1.0) or 1.0) > elo_cap:
                return False

            # Total goals band (loosened)
            xg = float(row.get('Book_Exp_Total_Goals', 2.5) or 2.5)
            lo = 1.7 - max(0.0, 0.45 - min_prob_slider) * 0.7  # down to ~1.2
            hi = 2.9 + max(0.0, 0.45 - min_prob_slider) * 1.0  # up to ~3.9
            if not (lo <= xg <= hi):
                return False

            # Optional: EV sanity only when user is strict on EV
            ev = float(row.get('EV', 0.0) or 0.0)
            if min_ev_slider >= 0.5 and ev < min_ev_slider:
                return False

            return True

        def is_good_kg(
    row,
    min_prob_slider: float,
     min_ev_slider: float) -> bool:
            """Extra sanity filter for KG Var/Yok. Must follow sliders; avoid hard 'EV>=4' locks."""
            s = row.get('Seçim')
            if s not in ('KG Var', 'KG Yok'):
                return True

            p = float(row.get('Prob', 0.0) or 0.0)
            ev = float(row.get('EV', 0.0) or 0.0)

            # Slightly different bases; sliders can relax below these.
            base_prob = 0.55 if s == 'KG Var' else 0.53
            # allow down to ~0.43 when slider is low
            req_prob = max(min_prob_slider, base_prob - 0.10)

            # EV: when slider is very loose (<0), don't force a positive EV hard floor.
            # When slider is strict (>=0), respect it.
            if min_ev_slider < 0:
                req_ev = min_ev_slider  # user explicitly allows negative EV for debugging
            else:
                req_ev = max(min_ev_slider, 0.5)

            if p < req_prob:
                return False
            if ev < req_ev:
                return False

            return True

        MIN_POOL_TARGET = 10
        _pre_stage11_rows = int(len(pool_bets)) if pool_bets is not None else 0
        _min_ev_for_gate = float(min_ev_val)
        if _pre_stage11_rows < 25 or _pre_stage11_rows < MIN_POOL_TARGET:
            _min_ev_for_gate = min(_min_ev_for_gate, -10.0)

        def _safe_series(df, col, default):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index)

        if not pool_bets.empty:
            pool_bets_base = pool_bets.copy()

            sel_col = "Seçim" if "Seçim" in pool_bets.columns else None
            if sel_col is None:
                pass
            else:
                st.session_state["_dbg_stage11"] = {
                    "pre_rows": int(len(pool_bets)),
                    "sel_counts_pre": pool_bets[sel_col].value_counts().to_dict(),
                    "match_ids_pre": (
                        pool_bets["Match_ID"].astype(str).head(10).tolist()
                        if "Match_ID" in pool_bets.columns else []
                    ),
                }
                _sel = pool_bets[sel_col].astype(str).fillna("")
                mask_keep = pd.Series(True, index=pool_bets.index, dtype=bool)

                is_draw_market = _sel.isin(["MS X", "X", "Draw", "Berabere"])
                mask_draw = None
                if is_draw_market.any():
                    mask_draw = pool_bets[is_draw_market].apply(
                        lambda r: is_good_draw(
                            r,
                            min_prob_slider=min_prob_val,
                            min_ev_slider=_min_ev_for_gate),
                        axis=1)
                    mask_draw_aligned = mask_draw.reindex(pool_bets.index).fillna(True)
                    _draw_fail = is_draw_market & (~mask_draw_aligned)
                    # Get DRAW_PRESSURE with NaN for missing values (not default 1.0)
                    _draw_pressure = pd.to_numeric(pool_bets.get("DRAW_PRESSURE", pd.Series(np.nan, index=pool_bets.index)), errors="coerce")
                    _eff_n = _safe_series(pool_bets, "EFFECTIVE_N", 999.0)
                    _simq = _safe_series(pool_bets, "SIM_QUALITY", 999.0)
                    # Only apply extreme conditions when we have real draw evidence
                    has_draw_info = _draw_pressure.notna() & (("EFFECTIVE_N" in pool_bets.columns) | ("SIM_QUALITY" in pool_bets.columns))
                    # Only apply extreme conditions when quality metrics are present
                    has_quality = ("EFFECTIVE_N" in pool_bets.columns) and ("SIM_QUALITY" in pool_bets.columns)
                    if has_quality and has_draw_info:
                        _extreme_fail = _draw_fail & (
                            (_draw_pressure < 0.10) | (_eff_n < 5) | (_simq < 0.20)
                        )
                    else:
                        # No quality metrics or draw info available, treat as unknown - no extreme fails
                        _extreme_fail = pd.Series(False, index=pool_bets.index)
                    _soft_fail = _draw_fail & ~_extreme_fail
                    if _soft_fail.any():
                        if "_GATE_PENALTY" not in pool_bets.columns:
                            pool_bets["_GATE_PENALTY"] = 0.0
                        pool_bets.loc[_soft_fail, "_GATE_PENALTY"] = (
                            pool_bets.loc[_soft_fail, "_GATE_PENALTY"] + 0.15
                        )
                        for _sc in ["Score", "GoldenScore"]:
                            if _sc in pool_bets.columns:
                                pool_bets.loc[_soft_fail, _sc] = (
                                    pd.to_numeric(
                                        pool_bets.loc[_soft_fail, _sc], errors="coerce"
                                    ).fillna(0.0) * (1.0 - 0.15)
                                )
                        if "_GATE_FLAGS" not in pool_bets.columns:
                            pool_bets["_GATE_FLAGS"] = [[] for _ in range(len(pool_bets))]
                        pool_bets.loc[_soft_fail, "_GATE_FLAGS"] = pool_bets.loc[_soft_fail, "_GATE_FLAGS"].apply(
                            lambda r: r + ["draw_gate_soft"]
                        )
                    mask_keep.loc[is_draw_market] = ~_extreme_fail.loc[is_draw_market]

                is_kg_market = _sel.isin(
                    ["KG Var", "KG Yok", "BTTS Yes", "BTTS No"])
                mask_kg = None
                if is_kg_market.any():
                    mask_kg = pool_bets[is_kg_market].apply(
                        lambda r: is_good_kg(
                            r,
                            min_prob_slider=min_prob_val,
                             min_ev_slider=_min_ev_for_gate),
                        axis=1)
                    mask_keep.loc[is_kg_market] = mask_kg

                # MUST-HAVE gates: missing or all-NaN => FAIL (no silent pass)
                _must_gate_issues = {}
                for _col in ["SIM_ANCHOR_STRENGTH", "EFFECTIVE_N", "SIM_QUALITY"]:
                    if _col not in pool_bets.columns or pool_bets[_col].isna().all():
                        _must_gate_issues[_col] = "missing_or_all_nan"
                        mask_keep.loc[:] = False
                if _must_gate_issues:
                    st.session_state.setdefault("_dbg_pool_stages", {})
                    st.session_state["_dbg_pool_stages"]["must_have_gate_fail"] = _must_gate_issues
                    st.warning(
                        f"Required columns missing/all-NaN -> gate FAIL: {_must_gate_issues}"
                    )

                drop_mask = ~mask_keep
                drop_counts = {}
                drop_reasons = []
                if drop_mask.any():
                    reasons = pd.Series([[] for _ in range(len(pool_bets))], index=pool_bets.index)
                    if mask_draw is not None:
                        _fail = is_draw_market & ~mask_draw.reindex(pool_bets.index, fill_value=True)
                        # Only extreme fails are actually dropped - count them
                        _extreme_dropped = _fail & drop_mask
                        drop_counts["draw_gate_extreme"] = int(_extreme_dropped.sum())
                        reasons[_extreme_dropped] = reasons[_extreme_dropped].apply(lambda r: r + ["draw_gate_extreme"])
                    if mask_kg is not None:
                        _fail = is_kg_market & ~mask_kg.reindex(pool_bets.index, fill_value=True)
                        _kg_dropped = _fail & drop_mask
                        drop_counts["kg_gate_fail"] = int(_kg_dropped.sum())
                        reasons[_kg_dropped] = reasons[_kg_dropped].apply(lambda r: r + ["kg_gate_fail"])
                    if "SIM_ANCHOR_STRENGTH" in pool_bets.columns and not pool_bets["SIM_ANCHOR_STRENGTH"].isna().all():
                        _min_anchor = float(st.session_state.get("gr_min_anchor_strength", 0.0))
                        _fail = pd.to_numeric(pool_bets["SIM_ANCHOR_STRENGTH"], errors="coerce").fillna(0.0) < _min_anchor
                        drop_counts["anchor_strength_fail"] = int((_fail & drop_mask).sum())
                        reasons[_fail & drop_mask] = reasons[_fail & drop_mask].apply(lambda r: r + ["anchor_strength_fail"])
                    if "EFFECTIVE_N" in pool_bets.columns and not pool_bets["EFFECTIVE_N"].isna().all():
                        _min_en = float(st.session_state.get("gr_min_effective_n", 0))
                        _fail = pd.to_numeric(pool_bets["EFFECTIVE_N"], errors="coerce").fillna(0.0) < _min_en
                        drop_counts["effective_n_fail"] = int((_fail & drop_mask).sum())
                        reasons[_fail & drop_mask] = reasons[_fail & drop_mask].apply(lambda r: r + ["effective_n_fail"])
                    if "SIM_QUALITY" in pool_bets.columns and not pool_bets["SIM_QUALITY"].isna().all():
                        _min_simq = st.session_state.get("knn_gate_pool_min_simq", None)
                        if _min_simq is not None:
                            _fail = pd.to_numeric(pool_bets["SIM_QUALITY"], errors="coerce").fillna(0.0) < float(_min_simq)
                            drop_counts["sim_quality_fail"] = int((_fail & drop_mask).sum())
                            reasons[_fail & drop_mask] = reasons[_fail & drop_mask].apply(lambda r: r + ["sim_quality_fail"])
                    if "Odd" in pool_bets.columns:
                        _odd_min = st.session_state.get("odd_min", None)
                        _odd_max = st.session_state.get("odd_max", None)
                        _odd = pd.to_numeric(pool_bets["Odd"], errors="coerce")
                        _fail = pd.Series(False, index=pool_bets.index)
                        if _odd_min is not None:
                            _fail = _fail | (_odd < float(_odd_min))
                        if _odd_max is not None:
                            _fail = _fail | (_odd > float(_odd_max))
                        drop_counts["odds_fail"] = int((_fail & drop_mask).sum())
                        reasons[_fail & drop_mask] = reasons[_fail & drop_mask].apply(lambda r: r + ["odds_fail"])
                    if "Trap_Flag" in pool_bets.columns:
                        _fail = pd.to_numeric(pool_bets["Trap_Flag"], errors="coerce").fillna(0).astype(bool)
                        drop_counts["trap_flag"] = int((_fail & drop_mask).sum())
                        reasons[_fail & drop_mask] = reasons[_fail & drop_mask].apply(lambda r: r + ["trap_flag"])
                    if "Danger_Flag" in pool_bets.columns:
                        _fail = pd.to_numeric(pool_bets["Danger_Flag"], errors="coerce").fillna(0).astype(bool)
                        drop_counts["danger_flag"] = int((_fail & drop_mask).sum())
                        reasons[_fail & drop_mask] = reasons[_fail & drop_mask].apply(lambda r: r + ["danger_flag"])
                    if "Market_Conf_Score" in pool_bets.columns:
                        _min_market_conf = float(st.session_state.get("gr_min_market_conf", 0.0))
                        _fail = pd.to_numeric(pool_bets["Market_Conf_Score"], errors="coerce").fillna(0.0) < _min_market_conf
                        drop_counts["market_conf_fail"] = int((_fail & drop_mask).sum())
                        reasons[_fail & drop_mask] = reasons[_fail & drop_mask].apply(lambda r: r + ["market_conf_fail"])
                    drop_reasons = [
                        {
                            "Match_ID": pool_bets.loc[i].get("Match_ID", None),
                            "Seçim": pool_bets.loc[i].get(sel_col, None),
                            "reasons": reasons.loc[i],
                            "DRAW_PRESSURE": pool_bets.loc[i].get("DRAW_PRESSURE", None),
                            "PROFILE_CONF": pool_bets.loc[i].get("PROFILE_CONF", None),
                            "Odd": pool_bets.loc[i].get("Odd", None),
                            "Prob": pool_bets.loc[i].get("Prob", None),
                            "EV": pool_bets.loc[i].get("EV", None),
                            "Score": pool_bets.loc[i].get("Score", None),
                            "SIM_QUALITY": pool_bets.loc[i].get("SIM_QUALITY", None),
                            "EFFECTIVE_N": pool_bets.loc[i].get("EFFECTIVE_N", None),
                        }
                        for i in pool_bets.index[drop_mask][:5]
                    ]
                # Create debug sample from reset index dataframe to avoid duplicate index issues
                _debug_df = pool_bets.reset_index(drop=True)
                _debug_drop_mask = drop_mask.reindex(_debug_df.index).fillna(False) if isinstance(drop_mask, pd.Series) else pd.Series(drop_mask, index=_debug_df.index)
                _debug_keep_mask = mask_keep.reindex(_debug_df.index).fillna(False) if isinstance(mask_keep, pd.Series) else pd.Series(mask_keep, index=_debug_df.index)

                st.session_state["_dbg_stage11_drop_sample"] = [
                    {
                        "Match_ID": row.get("Match_ID", None),
                        "Seçim": row.get(sel_col, None),
                        "reasons": [],  # reasons not available in this context
                        "DRAW_PRESSURE": row.get("DRAW_PRESSURE", None),
                        "PROFILE_CONF": row.get("PROFILE_CONF", None),
                        "Odd": row.get("Odd", None),
                        "Prob": row.get("Prob", None),
                        "EV": row.get("EV", None),
                        "Score": row.get("Score", None),
                        "SIM_QUALITY": row.get("SIM_QUALITY", None),
                        "EFFECTIVE_N": row.get("EFFECTIVE_N", None),
                    }
                    for _, row in _debug_df[_debug_drop_mask].drop_duplicates(['Match_ID', sel_col]).head(5).iterrows()
                ]
                st.session_state["_dbg_stage11_pass_sample"] = [
                    {
                        "Match_ID": row.get("Match_ID", None),
                        "Seçim": row.get(sel_col, None),
                        "DRAW_PRESSURE": row.get("DRAW_PRESSURE", None),
                        "PROFILE_CONF": row.get("PROFILE_CONF", None),
                        "Odd": row.get("Odd", None),
                        "Prob": row.get("Prob", None),
                        "EV": row.get("EV", None),
                        "Score": row.get("Score", None),
                        "SIM_QUALITY": row.get("SIM_QUALITY", None),
                        "EFFECTIVE_N": row.get("EFFECTIVE_N", None),
                        "gate_flags": row.get("_GATE_FLAGS", []),
                    }
                    for _, row in _debug_df[_debug_keep_mask].drop_duplicates(['Match_ID', sel_col]).head(5).iterrows()
                ]

                pool_bets = pool_bets[mask_keep].copy()
                st.session_state["_dbg_stage11"].update({
                    "post_rows": int(len(pool_bets)),
                    "sel_counts_post": pool_bets[sel_col].value_counts().to_dict() if not pool_bets.empty else {},
                    "match_ids_post": (
                        pool_bets["Match_ID"].astype(str).head(10).tolist()
                        if "Match_ID" in pool_bets.columns else []
                    ),
                })
        st.session_state["_dbg_pool_stages"]["11_after_market_specific_gate"] = int(
            len(pool_bets))
        if not pool_bets.empty:
            _r = pool_bets.iloc[0]
            st.session_state["_dbg_pool_last_row_pre_small_pool"] = {
                "Match_ID": _r.get("Match_ID", None),
                "Seçim": _r.get("Seçim", None),
                "Prob": _r.get("Prob", None),
                "Score": _r.get("Score", None),
                "EV": _r.get("EV", None),
            }
        else:
            st.session_state["_dbg_pool_last_row_pre_small_pool"] = {}
        # OPTION A/B: Small-pool handling
        auto_relax_small_pool = bool(
    st.session_state.get(
        "gr_auto_relax_small_pool", True))

        if len(pool_bets) < MIN_POOL_TARGET:
            pre_small_pool = pool_bets.copy()
            st.session_state["_dbg_small_pool_pre_rows"] = int(len(pre_small_pool)) if pre_small_pool is not None else 0
            st.session_state["_dbg_pool_stages"]["12a_small_pool_branch"] = (
                "auto_relax" if auto_relax_small_pool else "warn_only")
            if auto_relax_small_pool and 'pool_bets_base' in locals(
            ) and pool_bets_base is not None and not pool_bets_base.empty:
                _pre_relax = pool_bets.copy()
                _pre_relax_len = int(len(pool_bets))
                # Try controlled relaxation steps (keeps style intent, avoids
                # empty outputs)
                relax_steps = [
                    {"dprob": 0.00, "force_ev_floor": None},
                    {"dprob": 0.02, "force_ev_floor": 0.0},
                    {"dprob": 0.04, "force_ev_floor": 0.0},
                    {"dprob": 0.06, "force_ev_floor": 0.0},
                ]

                def _filter_pool_relaxed(
    df_in: pd.DataFrame,
    dprob: float = 0.0,
     force_ev_floor: float | None = None) -> pd.DataFrame:
                    df = df_in.copy()

                    def is_good_draw_relaxed(row):
                        p = float(
    pd.to_numeric(
        row.get(
            "Prob",
            0),
             errors="coerce") or 0)
                        ev = float(
    pd.to_numeric(
        row.get(
            "EV",
            0),
             errors="coerce") or 0)
                        odds = float(
    pd.to_numeric(
        row.get(
            "Odd",
            0),
             errors="coerce") or 0)

                        # Original logic but with relaxed prob floor and
                        # optional EV floor override
                        if odds < 2.40:
                            req_prob = max(float(min_prob_val) - dprob, 0.50)
                            req_ev = 0.5 if force_ev_floor is None else force_ev_floor
                        else:
                            req_prob = max(float(min_prob_val) - dprob, 0.45)
                            req_ev = 0.0 if force_ev_floor is None else force_ev_floor

                        return (p >= req_prob) and (ev >= req_ev)

                    def is_good_kg_relaxed(row):
                        p = float(
    pd.to_numeric(
        row.get(
            "Prob",
            0),
             errors="coerce") or 0)
                        ev = float(
    pd.to_numeric(
        row.get(
            "EV",
            0),
             errors="coerce") or 0)
                        odds = float(
    pd.to_numeric(
        row.get(
            "Odd",
            0),
             errors="coerce") or 0)

                        # Slightly more permissive than draw
                        if odds < 2.10:
                            req_prob = max(float(min_prob_val) - dprob, 0.50)
                            req_ev = 0.5 if force_ev_floor is None else force_ev_floor
                        else:
                            req_prob = max(float(min_prob_val) - dprob, 0.45)
                            req_ev = 0.0 if force_ev_floor is None else force_ev_floor

                        return (p >= req_prob) and (ev >= req_ev)

                    df = df[df.apply(
                        lambda r: is_good_draw_relaxed(r), axis=1)]
                    df = df[df.apply(lambda r: is_good_kg_relaxed(r), axis=1)]
                    return df

                _picked = pool_bets.copy()
                applied = None
                for stp in relax_steps:
                    cand = _filter_pool_relaxed(
    pool_bets_base,
    dprob=stp["dprob"],
     force_ev_floor=stp["force_ev_floor"])
                    if len(cand) >= 5:
                        _picked = cand
                        applied = stp
                        break
                    _picked = cand
                    applied = stp

                pool_bets = _picked
                st.session_state["_dbg_small_pool_relaxed_rows"] = int(len(pool_bets)) if pool_bets is not None else 0
                if pool_bets is None or getattr(pool_bets, "empty", True):
                    st.session_state["_dbg_small_pool_fallback"] = True
                    pool_bets = pre_small_pool if isinstance(pre_small_pool, pd.DataFrame) else pool_bets
                else:
                    st.session_state["_dbg_small_pool_fallback"] = False
                st.session_state["_dbg_small_pool_handling"] = {
                    "len_before_relax": _pre_relax_len,
                    "len_after_relax": int(len(pool_bets)) if pool_bets is not None else 0,
                    "applied_step": applied,
                    "mask_cols": ["Prob", "EV", "Odd"],
                    "relax_steps": relax_steps,
                    "min_prob_val": float(min_prob_val),
                    "min_score_val": float(min_score_val),
                    "min_ev_val": float(min_ev_val),
                }
                if pool_bets is None or pool_bets.empty:
                    st.session_state["_dbg_pool_stages"]["12c_fallback_prev_nonempty"] = int(len(pre_small_pool))
                    pool_bets = pre_small_pool
                if pool_bets is not None and not pool_bets.empty:
                    _sort_cols = [c for c in ["GoldenScore", "Score", "EV", "Prob_dec"] if c in pool_bets.columns]
                    st.session_state["_dbg_small_pool_sort_cols"] = _sort_cols
                    K = 25
                    if _sort_cols:
                        pool_bets = pool_bets.sort_values(
                            _sort_cols, ascending=[False] * len(_sort_cols)
                        ).head(K).copy()
                _src = st.session_state.get("pool_bets_raw", None)
                if (pool_bets is None or pool_bets.empty) and isinstance(_src, pd.DataFrame) and len(_src) > 0:
                    sort_cols = [c for c in ["GoldenScore", "Score", "EV", "Prob_dec", "Prob"] if c in _src.columns]
                    if sort_cols:
                        pool_bets = _src.sort_values(
                            sort_cols, ascending=[False] * len(sort_cols)
                        ).head(25).copy()
                st.info(
    f"Havuz küçük kaldığı için otomatik gevşettim: Prob -{
        applied['dprob']:.2f}, EV floor={
            applied['force_ev_floor']}. (son havuz: {
                len(pool_bets)})")
            else:
                st.warning(
    "Havuz çok küçük kaldı ({} maç). Daha fazla maç için Stil presetini sağa kaydır veya Gelişmiş filtrelerde eşikleri düşür.".format(
        len(pool_bets)))
        else:
            st.session_state["_dbg_pool_stages"]["12a_small_pool_branch"] = "skip"
        st.session_state["_dbg_pool_stages"]["12_after_small_pool_handling"] = int(
            len(pool_bets))
        if len(pool_bets) == 0:
            st.session_state["_dbg_pool_stages"]["12b_emptied_after_small_pool"] = True
            st.session_state["_dbg_pool_last_row_before_empty"] = st.session_state.get(
                "_dbg_pool_last_row_pre_small_pool", {})
        else:
            st.session_state["_dbg_pool_stages"]["12b_emptied_after_small_pool"] = False

        # ✅ PATCH 2: Pool seviyesinde pazar başı üst limit
        if not pool_bets.empty:
            _r = pool_bets.iloc[0]
            st.session_state["_dbg_pool_last_row_pre_market_cap"] = {
                "Match_ID": _r.get("Match_ID", None),
                "Seçim": _r.get("Seçim", None),
                "Prob": _r.get("Prob", None),
                "Score": _r.get("Score", None),
                "EV": _r.get("EV", None),
            }
            capped_parts = []
            # Robust: selection column name may vary; default to 'Seçim' when
            # present
            _sel_col_pool = 'Seçim' if 'Seçim' in pool_bets.columns else None
            if _sel_col_pool is None:
                for _c in ('Secim', 'Selection','Pick','PICK','Market','BET_TYPE','bet_type'):
                    if _c in pool_bets.columns:
                        _sel_col_pool = _c
                        break
            if _sel_col_pool is None:
                _norm = {re.sub(r'\s+', '',str(c)).lower(): c for c in pool_bets.columns}
                for _k in ('seçim', 'secim','selection','pick','market','bettype','bet_type'):
                    if _k in _norm:
                        _sel_col_pool = _norm[_k]
                        break
            if _sel_col_pool is None:
                _sel_col_pool = 'Seçim'
            for m, g in pool_bets.groupby(_sel_col_pool):
                limit = POOL_MAX_PER_MARKET.get(m, len(g))
                g_sorted = g.sort_values(
                    by=["Score", "EV"], ascending=[False, False])
                capped_parts.append(g_sorted.head(limit))
            pool_bets = pd.concat(capped_parts, ignore_index=True)
        st.session_state["_dbg_pool_stages"]["13a_market_cap_applied"] = bool(
            not getattr(pool_bets, "empty", True))
        st.session_state["_dbg_pool_stages"]["13_after_market_cap"] = int(
            len(pool_bets))
        if len(pool_bets) == 0:
            st.session_state["_dbg_pool_stages"]["13b_emptied_after_market_cap"] = True
            _pre = st.session_state.get("_dbg_pool_last_row_pre_market_cap", {})
            if _pre:
                st.session_state["_dbg_pool_last_row_before_empty"] = _pre
        else:
            st.session_state["_dbg_pool_stages"]["13b_emptied_after_market_cap"] = False
        if _dbg_col_audit_enabled():
            _maybe_record_col_lineage(
                pool_bets, "pool: final", "pool_bets")
            _store_col_audit(pool_bets, "pool_bets_final")

        if not bets_df.empty:
            c_d1, c_d2 = st.columns(2)
            c_d1.write(f"TÜM BETLER ({len(bets_df)}):")
            _sel_col_all = 'Seçim' if 'Seçim' in bets_df.columns else None
            if _sel_col_all is None:
                for _c in ('Secim', 'Selection','Pick','PICK','Market','BET_TYPE','bet_type'):
                    if _c in bets_df.columns:
                        _sel_col_all = _c
                        break
            if _sel_col_all is None:
                _norm = {re.sub(r'\s+', '',str(c)).lower(): c for c in bets_df.columns}
                for _k in ('seçim', 'secim','selection','pick','market','bettype','bet_type'):
                    if _k in _norm:
                        _sel_col_all = _norm[_k]
                        break
            if _sel_col_all is None:
                c_d1.caption('Seçim kolonu bulunamadı (tüm betler).')
            else:
                c_d1.dataframe(
    bets_df[_sel_col_all].value_counts().to_frame().T)
            # -----------------------------------------------------------
            # UI: Rozet Rehberi (Karakter -> Oyun Tarzı -> Tavsiye)
            # -----------------------------------------------------------
            badge_guide_df = pd.DataFrame([
                {"Rozet": "🔒BNK", "Oyun Tarzı": "Garanti", "Tavsiye": "3-4 maçı birleştir, arkana yaslan."},
                {"Rozet": "🧠ZEK", "Oyun Tarzı": "Profesyonel", "Tavsiye": "Tek maç gir. Sistemin zekasına güven."},
                {"Rozet": "💎VAL", "Oyun Tarzı": "Yatırımcı", "Tavsiye": "Oran yüksek olduğu için sistem kuponlarında kullan."},
                {"Rozet": "✅GÜV", "Oyun Tarzı": "Standart", "Tavsiye": "Bankoların yanına 1-2 tane ekle."},
                {"Rozet": "💣SPR", "Oyun Tarzı": "Eğlence", "Tavsiye": "Çerez parasına sistem kuponu yap."},
            ])
            c_d1.markdown("**Rozet Rehberi**")
            c_d1.dataframe(
badge_guide_df,
use_container_width=True,
 hide_index=True)

            c_d2.write(f"HAVUZ (Pool) BETLER ({len(pool_bets)}):")
            if not pool_bets.empty:
                c_d2.dataframe(pool_bets['Seçim'].value_counts().to_frame().T)

                c_d2.write("Pool EV / Prob özet:")
                tmp = pool_bets.copy()
                tmp['EV_Display'] = tmp['EV'].clip(-8.0, 8.0)
                summary = (
                    tmp.groupby('Seçim')[['Prob', 'EV_Display', 'Score']]
                       .agg(['mean', 'count'])
                       .rename(columns={'EV_Display': 'EV_%'})
                )
                c_d2.dataframe(summary)

                # -----------------------------
                # Similarity Anchor Diagnostics (global)
                # -----------------------------
                with st.expander("🧲 Similarity Anchor Diagnostics", expanded=False):
                    dist = st.session_state.get("sim_anchor_dist", {})
                    mean_strength = st.session_state.get(
                        "sim_anchor_strength_mean", None)
                    if isinstance(dist, dict) and dist:
                        st.write({k: f"{v*100:.1f}%" for k, v in dist.items()})
                    else:
                        st.caption(
                            "Anchor dağılımı henüz hesaplanmadı (havuz boş olabilir).")
                    if mean_strength is not None:
                        st.caption(
    f"Ortalama Anchor Strength: {
        float(mean_strength):.3f}")

            with st.expander("📊 Pazar Dağılımı (Geniş Liste Debug)", expanded=False):
                # Robust: selection column name may vary depending on upstream
                # schema
                _sel_col = None
                for _c in ('Seçim', 'Secim','Selection','Pick','PICK','Market','BET_TYPE','bet_type'):
                    if _c in pool_bets.columns:
                        _sel_col = _c
                        break
                if _sel_col is None:
                    _norm = {re.sub(r'\s+', '',str(c)).lower(): c for c in pool_bets.columns}
                    for _k in ('seçim', 'secim','selection','pick','market','bettype','bet_type'):
                        if _k in _norm:
                            _sel_col = _norm[_k]
                            break
                if _sel_col is not None:
                    st.write(pool_bets[_sel_col].value_counts())
                else:
                    st.caption('Seçim kolonu bulunamadı (debug).')
        else:
            _w = getattr(locals().get("c_d2", None), "warning", None)
            if callable(_w):
                c_d2.warning("Havuz boş! Filtreleri gevşetmeyi deneyin.")
            else:
                st.warning("Havuz boş! Filtreleri gevşetmeyi deneyin.")

        quota_map = BESTOF_QUOTA_MAP.copy()
        market_min_conf_ev = BESTOF_MARKET_MIN_CONF_EV.copy()

        # NEW: Similarity Anchor eligibility gates for BestOf (do NOT affect
        # Pool view)
        pool_bets_bestof = pool_bets.copy() if pool_bets is not None else pd.DataFrame()
        pool_bets_bestof = _ensure_unique_index(pool_bets_bestof)
        # --- Best practice: keep Pool vs BestOf distinct, but make BestOf gate *relative* to preset
        # Pool uses base slider/preset values; BestOf applies small strict
        # offsets to reduce "two independent systems" drift.
        try:
            _base_anchor = float(bestof_anchor_min)
        except Exception:
            _base_anchor = 0.0
        try:
            _base_simq = float(bestof_simq_min)
        except Exception:
            _base_simq = 0.0
        BESTOF_ANCHOR_OFFSET = 0.03
        BESTOF_SIMQ_OFFSET = 0.02
        bestof_anchor_min = max(
            0.0, min(1.0, _base_anchor + BESTOF_ANCHOR_OFFSET))
        bestof_simq_min = max(0.0, min(1.0, _base_simq + BESTOF_SIMQ_OFFSET))
        simq_col = "SIM_QUALITY" if "SIM_QUALITY" in pool_bets_bestof.columns else None
        min_sim_quality_val = st.session_state.get("gr_min_sim_quality", None)

        # guard: only apply SIM_QUALITY gate when both column and threshold exist
        apply_simq_gate = (simq_col is not None) and (min_sim_quality_val is not None)
        # (use apply_simq_gate below wherever SIM_QUALITY filtering happens)
        try:
            if not getattr(pool_bets_bestof, 'empty', True):
                # Anchor strength gate
                if 'SIM_ANCHOR_STRENGTH' in pool_bets_bestof.columns and not pool_bets_bestof['SIM_ANCHOR_STRENGTH'].isna().all():
                    pool_bets_bestof['SIM_ANCHOR_STRENGTH'] = pd.to_numeric(
                        pool_bets_bestof['SIM_ANCHOR_STRENGTH'], errors='coerce')
                    pool_bets_bestof = pool_bets_bestof[pool_bets_bestof['SIM_ANCHOR_STRENGTH'].fillna(
                        0.0) >= float(bestof_anchor_min)].copy()
                else:
                    st.session_state.setdefault("_dbg_pool_stages", {})
                    st.session_state["_dbg_pool_stages"]["bestof_missing_sim_anchor_strength"] = True
                    st.warning("SIM_ANCHOR_STRENGTH missing/all-NaN -> BestOf gate FAIL")
                    pool_bets_bestof = pool_bets_bestof.iloc[0:0].copy()
                # Effective_N gate (fallback: SIM_N)
                eff_col = 'EFFECTIVE_N' if 'EFFECTIVE_N' in pool_bets_bestof.columns else (
                    'SIM_N' if 'SIM_N' in pool_bets_bestof.columns else None)
                if eff_col is not None and not pool_bets_bestof[eff_col].isna().all():
                    pool_bets_bestof[eff_col] = pd.to_numeric(
                        pool_bets_bestof[eff_col], errors='coerce')
                    pool_bets_bestof = pool_bets_bestof[pool_bets_bestof[eff_col].fillna(
                        0.0) >= float(min_effective_n_val)].copy()
                else:
                    st.session_state.setdefault("_dbg_pool_stages", {})
                    st.session_state["_dbg_pool_stages"]["bestof_missing_effective_n"] = True
                    st.warning("EFFECTIVE_N missing/all-NaN -> BestOf gate FAIL")
                    pool_bets_bestof = pool_bets_bestof.iloc[0:0].copy()
                # SIM_QUALITY gate (BestOf) + transparent KNN_OK flag
                try:
                    if (
                        'use_knn_gate_bestof' in locals()
                        and bool(use_knn_gate_bestof)
                        and apply_simq_gate
                        and 'SIM_QUALITY' in pool_bets_bestof.columns
                        and not pool_bets_bestof['SIM_QUALITY'].isna().all()
                    ):
                        pool_bets_bestof['SIM_QUALITY'] = pd.to_numeric(
                            pool_bets_bestof['SIM_QUALITY'], errors='coerce')
                        # Build KNN_OK using SIM_QUALITY + Effective_N (if
                        # available)
                        knn_ok = pool_bets_bestof['SIM_QUALITY'].fillna(
                            0.0) >= float(bestof_simq_min)
                        if eff_col is not None and eff_col in pool_bets_bestof.columns:
                            knn_ok = knn_ok & (
    pool_bets_bestof[eff_col].fillna(0.0) >= float(min_effective_n_val))
                        pool_bets_bestof['KNN_OK'] = knn_ok
                        pool_bets_bestof = pool_bets_bestof[pool_bets_bestof['KNN_OK']].copy(
                        )
                    elif apply_simq_gate:
                        st.session_state.setdefault("_dbg_pool_stages", {})
                        st.session_state["_dbg_pool_stages"]["bestof_missing_sim_quality"] = True
                        st.warning("SIM_QUALITY missing/all-NaN -> BestOf gate FAIL")
                        pool_bets_bestof = pool_bets_bestof.iloc[0:0].copy()
                    else:
                        # If gate disabled or SIM_QUALITY missing, keep all
                        # (but still expose KNN_OK if possible)
                        if apply_simq_gate and eff_col is not None and eff_col in pool_bets_bestof.columns and 'SIM_QUALITY' in pool_bets_bestof.columns:
                            pool_bets_bestof['SIM_QUALITY'] = pd.to_numeric(
                                pool_bets_bestof['SIM_QUALITY'], errors='coerce')
                            pool_bets_bestof[eff_col] = pd.to_numeric(
                                pool_bets_bestof[eff_col], errors='coerce')
                            pool_bets_bestof['KNN_OK'] = (
    (pool_bets_bestof['SIM_QUALITY'].fillna(0.0) >= float(min_simq_bestof_val)) & (
        pool_bets_bestof[eff_col].fillna(0.0) >= float(min_effective_n_val)) )
                except Exception:
                    _track_view_error("bestof: knn_gate_bestof")

        except Exception:
            # fail-safe: never crash BestOf due to gating
            pool_bets_bestof = pool_bets.copy() if pool_bets is not None else pd.DataFrame()
            pool_bets_bestof = _ensure_unique_index(pool_bets_bestof)

        # -----------------------------------------------------------
        # Auto-relax for BestOf KNN gate (prevents Pool>0 but BestOf=0)
        # -----------------------------------------------------------
        try:
            _auto_relax_bo = bool(
    st.session_state.get(
        "gr_autorelax_small_pool", True))
            _preset_locked = bool(
    st.session_state.get(
        "gr_preset_locked", True))
            if _auto_relax_bo and _preset_locked and (
    pool_bets is not None) and (
        not getattr(
            pool_bets,
            "empty",
             True)):
                if getattr(
    pool_bets_bestof,
    "empty",
     True) and bool(use_knn_gate_bestof):
                    # Relax SIM_QUALITY / Effective_N thresholds stepwise
                    _simq0 = float(
                        min_sim_quality_val) if min_sim_quality_val is not None else 0.30
                    _en0 = float(
                        min_effective_n_val) if min_effective_n_val is not None else 30.0
                    _steps = [(0.00, 0), (0.03, 5), (0.06, 10), (0.10, 15)]
                    _picked = None
                    for _ds, _dn in _steps:
                        _simq = max(0.15, _simq0 - _ds)
                        _en = max(5.0, _en0 - float(_dn))
                        _cand = pool_bets.copy()
                        if apply_simq_gate and simq_col in _cand.columns:
                            _cand = _cand[pd.to_numeric(
                                _cand[simq_col], errors="coerce").fillna(0.0) >= _simq]
                        if eff_col in _cand.columns:
                            _cand = _cand[pd.to_numeric(
                                _cand[eff_col], errors="coerce").fillna(0.0) >= _en]
                        if not getattr(_cand, "empty", True):
                            _picked = _cand
                            # update visible thresholds so user sees what's
                            # happening
                            st.session_state["gr_min_sim_quality"] = float(
                                _simq)
                            st.session_state["gr_min_effective_n"] = float(_en)
                            st.session_state["__GR_AUTO_RELAX_BESTOF_APPLIED__"] = True
                            break
                    if _picked is not None:
                        pool_bets_bestof = _picked.copy()
                    else:
                        # Last resort: disable KNN gate for BestOf (keep Pool
                        # logic intact)
                        pool_bets_bestof = pool_bets.copy()
                        pool_bets_bestof = _ensure_unique_index(pool_bets_bestof)
                        st.session_state["gr_use_knn_gate_bestof"] = False
                        st.session_state["__GR_AUTO_RELAX_BESTOF_APPLIED__"] = True
        except Exception:
            _track_view_error("bestof: auto_relax_knn_gate")

        # --- Style profile pack (hard gates + soft bonuses) ---
        try:
            _style_mode_now = str(
    st.session_state.get(
        "gr_style_mode",
         "✅ GÜVENLİ"))
            pool_bets_bestof = _apply_style_profile_pack(
    pool_bets_bestof, _style_mode_now, scope="bestof")
        except Exception:
            _track_view_error("bestof: style_profile_pack")

        best_of = _ensure_unique_index(build_best_of_list(
            pool_bets_bestof,
            min_prob=min_prob_val,
            min_score=min_score_val,
            min_ev=min_ev_val,
            max_per_market=6,
            max_total=max_total_val,
            max_per_market_map=quota_map,
            market_quality=market_quality,
            market_min_conf_ev=market_min_conf_ev
        ))
        st.session_state["_dbg_bestof_rows"] = int(len(best_of)) if best_of is not None else 0

        # Apply BestOfRank slider filter (optional)
        if "BestOfRank" in best_of.columns:
            _mask = best_of["BestOfRank"].astype(float) >= float(min_bestofrank_val)
            _mask = _align_mask(_mask, best_of)
            best_of = best_of[_mask].copy()
        if _dbg_col_audit_enabled():
            _maybe_record_col_lineage(
                best_of, "bestof: selected", "bestof_df")
            _store_col_audit(best_of, "bestof_df")

        st.markdown("""
        <div style="display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
            <span style="background-color: #0d47a1; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">🟦 Çok Güçlü (dinamik)</span>
            <span style="background-color: #1b5e20; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">🟩 Güçlü (dinamik)</span>
            <span style="background-color: #f57c00; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">🟧 Zayıf (dinamik)</span>
            <span style="background-color: #b71c1c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">🟥 Çok Zayıf (dinamik)</span>
        </div>
        """, unsafe_allow_html=True)

        def calculate_strength_category(row):
            # Strength preview label (plain text, no emoji) to avoid confusion
            # with other icon columns
            s = row.get(
    'SIM_ANCHOR_STRENGTH', row.get(
        'AnchorStrength', row.get(
            'Score', np.nan)))
            try:
                s = float(s)
            except Exception:
                s = np.nan
            if pd.isna(s):
                return ""
            if s >= 0.65:
                return "Çok Güçlü"
            elif s >= 0.58: return "Güçlü"
            elif s >= 0.50:
                return "Zayıf"
            else: return "Çok Zayıf"

        def strength_icon_from_category(cat: str) -> str:
            # Use colored SQUARES to avoid mixing with CONF/AUTOMOD circle
            # icons
            c = (cat or "").strip().lower()
            if "çok güçlü" in c:
                return "🟦"
            if c == "güçlü" or "güçlü" in c: return "🟩"
            if "çok zayıf" in c:
                return "🟥"
            if "zayıf" in c: return "🟧"
            return ""

        def add_strength_cols(
    df_in: pd.DataFrame,
     strength_col: str = "SIM_ANCHOR_STRENGTH") -> pd.DataFrame:
            """Vectorized strength category + icon.
            Uses dynamic percentile cutoffs when possible so icons are informative even if the strength scale shifts.
            Falls back to fixed cutoffs if sample is too small.
            """
            if not isinstance(df_in, pd.DataFrame) or df_in.empty:
                return df_in
            s = pd.to_numeric(df_in.get(strength_col, np.nan), errors="coerce")
            df_in[strength_col] = s  # enforce numeric
            s_valid = s.dropna()
            # Dynamic cutoffs (percentiles) -> more stable UX across different
            # pools/days
            if len(s_valid) >= 8 and s_valid.nunique() >= 4:
                hi = float(s_valid.quantile(0.85))
                mid = float(s_valid.quantile(0.60))
                lo = float(s_valid.quantile(0.35))
                # enforce monotonic ordering (avoid edge cases where quantiles
                # collapse)
                eps = 1e-6
                mid = max(mid, lo + eps)
                hi = max(hi, mid + eps)
            else:
                # fallback to legacy fixed thresholds
                hi, mid, lo = 0.55, 0.45, 0.35

            cat = np.where(s >= hi, "Çok Güçlü",
                           np.where(s >= mid, "Güçlü",
                           np.where(s >= lo, "Zayıf", "Çok Zayıf")))

            df_in["AnchorStrengthCategory"] = cat
            df_in["StrengthCategory"] = df_in.get(
    "StrengthCategory", df_in["AnchorStrengthCategory"])
            df_in["STRENGTH_ICON"] = pd.Series(
    cat, index=df_in.index).apply(strength_icon_from_category)
            # keep cutoffs for debugging/export (optional)
            df_in["_STR_HI"] = hi
            df_in["_STR_MID"] = mid
            df_in["_STR_LO"] = lo
            return df_in

        def color_score(val):
            if val >= 0.58:
                color = '#0d47a1'
            elif val >= 0.50: color = '#1b5e20'
            else:
                color = '#fbc02d'
            return f'background-color: {color}'

        def get_stars(row):
            base_q = market_quality.get(row['Seçim'], 0.5)
            mq = row.get('Market_Conf_Score', base_q)
            lq = row.get('League_Conf', 0.5)
            s  = row.get('Score', 0.5)

            q = 0.4 * mq + 0.3 * lq + 0.3 * s
            q = max(0.0, min(1.0, q))

            if q >= 0.80:
                return "★★★★★"
            elif q >= 0.65: return "★★★★☆"
            elif q >= 0.50:
                return "★★★☆☆"
            elif q >= 0.35: return "★★☆☆☆"
            else:
                return "★☆☆☆☆"

        if not best_of.empty:
            st.subheader("🌟 Best Of – Golden Ratio Listesi")

            # Add/refresh BestOfRank + Stars for display
            best_of = add_star_rating_by_bestofrank(best_of)

            # NEW: Anchor-based strength columns for BestOf (align with Similarity Anchor system)
            # NEW: Anchor-based strength columns for BestOf (align with
            # Similarity Anchor system)
            try:
                if "SIM_ANCHOR_STRENGTH" in best_of.columns:
                    best_of["AnchorStrength"] = pd.to_numeric(
                        best_of["SIM_ANCHOR_STRENGTH"], errors="coerce")
                    _strength_src = "SIM_ANCHOR_STRENGTH"
                elif "AnchorStrength" in best_of.columns:
                    best_of["AnchorStrength"] = pd.to_numeric(
                        best_of["AnchorStrength"], errors="coerce")
                    _strength_src = "AnchorStrength"
                else:
                    best_of["AnchorStrength"] = pd.to_numeric(
                        best_of.get("Score", 0.0), errors="coerce")
                    _strength_src = "AnchorStrength"
                # add dynamic icon/category cols
                best_of = add_strength_cols(
    best_of, strength_col=_strength_src)
            except Exception:
                _track_view_error("bestof: add_strength_cols")

            st.write("Best Of Pazar Dağılımı:")
            if "AutoMod_Status" in best_of.columns:
                _dist = best_of.pivot_table(
    index="Seçim",
    columns="AutoMod_Status",
    values="EV",
    aggfunc="size",
     fill_value=0)
                _dist["Toplam"] = _dist.sum(axis=1)
                _dist = _dist.sort_values("Toplam", ascending=False)
                st.dataframe(_dist)
            else:
                st.dataframe(best_of["Seçim"].value_counts().to_frame().T)
            # Use global star/advanced toggles (from Golden Ratio filter panel)
            min_star_bo = int(st.session_state.get("gr_min_star_global", 1))
            show_adv_bo = bool(
    st.session_state.get(
        "gr_show_adv_global", False))
            # -----------------------------------------------------------

            # UI EKLENTİSİ: AKILLI RİSK ETİKETLERİ (DÜZELTİLMİŞ)

            # -----------------------------------------------------------


            def get_match_character_badge(row):
                """
                Rozet üretimi (rehberle senkron):
                🔒BNK / 🧠ZEK / 💎VAL / ✅GÜV / 💣SPR
                Not: Rozet yoksa "" döner. UI'da hücre boş kalır.
                """

                def _to_float(v, default=0.0):
                    try:
                        if v is None:
                            return float(default)
                        s = str(v).strip()
                        if s == "" or s.lower() in ("nan", "none", "null"):
                            return float(default)
                        # "58%" gibi
                        if s.endswith("%"):
                            return float(s[:-1].replace(",", ".")) / 100.0
                        return float(s.replace(",", "."))
                    except Exception:
                        return float(default)

                def _clean_icon(v):
                    s = str(v).strip()
                    return "" if s.lower() in ("nan", "none", "null") else s

                # Temel veriler
                prob = _to_float(row.get("Prob", 0), 0.0)
                ev = _to_float(row.get("EV", 0), 0.0)
                odd = _to_float(row.get("Odd", 0), 0.0)

                # Benzerlik bağlamı (yoksa 0 gelir)
                sim_p_over = _to_float(row.get("SIM_POver", 0), 0.0)
                sim_p_btts = _to_float(row.get("SIM_PBTTS", 0), 0.0)
                sim_q = _to_float(row.get("SIM_QUALITY", 0), 0.0)
                eff_n = _to_float(row.get("EFFECTIVE_N", row.get("SIM_N", 0)), 0.0)

                # Seçim normalize (TR)
                selection_raw = str(row.get("Seçim", "") or "")
                s = selection_raw.strip().lower()
                s_norm = (
                    s.replace("ü", "u")
                     .replace("ı", "i")
                     .replace("ş", "s")
                     .replace("ğ", "g")
                     .replace("ç", "c")
                     .replace("ö", "o")
                )

                # ikonlar / risk
                conf_icon = _clean_icon(row.get("CONF_ICON", ""))
                am_icon = _clean_icon(row.get("AUTOMOD_ICON", ""))
                trap_icon = _clean_icon(row.get("TRAP_ICON", ""))
                dang_icon = _clean_icon(row.get("DANGER_ICON", ""))

                has_trap_or_danger = (trap_icon != "") or (dang_icon != "")

                # -------------------------------------------------------
                # Contrarian (tarihsel çelişki)
                # -------------------------------------------------------
                history_conflict = False
                if ("alt" in s_norm) and (sim_p_over > 0.60):
                    history_conflict = True
                if (("ust" in s_norm) or ("üst" in s)) and (
                    0 < sim_p_over < 0.40):
                    history_conflict = True
                if ("kg var" in s_norm) and (0 < sim_p_btts < 0.40):
                    history_conflict = True
                if ("kg yok" in s_norm) and (sim_p_btts > 0.60):
                    history_conflict = True

                # -------------------------------------------------------
                # Etiket kuralları (sıra önemli)
                # Kısa etiket kullanıyoruz (UI kırpılmasın diye)
                # -------------------------------------------------------
                banko_sim_ok = (
    sim_q >= 0.35) or (
        (conf_icon == "🟢") and (
            am_icon == "🟢")) or (
                eff_n >= 35)

                # 1) 🔒BNK: yüksek olasılık + güçlü kanıt + risk yok
                if (prob >= 0.67) and banko_sim_ok and (
                    not has_trap_or_danger):
                    return "🔒BNK"

                # 2) 💎VAL: EV anlamlı + oran makul + risk yok
                if (ev >= 1.0) and (odd >= 1.90) and (not has_trap_or_danger):
                    return "💎VAL"

                # 3) 🧠ZEK: çelişki var ama sistem hâlâ makul avantaj görüyor
                if history_conflict and (
    (ev >= 0.5) or (
        prob >= 0.55)) and (
            odd >= 1.70) and (
                not has_trap_or_danger):
                    return "🧠ZEK"

                # 4) 💣SPR: yüksek oran (eğlence) — EV şartı yok (küçük stake
                # mantığı)
                if (odd >= 2.80) and (prob >= 0.35):
                    return "💣SPR"

                # 5) ✅GÜV: orta-üst olasılık + risk yok (EV negatif olabilir;
                # burada 'stabil' demek)
                if (prob >= 0.60) and (not has_trap_or_danger):
                    return "✅GÜV"

                return ""
# Fonksiyonu ANA TABLOYA (best_of) uygula

            if 'best_of' in locals() and best_of is not None and not best_of.empty:
                best_of["Karakter"] = best_of.apply(
                    get_match_character_badge, axis=1)

            # Aynı rozet mantığını HAVUZ (pool_bets) için de uygula (UI/Export
            # senkronu)
            if 'pool_bets' in locals() and pool_bets is not None and (
                not getattr(pool_bets, 'empty', True)):
                try:
                    pool_bets["Karakter"] = pool_bets.apply(
                        get_match_character_badge, axis=1)
                except Exception:
                    _track_view_error("pool_bets: karakter")


# Golden Ratio / global filtre sonrası görünüm

            best_view_src = best_of.copy()
            _alias_map = {
                "Date": ["DATE", "date", "MatchDate", "match_date"],
                "League": ["LEAGUE", "league", "Lig", "LIG"],
                "HomeTeam": ["Home_Team", "HOME", "Home", "HomeTeamName", "home_team"],
                "AwayTeam": ["Away_Team", "AWAY", "Away", "AwayTeamName", "away_team"],
                "Seçim": ["Secim", "Selection", "SELECTION", "Pick", "Market", "Choice"],
                "Odd": ["Odds", "ODD", "odd", "Price"],
                "Prob": ["Probability", "PROB", "prob", "Model_Prob", "ModelProb"],
                "EV": ["Ev", "EV_Calc", "EV_calc", "ExpectedValue", "expected_value"],
            }
            for _dst, _aliases in _alias_map.items():
                if _dst not in best_view_src.columns:
                    for _a in _aliases:
                        if _a in best_view_src.columns:
                            best_view_src[_dst] = best_view_src[_a]
                            break
            _bestof_view_cols = list(dict.fromkeys(REQUIRED_VIEW_COLS + [
                "Star_Rating", "BestOfRank", "StrengthCategory",
                "AnchorStrengthCategory", "AutoMod_Status", "AMQS_percentile",
                "Trap_Flag", "Danger_Flag", "Archetype", "AutoMod",
                "CONF_percentile", "CONF_Status", "P_1X", "P_2X", "P_O25",
                "P_O35", "Score", "GoldenScore", "League_Conf",
                "Market_Conf_Score", "Final_Confidence", "AMQS", "EV_pure_pct",
                "Implied_Prob", "Model_vs_Market", "Trap_Type",
                "SIM_ANCHOR_GROUP", "SIM_ALPHA", "SIM_QUALITY", "SIM_POver",
                "SIM_PBTTS", "SIM_MS_STRENGTH", "SIM_OU_STRENGTH",
                "SIM_BTTS_STRENGTH", "P_Over_Model", "P_Over_Final",
                "P_BTTS_Model", "P_BTTS_Final", "P_Home_Model",
                "P_Home_Final", "P_Draw_Model", "P_Draw_Final",
                "P_Away_Model", "P_Away_Final", "BLEND_MODE_MS",
                "LEAGUE_OK_MS", "BLEND_W_LEAGUE_MS", "MS_LEAGUE_SIM_P2",
                "MS_GLOBAL_SIM_P2", "BLEND_MODE_OB", "LEAGUE_OK_OB",
                "BLEND_W_LEAGUE_OB", "OB_LEAGUE_SIM_POver",
                "OB_GLOBAL_SIM_POver", "MDL_rank", "MDL_quantile",
                "MDL_BOOST", "GoldenScore_MDL",
            ]))
            best_view = _ensure_view_cols(best_view_src, _bestof_view_cols)

            st.session_state["_dbg_best_view_rows_pre_style"] = int(len(best_view))
            st.session_state["_dbg_best_view_cols_pre_style"] = list(best_view.columns)
            if _dbg_col_audit_enabled():
                _maybe_record_col_lineage(
                    best_view, "bestof: view_pre_style", "bestof_view_pre_style")
                _store_col_audit(best_view, "bestof_view_pre_style")

            # Stil rozetini (tek slider modu) BestOf görünümüne ekle (istersen
            # karakteri bununla kilitle)
            _m = str(st.session_state.get("gr_style_mode", "") or "")
            if ("BANKO" in _m) or ("BNK" in _m):
                _style_badge = "🔒BNK"
            elif ("ZEK" in _m) or ("ZEKİ" in _m) or ("ZEKI" in _m):
                _style_badge = "🧠ZEK"
            elif ("VALUE" in _m) or ("VAL" in _m):
                _style_badge = "💎VAL"
            elif ("SÜRPRİZ" in _m) or ("SURPRIZ" in _m) or ("SPR" in _m):
                _style_badge = "💣SPR"
            elif ("GÜVEN" in _m) or ("GUVEN" in _m) or ("GÜV" in _m):
                _style_badge = "✅GÜV"
            else:
                _style_badge = ""
            if _style_badge:
                best_view["Stil_Rozet"] = _style_badge
                if bool(st.session_state.get("gr_force_style_badge", True)):
                    best_view["Karakter"] = _style_badge

            best_view = _ensure_trap_cols(best_view)

            if "StrengthCategory" in best_view.columns and "STRENGTH_ICON" not in best_view.columns:
                try:
                    best_view["STRENGTH_ICON"] = best_view["StrengthCategory"].apply(
                        strength_icon_from_category)
                except Exception:
                    best_view["STRENGTH_ICON"] = ""
            # Alias for clarity in new format
            if 'StrengthCategory' in best_view.columns and 'AnchorStrengthCategory' not in best_view.columns:
                best_view['AnchorStrengthCategory'] = best_view['StrengthCategory']

# AutoMod icon for main view (UX)
            if "AutoMod_Status" in best_view.columns:
                best_view["AUTOMOD_ICON"] = best_view["AutoMod_Status"].map({"High": "🟢","Medium":"🟡","Low":"🔴"}).fillna("🟡")
            elif "AMQS_percentile" in best_view.columns:
                _p = pd.to_numeric(
    best_view["AMQS_percentile"],
     errors="coerce").fillna(0.50)
                best_view["AUTOMOD_ICON"] = np.select([_p >= 0.80, _p >= 0.60], ["🟢", "🟡"], default="🔴")
            else:
                best_view["AUTOMOD_ICON"] = "🟡"
            # IMPORTANT: Do NOT recompute/overwrite CONF on the displayed subset.
            # CONF_ICON is used as a filter; recomputing here makes it look
            # like the filter doesn't work.

            # Trap visibility (show without advanced columns)
            if "Trap_Flag" in best_view.columns:
                best_view["TRAP_ICON"] = np.where(
    pd.to_numeric(
        best_view["Trap_Flag"],
        errors="coerce").fillna(0).astype(bool),
        "⚠️",
         "")
            elif "Archetype" in best_view.columns:
                best_view["TRAP_ICON"] = np.where(
    best_view["Archetype"].astype(str).eq("TRAP_FAVORITE"), "⚠️", "")
            else:
                best_view["TRAP_ICON"] = ""
            if "DANGER_ICON" in best_view.columns:
                best_view["DANGER_ICON"] = best_view["DANGER_ICON"].fillna("")
            else:
                best_view["DANGER_ICON"] = ""
            # Icon inclusion flags (Golden Ratio panel)
            show_trap_icon = bool(st.session_state.get("bo_inc_trap", True))
            show_danger_icon = bool(
    st.session_state.get(
        "bo_inc_danger", True))
            show_arch_icon = bool(st.session_state.get("bo_inc_profile", True))

                       # ---- Phase-2 UI: lean default columns & clear naming ----
            if "AnchorStrengthCategory" not in best_view.columns and "StrengthCategory" in best_view.columns:
                best_view["AnchorStrengthCategory"] = best_view["StrengthCategory"]
            # Ensure core similarity columns exist (never show as blank if we
            # can derive)
            if "SIM_ANCHOR_STRENGTH" not in best_view.columns:
                _src_strength = best_view.get(
    "AnchorStrength", best_view.get(
        "Score", np.nan))
                best_view["SIM_ANCHOR_STRENGTH"] = pd.to_numeric(
                    _src_strength, errors="coerce")
            if "EFFECTIVE_N" not in best_view.columns:
                _src_en = best_view.get(
    "EFFECTIVE_N_REAL", best_view.get(
        "SIM_N", 0))
                best_view["EFFECTIVE_N"] = pd.to_numeric(
                    _src_en, errors="coerce")

            base_cols = [
                "Match_ID",
                "Date",
                "League",
                "MDL_rank",
                "MDL_quantile",
                "CONF_ICON",
                "AUTOMOD_ICON",
                "TRAP_ICON",
                "DANGER_ICON",
                "ARCH_ICON",
                "Karakter",
                "HomeTeam",
                "AwayTeam",
                "Seçim",
                "Odd",
                "Prob",
                "EV",
                "PROFILE_CONF",
                "DRAW_PRESSURE",
                "TG_Q75",
                "TG_Q90",
                "MDL_BOOST",
                "GoldenScore_MDL",
                "STRENGTH_ICON",
                "AnchorStrengthCategory",
                "SIM_ANCHOR_STRENGTH",
                "EFFECTIVE_N",
                "Star_Rating",
                "BestOfRank",
            ]

            cols_advanced_exact = [
                "Match_ID",
                "Date",
                "League",
                "CONF_ICON",
                "AUTOMOD_ICON",
                "TRAP_ICON",
                "DANGER_ICON",
                "ARCH_ICON",
                "Karakter",
                "HomeTeam",
                "AwayTeam",
                "Seçim",
                "Odd",
                "Prob",
                "EV",
                "PROFILE_CONF",
                "DRAW_PRESSURE",
                "TG_Q75",
                "TG_Q90",
                "MDL_BOOST",
                "GoldenScore_MDL",
                "P_1X",
                "P_2X",
                "P_O25",
                "P_O35",
                "STRENGTH_ICON",
                "AnchorStrengthCategory",
                "SIM_ANCHOR_STRENGTH",
                "EFFECTIVE_N",
                "Star_Rating",
                "BestOfRank",
                "Score",
                "GoldenScore",
                "League_Conf",
                "Market_Conf_Score",
                "Final_Confidence",
                "CONF_percentile",
                "CONF_Status",
                "AMQS",
                "AMQS_percentile",
                "AutoMod",
                "AutoMod_Status",
                "Archetype",
                "EV_pure_pct",
                "Implied_Prob",
                "Model_vs_Market",
                "Trap_Flag",
                "Danger_Flag",
                "Trap_Type",
                "SIM_ANCHOR_GROUP",
                "SIM_ALPHA",
                "SIM_QUALITY",
                "SIM_POver",
                "SIM_PBTTS",
                "SIM_MS_STRENGTH",
                "SIM_OU_STRENGTH",
                "SIM_BTTS_STRENGTH",
                "P_Over_Model",
                "P_Over_Final",
                "P_BTTS_Model",
                "P_BTTS_Final",
                "P_Home_Model",
                "P_Home_Final",
                "P_Draw_Model",
                "P_Draw_Final",
                "P_Away_Model",
                "P_Away_Final",
                "BLEND_MODE_MS",
                "LEAGUE_OK_MS",
                "BLEND_W_LEAGUE_MS",
                "MS_LEAGUE_SIM_P2",
                "MS_GLOBAL_SIM_P2",
                "BLEND_MODE_OB",
                "LEAGUE_OK_OB",
                "BLEND_W_LEAGUE_OB",
                "OB_LEAGUE_SIM_POver",
                "OB_GLOBAL_SIM_POver",
            ]

            # Optional: hide icon columns if user disables them (keeps
            # pool-view parity)
            if not show_trap_icon and "TRAP_ICON" in base_cols:
                base_cols.remove("TRAP_ICON")
            if not show_danger_icon and "DANGER_ICON" in base_cols:
                base_cols.remove("DANGER_ICON")
            if not show_arch_icon and "ARCH_ICON" in base_cols:
                base_cols.remove("ARCH_ICON")

            # Build display columns (BestOf should mirror Pool/Wide view; only
            # extra is Karakter)
            cols = cols_advanced_exact if show_adv_bo else base_cols
            # Apply the same icon-hide behavior to the advanced list
            if show_adv_bo:
                if not show_trap_icon and "TRAP_ICON" in cols:
                    cols = [c for c in cols if c != "TRAP_ICON"]
                if not show_danger_icon and "DANGER_ICON" in cols:
                    cols = [c for c in cols if c != "DANGER_ICON"]
                if not show_arch_icon and "ARCH_ICON" in cols:
                    cols = [c for c in cols if c != "ARCH_ICON"]

            # Keep only existing columns and dedupe (preserve order)
            cols = _safe_view_cols(best_view, cols, tag="bestof")
            cols = list(dict.fromkeys(cols))
            _df_show = best_view[cols] if cols else best_view
            _safe_view_cols(_df_show, REQUIRED_VIEW_COLS, tag="bestof")
            st.session_state["_dbg_best_view_rows_post_style"] = int(len(_df_show))
            st.session_state["_dbg_bestof_source_var"] = "_df_show"
            _sort_cols = [c for c in ["Star_Rating", "BestOfRank"] if c in _df_show.columns]
            if _sort_cols:
                _df_show = _df_show.sort_values(
                    _sort_cols, ascending=[False] * len(_sort_cols))
            _df_show = _df_show.reset_index(drop=True)
            _df_show = _df_show.loc[:, ~_df_show.columns.duplicated()].copy()

            _styler = _df_show.style.format({
                'Prob': '{:.2%}',
                'EV': '{:.2f}',
                'Score': '{:.3f}',
                'GoldenScore': '{:.3f}',
                'BestOfRank': '{:.3f}',
                'AMQS': '{:.2f}',
                'AMQS_percentile': '{:.0%}',
                'Final_Confidence': '{:.3f}',
                'CONF_percentile': '{:.0f}',
            })

            if "AutoMod_Status" in _df_show.columns:
                _styler = _styler.applymap(
    _automod_cell_style, subset=["AutoMod_Status"])
            # Color-code AutoMod status for instant readability
            if "AutoMod_Status" in _df_show.columns:
                _styler = _styler.applymap(
    _automod_css, subset=["AutoMod_Status"])
            if "AutoMod" in _df_show.columns:
                _styler = _styler.set_properties(
                    subset=["AutoMod"], **{"font-weight": "700"})

            st.dataframe(_styler)

            # --- Export (clean CSV) ---
            try:
                # Use the underlying best_of dataframe (not the display view) to ensure Match_ID
                _export_best = ensure_match_id(best_of.copy())
                # ensure MDL rank/quantile are present if GoldenScore_MDL exists
                if "MDL_rank" not in _export_best.columns or "MDL_quantile" not in _export_best.columns:
                    _tmp2 = ensure_bestofrank(_export_best.copy())
                    for c in ["MDL_rank", "MDL_quantile"]:
                        if c in _tmp2.columns:
                            _export_best[c] = _tmp2[c]
                if "Match_ID" not in _export_best.columns:
                    raise RuntimeError("Match_ID missing in export after ensure_match_id")
                if "AUTOMOD_ICON" not in _export_best.columns:
                    _auto_src = _export_best.get("AutoMod_Status", "")
                    _export_best["AUTOMOD_ICON"] = _auto_src.map(
                        lambda x: "??" if str(x).lower().startswith("g") else "??"
                    )
                export_cols = _safe_view_cols(
                    _export_best, REQUIRED_VIEW_COLS + ["Match_ID"], tag="bestof_export")
                if export_cols:
                    _export_best = _export_best[export_cols]
                if _dbg_col_audit_enabled():
                    _maybe_record_col_lineage(
                        _export_best, "bestof: export_df", "export_df")
                    _store_col_audit(_export_best, "bestof_export_df")
                csv_bytes = _export_best.to_csv(
    index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
    "⬇️ BestOf CSV (temiz)",
    data=csv_bytes,
    file_name="bestof_export.csv",
     mime="text/csv")
            except Exception:
                _track_view_error("bestof: export_csv")

            # -----------------------------------------------------------------
            # 🧲 Benzer Maçlar (kNN) - LAZY / Explainable
            # -----------------------------------------------------------------
            with st.expander("🧲 Benzer Maçlar (kNN) – Neden benzer / Ne olmuş?", expanded=False):
                train_core = st.session_state.get("train_core", None)
                if train_core is None or getattr(train_core, "empty", True):
                    st.info(
                        "Past havuzu (train_core) yok. 'Analizi Başlat' sonrası oluşuyor.")
                else:
                    # Make sure Match_ID exists for selection stability
                    _tmp = best_view.copy()
                    if "Match_ID" not in _tmp.columns:
                        _tmp["Match_ID"] = _tmp.apply(
                            lambda r: _make_match_id(r), axis=1)

                    # Pick a match from Best Of view
                    label_cols = [c for c in ["League", "HomeTeam","AwayTeam","Date","Seçim","Star_Rating","Prob","EV"] if c in _tmp.columns]
                    if label_cols:
                        # build label for UI, but select by stable id (root
                        # fix)
                        _tmp = _tmp.copy()
                        # Prefer a stable id column; Match_ID is created above
                        # in this block
                        id_col = 'Match_ID' if 'Match_ID' in _tmp.columns else None
                        if id_col is None:
                            _tmp['_ROW_ID'] = range(len(_tmp))
                            id_col = '_ROW_ID'

                        # Build display label (can be non-unique; we append id
                        # to avoid UI confusion)

                        _df_lbl = _tmp[label_cols].astype(str)

                        # Robust label build:
                        # - Some pandas edge-cases (e.g., empty/degenerate frames) can return a DataFrame from apply().
                        # - We always coerce to a 1D Series before assignment.
                        _lbl = _df_lbl.apply( lambda r: " | ".join(
                            [v for v in r.tolist() if v not in ("nan", "None", "")]), axis=1, )
                        if isinstance(_lbl, pd.DataFrame):
                            # Defensive: keep first col if apply unexpectedly
                            # produced a DataFrame
                            _lbl = _lbl.iloc[:, 0]

                        _tmp["_label"] = _lbl.astype(
                            str) + "  #" + _tmp[id_col].astype(str)

                        opt_map = dict(zip(_tmp[id_col].astype(
                            str).tolist(), _tmp['_label'].tolist()))
                        choice_id = st.selectbox(
    "Maç seç (Best Of)", options=list(
        opt_map.keys()), format_func=lambda k: opt_map.get(
            k, str(k)), key='bestof_match_choice')

                        sel_df = _tmp.loc[_tmp[id_col].astype(
                            str) == str(choice_id)]
                        if sel_df.empty:
                            st.warning(
                                'Seçim filtre sonrası bulunamadı. Filtreleri değiştirdiysen yeniden seçim yap.')
                            sel = None
                        else:
                            sel = sel_df.iloc[0]
                    else:
                        st.warning(
                            "Best Of tablosunda beklenen temel kolonlar yok (League/Home/Away/Date).")
                        sel = None

                    if sel is not None:
                        # Options
                        c1, c2, c3 = st.columns(3)
                        same_league = c1.checkbox(
    "Aynı lig öncelikli",
    value=bool(
        st.session_state.get(
            "SIM_SAME_LEAGUE",
            True)),
             key="SIM_SAME_LEAGUE")
                        k = c2.slider(
    "Komşu sayısı (K)", 5, 120, int(
        st.session_state.get(
            "SIM_K", 20)), 5, key="SIM_K")
                        max_pool = c3.slider(
    "Havuz limiti (hız)", 10000, 120000, int(
        st.session_state.get(
            "SIM_MAX_POOL", 60000)), 5000, key="SIM_MAX_POOL")

                        # Tightening controls
                        t1, t2, t3 = st.columns(3)
                        use_cutoff = t1.checkbox("Sadece güçlü benzerleri kullan", value=bool(
                            st.session_state.get("SIM_USE_CUTOFF", True)), key="SIM_USE_CUTOFF")
                        min_sim = t2.slider(
    "Min Similarity", 0.30, 0.80, float(
        st.session_state.get(
            "SIM_MIN_SIM", 0.55)), 0.01, key="SIM_MIN_SIM")
                        min_neighbors = t3.slider(
    "Min güçlü komşu", 5, 40, int(
        st.session_state.get(
            "SIM_MIN_NEIGHBORS", 10)), 1, key="SIM_MIN_NEIGHBORS")

                        y1, y2 = st.columns([1, 2])
                        years_back = y1.slider(
    "Son kaç yıl (0=kapalı)", 0, 8, int(
        st.session_state.get(
            "SIM_YEARS_BACK", 4)), 1, key="SIM_YEARS_BACK")
                        preset = y2.selectbox("Feature preset",
    ["Dual (Lig+Global, MS/OB)",
    "Auto (mevcut kolonlar)",
    "Over/BTTS odaklı (xG+Book)",
    "MS odaklı (Elo+Form+xGD)"],
    index=0,
     key="SIM_PRESET")
                        maybe_refresh_similarity_on_session_predictions()

                        feats = _pick_similarity_features(
                            best_view, train_core, preset=preset)
                        if preset == "Over/BTTS odaklı (xG+Book)":
                            pref = [c for c in ["Roll_xG_5_Home", "Roll_xG_5_Away","xGD_last5_Diff","Book_Exp_Total_Goals"] if c in feats]
                            feats = pref if len(pref) >= 2 else feats
                        elif preset == "MS odaklı (Elo+Form+xGD)":
                            pref = [c for c in ["Elo_Diff", "Form_TMB_Diff","xGD_last5_Diff"] if c in feats]
                            feats = pref if len(pref) >= 2 else feats

                        st.caption(
    f"Kullanılan similarity feature'ları: {
        ', '.join(feats) if feats else 'YOK'}")

                        cache_key = sel.get("Match_ID", _make_match_id(sel))
                        if "sim_cache" not in st.session_state:
                            st.session_state["sim_cache"] = {}
                        if st.button(
    "Benzerleri Getir",
     key=f"btn_sim_{cache_key}"):

                            # --- Sprint 5: Time-decay toggle (regime)
                            use_time_decay = st.checkbox(
    "Time-decay ağırlık (eski maçları azalt)",
    value=False,
     key=f"sim_time_decay_{cache_key}")
                            half_life_days = st.slider(
    "Half-life (gün)",
    min_value=180,
    max_value=1200,
    value=540,
    step=30,
     key=f"sim_half_life_{cache_key}")
                            # --- Sprint 6: Gate thresholds (reliability brake)
                            gate_min_simq = st.slider(
    "Min SIM_QUALITY",
    min_value=0.10,
    max_value=0.80,
    value=0.30,
    step=0.05,
     key=f"sim_gate_simq_{cache_key}")
                            gate_min_en = st.slider(
    "Min EFFECTIVE_N",
    min_value=1,
    max_value=30,
    value=6,
    step=1,
     key=f"sim_gate_en_{cache_key}")
                            neigh_df, sim_out = _similarity_neighbors_for_match(
                                train_core=train_core,
                                upcoming_row=sel,
                                feats=feats,
                                k=int(k),
                                same_league=bool(same_league),
                                max_pool=int(max_pool),
                                use_cutoff=bool(use_cutoff),
                                min_similarity=float(min_sim),
                                min_neighbors=int(min_neighbors),
                                year_window=int(years_back),
                                use_time_decay=bool(use_time_decay),
                                half_life_days=int(half_life_days),
                                gate_min_simq=float(gate_min_simq),
                                gate_min_en=int(gate_min_en),
                            )
                            st.session_state["sim_cache"][cache_key] = (
                                neigh_df, sim_out)

                        if cache_key in st.session_state.get("sim_cache", {}):
                            neigh_df, sim_out = st.session_state["sim_cache"][cache_key]
                            if sim_out.get("error"):
                                st.error(sim_out["error"])
                            else:
                                # Summary
                                s1, s2, s3, s4 = st.columns(4)
                                s1.metric(
    "Similarity Quality", f"{
        sim_out.get(
            'SIM_QUALITY', 0):.2f}")
                                s2.metric(
    "Güçlü Komşu", int(
        sim_out.get(
            "EFFECTIVE_N", sim_out.get(
                "SIM_N", 0))))
                                s3.metric(
    "Median Dist", f"{
        sim_out.get(
            'SIM_DIST_MED', 0):.3f}")
                                s4.metric("MS1 / MSX / MS2", f"{sim_out.get('SIM_P1', 0):.0%} | {sim_out.get('SIM_PX',0):.0%} | {sim_out.get('SIM_P2',0):.0%}")
                                st.caption(f"Havuz: raw={sim_out.get('POOL_RAW_N', 0)} → used={sim_out.get('POOL_USED_N',0)} | cutoff={sim_out.get('CUTOFF_APPLIED', False)} (min_sim={sim_out.get('MIN_SIM', float('nan'))})")

                                s5, s6 = st.columns(2)
                                if "SIM_POver" in sim_out:
                                    s5.metric("Over 2.5", f"{sim_out.get('SIM_POver', 0):.0%}")
                                if "SIM_PBTTS" in sim_out:
                                    s6.metric("BTTS", f"{sim_out.get('SIM_PBTTS', 0):.0%}")

                                # Why similar (closest neighbor)
                                why_df = sim_out.get("why_df", None)
                                if isinstance(
    why_df, pd.DataFrame) and not why_df.empty:
                                    st.markdown(
                                        "**Neden benzer? (en yakın komşuya göre)**")
                                    st.dataframe(why_df)

                                # Neigh list
                                if isinstance(
    neigh_df, pd.DataFrame) and not neigh_df.empty:
                                    st.markdown("**En yakın 50 benzer maç**")
                                    st.dataframe(neigh_df)
                                else:
                                    st.warning("Komşu listesi boş döndü.")

                        # -----------------------------------------------------------
            # JOURNAL (NEW): Save today's BestOf as Pick Log entries (profile-aware)
            # -----------------------------------------------------------
            c_save1, c_save2, c_save3 = st.columns([1.2, 1.2, 1.0])
            with c_save1:
                _coupon_name = st.text_input(
    "Kupon adı (opsiyonel)", value="", key="pl_coupon_name_input")
            with c_save2:
                _coupon_tag = st.text_input(
    "Not/etiket (opsiyonel)", value="", key="pl_coupon_tag_input")
            with c_save3:
                _snap_day = st.date_input(
    "Snapshot günü",
    value=pd.Timestamp.utcnow().date(),
     key="pl_snapshot_day")

            def _style_to_profile_name(s: str) -> str:
                s = str(s or "")
                if "BANKO" in s:
                    return "Banko"
                if "ZEK" in s: return "Zeki"
                if "VALUE" in s:
                    return "Value"
                if "GÜVEN" in s or "GUVEN" in s: return "Güvenli"
                if "SÜRPR" in s or "SURPR" in s:
                    return "Sürpriz"
                return s.strip() or "Profil"

            _profile_name = _style_to_profile_name(style_mode)

            if st.button(
    "✅ Bugünün Pick'lerini Kaydet (Pick Log)",
    use_container_width=True,
     key="pl_save_bestof_btn"):
                if best_view is None or best_view.empty:
                    st.warning("Best Of listesi boş. Önce Best Of üret.")
                else:
                    _filters = {
                        "style_mode": str(style_mode),
                        "min_prob": float(st.session_state.get("gr_min_prob", min_prob_val)),
                        "min_score": float(st.session_state.get("gr_min_score", min_score_val)),
                        "min_ev": float(st.session_state.get("gr_min_ev", min_ev_val)),
                        "max_total": int(st.session_state.get("gr_max_total", max_total_val)),
                        "min_league_conf": float(st.session_state.get("gr_min_league_conf", min_league_conf)),
                        "min_market_conf": float(st.session_state.get("gr_min_market_conf", min_market_conf)),
                        "hide_weak_leagues": bool(st.session_state.get("gr_hide_weak", hide_weak_leagues)),
                        "min_star_bo": int(min_star_bo),
                    }

                    _save_cols = [c for c in [
                        "Date",
                        "League",
                        "HomeTeam",
                        "AwayTeam",
                        "Seçim",
                        "Odd",
                        "Prob",
                        "EV",
                        "Karakter",
                        "Score",
                        "GoldenScore",
                        "League_Conf",
                        "Market_Conf_Score",
                        "AMQS",
                        "AMQS_percentile",
                        "AutoMod_Status",
                        "Archetype",
                        "Star_Rating",
                        "BestOfRank",
                        "CONF_ICON",
                        "AUTOMOD_ICON",
                        "ARCH_ICON",
                        "CONF_percentile",
                        "AMQS_percentile",
                        "AnchorStrength",
                        "SIM_ANCHOR_STRENGTH",
                        "AnchorStrengthCategory",
                        "SIM_QUALITY",
                        "SimQuality",
                        "EFFECTIVE_N",
                        "Effective_N",
                        "SIM_P1",
                        "SIM_P2",
                        "SIM_PX",
                        "Odd_Open",
                        "Odd_Close",
                        "Odd_Closing",
                        "ClosingOdd",
                        "Market",
                        "MarketType",
                        "Selection"
                    ] if c in best_view.columns]

                    _save = best_view[_save_cols].copy()

                    # Ensure stable Match_ID
                    if "Match_ID" not in _save.columns:
                        try:
                            _save["Match_ID"] = _save.apply(
                                _make_match_id_row, axis=1)
                        except Exception:
                            _save["Match_ID"] = ""

                    _save["Snapshot_Date"] = pd.to_datetime(_snap_day)
                    _save["Profile"] = _profile_name
                    _save["Coupon_Name"] = _coupon_name
                    _save["Coupon_Tag"] = _coupon_tag
                    _save["Saved_At"] = pd.Timestamp.utcnow()
                    _save["Filters_JSON"] = json.dumps(
                        _filters, ensure_ascii=False)

                    # Result placeholders
                    for c in ["FT_Home", "FT_Away","Hit","Profit"]:
                        if c not in _save.columns:
                            _save[c] = np.nan if c != "Profit" else 0.0

                    # Stable pick id to avoid duplicates
                    def _mk_pick_id(r):
                        # Snapshot_Date is a Timestamp; keep YYYY-MM-DD for
                        # stable id
                        sd = r.get("Snapshot_Date")
                        sd_s = str(sd)[:10] if sd is not None else ""
                        return f"{sd_s}|{str(r.get('Match_ID', ''))}|{str(r.get('Seçim',''))}|{str(r.get('Profile',''))}"
                    _save["Pick_ID"] = _save.apply(_mk_pick_id, axis=1)

                    pl = load_pick_log()
                    if pl is None or pl.empty:
                        pl2 = _save
                    else:
                        # pandas concat can crash if either DF has duplicate
                        # column labels
                        pl = pl.copy()
                        _save = _save.copy()
                        try:
                            pl = pl.loc[:, ~pl.columns.duplicated()].copy()
                        except Exception:
                            pass
                        try:
                            _save = _save.loc[:,
     ~_save.columns.duplicated()].copy()
                        except Exception:
                            pass
                        pl = pl.reset_index(drop=True)
                        _save = _save.reset_index(drop=True)
                        pl2 = pd.concat(
                            [pl, _save], ignore_index=True, sort=False)

                    # Dedup keep latest
                    if "Pick_ID" in pl2.columns:
                        pl2["Pick_ID"] = pl2["Pick_ID"].astype(str)
                        pl2 = pl2.drop_duplicates(
                            subset=["Pick_ID"], keep="last")

                    save_pick_log(pl2)
                    st.success(
                        "Pick Log kaydedildi. '📅 Günün Maçları (Kayıt)' tab'inde Pick Log ekranından görebilirsin.")
                    try:
                        st.rerun()
                    except Exception:
                        pass


        else:
            st.info("Bu slider ayarlarına göre Best Of listesi boş.")
        st.subheader("📂 Geniş Aday Listesi (Slider Filtreleri)")

        if not pool_bets.empty:
            # Strength + BestOfRank + Stars (Pool)
            # GÜNCELLEME: Artık dinamik/kalibre edilmiş fonksiyonu kullanıyoruz

            # Doğru strength kaynağını seç
            if "SIM_ANCHOR_STRENGTH" in pool_bets.columns:
                _str_col = "SIM_ANCHOR_STRENGTH"
            elif "AnchorStrength" in pool_bets.columns:
                _str_col = "AnchorStrength"
            else:
                _str_col = "Score"

            # Dinamik fonksiyonu çağır (Bu fonksiyon barajları havuzun durumuna
            # göre esnetir)
            pool_bets = add_strength_cols(pool_bets, strength_col=_str_col)

            # Yıldızları ekle
            pool_bets = add_star_rating_by_bestofrank(pool_bets)

            # -----------------------------
            # Prediction Log (Broad Candidate List snapshot)
            # -----------------------------
                       # (Removed legacy Prediction Log UI block)

            st.session_state["_dbg_pool_bets_rows"] = int(len(pool_bets)) if pool_bets is not None else 0

            with st.expander("🧭 KNN Güvenilirlik Filtresi (Geniş Liste)", expanded=False):
                use_knn_gate_pool = st.checkbox(
    "KNN Gate'i uygula (Geniş Liste)",
    value=bool(
        st.session_state.get(
            "knn_gate_pool_apply",
            False)),
            key="knn_gate_pool_apply",
             )
                min_simq_pool = st.slider(
    "Min SIM_QUALITY",
    0.0,
    1.0,
    float(
        st.session_state.get(
            "knn_gate_pool_min_simq",
            0.30)),
            0.01,
            key="knn_gate_pool_min_simq",
             )
                min_en_pool = st.slider(
                    "Min EFFECTIVE_N",
                    0, 50,
                    int(st.session_state.get("knn_gate_pool_min_en", 6)),
                    1,
                    key="knn_gate_pool_min_en",
                )

            if use_knn_gate_pool and pool_view is not None and not pool_view.empty:
                st.session_state["_dbg_pool_view_rows_before_simq_gate"] = 0 if pool_view is None else int(len(pool_view))
                st.session_state["_dbg_min_simq_pool"] = float(min_simq_pool)
                st.session_state["_dbg_min_en_pool"] = float(min_en_pool)
                _simq = pd.to_numeric(
    pool_view.get(
        "SIM_QUALITY",
        np.nan),
         errors="coerce")
                _en = pd.to_numeric(
    pool_view.get(
        "EFFECTIVE_N_REAL",
        pool_view.get(
            "EFFECTIVE_N",
            np.nan)),
            errors="coerce",
             )
                pool_view = pool_view[
                    (_simq.fillna(-1) >= float(min_simq_pool)) &
                    (_en.fillna(-1) >= float(min_en_pool))
                ].copy()
                pool_view = pool_view if pool_view is not None else pd.DataFrame()
                st.session_state["_dbg_pool_view_rows_after_simq_gate"] = 0 if pool_view is None else int(len(pool_view))
            else:
                st.session_state["_dbg_pool_view_rows_before_simq_gate"] = (
                    0 if pool_view is None else int(len(pool_view)))
                st.session_state["_dbg_pool_view_rows_after_simq_gate"] = (
                    0 if pool_view is None else int(len(pool_view)))

            # Safety: pool_view must be defined (some branches may skip its
            # creation)
            if 'pool_view' not in locals() or pool_view is None:
                try:
                    pool_view = _ensure_unique_index(pool_bets.copy()) if (
    'pool_bets' in locals() and pool_bets is not None) else pd.DataFrame()
                except Exception:
                    pool_view = pd.DataFrame()
            pool_view = pool_view if pool_view is not None else pd.DataFrame()

            st.session_state["_dbg_pool_view_rows_pre_style"] = (
                0 if pool_view is None else int(len(pool_view)))
            if pool_view is None:
                pool_view = pd.DataFrame()
            st.session_state["_dbg_pool_view_cols_pre_style"] = list(pool_view.columns)
            if _dbg_col_audit_enabled():
                _maybe_record_col_lineage(
                    pool_view, "pool: view_pre_style", "pool_view_pre_style")
                _store_col_audit(pool_view, "pool_view_pre_style")
            if pool_view is not None and not pool_view.empty:
                for _c in ["TG_Q75", "TG_Q90", "EFFECTIVE_N", "SIM_ANCHOR_STRENGTH", "_EN_PEN"]:
                    if _c in pool_view.columns:
                        pool_view[_c] = pd.to_numeric(pool_view[_c], errors="coerce").fillna(0.0)
                    else:
                        pool_view[_c] = 0.0
                for _c in ["CORE_ICON", "AUTOMOD_ICON"]:
                    if _c in pool_view.columns:
                        pool_view[_c] = pool_view[_c].fillna("—")
                    else:
                        pool_view[_c] = "—"

            # --- Style profile pack for Pool view (looser than BestOf; keeps radar behavior) ---
            st.session_state["_dbg_pool_view_rows_before_profile_pack"] = 0 if pool_view is None else int(len(pool_view))
            try:
                _style_mode_now = str(
    st.session_state.get(
        "gr_style_mode", "✅ GÜVENLİ"))
                pool_view = _apply_style_profile_pack(
                    pool_view, _style_mode_now, scope="pool")
            except Exception:
                _track_view_error("pool_view: style_profile_pack")
            pool_view = pool_view if pool_view is not None else pd.DataFrame()
            st.session_state["_dbg_pool_view_rows_after_profile_pack"] = 0 if pool_view is None else int(len(pool_view))

            if pool_view is None:
                pool_view = pd.DataFrame()
            st.session_state["_dbg_pool_view_rows_before_trap_cols"] = 0 if pool_view is None else int(len(pool_view))
            try:
                pool_view = _ensure_trap_cols(pool_view)
            except Exception:
                _track_view_error("pool_view: ensure_trap_cols")
            pool_view = pool_view if pool_view is not None else pd.DataFrame()
            st.session_state["_dbg_pool_view_rows_after_trap_cols"] = 0 if pool_view is None else int(len(pool_view))
            if pool_view is None:
                pool_view = pd.DataFrame()
            pool_view = pool_view if pool_view is not None else pd.DataFrame()

            st.session_state["_dbg_pool_view_rows_post_style"] = (
                0 if pool_view is None else int(len(pool_view)))
            _safe_view_cols(pool_view, REQUIRED_VIEW_COLS, tag="pool")

            # Karakter (Advanced Badge) — geniş liste ile BestOf senkron olsun
            pool_view = pool_view if isinstance(pool_view, pd.DataFrame) else pd.DataFrame()
            if pool_view is None:
                pool_view = pd.DataFrame()
            try:
                if pool_view is None:
                    pool_view = pd.DataFrame()
                if pool_view is None:
                    pool_view = pd.DataFrame()
                if "Karakter" not in pool_view.columns:
                    pool_view["Karakter"] = pool_view.apply(
                        get_match_character_badge, axis=1)
            except Exception:
                if pool_view is None:
                    pool_view = pd.DataFrame()
                _track_view_error("pool_view: karakter")
                pool_view["Karakter"] = ""

            # Archetype include/exclude toggles (fast UI control)
            pool_view = pool_view if isinstance(pool_view, pd.DataFrame) else pd.DataFrame()
            if pool_view is None:
                pool_view = pd.DataFrame()
            if "Archetype" in pool_view.columns:
                _arch = pool_view["Archetype"].astype(str).fillna("NONE")
                _arch_vals = sorted(
                    {a for a in _arch.unique() if a and a.upper() != "NONE"})
                _arch_desc = {
                    "1X2_TRAP": "Favori tarafında tuzak profili",
                    "OU_NO_EDGE": "OU var ama edge yok / zayıf",
                    "OU_NONE": "OU sinyali yok",
                    "BTTS_NO_EDGE": "KG var ama edge yok / zayıf",
                    "BTTS_NONE": "KG sinyali yok",
                    "1X2_NONE": "1X2 sinyali yok",
                }
                with st.expander("Archetype filtreleri (aç/kapat)", expanded=False):
                    st.caption(
                        "İşaretini kaldırdıkların tablodan çıkarılır (NONE satırları varsayılan olarak kalır).")
                    cA, cB = st.columns(2)
                    _allowed = set()
                    for i, a in enumerate(_arch_vals):
                        _label = f"{a} — {_arch_desc.get(a, 'Etiket')}"
                        _key = f"wf_arch_on__{a}"
                        _checked = (
    cA if i %
    2 == 0 else cB).checkbox(
        _label,
        value=True,
         key=_key)
                        if _checked:
                            _allowed.add(a)
                    _hide_none = st.checkbox(
    "NONE (etiketsiz) satırları gizle",
    value=False,
     key="wf_arch_hide_none")

                if _arch_vals:
                    _mask = _arch.isin(list(_allowed))
                    if not _hide_none:
                        _mask = _mask | (_arch == "NONE")
                    _mask = _align_mask(_mask, pool_view)
                    pool_view = pool_view[_mask].copy()
                    pool_view = pool_view if pool_view is not None else pd.DataFrame()

            # Ensure AutoMod columns + Final Confidence for pool view
# IMPORTANT: keep CONF_ICON stable; don't overwrite after filters.
            # AutoMod icon for main view (UX)
            pool_view = pool_view if isinstance(pool_view, pd.DataFrame) else pd.DataFrame()
            if pool_view is None:
                pool_view = pd.DataFrame()
            if "AutoMod_Status" in pool_view.columns:
                pool_view["AUTOMOD_ICON"] = pool_view["AutoMod_Status"].map({"High": "🟢","Medium":"🟡","Low":"🔴"}).fillna("🟡")
            else:
                if pool_view is None:
                    pool_view = pd.DataFrame()
                if "AMQS_percentile" in pool_view.columns:
                    _p = pd.to_numeric(
        pool_view["AMQS_percentile"],
         errors="coerce").fillna(0.50)
                    pool_view["AUTOMOD_ICON"] = np.select([_p >= 0.80, _p >= 0.60], ["🟢", "🟡"], default="🔴")
                else:
                    pool_view["AUTOMOD_ICON"] = "🟡"

            # Trap visibility (show without advanced columns)
            pool_view = pool_view if isinstance(pool_view, pd.DataFrame) else pd.DataFrame()
            if pool_view is None:
                pool_view = pd.DataFrame()
            if "Trap_Flag" in pool_view.columns:
                pool_view["TRAP_ICON"] = np.where(
    pd.to_numeric(
        pool_view["Trap_Flag"],
        errors="coerce").fillna(0).astype(bool),
        "⚠️",
         "")
            else:
                if pool_view is None:
                    pool_view = pd.DataFrame()
                if "Archetype" in pool_view.columns:
                    pool_view["TRAP_ICON"] = np.where(
        pool_view["Archetype"].astype(str).eq("TRAP_FAVORITE"), "⚠️", "")
                else:
                    pool_view["TRAP_ICON"] = ""
            # Icon inclusion flags (Pool panel)
            st.session_state["_dbg_pool_view_rows_before_label_filters"] = 0 if pool_view is None else int(len(pool_view))
            show_trap_icon = bool(st.session_state.get("bo_inc_trap", True))
            show_danger_icon = bool(
    st.session_state.get(
        "bo_inc_danger", True))
            show_arch_icon = bool(st.session_state.get("bo_inc_profile", True))

            # FILTER OUT rows when user disables a label
            if pool_view is None:
                pool_view = pd.DataFrame()
            if not show_trap_icon and "Trap_Flag" in pool_view.columns:
                pool_view = pool_view[~pd.to_numeric(
                    pool_view["Trap_Flag"], errors="coerce").fillna(0).astype(bool)].copy()
                pool_view = pool_view if pool_view is not None else pd.DataFrame()
            if pool_view is None:
                pool_view = pd.DataFrame()
            if not show_danger_icon and "Danger_Flag" in pool_view.columns:
                pool_view = pool_view[~pd.to_numeric(
                    pool_view["Danger_Flag"], errors="coerce").fillna(0).astype(bool)].copy()
                pool_view = pool_view if pool_view is not None else pd.DataFrame()
            if pool_view is None:
                pool_view = pd.DataFrame()
            if not show_arch_icon and "Archetype" in pool_view.columns:
                _arch = pool_view["Archetype"].astype(
                    str).fillna("NONE").str.upper()
                _is_neutral = _arch.eq("NONE") | _arch.str.endswith("_NONE")
                pool_view = pool_view[_is_neutral].copy()
                pool_view = pool_view if pool_view is not None else pd.DataFrame()
            st.session_state["_dbg_pool_view_rows_after_label_filters"] = 0 if pool_view is None else int(len(pool_view))
            st.session_state["_dbg_min_profile_conf"] = st.session_state.get(
                "gr_min_profile_conf", None)
            if pool_view is not None and not pool_view.empty:
                _row0 = pool_view.iloc[0]
                st.session_state["_dbg_pool_view_row0"] = {
                    "SIM_QUALITY": _row0.get("SIM_QUALITY", None),
                    "EFFECTIVE_N": _row0.get("EFFECTIVE_N", None),
                    "PROFILE_CONF": _row0.get("PROFILE_CONF", None),
                    "Trap_Flag": _row0.get("Trap_Flag", None),
                    "Danger_Flag": _row0.get("Danger_Flag", None),
                    "Archetype": _row0.get("Archetype", None),
                }
            else:
                st.session_state["_dbg_pool_view_row0"] = {}


                       # ---- Phase-2 UI: lean default columns & clear naming ----
            if pool_view is None:
                pool_view = pd.DataFrame()
            if "AnchorStrengthCategory" not in pool_view.columns and "StrengthCategory" in pool_view.columns:
                pool_view["AnchorStrengthCategory"] = pool_view["StrengthCategory"]
            # Ensure core similarity columns exist (never show as blank if we
            # can derive)
            if pool_view is None:
                pool_view = pd.DataFrame()
            if "SIM_ANCHOR_STRENGTH" not in pool_view.columns:
                _src_strength = pool_view.get(
    "AnchorStrength", pool_view.get(
        "Score", np.nan))
                pool_view["SIM_ANCHOR_STRENGTH"] = pd.to_numeric(
                    _src_strength, errors="coerce")
            if pool_view is None:
                pool_view = pd.DataFrame()
            if "EFFECTIVE_N" not in pool_view.columns:
                _src_en = pool_view.get(
    "EFFECTIVE_N_REAL", pool_view.get(
        "SIM_N", 0))
                pool_view["EFFECTIVE_N"] = pd.to_numeric(
                    _src_en, errors="coerce")

            base_cols = [
                "Match_ID",
                "Date", "League","MDL_rank","MDL_quantile",
                "CONF_ICON", "AUTOMOD_ICON","TRAP_ICON","DANGER_ICON","ARCH_ICON",
                "Karakter",
                "HomeTeam", "AwayTeam","Seçim","Odd","Prob","EV",
                "MDL_BOOST", "GoldenScore_MDL",
                "STRENGTH_ICON", "AnchorStrengthCategory","SIM_ANCHOR_STRENGTH","EFFECTIVE_N",
                "Star_Rating", "BestOfRank"
            ]

            if not show_trap_icon and "TRAP_ICON" in base_cols:
                base_cols.remove("TRAP_ICON")
            if not show_danger_icon and "DANGER_ICON" in base_cols:
                base_cols.remove("DANGER_ICON")
            if not show_arch_icon and "ARCH_ICON" in base_cols:
                base_cols.remove("ARCH_ICON")
            adv_cols = [
                "Score", "GoldenScore","League_Conf","Market_Conf_Score",
                "Final_Confidence", "CONF_percentile","CONF_Status",
                "AMQS", "AMQS_percentile","AutoMod","AutoMod_Status","Archetype"
            ]
            # Fail-safe: advanced columns toggle

            show_adv_cols = bool(st.session_state.get('show_adv_cols', False))
            cols = base_cols + (adv_cols if show_adv_cols else [])
            pool_view = pool_view if isinstance(pool_view, pd.DataFrame) else pd.DataFrame()
            cols = _safe_view_cols(pool_view, cols, tag="pool")
            cols = list(dict.fromkeys(cols))  # dedupe columns

            # Extra market-debug columns (optional)
            if show_adv_cols:
                _extra_dbg = ["EV_pure_pct", "Implied_Prob","Model_vs_Market","Trap_Flag","Danger_Flag","Trap_Type","Archetype","SIM_ANCHOR_GROUP","SIM_ANCHOR_STRENGTH","SIM_ALPHA","SIM_QUALITY","EFFECTIVE_N","SIM_POver","SIM_PBTTS","SIM_MS_STRENGTH","SIM_OU_STRENGTH","SIM_BTTS_STRENGTH","P_Over_Model","P_Over_Final","P_BTTS_Model","P_BTTS_Final","P_Home_Model","P_Home_Final","P_Draw_Model","P_Draw_Final","P_Away_Model","P_Away_Final","BLEND_MODE_MS","LEAGUE_OK_MS","BLEND_W_LEAGUE_MS","MS_LEAGUE_SIM_P2","MS_GLOBAL_SIM_P2","BLEND_MODE_OB","LEAGUE_OK_OB","BLEND_W_LEAGUE_OB","OB_LEAGUE_SIM_POver","OB_GLOBAL_SIM_POver"]
                if pool_view is None:
                    pool_view = pd.DataFrame()
                cols += [c for c in _extra_dbg if c in pool_view.columns and c not in cols]
            # AutoMod badge column (UX)
            if pool_view is None:
                pool_view = pd.DataFrame()
            if "AutoMod_Status" in pool_view.columns and "AutoMod" not in pool_view.columns:
                pool_view["AutoMod"] = pool_view["AutoMod_Status"].apply(
                    _automod_badge)

            # Deterministic pool sorting (stable) to prevent flickering
            pool_view = _ensure_tie_key(pool_view)
            pool_view = pool_view if pool_view is not None else pd.DataFrame()
            pool_view = pool_view if isinstance(pool_view, pd.DataFrame) else pd.DataFrame()
            _pv = _style_df_for_view(pool_view, cols)

            # Effective-N mid-band preference (optional)
            if "EFFECTIVE_N" in _pv.columns:
                _pv["_EN_PEN"] = (
    pd.to_numeric(
        _pv["EFFECTIVE_N"],
        errors="coerce") -
         40.0).abs()

            sort_cols_pool = []
            asc_pool = []

            for c, a in [
                ("Star_Rating", False),
                ("BestOfRank", False),
                ("SIM_QUALITY", False),
                ("_EN_PEN", True),
                ("Odd", True),
                ("Date", True),
                ("HomeTeam", True),
                ("AwayTeam", True),
                ("_TIE", True),
            ]:
                if c in _pv.columns:
                    sort_cols_pool.append(c)
                    asc_pool.append(a)

            _df_pool = (
                _pv.sort_values(sort_cols_pool, ascending=asc_pool, kind="mergesort")
                .reset_index(drop=True)
            )
            _df_pool = _df_pool.loc[:, ~_df_pool.columns.duplicated()].copy()

            _styler2 = _df_pool.style.format({
                'Prob': '{:.2%}',
                'EV': '{:.2f}',
                'Score': '{:.3f}',
                'BestOfRank': '{:.3f}',
                'AMQS': '{:.2f}',
                'AMQS_percentile': '{:.0%}',
                'Final_Confidence': '{:.3f}',
                'CONF_percentile': '{:.0f}',
            })

            if "AutoMod_Status" in _df_pool.columns:
                _styler2 = _styler2.applymap(
    _automod_css, subset=["AutoMod_Status"])
            if "AutoMod" in _df_pool.columns:
                _styler2 = _styler2.set_properties(
                    subset=["AutoMod"], **{"font-weight": "700"})

            st.dataframe(_styler2)

            # --- Export (clean CSV) ---
            try:
                # Use the underlying pool_view dataframe (not the display view) to ensure Match_ID
                _export_pool = ensure_match_id(pool_view.copy())
                if "MDL_rank" not in _export_pool.columns or "MDL_quantile" not in _export_pool.columns:
                    _tmp3 = ensure_bestofrank(_export_pool.copy())
                    for c in ["MDL_rank", "MDL_quantile"]:
                        if c in _tmp3.columns:
                            _export_pool[c] = _tmp3[c]
                if "Match_ID" not in _export_pool.columns:
                    raise RuntimeError("Match_ID missing in export after ensure_match_id")
                export_cols = _safe_view_cols(
                    _export_pool, REQUIRED_VIEW_COLS + ["Match_ID"], tag="pool_export")
                if export_cols:
                    _export_pool = _export_pool[export_cols]
                csv_bytes = _export_pool.to_csv(
    index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
    "⬇️ Havuz CSV (temiz)",
    data=csv_bytes,
    file_name="pool_export.csv",
     mime="text/csv")
            except Exception:
                pass

        else:
            st.info("Bu slider ayarlarına göre havuzda maç kalmadı.")


    with tab_qc:
        st.subheader("🧾 QC Panel – KNN & Liste Sağlık Kontrolü")
        st.caption("Bu panel, KNN hattındaki 1→6 adımlarının (Hijyen → Güvenilirlik → Market-aware → Risk/Dağılım → Zaman rejimi → Gating) UI üzerinden doğrulanması için tasarlandı.")

        def _qc_percentiles(s):
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return {"count": 0}
            return {
                "count": int(s.shape[0]),
                "min": float(s.min()),
                "p10": float(s.quantile(0.10)),
                "p50": float(s.quantile(0.50)),
                "p90": float(s.quantile(0.90)),
                "max": float(s.max()),
            }

        def _qc_table_from_dict(d, title):
            try:
                st.markdown(f"**{title}**")
                st.dataframe(
    pd.DataFrame(
        [d]),
        use_container_width=True,
         hide_index=True)
            except Exception:
                st.write(title, d)

        def _qc_df_summary(df, name):
            if df is None or getattr(df, "empty", True):
                st.warning(f"{name}: boş / yok.")
                return

            st.markdown(f"### {name}")
            st.write(
                f"Satır: **{len(df):,}** | Kolon: **{len(df.columns):,}**")

            # 1) Hijyen göstergeleri (varsa)
            hij_cols = [
    c for c in [
        "LEAK_OK",
        "POOL_MAX_DATE",
        "FILTERED_FUTURE_N",
        "FILTERED_SAME_DAY_N",
         "DUP_DROPPED_N"] if c in df.columns]
            if hij_cols:
                st.markdown("**Hijyen (leak / pool) bayrakları**")
                view = df[hij_cols].copy()
                # show simple aggregates
                agg = {}
                if "LEAK_OK" in view.columns:
                    agg["LEAK_OK_%"] = float(
    pd.to_numeric(
        view["LEAK_OK"],
        errors="coerce").fillna(0).mean() *
         100.0)
                for c in [
    "FILTERED_FUTURE_N",
    "FILTERED_SAME_DAY_N",
     "DUP_DROPPED_N"]:
                    if c in view.columns:
                        agg[c +"_sum"] = float(pd.to_numeric(view[c], errors="coerce").fillna(0).sum())
                if "POOL_MAX_DATE" in view.columns:
                    try:
                        agg["POOL_MAX_DATE_max"] = str(pd.to_datetime(
                            view["POOL_MAX_DATE"], errors="coerce").max())
                    except Exception:
                        pass
                _qc_table_from_dict(agg, "Özet")
                st.dataframe(
    view.head(50),
    use_container_width=True,
     hide_index=True)

            # 2) Güvenilirlik metrikleri
            rel_candidates = [
    c for c in [
        "SIM_QUALITY",
        "EFFECTIVE_N",
        "EFFECTIVE_N_REAL",
        "KNN_OK",
        "SIMQ_OU",
        "EN_OU",
        "SIMQ_BTTS",
        "EN_BTTS",
        "SIMQ_MS",
         "EN_MS"] if c in df.columns]
            if rel_candidates:
                st.markdown("**Güvenilirlik (SIM_QUALITY / EFFECTIVE_N)**")
                rows = []
                for c in [
    "SIM_QUALITY",
    "EFFECTIVE_N",
    "EFFECTIVE_N_REAL",
    "SIMQ_OU",
    "EN_OU",
    "SIMQ_BTTS",
    "EN_BTTS",
    "SIMQ_MS",
     "EN_MS"]:
                    if c in df.columns:
                        rows.append({"metric": c, **_qc_percentiles(df[c])})
                if rows:
                    st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
     hide_index=True)
                if "KNN_OK" in df.columns:
                    ok = pd.to_numeric(df["KNN_OK"], errors="coerce")
                    ok_rate = float(ok.fillna(0).mean() * 100.0)
                    st.info(
                        f"KNN_OK geçiş oranı: **{ok_rate:.1f}%** (bu oran slider eşiklerine göre değişmelidir)")

            # 3) Market-aware context (varsa)
            ctx_cols = [
    c for c in [
        "SIM_POver",
        "SIM_PBTTS",
        "SIM_P2",
        "SIM_POver_OU",
        "SIM_PBTTS_BTTS",
         "SIM_P2_MS"] if c in df.columns]
            if ctx_cols:
                st.markdown("**Market context (benzer maçlardan oranlar)**")
                rows = [{"metric": c,
     **_qc_percentiles(df[c])} for c in ctx_cols]
                st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
     hide_index=True)

            # 4) Risk/dağılım (Sprint 4)
            risk_cols = [
    c for c in [
        "SIM_RISK_OU",
        "SIM_RISK_BTTS",
        "SIM_TAIL_3PLUS_OU",
        "SIM_TG_STD_OU",
        "SIM_TG_IQR_OU",
        "SIM_TAIL_00_BTTS",
         "SIM_VAR_PBTTS_BTTS"] if c in df.columns]
            if risk_cols:
                st.markdown("**Risk/Dağılım (Sprint 4)**")
                # categorical counts
                cat = [c for c in risk_cols if str(df[c].dtype) == "object"]
                if cat:
                    for c in cat:
                        st.write(c)
                        st.dataframe(
    df[c].value_counts(
        dropna=False).to_frame("count").T,
        use_container_width=True,
         hide_index=True)
                num = [c for c in risk_cols if c not in cat]
                if num:
                    rows = [{"metric": c,
     **_qc_percentiles(df[c])} for c in num]
                    st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
     hide_index=True)

        # ---------------------------------------------------------------------
        # Sprint 5: Zaman rejimi (UI kontrol)
        # ---------------------------------------------------------------------
        st.markdown("## Sprint 5 – Zaman Rejimi (Time-decay) Kontrolü")
        use_time_decay = bool(
    st.session_state.get(
        "knn_use_time_decay",
        False) or st.session_state.get(
            "use_time_decay",
             False))
        half_life = st.session_state.get(
    "knn_half_life_days", st.session_state.get(
        "half_life_days", None))
        st.write(f"use_time_decay: **{use_time_decay}**")
        if half_life is not None:
            st.write(f"half_life_days: **{half_life}**")
        st.caption(
            "Beklenen davranış: time-decay aç/kapa ve half-life küçült/büyüt yaptığında SIM_QUALITY / SIM_P* bazı maçlarda değişmeli.")

        # ---------------------------------------------------------------------
        # Dataframes (mevcut olanları göster)
        # ---------------------------------------------------------------------
        st.markdown("## Veri Setleri (UI'da görünen)")
        # best_of / best_view / pool_view / pool_bets gibi değişkenler farklı
        # yerlerde oluşabiliyor.
        candidates = []
        for nm in [
    "best_of",
    "best_view",
    "df_display",
    "pool_view",
    "pool_bets",
    "pool_bets_bestof",
     "bets_df"]:
            if nm in locals():
                candidates.append(nm)

        if not candidates:
            st.warning("QC: Bu tab açıldığında henüz listeler oluşturulmamış olabilir. Önce BestOf veya Geniş Liste sekmelerinde bir kez üretim yapıp sonra QC'ye gel.")
        else:
            st.write("Bulunan tablo değişkenleri:", ", ".join(candidates))

        # Safely render the most useful ones
        if "best_of" in locals():
            _qc_df_summary(best_of, "BestOf (raw)")
        if "best_view" in locals():
            _qc_df_summary(best_view, "BestOf (display / best_view)")
        if "pool_view" in locals():
            _qc_df_summary(pool_view, "Geniş Liste (pool_view)")
        if "pool_bets" in locals():
            _qc_df_summary(pool_bets, "Aday Havuzu (pool_bets)")

        st.markdown("## Hızlı yorum – sağlıklı aralıklar")
        st.markdown("""
- **SIM_QUALITY**: çoğu lig/market için genelde 0.10–0.60 bandı normal. 0.90+ çok sık ise leak/ölçekleme şüphesi.
- **EFFECTIVE_N**: K=30 ise 5–25 arası doğal. 1–3 sürekli ise ağırlıklar aşırı tekilleşmiş demektir.
- **Gate testi**: KNN Gate slider’larını yükselttikçe BestOf ve Geniş Liste satır sayısı düşmeli.
""")
    with tab_wf:
        train_days = int(st.session_state.get('wf_train_days', 90))
        test_days = int(st.session_state.get('wf_test_days', 14))
        st.subheader(
    f"🧪 Walk-Forward Filter Lab – Filtre Aralığı Keşfi ({train_days}g öğren / {test_days}g test)")

        # Source selection: dynamic hist_df vs persistent prediction log
        hist_df = st.session_state.get("hist_df", pd.DataFrame())
        # Full historical core (for date-range awareness / UI bounds)
        core_df = st.session_state.get("train_core", pd.DataFrame())
        try:
            if core_df is not None and not core_df.empty and "Date" in core_df.columns:
                core_df = core_df.copy()
                core_df["Date"] = _robust_to_datetime(core_df.get("Date"))
                core_df = core_df.dropna(subset=["Date"]).copy()
        except Exception:
            core_df = st.session_state.get("train_core", pd.DataFrame())

        pred_log = load_prediction_log()

        # --- Retro snapshot builder (from existing hist_df) ---
        with st.expander("🔁 Retro Snapshot (hist_df üzerinden) – Prediction Log'u geriye doldur", expanded=False):
            st.caption(
    "Bu işlem model retrain etmez; sadece mevcut hist_df satırlarını gün gün snapshot olarak Prediction Log'a yazar. "
    "Ama WF Lab için çok faydalıdır (stabil aday evreni + daha çok gün).")
            if hist_df is None or hist_df.empty:
                st.warning("hist_df boş. Önce modeli çalıştırıp hist_df üret.")
            else:
                _min_d = pd.to_datetime(
    hist_df.get("Date"), errors="coerce").min()
                _max_d = pd.to_datetime(
    hist_df.get("Date"), errors="coerce").max()
                c1, c2,c3 = st.columns([2,2,1])
                with c1:
                    retro_start = st.date_input(
    "Retro başlangıç (hist_df)",
     value=_min_d.date() if pd.notna(_min_d) else None)
                with c2:
                    retro_end = st.date_input(
    "Retro bitiş (hist_df)",
     value=_max_d.date() if pd.notna(_max_d) else None)
                with c3:
                    max_days = st.number_input(
    "Max gün",
    min_value=1,
    max_value=5000,
    value=500,
    step=50,
     help="Çok geniş aralıkta UI'yi kilitlememek için güvenlik limiti.")
                if st.button("Retro snapshot üret", key="wf_retro_snap_btn"):
                    days_ok, days_skip, msgs = backfill_prediction_log_from_hist_df(
                        hist_df,
                        pd.to_datetime(retro_start),
                        pd.to_datetime(retro_end),
                        meta={"mode": "retro_from_hist_df","source":"wf_lab"},
                        max_days=int(max_days)
                    )
                    st.success(
    f"Retro snapshot tamam: {days_ok} gün eklendi, {days_skip} gün atlandı.")
                    for m in msgs[:10]:
                        st.caption(m)
                    st.info(
                        "Şimdi sayfayı bir kez yenile (R) veya sekmeden çıkıp gel: Prediction Log dolunca tarih kısıtı genişler.")

        use_pred_log = False
        if pred_log is not None and not pred_log.empty:
            use_pred_log = st.checkbox(
    "Prediction Log ile çalış (snapshots – önerilir)", value=True)
        else:
            st.caption(
                "Prediction Log boş. İstersen 'Best Of' tab'inde 'Prediction Log'a Kaydet' ile günlük snapshot al.")

        if use_pred_log:
            if hist_df is None or hist_df.empty:
                st.info(
                    "Prediction Log var ama sonuç (FT skor) için hist_df gerekiyor. Önce geçmiş CSV'leri yükleyip modeli çalıştır.")
                _h = pd.DataFrame()
            else:
                # Merge predictions (from log) with realized outcomes (from
                # hist_df)
                _log = pred_log.copy()
                _hist = hist_df.copy()

                for dcol in ["Date"]:
                    if dcol in _log.columns:
                        _log[dcol] = pd.to_datetime(
                            _log[dcol], errors="coerce")
                    if dcol in _hist.columns:
                        _hist[dcol] = pd.to_datetime(
                            _hist[dcol], errors="coerce")

                # Join on stable keys
                k = [c for c in ["Date", "League","HomeTeam","AwayTeam","Seçim"] if c in _log.columns and c in _hist.columns]
                if not k:
                    st.warning(
                        "Prediction Log ile hist_df arasında birleştirme anahtarı bulunamadı.")
                    _h = pd.DataFrame()
                else:
                    # Keep only necessary outcome columns from hist
                    hcol, acol = _detect_ft_cols(_hist)
                    keep_out = k.copy()
                    if hcol and acol:
                        keep_out += [hcol, acol]
                    _hist2 = _hist[keep_out].drop_duplicates(
                        subset=k, keep="last")

                    _h = _log.merge(_hist2, on=k, how="left")
                    # Ensure required policy columns exist
                    _h = ensure_amqs_columns(_h)
                    _h = ensure_conf_percentile_columns(_h, by_market=True)
        else:
            _h = hist_df

        # --- Default safety: exclude low-confidence result matches from WF/Lab ---
        # This prevents silent mis-joins from polluting policy search. Users
        # can opt-in to include them.
        exclude_low_conf_results = st.checkbox(
    "Sonuç eşleşme güveni düşük olanları dışla (önerilir)",
    value=True,
    help="Pick/Prediction log sonuç bağlama sırasında bazı maçlar NO_MATCH veya düşük güvenle eşleşebilir. "
    "Bu filtre varsayılan olarak RESULT_MATCH_CONFIDENCE < 0.80 olanları Walk-Forward analizine sokmaz." )
        if exclude_low_conf_results and _h is not None and not _h.empty:
            if "RESULT_MATCH_CONFIDENCE" in _h.columns:
                _conf = pd.to_numeric(
    _h.get("RESULT_MATCH_CONFIDENCE"), errors="coerce")
                _h = _h.loc[_conf.fillna(0.0) >= 0.80].copy()
            elif "RESULT_MATCH_METHOD" in _h.columns:
                _mth = _h.get("RESULT_MATCH_METHOD").astype(str)
                _h = _h.loc[_mth.isin(
                    ["STRICT", "DATE_SHIFT", "MANUAL"])].copy()
            else:
                # No confidence/method columns available; nothing to filter.
                pass

        if _h is None or _h.empty:
            st.info("Walk-forward için uygun veri bulunamadı (seçilen kaynak boş).")
        else:
            # Date range controls
            _h = _h.copy()
            _h["Date"] = _robust_to_datetime(_h.get("Date"))
            _h = _h.dropna(subset=["Date"]).copy()
            if _h.empty:
                st.warning("Hist veri içinde Date parse edilemedi.")
            else:
                pred_min_d = _h["Date"].min().date()
                pred_max_d = _h["Date"].max().date()

                # UI date bounds:
                # - If using Prediction Log, bounds are snapshot coverage.
                # - Otherwise, show full historical bounds (train_core) even if model outputs (hist_df) cover only a subset.
                if use_pred_log:
                    min_d, max_d = pred_min_d, pred_max_d
                else:
                    if core_df is not None and not core_df.empty and "Date" in core_df.columns:
                        try:
                            min_d = core_df["Date"].min().date()
                            max_d = core_df["Date"].max().date()
                        except Exception:
                            min_d, max_d = pred_min_d, pred_max_d
                    else:
                        min_d, max_d = pred_min_d, pred_max_d

                if (not use_pred_log) and (pred_min_d is not None) and (
                    pred_max_d is not None) and (min_d != pred_min_d or max_d != pred_max_d):
                    st.info(
    f"Not: Bu oturumda model çıktıları (hist_df) yalnızca **{pred_min_d} → {pred_max_d}** aralığını kapsıyor. Daha eski tarih seçebilirsin ama Walk-forward çalıştırırken bu aralık dışında veri olmayabilir. (Tüm geçmiş için: retro snapshot / prediction log gerekir.)")

                # If Prediction Log is enabled, the available date range is limited by snapshot coverage.
                # Explain this clearly so the user knows why they cannot go
                # further back.
                try:
                    if use_pred_log and (
    hist_df is not None) and (
        not hist_df.empty) and (
            "Date" in hist_df.columns):
                        _hist_tmp = hist_df.copy()
                        _hist_tmp["Date"] = _robust_to_datetime(
                            _hist_tmp.get("Date"))
                        _hist_tmp = _hist_tmp.dropna(subset=["Date"])
                        if not _hist_tmp.empty:
                            _hist_min = _hist_tmp["Date"].min().date()
                            if (_hist_min is not None) and (_hist_min < min_d):
                                st.warning(
                                    f"⚠️ Prediction Log seçiliyken başlangıç tarihi snapshot'ların başladığı güne kadar geri gidebilir: {min_d}. "
                                    f"Daha eski tarihleri test etmek için 'Prediction Log ile çalış' seçeneğini kapat."
                                )
                except Exception:
                    pass

                # Date widgets can get "stuck" / turn red if session_state keeps an out-of-range date.
                # Clamp them whenever min/max changes (e.g., when switching
                # Prediction Log on/off).
                if "wf_start" not in st.session_state:
                    st.session_state["wf_start"] = min_d
                if "wf_end" not in st.session_state:
                    st.session_state["wf_end"] = max_d
                try:
                    if st.session_state["wf_start"] < min_d:
                        st.session_state["wf_start"] = min_d
                except Exception:
                    st.session_state["wf_start"] = min_d
                try:
                    if st.session_state["wf_end"] > max_d:
                        st.session_state["wf_end"] = max_d
                except Exception:
                    st.session_state["wf_end"] = max_d
                # Also ensure ordering
                try:
                    if st.session_state["wf_end"] < st.session_state["wf_start"]:
                        st.session_state["wf_end"] = st.session_state["wf_start"]
                except Exception:
                    pass

                c1, c2, c3 = st.columns([2, 2,2])
                with c1:
                    start_d = st.date_input(
    "Başlangıç (hist)",
    key="wf_start",
    min_value=min_d,
     max_value=max_d)
                with c2:
                    end_d = st.date_input(
    "Bitiş (hist)",
    key="wf_end",
    min_value=min_d,
     max_value=max_d)
                with c3:
                    quick = st.checkbox(
    "Hızlı mod (daha az kombinasyon)", value=False)

                st.markdown("**Grid (policy) eşikleri**")

                # Kullanıcı ayarlı grid: min_CONF vb. eşikler buradan seçilir.
                # Not: Çok az pick çıkıyorsa önce burada eşikleri gevşet.
                with st.expander("Grid (policy) eşikleri", expanded=True):
                    if quick_mode:
                        st.caption("Hızlı mod: daha az kombinasyon.")
                    else:
                        st.caption(
                            "Normal mod: Sürpriz (düşük anchor) ve Banko (yüksek anchor) profillerini kapsayan geniş aralık.")

                    # --- 1. ESKİ METRİKLER (Genişletildi) ---
                    BESTOF_PRESETS = [-1_000_000_000, - \
                        2.0, -1.0, 0.0, 0.25, 0.50]
                    CONF_PRESETS = [0, 30, 50, 60, 70]
                    PROB_PRESETS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
                    EV_PRESETS = [-5, -2, 0, 1, 2]
                    AUTOMOD_PRESETS = [0, 30, 50, 70]

                    # --- 2. YENİ METRİKLER ---
                    ANCHOR_PRESETS = [
    0.0, 0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                    SIMQ_PRESETS = [0.0, 0.20, 0.25, 0.30, 0.33, 0.35, 0.36]
                    EFFN_PRESETS = [0, 8, 15, 20, 30]

                    # --- DEFAULT SEÇİMLER ---
                    d_bestof = [-1_000_000_000, 0.0]
                    d_prob = [0.35, 0.45, 0.55]
                    d_conf = [0, 50]
                    d_ev = [-2, 1]
                    d_am = [0, 50]
                    d_anc = [0.15, 0.30, 0.45]
                    d_simq = [0.20, 0.30, 0.35]
                    d_effn = [8, 20]

                    if quick_mode:
                        d_prob = [0.45]
                        d_anc = [0.30]
                        d_simq = [0.30]

                    g1, g2, g3, g4 = st.columns(4)
                    with g1:
                        bestof_grid = st.multiselect(
    "BestOfRank min", BESTOF_PRESETS, default=d_bestof)
                        prob_grid = st.multiselect(
    "Prob min", PROB_PRESETS, default=d_prob)
                    with g2:
                        conf_grid = st.multiselect(
    "CONF min", CONF_PRESETS, default=d_conf)
                        anchor_grid = st.multiselect(
    "⚓ Anchor Str. min", ANCHOR_PRESETS, default=d_anc)
                    with g3:
                        ev_grid = st.multiselect(
    "EV min", EV_PRESETS, default=d_ev)
                        simq_grid = st.multiselect(
    "🧬 Sim Quality min", SIMQ_PRESETS, default=d_simq)
                    with g4:
                        automod_grid = st.multiselect(
    "AutoMod min", AUTOMOD_PRESETS, default=d_am)
                        eff_n_grid = st.multiselect(
    "👥 Effective N min", EFFN_PRESETS, default=d_effn)

                    # --- Market / Seçim filtresi (opsiyonel) ---
                    MARKET_PRESETS = [
    "MS 1",
    "MS X",
    "MS 2",
    "2.5 Üst",
    "2.5 Alt",
    "KG Var",
     "KG Yok"]
                    market_filter = st.multiselect(
                        "🎯 Market filtresi (opsiyonel) — örn. sadece MS 1",
                        options=MARKET_PRESETS,
                        default=[],
                        key="wf_market_filter",
                    )

                    # Defensive sort
                    bestof_grid = tuple(
    sorted(
        set(bestof_grid),
         key=lambda x: float(x)))
                    conf_grid   = tuple(sorted(set(conf_grid), key=lambda x: float(x)))
                    ev_grid     = tuple(sorted(set(ev_grid), key=lambda x: float(x)))
                    prob_grid   = tuple(sorted(set(prob_grid), key=lambda x: float(x)))
                    automod_grid = tuple(sorted(set(automod_grid),key=lambda x: float(x)))

                    anchor_grid = tuple(
    sorted(
        set(anchor_grid),
         key=lambda x: float(x)))
                    simq_grid   = tuple(sorted(set(simq_grid), key=lambda x: float(x)))
                    eff_n_grid  = tuple(sorted(set(eff_n_grid), key=lambda x: float(x)))

                # Constraints

                cc0, cc1, cc2, cc3, cc4 = st.columns([1, 1,1,1,1.2])
                with cc0:
                    train_days = int(
    st.number_input(
        "Train gün (öğren)",
        min_value=7,
        max_value=365,
        value=90,
         step=1))
                with cc1:
                    test_days = int(
    st.number_input(
        "Test gün",
        min_value=3,
        max_value=120,
        value=14,
         step=1))
                st.session_state['wf_train_days'] = train_days
                st.session_state['wf_test_days'] = test_days
                with cc2:
                    min_ppd = float(
    st.number_input(
        "Min pick/gün (test)",
        min_value=0.0,
        max_value=20.0,
        value=0.5,
         step=0.5))
                with cc3:
                    min_total = int(
    st.number_input(
        "Min toplam pick (train)",
        min_value=0,
        max_value=5000,
        value=50,
         step=10))

                # Candidate Pool Mode: generate Top-K candidates per day BEFORE
                # policy thresholds.
                ccp1, ccp2, ccp3 = st.columns([1.2, 0.9, 0.9])
                with ccp1:
                    candidate_pool_mode = st.checkbox(
    "Candidate Pool Mode (WF için önerilir: blok başına Top-K aday üret)",
    value=True,
     help="WF'in blok başına 1 pick'e kilitlenmesini engeller: önce gün başına Top-K aday havuzu üretir, sonra policy filtreleri uygular." )
                with ccp2:
                    candidate_topk_per_day = int(
    st.number_input(
        "Top-K aday/gün",
        min_value=5,
        max_value=300,
        value=60,
         step=5))
                with ccp3:
                    candidate_topk_per_match_per_day = int(
    st.number_input(
        "Maç başına aday/gün",
        min_value=1,
        max_value=5,
        value=1,
         step=1))
                # Effective constraints (may be overridden by Exploration Mode)
                min_pick_per_day = float(min_ppd)
                min_total_pick_train = int(min_total)
                with cc4:
                    run_btn = st.button(
    "Walk-forward çalıştır", type="primary")
                hard_sanity = st.checkbox(
                    "Hard Sanity Mode (keşif için önerilir)",
                    value=True,
                    help="Policy-level min_pick guard bypass. Düşük örneklemli blokları LOW_SAMPLE / LOW_PPD olarak işaretler; sonuçlar daha oynak olabilir."
                )

                explore_mode = st.checkbox(
                    "Exploration Mode (min pick/min total override)",
                    value=True,
                    help="Keşif aşamasında pick guard + min pick kısıtlarını gevşetir: min pick/gün=0 ve min toplam pick(train)=0 uygulanır. Ayrıca Signal-only kapatılır."
                )
                if explore_mode:
                    # Keşif için en kritik kısım: kısıtlar WF'yi boğmasın.
                    min_pick_per_day = 0.0
                    min_total_pick_train = 0
                    signal_only = False
                st.caption("Hard Sanity açıkken: min_pick guard WF'yi BOĞMAZ; ama düşük örneklemli bloklar tabloya LOW_SAMPLE/LOW_PPD olarak düşer. Optimize kararını verirken bu bayraklara bak.")
                if run_btn:
                    sd = pd.to_datetime(start_d)
                    ed = pd.to_datetime(end_d)
                    if sd >= ed:
                        st.error("Başlangıç tarihi bitişten küçük olmalı.")
                    else:
                        with st.spinner("Walk-forward hesaplanıyor..."):
                            res_df, freq_df, summary_df = _run_walk_forward_policy_search(
                                hist_df=hist_df,
                                # Use the UI date inputs (start_d/end_d). Avoid NameError on date_start/date_end.
                                start_date=sd,
                                end_date=ed,
                                train_days=int(train_days),
                                test_days=int(test_days),
                                min_picks_per_day=float(min_pick_per_day),
                                min_total_picks_train=int(min_total_pick_train),
                                bestof_grid=bestof_grid,
                                conf_grid=conf_grid,
                                ev_grid=ev_grid,
                                prob_grid=prob_grid,
                                automod_grid=automod_grid,
                                anchor_grid=anchor_grid,
                                simq_grid=simq_grid,
                                eff_n_grid=eff_n_grid,
                                market_filter=tuple(market_filter) if market_filter else None,
                                odd_grid=odd_grid,
                                use_open_odds=bool(use_open_odds),
                                signal_only=signal_only,
                                candidate_pool_mode=bool(candidate_pool_mode),
                                candidate_topk_per_day=_wf_topk,
                                candidate_topk_per_match_per_day=_wf_per_match,
                                hard_sanity=bool(hard_sanity),
                                exploration_mode=exploration_mode,
                                quick_mode=bool(quick_mode),
                            )

                        if summary_df.empty or res_df.empty:
                            st.warning(
                                "Bu tarih aralığında/kurallarda yeterli pick bulunamadı (min toplam pick çok yüksek olabilir).")
                        else:
                            st.success("Tamamlandı.")

                            st.markdown("### ✅ Özet")
                            st.dataframe(summary_df)

                            st.markdown(
                                "### 🧩 En çok seçilen policy kombinasyonları")
                            if freq_df is None or freq_df.empty:
                                st.info("Frekans tablosu boş.")
                            else:
                                st.dataframe(freq_df.head(15))

                            st.markdown("### 📈 Walk-forward blok sonuçları")
                            st.dataframe(res_df)

                            # Save to session for later inspection/export
                            st.session_state["wf_results_df"] = res_df
                            st.session_state["wf_policy_freq_df"] = freq_df
                            st.session_state["wf_summary_df"] = summary_df

    with tab_saved:
        st.subheader(
            "📓 Journal – Tahmin → Pick Log → Sonuç Bağla → Performans")
        # -----------------------------------------------------------
        # SIMPLE MODE (stabil + hızlı): tek scope, 2 tablo, 1 metrik seti
        # -----------------------------------------------------------
        simple_mode = st.toggle(
    "⚡ Basit mod (önerilen)",
    value=True,
     key="jl_simple_mode")
        if simple_mode:
            # -----------------------------------------------------------
            # JOURNAL LITE: 3 blok (Açık / Sonuçlanan / İstatistik) + Manuel settle
            # -----------------------------------------------------------
            def _jl_mtime():
                p = _pick_log_path()
                csvp = p.replace(".parquet", ".csv")
                mt = -1.0
                try:
                    if os.path.exists(p):
                        mt = max(mt, os.path.getmtime(p))
                except Exception:
                    pass
                try:
                    if os.path.exists(csvp):
                        mt = max(mt, os.path.getmtime(csvp))
                except Exception:
                    pass
                return mt

            @st.cache_data(show_spinner=False)
            def _jl_load_picklog_cached(_mtime: float):
                try:
                    return load_pick_log()
                except Exception:
                    return pd.DataFrame()

            pl = _jl_load_picklog_cached(_jl_mtime())
            pl = pl.copy() if isinstance(pl, pd.DataFrame) else pd.DataFrame()

            if pl.empty:
                st.info(
                    "Pick Log boş görünüyor. Önce ana sayfadan pick kaydet veya journal klasöründeki pick_log dosyalarını kontrol et.")
                st.stop()

            # --- canonical view columns ---
            pl = canonicalize_picklog_df(pl)

            # Normalize numerics we use for stats
            for c in ["Odd", "Profit", "EV", "Prob"]:
                if c in pl.columns:
                    pl[c] = pd.to_numeric(pl[c], errors="coerce")

            # Normalize Hit to bool/NaN
            if "Hit" in pl.columns:
                # some logs store Hit as 'True'/'False' strings
                pl["Hit"] = pl["Hit"].map(lambda x: True if str(x).lower() in ["true", "1","yes"] else (False if str(x).lower() in ["false","0","no"] else x))
                pl["Hit"] = pl["Hit"].where(
                    pl["Hit"].isin([True, False]), np.nan)

            # ---------------- Scope controls ----------------
            c0, c1, c2, c3 = st.columns([1.2, 1.6, 1.2, 1.0])

            with c0:
                snapshots = sorted([s for s in pl.get("Snapshot_Date", pd.Series(
                    dtype=str)).dropna().astype(str).unique() if s and s != "nan"])
                snap_opt = ["(hepsi)"] + \
                              snapshots if snapshots else ["(hepsi)"]
                snap_sel = st.selectbox(
    "Snapshot", snap_opt, index=0, key="jl_snap_sel")

            with c1:
                prof_col = "Profile" if "Profile" in pl.columns else (
                    "Karakter" if "Karakter" in pl.columns else None)
                profiles = []
                if prof_col:
                    profiles = sorted([p for p in pl[prof_col].dropna().astype(
                        str).unique() if p and p != "nan"])
                prof_sel = st.multiselect(
    "Profil", profiles, default=[], key="jl_prof_sel")

            with c2:
                show_manual = st.checkbox(
    "MANUAL dahil", value=False, key="jl_show_manual")

            with c3:
                durum = st.radio(
    "Durum", [
        "Hepsi", "Açık", "Sonuçlanmış"], horizontal=True, key="jl_durum")

            view = pl
            if snap_sel != "(hepsi)" and "Snapshot_Date" in view.columns:
                view = view[view["Snapshot_Date"].astype(str) == str(snap_sel)]

            if prof_sel and prof_col:
                view = view[view[prof_col].astype(
                    str).isin([str(x) for x in prof_sel])]

            # manual filter (based on RESULT_MATCH_METHOD or
            # RESULT_SOURCE_ROW_ID)
            if not show_manual:
                for mc in ["RESULT_MATCH_METHOD", "RESULT_SOURCE_ROW_ID"]:
                    if mc in view.columns:
                        view = view[~view[mc].astype(
                            str).str.upper().eq("MANUAL")]

            # classify open/settled
            is_settled = view["Hit"].notna() if "Hit" in view.columns else pd.Series([False] *len(view))
            open_df = view[~is_settled].copy()
            settled_df = view[is_settled].copy()

            if durum == "Açık":
                view_main = open_df
            elif durum == "Sonuçlanmış":
                view_main = settled_df
            else:
                view_main = view

            # ---------------- Stats block (1) ----------------
            st.markdown("### 📌 İstatistik (aktif filtrelere göre)")
            total_n = len(view)
            open_n = len(open_df)
            settled_n = len(settled_df)

            total_profit = float(settled_df["Profit"].sum()) if (
                "Profit" in settled_df.columns and len(settled_df)) else 0.0
            hit_rate = float(
    settled_df["Hit"].mean()) if (
        "Hit" in settled_df.columns and len(settled_df)) else 0.0
            roi_per_pick = (total_profit / settled_n) if settled_n else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Toplam pick", f"{total_n}")
            k2.metric("Açık", f"{open_n}")
            k3.metric("Sonuçlanmış", f"{settled_n}")
            k4.metric("ROI/pick", f"{roi_per_pick:.3f}")

            k5, k6, k7 = st.columns(3)
            k5.metric("Toplam Profit", f"{total_profit:.2f}")
            k6.metric("Hit Rate", f"{hit_rate *100:.1f}%")
            k7.metric("Kaynak", "AUTO+MANUAL" if show_manual else "AUTO")

            # ---------------- Open block (2) ----------------
            st.markdown("### 🟡 Açık Pick'ler")
            open_show_cols = [c for c in ["Snapshot_Date", "Date","League","HomeTeam","AwayTeam","Seçim","Odd","Prob","EV","Pick_ID","Match_ID"] if c in open_df.columns]
            st.dataframe(
    open_df[open_show_cols].head(80) if open_show_cols else open_df.head(80),
     use_container_width=True)

            # ---------------- Manual settle (quick) ----------------
            with st.expander("🛠️ Manuel sonuçlandır (Basit)", expanded=False):
                if open_df.empty:
                    st.info("Açık pick bulunamadı (aktif filtrelere göre).")
                else:
                    # selection label
                    def _label(r):
                        try:
                            return f"{r.get('Date', '')} | {r.get('League','')} | {r.get('HomeTeam','')} - {r.get('AwayTeam','')} | {r.get('Seçim','')} | {r.get('Odd','')}"
                        except Exception:
                            return str(r.get("Pick_ID", ""))
                    # create options
                    opts = []
                    id_map = {}
                    for _, r in open_df.head(200).iterrows():
                        pid = str(r.get("Pick_ID", ""))
                        lab = _label(r)
                        opts.append(lab)
                        id_map[lab] = pid

                    sel_lab = st.selectbox(
    "Açık pick seç", opts, key="jl_manual_sel")
                    pid = id_map.get(sel_lab, None)

                    cc1, cc2, cc3 = st.columns([1, 1,1.2])
                    with cc1:
                        ft_h = st.number_input(
    "FT Home", min_value=0, max_value=20, value=0, step=1, key="jl_ft_h")
                    with cc2:
                        ft_a = st.number_input(
    "FT Away", min_value=0, max_value=20, value=0, step=1, key="jl_ft_a")
                    with cc3:
                        conf = st.slider(
    "Confidence", 0.5, 1.0, 1.0, 0.05, key="jl_conf")

                    def _eval_hit(selection: str, h: int, a: int):
                        sel = str(selection).strip().lower()
                        tg = h + a
                        # 1X2
                        if sel in ["ms 1", "ms1","1"]:
                            return h > a
                        if sel in ["ms x", "msx","x","beraber"]:
                            return h == a
                        if sel in ["ms 2", "ms2","2"]:
                            return a > h
                        # OU 2.5
                        if "2.5" in sel and (
    "üst" in sel or "over" in sel or "ou_o25" in sel):
                            return tg >= 3
                        if "2.5" in sel and (
    "alt" in sel or "under" in sel or "ou_u25" in sel):
                            return tg <= 2
                        # BTTS
                        if ("kg var" in sel) or (
                            "btts_y" in sel) or ("gg" in sel):
                            return (h >= 1) and (a >= 1)
                        if ("kg yok" in sel) or (
                            "btts_n" in sel) or ("ng" in sel):
                            return (h == 0) or (a == 0)
                        # fallback: unknown => no change
                        return None

                    if st.button(
    "✅ Manuel sonucu kilitle",
     key="jl_manual_lock"):
                        if not pid:
                            st.error("Pick_ID bulunamadı.")
                        else:
                            hit_val = None
                            # locate row in full pick log (not just filtered
                            # view)
                            full = load_pick_log()
                            full = canonicalize_picklog_df(full)

                            mask = full.get("Pick_ID", "").astype(str) == str(pid)
                            if mask.sum() == 0:
                                st.error(
                                    "Pick_ID pick log içinde bulunamadı. (Dedup/format sorunu)")
                            else:
                                row = full.loc[mask].iloc[0]
                                sel = row.get("Seçim", "")
                                odd = float(
    pd.to_numeric(
        row.get(
            "Odd",
            np.nan),
             errors="coerce")) if "Odd" in row else np.nan
                                hit_val = _eval_hit(sel, int(ft_h), int(ft_a))
                                if hit_val is None:
                                    st.warning(
                                        "Bu market otomatik değerlendirilemedi. Hit/Profit boş bırakıldı.")
                                # write fields
                                full.loc[mask, "FT_Home"] = int(ft_h)
                                full.loc[mask, "FT_Away"] = int(ft_a)
                                full.loc[mask, "Total_Goals"] = int(
                                    ft_h) + int(ft_a)
                                full.loc[mask, "Settled_At"] = pd.Timestamp.now(
                                ).isoformat()
                                full.loc[mask,
     "RESULT_MATCH_METHOD"] = "MANUAL"
                                full.loc[mask,
     "RESULT_MATCH_CONFIDENCE"] = float(conf)
                                full.loc[mask,
     "RESULT_SOURCE_ROW_ID"] = "MANUAL"
                                if hit_val is not None:
                                    full.loc[mask, "Hit"] = bool(hit_val)
                                    if not np.isnan(odd):
                                        full.loc[mask, "Profit"] = (
                                            odd - 1.0) if hit_val else -1.0
                                save_pick_log(full)
                                st.success("Manuel sonuç kaydedildi ✅")
                                st.rerun()

            # ---------------- Settled block (3) ----------------
            st.markdown("### ✅ Sonuçlanan Pick'ler")
            settled_show_cols = [c for c in ["Snapshot_Date", "Date","League","HomeTeam","AwayTeam","Seçim","Odd","Hit","Profit","FT_Home","FT_Away","Total_Goals","Settled_At","RESULT_MATCH_METHOD","Pick_ID"] if c in settled_df.columns]
            st.dataframe(
    settled_df[settled_show_cols].sort_values(
        by=["Date"],
        ascending=False).head(120) if settled_show_cols else settled_df.head(120),
         use_container_width=True)

            # stop here (skip heavy legacy panels)
            st.stop()

        else:
            st.caption(
                "Pick Log boş görünüyor. (Henüz pick kaydetmediysen normal)")

        # -----------------------------------------------------------
        # 1) PICK LOG OLUŞTUR (bugün seçtiklerini kaydet)
        # -----------------------------------------------------------
        with st.expander("1) Pick Log Oluştur (Aday listesinden pick seç ve kaydet)", expanded=False):
            st.caption(
                "Aday/Predict export (future/pool) dosyanı yükle. Buradan Auto Pick veya manuel seçimle pick_log'a yazarsın.")
            up = st.file_uploader("Aday listesi dosyası (csv / parquet)", type=["csv", "parquet"], key="jl_upl_cand")

            cand_df = pd.DataFrame()
            if up is not None:
                try:
                    if str(up.name).lower().endswith(".parquet"):
                        cand_df = pd.read_parquet(up)
                    else:
                        cand_df = pd.read_csv(up)
                except Exception as e:
                    st.error(f"Dosya okunamadı: {e}")
                    cand_df = pd.DataFrame()

            if not cand_df.empty:
                cand_df = ensure_match_id(cand_df)

                # Market filter (opsiyonel)
                mkt = st.selectbox("Market filtresi (opsiyonel)", ["(Hepsi)", "MS1","MSX","MS2","OU_O25","OU_U25","BTTS_Y","BTTS_N"], index=0, key="jl_market_filter")
                if mkt != "(Hepsi)" and "Seçim" in cand_df.columns:
                    cand_df["__mkt__"] = cand_df["Seçim"].astype(
                        str).apply(_infer_market_from_selection)
                    cand_df = cand_df[cand_df["__mkt__"] == mkt].copy()

                st.write(
    f"Aday satırı: **{
        len(cand_df)}** | Unique maç: **{
            cand_df['Match_ID'].nunique() if 'Match_ID' in cand_df.columns else '—'}**")
                st.dataframe(cand_df.head(50))

                mode = st.radio("Pick seçimi", ["Auto Pick (match başına 1)", "Manuel (checkbox)"], horizontal=True, key="jl_pick_mode")

                pick_df = pd.DataFrame()
                if mode.startswith("Auto"):
                    # Score: CONF_percentile öncelikli, BestOfRank tie-break
                    _tmp = cand_df.copy()
                    for c in ["CONF_percentile", "BestOfRank","EV","Prob","Odd","Odd_Open","AutoMod_percentile","SIM_ANCHOR_STRENGTH","SIM_QUALITY","EFFECTIVE_N"]:
                        if c in _tmp.columns:
                            _tmp[c] = pd.to_numeric(_tmp[c], errors="coerce")

                    if "CONF_percentile" not in _tmp.columns:
                        _tmp["CONF_percentile"] = 0.0
                    if "BestOfRank" not in _tmp.columns:
                        _tmp["BestOfRank"] = 0.0

                    _tmp["__pick_score__"] = _tmp["CONF_percentile"].fillna(0) *1000 + _tmp["BestOfRank"].fillna(0)
                    idx = _tmp.groupby("Match_ID")["__pick_score__"].idxmax()
                    pick_df = _tmp.loc[idx].drop(
    columns=["__pick_score__"], errors="ignore").copy()
                    st.success(f"Auto Pick üretildi: **{len(pick_df)}** pick")
                    st.dataframe(pick_df[["Date", "League","HomeTeam","AwayTeam","Seçim","Odd","Prob","EV","CONF_percentile","AutoMod_percentile","SIM_ANCHOR_STRENGTH","SIM_QUALITY","EFFECTIVE_N","Match_ID"]].head(100), use_container_width=True)
                else:
                    _tmp = cand_df.copy()
                    _tmp["_pick_"] = False
                    edited = st.data_editor(
    _tmp,
    num_rows="dynamic",
    use_container_width=True,
     key="jl_manual_editor")
                    pick_df = edited[edited["_pick_"]].copy() if isinstance(edited, pd.DataFrame) else pd.DataFrame()
                    st.info(f"Seçili pick: **{len(pick_df)}**")

                c1, c2 = st.columns([1, 2])
                with c1:
                    snapshot_date = st.date_input(
    "Snapshot_Date", value=pd.Timestamp.now(
        tz=None).date(), key="jl_snapshot_date")
                with c2:
                    profile_name = st.selectbox("Profil etiketi", ["BANKO", "ZEKİ","VALUE","GÜVENLİ","SÜRPRİZ","(NA)"], index=0, key="jl_profile_tag")

                if st.button(
    "✅ Pick'leri pick_log'a kaydet",
     key="jl_save_picks"):
                    if pick_df is None or pick_df.empty:
                        st.warning("Kaydedilecek pick yok.")
                    else:
                        out = pick_df.copy()
                        out["Snapshot_Date"] = pd.to_datetime(snapshot_date)
                        out["Profile"] = profile_name
                        out["is_pick"] = 1
                        # standart alanlar
                        for col in ["Hit", "Profit","Settled_At","FT_Home","FT_Away"]:
                            if col not in out.columns:
                                out[col] = np.nan
                        n_added = append_to_pick_log(out)
                        st.success(f"Pick log'a eklendi: **{n_added}** satır")

        # -----------------------------------------------------------
        # 2) AÇIK POZİSYONLAR (sonuç bağla)
        # -----------------------------------------------------------
        with st.expander("2) Açık Pozisyonlar – Past'tan Sonuçları Çek (Auto) + Kilitle", expanded=True):
            pl = load_pick_log()
            if pl.empty:
                st.info("pick_log boş. Önce 1) bölümünden pick kaydet.")
            else:
                pl = ensure_match_id(pl)
                open_mask = ((pl.get("Hit").isna()) if "Hit" in pl.columns else pd.Series([True] *len(pl)))
                if "Settled_At" in pl.columns:
                    open_mask = open_mask & (pl["Settled_At"].isna())
                st.write(
                    f"Açık pick: **{int(open_mask.sum())}** | Toplam pick: **{len(pl)}**")
                st.dataframe(
    pl.loc[open_mask].head(200),
     use_container_width=True)

                past_up = st.file_uploader("cleaned_past.csv (veya parquet) yükle", type=["csv", "parquet"], key="jl_upl_past")
                past_df = pd.DataFrame()
                if past_up is not None:
                    try:
                        if str(past_up.name).lower().endswith(".parquet"):
                            past_df = pd.read_parquet(past_up)
                        else:
                            past_df = pd.read_csv(past_up)
                    except Exception as e:
                        st.error(f"Past dosyası okunamadı: {e}")

                # --- Controls: Dry-run / Force rebind ---
                c_dry, c_force = st.columns(2)
                dry_run = c_dry.checkbox(
    "Dry-run (sadece raporla, kaydetme)",
    value=True,
     help="Önce eşleşme kalitesini gör. Kaydetmez.")
                force_rebind = c_force.checkbox(
    "Force (MANUAL hariç yeniden bağla)",
    value=False,
     help="Mevcut sonuçları (MANUAL olmayan) yeniden bağlamayı dener.")

                if st.button("🔄 Past'tan Sonuçları Çek", key="jl_settle"):
                    if past_df.empty:
                        st.warning("Past dosyası yok.")
                    else:
                        updated = settle_from_past(
    pl, past_df, force=bool(force_rebind))
                        # Summary
                        if "RESULT_MATCH_METHOD" in updated.columns:
                            s = updated["RESULT_MATCH_METHOD"].value_counts(
                                dropna=False)
                            st.info("Eşleşme özeti: " +
     " | ".join([f"{k}: {int(v)}" for k, v in s.items()]))
                        # Show suspicious
                        if "RESULT_MATCH_CONFIDENCE" in updated.columns:
                            conf = pd.to_numeric(
    updated["RESULT_MATCH_CONFIDENCE"],
     errors="coerce").fillna(0.0)
                            bad = updated.loc[conf < 0.80]
                            if not bad.empty:
                                st.warning(
                                    f"Düşük güven / eşleşmeyen: {len(bad)} satır (conf < 0.80)")
                                st.dataframe(
    bad.head(200), use_container_width=True)

                        if dry_run:
                            st.success("Dry-run tamamlandı. (Kaydetmedim)")
                        else:
                            save_pick_log(updated)
                            st.success(
                                "Sonuçlar işlendi ve pick_log güncellendi.")

                # -----------------------------------------------------------
                # 🛟 MANUEL DÜZELT + KİLİTLE (MANUAL OVERRIDE)
                # -----------------------------------------------------------
                st.divider()
                st.markdown("### 🛟 Manuel Düzelt + Kilitle (MANUAL)")
                st.caption(
                    "Eşleşmeyen / şüpheli (conf<0.80) maçları elle skor girerek kilitle. MANUAL olanlar bir daha otomatik overwrite edilmez.")

                pl_cur = load_pick_log()
                if not pl_cur.empty:
                    pl_cur = ensure_match_id(pl_cur)
                    # Aday: sonucu olmayanlar veya düşük güven
                    conf_col = "RESULT_MATCH_CONFIDENCE" if "RESULT_MATCH_CONFIDENCE" in pl_cur.columns else None
                    if conf_col:
                        conf = pd.to_numeric(
    pl_cur[conf_col], errors="coerce").fillna(0.0)
                        cand = pl_cur[(pl_cur.get("Hit").isna(
                        ) if "Hit" in pl_cur.columns else True) | (conf < 0.80)].copy()
                    else:
                        cand = pl_cur[(pl_cur.get("Hit").isna(
                        ) if "Hit" in pl_cur.columns else True)].copy()

                    # Pick_ID yoksa üret (stabil)
                    if "Pick_ID" not in cand.columns:
                        cand["Pick_ID"] = (
    cand.get(
        "Snapshot_Date",
        cand.get(
            "Date",
            pd.Timestamp.now())).astype(str) +
            " | " +
            cand.get(
                "Match_ID",
                "").astype(str) +
                " | " +
                cand.get(
                    "Seçim",
                    "").astype(str) +
                    " | " +
                    cand.get(
                        "Profile",
                         "").astype(str) )

                    if cand.empty:
                        st.success(
                            "Manuel düzeltilecek şüpheli/boş sonuç bulunamadı.")
                    else:
                        # Kullanıcı seçim
                        cand = cand.reset_index(drop=True)
                        opts = cand["Pick_ID"].astype(str).tolist()
                        sel_pid = st.selectbox(
    "Manuel düzeltilecek Pick_ID seç", opts, key="jl_manual_pid")

                        row = cand.loc[cand["Pick_ID"].astype(
                            str) == str(sel_pid)].iloc[0]
                        c0, c1, c2, c3 = st.columns([2, 2,1,1])
                        c0.write(f"**{row.get('HomeTeam', '')} vs {row.get('AwayTeam','')}**")
                        c1.write(f"**{row.get('League', '')}** | {str(row.get('Date',''))[:10]}")
                        c2.write(f"Seçim: **{row.get('Seçim', '')}**")
                        c3.write(f"Odd: **{row.get('Odd', '')}**")

                        # Mevcut skor varsa doldur
                        def _to_int(x, default=0):
                            try:
                                if pd.isna(x):
                                    return default
                                return int(float(x))
                            except Exception:
                                return default

                        ft_h_def = _to_int(row.get("FT_Home", np.nan), 0)
                        ft_a_def = _to_int(row.get("FT_Away", np.nan), 0)

                        cft1, cft2 = st.columns(2)
                        ft_home = cft1.number_input(
    "FT Home",
    min_value=0,
    max_value=25,
    value=int(ft_h_def),
    step=1,
     key="jl_ft_home")
                        ft_away = cft2.number_input(
    "FT Away",
    min_value=0,
    max_value=25,
    value=int(ft_a_def),
    step=1,
     key="jl_ft_away")

                        # Hesaplayıcı
                        def _hit_profit(selection, ft_h, ft_a, odd):
                            sel = str(selection or "").strip().lower()
                            hit = np.nan
                            # 1X2
                            if sel in ["ms1", "ms 1","1","home","ev"] or sel.startswith("ms1"):
                                hit = 1 if ft_h > ft_a else 0
                            elif sel in ["ms2", "ms 2","2","away","dep"] or sel.startswith("ms2"):
                                hit = 1 if ft_a > ft_h else 0
                            elif sel in ["msx", "ms x","x","draw","ber"] or sel.replace(" ","") in ["msx"]:
                                hit = 1 if ft_h == ft_a else 0
                            # OU 2.5
                            elif "2.5" in sel and ("üst" in sel or "ust" in sel or "over" in sel):
                                hit = 1 if (ft_h + ft_a) >= 3 else 0
                            elif "2.5" in sel and ("alt" in sel or "under" in sel):
                                hit = 1 if (ft_h + ft_a) <= 2 else 0
                            # BTTS
                            elif "kg var" in sel or "btts yes" in sel or "btts:yes" in sel or "btts_yes" in sel:
                                hit = 1 if (ft_h > 0 and ft_a > 0) else 0
                            elif "kg yok" in sel or "btts no" in sel or "btts:no" in sel or "btts_no" in sel:
                                hit = 1 if not (ft_h > 0 and ft_a > 0) else 0
                            else:
                                hit = np.nan

                            try:
                                o = float(pd.to_numeric(odd, errors="coerce"))
                            except Exception:
                                o = np.nan
                            profit = np.nan
                            if not pd.isna(hit) and not pd.isna(o):
                                profit = (o - 1.0) if int(hit) == 1 else -1.0
                            return hit, profit

                        preview_hit, preview_profit = _hit_profit(row.get("Seçim", ""), int(ft_home), int(ft_away), row.get("Odd", np.nan))
                        st.info(
    f"Önizleme → Hit: **{preview_hit}** | Profit: **{preview_profit}**")

                        if st.button(
    "🛟 Manuel sonucu kilitle (MANUAL)",
     key="jl_manual_lock"):
                            pl2 = pl_cur.copy()
                            # Pick_ID kolonu yoksa ana tabloda da üret
                            if "Pick_ID" not in pl2.columns:
                                pl2["Pick_ID"] = (
    pl2.get(
        "Snapshot_Date",
        pl2.get(
            "Date",
            pd.Timestamp.now())).astype(str) +
            " | " +
            pl2.get(
                "Match_ID",
                "").astype(str) +
                " | " +
                pl2.get(
                    "Seçim",
                    "").astype(str) +
                    " | " +
                    pl2.get(
                        "Profile",
                         "").astype(str) )

                            mask = pl2["Pick_ID"].astype(str) == str(sel_pid)
                            if mask.sum() == 0:
                                st.error(
                                    "Seçilen Pick_ID pick_log içinde bulunamadı.")
                            else:
                                # Update score + outcomes
                                pl2.loc[mask, "FT_Home"] = int(ft_home)
                                pl2.loc[mask, "FT_Away"] = int(ft_away)
                                pl2.loc[mask, "Total_Goals"] = int(
                                    ft_home) + int(ft_away)

                                # Result labels
                                res_1x2 = "1" if ft_home > ft_away else (
                                    "2" if ft_away > ft_home else "X")
                                pl2.loc[mask, "Result_1X2"] = res_1x2
                                pl2.loc[mask, "OU25_Result"] = "Over" if (
                                    ft_home + ft_away) >= 3 else "Under"
                                pl2.loc[mask, "BTTS_Result"] = "Yes" if (
                                    ft_home > 0 and ft_away > 0) else "No"

                                # Hit/Profit
                                h, pr = _hit_profit(pl2.loc[mask, "Seçim"].iloc[0], int(
                                    ft_home), int(ft_away), pl2.loc[mask, "Odd"].iloc[0])
                                pl2.loc[mask, "Hit"] = h
                                pl2.loc[mask, "Profit"] = pr

                                # Override flags
                                pl2.loc[mask, "RESULT_OVERRIDE"] = True
                                pl2.loc[mask, "RESULT_MATCH_METHOD"] = "MANUAL"
                                pl2.loc[mask, "RESULT_MATCH_CONFIDENCE"] = 1.0
                                pl2.loc[mask,
     "RESULT_SOURCE_ROW_ID"] = "MANUAL"
                                pl2.loc[mask,
     "Settled_At"] = pd.Timestamp.now(tz=None)

                                # Dedup safety
                                pl2 = pl2.loc[:,
     ~pl2.columns.duplicated()].copy()
                                save_pick_log(pl2)
                                st.success(
                                    "MANUAL sonuç kilitlendi ve pick_log güncellendi.")
                                st.rerun()

        # -----------------------------------------------------------
        # 3) PERFORMANS (özet)
        # -----------------------------------------------------------
        with st.expander("3) Performans – ROI / Hit / Market kırılımı", expanded=False):
            pl = load_pick_log()
            if pl.empty:
                st.info("pick_log boş.")
            else:
                pl = ensure_match_id(pl)
                if "Profit" in pl.columns:
                    pl["Profit"] = pd.to_numeric(pl["Profit"], errors="coerce")
                if "Hit" in pl.columns:
                    hit = pl["Hit"].dropna()
                settled = pl[pl.get("Hit").notna()
                                    ] if "Hit" in pl.columns else pl.iloc[0:0]
                st.write(f"Settled pick: **{len(settled)}** / {len(pl)}")
                if not settled.empty:
                    total_profit = float(
    settled["Profit"].sum(
        skipna=True)) if "Profit" in settled.columns else 0.0
                    roi = total_profit / \
                        float(len(settled)) if len(settled) else 0.0
                    hit_rate = float(
    settled["Hit"].mean()) if "Hit" in settled.columns else np.nan
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ROI (1u)", f"{roi:.3f}")
                    c2.metric("Toplam Profit", f"{total_profit:.2f}")
                    c3.metric("Hit Rate", f"{hit_rate:.1%}" if hit_rate == hit_rate else "—")

                    if "Seçim" in settled.columns:
                        settled["Market"] = settled["Seçim"].astype(
                            str).apply(_infer_market_from_selection)
                        g = settled.groupby("Market").agg(
                            n=("Market", "size"),
                            roi=("Profit", lambda x: float(x.sum()) /float(len(x)) if len(x) else 0.0),
                            hit=("Hit", "mean")
                        ).reset_index().sort_values("n", ascending=False)
                        st.dataframe(g, use_container_width=True)
    with tab2:
        st.subheader("🔬 Derin Metrikler (Validation Set)")
        st.divider()

        market_options = [
    'MS 1',
    'MS X',
    'MS 2',
    '2.5 Üst',
    '2.5 Alt',
    'KG Var',
     'KG Yok']
        selected_market = st.selectbox(
    "Analiz Edilecek Pazar Seçiniz:", market_options)
        st.divider()
        st.markdown(
            "**Not:** Bu metrikler modelin *hiç görmediği* son %20'lik dilimdeki (Validasyon seti) performansıdır. Gerçekçi sonuçlardır.")

        if not hist_df.empty:
            def get_metrics(y_true, y_prob, name, threshold=0.5):
                try:
                    ll = log_loss(y_true, y_prob)
                    auc = roc_auc_score(
    y_true, y_prob) if len(
        np.unique(y_true)) > 1 else 0.5
                    brier = brier_score_loss(y_true, y_prob)
                    ece = calculate_ece_clean(y_true, y_prob)
                    sharpness = np.std(y_prob)

                    val_stats = st.session_state.get("market_threshold_stats")
                    threshold = 0.50
                    if val_stats is not None and not val_stats.empty:
                        row = val_stats[val_stats["Pazar"] == name]
                        if not row.empty:
                            val = row["En_iyi_Prob_Eşiği"].iloc[0]
                            if not np.isnan(val):
                                threshold = float(val)

                    if name == "MS X" and threshold == 0.50:
                        threshold = 0.30

                    y_pred = (y_prob > threshold).astype(int)
                    if y_pred.sum() == 0:
                        prec = np.nan
                    else:
                        prec = precision_score(y_true, y_pred, zero_division=0)

                    mcc = matthews_corrcoef(y_true, y_pred)
                    return {
    "Tür": name,
    "Log Loss": ll,
    "AUC": auc,
    "Brier": brier,
    "ECE": ece,
    "Sharpness": sharpness,
    "MCC": mcc,
    "Precision": prec,
     "Ort. Güven": y_prob.mean()}
                except Exception:
                    return {"Tür": name, "Log Loss": np.nan, "AUC": np.nan, "Brier": np.nan, "ECE": np.nan, "Sharpness": np.nan, "MCC": np.nan, "Precision": np.nan, "Ort. Güven": np.nan}

                       # -----------------------------
            # Two metric views:
            # 1) RAW: pre-calibration probabilities (if available)
            # 2) FINAL: calibrated/final probabilities (current behavior)
            # -----------------------------
            def _get_col(
    df_,
    raw_name: str,
    final_name: str,
     fallback: str = None):
                if raw_name in df_.columns:
                    return raw_name
                if final_name in df_.columns:
                    return final_name
                return fallback

            # FINAL metrics (existing)
            metrics_list = []
            metrics_list.append(
    get_metrics(
        (hist_df['Target_Result'] == 2).astype(int),
        hist_df['P_Home_Final'],
         "MS 1"))
            metrics_list.append(
    get_metrics(
        (hist_df['Target_Result'] == 1).astype(int),
        hist_df['P_Draw_Final'],
         "MS X"))
            metrics_list.append(
    get_metrics(
        (hist_df['Target_Result'] == 0).astype(int),
        hist_df['P_Away_Final'],
         "MS 2"))

            p_over_final = _get_col(
    hist_df,
    "P_Over_Raw",
    "P_Over_Final",
     fallback="Prob_Over")
            p_btts_final = _get_col(
    hist_df,
    "P_BTTS_Raw",
    "P_BTTS_Final",
     fallback="Prob_BTTS")

            if 'Target_Over' in hist_df.columns and p_over_final in hist_df.columns:
                metrics_list.append(
    get_metrics(
        hist_df['Target_Over'],
        hist_df[p_over_final],
         "2.5 Üst"))
                metrics_list.append(
    get_metrics(
        1 - hist_df['Target_Over'],
        1 - hist_df[p_over_final],
         "2.5 Alt"))

            if 'Target_BTTS' in hist_df.columns and p_btts_final in hist_df.columns:
                metrics_list.append(
    get_metrics(
        hist_df['Target_BTTS'],
        hist_df[p_btts_final],
         "KG Var"))
                metrics_list.append(
    get_metrics(
        1 - hist_df['Target_BTTS'],
        1 - hist_df[p_btts_final],
         "KG Yok"))

            metrics_df = pd.DataFrame(metrics_list).set_index("Tür")

            # RAW metrics (pre-calibration) – only differs for 1X2 right now
            # unless raw OU/KG cols exist
            metrics_list_raw = []
            p_home_raw = _get_col(hist_df, "P_Home_Raw", "P_Home_Final")
            p_draw_raw = _get_col(hist_df, "P_Draw_Raw", "P_Draw_Final")
            p_away_raw = _get_col(hist_df, "P_Away_Raw", "P_Away_Final")

            metrics_list_raw.append(
    get_metrics(
        (hist_df['Target_Result'] == 2).astype(int),
        hist_df[p_home_raw],
         "MS 1"))
            metrics_list_raw.append(
    get_metrics(
        (hist_df['Target_Result'] == 1).astype(int),
        hist_df[p_draw_raw],
         "MS X"))
            metrics_list_raw.append(
    get_metrics(
        (hist_df['Target_Result'] == 0).astype(int),
        hist_df[p_away_raw],
         "MS 2"))

            p_over_raw = _get_col(
    hist_df,
    "P_Over_Raw",
    "P_Over_Final",
     fallback="Prob_Over")
            p_btts_raw = _get_col(
    hist_df,
    "P_BTTS_Raw",
    "P_BTTS_Final",
     fallback="Prob_BTTS")

            if 'Target_Over' in hist_df.columns and p_over_raw in hist_df.columns:
                metrics_list_raw.append(
    get_metrics(
        hist_df['Target_Over'],
        hist_df[p_over_raw],
         "2.5 Üst"))
                metrics_list_raw.append(
    get_metrics(
        1 - hist_df['Target_Over'],
        1 - hist_df[p_over_raw],
         "2.5 Alt"))

            if 'Target_BTTS' in hist_df.columns and p_btts_raw in hist_df.columns:
                metrics_list_raw.append(
    get_metrics(
        hist_df['Target_BTTS'],
        hist_df[p_btts_raw],
         "KG Var"))
                metrics_list_raw.append(
    get_metrics(
        1 - hist_df['Target_BTTS'],
        1 - hist_df[p_btts_raw],
         "KG Yok"))

            metrics_df_raw = pd.DataFrame(metrics_list_raw).set_index("Tür")

            def style_metrics_table(df: pd.DataFrame):
                auc_bench = {'MS 1': (0.60, 0.65), 'MS X':(0.54, 0.58), 'MS 2':(0.60, 0.65), '2.5 Üst':(0.55, 0.60), '2.5 Alt':(0.55, 0.60), 'KG Var':(0.54, 0.58), 'KG Yok':(0.54, 0.58)}
                logloss_good = df['Log Loss'].quantile(0.3)
                logloss_bad  = df['Log Loss'].quantile(0.8)
                brier_good   = df['Brier'].quantile(0.3)
                brier_bad    = df['Brier'].quantile(0.8)

                def color_cell(v, market, metric):
                    if pd.isna(v):
                        return ''
                    RED, YEL, GREEN = '#b71c1c', '#fbc02d', '#1b5e20'
                    if metric == 'AUC':
                        low, high = auc_bench.get(market, (0.55, 0.60))
                        if v < low:
                            return f'background-color: {RED}; color: white'
                        elif v < high: return f'background-color: {YEL}; color: black'
                        else:
                            return f'background-color: {GREEN}; color: white'
                    if metric == 'Log Loss':
                        if v <= logloss_good:
                            return f'background-color: {GREEN}; color: white'
                        elif v <= logloss_bad: return f'background-color: {YEL}; color: black'
                        else:
                            return f'background-color: {RED}; color: white'
                    if metric == 'Brier':
                        if v <= brier_good:
                            return f'background-color: {GREEN}; color: white'
                        elif v <= brier_bad: return f'background-color: {YEL}; color: black'
                        else:
                            return f'background-color: {RED}; color: white'
                    if metric == 'ECE':
                        if v <= 0.02:
                            return f'background-color: {GREEN}; color: white'
                        elif v <= 0.04: return f'background-color: {YEL}; color: black'
                        else:
                            return f'background-color: {RED}; color: white'
                    if metric in ['MCC', 'Precision']:
                        if v >= 0.15:
                            return f'background-color: {GREEN}; color: white'
                        elif v >= 0.05: return f'background-color: {YEL}; color: black'
                        else:
                            return f'background-color: {RED}; color: white'
                    if metric == 'Sharpness': return 'background-color: #424242; color: white'
                    return ''

                def style_row(row: pd.Series):
                    market = row.name
                    styles = []
                    for metric, v in row.items():
                        styles.append(color_cell(v, market, metric))
                    return styles

                return df.style.format("{:.4f}").apply(style_row, axis=1)

            # Show RAW vs FINAL metrics side-by-side
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.markdown("### 🧪 RAW (pre-calibration) Metrikler")
                st.table(style_metrics_table(metrics_df_raw))
            with c_m2:
                st.markdown("### ✅ FINAL (calibrated) Metrikler")
                st.table(style_metrics_table(metrics_df))
            # --- SAFE prob column resolution for analysis charts (prevents NameError) ---

            # --- SAFE prob column resolution for analysis charts (prevents NameError / KeyError) ---
            def _resolve_first_col(
    _df: pd.DataFrame,
     _cands: List[str]) -> Optional[str]:
                try:
                    for _c in _cands:
                        if _c and (_c in _df.columns):
                            return _c
                except Exception:
                    pass
                return None

            p_home_col = _resolve_first_col(hist_df, ["P_Home_Final", "P_Home","Prob_Home","Prob_MS1"])
            p_draw_col = _resolve_first_col(hist_df, ["P_Draw_Final", "P_Draw","Prob_Draw","Prob_MSX","Prob_X"])
            p_away_col = _resolve_first_col(hist_df, ["P_Away_Final", "P_Away","Prob_Away","Prob_MS2"])

            p_over_col = _resolve_first_col(hist_df, ["P_Over_Final", "Prob_Over","P_Over"])
            p_btts_col = _resolve_first_col(hist_df, ["P_BTTS_Final", "Prob_BTTS","P_BTTS"])

            if selected_market == 'MS 1':
                if ('Target_Result' not in hist_df.columns) or (
                    p_home_col is None):
                    st.warning(
                        "MS 1 analizi için olasılık kolonu bulunamadı (P_Home_Final / Prob_Home).")
                    y_true = pd.Series(dtype=float)
                    y_prob = pd.Series(dtype=float); odd_col = 'ClosingHomeOdd'
                else:
                    y_true = (hist_df['Target_Result'] == 2).astype(int)
                    y_prob = pd.to_numeric(
    hist_df[p_home_col], errors="coerce")
                    odd_col = 'ClosingHomeOdd'
            elif selected_market == 'MS X':
                if ('Target_Result' not in hist_df.columns) or (
                    p_draw_col is None):
                    st.warning(
                        "MS X analizi için olasılık kolonu bulunamadı (P_Draw_Final / Prob_Draw).")
                    y_true = pd.Series(dtype=float)
                    y_prob = pd.Series(dtype=float); odd_col = 'ClosingDrawOdd'
                else:
                    y_true = (hist_df['Target_Result'] == 1).astype(int)
                    y_prob = pd.to_numeric(
    hist_df[p_draw_col], errors="coerce")
                    odd_col = 'ClosingDrawOdd'
            elif selected_market == 'MS 2':
                if ('Target_Result' not in hist_df.columns) or (
                    p_away_col is None):
                    st.warning(
                        "MS 2 analizi için olasılık kolonu bulunamadı (P_Away_Final / Prob_Away).")
                    y_true = pd.Series(dtype=float)
                    y_prob = pd.Series(dtype=float); odd_col = 'ClosingAwayOdd'
                else:
                    y_true = (hist_df['Target_Result'] == 0).astype(int)
                    y_prob = pd.to_numeric(
    hist_df[p_away_col], errors="coerce")
                    odd_col = 'ClosingAwayOdd'
            elif selected_market == '2.5 Üst':
                if (p_over_col is None) or ('Target_Over' not in hist_df.columns) or (
                    p_over_col not in hist_df.columns):
                    st.warning(
                        "OU analizi için olasılık kolonu bulunamadı (P_Over_Final / Prob_Over).")
                    y_true = pd.Series(dtype=float)
                    y_prob = pd.Series(dtype=float); odd_col = 'ClosingO25'
                else:
                    y_true = hist_df['Target_Over'].astype(int)
                    y_prob = pd.to_numeric(
    hist_df[p_over_col], errors="coerce")
                    odd_col = 'ClosingO25'
            elif selected_market == '2.5 Alt':
                if (p_over_col is None) or ('Target_Over' not in hist_df.columns) or (
                    p_over_col not in hist_df.columns):
                    st.warning(
                        "OU analizi için olasılık kolonu bulunamadı (P_Over_Final / Prob_Over).")
                    y_true = pd.Series(dtype=float)
                    y_prob = pd.Series(dtype=float); odd_col = 'ClosingU25'
                else:
                    y_true = 1 - hist_df['Target_Over'].astype(int)
                    y_prob = 1 - \
                        pd.to_numeric(hist_df[p_over_col], errors="coerce")
                    odd_col = 'ClosingU25'
            elif selected_market == 'KG Var':
                if (p_btts_col is None) or ('Target_BTTS' not in hist_df.columns) or (
                    p_btts_col not in hist_df.columns):
                    st.warning(
                        "KG analizi için olasılık kolonu bulunamadı (P_BTTS_Final / Prob_BTTS).")
                    y_true = pd.Series(dtype=float)
                    y_prob = pd.Series(dtype=float); odd_col = 'ClosingBTTSY'
                else:
                    y_true = hist_df['Target_BTTS'].astype(int)
                    y_prob = pd.to_numeric(
    hist_df[p_btts_col], errors="coerce")
                    odd_col = 'ClosingBTTSY'
            else:
                if (p_btts_col is None) or ('Target_BTTS' not in hist_df.columns) or (
                    p_btts_col not in hist_df.columns):
                    st.warning(
                        "KG analizi için olasılık kolonu bulunamadı (P_BTTS_Final / Prob_BTTS).")
                    y_true = pd.Series(dtype=float)
                    y_prob = pd.Series(dtype=float); odd_col = 'ClosingBTTSN'
                else:
                    y_true = 1 - hist_df['Target_BTTS'].astype(int)
                    y_prob = 1 - \
                        pd.to_numeric(hist_df[p_btts_col], errors="coerce")
                    odd_col = 'ClosingBTTSN'

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_calibration_curve_plotly(y_true, y_prob, f"{selected_market} Kalibrasyonu"))
            with c2: st.plotly_chart(plot_gain_chart(y_true, y_prob, f"{selected_market} Gain Chart"))

            ev_df = hist_df.copy()
            ev_df['_target_binary'] = y_true
            ev_df['_prob'] = y_prob.values
            st.plotly_chart(
    plot_ev_analysis(
        ev_df,
        '_prob',
        odd_col,
        '_target_binary',
         f"{selected_market} EV Analizi"))
        else:
            st.info("Analiz için yeterli geçmiş veri yok.")

    with tab3:
        st.subheader("🛡️ Lig Güven Karnesi (Validation Set)")
        if not league_metrics_df.empty:
            cols = [
                "League", "Matches",
                "AUC_MS1", "ROI_MS1_%","Bets_MS1",
                "AUC_MSX", "ROI_MSX_%","Bets_MSX",
                "AUC_MS2", "ROI_MS2_%","Bets_MS2",
                "AUC_O25", "ROI_O25_%","Bets_O25",
                "AUC_U25", "ROI_U25_%","Bets_U25",
                "AUC_KGY", "ROI_KGY_%","Bets_KGY",
                "AUC_KGN", "ROI_KGN_%","Bets_KGN",
                "Best_Market", "Best_Market_AUC","Best_Market_ROI_%",
                "Home_ROI_%", "Home_Bets",
                "Weak_League", "League_Score"
            ]
            valid_cols = [c for c in cols if c in league_metrics_df.columns]

            st.dataframe(
                league_metrics_df[valid_cols],
                use_container_width=True,
                height=600
            )

    with tab4:
        st.subheader("Backtest (Home Bets - Validation Set)")
        if not hist_df.empty and 'P_Home_Final' in hist_df.columns:
            bt_df = hist_df.sort_values('Date').copy()
            bt_df = bt_df.drop_duplicates(
    subset=[
        'Date',
        'HomeTeam',
        'AwayTeam'],
         keep='first')
            valid_odds_mask = (
    bt_df['ClosingHomeOdd'] > 1.0) & (
        bt_df['ClosingHomeOdd'].notna())
            bt_df = bt_df[valid_odds_mask]

            threshold = st.slider("Min Prob (Home)", 0.4, 0.9, 0.5)
            selected_bets = bt_df[bt_df['P_Home_Final'] > threshold].copy()

            if not selected_bets.empty:
                selected_bets['Profit'] = np.where(
    selected_bets['Target_Result'] == 2, selected_bets['ClosingHomeOdd'] - 1, -1)
                selected_bets['CumProfit'] = selected_bets['Profit'].cumsum()

                # --- Safe chart guard (prevents Vega-Lite infinite extent warnings) ---
                _lc = selected_bets[['Date', 'CumProfit']].copy()
                _lc['Date'] = pd.to_datetime(_lc['Date'], errors='coerce')
                _lc['CumProfit'] = pd.to_numeric(
                    _lc['CumProfit'], errors='coerce')
                _lc = _lc[_lc['Date'].notna() & np.isfinite(
                    _lc['CumProfit'].values)]
                if len(_lc) >= 1:
                    _lc = _lc.sort_values('Date').set_index('Date')
                    st.line_chart(_lc['CumProfit'])
                else:
                    st.caption(
                        "📉 Grafik çizilemedi: Date/CumProfit boş veya geçersiz (NaN/∞).")

                roi = selected_bets['Profit'].mean() * 100
                total_profit = selected_bets['Profit'].sum()
                match_count = len(selected_bets)
                win_rate = ((selected_bets['Profit'] > 0).sum() / match_count) * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Toplam Bahis", match_count)
                c2.metric("Toplam Kâr (Birim)", f"{total_profit:.2f}")
                c3.metric("ROI", f"%{roi:.2f}")
                c4.metric("Kazanma Oranı", f"%{win_rate:.1f}")

                with st.expander("Bahis Listesi"):
                    st.dataframe(selected_bets[['Date',
    'HomeTeam',
    'AwayTeam',
    'ClosingHomeOdd',
    'P_Home_Final',
    'Profit',
     'Target_Result']])
            else:
                st.warning("Bu filtreleme kriterlerine uygun (ve oranı geçerli) maç bulunamadı.")
        else: st.info("Backtest için yeterli veri yok.")

        st.markdown("---")
        st.subheader("⏱ Walk-forward Zaman Serisi Testi (Multi-Fold)")

        if st.session_state.get("FAST_MODE", False):
            st.info(
                "⚡ FAST MODE aktifken Walk-Forward analizi devre dışı. Tam tarihsel kalite için Full Mode'a geçin.")
        else:
            if st.button("🚀 Walk-Forward Analizini Başlat"):
                if 'train_core' in st.session_state:
                    with st.spinner("Zaman tünelinde geriye gidiliyor (4-Fold Cross Validation)..."):
                        train_core_data = st.session_state.train_core
                        walk_df = run_walkforward_eval(train_core_data)
                        st.session_state.walkforward_df = walk_df
                else:
                    st.error("Önce ana analizi başlatmalısınız.")

            walk_df = st.session_state.get("walkforward_df", pd.DataFrame())
            if not walk_df.empty:
                st.dataframe(walk_df.style.format(
                    {"Home_AUC": "{:.3f}", "Home_ROI_%": "{:.1f}"}))

    # -----------------------------------------------------------
    # 3b) SONUÇLANAN MAÇLAR (tuttu / tutmadı) — aynı sayfada hızlı kontrol
    # -----------------------------------------------------------
    with st.expander("✅ Sonuçlanan Maçlar (tuttu / tutmadı)", expanded=False):
        pl = load_pick_log()
        if pl.empty:
            st.info("pick_log boş.")
        else:
            pl = ensure_match_id(pl)

            # Snapshot filtresi (Journal ekranındaki date_input key'i)
            snap_day = st.session_state.get("pl_snapshot_day", None)
            # normalize date columns
            for c in ["Snapshot_Date", "Date", "Settled_At"]:
                if c in pl.columns:
                    pl[c] = pd.to_datetime(pl[c], errors="coerce")

            # settled tanımı
            if "Hit" in pl.columns:
                settled = pl[pl["Hit"].notna()].copy()
            elif "Settled_At" in pl.columns:
                settled = pl[pl["Settled_At"].notna()].copy()
            else:
                settled = pl.iloc[0:0].copy()

            if settled.empty:
                st.info("Henüz sonuçlanmış pick yok (Hit/Settled_At boş).")
            else:
                # optional snapshot/day filter
                if snap_day is not None:
                    try:
                        _sd = pd.to_datetime(snap_day).date()
                        if "Snapshot_Date" in settled.columns and settled["Snapshot_Date"].notna(
                        ).any():
                            settled = settled[settled["Snapshot_Date"].dt.date == _sd]
                        elif "Date" in settled.columns and settled["Date"].notna().any():
                            settled = settled[settled["Date"].dt.date == _sd]
                    except Exception:
                        pass

                # özet
                if "Profit" in settled.columns:
                    settled["Profit"] = pd.to_numeric(
                        settled["Profit"], errors="coerce")
                total_profit = float(
    settled["Profit"].sum(
        skipna=True)) if "Profit" in settled.columns else 0.0
                roi = total_profit / \
                    float(len(settled)) if len(settled) else 0.0
                hit_rate = float(settled["Hit"].mean(
                )) if "Hit" in settled.columns and settled["Hit"].notna().any() else np.nan

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sonuçlanan", int(len(settled)))
                c2.metric("Hit Rate", f"{hit_rate*100:.1f}%" if hit_rate == hit_rate else "—")
                c3.metric("Toplam Profit", f"{total_profit:.2f}")
                c4.metric("ROI / pick", f"{roi:.3f}")

                # tablo

                def _hit_icon(v):
                    try:
                        if pd.isna(v):
                            return ""
                    except Exception:
                        pass
                    try:
                        iv = int(float(v))
                    except Exception:
                        return ""
                    return "✅" if iv == 1 else ("❌" if iv == 0 else "")

                if "Hit" in settled.columns:
                    settled["Sonuç"] = settled["Hit"].apply(_hit_icon)
                else:
                    settled["Sonuç"] = ""

                base_cols = ["Sonuç", "Date","League","HomeTeam","AwayTeam","Seçim","Odd","Prob","EV",
                            "FT_Home", "FT_Away","Total_Goals","Hit","Profit",
                            "RESULT_MATCH_METHOD", "RESULT_MATCH_CONFIDENCE","Settled_At",
                            "Profile", "Pick_ID","Match_ID"]
                cols = [c for c in base_cols if c in settled.columns]

                # son görüneni en üste al
                sort_col = "Settled_At" if "Settled_At" in settled.columns else (
                    "Date" if "Date" in settled.columns else None)
                if sort_col:
                    settled = settled.sort_values(
    sort_col, ascending=False, na_position="last")

                st.dataframe(settled[cols].head(200), use_container_width=True)

    with st.expander("Raw Data Check"):
        st.write("Validation Set (hist_df) - Last 20%")
        st.dataframe(hist_df.head())
        st.write("Future Predictions (final_df)")
        st.dataframe(df.head())

        st.markdown("### 🧪 Oran Sağlık Kontrolü")
        check_cols = ['HomeOdd', 'DrawOdd','AwayOdd','O25','U25','BTTSY','BTTSN']
        existing = [c for c in check_cols if c in df.columns]
        if existing:
            bad_rows = df[(df[existing].isna().any(axis=1)) | (df[existing] < 1.10).any(axis=1)][['Date', 'League','HomeTeam','AwayTeam'] + existing].head(50)
            if bad_rows.empty:
                st.success("Şüpheli oran bulunamadı ✅")
            else: st.write("Şüpheli oran satırları:"); st.dataframe(bad_rows)
        else:
            st.info("Oran kolonları bulunamadı, QC yapılamadı.")
