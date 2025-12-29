import streamlit as st

import pandas as pd

import numpy as np

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler

import hashlib

import json

import math
import os
import glob


def _get_base_dir():
    try:
        base = os.path.dirname(__file__)
    except Exception:
        base = ""
    return base if base else os.getcwd()



st.set_page_config(page_title="Futbol Analist: Reborn (KNN Core)", layout="wide")

st.title("Futbol Analist: Reborn (KNN Core)")

st.caption("Strict schema + Open odds standard + weighted global numeric KNN core.")



# -----------------------------------------------------------------------------#

# Helpers

# -----------------------------------------------------------------------------#

def _norm_colname(c: str) -> str:

    return str(c).strip().replace("\ufeff", "")



def _coerce_datetime(s: pd.Series) -> pd.Series:

    return pd.to_datetime(s, errors="coerce", utc=False)



def _to_num(s: pd.Series) -> pd.Series:

    return pd.to_numeric(s, errors="coerce")



def _require_cols(df: pd.DataFrame, cols: list[str], context: str):

    missing = [c for c in cols if c not in df.columns]

    if missing:

        raise ValueError(f"[{context}] Missing required columns: {missing}")



def _nonempty_or_fail(df: pd.DataFrame, cols: list[str], context: str, allow_all_nan: bool = False):

    bad = []

    for c in cols:

        if c not in df.columns:

            bad.append((c, "missing"))

            continue

        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        if (not allow_all_nan) and col.isna().all():

            bad.append((c, "all_nan"))

    if bad:

        raise ValueError(f"[{context}] Required columns are missing/all-NaN: {bad}")



def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:

    b2 = b.replace(0, np.nan)

    return a / b2


# Pick/switch defaults (single source of truth).
ANCHOR_ALPHA_DEFAULT = 1.10
FAIR_CLOSE_BASE_DEFAULT = 0.95
EVID_TH_DEFAULT = 0.03
MIS_BETTER_SOFT_DEFAULT = 0.01
MIS_BETTER_HARD_DEFAULT = 0.02


def _resolve_param(name, default, params=None, debug_context=None):
    # 1) start with default
    val = default
    # 2) params override if present
    if isinstance(params, dict) and name in params:
        val = params.get(name, default)
    # 3) debug_context nested override (highest priority) if present
    if isinstance(debug_context, dict):
        cp = debug_context.get("chosen_params") or {}
        wc = (cp.get("weight_config") or cp.get("weights") or cp.get("params") or {})
        if isinstance(wc, dict) and name in wc:
            val = wc.get(name, val)
        if name in debug_context:
            val = debug_context.get(name, val)
    # 4) cast to finite float
    try:
        f = float(val)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return float(default)


def build_decision_trace(row, debug_context=None):
    def _fnum(v, default=np.nan):
        try:
            f = float(v)
            return f if np.isfinite(f) else default
        except Exception:
            return default

    def _get(obj, key, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return obj.get(key, default) if hasattr(obj, "get") else default
        except Exception:
            return default

    params = {}
    if isinstance(debug_context, dict):
        params = debug_context.get("chosen_params", {}) or {}

    anchor_alpha = _resolve_param("ANCHOR_ALPHA", ANCHOR_ALPHA_DEFAULT, params, debug_context)
    fair_close_base = _resolve_param("FAIR_CLOSE_BASE", FAIR_CLOSE_BASE_DEFAULT, params, debug_context)
    evid_th = _resolve_param("EVID_TH", EVID_TH_DEFAULT, params, debug_context)
    mis_soft = _resolve_param("MIS_BETTER_SOFT", MIS_BETTER_SOFT_DEFAULT, params, debug_context)
    mis_hard = _resolve_param("MIS_BETTER_HARD", MIS_BETTER_HARD_DEFAULT, params, debug_context)

    match_id = _get(row, "Match_ID", "")
    league = _get(row, "League", "")
    home = _get(row, "HomeTeam", "")
    away = _get(row, "AwayTeam", "")

    base_pick = _get(row, "PICK_BASE", "")
    final_pick = _get(row, "PICK_FINAL", _get(row, "PICK", ""))
    switched = int(_fnum(_get(row, "PICK_SWITCHED", 0), 0) or 0)
    switch_reason = _get(row, "SWITCH_REASON", "")

    pick_prob = _fnum(_get(row, "PICK_PROB", np.nan))
    pick_prob_eff = _fnum(_get(row, "PICK_PROB_EFF", np.nan))
    pick_prob_adj = _fnum(_get(row, "PICK_PROB_ADJ", np.nan))
    sel_p_adj = _fnum(_get(row, "SEL_P_ADJ", np.nan))
    sel_alpha = _fnum(_get(row, "SEL_ALPHA", np.nan))
    sel_sim = _fnum(_get(row, "SEL_SIM", np.nan))
    sel_score = _fnum(_get(row, "SEL_SCORE", np.nan))
    sel_mispricing = _fnum(_get(row, "SEL_MISPRICING", np.nan))
    sel_mis_bonus = _fnum(_get(row, "SEL_MIS_BONUS", np.nan))

    pick_score_total = _fnum(_get(row, "PICK_SCORE_TOTAL", np.nan))
    pick_mis_bonus = _fnum(_get(row, "PICK_MIS_BONUS", np.nan))
    pick_sim_bonus = _fnum(_get(row, "PICK_SIM_BONUS", np.nan))
    market_weight = _fnum(_get(row, "MARKET_WEIGHT", np.nan))

    sim_quality = _fnum(_get(row, "SIM_QUALITY", np.nan))
    effective_n = _fnum(_get(row, "EFFECTIVE_N", np.nan))
    conf = _fnum(_get(row, "CONF", np.nan))
    trust_pct = _fnum(_get(row, "TRUST_PCT", np.nan))
    best_of_rank = _fnum(_get(row, "BestOfRank", np.nan))

    xg_weight = _fnum(_get(row, "XG_WEIGHT_CFG", _get(row, "XG_BIAS_W", np.nan)))
    xg_micro = {
        "home": _fnum(_get(row, "XG_MICRO_BIAS_HOME", np.nan)),
        "draw": _fnum(_get(row, "XG_MICRO_BIAS_DRAW", np.nan)),
        "away": _fnum(_get(row, "XG_MICRO_BIAS_AWAY", np.nan)),
        "over25": _fnum(_get(row, "XG_MICRO_BIAS_OVER25", np.nan)),
        "btts": _fnum(_get(row, "XG_MICRO_BIAS_BTTS", np.nan)),
    }

    cands_reason = _get(row, "CANDS_REASON", "")
    has_cands = int(_fnum(_get(row, "HAS_CANDS", 0), 0) or 0)
    found_final = int(_fnum(_get(row, "FOUND_FINAL", 0), 0) or 0)

    bonus_total = _fnum(pick_mis_bonus, 0.0) + _fnum(pick_sim_bonus, 0.0)
    score_est = np.nan
    if np.isfinite(pick_prob_adj) and np.isfinite(market_weight):
        score_est = pick_prob_adj * (1.0 + bonus_total) * market_weight

    breakdown = {
        "base_prob_adj": pick_prob_adj,
        "mispricing_component": pick_prob_adj * _fnum(pick_mis_bonus, 0.0) if np.isfinite(pick_prob_adj) else np.nan,
        "sim_bonus_component": pick_prob_adj * _fnum(pick_sim_bonus, 0.0) if np.isfinite(pick_prob_adj) else np.nan,
        "market_weight_component": (pick_prob_adj * (1.0 + bonus_total) * (market_weight - 1.0))
        if np.isfinite(pick_prob_adj) and np.isfinite(market_weight)
        else np.nan,
        "xg_micro_component": xg_weight * sum([v for v in xg_micro.values() if np.isfinite(v)])
        if np.isfinite(xg_weight)
        else np.nan,
        "fair_override": 1.0 if str(switch_reason).startswith("FAIR_OVERRIDE_") else 0.0,
        "note": "components are proxy deltas from exported fields",
    }

    weights_resolved = {
        "ANCHOR_ALPHA": anchor_alpha,
        "FAIR_CLOSE_BASE": fair_close_base,
        "EVID_TH": evid_th,
        "MIS_BETTER_SOFT": mis_soft,
        "MIS_BETTER_HARD": mis_hard,
        "XG_WEIGHT_CFG": xg_weight,
    }

    params_snapshot = {}
    if isinstance(params, dict):
        for k in ["ANCHOR_ALPHA", "FAIR_CLOSE_BASE", "EVID_TH", "MIS_BETTER_SOFT", "MIS_BETTER_HARD"]:
            if k in params:
                params_snapshot[k] = params.get(k)
        wc = params.get("weight_config") if isinstance(params.get("weight_config"), dict) else {}
        if wc:
            params_snapshot["weight_config"] = {k: wc.get(k) for k in wc}

    trace = {
        "inputs": {
            "match_id": match_id,
            "league": league,
            "home": home,
            "away": away,
        },
        "decision": {
            "pick_base": base_pick,
            "pick_final": final_pick,
            "switched": switched,
            "switch_reason": switch_reason,
            "has_cands": has_cands,
            "found_final": found_final,
            "cands_reason": cands_reason,
        },
        "probabilities": {
            "pick_prob": pick_prob,
            "pick_prob_eff": pick_prob_eff,
            "pick_prob_adj": pick_prob_adj,
            "sel_p_adj": sel_p_adj,
        },
        "components": {
            "sel_alpha": sel_alpha,
            "sel_sim": sel_sim,
            "sel_score": sel_score,
            "sel_mispricing": sel_mispricing,
            "sel_mis_bonus": sel_mis_bonus,
            "pick_mis_bonus": pick_mis_bonus,
            "pick_sim_bonus": pick_sim_bonus,
            "market_weight": market_weight,
            "score_total": pick_score_total,
            "score_est": score_est,
        },
        "breakdown": breakdown,
        "quality": {
            "sim_quality": sim_quality,
            "effective_n": effective_n,
            "conf": conf,
            "trust_pct": trust_pct,
            "best_of_rank": best_of_rank,
        },
        "xg": {
            "xg_weight_cfg": xg_weight,
            "xg_micro": xg_micro,
        },
        "weights": {
            "resolved": weights_resolved,
            "params_snapshot": params_snapshot,
        },
    }

    lines = [
        f"Match: {home} vs {away} | {league} | {match_id}",
        f"PICK_BASE={base_pick} -> PICK_FINAL={final_pick} | switched={switched} reason={switch_reason}",
        f"Prob: base={pick_prob:.3f} eff={pick_prob_eff:.3f} adj={pick_prob_adj:.3f} sel_adj={sel_p_adj:.3f}",
        f"Components: alpha={sel_alpha:.3f} sim={sel_sim:.3f} mispricing={sel_mispricing:.3f} mis_bonus={sel_mis_bonus:.3f}",
        f"Score: sel_score={sel_score:.3f} total={pick_score_total:.3f} approx={score_est:.3f} mw={market_weight:.3f}",
        f"Quality: SIM_QUALITY={sim_quality:.3f} EFFECTIVE_N={effective_n:.2f} CONF={conf:.2f} TRUST_PCT={trust_pct:.1f} BestOfRank={best_of_rank:.1f}",
        f"Cands: has={has_cands} found={found_final} reason={cands_reason}",
        "Breakdown (proxy deltas): "
        f"base={breakdown['base_prob_adj']} "
        f"+mis={breakdown['mispricing_component']} "
        f"+sim={breakdown['sim_bonus_component']} "
        f"+mw={breakdown['market_weight_component']}",
    ]
    trace_text = "\n".join(lines)
    return trace, trace_text


_PICK_SWITCH_AUDIT_ROWS = []


def _audit_pick_row(row, best, candidates_sorted, topn=6):
    def _g(obj, key, default=np.nan):
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return obj.get(key, default) if hasattr(obj, "get") else default
        except Exception:
            return default

    try:
        match_id = _g(row, "Match_ID", "")
        if not match_id:
            match_id = f"{_g(row, 'Date', '')}|{_g(row, 'League', '')}|{_g(row, 'HomeTeam', '')}|{_g(row, 'AwayTeam', '')}"
    except Exception:
        match_id = ""

    base_market = _g(best, "base_market", "")
    final_market = _g(best, "final_market", _g(best, "market", ""))
    switched = int(base_market != final_market)

    out = {
        "Match_ID": match_id,
        "Date": _g(row, "Date", np.nan),
        "League": _g(row, "League", ""),
        "HomeTeam": _g(row, "HomeTeam", ""),
        "AwayTeam": _g(row, "AwayTeam", ""),
        "PICK_BASE": base_market,
        "PICK_FINAL": final_market,
        "PICK_SWITCHED": switched,
        "SWITCH_REASON": _g(best, "switch_reason", ""),
        "BASE_SCORE": _g(best, "base_score", np.nan),
        "FINAL_SCORE": _g(best, "score_total", _g(best, "score", np.nan)),
        "TRUST_PCT": _g(row, "TRUST_PCT", np.nan),
        "SIM_QUALITY": _g(row, "SIM_QUALITY", np.nan),
        "EFFECTIVE_N": _g(row, "EFFECTIVE_N", np.nan),
        "ANCHOR_ALPHA": _g(row, "ANCHOR_ALPHA", np.nan),
    }

    summary_parts = []
    for i, cand in enumerate(candidates_sorted[:topn], start=1):
        out[f"TOP{i}_MKT"] = _g(cand, "market", "")
        out[f"TOP{i}_SCORE"] = _g(cand, "score", np.nan)
        out[f"TOP{i}_PROB"] = _g(cand, "prob", np.nan)
        out[f"TOP{i}_PROB_EFF"] = _g(cand, "prob_eff", np.nan)
        out[f"TOP{i}_PROB_ADJ"] = _g(cand, "prob_adj", np.nan)
        out[f"TOP{i}_ALPHA"] = _g(cand, "alpha", np.nan)
        out[f"TOP{i}_ODD"] = _g(cand, "odd", np.nan)
        out[f"TOP{i}_MIS"] = _g(cand, "mis_bonus", _g(cand, "mispricing", np.nan))
        out[f"TOP{i}_SIM"] = _g(cand, "sim_bonus", np.nan)

        summary_parts.append(
            f"{out[f'TOP{i}_MKT']} s={_g(cand,'score',np.nan)} p={_g(cand,'prob',np.nan)} "
            f"pe={_g(cand,'prob_eff',np.nan)} pa={_g(cand,'prob_adj',np.nan)} "
            f"a={_g(cand,'alpha',np.nan)} o={_g(cand,'odd',np.nan)}"
        )

    out["TOPN_SUMMARY"] = " | ".join(summary_parts)

    _PICK_SWITCH_AUDIT_ROWS.append(out)



def _sha1_bytes(b: bytes) -> str:

    return hashlib.sha1(b).hexdigest()



def _stable_params_key(params: dict) -> str:

    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()



def _form_metrics(value):

    if pd.isna(value):

        return (0.0, 0.0, 0.0, 0.0)

    text = str(value).upper()

    tokens = [ch for ch in text if ch in {"W", "D", "L"}]

    total = len(tokens)

    if total == 0:

        return (0.0, 0.0, 0.0, 0.0)

    wins = tokens.count("W")

    draws = tokens.count("D")

    losses = tokens.count("L")

    points = wins * 3 + draws

    return (float(points), wins / total, draws / total, losses / total)



def _clamp(value: float, min_val: float, max_val: float) -> float:

    return max(min(value, max_val), min_val)



def clamp(x, lo=0.0, hi=1.0):
    try:
        x = float(x)
    except Exception:
        return lo
    return lo if x < lo else hi if x > hi else x

def _get_total_goals_series(df: pd.DataFrame):
    """
    Guvenli TG uretimi:
    - FT_Score_Home/FT_Score_Away varsa onu kullan
    - FTHG/FTAG varsa onu kullan
    - TG zaten varsa onu kullan
    - Hicbiri yoksa None don
    """
    if df is None or df.empty:
        return None
    cols = set(df.columns)
    if "TG" in cols:
        return pd.to_numeric(df["TG"], errors="coerce")
    if "FT_Score_Home" in cols and "FT_Score_Away" in cols:
        return pd.to_numeric(df["FT_Score_Home"], errors="coerce") + pd.to_numeric(df["FT_Score_Away"], errors="coerce")
    if "FTHG" in cols and "FTAG" in cols:
        return pd.to_numeric(df["FTHG"], errors="coerce") + pd.to_numeric(df["FTAG"], errors="coerce")
    return None

def compute_league_profile(df, target_league=None, W_short=10, W_long=50):
    league_filter_fallback = False
    if df is None or df.empty:
        return {
            "league": "",
            "league_regime": "n/a",
            "league_regime_score": 0.0,
            "goals_avg_10": np.nan,
            "goals_avg_50": np.nan,
            "n_long": 0,
            "n_short": 0,
            "conf": 0.0,
            "league_filter_fallback": league_filter_fallback,
        }

    df2 = df.copy()
    if target_league and "League" in df2.columns:
        filt = df2[df2["League"] == target_league].copy()
        min_rows = max(8, W_short // 2)
        if len(filt) >= min_rows:
            df2 = filt
        else:
            league_filter_fallback = True

    tg = _get_total_goals_series(df2)
    if tg is None:
        league = target_league or (df2["League"].iloc[0] if "League" in df2.columns else "")
        return {
            "league": league,
            "league_regime": "n/a",
            "league_regime_score": 0.0,
            "goals_avg_10": np.nan,
            "goals_avg_50": np.nan,
            "n_long": 0,
            "n_short": 0,
            "conf": 0.0,
            "league_filter_fallback": league_filter_fallback,
        }

    df2["TG"] = tg
    league = target_league or (df2["League"].iloc[0] if "League" in df2.columns else "")
    sub = df2.tail(W_long)
    sub2 = df2.tail(W_short)
    goals_avg_50 = sub["TG"].mean()
    goals_avg_10 = sub2["TG"].mean()

    n_long = len(sub)
    n_short = len(sub2)
    conf = min(1, n_short / W_short) * min(1, n_long / W_long)

    trend = goals_avg_10 - goals_avg_50
    strength = abs(trend)
    raw = (strength * conf * 5.0)
    raw = raw if trend > 0 else -raw
    score = max(-1.0, min(1.0, raw / 2.0))
    if not np.isfinite(score):
        score = 0.0

    if score == 0.0 and not np.isfinite(trend):
        regime = "n/a"
    else:
        regime = "NORMAL"
        if score > 0.3:
            regime = "HIGH-TEMPO"
        elif score < -0.3:
            regime = "LOW-TEMPO"

    return {
        "league": league,
        "league_regime": regime,
        "league_regime_score": score,
        "goals_avg_10": goals_avg_10,
        "goals_avg_50": goals_avg_50,
        "n_long": n_long,
        "n_short": n_short,
        "conf": conf,
        "league_filter_fallback": league_filter_fallback,
    }


def compute_team_trend(df, team, W_short=10, W_long=50):
    def _trend_from_series(series, metric_used, index=None):
        if series is None:
            return {
                "metric_used": "n/a",
                "n_matches": 0,
                "n_short": 0,
                "n_long": 0,
                "short_mean": np.nan,
                "long_mean": np.nan,
                "conf": 0.0,
                "trend_score": 0.0,
                "trend_reason": "metric_n/a",
            }
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=index)
        s = pd.to_numeric(series, errors="coerce").dropna()
        n_matches = int(len(s))
        if n_matches == 0:
            return {
                "metric_used": "n/a",
                "n_matches": 0,
                "n_short": 0,
                "n_long": 0,
                "short_mean": np.nan,
                "long_mean": np.nan,
                "conf": 0.0,
                "trend_score": 0.0,
                "trend_reason": "metric_n/a",
            }
        if n_matches <= 3:
            return {
                "metric_used": metric_used,
                "n_matches": n_matches,
                "n_short": 0,
                "n_long": 0,
                "short_mean": np.nan,
                "long_mean": np.nan,
                "conf": 0.0,
                "trend_score": 0.0,
                "trend_reason": "N_low",
            }

        if n_matches <= W_short:
            short_n = max(3, int(np.ceil(n_matches * 0.5)))
            long_n = n_matches
            s_short = s.tail(short_n)
            s_long = s.tail(long_n)
        else:
            s_short = s.tail(W_short)
            s_long = s.tail(min(W_long, n_matches))
        n_short = int(len(s_short))
        n_long = int(len(s_long))
        short_mean = s_short.mean()
        long_mean = s_long.mean()
        conf = min(1, n_short / W_short) * min(1, n_long / W_long)
        trend = short_mean - long_mean
        strength = abs(trend)
        raw = (strength * conf * 5.0)
        raw = raw if trend > 0 else -raw
        score = max(-1.0, min(1.0, raw / 2.0))
        if not np.isfinite(score) or abs(score) < 1e-9:
            score = 0.0
        if abs(trend) < 1e-12:
            trend_reason = "short_eq_long"
        else:
            trend_reason = "ok"

        return {
            "metric_used": metric_used,
            "n_matches": n_matches,
            "n_short": n_short,
            "n_long": n_long,
            "short_mean": short_mean,
            "long_mean": long_mean,
            "conf": conf,
            "trend_score": score,
            "trend_reason": trend_reason,
        }

    if df is None or df.empty or not team:
        base = {
            "team": team,
            "metric_used": "n/a",
            "n_matches": 0,
            "n_short": 0,
            "n_long": 0,
            "short_mean": np.nan,
            "long_mean": np.nan,
            "conf": 0.0,
            "trend_score": 0.0,
            "xg_metric_used": "n/a",
            "xg_short_mean": np.nan,
            "xg_long_mean": np.nan,
            "xg_conf": 0.0,
            "xg_trend_score": 0.0,
        }
        return base

    df2 = df.copy()
    has_home = "HomeTeam" in df2.columns
    has_away = "AwayTeam" in df2.columns
    if not (has_home or has_away):
        return {
            "team": team,
            "metric_used": "n/a",
            "n_matches": 0,
            "n_short": 0,
            "n_long": 0,
            "short_mean": np.nan,
            "long_mean": np.nan,
            "conf": 0.0,
            "trend_score": 0.0,
            "xg_metric_used": "n/a",
            "xg_short_mean": np.nan,
            "xg_long_mean": np.nan,
            "xg_conf": 0.0,
            "xg_trend_score": 0.0,
        }

    m = None
    if has_home:
        m = (df2["HomeTeam"] == team)
    if has_away:
        m2 = (df2["AwayTeam"] == team)
        m = (m2 if m is None else (m | m2))
    team_df = df2[m].copy() if m is not None else pd.DataFrame()
    if "Date" in team_df.columns:
        team_df["Date"] = pd.to_datetime(team_df["Date"], errors="coerce")
        team_df = team_df.sort_values("Date", kind="mergesort")
        team_df = team_df.reset_index(drop=True)
    if team_df.empty:
        return {
            "team": team,
            "metric_used": "n/a",
            "n_matches": 0,
            "n_short": 0,
            "n_long": 0,
            "short_mean": np.nan,
            "long_mean": np.nan,
            "conf": 0.0,
            "trend_score": 0.0,
            "xg_metric_used": "n/a",
            "xg_short_mean": np.nan,
            "xg_long_mean": np.nan,
            "xg_conf": 0.0,
            "xg_trend_score": 0.0,
        }

    gf_series = None
    if {"FTHG", "FTAG"}.issubset(team_df.columns):
        gf_series = np.where(
            team_df["HomeTeam"] == team,
            team_df["FTHG"],
            team_df["FTAG"],
        )
    elif {"FT_Score_Home", "FT_Score_Away"}.issubset(team_df.columns):
        gf_series = np.where(
            team_df["HomeTeam"] == team,
            team_df["FT_Score_Home"],
            team_df["FT_Score_Away"],
        )

    gf_trend = _trend_from_series(gf_series, "GF", index=team_df.index)

    xg_series = None
    if {"xG_Home", "xG_Away"}.issubset(team_df.columns):
        xg_series = np.where(
            team_df["HomeTeam"] == team,
            team_df["xG_Home"],
            team_df["xG_Away"],
        )
    xg_trend = _trend_from_series(xg_series, "xG_for", index=team_df.index) if xg_series is not None else {
        "metric_used": "n/a",
        "n_matches": 0,
        "n_short": 0,
        "n_long": 0,
        "short_mean": np.nan,
        "long_mean": np.nan,
        "conf": 0.0,
        "trend_score": 0.0,
        "trend_reason": "metric_n/a",
    }

    return {
        "team": team,
        "metric_used": gf_trend["metric_used"],
        "n_matches": gf_trend["n_matches"],
        "n_short": gf_trend["n_short"],
        "n_long": gf_trend["n_long"],
        "short_mean": gf_trend["short_mean"],
        "long_mean": gf_trend["long_mean"],
        "conf": gf_trend["conf"],
        "trend_score": gf_trend["trend_score"],
        "trend_reason": gf_trend.get("trend_reason", "metric_n/a"),
        "xg_metric_used": xg_trend["metric_used"],
        "xg_short_mean": xg_trend["short_mean"],
        "xg_long_mean": xg_trend["long_mean"],
        "xg_conf": xg_trend["conf"],
        "xg_trend_score": xg_trend["trend_score"],
        "xg_trend_reason": xg_trend.get("trend_reason", "metric_n/a"),
    }


def regime_bias_for_top3(league_score, home_team_score, away_team_score, PICK_MARGIN):
    if not (np.isfinite(PICK_MARGIN) and PICK_MARGIN < 3.0):
        return 0.0
    raw = 0.6 * league_score + 0.2 * home_team_score + 0.2 * away_team_score
    raw = max(-1, min(1, raw))
    bias = raw * 2.0
    return max(-2, min(2, bias))

def parse_form_string(s: str):
    if not isinstance(s, str) or len(s.strip()) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    s = s.strip().upper()

    n = len(s)

    w = s.count("W")

    d = s.count("D")

    l = s.count("L")

    points = (3 * w + d) / (3 * n) if n > 0 else 0.0

    return points, (w / n if n > 0 else 0.0), (d / n if n > 0 else 0.0), (l / n if n > 0 else 0.0), n



def market_confidence(odds_moves_raw: str):

    """

    Odds_Moves_Raw example: "(5.0>6.1),(3.9>3.4)"

    Parse all pairs and return avg (open-close)/open.

    Positive -> odds dropped -> money in.

    """

    if not isinstance(odds_moves_raw, str) or ">" not in odds_moves_raw:

        return 0.0

    parts = [p.strip() for p in odds_moves_raw.split(",") if ">" in p]

    vals = []

    for p in parts:

        p = p.replace("(", "").replace(")", "")

        try:

            o, c = p.split(">")

            o, c = float(o), float(c)

            if o > 0:

                vals.append((o - c) / o)

        except Exception:

            continue

    if not vals:

        return 0.0

    return float(clamp(sum(vals) / len(vals), -0.30, 0.30))



def manager_penalty(games):

    try:

        n = float(games)

    except Exception:

        return 0.0

    if n < 3:

        return -8.0

    if n < 5:

        return -5.0

    if n < 8:

        return -3.0

    return 0.0



def xg_micro_bias(xg_home, xg_away):

    """

    Returns small biases for (home, draw, away, over25, btts_yes).

    Biases are in probability points (e.g. +0.03).

    Clamped to stay small.

    """

    try:

        xh = float(xg_home)

        xa = float(xg_away)

    except Exception:

        return 0.0, 0.0, 0.0, 0.0, 0.0



    if not np.isfinite(xh) or not np.isfinite(xa):

        return 0.0, 0.0, 0.0, 0.0, 0.0



    xt = xh + xa

    xdiff = xh - xa



    b_home = float(clamp(xdiff * 0.03, -0.03, 0.03))

    b_away = -b_home

    b_draw = float(clamp(-abs(xdiff) * 0.02, -0.02, 0.0))



    b_over = float(clamp((xt - 2.6) * 0.04, -0.04, 0.04))

    b_btts = float(clamp((min(xh, xa) - 0.9) * 0.05, -0.04, 0.04))



    return b_home, b_draw, b_away, b_over, b_btts



def _safe_float(v):

    try:

        f = float(v)

        if not np.isfinite(f):

            return np.nan

        return f

    except Exception:

        return np.nan



def _weighted_xg_residual(sim_df: pd.DataFrame, w_power: float = 2.0, clip: float = 1.5):

    """

    Compute weighted residuals:

      rH = FTHG - xG_Home

      rA = FTAG - xG_Away

    Weights from Similarity_Score^w_power, normalized.

    Returns: mu_rh, mu_ra, eff_n

    """

    if sim_df is None or getattr(sim_df, "empty", True):

        return 0.0, 0.0, 0.0



    need = {"FTHG", "FTAG", "xG_Home", "xG_Away"}

    if not need.issubset(set(sim_df.columns)):

        return 0.0, 0.0, 0.0



    df = sim_df.copy()

    df["xG_Home"] = pd.to_numeric(df["xG_Home"], errors="coerce")

    df["xG_Away"] = pd.to_numeric(df["xG_Away"], errors="coerce")

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")

    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")



    df = df.dropna(subset=["xG_Home", "xG_Away", "FTHG", "FTAG"])

    if df.empty:

        return 0.0, 0.0, 0.0



    rH = (df["FTHG"] - df["xG_Home"]).clip(-clip, clip)

    rA = (df["FTAG"] - df["xG_Away"]).clip(-clip, clip)



    if "Similarity_Score" in df.columns:

        w = pd.to_numeric(df["Similarity_Score"], errors="coerce").fillna(0.0).clip(lower=0.0)

        w = np.power(w.values, w_power)

    else:

        w = np.ones(len(df), dtype=float)



    w_sum = float(np.sum(w))

    if w_sum <= 0:

        w = np.ones(len(df), dtype=float)

        w_sum = float(np.sum(w))



    w = w / w_sum



    mu_rh = float(np.sum(w * rH.values))

    mu_ra = float(np.sum(w * rA.values))

    eff_n = float(1.0 / max(1e-9, np.sum(w * w)))



    return mu_rh, mu_ra, eff_n



def _poisson_pmf(k: int, lam: float) -> float:

    try:

        if lam <= 0:

            return 0.0

        return float(np.exp(-lam) * (lam ** k) / math.factorial(k))

    except Exception:

        return 0.0



def score_expectation_from_xg(xh, xa, mu_rh=0.0, mu_ra=0.0, k_resid=0.35, max_goals=5):

    """

    Build adjusted lambdas:

      lamH = clamp(xh + k_resid*mu_rh, 0.1, 3.5)

      lamA = clamp(xa + k_resid*mu_ra, 0.1, 3.5)

    Compute joint Poisson distribution (independent) for 0..max_goals.

    Return: lamH, lamA, mode_score_str, top3_str

    """

    xh_f = _safe_float(xh)

    xa_f = _safe_float(xa)

    if not np.isfinite(xh_f) or not np.isfinite(xa_f):

        return np.nan, np.nan, "", ""



    lamH = float(clamp(xh_f + k_resid * float(mu_rh), 0.1, 3.5))

    lamA = float(clamp(xa_f + k_resid * float(mu_ra), 0.1, 3.5))



    probs = {}

    total = 0.0

    for h in range(0, max_goals + 1):

        ph = _poisson_pmf(h, lamH)

        for a in range(0, max_goals + 1):

            pa = _poisson_pmf(a, lamA)

            p = ph * pa

            probs[(h, a)] = p

            total += p



    if total > 0:

        for k in list(probs.keys()):

            probs[k] = probs[k] / total



    (mh, ma), _ = max(probs.items(), key=lambda kv: kv[1])

    mode = f"{mh}-{ma}"



    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    parts = [f"{h}-{a}({p*100:.0f}%)" for (h, a), p in top3]

    top3_str = " · ".join(parts)



    return lamH, lamA, mode, top3_str



def _pct(v):

    try:

        x = float(v)

        if not np.isfinite(x):

            return np.nan

        return x * 100.0 if 0.0 <= x <= 1.0 else x

    except Exception:

        return np.nan



def _fmt1(x):

    try:

        return f"{float(x):.1f}"

    except Exception:

        return "NA"



def _pick_first(stats: dict, keys: list, default=np.nan):

    for k in keys:

        if k in stats and stats.get(k, None) is not None:

            return stats.get(k)

    return default



def select_scenario(stats: dict):

    pH = _pct(_pick_first(stats, ["Sim_Home%", "Sim_Home_Win", "Home_%","HOME_PCT","EV_%","EVP","p_home","HOME"], np.nan))

    pD = _pct(_pick_first(stats, ["Sim_Draw%", "Sim_Draw", "Draw_%","DRAW_PCT","Berabere_%","p_draw","DRAW"], np.nan))

    pA = _pct(_pick_first(stats, ["Sim_Away%", "Sim_Away_Win", "Away_%","AWAY_PCT","Deplasman_%","p_away","AWAY"], np.nan))



    pOver = _pct(_pick_first(stats, ["Sim_Over25%", "Sim_Over25", "Over25_%","OVER25_PCT","Ust 2.5 %","?st 2.5 %","p_over25","OVER25"], np.nan))

    pUnder = _pct(_pick_first(stats, ["Sim_Under25%", "Sim_Under25", "Under25_%","UNDER25_PCT","Alt 2.5 %","p_under25","UNDER25"], np.nan))



    pBttsY = _pct(_pick_first(stats, ["Sim_BTTS_Yes%", "Sim_BTTS_Yes", "BTTS_Yes_%","BTTS_YES_PCT","BTTS Evet %","p_btts_yes","BTTS_YES"], np.nan))

    pBttsN = _pct(_pick_first(stats, ["Sim_BTTS_No%", "Sim_BTTS_No", "BTTS_No_%","BTTS_NO_PCT","BTTS Hayir %","BTTS Hay?r %","p_btts_no","BTTS_NO"], np.nan))



    trust = float(_pick_first(stats, ["TRUST_PCT","Trust","trust_pct","Guven","G?ven"], np.nan)) if stats is not None else np.nan



    score_mode = str(stats.get("SCORE_MODE","")) if stats else ""

    score_top3 = str(stats.get("SCORE_TOP3","")) if stats else ""

    score_exp = ""

    try:

        lh = stats.get("SCORE_LAM_H", np.nan)

        la = stats.get("SCORE_LAM_A", np.nan)

        if np.isfinite(float(lh)) and np.isfinite(float(la)):

            score_exp = f"{float(lh):.2f}-{float(la):.2f}"

    except Exception:

        score_exp = ""



    if np.isfinite(trust):

        if trust < 55:

            pre = "Guven dusuk: Veriler karisik. "

        elif trust < 70:

            pre = "Orta guven: Eglim var ama kesin degil. "

        else:

            pre = "Guven yuksek: Sinyal netlesiyor. "

    else:

        pre = ""



    ev = f"Ev:{_fmt1(pH)}% • Ber:{_fmt1(pD)}% • Dep:{_fmt1(pA)}% | Ust:{_fmt1(pOver)}% • Alt:{_fmt1(pUnder)}% | KGV:{_fmt1(pBttsY)}% • KGY:{_fmt1(pBttsN)}% | TRUST:{_fmt1(trust)}"

    if score_mode or score_top3:

        ev += f" | Skor:{score_mode} • {score_top3}"



    def strength(val):

        try:

            return float(val)

        except Exception:

            return 0.0



    rules = []



    rules.append(("EDGE_ANTI_DRAW","EDGE","Ya Hep Ya Hic (Anti-Beraberlik)",

                 (np.isfinite(pD) and pD < 20),

                 strength(20 - pD),

                 "Bu macta beraberlik beklemek buyuk hata olur. Iki takimin oyun yapisi 'kazan ya da kaybet' uzerine kurulu. Orta sahalarin cabuk gecilecegi, beraberligin neredeyse imkansiz oldugu bir mac. Ya 2-0 ya 0-2; ortasi yok."))



    rules.append(("EDGE_PARK_BUS","EDGE","Deplasman Kilidi (Park The Bus)",

                 (np.isfinite(pH) and np.isfinite(pUnder) and pH > 50 and pUnder > 65),

                 strength((pH-50) + (pUnder-65)),

                 "Ev sahibi favori ancak rakip tam bir savunma takimi. Deplasman ekibi 'Canakkale Gecilmez'i oynayacak. Ev sahibi kilidi acmakta cok zorlanabilir. 1-0'lik zoraki bir galibiyet veya surpriz bir 0-0 ihtimali cok yuksek."))



    rules.append(("EDGE_TRAP","EDGE","Yalanci Favori (Trap Match)",

                 (np.isfinite(pH) and np.isfinite(pD) and np.isfinite(pA) and (40 <= pH <= 50) and (pD + pA) > pH),

                 strength((pD+pA) - pH),

                 "Dikkat! Ev sahibi kagit uzerinde favori gorunse de veriler guven vermiyor. Puan kaybi ihtimali, kazanma ihtimalinden daha yuksek. Bu mac 'Banko 1' oynamak icin cok riskli, 'Cifte Sans (X2)' veya uzak durmak daha mantikli."))



    rules.append(("EDGE_21_PARADOX","EDGE","2-1 Paradoksu (Dengeli Gol)",

                 (np.isfinite(pBttsY) and np.isfinite(pOver) and pBttsY > 60 and (45 <= pOver <= 55)),

                 strength(min(pBttsY-60, 10) + (55-abs(pOver-50))),

                 "Istatistikler cok ozel bir sikismayi isaret ediyor. Iki takimin da gol atmasi neredeyse kesin (KG Var) ama macin gol yagmuruna donusmesi beklenmiyor. Veriler adeta 2-1 veya 1-1 skorunu bagiriyor."))



    rules.append(("FAV_HOME_UNDER","FAV","Profesyonel Is (Ev + Alt)",

                 (np.isfinite(pH) and np.isfinite(pUnder) and pH > 55 and pUnder > 55),

                 strength((pH-55)+(pUnder-55)),

                 "Ev sahibi maci rolantide goturup fisi ceker. Skor uretmekte zorlanmazlar ama farka da kosmazlar. Rakibin direnci zayif. 2-0 gibi temiz, 'profesyonel' ve gol yemeden bir galibiyet bekleniyor."))



    rules.append(("FAV_HOME_OVER_KGY","FAV","Govde Gosterisi (Ev + Ust)",

                 (np.isfinite(pH) and np.isfinite(pOver) and np.isfinite(pBttsN) and pH > 55 and pOver > 55 and pBttsN > 50),

                 strength((pH-55)+(pOver-55)+(pBttsN-50)),

                 "Saha tek yonlu egimli gibi. Ev sahibi rakibini sahadan silebilir. Deplasman ekibinin gol atma sansi dusukken, ev sahibinin tek basina 3 gol bulabilecegi bir mac. Handikapli galibiyet tercihi mantikli."))



    rules.append(("FAV_HOME_KGV","FAV","Gollu Zafer (Ev + KG Var)",

                 (np.isfinite(pH) and np.isfinite(pBttsY) and pH > 55 and pBttsY > 55),

                 strength((pH-55)+(pBttsY-55)),

                 "Ev sahibinin kazanmasina kesin gozuyle bakiliyor ancak savunma konsantrasyonu dusuk. Hucum gucleriyle kazanirlar ama kalelerinde gol gormeleri muhtemel. 3-1 veya 2-1 gibi skorlar masada."))



    rules.append(("FAV_AWAY_OVER","FAV","Surpriz Deplasman Sovu (Dep + Ust)",

                 (np.isfinite(pA) and np.isfinite(pOver) and pA > 50 and pOver > 55),

                 strength((pA-50)+(pOver-55)),

                 "Ev sahibi sahada yoklari oynuyor. Konuk ekip cok formda ve gol yollarinda etkili. Deplasman ekibinin sov yapip farka gidebilecegi, bol gollu bir dis saha galibiyeti."))



    rules.append(("FAV_AWAY_UNDER","FAV","Stratejik Deplasman (Dep + Alt)",

                 (np.isfinite(pA) and np.isfinite(pUnder) and pA > 50 and pUnder > 55),

                 strength((pA-50)+(pUnder-55)),

                 "Deplasman ekibi maci kontrol edip, risk almadan sonuca gidecek. Ev sahibinin gol atmasi cok zor gorunuyor. 0-1 veya 0-2 bitmesi muhtemel, kontrollu bir mac."))



    rules.append(("BAL_CHESS_UNDER","BAL","Satranc Maci (Denge + Alt)",

                 (np.isfinite(pH) and np.isfinite(pA) and np.isfinite(pUnder) and abs(pH-pA) < 15 and pUnder > 60),

                 strength((15-abs(pH-pA)) + (pUnder-60)),

                 "Tam bir taktik savasi. Iki takim da hata yapmaktan korkuyor. Ilk golu atan maci kilitler ve uzerine yatar. Taraf bahsi cok riskli. 1-0 veya 0-0 kilitlenmesi en muhtemel senaryo."))



    rules.append(("BAL_RUSSIAN_OVER","BAL","Rus Ruleti (Denge + Ust)",

                 (np.isfinite(pH) and np.isfinite(pA) and np.isfinite(pOver) and abs(pH-pA) < 15 and pOver > 60),

                 strength((15-abs(pH-pA)) + (pOver-60)),

                 "Savunmalarin guven vermedigi, her turlu sonuca acik bir kaos maci. Kimin kazanacagini kestirmek zor ama topun iki kalede de gidip gelecegi kesin. Taraf bahsinden kacip gollere yonelmeli. 2-2, 3-2 gibi skorlar cikabilir."))



    rules.append(("BAL_DRAW_UNDER","BAL","Kisir Beraberlik (X + Alt)",

                 (np.isfinite(pD) and np.isfinite(pUnder) and (pD > 30) and pUnder > 55),

                 strength((pD-30)+(pUnder-55)),

                 "Bultenin en 'beraberlik' kokan maci. Iki denk gucun mucadelesinde gol sesi zor cikar. Sikici, pozisyonsuz bir mucadele ve 0-0 / 1-1 bitmeye en yakin aday."))



    rules.append(("BAL_LIGHT_HOME_UNDER","BAL","Zoraki Ev Sahibi (Hafif Ev + Alt)",

                 (np.isfinite(pH) and np.isfinite(pUnder) and (40 <= pH <= 50) and pUnder > 55),

                 strength((pUnder-55) + (pH-40)),

                 "Ev sahibi saha avantajiyla bir adim onde ama rahat bir mac olmayacak. Rakip direnecektir. Tek farkli, taraftari terleterek gelen bir galibiyet (1-0, 2-0) beklentisi hakim."))



    rules.append(("BAL_DRAW_KGV","BAL","Gollu Beraberlik Eglimi (X + KG Var)",

                 (np.isfinite(pD) and np.isfinite(pBttsY) and pD > 28 and pBttsY > 60),

                 strength((pD-28)+(pBttsY-60)),

                 "Iki takim da birbirine dis gecirebilecek gucde ancak yenisemiyorlar. Ikisi de gol bulur ama kazanan cikmaz. 1-1 veya 2-2'lik skorla puanlarin kardesce paylasilmasi en guclu ihtimal."))



    rules.append(("BAL_LIGHT_AWAY_UNDER","BAL","Deplasman Direnisi (Hafif Dep + Alt)",

                 (np.isfinite(pA) and np.isfinite(pUnder) and (40 <= pA <= 50) and pUnder > 55),

                 strength((pUnder-55) + (pA-40)),

                 "Konuk ekip biraz daha kaliteli ayaklara sahip ama ev sahibi kati savunma yapacaktir. Deplasman ekibi kilidi acmakta zorlanabilir ama kaybetmesi beklenmiyor. 0-1 veya beraberlik on planda."))



    rules.append(("BAL_LIGHT_AWAY_OVER","BAL","Ters Kose (Hafif Dep + Ust)",

                 (np.isfinite(pA) and np.isfinite(pOver) and np.isfinite(pBttsY) and (40 <= pA <= 50) and pOver > 55 and pBttsY > 55),

                 strength((pOver-55)+(pBttsY-55)+(pA-40)),

                 "Ev sahibi gol bulsa da savunma zaaflari basina is acacak. Mac ortada gorunse de gol duellosundan deplasman ekibinin galip ayrilmasi veya macin yuksek skorlu berabere bitmesi bekleniyor."))



    best = None

    best_score = -1e9

    for sid, grp, title, cond, sc, body in rules:

        if not cond:

            continue

        bonus = 1000 if grp == "EDGE" else (500 if grp == "FAV" else 0)

        trust_boost = 0.0

        if np.isfinite(trust):

            trust_boost = 0.02 * trust

        total_sc = float(bonus + sc + trust_boost)

        if total_sc > best_score:

            best_score = total_sc

            best = (sid, grp, title, body)



    if best is None:

        sid, grp, title = "DEFAULT", "MIXED", "Varsayilan"

        body = "Veriler cok karmasik ve birbirine zit sinyaller veriyor. Bu macta istatistiksel bir trend yakalamak zor, canli bahiste gidisati gormek daha saglikli olabilir."

    else:

        sid, grp, title, body = best



    score_line = ""

    if score_mode or score_top3 or score_exp:

        if score_exp:

            score_line += f" Skor beklentisi: {score_exp}. "

        if score_mode:

            score_line += f"En olasi skor: {score_mode}. "

        if score_top3:

            score_line += f"Koridor: {score_top3}."



    text = pre + body + (" " + score_line.strip() if score_line else "")

    return sid, grp, title, text, ev



def _add_form_features(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:

        return df

    for base in ["Form_TMB", "Form_THBH"]:

        hcol = f"{base}_Home"

        acol = f"{base}_Away"

        if hcol in df.columns:

            series = df[hcol]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            parsed = series.apply(parse_form_string)

            if len(parsed) > 0:

                pts, wr, dr, lr, ln = zip(*parsed)

                df[f"{base}_Home_PPGN"] = pts

                df[f"{base}_Home_WinRate"] = wr

                df[f"{base}_Home_DrawRate"] = dr

                df[f"{base}_Home_LossRate"] = lr

                df[f"{base}_Home_Len"] = ln

        if acol in df.columns:

            series = df[acol]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            parsed = series.apply(parse_form_string)

            if len(parsed) > 0:

                pts, wr, dr, lr, ln = zip(*parsed)

                df[f"{base}_Away_PPGN"] = pts

                df[f"{base}_Away_WinRate"] = wr

                df[f"{base}_Away_DrawRate"] = dr

                df[f"{base}_Away_LossRate"] = lr

                df[f"{base}_Away_Len"] = ln

        if f"{base}_Home_PPGN" in df.columns and f"{base}_Away_PPGN" in df.columns:

            df[f"{base}_PPGN_Diff"] = (df[f"{base}_Home_PPGN"] - df[f"{base}_Away_PPGN"]) * 0.5

    return df



def _add_elo_diff(df: pd.DataFrame) -> pd.DataFrame:

    if "Elo_Home" in df.columns and "Elo_Away" in df.columns:

        df["Elo_Diff"] = (

            pd.to_numeric(df["Elo_Home"], errors="coerce")

            - pd.to_numeric(df["Elo_Away"], errors="coerce")

        )

        df["Elo_Diff"] = df["Elo_Diff"].fillna(0.0) * 0.01

    return df



# -----------------------------------------------------------------------------#

# Data Engine

# -----------------------------------------------------------------------------#

class DataEngine:

    BASE_REQUIRED = ["Date", "League", "Home_Team", "Away_Team"]

    ODDS_OPEN_REQUIRED = ["Odds_Open_Home", "Odds_Open_Draw", "Odds_Open_Away"]

    SCORE_REQUIRED_PAST = ["FT_Score_Home", "FT_Score_Away"]

    TEXT_COLUMNS = {"Date", "League", "HomeTeam", "AwayTeam", "Season", "Time", "Odds_Moves_Raw"}

    FEATURE_EXCLUDE = {

        "FTHG", "FTAG", "HomeScore", "AwayScore",

        "Result", "TotalGoals", "Total_Goals",

        "Is_Over25", "Is_BTTS",

        "Date", "League", "HomeTeam", "AwayTeam",

        "Season", "Time",

        "xG_Home_Actual", "xG_Away_Actual",

        "Match_ID",

    }

    EXCLUDE_PREFIXES = ("Sim_",)

    RENAME_MAP = {

        "Home_Team": "HomeTeam",

        "Away_Team": "AwayTeam",

        "FT_Score_Home": "FTHG",

        "FT_Score_Away": "FTAG",

        "HomeScore": "FTHG",

        "AwayScore": "FTAG",

        "Odds_Open_Home": "HomeOdd",

        "Odds_Open_Draw": "DrawOdd",

        "Odds_Open_Away": "AwayOdd",

        "Odds_Open_Over25": "Over25Odd",

        "Odds_Open_Under25": "Under25Odd",

        "Odds_Open_BTTS_Yes": "BTTSYesOdd",

        "Odds_Open_BTTS_No": "BTTSNoOdd",

    }



    @staticmethod

    def make_match_id(row: pd.Series) -> str:

        date_str = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "unknown"

        league = str(row["League"]).lower().strip().replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_")

        home = str(row["HomeTeam"]).lower().strip().replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_")

        away = str(row["AwayTeam"]).lower().strip().replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_")

        return f"{date_str}|{league}|{home}|{away}"



    @staticmethod

    def load_csv(uploaded_file) -> pd.DataFrame:

        import os

        # Determine file size

        if hasattr(uploaded_file, 'getvalue'):  # Streamlit UploadedFile

            file_size = len(uploaded_file.getvalue())

        elif hasattr(uploaded_file, 'read'):  # File-like object

            current_pos = uploaded_file.tell() if hasattr(uploaded_file, 'tell') else 0

            uploaded_file.seek(0, 2)  # Seek to end

            file_size = uploaded_file.tell()

            uploaded_file.seek(current_pos)  # Back to original

        elif isinstance(uploaded_file, str) and os.path.isfile(uploaded_file):  # Path

            file_size = os.path.getsize(uploaded_file)

        else:

            file_size = 0

        

        if file_size == 0:

            raise ValueError(f"CSV file appears empty. File size: {file_size} bytes. Please save as 'CSV UTF-8' from Excel and re-upload.")

        

        # Try reading with default settings

        try:

            if hasattr(uploaded_file, 'seek'):

                uploaded_file.seek(0)

            df = pd.read_csv(uploaded_file)

        except pd.errors.EmptyDataError:

            # Fallback: try with sep=None, engine="python"

            try:

                if hasattr(uploaded_file, 'seek'):

                    uploaded_file.seek(0)

                df = pd.read_csv(uploaded_file, sep=None, engine="python")

            except Exception as e:

                raise ValueError(f"CSV parsing failed even with fallback. File size: {file_size} bytes. Error: {str(e)}. Please check file format and encoding.")

        except Exception as e:

            raise ValueError(f"CSV loading failed. File size: {file_size} bytes. Error: {str(e)}. Please save as 'CSV UTF-8' from Excel and re-upload.")

        

        df.columns = [_norm_colname(c) for c in df.columns]

        return df



    @staticmethod

    def _coerce_numeric_columns(df: pd.DataFrame):

        for c in df.columns:

            if c in DataEngine.TEXT_COLUMNS:

                continue

            if c.startswith("Form_"):

                continue

            col = df[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            df[c] = _to_num(col)



    @staticmethod

    def _encode_form_columns(df: pd.DataFrame):

        form_cols = [c for c in df.columns if c.startswith("Form_")]

        for col in form_cols:

            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            parsed = [_form_metrics(v) for v in series]

            df[f"{col}_Points"] = [p[0] for p in parsed]

            df[f"{col}_WinRate"] = [p[1] for p in parsed]

            df[f"{col}_DrawRate"] = [p[2] for p in parsed]

            df[f"{col}_LossRate"] = [p[3] for p in parsed]



    @staticmethod

    def standardize(df: pd.DataFrame, *, is_past: bool) -> pd.DataFrame:

        d = df.copy()

        # PATCH 1: Alias normalization

        ALIASES = {

            # team name aliases

            "HomeTeam": "Home_Team",

            "AwayTeam": "Away_Team",

            # odds aliases (bazi exportlar direkt HomeOdd ile gelebilir)

            "HomeOdd": "Odds_Open_Home",

            "DrawOdd": "Odds_Open_Draw",

            "AwayOdd": "Odds_Open_Away",

            "Over25Odd": "Odds_Open_Over25",

            "Under25Odd": "Odds_Open_Under25",

            "BTTSYesOdd": "Odds_Open_BTTS_Yes",

            "BTTSNoOdd": "Odds_Open_BTTS_No",

        }

        for src, dst in ALIASES.items():

            # Eger dst beklenen ama yoksa, src varsa kopyala

            if dst not in d.columns and src in d.columns:

                val = d[src]
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                d[dst] = val

        _require_cols(d, DataEngine.BASE_REQUIRED, "INGEST")

        _require_cols(d, DataEngine.ODDS_OPEN_REQUIRED, "INGEST")

        if is_past:

            _require_cols(d, DataEngine.SCORE_REQUIRED_PAST, "INGEST(PAST)")

        d = d.rename(columns=DataEngine.RENAME_MAP)

        d["Date"] = _coerce_datetime(d["Date"])

        if d["Date"].isna().all():

            raise ValueError("[INGEST] Date parsing failed (all NaT). Check Date column format.")

        DataEngine._encode_form_columns(d)

        DataEngine._coerce_numeric_columns(d)

        for col in ["HomeOdd", "DrawOdd", "AwayOdd"]:

            if col not in d.columns:

                raise ValueError(f"[INGEST] Required column {col} not found after rename.")

            col_data = d[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            d[col] = _to_num(col_data)

        _nonempty_or_fail(d, ["HomeOdd", "DrawOdd", "AwayOdd"], "INGEST(ODDS)", allow_all_nan=False)

        if is_past:

            for col in ["FTHG", "FTAG"]:

                if col not in d.columns:

                    raise ValueError(f"[INGEST(PAST)] Missing score column {col}.")

            hg = d["FTHG"]
            if isinstance(hg, pd.DataFrame):
                hg = hg.iloc[:, 0]
            ag = d["FTAG"]
            if isinstance(ag, pd.DataFrame):
                ag = ag.iloc[:, 0]
            d["FTHG"] = _to_num(hg)

            d["FTAG"] = _to_num(ag)

            _nonempty_or_fail(d, ["FTHG", "FTAG"], "INGEST(PAST:SCORES)", allow_all_nan=False)

            conditions = [

                d["FTHG"] > d["FTAG"],

                d["FTHG"] < d["FTAG"],

            ]

            d["Result"] = np.select(conditions, ["HOME", "AWAY"], default="DRAW")

            d["TotalGoals"] = d["FTHG"].fillna(0) + d["FTAG"].fillna(0)

            d["Is_Over25"] = (d["TotalGoals"] > 2.5).astype(int)

            d["Is_BTTS"] = ((d["FTHG"] > 0) & (d["FTAG"] > 0)).astype(int)

        for imp_col, odd_col in [("Imp_Home", "HomeOdd"), ("Imp_Draw", "DrawOdd"), ("Imp_Away", "AwayOdd")]:

            if imp_col not in d.columns:

                col = d[odd_col]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                d[imp_col] = 1.0 / col.replace(0, np.nan)

        imp_sum = (d["Imp_Home"] + d["Imp_Draw"] + d["Imp_Away"]).replace(0, np.nan)

        d["Imp_Home_N"] = d["Imp_Home"] / imp_sum

        d["Imp_Draw_N"] = d["Imp_Draw"] / imp_sum

        d["Imp_Away_N"] = d["Imp_Away"] / imp_sum

        if "xG_Home" in d.columns and "xG_Away" in d.columns:

            d["xG_Total"] = (pd.to_numeric(d["xG_Home"], errors="coerce") + pd.to_numeric(d["xG_Away"], errors="coerce"))

        d["Match_ID"] = d.apply(DataEngine.make_match_id, axis=1)

        if not d["Match_ID"].is_unique:

            d["Match_ID"] = d["Match_ID"] + "|dup" + (d.groupby("Match_ID").cumcount() + 1).astype(str)

        _require_cols(d, ["Date", "League", "HomeTeam", "AwayTeam", "HomeOdd", "DrawOdd", "AwayOdd"], "STANDARDIZE(OUTPUT)")

        d = _add_form_features(d)

        d = _add_elo_diff(d)

        return d



# -----------------------------------------------------------------------------#

# Feature selection + weighting helpers

# -----------------------------------------------------------------------------#



def build_feature_metadata(past: pd.DataFrame, future: pd.DataFrame, null_threshold: float, manual_features: list[str] = None):

    mandatory_xg = ("xG_Home", "xG_Away")

    missing_xg = [col for col in mandatory_xg if col not in past.columns or col not in future.columns]

    if missing_xg:

        raise ValueError(f"Mandatory xG columns missing: {missing_xg}")

    common_cols = sorted(set(past.columns).intersection(future.columns))

    if manual_features:

        # Manual mode: start with selected features, ensure xG included

        feature_cols = list(set(manual_features + list(mandatory_xg)))

        # Filter to common numeric

        feature_cols = [col for col in feature_cols if col in common_cols and pd.api.types.is_numeric_dtype(past[col]) and pd.api.types.is_numeric_dtype(future[col])]

        drop_records = []

        null_info = []

        for col in common_cols:

            if col in feature_cols:

                null_ratio = float(future[col].isna().mean())

                null_info.append({"column": col, "null_ratio": null_ratio})

                if null_ratio > null_threshold and col not in mandatory_xg:

                    drop_records.append({"column": col, "reason": f"null_ratio {null_ratio:.1%} > threshold", "null_ratio": null_ratio})

                    feature_cols.remove(col)

        # Ensure xG stays if possible

        for col in mandatory_xg:

            if col in feature_cols or future[col].isna().mean() <= null_threshold:

                if col not in feature_cols:

                    feature_cols.append(col)

                    null_ratio = float(future[col].isna().mean())

                    null_info.append({"column": col, "null_ratio": null_ratio})

    else:

        # Auto mode: existing logic

        feature_cols = []

        drop_records = []

        null_info = []

        for col in common_cols:

            past_numeric = pd.api.types.is_numeric_dtype(past[col])

            future_numeric = pd.api.types.is_numeric_dtype(future[col])

            col_future = future[col]
            if isinstance(col_future, pd.DataFrame):
                col_future = col_future.iloc[:, 0]
            null_ratio = float(col_future.isna().mean())

            if not past_numeric or not future_numeric:

                drop_records.append({"column": col, "reason": "non-numeric column", "null_ratio": null_ratio})

                continue

            if col in DataEngine.FEATURE_EXCLUDE:

                drop_records.append({"column": col, "reason": "identity/target column", "null_ratio": null_ratio})

                continue

            if any(col.startswith(prefix) for prefix in DataEngine.EXCLUDE_PREFIXES):

                drop_records.append({"column": col, "reason": "excluded prefix", "null_ratio": null_ratio})

                continue

            null_info.append({"column": col, "null_ratio": null_ratio})

            if col in mandatory_xg:

                feature_cols.append(col)

                continue

            if null_ratio > null_threshold:

                drop_records.append({"column": col, "reason": f"null_ratio {null_ratio:.1%} > threshold", "null_ratio": null_ratio})

                continue

            feature_cols.append(col)

    extra_knn = []

    for c in ["Form_TMB_PPGN_Diff", "Form_THBH_PPGN_Diff", "Elo_Diff", "xG_Home", "xG_Away", "xG_Total"]:

        if c in past.columns and c in future.columns:

            extra_knn.append(c)

    feature_cols = list(dict.fromkeys(feature_cols + extra_knn))



    xg_info = {}

    for col in mandatory_xg:

        included = col in feature_cols

        reason = "meets threshold and present in both datasets" if included else "missing or filtered"

        xg_info[col] = {"included": included, "reason": reason}

    if len(feature_cols) < 5:

        raise ValueError("Feature set too small after filtering. Adjust null threshold or inspect data.")

    null_info_sorted = sorted(null_info, key=lambda r: r["null_ratio"], reverse=True)

    return feature_cols, drop_records, xg_info, null_info_sorted





def build_feature_weights(feature_cols: list[str], weight_config: dict[str, float]) -> dict[str, float]:

    weights: dict[str, float] = {}

    for col in feature_cols:

        if col in ("xG_Home", "xG_Away"):

            weights[col] = weight_config.get("xg", 2.0)

        elif col.startswith("Elo"):

            weights[col] = weight_config.get("elo", 1.2)

        elif col.startswith("League_Pos"):

            weights[col] = weight_config.get("league_pos", 1.1)

        else:

            weights[col] = weight_config.get("default", 1.0)

    return weights



# -----------------------------------------------------------------------------#

# Weighted scaler and KNN

# -----------------------------------------------------------------------------#



class WeightedStandardScaler:

    def __init__(self, feature_cols: list[str], weights: dict[str, float]):

        self.feature_cols = list(feature_cols)

        self.weights = np.array([weights.get(col, 1.0) for col in self.feature_cols], dtype=float)

        self.scaler = StandardScaler()



    def fit(self, X: np.ndarray):

        self.scaler.fit(X)



    def transform(self, X: np.ndarray) -> np.ndarray:

        scaled = self.scaler.transform(X)

        return scaled * self.weights





class KNNEngine:

    def __init__(

        self,

        history_df: pd.DataFrame,

        feature_cols: list[str],

        feature_weights: dict[str, float],

        k_same: int,

        k_global: int,

        min_same_found: int,

        same_league_mode: bool,

        conf_quality_floor: float,

    ):

        self.history_df = history_df.copy()

        self.feature_cols = list(feature_cols)

        self.feature_weights = feature_weights

        self.k_same = max(1, k_same)

        self.k_global = max(1, k_global)

        self.min_same_found = max(1, min_same_found)

        self.same_league_mode = same_league_mode

        self.conf_quality_floor = conf_quality_floor

        self.total_limit = self.k_same + self.k_global

        _require_cols(self.history_df, self.feature_cols, "KNN(history features)")

        self.global_wrapper = self._build_model(self.history_df, "global")

        self.league_wrappers: dict[str, dict] = {}

        for league, group in self.history_df.groupby("League"):

            if group.empty:

                continue

            self.league_wrappers[league] = self._build_model(group, "same")



    def _build_model(self, df: pd.DataFrame, source: str) -> dict:

        df2 = df.copy().reset_index()

        df2 = df2.rename(columns={"index": "orig_index"})

        X = df2[self.feature_cols].copy()

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        scaler = WeightedStandardScaler(self.feature_cols, self.feature_weights)

        scaler.fit(X.values)

        Xs = scaler.transform(X.values)

        max_neighbors = max(1, min(len(df2), self.total_limit))

        model = NearestNeighbors(n_neighbors=max_neighbors, metric="euclidean")

        model.fit(Xs)

        return {

            "df": df2,

            "scaler": scaler,

            "model": model,

            "max_neighbors": max_neighbors,

            "source": source,

        }



    def _run_knn(self, wrapper: dict, feature_vector: np.ndarray, n_neighbors: int) -> pd.DataFrame:

        if wrapper is None or wrapper["df"].empty:

            return pd.DataFrame()

        vector = np.array(feature_vector, dtype=float).reshape(1, -1)

        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        scaled = wrapper["scaler"].transform(vector)

        limit = min(n_neighbors, wrapper["max_neighbors"])

        if limit < 1:

            return pd.DataFrame()

        distances, indices = wrapper["model"].kneighbors(scaled, n_neighbors=limit)

        neighbors = wrapper["df"].iloc[indices[0]].copy()

        neighbors["Similarity_Score"] = 1.0 / (1.0 + distances[0])

        neighbors["neighbor_source"] = wrapper["source"]

        return neighbors



    def analyze_match(self, future_row: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:

        feature_vector = future_row[self.feature_cols].copy().values



        # --- DEFAULTS (HER PATH IÇIN TANIMLI) ---

        avg_similarity = 0.0

        sim_quality = 0.0

        effective_n = 0.0

        match_count = 0

        same_league_count = 0

        global_count = 0

        league_bonus = 0.85

        conf_raw = 0.0

        conf_ok = False

        conf_reason = "NO_NEIGHBORS"



        # Fillna for mandatory xG

        for i, col in enumerate(self.feature_cols):

            if col in ("xG_Home", "xG_Away") and pd.isna(feature_vector[i]):

                median_val = self.history_df[col].median()

                feature_vector[i] = median_val if not pd.isna(median_val) else 0.0



        neighbors_list = []



        # Same league

        if self.same_league_mode:

            league = future_row.get("League")

            league_wrapper = self.league_wrappers.get(league)

            if league_wrapper is not None and len(league_wrapper["df"]) >= self.min_same_found:

                same_neighbors = self._run_knn(league_wrapper, feature_vector, self.k_same)

                if not same_neighbors.empty:

                    same_league_count = len(same_neighbors)

                    neighbors_list.append(same_neighbors)



        # Global

        global_neighbors = self._run_knn(self.global_wrapper, feature_vector, self.k_global)

        if not global_neighbors.empty:

            global_count = len(global_neighbors)

            neighbors_list.append(global_neighbors)



        if not neighbors_list:

            sim_df = pd.DataFrame()

        else:

            sim_df = (

                pd.concat(neighbors_list, ignore_index=True)

                .sort_values("Similarity_Score", ascending=False)

                .drop_duplicates(subset="orig_index", keep="first")

                .head(self.total_limit)

                .reset_index(drop=True)

            )



        # --- ASIL HESAPLAR (SADECE sim_df VARSA) ---

        if not sim_df.empty:

            avg_similarity = float(sim_df["Similarity_Score"].mean())

            match_count = int(len(sim_df))



            top_quality = sim_df["Similarity_Score"].head(10)

            sim_quality = float(top_quality.mean()) if not top_quality.empty else 0.0



            effective_n = float(sim_df["Similarity_Score"].head(50).sum())



            league_bonus = 1.0 if same_league_count >= self.min_same_found else 0.85

            n_boost = _clamp(effective_n / 25.0, 0.0, 1.0)



            conf_raw = 100.0 * (sim_quality / 0.30) * (0.6 + 0.4 * n_boost) * league_bonus

            conf_raw = float(_clamp(conf_raw, 0.0, 100.0))



            conf_ok = sim_quality >= self.conf_quality_floor

            conf_reason = "OK" if conf_ok else f"SIM_QUALITY_BELOW_{self.conf_quality_floor:.2f}"



        # --- Anchor + Consensus Blend (Top-10 + All Neighbors) ---

        # NOTE: Bizim SIM_QUALITY olcegimiz dusuk (tepe ~0.25). Bu yuzden alpha'yi o olcege gore kalibre ediyoruz.

        # Goal: Top-10 (anchor) bilgisi etkilesin ama ziplatmasin (alpha max ~0.35).

        alpha = 0.0

        if not sim_df.empty:

            top10 = sim_df.head(10).copy()



            # All-neighbors (consensus)

            sim_home = float((sim_df["Result"] == "HOME").mean()) if "Result" in sim_df.columns else 0.0

            sim_draw = float((sim_df["Result"] == "DRAW").mean()) if "Result" in sim_df.columns else 0.0

            sim_away = float((sim_df["Result"] == "AWAY").mean()) if "Result" in sim_df.columns else 0.0



            sim_over = float(sim_df["Is_Over25"].mean()) if "Is_Over25" in sim_df.columns else 0.0

            sim_btts = float(sim_df["Is_BTTS"].mean()) if "Is_BTTS" in sim_df.columns else 0.0



            # Top-10 (anchor)

            top10_home = float((top10["Result"] == "HOME").mean()) if "Result" in top10.columns and not top10.empty else sim_home

            top10_draw = float((top10["Result"] == "DRAW").mean()) if "Result" in top10.columns and not top10.empty else sim_draw

            top10_away = float((top10["Result"] == "AWAY").mean()) if "Result" in top10.columns and not top10.empty else sim_away



            top10_over = float(top10["Is_Over25"].mean()) if "Is_Over25" in top10.columns and not top10.empty else sim_over

            top10_btts = float(top10["Is_BTTS"].mean()) if "Is_BTTS" in top10.columns and not top10.empty else sim_btts



            # alpha calibration for low-sim regime:

            # 0.15 -> 0.0 , 0.25 -> 1.0  (clip)

            sim_norm = _clamp((sim_quality - 0.15) / (0.25 - 0.15), 0.0, 1.0) if (0.25 - 0.15) > 0 else 0.0

            eff_norm = _clamp(effective_n / 20.0, 0.0, 1.0)

            alpha = 0.10 + 0.25 * sim_norm * (0.50 + 0.50 * eff_norm)  # ~0.10..0.35

            if sim_quality < 0.18:

                alpha = 0.0  # very low quality => disable anchor influence



            # blended finals (these are what pick_best_market will effectively use via res["Sim_*%"])

            final_home = (1.0 - alpha) * sim_home + alpha * top10_home

            final_draw = (1.0 - alpha) * sim_draw + alpha * top10_draw

            final_away = (1.0 - alpha) * sim_away + alpha * top10_away

            final_over = (1.0 - alpha) * sim_over + alpha * top10_over

            final_btts = (1.0 - alpha) * sim_btts + alpha * top10_btts

        else:

            final_home = final_draw = final_away = 0.0

            final_over = final_btts = 0.0



        # --- Poisson blend (small weight into final 1X2) ---

        if (

            "Poisson_Home_Pct" in future_row

            and "Poisson_Draw_Pct" in future_row

            and "Poisson_Away_Pct" in future_row

        ):

            ph = float(future_row.get("Poisson_Home_Pct") or 0.0)

            pd_ = float(future_row.get("Poisson_Draw_Pct") or 0.0)

            pa = float(future_row.get("Poisson_Away_Pct") or 0.0)

            s = ph + pd_ + pa

            if s > 0:

                ph, pd_, pa = ph / s, pd_ / s, pa / s

                w = 0.15

                final_home = (1 - w) * final_home + w * ph

                final_draw = (1 - w) * final_draw + w * pd_

                final_away = (1 - w) * final_away + w * pa



        # --- xG micro-bias (small tilt, then clamp/renorm) ---

        xh = future_row.get("xG_Home", None)

        xa = future_row.get("xG_Away", None)

        bh, bd, ba, bo, bb = xg_micro_bias(xh, xa)

        w_xg = 0.20

        final_home = final_home + w_xg * bh

        final_draw = final_draw + w_xg * bd

        final_away = final_away + w_xg * ba

        final_over = final_over + w_xg * bo

        final_btts = final_btts + w_xg * bb



        final_home = float(clamp(final_home, 0.0, 1.0))

        final_draw = float(clamp(final_draw, 0.0, 1.0))

        final_away = float(clamp(final_away, 0.0, 1.0))

        s = final_home + final_draw + final_away

        if s > 0:

            final_home, final_draw, final_away = final_home / s, final_draw / s, final_away / s

        final_over = float(clamp(final_over, 0.0, 1.0))

        final_btts = float(clamp(final_btts, 0.0, 1.0))

        sim_imp_home = np.nan
        sim_imp_draw = np.nan
        sim_imp_away = np.nan
        sim_imp_over = np.nan
        sim_imp_under = np.nan
        sim_imp_btts_yes = np.nan
        sim_imp_btts_no = np.nan
        if not sim_df.empty and "Similarity_Score" in sim_df.columns:
            w = pd.to_numeric(sim_df["Similarity_Score"], errors="coerce").fillna(0.0)
            w = w.clip(lower=0.0)
            w_sum = float(w.sum())

            def _wavg_implied(col):
                if col not in sim_df.columns or w_sum <= 0.0:
                    return np.nan
                series = sim_df[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                odds = pd.to_numeric(series, errors="coerce")
                implied = odds.apply(implied_prob_from_odd)
                m = implied.notna() & (w > 0.0)
                if not m.any():
                    return np.nan
                return float(np.average(implied[m], weights=w[m]))

            sim_imp_home = _wavg_implied("HomeOdd")
            sim_imp_draw = _wavg_implied("DrawOdd")
            sim_imp_away = _wavg_implied("AwayOdd")
            sim_imp_over = _wavg_implied("Over25Odd")
            sim_imp_under = _wavg_implied("Under25Odd")
            sim_imp_btts_yes = _wavg_implied("BTTSYesOdd")
            sim_imp_btts_no = _wavg_implied("BTTSNoOdd")

        xh_f = _safe_float(xh)

        xa_f = _safe_float(xa)

        mu_rh, mu_ra, eff_n = _weighted_xg_residual(sim_df)

        lamH, lamA, mode, top3 = score_expectation_from_xg(

            xh,

            xa,

            mu_rh=mu_rh,

            mu_ra=mu_ra,

            k_resid=0.35,

            max_goals=5,

        )



        stats = {

            # NOTE: final_* are blended (Top-10 + All neighbors) depending on alpha

            "Sim_Home_Win": float(final_home),

            "Sim_Draw": float(final_draw),

            "Sim_Away_Win": float(final_away),

            "Sim_Over25": float(final_over),

            "Sim_Under25": float(1.0 - final_over),

            "Sim_BTTS_Yes": float(final_btts),

            "Sim_BTTS_No": float(1.0 - final_btts),

            "Avg_Similarity": avg_similarity,

            "Match_Count": match_count,

            "SIM_QUALITY": sim_quality,

            "EFFECTIVE_N": effective_n,

            "CONF": conf_raw,

            "CONF_RAW": conf_raw,

            "CONF_OK": conf_ok,

            "CONF_REASON": conf_reason,

            "same_league_count": same_league_count,

            "global_count": global_count,

            "ANCHOR_ALPHA": float(alpha),

            "XG_Home": float(xh_f) if np.isfinite(xh_f) else np.nan,

            "XG_Away": float(xa_f) if np.isfinite(xa_f) else np.nan,

            "XG_Total": float(xh_f + xa_f) if np.isfinite(xh_f) and np.isfinite(xa_f) else np.nan,

            "XG_WEIGHT_CFG": float(w_xg),
            "XG_BIAS_W": float(w_xg),
            "XG_MICRO_BIAS_HOME": float(bh),
            "XG_MICRO_BIAS_DRAW": float(bd),
            "XG_MICRO_BIAS_AWAY": float(ba),
            "XG_MICRO_BIAS_OVER25": float(bo),
            "XG_MICRO_BIAS_BTTS": float(bb),

            "SCORE_LAM_H": float(lamH) if np.isfinite(lamH) else np.nan,

            "SCORE_LAM_A": float(lamA) if np.isfinite(lamA) else np.nan,

            "SCORE_MODE": mode,

            "SCORE_TOP3": top3,

            "SCORE_RES_RH": float(mu_rh),

            "SCORE_RES_RA": float(mu_ra),

            "SCORE_RES_EFFN": float(eff_n),

            "SCORE_RES_K": float(0.35),

            "SIM_IMP_HOME": sim_imp_home,
            "SIM_IMP_DRAW": sim_imp_draw,
            "SIM_IMP_AWAY": sim_imp_away,
            "SIM_IMP_OVER25": sim_imp_over,
            "SIM_IMP_UNDER25": sim_imp_under,
            "SIM_IMP_BTTS_YES": sim_imp_btts_yes,
            "SIM_IMP_BTTS_NO": sim_imp_btts_no,

        }



        sid, grp, title, text, evidence = select_scenario(stats)

        stats.update({

            "SCENARIO_ID": sid,

            "SCENARIO_GROUP": grp,

            "SCENARIO_TITLE": title,

            "SCENARIO_TEXT": text,

            "SCENARIO_EVIDENCE": evidence,

        })



        return stats, sim_df



# -----------------------------------------------------------------------------#

# Scoring

# -----------------------------------------------------------------------------#

def bestof_score(row: pd.Series) -> float:

    candidates = []

    for key in ["Sim_Home%", "Sim_Draw%", "Sim_Away%", "Sim_Over25%", "Sim_BTTS_Yes%"]:

        value = row.get(key)

        if pd.notna(value):

            candidates.append(value)

    prob_max = max(candidates) if candidates else 0.0

    quality = float(row.get("Avg_Similarity", 0.0)) if pd.notna(row.get("Avg_Similarity")) else 0.0

    ev_candidates = []

    market_map = [

        ("Sim_Home%", "HomeOdd"),

        ("Sim_Draw%", "DrawOdd"),

        ("Sim_Away%", "AwayOdd"),

        ("Sim_Over25%", "Over25Odd"),

        ("Sim_Under25%", "Under25Odd"),

        ("Sim_BTTS_Yes%", "BTTSYesOdd"),

        ("Sim_BTTS_No%", "BTTSNoOdd"),

    ]

    for prob_col, odd_col in market_map:

        prob = row.get(prob_col)

        odd = row.get(odd_col)

        if isinstance(prob, pd.DataFrame):
            prob = prob.iloc[:, 0]
        elif isinstance(prob, pd.Series):
            prob = prob.iloc[0] if len(prob) > 0 else np.nan
        if isinstance(odd, pd.DataFrame):
            odd = odd.iloc[:, 0]
        elif isinstance(odd, pd.Series):
            odd = odd.iloc[0] if len(odd) > 0 else np.nan
        if pd.isna(prob) or pd.isna(odd):

            continue

        if odd <= 1.01:

            continue

        ev = prob * odd - 1.0

        ev_candidates.append(ev)

    ev_bonus = max([ev for ev in ev_candidates if ev > 0.0], default=0.0)

    # EV bonusunu tamamen devreden çikardik.

    # Ana hedef: benzerlik orani + benzerlik kalitesi.

    score = (prob_max * 70.0) + (quality * 220.0)

    return float(score)



def implied_prob_from_odd(odd):
    try:
        odd = float(odd)
        if odd <= 1.01 or not math.isfinite(odd):
            return np.nan
        return 1.0 / odd
    except Exception:
        return np.nan


def poisson_over25(mu):
    """P(G>=3) for Poisson total goals."""
    if mu is None or not np.isfinite(mu):
        return np.nan
    return 1.0 - math.exp(-mu) * (1.0 + mu + (mu * mu) / 2.0)


def btts_prob(lam_h, lam_a):
    """P(BTTS Yes) from independent Poisson lambdas."""
    if not (np.isfinite(lam_h) and np.isfinite(lam_a)):
        return np.nan
    p_h = 1.0 - math.exp(-lam_h)
    p_a = 1.0 - math.exp(-lam_a)
    return p_h * p_a


def _load_policy_context(log_path):
    ctx = {
        "under_xg_p50": np.nan,
        "under_leagues": set(),
        "ms1_p_over25_q60": np.nan,
        "ms1_pick_margin_q50": np.nan,
        "ms2_pick_score_q80": np.nan,
        "ms2_pick_margin_q60": np.nan,
        "x_pick_score_q90": np.nan,
        "x_xg_total_q50": np.nan,
        "over_p_over25_q70": np.nan,
        "over_bestofrank_q70": np.nan,
        "under_pick_prob_q80": np.nan,
        "under_xg_home_q60": np.nan,
        "bttsy_p_bttsy_q80": np.nan,
        "bttsy_xg_home_q50": np.nan,
        "bttsn_p_bttsy_q50": np.nan,
        "bttsn_pick_margin_q50": np.nan,
    }
    if not isinstance(log_path, str) or not log_path:
        return ctx
    if not os.path.exists(log_path):
        return ctx
    try:
        log_df = pd.read_csv(log_path)
    except Exception:
        return ctx
    if log_df.empty:
        return ctx

    def _q(s, q):
        if s is None:
            return np.nan
        if isinstance(s, (int, float, np.number)):
            s = pd.Series([s])
        elif not isinstance(s, (pd.Series, list, tuple, np.ndarray)):
            s = pd.Series([s])
        s = pd.to_numeric(s, errors="coerce")
        if isinstance(s, pd.Series) and s.notna().any():
            return float(s.quantile(q))
        return np.nan

    def _norm_pick(v):
        s = str(v or "").strip()
        su = s.upper().replace(" ", "_")
        if su == "MSX":
            return "X"
        if su in ("MS1", "MS2", "X"):
            return su
        if su in ("OVER2.5", "OVER25"):
            return "Over2.5"
        if su in ("UNDER2.5", "UNDER25"):
            return "Under2.5"
        if su in ("BTTS_YES", "BTTSYES"):
            return "BTTS_Yes"
        if su in ("BTTS_NO", "BTTSNO"):
            return "BTTS_No"
        return s

    df = log_df.copy()
    df["PICK_FINAL_NORM"] = df.get("PICK_FINAL", "").map(_norm_pick)

    def _subset(group, pick):
        return df[
            (df.get("PICK_GROUP") == group)
            & (df.get("PICK_FINAL_NORM") == pick)
        ].copy()

    # 1X2 quantiles: global (all rows)
    ctx["ms1_p_over25_q60"] = _q(df.get("P_OVER25_ADJ"), 0.60)
    ctx["ms1_pick_margin_q50"] = _q(df.get("PICK_MARGIN"), 0.50)
    ctx["ms2_pick_score_q80"] = _q(df.get("PICK_SCORE_TOTAL"), 0.80)
    ctx["ms2_pick_margin_q60"] = _q(df.get("PICK_MARGIN"), 0.60)
    ctx["x_pick_score_q90"] = _q(df.get("PICK_SCORE_TOTAL"), 0.90)
    xg_total_all = df.get("xG_Total")
    if xg_total_all is None and {"xG_Home", "xG_Away"}.issubset(df.columns):
        xg_total_all = pd.to_numeric(df["xG_Home"], errors="coerce") + pd.to_numeric(df["xG_Away"], errors="coerce")
    ctx["x_xg_total_q50"] = _q(xg_total_all, 0.50)

    over = _subset("OU", "Over2.5")
    if not over.empty:
        ctx["over_p_over25_q70"] = _q(over.get("P_OVER25_ADJ"), 0.70)
        ctx["over_bestofrank_q70"] = _q(over.get("BestOfRank"), 0.70)

    under = _subset("OU", "Under2.5")
    if not under.empty:
        ctx["under_pick_prob_q80"] = _q(under.get("PICK_PROB_ADJ"), 0.80)
        ctx["under_xg_home_q60"] = _q(under.get("xG_Home"), 0.60)

    bttsy = _subset("BTTS", "BTTS_Yes")
    if not bttsy.empty:
        ctx["bttsy_p_bttsy_q80"] = _q(bttsy.get("P_BTTSY_ADJ"), 0.80)
        ctx["bttsy_xg_home_q50"] = _q(bttsy.get("xG_Home"), 0.50)

    bttsn = _subset("BTTS", "BTTS_No")
    if not bttsn.empty:
        ctx["bttsn_p_bttsy_q50"] = _q(bttsn.get("P_BTTSY_ADJ"), 0.50)
        ctx["bttsn_pick_margin_q50"] = _q(bttsn.get("PICK_MARGIN"), 0.50)

    under = log_df[
        (log_df.get("PICK_GROUP") == "OU")
        & (log_df.get("PICK_FINAL") == "Under2.5")
    ].copy()
    if not under.empty:
        if "xG_Total" not in under.columns and {"xG_Home", "xG_Away"}.issubset(under.columns):
            under["xG_Total"] = (
                pd.to_numeric(under["xG_Home"], errors="coerce")
                + pd.to_numeric(under["xG_Away"], errors="coerce")
            )
        xg = pd.to_numeric(under.get("xG_Total"), errors="coerce")
        if xg.notna().any():
            ctx["under_xg_p50"] = float(xg.quantile(0.50))

        if "RES_RESULT_OU" in under.columns and "League" in under.columns:
            under["HIT"] = np.where(
                under["RES_RESULT_OU"] == "Under2.5", 1, 0
            )
            by_league = under.groupby("League")["HIT"].agg(["count", "mean"]).reset_index()
            good = by_league[(by_league["count"] >= 3) & (by_league["mean"] >= 0.50)]
            ctx["under_leagues"] = set(good["League"].tolist())

    return ctx


def _policy_flag(row, ctx):
    def _f(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    group = str(row.get("PICK_GROUP", "") or "")
    pick = str(row.get("PICK_FINAL", "") or "")
    su = str(pick).strip().upper().replace(" ", "_")
    if su == "MSX":
        pick = "X"
    elif su in ("OVER2.5", "OVER25"):
        pick = "Over2.5"
    elif su in ("UNDER2.5", "UNDER25"):
        pick = "Under2.5"
    elif su in ("BTTS_YES", "BTTSYES"):
        pick = "BTTS_Yes"
    elif su in ("BTTS_NO", "BTTSNO"):
        pick = "BTTS_No"

    # MS1: P_OVER25_ADJ >= q60 & PICK_MARGIN <= q50
    if group == "1X2" and pick == "MS1":
        p_over = _f(row.get("P_OVER25_ADJ"))
        margin = _f(row.get("PICK_MARGIN"))
        t_over = _f(ctx.get("ms1_p_over25_q60"))
        t_margin = _f(ctx.get("ms1_pick_margin_q50"))
        if np.isfinite(p_over) and np.isfinite(t_over) and (p_over >= t_over):
            if np.isfinite(margin) and np.isfinite(t_margin) and (margin <= t_margin):
                return 1, "MS1_POVER_Q60_MARGIN_Q50"

    # MS2: PICK_SCORE_TOTAL <= q80 & PICK_MARGIN >= q60
    if group == "1X2" and pick == "MS2":
        pick_score = _f(row.get("PICK_SCORE_TOTAL"))
        margin = _f(row.get("PICK_MARGIN"))
        t_score = _f(ctx.get("ms2_pick_score_q80"))
        t_margin = _f(ctx.get("ms2_pick_margin_q60"))
        if np.isfinite(pick_score) and np.isfinite(t_score) and (pick_score <= t_score):
            if np.isfinite(margin) and np.isfinite(t_margin) and (margin >= t_margin):
                return 1, "MS2_SCORE_Q80_MARGIN_Q60"

    # MSX (Draw): PICK_SCORE_TOTAL >= q90 & xG_Total <= q50
    if group == "1X2" and pick == "X":
        pick_score = _f(row.get("PICK_SCORE_TOTAL"))
        xg_total = row.get("xG_Total")
        if xg_total is None or (isinstance(xg_total, float) and np.isnan(xg_total)):
            if "xG_Home" in row and "xG_Away" in row:
                xg_total = _f(row.get("xG_Home")) + _f(row.get("xG_Away"))
        xg_total = _f(xg_total)
        t_score = _f(ctx.get("x_pick_score_q90"))
        t_xg = _f(ctx.get("x_xg_total_q50"))
        if np.isfinite(pick_score) and np.isfinite(t_score) and (pick_score >= t_score):
            if np.isfinite(xg_total) and np.isfinite(t_xg) and (xg_total <= t_xg):
                return 1, "MSX_SCORE_Q90_XGT_Q50"

    # Over 2.5: P_OVER25_ADJ >= q70 & BestOfRank <= q70
    if group == "OU" and pick == "Over2.5":
        p_over = _f(row.get("P_OVER25_ADJ"))
        bestof = _f(row.get("BestOfRank"))
        t_over = _f(ctx.get("over_p_over25_q70"))
        t_bor = _f(ctx.get("over_bestofrank_q70"))
        if np.isfinite(p_over) and np.isfinite(t_over) and (p_over >= t_over):
            if np.isfinite(bestof) and np.isfinite(t_bor) and (bestof <= t_bor):
                return 1, "OU_OVER_POVER_Q70_BOR_Q70"

    # Under 2.5: PICK_PROB_ADJ >= q80 & xG_Home <= q60
    if group == "OU" and pick == "Under2.5":
        pick_prob = _f(row.get("PICK_PROB_ADJ"))
        xg_home = _f(row.get("xG_Home"))
        t_prob = _f(ctx.get("under_pick_prob_q80"))
        t_xg = _f(ctx.get("under_xg_home_q60"))
        if np.isfinite(pick_prob) and np.isfinite(t_prob) and (pick_prob >= t_prob):
            if np.isfinite(xg_home) and np.isfinite(t_xg) and (xg_home <= t_xg):
                return 1, "OU_UNDER_PROB_Q80_XGH_Q60"

    # BTTS Yes: P_BTTSY_ADJ >= q80 & xG_Home >= q50
    if group == "BTTS" and pick == "BTTS_Yes":
        p_bttsy = _f(row.get("P_BTTSY_ADJ"))
        xg_home = _f(row.get("xG_Home"))
        t_bttsy = _f(ctx.get("bttsy_p_bttsy_q80"))
        t_xg = _f(ctx.get("bttsy_xg_home_q50"))
        if np.isfinite(p_bttsy) and np.isfinite(t_bttsy) and (p_bttsy >= t_bttsy):
            if np.isfinite(xg_home) and np.isfinite(t_xg) and (xg_home >= t_xg):
                return 1, "BTTSY_PBTTS_Q80_XGH_Q50"

    # BTTS No: P_BTTSY_ADJ <= q50 & PICK_MARGIN <= q50
    if group == "BTTS" and pick == "BTTS_No":
        p_bttsy = _f(row.get("P_BTTSY_ADJ"))
        margin = _f(row.get("PICK_MARGIN"))
        t_bttsy = _f(ctx.get("bttsn_p_bttsy_q50"))
        t_margin = _f(ctx.get("bttsn_pick_margin_q50"))
        if np.isfinite(p_bttsy) and np.isfinite(t_bttsy) and (p_bttsy <= t_bttsy):
            if np.isfinite(margin) and np.isfinite(t_margin) and (margin <= t_margin):
                return 1, "BTTSN_PBTTS_Q50_MARGIN_Q50"

    return 0, ""


def pick_best_market(row: pd.Series, params=None, debug_context=None):
    def _f(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    def _prob01(x):
        v = _f(x)
        if not np.isfinite(v):
            return np.nan
        if v > 1.0 and v <= 100.0:
            return v / 100.0
        return v

    markets = [
        ("MS1", "Sim_Home%", "HomeOdd"),
        ("X", "Sim_Draw%", "DrawOdd"),
        ("MS2", "Sim_Away%", "AwayOdd"),
        ("Over2.5", "Sim_Over25%", "Over25Odd"),
        ("Under2.5", "Sim_Under25%", "Under25Odd"),
        ("BTTS_Yes", "Sim_BTTS_Yes%", "BTTSYesOdd"),
        ("BTTS_No", "Sim_BTTS_No%", "BTTSNoOdd"),
    ]
    odd_alias = {
        "HomeOdd": "Odds_Open_Home",
        "DrawOdd": "Odds_Open_Draw",
        "AwayOdd": "Odds_Open_Away",
        "Over25Odd": "Odds_Open_Over25",
        "Under25Odd": "Odds_Open_Under25",
        "BTTSYesOdd": "Odds_Open_BTTS_Yes",
        "BTTSNoOdd": "Odds_Open_BTTS_No",
    }

    # prob (benzerlik orani) ayniysa kalite ile tie-break yapalim
    row_simq = float(row.get("SIM_QUALITY", 0.0)) if pd.notna(row.get("SIM_QUALITY")) else 0.0

    # Prob fallbacks: bazı exportlarda Sim_* kolonları yok, SIM_IMP_* / P_*_ADJ var
    p_over = _prob01(row.get("Sim_Over25%"))
    if not np.isfinite(p_over):
        p_over = _prob01(row.get("SIM_IMP_OVER25"))
    if not np.isfinite(p_over):
        p_over = _prob01(row.get("P_OVER25_ADJ"))

    p_btts = _prob01(row.get("Sim_BTTS_Yes%"))
    if not np.isfinite(p_btts):
        p_btts = _prob01(row.get("SIM_IMP_BTTS_YES"))
    if not np.isfinite(p_btts):
        p_btts = _prob01(row.get("P_BTTSY_ADJ"))

    lam_h = _f(row.get("SCORE_LAM_H"))
    lam_a = _f(row.get("SCORE_LAM_A"))
    mu = (lam_h + lam_a) if (np.isfinite(lam_h) and np.isfinite(lam_a)) else np.nan

    beta = _f(row.get("XG_WEIGHT_CFG", row.get("XG_BIAS_W", 0.0)))
    if not np.isfinite(beta):
        beta = 0.0
    beta = float(np.clip(beta, 0.0, 0.35))

    p_over_lam = poisson_over25(mu)
    p_btts_lam = btts_prob(lam_h, lam_a)

    p_over_adj = p_over
    if np.isfinite(p_over) and np.isfinite(p_over_lam) and beta > 0.0:
        p_over_adj = (1.0 - beta) * p_over + (beta * p_over_lam)

    p_btts_adj = p_btts
    if np.isfinite(p_btts) and np.isfinite(p_btts_lam) and beta > 0.0:
        p_btts_adj = (1.0 - beta) * p_btts + (beta * p_btts_lam)

    p_under_adj = (1.0 - p_over_adj) if np.isfinite(p_over_adj) else np.nan
    p_btts_no_adj = (1.0 - p_btts_adj) if np.isfinite(p_btts_adj) else np.nan

    p_home = _prob01(row.get("Sim_Home%"))
    if not np.isfinite(p_home):
        p_home = _prob01(row.get("SIM_IMP_HOME"))

    p_draw = _prob01(row.get("Sim_Draw%"))
    if not np.isfinite(p_draw):
        p_draw = _prob01(row.get("SIM_IMP_DRAW"))
    if not np.isfinite(p_draw):
        p_draw = _prob01(row.get("Poisson_Draw_Pct"))

    p_away = _prob01(row.get("Sim_Away%"))
    if not np.isfinite(p_away):
        p_away = _prob01(row.get("SIM_IMP_AWAY"))

    prob_map = {
        "MS1": p_home,
        "X": p_draw,
        "MS2": p_away,
        "Over2.5": p_over_adj,
        "Under2.5": p_under_adj,
        "BTTS_Yes": p_btts_adj,
        "BTTS_No": p_btts_no_adj,
    }

    sim_imp_map = {
        "MS1": _f(row.get("SIM_IMP_HOME")),
        "X": _f(row.get("SIM_IMP_DRAW")),
        "MS2": _f(row.get("SIM_IMP_AWAY")),
        "Over2.5": _f(row.get("SIM_IMP_OVER25")),
        "Under2.5": _f(row.get("SIM_IMP_UNDER25")),
        "BTTS_Yes": _f(row.get("SIM_IMP_BTTS_YES")),
        "BTTS_No": _f(row.get("SIM_IMP_BTTS_NO")),
    }

    candidates = []
    cands_reason = ""
    saw_prob_finite = False
    saw_odd_finite = False
    k_mis = 0.75

    def _group_of(label: str) -> str:
        if label in ("MS1", "X", "MS2"):
            return "1X2"
        if label in ("Over2.5", "Under2.5"):
            return "OU"
        if label in ("BTTS_Yes", "BTTS_No"):
            return "BTTS"
        return ""

    def _scale_pos(b):
        try:
            b = float(b)
        except Exception:
            return 0.0
        if not np.isfinite(b):
            return 0.0
        return float(np.clip(b, 0.0, 0.25))

    def _bump_counter(key):
        try:
            st.session_state[key] = int(st.session_state.get(key, 0)) + 1
        except Exception:
            pass

    for label, _p_col, o_col in markets:
        prob = prob_map.get(label, np.nan)
        prob_f = _f(prob)
        if np.isfinite(prob_f):
            saw_prob_finite = True
        if not np.isfinite(prob):
            continue

        odd = row.get(o_col)
        if isinstance(odd, pd.DataFrame):
            odd = odd.iloc[:, 0]
        elif isinstance(odd, pd.Series):
            odd = odd.iloc[0] if len(odd) > 0 else np.nan
        if odd is None or (isinstance(odd, float) and np.isnan(odd)):
            alias = odd_alias.get(o_col)
            if alias:
                odd = row.get(alias)
        if label == "X" and (odd is None or (isinstance(odd, float) and np.isnan(odd))):
            sim_imp_draw = sim_imp_map.get("X", np.nan)
            if np.isfinite(sim_imp_draw) and sim_imp_draw > 0:
                odd = 1.0 / sim_imp_draw
        odd_f = _f(odd)
        if np.isfinite(odd_f) and odd_f > 1.01:
            saw_odd_finite = True
        ev = None
        mis = 0.0
        mis_bonus = 0.0
        implied_today = implied_prob_from_odd(odd)
        if pd.notna(odd) and odd > 1.01:
            ev = prob * odd - 1.0
        if np.isfinite(implied_today):
            mis = prob - implied_today
            mis_clip = float(np.clip(mis, -0.10, 0.15))
            mis_bonus = k_mis * max(0.0, mis_clip)

        sim_bonus = 0.0
        sim_imp = sim_imp_map.get(label, np.nan)
        if np.isfinite(sim_imp) and np.isfinite(implied_today):
            sim_gap = sim_imp - implied_today
            sim_bonus = float(np.clip(sim_gap * 0.4, -0.05, 0.05))

        scaled_bonus = _scale_pos(mis_bonus) + _scale_pos(sim_bonus)

        elo_h = _f(row.get("Elo_Home"))
        elo_a = _f(row.get("Elo_Away"))
        elo_factor = 0.0
        if np.isfinite(elo_h) and np.isfinite(elo_a):
            elo_gap = abs(float(elo_h) - float(elo_a))
            elo_factor = min(elo_gap / 60.0, 0.20)

        margin = _f(row.get("PICK_MARGIN"))
        open_factor = 0.0
        if np.isfinite(margin):
            opening_edge = max(0.0, float(margin) - 0.05)
            open_factor = min(opening_edge, 0.15)

        draw_pressure = _f(row.get("DRAW_PRESSURE"))
        draw_penalty = 0.0
        if np.isfinite(draw_pressure):
            draw_penalty = min(float(draw_pressure) * 0.10, 0.05)

        draw_odd = odd_f if np.isfinite(odd_f) else np.nan
        x_lift = 0.0
        if label == "X":
            if (
                np.isfinite(prob_f)
                and np.isfinite(draw_odd)
                and np.isfinite(p_over_adj)
                and prob_f >= 0.30
                and draw_odd >= 3.0
                and p_over_adj <= 0.58
            ):
                x_lift = 0.015

        bonus = scaled_bonus + elo_factor + open_factor - draw_penalty + x_lift
        bonus = float(np.clip(bonus, 0.0, 0.25))

        group = _group_of(label)
        market_weight = 1.0
        opening_edge = 0.0
        if np.isfinite(margin):
            opening_edge = float(np.clip(max(0.0, float(margin) - 0.05), 0.0, 0.10))

        imp_gap = 0.0
        if np.isfinite(sim_imp) and np.isfinite(implied_today):
            imp_gap = float(np.clip(sim_imp - implied_today, -0.06, 0.06))
        pos_gap = max(0.0, imp_gap)

        if group == "OU":
            lg10 = _f(row.get("SIM_LEAGUE_GOALS_AVG_10"))
            league_factor = 0.0
            if np.isfinite(lg10):
                league_factor = float(np.clip((float(lg10) - 2.45) / 0.60, 0.0, 1.0))
            market_weight = 1.0 + (0.10 * league_factor) + (0.15 * pos_gap) + (0.10 * opening_edge)
            market_weight = float(np.clip(market_weight, 1.0, 1.15))
        elif group == "BTTS":
            market_weight = 1.0 + (0.18 * pos_gap) + (0.10 * opening_edge)
            market_weight = float(np.clip(market_weight, 1.0, 1.18))
        elif group == "1X2":
            sim_q = _f(row.get("SIM_QUALITY"))
            sim_q_norm = 0.0
            if np.isfinite(sim_q):
                sim_q_norm = float(np.clip((float(sim_q) - 0.15) / (0.30 - 0.15), 0.0, 1.0))
            market_weight = 1.0 + (0.12 * sim_q_norm) + (0.20 * max(0.0, float(np.clip(imp_gap, -0.05, 0.05))))
            market_weight = float(np.clip(market_weight, 1.0, 1.15))

        synth_prob_pct = _f(row.get("synth_prob_pct", np.nan))
        prob_eff = prob
        if np.isfinite(synth_prob_pct):
            prob_eff = float(np.clip(synth_prob_pct / 100.0, 0.0, 1.0))

        p = float(np.clip(prob_eff, 0.0, 1.0))
        # Market-agnostic anchor alpha (equal for ALL groups)
        anchor_alpha = _resolve_param("ANCHOR_ALPHA", ANCHOR_ALPHA_DEFAULT, params, debug_context)
        alpha = float(anchor_alpha)
        p_adj = p ** alpha

        score = p_adj * (1.0 + bonus) * market_weight
        try:
            candidates.append(
                {
                    "market": label,
                    "mkt": label,
                    "prob": prob,
                    "prob_eff": prob_eff,
                    "prob_adj": p_adj,
                    "alpha": alpha,
                    "odd": odd,
                    "ev": ev,
                    "row_simq": row_simq,
                    "score": score,
                    "mispricing": float(mis_bonus) if np.isfinite(mis_bonus) else 0.0,
                    "mis_bonus": mis_bonus,
                    "sim_bonus": sim_bonus,
                    "implied_today": implied_today,
                    "sim_imp": sim_imp,
                    "scaled_bonus": scaled_bonus,
                    "bonus": bonus,
                    "x_lift": x_lift,
                    "group": group,
                    "market_weight": market_weight,
                    "opening_edge": opening_edge,
                    "pos_gap": pos_gap,
                    "p_over_adj": p_over_adj,
                    "p_draw": prob_f,
                    "draw_odd": draw_odd,
                }
            )
        except Exception:
            cands_reason = "EXCEPTION_IN_CAND_BUILD"
            continue

    if not candidates:
        if not saw_prob_finite:
            cands_reason = "MISSING_PROBS"
        elif not saw_odd_finite:
            cands_reason = "MISSING_ODDS"
        elif not cands_reason:
            cands_reason = "FILTERED_ALL_OR_INVALID"
        return None

    base = sorted(
        candidates,
        key=lambda x: (x.get("prob", 0.0), x.get("row_simq", 0.0)),
        reverse=True,
    )[0]

    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x.get("score", 0.0), x.get("row_simq", 0.0)),
        reverse=True,
    )

    best = candidates_sorted[0]
    best["_candidates"] = candidates
    best["_cands_reason"] = cands_reason or "OK"
    base_group = _group_of(base.get("market", ""))
    best_group = _group_of(best.get("market", ""))

    # --- FAIR OVERRIDE: market-agnostic ---
    if len(candidates_sorted) > 1 and best.get("score", 0.0) > 0:
        top_score = float(best.get("score", 0.0))
        top_mis = float(best.get("mispricing", 0.0) or 0.0)
        # --- Fair override knobs (market-agnostic) ---
        # Base close band (will be adjusted by trust): 0.95 = %5
        FAIR_CLOSE_BASE = _resolve_param("FAIR_CLOSE_BASE", FAIR_CLOSE_BASE_DEFAULT, params, debug_context)
        # Evidence thresholds
        EVID_TH = _resolve_param("EVID_TH", EVID_TH_DEFAULT, params, debug_context)
        # Mispricing thresholds:
        # - SOFT: allow switch only if there is evidence
        # - HARD: allow switch even without evidence
        MIS_BETTER_SOFT = _resolve_param("MIS_BETTER_SOFT", MIS_BETTER_SOFT_DEFAULT, params, debug_context)
        MIS_BETTER_HARD = _resolve_param("MIS_BETTER_HARD", MIS_BETTER_HARD_DEFAULT, params, debug_context)

        base_trust = float(best.get("trust", best.get("trust_pct", 0.0)) or 0.0)  # 0-100 or 0-1
        trust01 = base_trust / 100.0 if base_trust > 1.0 else base_trust
        close_enough = FAIR_CLOSE_BASE + (0.02 * (1.0 - float(np.clip(trust01, 0.0, 1.0))))  # trust high -> easier

        # Scan top-4 alternatives (not just 1-2)
        for alt_best in candidates_sorted[1:5]:
            alt_score = float(alt_best.get("score", 0.0) or 0.0)
            if alt_score <= 0:
                continue
            gap = (alt_score / top_score) if (top_score > 0 and alt_score > 0) else 0.0
            if not (gap >= close_enough):
                continue

            # Prob guard: don't allow very low-prob alts to override
            p = float(alt_best.get("prob_eff", alt_best.get("prob", 0.0)) or 0.0)
            if p < 0.18:
                continue

            ev_open = float(alt_best.get("opening_edge", 0.0) or 0.0)
            ev_scaled_bonus = float(alt_best.get("scaled_bonus", 0.0) or 0.0)
            ev_pos_gap = float(alt_best.get("pos_gap", 0.0) or 0.0)
            # market_weight evidence: only count excess above 1.0
            mw = float(alt_best.get("market_weight", 1.0) or 1.0)
            mw_excess = max(0.0, mw - 1.0)
            has_evidence = (ev_open >= EVID_TH) or (ev_scaled_bonus >= EVID_TH) or (ev_pos_gap >= EVID_TH) or (mw_excess >= EVID_TH)

            alt_mis = float(alt_best.get("mispricing", 0.0) or 0.0)
            better_mis_soft = alt_mis > (top_mis + MIS_BETTER_SOFT)
            better_mis_hard = alt_mis > (top_mis + MIS_BETTER_HARD)

            # FAIR_OVERRIDE: only allow when implied edge is truly better
            # Policy:
            # - hard value gap -> switch
            # - soft value gap -> switch only with evidence
            if better_mis_hard or (better_mis_soft and has_evidence):
                best = alt_best
                best["switch_reason"] = "FAIR_OVERRIDE_HARD_VALUE" if better_mis_hard else "FAIR_OVERRIDE_SOFT_VALUE_EVID"
                break

    if len(candidates_sorted) > 1 and best_group == "1X2" and best.get("score", 0.0) > 0:
        top_score = best.get("score", 0.0)
        top_mis = best.get("mispricing", 0.0)
        for alt in candidates_sorted[1:4]:
            alt_score = alt.get("score", 0.0)
            if alt_score <= 0:
                continue
            gap = alt_score / top_score
            # yakinlik: %5 bandi (istersen 0.97 yapariz)
            close = gap >= FAIR_CLOSE_BASE
            # Evidence: market_weight ham 1.0 oldugu icin "1.0 >= 0.03" bug'ini engelle
            opening_edge = float(alt.get("opening_edge", 0.0) or 0.0)
            mw = float(alt.get("market_weight", 1.0) or 1.0)
            mw_excess = max(0.0, mw - 1.0)  # yalnizca 1.0 ustu "kanit"
            scaled_bonus = float(alt.get("scaled_bonus", 0.0) or 0.0)
            pos_gap = float(alt.get("pos_gap", 0.0) or 0.0)

            has_evidence = (opening_edge >= EVID_TH) or (mw_excess >= EVID_TH) or (scaled_bonus >= EVID_TH) or (pos_gap >= EVID_TH)
            # pricing ustunlugu: top_mis'e gore kucuk ama anlamli fark (0.5% yerine 0.3% da deneyebiliriz)
            better_mis = float(alt.get("mispricing", 0.0) or 0.0) > (top_mis + 0.005)
            if close and (better_mis or has_evidence):
                best = alt
                best["switch_reason"] = "1X2_VALUE_OVERRIDE"
                break

    # --- X override gate ---
    x_candidate = next((c for c in candidates_sorted if c.get("mkt", c.get("market", "")) == "X"), None)
    if x_candidate is not None and best.get("market") != "X":
        top_score = float(best.get("score", 0.0) or 0.0)
        x_score = float(x_candidate.get("score", 0.0) or 0.0)
        ratio = (x_score / top_score) if (top_score > 0 and x_score > 0) else 0.0

        x_close = _resolve_param("X_SWITCH_CLOSE", 0.62, params, debug_context)
        x_topn = int(_resolve_param("X_SWITCH_TOPN", 3, params, debug_context))
        x_min_draw = _resolve_param("X_SWITCH_MIN_DRAW", 0.30, params, debug_context)
        x_min_draw_odd = _resolve_param("X_SWITCH_MIN_DRAW_ODD", 3.0, params, debug_context)
        x_max_over = _resolve_param("X_SWITCH_MAX_OVER", 0.58, params, debug_context)

        x_rank = None
        try:
            x_rank = 1 + [c.get("mkt", c.get("market", "")) for c in candidates_sorted].index("X")
        except Exception:
            x_rank = None

        p_draw_gate = _f(x_candidate.get("prob_eff", x_candidate.get("prob", np.nan)))
        draw_odd = _f(x_candidate.get("draw_odd", x_candidate.get("odd", np.nan)))
        p_over_gate = _f(x_candidate.get("p_over_adj", p_over_adj))

        if (
            (x_rank is not None and x_rank <= x_topn)
            and ratio >= x_close
            and np.isfinite(p_draw_gate)
            and p_draw_gate >= x_min_draw
            and np.isfinite(draw_odd)
            and draw_odd >= x_min_draw_odd
            and np.isfinite(p_over_gate)
            and p_over_gate <= x_max_over
        ):
            best = x_candidate
            best["switch_reason"] = "X_VALUE_OVERRIDE"
            _bump_counter("_dbg_x_switch_applied")

    best_group = _group_of(best.get("market", ""))
    final_market = best.get("market")
    best["final_market"] = final_market
    best["final_group"] = best_group
    best["base_market"] = base.get("market")
    best["base_prob"] = base.get("prob", np.nan)
    best["base_score"] = base.get("score", np.nan)
    best["switched"] = int(base.get("market") != final_market)
    best["mis_bonus"] = best.get("mis_bonus", np.nan)
    best["sim_bonus"] = best.get("sim_bonus", np.nan)
    best["score_total"] = best.get("score", np.nan)
    best["prob_eff"] = best.get("prob_eff", best.get("prob", np.nan))
    if not best.get("switch_reason"):
        best["switch_reason"] = ""
        if best["switched"] == 1:
            if np.isfinite(best.get("sim_bonus", np.nan)) and best.get("sim_bonus", 0.0) > 0.001:
                best["switch_reason"] = "SIM_GAP"
            elif np.isfinite(best.get("mis_bonus", np.nan)) and best.get("mis_bonus", 0.0) > 0.001:
                best["switch_reason"] = "MISPRICING"
            else:
                best["switch_reason"] = "OTHER"

    best["p_over25_adj"] = p_over_adj
    best["p_bttsy_adj"] = p_btts_adj

    try:
        _audit_pick_row(row, best, candidates_sorted, topn=6)
    except Exception:
        pass

    return best



def _to_float(x, default=np.nan):

    try:

        f = float(x)

        return f if np.isfinite(f) else default

    except Exception:

        return default



def _pct01_to_pct(v):

    """0-1 ise %'e cevir, 0-100 ise oldugu gibi birak."""

    f = _to_float(v, np.nan)

    if not np.isfinite(f):

        return np.nan

    return f * 100.0 if f <= 1.0 else f



def _get_prob_pct(row: dict, keys: list):

    for k in keys:

        if k in row:

            v = _pct01_to_pct(row.get(k))

            if np.isfinite(v):

                return v

    return np.nan



def _scenario_bias(market: str, scenario_id: str = "", scenario_title: str = ""):

    """

    Small biases (PCT points, percent-scale). Keep it tiny.

    Return value is in PCT points, e.g. +3.0 => +3%.

    Prefer SCENARIO_ID mapping (stable) over title keyword parsing (fragile).

    """

    sid = (scenario_id or "").strip().upper()

    t = (scenario_title or "").lower()



    b = 0.0



    # --- ID-based mapping (primary) ---

    # Notes:

    # markets used in synth: "MS1","MS2","X","Under2.5","Over2.5","BTTS_No","BTTS_Yes"

    if sid == "EDGE_ANTI_DRAW":

        # anti-draw: X down, slight push to sides

        if market == "X":

            b -= 3.0

        if market in ("MS1", "MS2"):

            b += 1.0



    elif sid == "EDGE_TRAP":

        # trap: avoid confident side picks

        if market in ("MS1", "MS2"):

            b -= 2.0



    elif sid == "EDGE_PARK_BUS":

        # park the bus: under + btts no

        if market == "Under2.5":

            b += 3.0

        if market == "BTTS_No":

            b += 2.0



    elif sid == "EDGE_21_PARADOX":

        # "2-1 / 1-1 corridor": BTTS yes is the main tilt, over is mild (not a goal-fest)

        if market == "BTTS_Yes":

            b += 3.0

        if market == "Over2.5":

            b += 1.0

        if market == "X":

            b += 0.5



    elif sid in ("BAL_CHESS_UNDER", "BAL_DRAW_UNDER", "FAV_HOME_UNDER", "FAV_AWAY_UNDER",

                 "BAL_LIGHT_HOME_UNDER", "BAL_LIGHT_AWAY_UNDER"):

        # all "ALT" scenarios

        if market == "Under2.5":

            b += 3.0

        if market == "BTTS_No":

            b += 2.0

        # draw-under specifically leans X a bit

        if sid == "BAL_DRAW_UNDER" and market == "X":

            b += 1.0



    elif sid in ("BAL_RUSSIAN_OVER", "FAV_AWAY_OVER", "BAL_LIGHT_AWAY_OVER"):

        # all "UST" scenarios (generally)

        if market == "Over2.5":

            b += 3.0

        if market == "BTTS_Yes":

            b += 2.0



    elif sid == "FAV_HOME_OVER_KGY":

        # "Ev + Ust" but condition includes BTTS_No (KG Yok) in your rules

        if market == "Over2.5":

            b += 3.0

        if market == "BTTS_No":

            b += 2.0



    elif sid == "FAV_HOME_KGV":

        # "Ev + KG Var" (BTTS yes)

        if market == "BTTS_Yes":

            b += 3.0

        # optional: mild over push (not strict in rule, keep small)

        if market == "Over2.5":

            b += 1.0



    elif sid == "BAL_DRAW_KGV":

        # "X + KG Var"

        if market == "BTTS_Yes":

            b += 3.0

        if market == "X":

            b += 1.0



    # --- Title keyword fallback (secondary, backward compatible) ---

    # Keep your old behavior so older stats without SCENARIO_ID still get some bias.

    if not sid:

        if "anti-beraberlik" in t or "ya hep ya hic" in t:

            if market == "X":

                b -= 3.0



        if ("park the bus" in t) or ("satranc" in t) or ("kisir beraberlik" in t) or ("denge + alt" in t):

            if market == "Under2.5":

                b += 3.0

            if market == "BTTS_No":

                b += 2.0



        if ("rus ruleti" in t) or ("denge + ust" in t):

            if market == "Over2.5":

                b += 3.0

            if market == "BTTS_Yes":

                b += 2.0



        if "yalanci favori" in t or "trap" in t:

            if market in ("MS1", "MS2"):

                b -= 2.0



        # quick support for short tags if they appear in titles

        if ("ev + alt" in t) or ("ev+alt" in t) or ("dep + alt" in t) or ("dep+alt" in t):

            if market == "Under2.5":

                b += 3.0

            if market == "BTTS_No":

                b += 2.0

        if ("ev + ust" in t) or ("ev+ust" in t) or ("dep + ust" in t) or ("dep+ust" in t):

            if market == "Over2.5":

                b += 3.0

            if market == "BTTS_Yes":

                b += 2.0



    return float(np.clip(b, -4.0, 4.0))



def _scorefit_bias(market: str, lamH, lamA):

    """

    Use SCORE_LAM_H/A to give tiny compatibility bias (percent-scale).

    """

    lh = _to_float(lamH, np.nan)

    la = _to_float(lamA, np.nan)

    if not (np.isfinite(lh) and np.isfinite(la)):

        return 0.0



    lt = lh + la

    diff = abs(lh - la)

    b = 0.0



    if market == "Over2.5":

        b += np.clip((lt - 2.7) * 6.0, -4.0, 4.0)

    if market == "Under2.5":

        b += np.clip((2.4 - lt) * 6.0, -4.0, 4.0)



    mn = min(lh, la)

    if market == "BTTS_Yes":

        b += np.clip((mn - 0.95) * 8.0, -4.0, 4.0)

    if market == "BTTS_No":

        b += np.clip((0.85 - mn) * 8.0, -4.0, 4.0)



    if market in ("MS1", "MS2", "X"):

        if diff >= 0.45:

            if market == "MS1" and (lh > la):

                b += 2.0

            if market == "MS2" and (la > lh):

                b += 2.0

            if market == "X":

                b -= 2.0

        else:

            if market == "X":

                b += 1.5



    return float(np.clip(b, -4.0, 4.0))



def _rank_bias(bestofrank):

    r = _to_float(bestofrank, np.nan)

    if not np.isfinite(r):

        return 0.0

    return float(np.clip((r - 70.0) * 0.05, -2.0, 2.0))



def _trust_bias(trust_pct):

    t = _to_float(trust_pct, np.nan)

    if not np.isfinite(t):

        return 0.0

    return float(np.clip((t - 60.0) * 0.10, -2.0, 3.0))



def _market_group(market: str):

    if market in ("MS1", "X", "MS2"):

        return "1X2"

    if market in ("Over2.5", "Under2.5"):

        return "OU"

    if market in ("BTTS_Yes", "BTTS_No"):

        return "BTTS"

    return "OTHER"



def pick_best_market_synth(row, return_debug: bool = False, params=None, debug_context=None):

    """

    Sentezli pick secimi (EV YOK):



    1) Similarity (KNN) market olasiliklarindan base prob (%) cikar

    2) Grup-cakismasi olmadan Top-3 aday sec (1X2, OU, BTTS)

    3) Kucuk bias'lar uygula: TRUST + SCORE_LAM + SCENARIO + BestOfRank

    4) Top-3 icinde en iyi synth sonucu PICK yap

    5) Fallback: data yetersizse pick_best_market()



    return_debug=True ise adaylar ve bias parcasi doner (UI debug icin).

    """

    if row is None:

        return pick_best_market(row, params=params, debug_context=debug_context)



    if isinstance(row, pd.Series):

        row = row.to_dict()

    def _bump_counter(key):
        try:
            st.session_state[key] = int(st.session_state.get(key, 0)) + 1
        except Exception:
            pass

    def _get_draw_odd(row_obj):
        odd = _to_float(row_obj.get("DrawOdd", np.nan), np.nan)
        if not np.isfinite(odd):
            odd = _to_float(row_obj.get("Odds_Open_Draw", np.nan), np.nan)
        if not np.isfinite(odd):
            sim_imp_draw = _to_float(row_obj.get("SIM_IMP_DRAW", np.nan), np.nan)
            if np.isfinite(sim_imp_draw) and sim_imp_draw > 0:
                odd = 1.0 / sim_imp_draw
        return odd



    markets = [

        ("MS1",      ["Sim_Home%", "Sim_Home_Win", "HOME_PCT", "Home_%", "p_home"], "HomeOdd"),

        ("X",        ["Sim_Draw%", "Sim_Draw",     "DRAW_PCT", "Draw_%", "p_draw"], "DrawOdd"),

        ("MS2",      ["Sim_Away%", "Sim_Away_Win", "AWAY_PCT", "Away_%", "p_away"], "AwayOdd"),

        ("Over2.5",  ["Sim_Over25%",  "Sim_Over25",  "OVER25_PCT",  "Over25_%",  "p_over25"], "Over25Odd"),

        ("Under2.5", ["Sim_Under25%", "Sim_Under25", "UNDER25_PCT", "Under25_%", "p_under25"], "Under25Odd"),

        ("BTTS_Yes", ["Sim_BTTS_Yes%", "Sim_BTTS_Yes", "BTTS_YES_PCT", "BTTS_Yes_%", "p_btts_yes"], "BTTSYesOdd"),

        ("BTTS_No",  ["Sim_BTTS_No%",  "Sim_BTTS_No",  "BTTS_NO_PCT",  "BTTS_No_%",  "p_btts_no"],  "BTTSNoOdd"),

    ]



    items = []

    for mkt, keys, odd_col in markets:

        p = _get_prob_pct(row, keys)  # percent-scale

        if np.isfinite(p):

            items.append({

                "market": mkt,

                "prob_pct": float(p),

                "group": _market_group(mkt),

                "odd": row.get(odd_col, np.nan),

            })



    if len(items) < 2:

        return pick_best_market(pd.Series(row), params=params, debug_context=debug_context) if isinstance(row, dict) else pick_best_market(row, params=params, debug_context=debug_context)



    # base ranking

    items = sorted(items, key=lambda x: x["prob_pct"], reverse=True)



    # Top-3 candidates with group uniqueness

    top3 = []

    used_groups = set()

    for it in items:

        if it["group"] in used_groups:

            continue

        top3.append(it)

        used_groups.add(it["group"])

        if len(top3) == 3:

            break

    # Allow X (Draw) to enter Top-3 if it's close to top 1X2 and passes a basic gate.
    # This keeps group-unique behavior but doesn't starve X completely.
    x_gate_ok = False
    x_min_draw_pct = _resolve_param("X_MIN_DRAW_PCT", 22.0, params, debug_context)
    x_gap_max_pct = _resolve_param("X_GAP_MAX_PCT", 4.0, params, debug_context)
    x_max_draw_odd = _resolve_param("X_MAX_DRAW_ODD", 4.8, params, debug_context)

    x_item = next((it for it in items if it.get("market") == "X"), None)
    top1_1x2 = next((it for it in top3 if it.get("group") == "1X2"), None)
    if x_item is not None and top1_1x2 is not None:
        x_prob = _to_float(x_item.get("prob_pct"), np.nan)
        top_prob = _to_float(top1_1x2.get("prob_pct"), np.nan)
        x_gap = top_prob - x_prob if np.isfinite(top_prob) and np.isfinite(x_prob) else np.nan
        x_odd = _to_float(x_item.get("odd"), np.nan)
        if (
            np.isfinite(x_prob)
            and x_prob >= x_min_draw_pct
            and np.isfinite(x_gap)
            and x_gap <= x_gap_max_pct
            and (not np.isfinite(x_odd) or x_odd <= x_max_draw_odd)
        ):
            if all((t.get("market") != "X") for t in top3):
                top3.append(x_item)
                x_gate_ok = True



    if not top3:

        return pick_best_market(pd.Series(row), params=params, debug_context=debug_context) if isinstance(row, dict) else pick_best_market(row, params=params, debug_context=debug_context)



    trust = row.get("TRUST_PCT", row.get("PROFILE_CONF", np.nan))

    lamH = row.get("SCORE_LAM_H", np.nan)

    lamA = row.get("SCORE_LAM_A", np.nan)

    scenario_id = row.get("SCENARIO_ID", row.get("SCENARIO", ""))

    scenario_title = row.get("SCENARIO_TITLE", "")

    bestofrank = row.get("BestOfRank", np.nan)



    # weights (trust dusukse score/scenario etkisini biraz kis)

    t_val = _to_float(trust, np.nan)

    if np.isfinite(t_val) and t_val < 55:

        trust_w, score_w, scen_w, rank_w = 0.8, 0.6, 0.6, 0.8

    else:

        trust_w, score_w, scen_w, rank_w = 0.8, 1.0, 1.0, 0.8



    scored = []

    for it in top3:

        mkt = it["market"]

        base = float(it["prob_pct"])



        b_trust = _trust_bias(trust) * trust_w

        b_score = _scorefit_bias(mkt, lamH, lamA) * score_w

        b_scen = _scenario_bias(mkt, scenario_id, scenario_title) * scen_w

        b_rank = _rank_bias(bestofrank) * rank_w

        b_total = float(b_trust + b_score + b_scen + b_rank)



        synth = float(base + b_total)

        scored.append({

            "market": mkt,

            "group": it["group"],

            "odd": it.get("odd", np.nan),

            "base_prob_pct": base,

            "bias_trust": float(b_trust),

            "bias_score": float(b_score),

            "bias_scenario": float(b_scen),

            "bias_rank": float(b_rank),

            "bias_total": b_total,

            "synth_prob_pct": synth,

        })



    scored_sorted = sorted(scored, key=lambda x: x["synth_prob_pct"], reverse=True)

    best_it = scored_sorted[0]



    # margin: how close was the decision?

    try:

        second = scored_sorted[1] if len(scored_sorted) > 1 else None

        pick_margin = float(best_it["synth_prob_pct"] - (second["synth_prob_pct"] if second else best_it["synth_prob_pct"]))

    except Exception:

        pick_margin = np.nan



    # close-call: base Top1'i koru (UI stabilitesi)

    top1 = top3[0]

    top1_row = next((x for x in scored_sorted if x["market"] == top1["market"]), None)

    if top1_row is not None:

        if (best_it["synth_prob_pct"] - top1_row["synth_prob_pct"]) < 1.0:

            if not (x_gate_ok and best_it.get("market") == "X"):
                best_it = top1_row

    # --- X override gate (synth) ---
    x_scored = next((x for x in scored_sorted if x.get("market") == "X"), None)
    if x_scored is None:
        x_item = next((it for it in items if it.get("market") == "X"), None)
        if x_item is not None:
            base = float(x_item["prob_pct"])
            b_trust = _trust_bias(trust) * trust_w
            b_score = _scorefit_bias("X", lamH, lamA) * score_w
            b_scen = _scenario_bias("X", scenario_id, scenario_title) * scen_w
            b_rank = _rank_bias(bestofrank) * rank_w
            b_total = float(b_trust + b_score + b_scen + b_rank)
            synth = float(base + b_total)
            x_scored = {
                "market": "X",
                "group": x_item.get("group"),
                "odd": x_item.get("odd", np.nan),
                "base_prob_pct": base,
                "bias_trust": float(b_trust),
                "bias_score": float(b_score),
                "bias_scenario": float(b_scen),
                "bias_rank": float(b_rank),
                "bias_total": b_total,
                "synth_prob_pct": synth,
            }
            x_scored["_synthed_fallback"] = True
    if x_scored is not None and best_it.get("market") != "X":
        top_score = float(best_it.get("synth_prob_pct", 0.0) or 0.0)
        x_score = float(x_scored.get("synth_prob_pct", 0.0) or 0.0)
        ratio = (x_score / top_score) if (top_score > 0 and x_score > 0) else 0.0

        x_close = _resolve_param("X_SWITCH_CLOSE", 0.62, params, debug_context)
        x_topn = int(_resolve_param("X_SWITCH_TOPN", 3, params, debug_context))
        x_min_draw_pct = _resolve_param("X_SWITCH_MIN_DRAW_PCT", 30.0, params, debug_context)
        x_min_draw_odd = _resolve_param("X_SWITCH_MIN_DRAW_ODD", 3.0, params, debug_context)
        x_max_over_pct = _resolve_param("X_SWITCH_MAX_OVER_PCT", 58.0, params, debug_context)

        p_draw_pct = _get_prob_pct(row, ["Sim_Draw%", "Sim_Draw", "DRAW_PCT", "Draw_%", "p_draw"])
        p_over_pct = _get_prob_pct(row, ["P_OVER25_ADJ", "Sim_Over25%", "Sim_Over25", "OVER25_PCT", "Over25_%", "p_over25"])
        draw_odd = _get_draw_odd(row)
        x_rank = None
        try:
            rank_pool = list(scored_sorted)
            if all((x.get("market") != "X") for x in rank_pool):
                rank_pool.append(x_scored)
            rank_pool = sorted(rank_pool, key=lambda x: x.get("synth_prob_pct", -1e9), reverse=True)
            x_rank = 1 + [x.get("market") for x in rank_pool].index("X")
        except Exception:
            x_rank = None

        if (
            (x_rank is not None and x_rank <= x_topn)
            and ratio >= x_close
            and np.isfinite(p_draw_pct)
            and p_draw_pct >= x_min_draw_pct
            and np.isfinite(draw_odd)
            and draw_odd >= x_min_draw_odd
            and np.isfinite(p_over_pct)
            and p_over_pct <= x_max_over_pct
        ):
            best_it = x_scored
            best_it["switch_reason"] = "X_VALUE_OVERRIDE"
            _bump_counter("_dbg_x_switch_applied_synth")
            others = [x for x in scored_sorted if x.get("market") != best_it.get("market")]
            if others:
                pick_margin = float(best_it["synth_prob_pct"] - others[0]["synth_prob_pct"])



    out = {

        "market": best_it["market"],

        "prob": float(np.clip(best_it["base_prob_pct"] / 100.0, 0.0, 1.0)),

        "odd": best_it.get("odd", np.nan),

        "synth_prob_pct": float(best_it["synth_prob_pct"]),

        "pick_margin": pick_margin,

        "_candidates": best_it.get("_candidates", []),

    }
    out["switch_reason"] = best_it.get("switch_reason", "")

    # --- Backfill score components from pick_best_market candidates (no recompute) ---
    final_mkt = out.get("market", "") or out.get("mkt", "") or out.get("FINAL_MKT", "")
    cands = out.get("_candidates", None)
    if (not isinstance(cands, list)) or (isinstance(cands, list) and not cands):
        _row = locals().get("row", None)
        if isinstance(_row, dict):
            cands = _row.get("_candidates")
    cand_final = None
    if isinstance(cands, list) and cands:
        cand_final = next((c for c in cands if c.get("mkt", c.get("market", "")) == final_mkt), None)
    if isinstance(cand_final, dict):
        # group backfill (keep both keys for compatibility)
        _g = cand_final.get("final_group", None)
        if _g is None:
            _g = cand_final.get("group", None)
        if _g is None:
            _g = cand_final.get("pick_group", None)
        if _g is not None:
            out["final_group"] = out.get("final_group", _g)
            out["group"] = out.get("group", _g)

        # score_total backfill (prefer score_total, else score)
        _st = cand_final.get("score_total", None)
        if _st is None:
            _st = cand_final.get("score", None)
        if _st is None:
            _st = cand_final.get("sel_score", None)
        if _st is not None:
            out["score_total"] = out.get("score_total", _st)

        out["market_weight"] = cand_final.get("market_weight", out.get("market_weight", 1.0))
        out["mis_bonus"] = cand_final.get("mis_bonus", out.get("mis_bonus", 0.0))
        out["sim_bonus"] = cand_final.get("sim_bonus", out.get("sim_bonus", 0.0))
        out["implied_today"] = cand_final.get("implied_today", out.get("implied_today", np.nan))
        out["sim_imp"] = cand_final.get("sim_imp", out.get("sim_imp", np.nan))
        if np.isfinite(_to_float(out.get("sim_imp"), np.nan)) and np.isfinite(_to_float(out.get("implied_today"), np.nan)):
            out["sim_gap"] = _to_float(out.get("sim_imp"), np.nan) - _to_float(out.get("implied_today"), np.nan)
        out["bonus_total"] = cand_final.get("bonus_total", out.get("bonus_total", np.nan))
        if not np.isfinite(_to_float(out.get("bonus_total"), np.nan)):
            mb = _to_float(out.get("mis_bonus"), 0.0)
            sb = _to_float(out.get("sim_bonus"), 0.0)
            out["bonus_total"] = float(np.clip(mb, 0.0, 0.25) + np.clip(sb, 0.0, 0.25))
        out["synth_backfilled"] = True
    else:
        out.setdefault("final_group", "")
        out.setdefault("group", "")
        out.setdefault("score_total", np.nan)
        out.setdefault("market_weight", 1.0)
        out.setdefault("mis_bonus", 0.0)
        out.setdefault("sim_bonus", 0.0)
        out.setdefault("bonus_total", 0.0)
        out["synth_backfilled"] = False



    if return_debug:

        for x in scored_sorted:

            x["chosen"] = (x["market"] == best_it["market"])

        out["debug"] = {

            "weights": {"trust_w": trust_w, "score_w": score_w, "scen_w": scen_w, "rank_w": rank_w},

            "trust": _to_float(trust, np.nan),

            "bestofrank": _to_float(bestofrank, np.nan),

            "lamH": _to_float(lamH, np.nan),

            "lamA": _to_float(lamA, np.nan),

            "scenario_id": scenario_id,

            "scenario_title": scenario_title,

            "pick_margin": _to_float(pick_margin, np.nan),

            "candidates": scored_sorted,

            "top1_base": top1.get("market"),

        }



    return out

def build_prediction_table(res: pd.DataFrame, policy_ctx: dict | None = None) -> pd.DataFrame:

    def _fill_sel_defaults_for_no_cands(row: dict, reason: str):
        """
        Candidate-less fallback / fair-override path:
        Ensure SEL_* are never blank, but clearly marked as gated/low-evidence.
        """
        sel_alpha = row.get("SEL_ALPHA")
        if sel_alpha is None or (isinstance(sel_alpha, float) and (sel_alpha != sel_alpha)):
            row["SEL_ALPHA"] = 0.0

        sel_sim = row.get("SEL_SIM")
        if sel_sim is None or (isinstance(sel_sim, float) and (sel_sim != sel_sim)):
            row["SEL_SIM"] = 0.0

        p = row.get("PICK_PROB_ADJ")
        if p is None or (isinstance(p, float) and (p != p)):
            p = row.get("SEL_P_ADJ")
        if p is None or (isinstance(p, float) and (p != p)):
            p = row.get("PICK_PROB", 0.0)
        row["SEL_SCORE"] = float(p) if p is not None else 0.0

        sel_p_adj = row.get("SEL_P_ADJ")
        if sel_p_adj is None or (isinstance(sel_p_adj, float) and (sel_p_adj != sel_p_adj)):
            row["SEL_P_ADJ"] = row.get("PICK_PROB_ADJ", row.get("PICK_PROB", 0.0))

        row["SEL_MISPRICING"] = 0.0
        row["SEL_MIS_BONUS"] = 0.0
        row["CANDS_REASON"] = reason

    rows = []
    if policy_ctx is None:
        log_path = os.path.join(_get_base_dir(), "pred_results_log.csv")
        policy_ctx = _load_policy_context(log_path)
    debug_global = st.session_state.get("_debug", {}) if hasattr(st, "session_state") else {}
    params = {}
    if isinstance(debug_global, dict):
        params = debug_global.get("chosen_params", {}) or {}

    _dd = locals().get("debug_dict")
    ctx = _dd if isinstance(_dd, dict) and _dd else debug_global

    if st.session_state.get("DEBUG_MODE", False):
        _PICK_SWITCH_AUDIT_ROWS.clear()

    for _, r in res.iterrows():

        try:

            best = pick_best_market_synth(r, params=params, debug_context=ctx)

        except Exception:

            best = pick_best_market(r, params=params, debug_context=ctx)

        if not best:

            continue



        # Trust tek yerden gelsin

        trust = r.get("TRUST_PCT")

        if trust is None or (isinstance(trust, float) and np.isnan(trust)):

            trust = r.get("PROFILE_CONF", np.nan)



        # DISPLAY PROB: synth varsa onu goster (yoksa base)

        prob = best.get("prob", np.nan)

        sp = best.get("synth_prob_pct", np.nan)

        if np.isfinite(_to_float(sp, np.nan)):

            prob = float(np.clip(_to_float(sp) / 100.0, 0.0, 1.0))

        def _norm_mkt(v):
            if v is None:
                return ""
            s = str(v).strip().lower()
            s = s.replace(" ", "").replace("_", "")
            aliases = {
                "1": "MS1",
                "ms1": "MS1",
                "home": "MS1",
                "ev": "MS1",
                "h": "MS1",
                "x": "X",
                "draw": "X",
                "berabere": "X",
                "2": "MS2",
                "ms2": "MS2",
                "away": "MS2",
                "dep": "MS2",
                "a": "MS2",
                "over25": "Over2.5",
                "over2.5": "Over2.5",
                "o25": "Over2.5",
                "o2.5": "Over2.5",
                "under25": "Under2.5",
                "under2.5": "Under2.5",
                "u25": "Under2.5",
                "u2.5": "Under2.5",
                "bttsyes": "BTTS_Yes",
                "bttsvar": "BTTS_Yes",
                "kgvar": "BTTS_Yes",
                "bttsno": "BTTS_No",
                "kgyok": "BTTS_No",
            }
            return aliases.get(s, v if isinstance(v, str) else str(v))

        # Final pick alignment (PICK_ODD / PICK_PROB should match PICK_FINAL)
        final_market = best.get("final_market", best.get("market", ""))
        pick_odd = best.get("odd", np.nan)
        pick_prob_base = prob
        pick_prob_eff = prob
        pick_prob_adj = np.nan
        sel_alpha = np.nan
        sel_sim = np.nan
        sel_score = np.nan
        sel_mispricing = np.nan
        sel_mis_bonus = np.nan
        cands = r.get("_candidates", None) if isinstance(r, (dict, pd.Series)) else None
        if cands is None:
            cands = best.get("_candidates", None) or best.get("candidates", None)
        if isinstance(cands, str) and cands.strip().startswith("["):
            try:
                import ast
                cands = ast.literal_eval(cands)
            except Exception:
                cands = []
        has_cands = isinstance(cands, list) and len(cands) > 0
        cands_n = len(cands) if isinstance(cands, list) else 0
        cands_mkts = []
        found_final = 0
        cands_reason = ""
        top1_mkt = ""
        top2_mkt = ""
        top3_mkt = ""
        top1_score = np.nan
        top2_score = np.nan
        top3_score = np.nan
        top1_prob_eff = np.nan
        top2_prob_eff = np.nan
        top3_prob_eff = np.nan
        cand_final = None
        if isinstance(r, (dict, pd.Series)):
            cands_reason = r.get("_cands_reason", "") if hasattr(r, "get") else ""
        if not cands_reason and isinstance(best, dict):
            cands_reason = best.get("_cands_reason", "")
        if not cands_reason:
            cands_reason = "CANDS_EMPTY" if not has_cands else "OK"
        if has_cands:
            cands_mkts = [str(c.get("mkt", c.get("market", ""))) for c in cands]
            try:
                cands_sorted = sorted(
                    cands,
                    key=lambda x: (x.get("score", 0.0), x.get("row_simq", 0.0)),
                    reverse=True,
                )
                if len(cands_sorted) > 0:
                    top1_mkt = str(cands_sorted[0].get("mkt", cands_sorted[0].get("market", "")))
                    top1_score = _to_float(cands_sorted[0].get("score", np.nan), np.nan)
                    top1_prob_eff = _to_float(cands_sorted[0].get("prob_eff", cands_sorted[0].get("prob", np.nan)), np.nan)
                if len(cands_sorted) > 1:
                    top2_mkt = str(cands_sorted[1].get("mkt", cands_sorted[1].get("market", "")))
                    top2_score = _to_float(cands_sorted[1].get("score", np.nan), np.nan)
                    top2_prob_eff = _to_float(cands_sorted[1].get("prob_eff", cands_sorted[1].get("prob", np.nan)), np.nan)
                if len(cands_sorted) > 2:
                    top3_mkt = str(cands_sorted[2].get("mkt", cands_sorted[2].get("market", "")))
                    top3_score = _to_float(cands_sorted[2].get("score", np.nan), np.nan)
                    top3_prob_eff = _to_float(cands_sorted[2].get("prob_eff", cands_sorted[2].get("prob", np.nan)), np.nan)
            except Exception:
                pass
            final_norm = _norm_mkt(final_market)
            cand_final = next(
                (c for c in cands if _norm_mkt(c.get("mkt", c.get("market", ""))) == final_norm),
                None,
            )
            if isinstance(cand_final, dict):
                found_final = 1
                pick_odd = cand_final.get("odd", pick_odd)
                pick_prob_base = cand_final.get("prob", pick_prob_base)
                pick_prob_eff = cand_final.get("prob_eff", cand_final.get("prob", pick_prob_eff))
                pick_prob_adj = cand_final.get("prob_adj", pick_prob_adj)
                sel_alpha = cand_final.get("alpha", sel_alpha)
                sel_sim = cand_final.get("sim_bonus", sel_sim)
                sel_score = cand_final.get("score", sel_score)
                sel_mispricing = cand_final.get("mispricing", sel_mispricing)
                sel_mis_bonus = cand_final.get("mis_bonus", sel_mis_bonus)

        if found_final != 1:
            pick_prob_adj = np.nan
            sel_alpha = np.nan
            sel_sim = np.nan
            sel_score = np.nan
            sel_mispricing = np.nan
            sel_mis_bonus = np.nan
            sel_p_adj = np.nan



        def _prob01_local(x):
            v = _to_float(x, np.nan)
            if not np.isfinite(v):
                return np.nan
            if v > 1.0 and v <= 100.0:
                return v / 100.0
            return v

        lam_h = _to_float(r.get("SCORE_LAM_H"), np.nan)
        lam_a = _to_float(r.get("SCORE_LAM_A"), np.nan)
        mu = (lam_h + lam_a) if (np.isfinite(lam_h) and np.isfinite(lam_a)) else np.nan
        beta = _to_float(r.get("XG_WEIGHT_CFG", r.get("XG_BIAS_W")), 0.0)
        beta = float(np.clip(beta, 0.0, 0.35)) if np.isfinite(beta) else 0.0

        p_over_base = _prob01_local(r.get("Sim_Over25%"))
        p_btts_base = _prob01_local(r.get("Sim_BTTS_Yes%"))
        p_over_lam = poisson_over25(mu)
        p_btts_lam = btts_prob(lam_h, lam_a)

        p_over_adj = p_over_base
        if np.isfinite(p_over_base) and np.isfinite(p_over_lam) and beta > 0.0:
            p_over_adj = (1.0 - beta) * p_over_base + (beta * p_over_lam)
        p_btts_adj = p_btts_base
        if np.isfinite(p_btts_base) and np.isfinite(p_btts_lam) and beta > 0.0:
            p_btts_adj = (1.0 - beta) * p_btts_base + (beta * p_btts_lam)

        sel_p_adj = np.nan
        if final_market == "Over2.5":
            sel_p_adj = p_over_adj
        elif final_market == "Under2.5" and np.isfinite(p_over_adj):
            sel_p_adj = 1.0 - p_over_adj
        elif final_market == "BTTS_Yes" and np.isfinite(p_btts_adj):
            sel_p_adj = p_btts_adj
        elif final_market == "BTTS_No" and np.isfinite(p_btts_adj):
            sel_p_adj = 1.0 - p_btts_adj

        mispricing_pick = np.nan
        if np.isfinite(_to_float(sel_mispricing, np.nan)):
            mispricing_pick = float(sel_mispricing)
        else:
            if isinstance(pick_odd, pd.DataFrame):
                pick_odd = pick_odd.iloc[:, 0]
            elif isinstance(pick_odd, pd.Series):
                pick_odd = pick_odd.iloc[0] if len(pick_odd) > 0 else np.nan
            if np.isfinite(_to_float(pick_prob_eff, np.nan)) and pd.notna(pick_odd) and pick_odd > 1.01:
                mispricing_pick = float(pick_prob_eff) - (1.0 / pick_odd)

        base_market = best.get("base_market", "")
        base_prob = best.get("base_prob", np.nan)
        base_score = best.get("base_score", np.nan)
        switched = best.get("switched", 0)
        switch_reason = best.get("switch_reason", "")
        sim_bonus = best.get("sim_bonus", np.nan)
        mis_bonus = best.get("mis_bonus", np.nan)
        if not base_market:
            base_pick = pick_best_market(r, params=params, debug_context=ctx)
            if base_pick:
                base_market = base_pick.get("base_market", base_pick.get("market", ""))
                base_prob = base_pick.get("base_prob", base_pick.get("prob", np.nan))
                base_score = base_pick.get("base_score", base_pick.get("score", np.nan))
                switched = int(base_market != best.get("market"))
                switch_reason = base_pick.get("switch_reason", "")
                if not np.isfinite(_to_float(sim_bonus, np.nan)):
                    sim_bonus = base_pick.get("sim_bonus", np.nan)
                if not np.isfinite(_to_float(mis_bonus, np.nan)):
                    mis_bonus = base_pick.get("mis_bonus", np.nan)
        if switched and not switch_reason:
            sim_bonus = _to_float(sim_bonus, np.nan)
            mis_bonus = _to_float(mis_bonus, np.nan)
            if np.isfinite(sim_bonus) and sim_bonus > 0.001:
                switch_reason = "SIM_GAP"
            elif np.isfinite(mis_bonus) and mis_bonus > 0.001:
                switch_reason = "MISPRICING"
            else:
                switch_reason = "OTHER"

        mw = best.get("market_weight", None)
        if mw is None and isinstance(cand_final, dict):
            mw = cand_final.get("market_weight", None)
        if mw is None:
            mw = 1.0

        misb = best.get("mis_bonus", None)
        if misb is None and isinstance(cand_final, dict):
            misb = cand_final.get("mis_bonus", None)
        if misb is None:
            misb = 0.0

        simb = best.get("sim_bonus", None)
        if simb is None and isinstance(cand_final, dict):
            simb = cand_final.get("sim_bonus", None)
        if simb is None:
            simb = 0.0

        bonus_total = best.get("bonus_total", None)
        if bonus_total is None and isinstance(cand_final, dict):
            bonus_total = cand_final.get("bonus_total", None)
        if bonus_total is None:
            bonus_total = float(np.clip(_to_float(misb, 0.0), 0.0, 0.25) + np.clip(_to_float(simb, 0.0), 0.0, 0.25))

        mw_source = "best"
        if best.get("market_weight", None) is None and isinstance(cand_final, dict) and cand_final.get("market_weight", None) is not None:
            mw_source = "cand_final"
        if (best.get("market_weight", None) is None) and (not isinstance(cand_final, dict) or cand_final.get("market_weight", None) is None):
            mw_source = "default"

        # group fallback (best -> cand_final -> "")
        pick_group = best.get("final_group", None)
        if not pick_group:
            pick_group = best.get("group", None)
        if (not pick_group) and isinstance(cand_final, dict):
            pick_group = cand_final.get("final_group") or cand_final.get("group") or cand_final.get("pick_group")
        if not pick_group:
            pick_group = ""

        # score_total fallback (best -> cand_final -> NaN)
        score_total = best.get("score_total", None)
        if score_total is None and isinstance(cand_final, dict):
            score_total = cand_final.get("score_total", None)
            if score_total is None:
                score_total = cand_final.get("score", None)

        mw_implied = np.nan
        if np.isfinite(_to_float(pick_prob_adj, np.nan)) and np.isfinite(_to_float(bonus_total, np.nan)) and np.isfinite(_to_float(score_total, np.nan)):
            denom = _to_float(pick_prob_adj, np.nan) * (1.0 + _to_float(bonus_total, 0.0))
            if np.isfinite(denom) and denom != 0:
                mw_implied = _to_float(score_total, np.nan) / denom

        rec = {

            "Match_ID": r.get("Match_ID", ""),
            "Date": r.get("Date"),

            "League": r.get("League"),

            "HomeTeam": r.get("HomeTeam"),

            "AwayTeam": r.get("AwayTeam"),

            "PICK": final_market,

            "PICK_PROB": pick_prob_base,
            "PICK_PROB_EFF": pick_prob_eff,
            "PICK_PROB_ADJ": pick_prob_adj,
            "SEL_ALPHA": sel_alpha,
            "SEL_SIM": sel_sim,
            "SEL_SCORE": sel_score,
            "SEL_P_ADJ": sel_p_adj,
            "SEL_MISPRICING": sel_mispricing,
            "SEL_MIS_BONUS": sel_mis_bonus,
            "TOP1_MKT": top1_mkt,
            "TOP1_SCORE": top1_score,
            "TOP1_PROB_EFF": top1_prob_eff,
            "TOP2_MKT": top2_mkt,
            "TOP2_SCORE": top2_score,
            "TOP2_PROB_EFF": top2_prob_eff,
            "TOP3_MKT": top3_mkt,
            "TOP3_SCORE": top3_score,
            "TOP3_PROB_EFF": top3_prob_eff,
            "HAS_CANDS": int(has_cands),
            "CANDS_N": cands_n,
            "FOUND_FINAL": found_final,
            "FINAL_MKT": final_market,
            "CANDS_MKTS": ", ".join([m for m in cands_mkts if m]),
            "CANDS_REASON": cands_reason,

            "PICK_ODD": pick_odd,

            "PICK_MARGIN": best.get("pick_margin"),

            "TRUST_PCT": trust,

            "BestOfRank": r.get("BestOfRank"),

            "SCORE_LAM_H": r.get("SCORE_LAM_H"),

            "SCORE_LAM_A": r.get("SCORE_LAM_A"),

            "SCORE_MODE": r.get("SCORE_MODE"),

            "SCORE_TOP3": r.get("SCORE_TOP3"),

            "SCENARIO_TITLE": r.get("SCENARIO_TITLE"),

            "SCENARIO_GROUP": r.get("SCENARIO_GROUP"),

            "XG_WEIGHT_CFG": r.get("XG_WEIGHT_CFG", r.get("XG_BIAS_W", np.nan)),
            "XG_BIAS_W": r.get("XG_BIAS_W", np.nan),
            "XG_MICRO_BIAS_HOME": r.get("XG_MICRO_BIAS_HOME", np.nan),
            "XG_MICRO_BIAS_DRAW": r.get("XG_MICRO_BIAS_DRAW", np.nan),
            "XG_MICRO_BIAS_AWAY": r.get("XG_MICRO_BIAS_AWAY", np.nan),
            "XG_MICRO_BIAS_OVER25": r.get("XG_MICRO_BIAS_OVER25", np.nan),
            "XG_MICRO_BIAS_BTTS": r.get("XG_MICRO_BIAS_BTTS", np.nan),

            "P_OVER25_ADJ": p_over_adj,
            "P_BTTSY_ADJ": p_btts_adj,
            "MISPRICING_PICK": mispricing_pick,

            "SIM_IMP_HOME": r.get("SIM_IMP_HOME", np.nan),
            "SIM_IMP_DRAW": r.get("SIM_IMP_DRAW", np.nan),
            "SIM_IMP_AWAY": r.get("SIM_IMP_AWAY", np.nan),
            "SIM_IMP_OVER25": r.get("SIM_IMP_OVER25", np.nan),
            "SIM_IMP_UNDER25": r.get("SIM_IMP_UNDER25", np.nan),
            "SIM_IMP_BTTS_YES": r.get("SIM_IMP_BTTS_YES", np.nan),
            "SIM_IMP_BTTS_NO": r.get("SIM_IMP_BTTS_NO", np.nan),

            "PICK_GROUP": pick_group,
            "PICK_BASE": base_market,
            "PICK_FINAL": best.get("final_market", best.get("market", "")),
            "PICK_SWITCHED": switched,
            "PICK_SCORE_TOTAL": _to_float(score_total, np.nan),
            "PICK_MIS_BONUS": misb,
            "PICK_SIM_BONUS": simb,
            "BONUS_TOTAL": bonus_total,
            "MARKET_WEIGHT": mw,
            "MW_SOURCE": mw_source,
            "MW_IMPLIED": mw_implied,
            "SWITCH_REASON": switch_reason,

        }
        rec["_candidates"] = cands if has_cands else []

        flag, rule = _policy_flag(rec, policy_ctx)
        rec["POLICY_FLAG"] = int(flag)
        rec["POLICY_RULE"] = rule

        _has = int(rec.get("HAS_CANDS", 0) or 0)
        _cn = int(rec.get("CANDS_N", 0) or 0)
        _cr = str(rec.get("CANDS_REASON", "") or "")
        _sr = str(rec.get("SWITCH_REASON", "") or "")
        if (_has == 0 or _cn == 0 or _cr == "CANDS_EMPTY") and _sr.startswith("FAIR_OVERRIDE_"):
            _fill_sel_defaults_for_no_cands(rec, reason="CANDS_EMPTY/FAIR_OVERRIDE_GATED")

        pb = str(rec.get("PICK_BASE", "") or "")
        pf = str(rec.get("PICK_FINAL", "") or rec.get("PICK", "") or "")
        rec["PICK_FINAL"] = pf
        if pb and pf and pb != pf:
            rec["PICK_SWITCHED"] = 1
            if not str(rec.get("SWITCH_REASON", "") or ""):
                rec["SWITCH_REASON"] = "OVERRIDE_SWITCH_NO_CANDS"
        else:
            rec["PICK_SWITCHED"] = int(rec.get("PICK_SWITCHED", 0) or 0)

        rows.append(rec)



    pred = pd.DataFrame(rows)



    # --- kolon garantisi (UI PRED_COLS patlamasin) ---

    must = {

        "PICK": None,

        "PICK_PROB": np.nan,

        "PICK_ODD": np.nan,

        "PICK_MARGIN": np.nan,

        "TRUST_PCT": np.nan,

        "BestOfRank": np.nan,

        "SCORE_LAM_H": np.nan,

        "SCORE_LAM_A": np.nan,

        "SCORE_MODE": "",

        "SCORE_TOP3": "",

        "SCENARIO_TITLE": "",

        "SCENARIO_GROUP": "",

    }

    for c, d in must.items():

        if c not in pred.columns:

            pred[c] = d



    if not pred.empty:

        pred = pred.sort_values(["TRUST_PCT", "PICK_PROB"], ascending=False).reset_index(drop=True)

    return pred



# -----------------------------------------------------------------------------#

# Pipeline and Render

# -----------------------------------------------------------------------------#

def run_pipeline(past_file, future_file, params):

    debug_dict = {"stage": "start"}

    try:

        debug_dict["stage"] = "load_csv"

        raw_past = DataEngine.load_csv(past_file)

        raw_future = DataEngine.load_csv(future_file)

        debug_dict["stage"] = "standardize"

        past = DataEngine.standardize(raw_past, is_past=True)

        future = DataEngine.standardize(raw_future, is_past=False)

        # Debug: Check which odds columns are present in future

        odds_cols = ["HomeOdd", "DrawOdd", "AwayOdd", "Over25Odd", "Under25Odd", "BTTSYesOdd", "BTTSNoOdd"]

        present_odds = [col for col in odds_cols if col in future.columns]

        missing_odds = [col for col in odds_cols if col not in future.columns]

        debug_dict["future_odds_present"] = present_odds

        debug_dict["future_odds_missing"] = missing_odds

        debug_dict["stage"] = "fillna_xg"

        # Fillna for mandatory xG in past and future

        warnings = []

        for col in ("xG_Home", "xG_Away"):

            if col in past.columns:

                null_count_past = past[col].isna().sum()

                if null_count_past > 0:

                    median_past = past[col].median()

                    fill_val = median_past if not pd.isna(median_past) else 0.0

                    past[col] = past[col].fillna(fill_val)

                    warnings.append(f"Past {col}: {null_count_past} nulls filled with {fill_val}")

            if col in future.columns:

                null_count_future = future[col].isna().sum()

                if null_count_future > 0:

                    median_future = future[col].median()

                    fill_val = median_future if not pd.isna(median_future) else 0.0

                    future[col] = future[col].fillna(fill_val)

                    warnings.append(f"Future {col}: {null_count_future} nulls filled with {fill_val}")

        debug_dict["stage"] = "duplicates"

        duplicate_past = len(past) - past["Match_ID"].nunique()

        duplicate_future = len(future) - future["Match_ID"].nunique()

        debug_dict["stage"] = "feature_metadata"

        feature_cols, drop_records, xg_info, null_info = build_feature_metadata(past, future, params["null_threshold"], params.get("manual_features"))

        debug_dict["stage"] = "feature_weights"

        feature_weights = build_feature_weights(feature_cols, params["weight_config"])

        debug_dict["stage"] = "knn_init"

        knn = KNNEngine(

            past,

            feature_cols=feature_cols,

            feature_weights=feature_weights,

            k_same=params["k_same"],

            k_global=params["k_global"],

            min_same_found=params["min_same_found"],

            same_league_mode=params["same_league_mode"],

            conf_quality_floor=params["conf_quality_floor"],

        )

        debug_dict["stage"] = "analyze_matches"

        rows = []
        league_profile_cache = {}

        for idx, (_, frow) in enumerate(future.iterrows()):

            stats, sim_df = knn.analyze_match(frow)

            st.session_state.setdefault("_sim_cache", {})[frow["Match_ID"]] = sim_df

            league_name = frow.get("League", "")
            lp = league_profile_cache.get(league_name)
            if lp is None:
                lp = compute_league_profile(past, target_league=league_name)
                league_profile_cache[league_name] = lp

            rows.append({

                "Future_Index": idx,

                "Match_ID": frow["Match_ID"],

                "Date": frow["Date"],

                "League": frow["League"],

                "HomeTeam": frow["HomeTeam"],

                "AwayTeam": frow["AwayTeam"],

                "HomeOdd": frow.get("HomeOdd", np.nan),

                "DrawOdd": frow.get("DrawOdd", frow.get("Odds_Open_Draw", np.nan)),

                "AwayOdd": frow.get("AwayOdd", np.nan),

                "Over25Odd": frow.get("Over25Odd", np.nan),

                "Under25Odd": frow.get("Under25Odd", np.nan),

                "BTTSYesOdd": frow.get("BTTSYesOdd", np.nan),

                "BTTSNoOdd": frow.get("BTTSNoOdd", np.nan),

                "Odds_Moves_Raw": frow.get("Odds_Moves_Raw", np.nan),

                "Manager_Games_Home": frow.get("Manager_Games_Home", np.nan),

                "Manager_Games_Away": frow.get("Manager_Games_Away", np.nan),

                "Sim_Home%": stats["Sim_Home_Win"],

                "Sim_Draw%": stats["Sim_Draw"],

                "Sim_Away%": stats["Sim_Away_Win"],

                "Sim_Over25%": stats["Sim_Over25"],

                "Sim_Under25%": stats["Sim_Under25"],

                "Sim_BTTS_Yes%": stats["Sim_BTTS_Yes"],

                "Sim_BTTS_No%": stats["Sim_BTTS_No"],

                "Avg_Similarity": stats["Avg_Similarity"],

                "SIM_QUALITY": stats["SIM_QUALITY"],

                "EFFECTIVE_N": stats["EFFECTIVE_N"],

                "CONF": stats["CONF"],

                "CONF_RAW": stats["CONF_RAW"],

                "CONF_OK": stats["CONF_OK"],

                "CONF_REASON": stats.get("CONF_REASON", "OK"),

                "TARGET_LEAGUE_GOALS_AVG_10": lp.get("goals_avg_10", np.nan),

                "TARGET_LEAGUE_GOALS_AVG_50": lp.get("goals_avg_50", np.nan),

                "XG_Home": stats.get("XG_Home", np.nan),

                "XG_Away": stats.get("XG_Away", np.nan),

                "XG_Total": stats.get("XG_Total", np.nan),

                "XG_WEIGHT_CFG": stats.get("XG_WEIGHT_CFG", stats.get("XG_BIAS_W", np.nan)),
                "XG_BIAS_W": stats.get("XG_BIAS_W", np.nan),
                "XG_MICRO_BIAS_HOME": stats.get("XG_MICRO_BIAS_HOME", np.nan),
                "XG_MICRO_BIAS_DRAW": stats.get("XG_MICRO_BIAS_DRAW", np.nan),
                "XG_MICRO_BIAS_AWAY": stats.get("XG_MICRO_BIAS_AWAY", np.nan),
                "XG_MICRO_BIAS_OVER25": stats.get("XG_MICRO_BIAS_OVER25", np.nan),
                "XG_MICRO_BIAS_BTTS": stats.get("XG_MICRO_BIAS_BTTS", np.nan),

                "SCORE_LAM_H": stats.get("SCORE_LAM_H", np.nan),

                "SCORE_LAM_A": stats.get("SCORE_LAM_A", np.nan),

                "SCORE_MODE": stats.get("SCORE_MODE", ""),

                "SCORE_TOP3": stats.get("SCORE_TOP3", ""),

                "SCORE_RES_RH": stats.get("SCORE_RES_RH", np.nan),

                "SCORE_RES_RA": stats.get("SCORE_RES_RA", np.nan),

                "SCORE_RES_EFFN": stats.get("SCORE_RES_EFFN", np.nan),

                "SCORE_RES_K": stats.get("SCORE_RES_K", np.nan),

                "SIM_IMP_HOME": stats.get("SIM_IMP_HOME", np.nan),
                "SIM_IMP_DRAW": stats.get("SIM_IMP_DRAW", np.nan),
                "SIM_IMP_AWAY": stats.get("SIM_IMP_AWAY", np.nan),
                "SIM_IMP_OVER25": stats.get("SIM_IMP_OVER25", np.nan),
                "SIM_IMP_UNDER25": stats.get("SIM_IMP_UNDER25", np.nan),
                "SIM_IMP_BTTS_YES": stats.get("SIM_IMP_BTTS_YES", np.nan),
                "SIM_IMP_BTTS_NO": stats.get("SIM_IMP_BTTS_NO", np.nan),

                "SCENARIO_ID": stats.get("SCENARIO_ID", "DEFAULT"),

                "SCENARIO_GROUP": stats.get("SCENARIO_GROUP", "MIXED"),

                "SCENARIO_TITLE": stats.get("SCENARIO_TITLE", ""),

                "SCENARIO_TEXT": stats.get("SCENARIO_TEXT", ""),

                "SCENARIO_EVIDENCE": stats.get("SCENARIO_EVIDENCE", ""),

                "same_league_count": stats["same_league_count"],

                "global_count": stats["global_count"],

                "N": stats["Match_Count"],

            })

        debug_dict["stage"] = "build_res"

        res = pd.DataFrame(rows)



        # --- Guarantee required columns always exist (prevents KeyError in UI/export) ---

        _must_cols_defaults = {

            "CONF_REASON": "OK",

            "CONF_OK": True,

            "CONF_RAW": np.nan,

            "SIM_QUALITY": np.nan,

            "EFFECTIVE_N": np.nan,

            "Avg_Similarity": np.nan,

            "N": 0,

            "same_league_count": 0,

            "global_count": 0,

            "SCENARIO_ID": "DEFAULT",

            "SCENARIO_GROUP": "MIXED",

            "SCENARIO_TITLE": "",

            "SCENARIO_TEXT": "",

            "SCENARIO_EVIDENCE": "",

        }

        for _c, _default in _must_cols_defaults.items():

            if _c not in res.columns:

                res[_c] = _default

        # If exists but has NaN -> fill

        res["CONF_REASON"] = res["CONF_REASON"].fillna("OK")

        if "CONF_OK" in res.columns:

            res["CONF_OK"] = res["CONF_OK"].fillna(False)



        res["BestOfScore"] = res.apply(bestof_score, axis=1)



        # PROFILE_CONF calculation

        res["PROFILE_CONF"] = res.apply(lambda row: _clamp(100 * (

            0.6 * (row["SIM_QUALITY"] / 0.30) +

            0.4 * min(row["EFFECTIVE_N"] / 25.0, 1)

        ) * (0.85 if row["same_league_count"] < params["min_same_found"] else 1), 0, 100), axis=1)



        # TRUST_PCT calculation

        sim_norm = res["SIM_QUALITY"].fillna(0.0).apply(lambda x: clamp(x / 0.30, 0, 1)) if "SIM_QUALITY" in res.columns else pd.Series(0.0, index=res.index)

        eff_norm = res["EFFECTIVE_N"].fillna(0.0).apply(lambda x: clamp(np.log1p(x) / np.log1p(25.0), 0, 1)) if "EFFECTIVE_N" in res.columns else pd.Series(0.0, index=res.index)

        conf_norm = res["CONF"].fillna(0.0).apply(lambda x: clamp(x / 100.0, 0, 1)) if "CONF" in res.columns else pd.Series(0.0, index=res.index)

        base_trust = 100.0 * (0.40 * conf_norm + 0.35 * sim_norm + 0.25 * eff_norm)

        res["TRUST_PCT"] = base_trust



        if "Odds_Moves_Raw" in res.columns:

            res["Market_Conf"] = res["Odds_Moves_Raw"].apply(market_confidence)

        else:

            res["Market_Conf"] = 0.0



        mg_h = res["Manager_Games_Home"] if "Manager_Games_Home" in res.columns else pd.Series(0.0, index=res.index)

        mg_a = res["Manager_Games_Away"] if "Manager_Games_Away" in res.columns else pd.Series(0.0, index=res.index)



        res["TRUST_PCT"] = res["TRUST_PCT"] + (res["Market_Conf"] * 20.0).clip(-5.0, 5.0)

        res["TRUST_PCT"] = res["TRUST_PCT"] + mg_h.apply(manager_penalty) + mg_a.apply(manager_penalty)

        res["TRUST_PCT"] = res["TRUST_PCT"].clip(0.0, 100.0).round(1)



        # BestOfRank

        res["BestOfRank"] = (

            res["BestOfScore"]

            .rank(pct=True)

            .mul(100)

            .round(1)

        )



        res = res.sort_values("BestOfScore", ascending=False).reset_index(drop=True)

        debug_dict["stage"] = "simq_calc"

        simq = res["SIM_QUALITY"].replace([np.inf, -np.inf], np.nan).dropna()

        if len(simq) > 0:

            suggested_conf_floor = float(simq.quantile(0.60))

        else:

            suggested_conf_floor = params["conf_quality_floor"]

        debug_dict["stage"] = "final_debug"

        debug_dict.update({

            "feature_cols": feature_cols,

            "drop_records": drop_records,

            "null_info": null_info,

            "xg_info": xg_info,

            "duplicate_match_id_past": duplicate_past,

            "duplicate_match_id_future": duplicate_future,

            "chosen_params": params,

            "feature_weights": feature_weights,

            "warnings": warnings,

            "conf_floor": params["conf_quality_floor"],

            "conf_zero_rate": float((res["CONF"] <= 0.0).mean()),

            "conf_reason_counts": res["CONF_REASON"].value_counts().to_dict(),

            "sim_quality_summary": res["SIM_QUALITY"].describe().to_dict(),

            "conf_floor_used": params["conf_quality_floor"],

            "suggested_conf_floor": suggested_conf_floor,

            "sim_quality_q": {

                "q30": float(simq.quantile(0.30)) if len(simq) else None,

                "q50": float(simq.quantile(0.50)) if len(simq) else None,

                "q60": float(simq.quantile(0.60)) if len(simq) else None,

                "q70": float(simq.quantile(0.70)) if len(simq) else None,

            },

        })

        debug_dict["stage"] = "predictions"

        picks = []

        for _, row in res.iterrows():

            best = pick_best_market(row, params=params, debug_context=debug_dict)

            rec = {

                "Date": row["Date"],

                "League": row["League"],

                "HomeTeam": row["HomeTeam"],

                "AwayTeam": row["AwayTeam"],

                "PICK_MARKET": best["market"] if isinstance(best, dict) else "",

                "PICK_LABEL": best["market"] if isinstance(best, dict) else "",

                "PICK_ODD": best["odd"] if isinstance(best, dict) else np.nan,

                "PICK_PROB": best["prob"] if isinstance(best, dict) else np.nan,

                "PICK_EV": best["ev"] if isinstance(best, dict) else np.nan,

                "SIM_QUALITY": row["SIM_QUALITY"],

                "EFFECTIVE_N": row["EFFECTIVE_N"],

                "CONF": row["CONF"],

                "CONF_REASON": row["CONF_REASON"],

            }
            rec["_candidates"] = best.get("_candidates", []) if isinstance(best, dict) else []
            rec["_cands_reason"] = best.get("_cands_reason", "") if isinstance(best, dict) else "BEST_NONE"
            picks.append(rec)

        pred_df = pd.DataFrame(picks)
        try:
            if isinstance(res, pd.DataFrame) and len(res) == len(pred_df):
                res = res.copy()
                res["_candidates"] = pred_df["_candidates"].tolist()
                res["_cands_reason"] = pred_df["_cands_reason"].tolist()
        except Exception:
            pass

        debug_dict["stage"] = "done"

        pred_df = build_prediction_table(res)

        return pred_df, res, knn, past, future, debug_dict

    except Exception as e:

        debug_dict["error"] = str(e)

        debug_dict["stage"] = f"failed_at_{debug_dict.get('stage', 'unknown')}"

        raise



def render_tabs(pred_df, res, knn, past, future, debug):

    # Tahmin tablosu = sade. Dump ayri.

    PRED_COLS = [

        "Date","League","POLICY_ICON","Hitrate","HomeTeam","AwayTeam",

        "PICK_FINAL","PICK_ODD","PICK_PROB","TRUST_PCT",

        "Score_Exp","SCORE_MODE","SCORE_TOP3",

        "SCENARIO_TITLE",

        "BestOfRank",

    ]
    if st.session_state.get("DEBUG_MODE", False):
        PRED_COLS = ["Match_ID"] + PRED_COLS

    PRED_FMT = {"PICK_ODD":"{:.2f}", "PICK_PROB":"{:.1%}", "TRUST_PCT":"{:.1f}", "BestOfRank":"{:.0f}"}



    # Full dump kolon seti (varsa göster)

    show_cols = [

        "Match_ID", "Date", "League", "Hitrate", "HomeTeam", "AwayTeam",
        "POLICY_FLAG", "POLICY_RULE",

        "HomeOdd", "DrawOdd", "AwayOdd",

        "Over25Odd", "Under25Odd", "BTTSYesOdd", "BTTSNoOdd",

        "Sim_Home%", "Sim_Draw%", "Sim_Away%",

        "Sim_Over25%", "Sim_Under25%",

        "Sim_BTTS_Yes%", "Sim_BTTS_No%",

        "Avg_Similarity", "SIM_QUALITY", "EFFECTIVE_N", "TRUST_PCT",

        "same_league_count", "global_count", "BestOfScore",

        "P_OVER25_ADJ", "P_BTTSY_ADJ", "MISPRICING_PICK",
        "XG_WEIGHT_CFG",
        "XG_MICRO_BIAS_HOME", "XG_MICRO_BIAS_DRAW", "XG_MICRO_BIAS_AWAY",
        "XG_MICRO_BIAS_OVER25", "XG_MICRO_BIAS_BTTS",
        "SCORE_LAM_H", "SCORE_LAM_A",

    ]

    show_cols = [c for c in show_cols if isinstance(res, pd.DataFrame) and c in res.columns]



    # ? format_map önce tanimlanmali (senin hatanin ana kaynagi buydu)

    format_map = {

        "HomeOdd": "{:.2f}",

        "DrawOdd": "{:.2f}",

        "AwayOdd": "{:.2f}",

        "Over25Odd": "{:.2f}",

        "Under25Odd": "{:.2f}",

        "BTTSYesOdd": "{:.2f}",

        "BTTSNoOdd": "{:.2f}",

        "Sim_Home%": "{:.1%}",

        "Sim_Draw%": "{:.1%}",

        "Sim_Away%": "{:.1%}",

        "Sim_Over25%": "{:.1%}",

        "Sim_Under25%": "{:.1%}",

        "Sim_BTTS_Yes%": "{:.1%}",

        "Sim_BTTS_No%": "{:.1%}",

        "Avg_Similarity": "{:.3f}",

        "SIM_QUALITY": "{:.3f}",

        "EFFECTIVE_N": "{:.2f}",

        "TRUST_PCT": "{:.1f}",

        "BestOfScore": "{:.1f}",

        "same_league_count": "{:.0f}",

        "global_count": "{:.0f}",

        "P_OVER25_ADJ": "{:.1%}",
        "P_BTTSY_ADJ": "{:.1%}",
        "MISPRICING_PICK": "{:+.2%}",
        "XG_WEIGHT_CFG": "{:.2f}",
        "XG_MICRO_BIAS_HOME": "{:+.3f}",
        "XG_MICRO_BIAS_DRAW": "{:+.3f}",
        "XG_MICRO_BIAS_AWAY": "{:+.3f}",
        "XG_MICRO_BIAS_OVER25": "{:+.3f}",
        "XG_MICRO_BIAS_BTTS": "{:+.3f}",

    }

    # ? sadece mevcut kolonlarin formatini uygula

    format_map = {k: v for k, v in format_map.items() if k in show_cols}



    tab1, tab2, tab3, tab4 = st.tabs(["📊 Tahmin Tablosu", "🔎 Maç Detayı", "🧩 Şema & Debug", "📈 Tahmin Sonuçları"])



    if isinstance(res, pd.DataFrame) and not res.empty:

        with tab1:

            st.subheader("Tahmin Tablosu")




            _pred = pred_df.copy() if isinstance(pred_df, pd.DataFrame) else pd.DataFrame()

            # kolon fallback + garantisi (disaridan bozuk gelirse bile patlamasin)
            if "PICK_FINAL" not in _pred.columns and "PICK" in _pred.columns:
                _pred["PICK_FINAL"] = _pred["PICK"]

            for c in PRED_COLS:
                if c not in _pred.columns:
                    _pred[c] = np.nan
            if "POLICY_FLAG" not in _pred.columns:
                _pred["POLICY_FLAG"] = 0
            if "POLICY_RULE" not in _pred.columns:
                _pred["POLICY_RULE"] = ""

            if "PICK" in _pred.columns:
                if ("PICK_FINAL" not in _pred.columns) or _pred["PICK_FINAL"].isna().all():
                    _pred["PICK_FINAL"] = _pred["PICK"]



            view = _pred.copy()

            # policy flag'i log'dan esitle (tek kaynak)
            if "log_df" not in locals():
                try:
                    log_path = os.path.join(_get_base_dir(), "pred_results_log.csv")
                    log_df = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
                except Exception:
                    log_df = pd.DataFrame()
            if (
                isinstance(log_df, pd.DataFrame)
                and not log_df.empty
                and "Match_ID" in log_df.columns
                and "Match_ID" in view.columns
            ):
                pm = log_df.set_index("Match_ID")[["POLICY_FLAG", "POLICY_RULE"]]
                if "POLICY_FLAG" in view.columns:
                    view["POLICY_FLAG"] = view["Match_ID"].map(pm["POLICY_FLAG"]).fillna(view["POLICY_FLAG"])
                if "POLICY_RULE" in view.columns:
                    view["POLICY_RULE"] = view["Match_ID"].map(pm["POLICY_RULE"]).fillna(view["POLICY_RULE"])

            if isinstance(view, pd.DataFrame) and not view.empty:
                view["POLICY_ICON"] = np.where(view["POLICY_FLAG"].fillna(0) == 1, "\u00A0\u00A0✨\u00A0\u00A0", "")

                if "SCORE_LAM_H" in view.columns and "SCORE_LAM_A" in view.columns:

                    h = pd.to_numeric(view["SCORE_LAM_H"], errors="coerce").round(2)

                    a = pd.to_numeric(view["SCORE_LAM_A"], errors="coerce").round(2)

                    view["Score_Exp"] = h.astype(str) + "-" + a.astype(str)

                else:

                    view["Score_Exp"] = ""
                # PICK_FINAL rozetleri kaldirildi (sadece policy ikonu kalacak)

            # --- Policy Summary Cards (non-interactive) ---
            try:
                log_path = os.path.join(_get_base_dir(), "pred_results_log.csv")
                log_df = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
            except Exception:
                log_df = pd.DataFrame()

            # Keep policy exports in sync with log (dynamic refresh).
            if isinstance(log_df, pd.DataFrame) and not log_df.empty:
                try:
                    out_dir = os.path.join(_get_base_dir(), "Backtest-Out")
                    os.makedirs(out_dir, exist_ok=True)
                    _lg = log_df.dropna(subset=["RES_FTHG", "RES_FTAG"]).copy()

                    # league_picktype_matrix.csv
                    if {"League", "PICK_FINAL"}.issubset(_lg.columns):
                        _lg["HIT"] = np.nan
                        if "PICK_GROUP" in _lg.columns:
                            def _norm_1x2(v):
                                s = str(v or "")
                                return {"HOME": "MS1", "AWAY": "MS2", "DRAW": "X"}.get(s, s)
                            def _norm_btts(v):
                                s = str(v or "")
                                return {"BTTS Yes": "BTTS_Yes", "BTTS No": "BTTS_No"}.get(s, s)
                            if "RES_RESULT_1X2" in _lg.columns:
                                _lg.loc[_lg["PICK_GROUP"] == "1X2", "HIT"] = (
                                    _lg["PICK_FINAL"] == _lg["RES_RESULT_1X2"].map(_norm_1x2)
                                )
                            if "RES_RESULT_OU" in _lg.columns:
                                _lg.loc[_lg["PICK_GROUP"] == "OU", "HIT"] = (
                                    _lg["PICK_FINAL"] == _lg["RES_RESULT_OU"]
                                )
                            if "RES_RESULT_BTTS" in _lg.columns:
                                _lg.loc[_lg["PICK_GROUP"] == "BTTS", "HIT"] = (
                                    _lg["PICK_FINAL"] == _lg["RES_RESULT_BTTS"].map(_norm_btts)
                                )
                        _m = _lg[_lg["HIT"].notna()]
                        if not _m.empty:
                            matrix = (
                                _m.groupby(["League", "PICK_FINAL"])["HIT"]
                                .agg(["count", "mean"])
                                .reset_index()
                                .rename(columns={"mean": "hit"})
                            )
                            matrix.to_csv(os.path.join(out_dir, "league_picktype_matrix.csv"), index=False)

                    # policy_bucket_stats.csv
                    if {"League", "PICK_FINAL"}.issubset(_lg.columns):
                        stats = (
                            _lg[_lg["HIT"].notna()]
                            .groupby(["League", "PICK_FINAL"])["HIT"]
                            .mean()
                            .reset_index()
                        )
                        stats["key"] = stats["League"].astype(str) + "||" + stats["PICK_FINAL"].astype(str)
                        hit_map = dict(zip(stats["key"], stats["HIT"]))
                        _lg["key"] = _lg["League"].astype(str) + "||" + _lg["PICK_FINAL"].astype(str)
                        _lg["HIT_RATE"] = _lg["key"].map(hit_map)

                        bucket_bins = [0, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 1.0]
                        bucket_labels = ["0-14%", "15-28%", "29-42%", "43-56%", "57-70%", "71-84%", "85-100%"]
                        _lg["HIT_BUCKET"] = pd.cut(
                            _lg["HIT_RATE"], bins=bucket_bins, labels=bucket_labels, include_lowest=True, right=True
                        )
                        pf = _lg[_lg.get("POLICY_FLAG", 0) == 1]
                        if not pf.empty:
                            out = pf.groupby("HIT_BUCKET")["HIT"].agg(["count", "mean"]).reset_index()
                            out.to_csv(os.path.join(out_dir, "policy_bucket_stats.csv"), index=False)
                except Exception:
                    pass

            if isinstance(log_df, pd.DataFrame) and not log_df.empty:
                def _build_hit_df(df_in):
                    required = [
                        "RES_FTHG",
                        "RES_FTAG",
                        "RES_RESULT_1X2",
                        "RES_RESULT_OU",
                        "RES_RESULT_BTTS",
                    ]
                    if not all(c in df_in.columns for c in required):
                        return pd.DataFrame()
                    out = df_in.dropna(subset=required).copy()
                    if out.empty:
                        return out
                    out["PICK_RESULT_NORM"] = out.get("PICK_FINAL", out.get("PICK", ""))
                    out["HIT"] = np.nan

                    def _norm_1x2(v):
                        s = str(v or "")
                        return {"HOME": "MS1", "AWAY": "MS2", "DRAW": "X"}.get(s, s)

                    def _norm_btts(v):
                        s = str(v or "")
                        return {"BTTS Yes": "BTTS_Yes", "BTTS No": "BTTS_No"}.get(s, s)

                    out.loc[out["PICK_GROUP"] == "1X2", "HIT"] = (
                        out["PICK_RESULT_NORM"] == out["RES_RESULT_1X2"].map(_norm_1x2)
                    )
                    out.loc[out["PICK_GROUP"] == "OU", "HIT"] = (
                        out["PICK_RESULT_NORM"] == out["RES_RESULT_OU"]
                    )
                    out.loc[out["PICK_GROUP"] == "BTTS", "HIT"] = (
                        out["PICK_RESULT_NORM"] == out["RES_RESULT_BTTS"].map(_norm_btts)
                    )
                    return out

                hit_all = _build_hit_df(log_df)
                hit_all = hit_all[hit_all["HIT"].notna()].copy()
                if not hit_all.empty:
                    flag_all = hit_all[hit_all.get("POLICY_FLAG", 0) == 1]
                    total_cnt = len(hit_all)
                    total_hit = float(hit_all["HIT"].mean())
                    flag_cnt = len(flag_all)
                    flag_hit = float(flag_all["HIT"].mean()) if flag_cnt > 0 else np.nan

                    gen_grp_rows = []
                    grp_sum_all = hit_all.groupby("PICK_GROUP")["HIT"].agg(["count", "mean"]).reset_index()
                    for _, r in grp_sum_all.iterrows():
                        gen_grp_rows.append(f"{r['PICK_GROUP']} {r['mean']:.0%} (n={int(r['count'])})")

                    grp_rows = []
                    if flag_cnt > 0:
                        grp_sum = flag_all.groupby("PICK_GROUP")["HIT"].agg(["count", "mean"]).reset_index()
                        for _, r in grp_sum.iterrows():
                            grp_rows.append(f"{r['PICK_GROUP']} {r['mean']:.0%} (n={int(r['count'])})")

                    league_rows = []
                    if flag_cnt > 0 and "League" in flag_all.columns:
                        lg = flag_all.groupby("League")["HIT"].agg(["count", "mean"]).reset_index()
                        lg = lg[lg["count"] >= 3].sort_values("mean", ascending=False).head(5)
                        for _, r in lg.iterrows():
                            league_rows.append(f"{r['League']}: {r['mean']:.0%} (n={int(r['count'])})")

                    card_style = """
                    <style>
                    .policy-cards {display:flex; gap:10px; margin: 6px 0 12px 0;}
                    .policy-card {flex:1; padding:10px 12px; border-radius:12px; background: linear-gradient(135deg, #f7f2e8 0%, #fff7ea 100%); border:1px solid #eee2cc;}
                    .policy-card h4 {margin:0 0 6px 0; font-size:13px; color:#7b6a52; letter-spacing:0.2px;}
                    .policy-card .big {font-size:26px; font-weight:700; color:#2f2a1f; margin-bottom:2px;}
                    .policy-card .sub {font-size:12px; color:#8a7a66;}
                    .policy-card .grid {display:flex; gap:16px; align-items:flex-start;}
                    .policy-card .left {min-width:120px;}
                    .policy-card .right {flex:1;}
                    .policy-card .list {margin-top:4px; font-size:13px; color:#5b4d3a; line-height:1.45; display:grid; grid-template-columns: 1fr 1fr; gap:4px 12px;}
                    .policy-card .list span {display:block;}
                    .legend {display:grid; grid-template-columns: 1fr; gap:6px; margin-top:4px;}
                    .legend-row {display:flex; align-items:center; gap:8px; font-size:12px; color:#5b4d3a;}
                    .legend-chip {width:14px; height:14px; border-radius:3px; border:1px solid rgba(0,0,0,0.05);}
                    </style>
                    """
                    st.markdown(card_style, unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div class="policy-cards">
                          <div class="policy-card">
                            <h4>Genel Basari</h4>
                            <div class="grid">
                              <div class="left">
                                <div class="big">{total_hit:.0%}</div>
                                <div class="sub">Toplam basari (n={total_cnt})</div>
                              </div>
                              <div class="right">
                                <div class="sub">Pick group dagilimi</div>
                                <div class="list">{''.join([f'<span>{x}</span>' for x in gen_grp_rows]) if gen_grp_rows else 'Yok'}</div>
                              </div>
                            </div>
                          </div>
                          <div class="policy-card">
                            <h4>Policy Basari</h4>
                            <div class="grid">
                              <div class="left">
                                <div class="big">{(flag_hit if np.isfinite(flag_hit) else 0):.0%}</div>
                                <div class="sub">Policy basari (n={flag_cnt})</div>
                              </div>
                              <div class="right">
                                <div class="sub">Pick group dagilimi</div>
                                <div class="list">{''.join([f'<span>{x}</span>' for x in grp_rows]) if grp_rows else 'Group kirilimi yok'}</div>
                              </div>
                            </div>
                          </div>
                          <div class="policy-card">
                            <h4>En Iyi Ligler (Policy)</h4>
                            <div class="grid">
                              <div class="left">
                                <div class="big">{len(league_rows) if league_rows else 0}</div>
                                <div class="sub">Uygun lig sayisi</div>
                              </div>
                              <div class="right">
                                <div class="list">{''.join([f'<span>{x}</span>' for x in league_rows]) if league_rows else 'Yeterli veri yok'}</div>
                              </div>
                            </div>
                          </div>
                          <div class="policy-card">
                            <h4>Hit Rate Renkleri</h4>
                            <div class="legend">
                              <div class="legend-row"><span class="legend-chip" style="background:#f8d7da;"></span>0–14% · Zayıf</div>
                              <div class="legend-row"><span class="legend-chip" style="background:#f3b7b7;"></span>15–28% · Düşük</div>
                              <div class="legend-row"><span class="legend-chip" style="background:#f7d6b5;"></span>29–42% · Orta</div>
                              <div class="legend-row"><span class="legend-chip" style="background:#fff3cd;"></span>43–56% · Dengeli</div>
                              <div class="legend-row"><span class="legend-chip" style="background:#e2f0d9;"></span>57–70% · İyi</div>
                              <div class="legend-row"><span class="legend-chip" style="background:#cbe8c1;"></span>71–84% · Çok İyi</div>
                              <div class="legend-row"><span class="legend-chip" style="background:#b7e4c7;"></span>85–100% · Elit</div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Bucket summary table (general vs policy) from live log.
                    bucket_bins = [0, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 1.0]
                    bucket_labels = [
                        "0–14% · Zayıf",
                        "15–28% · Düşük",
                        "29–42% · Orta",
                        "43–56% · Dengeli",
                        "57–70% · İyi",
                        "71–84% · Çok İyi",
                        "85–100% · Elit",
                    ]
                    if {"League", "PICK_FINAL", "PICK_GROUP", "RES_FTHG", "RES_FTAG"}.issubset(log_df.columns):
                        lg = log_df.dropna(subset=["RES_FTHG", "RES_FTAG"]).copy()
                        if not lg.empty:
                            lg["PICK_RESULT_NORM"] = lg.get("PICK_FINAL", lg.get("PICK", ""))
                            lg["HIT"] = np.nan

                            def _norm_1x2(v):
                                s = str(v or "")
                                return {"HOME": "MS1", "AWAY": "MS2", "DRAW": "X"}.get(s, s)

                            def _norm_btts(v):
                                s = str(v or "")
                                return {"BTTS Yes": "BTTS_Yes", "BTTS No": "BTTS_No"}.get(s, s)

                            lg.loc[lg["PICK_GROUP"] == "1X2", "HIT"] = (
                                lg["PICK_RESULT_NORM"] == lg["RES_RESULT_1X2"].map(_norm_1x2)
                            )
                            lg.loc[lg["PICK_GROUP"] == "OU", "HIT"] = (
                                lg["PICK_RESULT_NORM"] == lg["RES_RESULT_OU"]
                            )
                            lg.loc[lg["PICK_GROUP"] == "BTTS", "HIT"] = (
                                lg["PICK_RESULT_NORM"] == lg["RES_RESULT_BTTS"].map(_norm_btts)
                            )

                            stats = (
                                lg[lg["HIT"].notna()]
                                .groupby(["League", "PICK_FINAL"])["HIT"]
                                .agg(["count", "mean"])
                                .reset_index()
                                .rename(columns={"mean": "hit"})
                            )
                            market_map = {
                                "MS1": "1X2",
                                "MS2": "1X2",
                                "X": "1X2",
                                "Over2.5": "OU",
                                "Under2.5": "OU",
                                "BTTS_Yes": "BTTS",
                                "BTTS_No": "BTTS",
                            }
                            stats["MARKET"] = stats["PICK_FINAL"].map(market_map).fillna("OTHER")
                            k_shrink = 3
                            global_by_market = (
                                stats.groupby("MARKET")
                                .apply(lambda g: np.average(g["hit"], weights=g["count"]))
                                .to_dict()
                            )
                            def _adj_hit(row):
                                gh = global_by_market.get(
                                    row["MARKET"],
                                    np.average(stats["hit"], weights=stats["count"]),
                                )
                                wins = row["hit"] * row["count"]
                                return (wins + k_shrink * gh) / (row["count"] + k_shrink)
                            stats["hit_adj_mkt"] = stats.apply(_adj_hit, axis=1)

                            stats["key"] = stats["League"].astype(str) + "||" + stats["PICK_FINAL"].astype(str)
                            hit_map = dict(zip(stats["key"], stats["hit_adj_mkt"]))
                            lg["key"] = lg["League"].astype(str) + "||" + lg["PICK_FINAL"].astype(str)
                            lg["HIT_RATE"] = lg["key"].map(hit_map)
                            lg["HIT_BUCKET"] = pd.cut(
                                lg["HIT_RATE"], bins=bucket_bins, labels=bucket_labels, include_lowest=True, right=True
                            )

                            def _bucket_rates(df_in):
                                out = df_in.groupby("HIT_BUCKET")["HIT"].agg(["count", "mean"]).reset_index()
                                out = out.set_index("HIT_BUCKET")
                                return out

                            all_b = _bucket_rates(lg[lg["HIT"].notna()])
                            pf = lg[(lg.get("POLICY_FLAG", 0) == 1) & (lg["HIT"].notna())]
                            pf_b = _bucket_rates(pf) if not pf.empty else pd.DataFrame()

                            rows = []
                            for lbl in bucket_labels:
                                g = all_b.loc[lbl, "mean"] if lbl in all_b.index else np.nan
                                p = pf_b.loc[lbl, "mean"] if (not pf_b.empty and lbl in pf_b.index) else np.nan
                                rows.append({"Aralık": lbl, "Genel Başarı": g, "Policy Başarı": p})

                            bucket_df = pd.DataFrame(rows)
                            bucket_df["Genel Başarı"] = bucket_df["Genel Başarı"].map(
                                lambda v: f"{v:.0%}" if pd.notna(v) else "-"
                            )
                            bucket_df["Policy Başarı"] = bucket_df["Policy Başarı"].map(
                                lambda v: f"{v:.0%}" if pd.notna(v) else "-"
                            )
                            bucket_color_map = {
                                "0–14% · Zayıf": "#f8d7da",
                                "15–28% · Düşük": "#f3b7b7",
                                "29–42% · Orta": "#f7d6b5",
                                "43–56% · Dengeli": "#fff3cd",
                                "57–70% · İyi": "#e2f0d9",
                                "71–84% · Çok İyi": "#cbe8c1",
                                "85–100% · Elit": "#b7e4c7",
                            }

                            def _bucket_row_bg(r):
                                color = bucket_color_map.get(r.get("Aralık"), "")
                                if not color:
                                    return [""] * len(bucket_df.columns)
                                return [f"background-color: {color}"] * len(bucket_df.columns)

                            st.dataframe(
                                bucket_df.style.apply(_bucket_row_bg, axis=1),
                                use_container_width=True,
                            )



            if view.empty:

                st.warning("Tahmin bulunamadi.")

            else:

                # Row background by league+pick hit rate (dynamic from log).
                try:
                    matrix_path = os.path.join(_get_base_dir(), "pred_results_log.csv")
                    if os.path.exists(matrix_path):
                        mtx = pd.read_csv(matrix_path)
                    else:
                        mtx = pd.DataFrame()
                except Exception:
                    mtx = pd.DataFrame()
                hit_map = {}
                if isinstance(mtx, pd.DataFrame) and not mtx.empty:
                    required = {"League", "PICK_GROUP", "PICK_FINAL", "RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS"}
                    if required.issubset(mtx.columns):
                        _m = mtx.dropna(subset=["RES_FTHG", "RES_FTAG"]).copy()
                        if not _m.empty:
                            def _norm_1x2(v):
                                s = str(v or "")
                                return {"HOME": "MS1", "AWAY": "MS2", "DRAW": "X"}.get(s, s)

                            def _norm_btts(v):
                                s = str(v or "")
                                return {"BTTS Yes": "BTTS_Yes", "BTTS No": "BTTS_No"}.get(s, s)

                            _m["PICK_FINAL_NORM"] = _m["PICK_FINAL"].replace({"MSX": "X"})
                            _m["HIT"] = np.nan
                            _m.loc[_m["PICK_GROUP"] == "1X2", "HIT"] = (
                                _m["PICK_FINAL_NORM"] == _m["RES_RESULT_1X2"].map(_norm_1x2)
                            )
                            _m.loc[_m["PICK_GROUP"] == "OU", "HIT"] = (
                                _m["PICK_FINAL"] == _m["RES_RESULT_OU"]
                            )
                            _m.loc[_m["PICK_GROUP"] == "BTTS", "HIT"] = (
                                _m["PICK_FINAL"] == _m["RES_RESULT_BTTS"].map(_norm_btts)
                            )
                            stats = (
                                _m[_m["HIT"].notna()]
                                .groupby(["League", "PICK_FINAL_NORM"])["HIT"]
                                .agg(["count", "mean"])
                                .reset_index()
                                .rename(columns={"mean": "hit"})
                            )
                            market_map = {
                                "MS1": "1X2",
                                "MS2": "1X2",
                                "X": "1X2",
                                "Over2.5": "OU",
                                "Under2.5": "OU",
                                "BTTS_Yes": "BTTS",
                                "BTTS_No": "BTTS",
                            }
                            stats["MARKET"] = stats["PICK_FINAL_NORM"].map(market_map).fillna("OTHER")
                            k_shrink = 3
                            global_by_market = (
                                stats.groupby("MARKET")
                                .apply(lambda g: np.average(g["hit"], weights=g["count"]))
                                .to_dict()
                            )
                            def _adj_hit(row):
                                gh = global_by_market.get(
                                    row["MARKET"],
                                    np.average(stats["hit"], weights=stats["count"]),
                                )
                                wins = row["hit"] * row["count"]
                                return (wins + k_shrink * gh) / (row["count"] + k_shrink)
                            stats["hit_adj_mkt"] = stats.apply(_adj_hit, axis=1)
                            keys = stats["League"].astype(str) + "||" + stats["PICK_FINAL_NORM"].astype(str)
                            hit_map = dict(zip(keys, pd.to_numeric(stats["hit_adj_mkt"], errors="coerce")))

                bucket_labels = [
                    ("Zayıf", 0.14),
                    ("Düşük", 0.28),
                    ("Orta", 0.42),
                    ("Dengeli", 0.56),
                    ("İyi", 0.70),
                    ("Çok İyi", 0.84),
                    ("Elit", 1.00),
                ]

                def _bucket_label(hr):
                    if not np.isfinite(hr):
                        return ""
                    for label, thr in bucket_labels:
                        if hr <= thr:
                            return label
                    return "Elit"

                def _row_bg(r):
                    pick_val = r.get("PICK_FINAL_RAW", r.get("PICK_FINAL", ""))
                    if str(pick_val).strip().upper() == "MSX":
                        pick_val = "X"
                    key = f"{r.get('League','')}||{pick_val}"
                    hr = hit_map.get(key, np.nan)
                    if not np.isfinite(hr):
                        return [""] * len(PRED_COLS)
                    if hr <= 0.14:
                        color = "#f8d7da"
                    elif hr <= 0.28:
                        color = "#f3b7b7"
                    elif hr <= 0.42:
                        color = "#f7d6b5"
                    elif hr <= 0.56:
                        color = "#fff3cd"
                    elif hr <= 0.70:
                        color = "#e2f0d9"
                    elif hr <= 0.84:
                        color = "#cbe8c1"
                    else:
                        color = "#b7e4c7"
                    return [f"background-color: {color}"] * len(PRED_COLS)

                if hit_map:
                    if "PICK_FINAL_RAW" not in view.columns and "PICK_FINAL" in view.columns:
                        view["PICK_FINAL_RAW"] = view["PICK_FINAL"]
                    if "PICK_FINAL_RAW" in view.columns:
                        view["PICK_FINAL_RAW"] = view["PICK_FINAL_RAW"].replace({"MSX": "X"})
                    view["Hitrate"] = view.apply(
                        lambda r: _bucket_label(hit_map.get(f"{r.get('League','')}||{r.get('PICK_FINAL_RAW', r.get('PICK_FINAL',''))}", np.nan)),
                        axis=1,
                    )
                    view["Hitrate"] = view["Hitrate"].fillna("n/a")
                else:
                    view["Hitrate"] = "n/a"

                view["Hitrate"] = view["Hitrate"].replace(["None", "nan", None, ""], "n/a")
                _pred["Hitrate"] = view["Hitrate"]

                if "PICK_FINAL" in view.columns:
                    view["PICK_FINAL"] = view["PICK_FINAL"].replace({"X": "MSX"})

                bucket_order = {
                    "Elit": 0,
                    "Çok İyi": 1,
                    "İyi": 2,
                    "Dengeli": 3,
                    "Orta": 4,
                    "Düşük": 5,
                    "Zayıf": 6,
                    "n/a": 7,
                }
                view["_hitrate_rank"] = view["Hitrate"].map(bucket_order).fillna(7)
                view = view.sort_values(
                    ["_hitrate_rank", "TRUST_PCT", "PICK_PROB"],
                    ascending=[True, False, False],
                ).reset_index(drop=True)
                view = view.drop(columns=["_hitrate_rank"])

                _style = view[PRED_COLS].style.format(PRED_FMT)
                if hit_map:
                    _style = _style.apply(_row_bg, axis=1)
                if "POLICY_ICON" in PRED_COLS:
                    try:
                        idx = PRED_COLS.index("POLICY_ICON")
                        _style = _style.set_table_styles(
                            [
                                {"selector": f"th.col{idx}", "props": [("text-align", "center")]},
                                {"selector": f"td.col{idx}", "props": [("text-align", "center")]},
                            ],
                            overwrite=False,
                        )
                    except Exception:
                        _style = _style.set_properties(subset=["POLICY_ICON"], **{"text-align": "center"})
                st.dataframe(_style, use_container_width=True)
                if st.session_state.get("DEBUG_MODE", False):
                    try:
                        total_rows = len(view)
                        na_rows = int((view["Hitrate"] == "n/a").sum()) if "Hitrate" in view.columns else total_rows
                        st.write(
                            f"Hitrate debug: rows={total_rows}, n/a={na_rows}, hit_map={len(hit_map) if hit_map else 0}"
                        )
                        if isinstance(_pred, pd.DataFrame) and "PICK_FINAL" in _pred.columns:
                            x_picks = int((_pred["PICK_FINAL"] == "X").sum())
                            st.write(f"X picks (view): {x_picks}")
                    except Exception:
                        pass
                    try:
                        dbg_src = None
                        if isinstance(res, pd.DataFrame) and "_candidates" in res.columns and "PICK_FINAL" in res.columns:
                            dbg_src = res
                        elif isinstance(_pred, pd.DataFrame) and "_candidates" in _pred.columns:
                            dbg_src = _pred
                        else:
                            dbg_src = view

                        def _fnum_dbg(v):
                            try:
                                f = float(v)
                                return f if np.isfinite(f) else np.nan
                            except Exception:
                                return np.nan

                        def _prob01_dbg(v):
                            v = _fnum_dbg(v)
                            if not np.isfinite(v):
                                return np.nan
                            return v / 100.0 if v > 1.0 else v

                        def _get_draw_odd_dbg(row_obj):
                            odd = _fnum_dbg(row_obj.get("DrawOdd", np.nan))
                            if not np.isfinite(odd):
                                odd = _fnum_dbg(row_obj.get("Odds_Open_Draw", np.nan))
                            if not np.isfinite(odd):
                                sim_imp = _prob01_dbg(row_obj.get("SIM_IMP_DRAW"))
                                if np.isfinite(sim_imp) and sim_imp > 0:
                                    odd = 1.0 / sim_imp
                            return odd

                        if isinstance(dbg_src, pd.DataFrame):
                            if "Sim_Draw%" in dbg_src.columns:
                                sd = pd.to_numeric(dbg_src["Sim_Draw%"], errors="coerce")
                                st.write(f"Sim_Draw% finite: {int(sd.notna().sum())} | mean={sd.mean():.3f}")
                            if "DrawOdd" in dbg_src.columns:
                                do = pd.to_numeric(dbg_src["DrawOdd"], errors="coerce")
                                st.write(f"DrawOdd finite: {int(do.notna().sum())} | mean={do.mean():.2f}")

                        if isinstance(dbg_src, pd.DataFrame) and "_candidates" in dbg_src.columns:
                            params_local = debug.get("chosen_params", {}) if isinstance(debug, dict) else {}
                            x_close = _resolve_param("X_SWITCH_CLOSE", 0.62, params_local, debug)
                            x_topn_gate = int(_resolve_param("X_SWITCH_TOPN", 3, params_local, debug))
                            x_min_draw = _resolve_param("X_SWITCH_MIN_DRAW", 0.30, params_local, debug)
                            x_min_draw_odd = _resolve_param("X_SWITCH_MIN_DRAW_ODD", 3.0, params_local, debug)
                            x_max_over = _resolve_param("X_SWITCH_MAX_OVER", 0.58, params_local, debug)

                            topn = int(st.session_state.get("_dbg_topn", 3))
                            rows_with_cands = 0
                            x_in_cands = 0
                            x_topn = 0
                            x_rank_list = []
                            x_ratio_list = []
                            gate_gap = 0
                            gate_topn = 0
                            gate_draw = 0
                            gate_odd = 0
                            gate_over = 0
                            gate_all = 0

                            for _, r in dbg_src.iterrows():
                                cands = r.get("_candidates")
                                if not isinstance(cands, list) or not cands:
                                    continue
                                rows_with_cands += 1
                                scored = []
                                for c in cands:
                                    if not isinstance(c, dict):
                                        continue
                                    m = str(c.get("mkt", c.get("market", "")))
                                    sc = _fnum_dbg(c.get("score", c.get("score_total", np.nan)))
                                    scored.append((m, sc, c))
                                scored = sorted(scored, key=lambda t: (t[1] if np.isfinite(t[1]) else -1e9), reverse=True)
                                if not scored:
                                    continue
                                top_score = scored[0][1]
                                x_rows = [t for t in scored if t[0] == "X"]
                                ratio = np.nan
                                x_rank = None
                                if x_rows:
                                    x_in_cands += 1
                                    x_score = x_rows[0][1]
                                    x_rank = 1 + [t[0] for t in scored].index("X")
                                    x_rank_list.append(x_rank)
                                    if np.isfinite(top_score) and np.isfinite(x_score) and top_score > 0 and x_score > 0:
                                        ratio = x_score / top_score
                                        x_ratio_list.append(ratio)
                                    if x_rank <= topn:
                                        x_topn += 1

                                p_draw = _prob01_dbg(r.get("Sim_Draw%"))
                                if not np.isfinite(p_draw):
                                    p_draw = _prob01_dbg(r.get("SIM_IMP_DRAW"))
                                if not np.isfinite(p_draw):
                                    p_draw = _prob01_dbg(r.get("Poisson_Draw_Pct"))

                                p_over = _prob01_dbg(r.get("P_OVER25_ADJ"))
                                if not np.isfinite(p_over):
                                    p_over = _prob01_dbg(r.get("Sim_Over25%"))

                                draw_odd = _get_draw_odd_dbg(r)

                                cond_gap = np.isfinite(ratio) and ratio >= x_close
                                cond_topn = (x_rank is not None) and x_rank <= x_topn_gate
                                cond_draw = np.isfinite(p_draw) and p_draw >= x_min_draw
                                cond_odd = np.isfinite(draw_odd) and draw_odd >= x_min_draw_odd
                                cond_over = np.isfinite(p_over) and p_over <= x_max_over
                                if cond_gap:
                                    gate_gap += 1
                                if cond_topn:
                                    gate_topn += 1
                                if cond_draw:
                                    gate_draw += 1
                                if cond_odd:
                                    gate_odd += 1
                                if cond_over:
                                    gate_over += 1
                                if cond_gap and cond_topn and cond_draw and cond_odd and cond_over:
                                    gate_all += 1

                            st.write(f"Rows with _candidates: {rows_with_cands}")
                            st.write(
                                f"X in Top-{topn} (rows with cands): {x_topn}/{rows_with_cands} | rows_with_cands={rows_with_cands}"
                            )
                            if x_rank_list:
                                st.write(
                                    f"X in candidates: {x_in_cands} | avg rank={float(np.mean(x_rank_list)):.2f} | min rank={min(x_rank_list)}"
                                )
                            if x_ratio_list:
                                sr = pd.Series(x_ratio_list)
                                st.write(
                                    f"X gap stats: min={sr.min():.3f} mean={sr.mean():.3f} p50={sr.quantile(0.50):.3f} p75={sr.quantile(0.75):.3f} p90={sr.quantile(0.90):.3f}"
                                )
                            st.write(
                                f"X gate rows={gate_all} | gap>={x_close:.2f}={gate_gap} | topn<={x_topn_gate}={gate_topn} | p_draw>={x_min_draw:.2f}={gate_draw} | DrawOdd>={x_min_draw_odd:.2f}={gate_odd} | over<={x_max_over:.2f}={gate_over} | all={gate_all}"
                            )

                            if isinstance(_pred, pd.DataFrame) and "SWITCH_REASON" in _pred.columns:
                                x_switch = int((_pred["SWITCH_REASON"] == "X_VALUE_OVERRIDE").sum())
                            else:
                                x_switch = 0
                            x_switch_base = int(st.session_state.get("_dbg_x_switch_applied", 0))
                            x_switch_synth = int(st.session_state.get("_dbg_x_switch_applied_synth", 0))
                            st.write(f"X switch applied: {x_switch} | base={x_switch_base} | synth={x_switch_synth}")
                    except Exception:
                        pass
                if isinstance(_pred, pd.DataFrame) and not _pred.empty:
                    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    csv_bytes = _pred.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Predictions CSV (FULL)",
                        data=csv_bytes,
                        file_name=f"pred_full_{ts}.csv",
                        mime="text/csv",
                    )



            with st.expander("Detay Görünüm (Full Dump)", expanded=False):

                FULL_COLS = [

                    "Match_ID","Date","League","Hitrate","HomeTeam","AwayTeam",

                    "POLICY_FLAG","POLICY_RULE",

                    "PICK","PICK_PROB","PICK_ODD",

                "Sim_Home%","Sim_Draw%","Sim_Away%","Sim_Over25%","Sim_Under25%","Sim_BTTS_Yes%","Sim_BTTS_No%",

                "SIM_QUALITY","EFFECTIVE_N","TRUST_PCT",
                "P_OVER25_ADJ","P_BTTSY_ADJ","MISPRICING_PICK",
                "XG_WEIGHT_CFG",
                "XG_MICRO_BIAS_HOME","XG_MICRO_BIAS_DRAW","XG_MICRO_BIAS_AWAY",
                "XG_MICRO_BIAS_OVER25","XG_MICRO_BIAS_BTTS",
                "SCORE_LAM_H","SCORE_LAM_A",

                ]

                FULL_COLS = [c for c in FULL_COLS if c in res.columns]

                res_view = res.copy()
                if "PICK_FINAL" in res_view.columns:
                    res_view["PICK_FINAL"] = res_view["PICK_FINAL"].replace({"X": "MSX"})
                if "PICK" in res_view.columns:
                    res_view["PICK"] = res_view["PICK"].replace({"X": "MSX"})
                st.dataframe(res_view[FULL_COLS].style.format(format_map), use_container_width=True)



        with tab2:

            st.subheader("Maç detayı")

            label_options = []

            for _, row in res.iterrows():

                d = row.get("Date")

                date_str = d.strftime("%Y-%m-%d") if pd.notna(d) else ""

                label_options.append(

                    f"{row['HomeTeam']} vs {row['AwayTeam']} | {row['League']} | {date_str} | {row['Match_ID']}"

                )

            pick = st.selectbox("Maç seç", label_options, index=0)

            if pick:

                selected_match_id = pick.split(" | ")[-1]

                matching_futures = future[future["Match_ID"] == selected_match_id]

                if matching_futures.empty:

                    st.error("Seçilen maç bulunamadi.")

                else:

                    frow = matching_futures.iloc[0]  # Ilk esleseni al

                    selected_idx = frow.name  # index

                    stats = res[res["Match_ID"] == selected_match_id].iloc[0].to_dict()

                    sim_df = st.session_state.get("_sim_cache", {}).get(selected_match_id, pd.DataFrame())

                    row1 = st.columns(3)

                    row1[0].metric("Ev %", f"{float(stats.get('Sim_Home%', 0.0)):.1%}")

                    row1[1].metric("Berabere %", f"{float(stats.get('Sim_Draw%', 0.0)):.1%}")

                    row1[2].metric("Deplasman %", f"{float(stats.get('Sim_Away%', 0.0)):.1%}")

                    row2 = st.columns(3)

                    row2[0].metric("Üst 2.5 %", f"{float(stats.get('Sim_Over25%', 0.0)):.1%}")

                    row2[1].metric("Alt 2.5 %", f"{float(stats.get('Sim_Under25%', 0.0)):.1%}")

                    row2[2].metric("Avg Benzerlik", f"{float(stats.get('Avg_Similarity', 0.0)):.3f}")

                    row3 = st.columns(3)

                    row3[0].metric("BTTS Evet %", f"{float(stats.get('Sim_BTTS_Yes%', 0.0)):.1%}")

                    row3[1].metric("BTTS Hayir %", f"{float(stats.get('Sim_BTTS_No%', 0.0)):.1%}")

                    row3[2].metric("Benzer Maç Sayısı", int(stats.get('N', 0)))

                    row4 = st.columns(3)

                    row4[0].metric("SIM_QUALITY", f"{stats['SIM_QUALITY']:.3f}")

                    row4[1].metric("EFFECTIVE_N", f"{stats['EFFECTIVE_N']:.2f}")

                    row4[2].metric("TRUST_PCT", f"{float(stats.get('TRUST_PCT', 0.0)):.1f}")

                    debug_context = st.session_state.get("_debug", {}) if hasattr(st, "session_state") else {}
                    try:
                        trace_dict, trace_text = build_decision_trace(stats, debug_context=debug_context)
                        with st.expander("🧾 Decision Trace", expanded=False):
                            st.code(trace_text)
                            st.json(trace_dict)
                    except Exception as e:
                        if st.session_state.get("DEBUG_MODE", False):
                            st.error(f"Decision Trace error: {e}")

                    row5 = st.columns(2)

                    row5[0].metric("Same league neighbors", stats['same_league_count'])

                    row5[1].metric("Global neighbors", stats['global_count'])

                    st.markdown("### 🧩 Senaryo")

                    scenario_text = stats.get("SCENARIO_TEXT", "")

                    scenario_ev = stats.get("SCENARIO_EVIDENCE", "")

                    if scenario_text:

                        st.info(scenario_text)

                        if scenario_ev:

                            st.caption(scenario_ev)

                    else:

                        st.info("Veriler cok karmasik ve birbirine zit sinyaller veriyor. Bu macta istatistiksel bir trend yakalamak zor, canli bahiste gidisati gormek daha saglikli olabilir.")

                    with st.expander("🧪 Sentez Debug (Top-3 adaylar + kurallar)", expanded=False):

                        try:

                            best_dbg = pick_best_market_synth(stats, return_debug=True, params=debug.get("chosen_params") if isinstance(debug, dict) else None, debug_context=debug)
                            dbg = (best_dbg or {}).get("debug", {}) if isinstance(best_dbg, dict) else {}
                            cand = dbg.get("candidates", [])
                            st.write("SCENARIO_ID:", dbg.get("scenario_id"))
                            st.write("SCENARIO_TITLE:", dbg.get("scenario_title"))
                            st.write("lamH / lamA:", dbg.get("lamH"), dbg.get("lamA"))
                            st.write("PICK_MARGIN:", dbg.get("pick_margin"))
                            if cand:
                                df_dbg = pd.DataFrame(cand)
                                show_cols = ["chosen","market","group","base_prob_pct","bias_trust","bias_score","bias_scenario","bias_rank","bias_total","synth_prob_pct","odd"]

                                df_dbg = df_dbg[[c for c in show_cols if c in df_dbg.columns]]

                                for c in ["base_prob_pct","bias_trust","bias_score","bias_scenario","bias_rank","bias_total","synth_prob_pct","odd"]:

                                    if c in df_dbg.columns:

                                        df_dbg[c] = pd.to_numeric(df_dbg[c], errors="coerce").round(2)

                                st.dataframe(df_dbg, use_container_width=True, hide_index=True)

                                chosen_rows = df_dbg[df_dbg.get("chosen", False) == True] if "chosen" in df_dbg.columns else pd.DataFrame()

                                if len(chosen_rows) > 0:

                                    ch = chosen_rows.iloc[0].to_dict()

                                    st.markdown(

                                        f"**Seçilen:** `{ch.get('market')}`  |  "

                                        f"Base={ch.get('base_prob_pct')}% ? Synth={ch.get('synth_prob_pct')}%  "

                                        f"(bias {float(ch.get('bias_total', 0.0)):+.2f} puan)"

                                    )

                            st.caption(

                                "Not: SCORE_LAM toplami yüksekse Under2.5 küçük ceza yer, düsükse bonus alir. "

                                "Senaryo bias'lari ve diger bias'lar küçük (±2-4 puan) tutulur."

                            )

                        except Exception as e:

                            st.error(f"Sentez debug üretilemedi: {e}")



                    league_goals_avg_10 = np.nan


                    league_goals_avg_50 = np.nan


                    league_regime = "n/a"


                    league_regime_score = 0.0


                    home_trend_score = 0.0


                    away_trend_score = 0.0
                    _regime_cache = st.session_state.setdefault("_regime_cache", {})
                    league_name = frow.get("League")
                    home_team = frow.get("HomeTeam")
                    away_team = frow.get("AwayTeam")
                    league_key = ("LP", league_name) if league_name else None
                    home_key = ("TT", home_team) if home_team else None
                    away_key = ("TT", away_team) if away_team else None

                    league_goals_avg_10 = np.nan
                    league_goals_avg_50 = np.nan
                    league_regime = "n/a"
                    league_regime_score = 0.0
                    league_conf = 0.0
                    league_n_long = 0
                    home_gf_avg_10 = np.nan
                    home_gf_avg_50 = np.nan
                    home_trend_score = 0.0
                    home_trend_n = 0
                    away_gf_avg_10 = np.nan
                    away_gf_avg_50 = np.nan
                    away_trend_score = 0.0
                    away_trend_n = 0
                    home_xg_trend_score = 0.0
                    away_xg_trend_score = 0.0
                    home_metric_used = "n/a"
                    away_metric_used = "n/a"
                    home_xg_metric_used = "n/a"
                    away_xg_metric_used = "n/a"
                    home_trend_reason = "metric_n/a"
                    away_trend_reason = "metric_n/a"
                    home_xg_trend_reason = "metric_n/a"
                    away_xg_trend_reason = "metric_n/a"

                    if isinstance(past, pd.DataFrame) and not past.empty:
                        past_league = (
                            past[past["League"] == league_name]
                            if ("League" in past.columns and league_name)
                            else past
                        )
                        if {"HomeTeam", "AwayTeam"}.issubset(past.columns) and home_team:
                            past_home = past[
                                (past["HomeTeam"] == home_team)
                                | (past["AwayTeam"] == home_team)
                            ]
                        else:
                            past_home = past
                        if {"HomeTeam", "AwayTeam"}.issubset(past.columns) and away_team:
                            past_away = past[
                                (past["HomeTeam"] == away_team)
                                | (past["AwayTeam"] == away_team)
                            ]
                        else:
                            past_away = past

                        if league_key and league_key in _regime_cache:
                            lp = _regime_cache.get(league_key, {})
                        else:
                            lp = compute_league_profile(past_league, target_league=league_name)
                            if league_key:
                                _regime_cache[league_key] = lp
                        if home_key and home_key in _regime_cache:
                            ht = _regime_cache.get(home_key, {})
                        else:
                            ht = compute_team_trend(past_home, home_team)
                            if home_key:
                                _regime_cache[home_key] = ht
                        if away_key and away_key in _regime_cache:
                            at = _regime_cache.get(away_key, {})
                        else:
                            at = compute_team_trend(past_away, away_team)
                            if away_key:
                                _regime_cache[away_key] = at
                        if len(_regime_cache) > 500:
                            _regime_cache.pop(next(iter(_regime_cache)))

                        league_goals_avg_10 = lp.get("goals_avg_10", np.nan)
                        league_goals_avg_50 = lp.get("goals_avg_50", np.nan)
                        league_regime = lp.get("league_regime", "n/a")
                        league_regime_score = lp.get("league_regime_score", 0.0)
                        league_conf = lp.get("conf", 0.0)
                        league_n_long = lp.get("n_long", 0)

                        home_gf_avg_10 = ht.get("short_mean", np.nan)
                        home_gf_avg_50 = ht.get("long_mean", np.nan)
                        home_trend_score = ht.get("trend_score", 0.0)
                        home_trend_n = ht.get("n_matches", 0)
                        away_gf_avg_10 = at.get("short_mean", np.nan)
                        away_gf_avg_50 = at.get("long_mean", np.nan)
                        away_trend_score = at.get("trend_score", 0.0)
                        away_trend_n = at.get("n_matches", 0)

                        home_xg_trend_score = ht.get("xg_trend_score", 0.0)
                        away_xg_trend_score = at.get("xg_trend_score", 0.0)

                        home_metric_used = ht.get("metric_used", "n/a")
                        away_metric_used = at.get("metric_used", "n/a")
                        home_xg_metric_used = ht.get("xg_metric_used", "n/a")
                        away_xg_metric_used = at.get("xg_metric_used", "n/a")
                        home_trend_reason = ht.get("trend_reason", "metric_n/a")
                        away_trend_reason = at.get("trend_reason", "metric_n/a")
                        home_xg_trend_reason = ht.get("xg_trend_reason", "metric_n/a")
                        away_xg_trend_reason = at.get("xg_trend_reason", "metric_n/a")

                        if not isinstance(league_regime, str) or league_regime.strip() == "" or league_regime.lower() == "nan":
                            league_regime = "n/a"
                        if not np.isfinite(league_regime_score):
                            league_regime_score = 0.0
                        if not np.isfinite(home_trend_score):
                            home_trend_score = 0.0
                        if not np.isfinite(away_trend_score):
                            away_trend_score = 0.0
                        if not np.isfinite(home_xg_trend_score):
                            home_xg_trend_score = 0.0
                        if not np.isfinite(away_xg_trend_score):
                            away_xg_trend_score = 0.0

                        sim_df = sim_df.copy()
                        sim_defaults = {
                            "SIM_LEAGUE_REGIME": "n/a",
                            "SIM_LEAGUE_REGIME_SCORE": 0.0,
                            "SIM_LEAGUE_GOALS_AVG_10": np.nan,
                            "SIM_LEAGUE_GOALS_AVG_50": np.nan,
                            "SIM_LEAGUE_CONF": 0.0,
                            "SIM_LEAGUE_N_LONG": 0,
                        }
                        if "League" in sim_df.columns and isinstance(past, pd.DataFrame) and not past.empty:
                            league_profile_map = {}
                            for row_league in sim_df["League"].dropna().unique():
                                cache_key = ("LP_SIM", row_league)
                                if cache_key in _regime_cache:
                                    lp_sim = _regime_cache.get(cache_key, {})
                                else:
                                    if "League" in past.columns and row_league:
                                        past_league = past[past["League"] == row_league]
                                    else:
                                        past_league = past
                                    lp_sim = compute_league_profile(past_league, target_league=row_league)
                                    _regime_cache[cache_key] = lp_sim
                                league_profile_map[row_league] = lp_sim

                            sim_df["SIM_LEAGUE_REGIME"] = sim_df["League"].map(
                                lambda lg: league_profile_map.get(lg, {}).get("league_regime", "n/a")
                            )
                            sim_df["SIM_LEAGUE_REGIME_SCORE"] = sim_df["League"].map(
                                lambda lg: league_profile_map.get(lg, {}).get("league_regime_score", 0.0)
                            )
                            sim_df["SIM_LEAGUE_GOALS_AVG_10"] = sim_df["League"].map(
                                lambda lg: league_profile_map.get(lg, {}).get("goals_avg_10", np.nan)
                            )
                            sim_df["SIM_LEAGUE_GOALS_AVG_50"] = sim_df["League"].map(
                                lambda lg: league_profile_map.get(lg, {}).get("goals_avg_50", np.nan)
                            )
                            sim_df["SIM_LEAGUE_CONF"] = sim_df["League"].map(
                                lambda lg: league_profile_map.get(lg, {}).get("conf", 0.0)
                            )
                            sim_df["SIM_LEAGUE_N_LONG"] = sim_df["League"].map(
                                lambda lg: league_profile_map.get(lg, {}).get("n_long", 0)
                            )
                        else:
                            for col, default in sim_defaults.items():
                                sim_df[col] = default

                        sim_has_xg = isinstance(past, pd.DataFrame) and {"xG_Home", "xG_Away"}.issubset(past.columns)
                        if {"HomeTeam", "AwayTeam"}.issubset(sim_df.columns) and isinstance(past, pd.DataFrame) and not past.empty:
                            teams = pd.concat(
                                [sim_df["HomeTeam"], sim_df["AwayTeam"]],
                                ignore_index=True,
                            ).dropna().unique()
                            team_trend_map = {}
                            for team in teams:
                                cache_key = ("TT_SIM", team)
                                if cache_key in _regime_cache:
                                    tt = _regime_cache.get(cache_key, {})
                                else:
                                    tt = compute_team_trend(past, team)
                                    _regime_cache[cache_key] = tt
                                team_trend_map[team] = tt

                            sim_df["SIM_HOME_TREND_SCORE"] = sim_df["HomeTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("trend_score", 0.0)
                            )
                            sim_df["SIM_HOME_TREND_N"] = sim_df["HomeTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("n_matches", 0)
                            )
                            sim_df["SIM_HOME_TREND_REASON"] = sim_df["HomeTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("trend_reason", "metric_n/a")
                            )
                            sim_df["SIM_HOME_METRIC_USED"] = sim_df["HomeTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("metric_used", "n/a")
                            )
                            sim_df["SIM_AWAY_TREND_SCORE"] = sim_df["AwayTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("trend_score", 0.0)
                            )
                            sim_df["SIM_AWAY_TREND_N"] = sim_df["AwayTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("n_matches", 0)
                            )
                            sim_df["SIM_AWAY_TREND_REASON"] = sim_df["AwayTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("trend_reason", "metric_n/a")
                            )
                            sim_df["SIM_AWAY_METRIC_USED"] = sim_df["AwayTeam"].map(
                                lambda t: team_trend_map.get(t, {}).get("metric_used", "n/a")
                            )
                            if sim_has_xg:
                                sim_df["SIM_HOME_XG_TREND_SCORE"] = sim_df["HomeTeam"].map(
                                    lambda t: team_trend_map.get(t, {}).get("xg_trend_score", 0.0)
                                )
                                sim_df["SIM_AWAY_XG_TREND_SCORE"] = sim_df["AwayTeam"].map(
                                    lambda t: team_trend_map.get(t, {}).get("xg_trend_score", 0.0)
                                )
                                sim_df["SIM_HOME_XG_TREND_REASON"] = sim_df["HomeTeam"].map(
                                    lambda t: team_trend_map.get(t, {}).get("xg_trend_reason", "metric_n/a")
                                )
                                sim_df["SIM_AWAY_XG_TREND_REASON"] = sim_df["AwayTeam"].map(
                                    lambda t: team_trend_map.get(t, {}).get("xg_trend_reason", "metric_n/a")
                                )
                        else:
                            sim_df["SIM_HOME_TREND_SCORE"] = 0.0
                            sim_df["SIM_HOME_TREND_N"] = 0
                            sim_df["SIM_HOME_TREND_REASON"] = "metric_n/a"
                            sim_df["SIM_HOME_METRIC_USED"] = "n/a"
                            sim_df["SIM_AWAY_TREND_SCORE"] = 0.0
                            sim_df["SIM_AWAY_TREND_N"] = 0
                            sim_df["SIM_AWAY_TREND_REASON"] = "metric_n/a"
                            sim_df["SIM_AWAY_METRIC_USED"] = "n/a"
                            sim_df["SIM_HOME_XG_TREND_SCORE"] = 0.0
                            sim_df["SIM_AWAY_XG_TREND_SCORE"] = 0.0
                            sim_df["SIM_HOME_XG_TREND_REASON"] = "metric_n/a"
                            sim_df["SIM_AWAY_XG_TREND_REASON"] = "metric_n/a"

                        for col in [
                            "SIM_HOME_TREND_SCORE",
                            "SIM_AWAY_TREND_SCORE",
                            "SIM_HOME_XG_TREND_SCORE",
                            "SIM_AWAY_XG_TREND_SCORE",
                        ]:
                            if col in sim_df.columns:
                                s = pd.to_numeric(sim_df[col], errors="coerce").fillna(0.0)
                                sim_df[col] = s.mask(s.abs() < 1e-9, 0.0)

                    def _fmt2(v):




                        f = _safe_float(v)



                        return f"{f:.2f}" if np.isfinite(f) else "NA"
                    st.session_state["dbg_league_regime"] = {
                        "TARGET_LEAGUE_REGIME": league_regime,
                        "TARGET_LEAGUE_REGIME_SCORE": league_regime_score,
                        "TARGET_LEAGUE_GOALS_AVG_10": league_goals_avg_10,
                        "TARGET_LEAGUE_GOALS_AVG_50": league_goals_avg_50,
                        "TARGET_LEAGUE_CONF": league_conf,
                        "TARGET_LEAGUE_N_LONG": league_n_long,
                        "TARGET_HOME_GF_AVG_10": home_gf_avg_10,
                        "TARGET_HOME_GF_AVG_50": home_gf_avg_50,
                        "TARGET_HOME_TREND_SCORE": home_trend_score,
                        "TARGET_HOME_TREND_N": home_trend_n,
                        "TARGET_HOME_METRIC_USED": home_metric_used,
                        "TARGET_HOME_TREND_REASON": home_trend_reason,
                        "TARGET_AWAY_GF_AVG_10": away_gf_avg_10,
                        "TARGET_AWAY_GF_AVG_50": away_gf_avg_50,
                        "TARGET_AWAY_TREND_SCORE": away_trend_score,
                        "TARGET_AWAY_TREND_N": away_trend_n,
                        "TARGET_AWAY_METRIC_USED": away_metric_used,
                        "TARGET_AWAY_TREND_REASON": away_trend_reason,
                        "TARGET_HOME_XG_TREND_SCORE": home_xg_trend_score,
                        "TARGET_AWAY_XG_TREND_SCORE": away_xg_trend_score,
                        "TARGET_HOME_XG_TREND_REASON": home_xg_trend_reason,
                        "TARGET_AWAY_XG_TREND_REASON": away_xg_trend_reason,
                    }
                    st.caption(
                        f"Regime={league_regime} score={_fmt2(league_regime_score)} L10={_fmt2(league_goals_avg_10)} L50={_fmt2(league_goals_avg_50)} | "
                        f"H_GF10={_fmt2(home_gf_avg_10)} H_GF50={_fmt2(home_gf_avg_50)} Hscore={_fmt2(home_trend_score)} HN={home_trend_n} | "
                        f"A_GF10={_fmt2(away_gf_avg_10)} A_GF50={_fmt2(away_gf_avg_50)} Ascore={_fmt2(away_trend_score)} AN={away_trend_n} | "
                        f"Hr={home_trend_reason} Ar={away_trend_reason}"
                    )
                    if home_xg_metric_used == "xG_for" or away_xg_metric_used == "xG_for":
                        st.caption(
                            f"HxGr={home_xg_trend_reason} AxGr={away_xg_trend_reason}"
                        )
                    st.markdown("### 📊 Benzer Maçlara Göre Dağılım (Top 10)")
                    top10 = sim_df.head(10).copy()

                    c1, c2, c3 = st.columns(3)

                    # 1X2

                    with c1:

                        st.markdown("**1X2**")

                        df1 = pd.DataFrame([

                            ["MS1", float((top10["Result"]=="HOME").mean())],

                            ["X",   float((top10["Result"]=="DRAW").mean())],

                            ["MS2", float((top10["Result"]=="AWAY").mean())],

                        ], columns=["Seçenek","Oran"])

                        st.dataframe(df1.style.format({"Oran":"{:.1%}"}), use_container_width=True)

                    # Over / Under

                    with c2:

                        st.markdown("**Over / Under 2.5**")

                        over = float(top10["Is_Over25"].mean()) if "Is_Over25" in top10.columns else 0.0

                        df2 = pd.DataFrame([["Over 2.5", over], ["Under 2.5", 1-over]], columns=["Seçenek","Oran"])

                        st.dataframe(df2.style.format({"Oran":"{:.1%}"}), use_container_width=True)

                    # BTTS

                    with c3:

                        st.markdown("**BTTS**")

                        btts = float(top10["Is_BTTS"].mean()) if "Is_BTTS" in top10.columns else 0.0

                        df3 = pd.DataFrame([["BTTS Yes", btts], ["BTTS No", 1-btts]], columns=["Seçenek","Oran"])

                        st.dataframe(df3.style.format({"Oran":"{:.1%}"}), use_container_width=True)

                    st.caption(f"KNN feature set: {len(debug['feature_cols'])} sütun (xG agirlik {debug['chosen_params']['weight_config']['xg']})")

                    show_advanced_cols = st.checkbox("Advanced (Regime/Trend kolonlari)", value=False)
                    st.write("Benzer maçlar")
                    sim_cols = [
                        "Date", "League", "HomeTeam", "AwayTeam",
                        "FTHG", "FTAG", "FT_Score",
                        "Result_1X2", "Result_OU", "Result_BTTS",
                        "xG_Home", "xG_Away", "xG_Total",
                        "TotalGoals", "Similarity_Score",
                    ]
                    if show_advanced_cols:
                        extra_cols = [
                            "TARGET_LEAGUE_REGIME", "TARGET_LEAGUE_REGIME_SCORE",
                            "TARGET_LEAGUE_GOALS_AVG_10", "TARGET_LEAGUE_GOALS_AVG_50",
                            "TARGET_LEAGUE_CONF", "TARGET_LEAGUE_N_LONG",
                            "TARGET_HOME_GF_AVG_10", "TARGET_HOME_GF_AVG_50", "TARGET_HOME_TREND_SCORE", "TARGET_HOME_TREND_N",
                            "TARGET_AWAY_GF_AVG_10", "TARGET_AWAY_GF_AVG_50", "TARGET_AWAY_TREND_SCORE", "TARGET_AWAY_TREND_N",
                            "SIM_LEAGUE_REGIME", "SIM_LEAGUE_REGIME_SCORE",
                            "SIM_LEAGUE_GOALS_AVG_10", "SIM_LEAGUE_GOALS_AVG_50",
                            "SIM_LEAGUE_CONF", "SIM_LEAGUE_N_LONG",
                            "SIM_HOME_TREND_SCORE", "SIM_HOME_TREND_N",
                            "SIM_AWAY_TREND_SCORE", "SIM_AWAY_TREND_N",
                            "SIM_HOME_TREND_REASON", "SIM_HOME_METRIC_USED",
                            "SIM_AWAY_TREND_REASON", "SIM_AWAY_METRIC_USED",
                        ]
                        if "SIM_HOME_XG_TREND_SCORE" in sim_df.columns or "SIM_AWAY_XG_TREND_SCORE" in sim_df.columns:
                            extra_cols += ["SIM_HOME_XG_TREND_SCORE", "SIM_AWAY_XG_TREND_SCORE"]
                        sim_cols = sim_cols + extra_cols
                    for c in ["FTHG", "FTAG", "TotalGoals"]:

                        if c in sim_df.columns:

                            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce").round(0).astype("Int64")

                    for c in ["xG_Home", "xG_Away"]:

                        if c in sim_df.columns:

                            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce").round(2)

                    if "xG_Home" in sim_df.columns and "xG_Away" in sim_df.columns:

                        sim_df["xG_Total"] = (sim_df["xG_Home"] + sim_df["xG_Away"]).round(2)

                    if "FTHG" in sim_df.columns and "FTAG" in sim_df.columns:
                        hg = pd.to_numeric(sim_df["FTHG"], errors="coerce")
                        ag = pd.to_numeric(sim_df["FTAG"], errors="coerce")
                        sim_df["Result_1X2"] = np.where(
                            hg > ag,
                            "HOME",
                            np.where(hg < ag, "AWAY", "DRAW"),
                        )
                        tg = hg + ag
                        sim_df["Result_OU"] = np.where(tg >= 3, "Over2.5", "Under2.5")
                        sim_df["Result_BTTS"] = np.where((hg > 0) & (ag > 0), "BTTS Yes", "BTTS No")

                    if "FTHG" in sim_df.columns and "FTAG" in sim_df.columns:

                        sim_df["FT_Score"] = (

                            sim_df["FTHG"].astype("Int64").astype(str)

                            + "-"

                            + sim_df["FTAG"].astype("Int64").astype(str)

                        )

                    sim_cols = [c for c in sim_cols if c in sim_df.columns]

                    display_count = min(len(sim_df), debug['chosen_params']['k_same'] + debug['chosen_params']['k_global'])

                    if isinstance(sim_df, pd.DataFrame) and not sim_df.empty:
                        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        sim_csv = sim_df[sim_cols].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Similar Matches CSV",
                            data=sim_csv,
                            file_name=f"sim_matches_{ts}.csv",
                            mime="text/csv",
                        )

                    fmt = {"Similarity_Score": "{:.3f}"}
                    for c in ["xG_Home", "xG_Away", "xG_Total"]:
                        if c in sim_df.columns:
                            fmt[c] = "{:.2f}"
                    st.dataframe(

                        sim_df[sim_cols].head(display_count).style.format(fmt),

                        use_container_width=True,

                    )

        with tab3:

            st.subheader("Sema & Debug")
            dbg_cols = [
                "Date", "League", "HomeTeam", "AwayTeam",
                "PICK", "PICK_ODD", "PICK_PROB",
                "P_OVER25_ADJ", "P_BTTSY_ADJ", "MISPRICING_PICK",
                "XG_WEIGHT_CFG", "XG_MICRO_BIAS_HOME", "XG_MICRO_BIAS_DRAW", "XG_MICRO_BIAS_AWAY",
                "XG_MICRO_BIAS_OVER25", "XG_MICRO_BIAS_BTTS",
                "PICK_BASE", "PICK_FINAL", "PICK_SWITCHED",
                "PICK_MIS_BONUS", "PICK_SIM_BONUS", "SWITCH_REASON",
            ]
            if isinstance(pred_df, pd.DataFrame):
                if st.session_state.get("DEBUG_MODE", False) and "PICK_SWITCHED" in pred_df.columns:
                    n_switched = int((pred_df["PICK_SWITCHED"] == 1).sum())
                    st.write(f"Switch count: {n_switched}")
                dbg_cols = [c for c in dbg_cols if c in pred_df.columns]
                if dbg_cols:
                    st.dataframe(pred_df[dbg_cols].head(15), use_container_width=True)
                if st.session_state.get("DEBUG_MODE", False):
                    audit_df = pd.DataFrame(_PICK_SWITCH_AUDIT_ROWS)
                    st.write(f"Audit rows: {len(audit_df)}")
                    if not audit_df.empty:
                        st.write(f"Switched: {int((audit_df['PICK_SWITCHED'] == 1).sum())}")
                        st.dataframe(audit_df.head(50), use_container_width=True)
                        st.download_button(
                            "Download Pick Switch Audit CSV",
                            audit_df.to_csv(index=False).encode("utf-8"),
                            file_name="pick_switch_audit.csv",
                            mime="text/csv",
                        )
            dbg_regime = st.session_state.get("dbg_league_regime")
            def _fmt2_dbg(v):
                f = _safe_float(v)
                return f"{f:.2f}" if np.isfinite(f) else "NA"
            if isinstance(dbg_regime, dict):
                league_regime = dbg_regime.get("TARGET_LEAGUE_REGIME", "n/a")
                if not isinstance(league_regime, str) or league_regime.strip() == "" or league_regime.lower() == "nan":
                    league_regime = "n/a"
                st.write(
                    f"Regime={league_regime} score={_fmt2_dbg(dbg_regime.get('TARGET_LEAGUE_REGIME_SCORE'))} L10={_fmt2_dbg(dbg_regime.get('TARGET_LEAGUE_GOALS_AVG_10'))} L50={_fmt2_dbg(dbg_regime.get('TARGET_LEAGUE_GOALS_AVG_50'))} | "
                    f"Hscore={_fmt2_dbg(dbg_regime.get('TARGET_HOME_TREND_SCORE'))} HN={dbg_regime.get('TARGET_HOME_TREND_N')} Hr={dbg_regime.get('TARGET_HOME_TREND_REASON')} Hm={dbg_regime.get('TARGET_HOME_METRIC_USED')} | "
                    f"Ascore={_fmt2_dbg(dbg_regime.get('TARGET_AWAY_TREND_SCORE'))} AN={dbg_regime.get('TARGET_AWAY_TREND_N')} Ar={dbg_regime.get('TARGET_AWAY_TREND_REASON')} Am={dbg_regime.get('TARGET_AWAY_METRIC_USED')}"
                )
                st.caption(
                    f"scores={{regime:{league_regime}, score:{_fmt2_dbg(dbg_regime.get('TARGET_LEAGUE_REGIME_SCORE'))}, L10:{_fmt2_dbg(dbg_regime.get('TARGET_LEAGUE_GOALS_AVG_10'))}, L50:{_fmt2_dbg(dbg_regime.get('TARGET_LEAGUE_GOALS_AVG_50'))}, H:{_fmt2_dbg(dbg_regime.get('TARGET_HOME_TREND_SCORE'))}, HN:{dbg_regime.get('TARGET_HOME_TREND_N')}, Hr:{dbg_regime.get('TARGET_HOME_TREND_REASON')}, Hm:{dbg_regime.get('TARGET_HOME_METRIC_USED')}, HxGr:{dbg_regime.get('TARGET_HOME_XG_TREND_REASON')}, A:{_fmt2_dbg(dbg_regime.get('TARGET_AWAY_TREND_SCORE'))}, AN:{dbg_regime.get('TARGET_AWAY_TREND_N')}, Ar:{dbg_regime.get('TARGET_AWAY_TREND_REASON')}, Am:{dbg_regime.get('TARGET_AWAY_METRIC_USED')}, AxGr:{dbg_regime.get('TARGET_AWAY_XG_TREND_REASON')}}}"
                )
            else:
                st.write("Regime: n/a")

            st.write(f"Past row sayisi: {len(past)}")

            st.write(f"Future row sayisi: {len(future)}")

            st.write(f"Feature null threshold: {debug['chosen_params']['null_threshold']:.0%}")

            st.write(f"K_SAME: {debug['chosen_params']['k_same']}, K_GLOBAL: {debug['chosen_params']['k_global']}, MIN_SAME_FOUND: {debug['chosen_params']['min_same_found']}")

            st.write("Past kolonlari:")

            st.code(sorted(list(past.columns)))

            st.write("Future kolonlari:")

            st.code(sorted(list(future.columns)))

            if "future_odds_present" in debug:

                st.write("Future'da bulunan odds kolonlari:")

                st.code(debug["future_odds_present"])

                if debug["future_odds_missing"]:

                    st.write("Future'da eksik odds kolonlari:")

                    st.code(debug["future_odds_missing"])

            feature_weight_df = pd.DataFrame(list(debug["feature_weights"].items()), columns=["feature", "weight"])

            st.write("Seçilen featurelar ve agirliklari:")

            st.dataframe(feature_weight_df)

            wc = debug["chosen_params"]["weight_config"]

            st.write(f"Weight config summary: xG={wc.get('xg', 1.0)}, elo={wc.get('elo', 1.0)}, league_pos={wc.get('league_pos', 1.0)}, default={wc.get('default', 1.0)}")

            null_top = debug["null_info"][:20]

            if null_top:

                null_df = pd.DataFrame(null_top)

                null_df["null_ratio"] = null_df["null_ratio"].map("{:.1%}".format)

                st.write("Null ratio (en yüksek 20):")

                st.dataframe(null_df)

            else:

                st.write("Null ratio bilgisi yok.")

            if debug["drop_records"]:

                st.write("Düsürülen kolonlar (sebep):")

                st.dataframe(pd.DataFrame(debug["drop_records"]))

            else:

                st.write("Düsürülen kolon yok.")

            st.write("xG bilgisi:")

            for col in ("xG_Home", "xG_Away"):

                info = debug["xg_info"].get(col, {"included": False, "reason": "veride yok"})

                status = "Dahil" if info["included"] else "Hariç"

                st.write(f"- {col}: {status} ({info['reason']})")

            if debug.get("warnings"):

                st.write("Uyarilar:")

                for w in debug["warnings"]:

                    st.write(f"- {w}")

            st.write("Feature weights:")

            st.dataframe(pd.DataFrame(list(debug["feature_weights"].items()), columns=["feature", "weight"]))

            st.write(f"Duplicate Match_ID past: {debug['duplicate_match_id_past']}")

            st.write(f"Duplicate Match_ID future: {debug['duplicate_match_id_future']}")

            mix_summary = res[["same_league_count", "global_count"]]

            st.write("Neighbor mix summary:")

            st.dataframe(mix_summary.describe().T)

            st.write("Session knn bilgiler:")

            st.json({

                "feature_count": len(debug["feature_cols"]),

                "knn_feature_list": debug["feature_cols"],

                "knn_feature_weights": debug["feature_weights"],

                "same_league_mode": debug["chosen_params"]["same_league_mode"],

            })

            st.write(f"CONF floor: {debug['conf_floor']:.2f} | CONF=0 orani: {debug['conf_zero_rate']:.1%}")

            st.json(debug["conf_reason_counts"])

            st.markdown("### CONF / SIM_QUALITY Teshis")

            st.write({

                "CONF floor (used)": debug["conf_floor_used"],

                "CONF floor (suggested q60)": debug["suggested_conf_floor"],

            })

            st.write("SIM_QUALITY quantiles:")

            st.json(debug["sim_quality_q"])

        with tab4:
            st.subheader("Tahmin Sonuçları")

            log_path = os.path.join(_get_base_dir(), "pred_results_log.csv")
            result_cols = ["RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS", "RES_UPDATED_AT"]

            log_df = pd.DataFrame()
            if os.path.exists(log_path):
                try:
                    log_df = pd.read_csv(log_path)
                except Exception:
                    log_df = pd.DataFrame()
            if not log_df.empty:
                def _infer_pick_group(pick_val: str) -> str:
                    s = str(pick_val or "").strip().upper()
                    if s in ("MS1", "MS2", "X", "MSX"):
                        return "1X2"
                    if s in ("OVER2.5", "UNDER2.5"):
                        return "OU"
                    if s in ("BTTS_YES", "BTTS_NO", "BTTS YES", "BTTS NO"):
                        return "BTTS"
                    return ""

                if "PICK_GROUP" not in log_df.columns:
                    log_df["PICK_GROUP"] = ""
                if "PICK_FINAL" in log_df.columns:
                    pg = log_df["PICK_GROUP"].astype(str).str.strip()
                    missing_pg = pg.eq("") | pg.eq("nan") | pg.isna()
                    if missing_pg.any():
                        log_df.loc[missing_pg, "PICK_GROUP"] = log_df.loc[missing_pg, "PICK_FINAL"].map(_infer_pick_group)
                        log_df.to_csv(log_path, index=False)

                if "PICK_GROUP" in log_df.columns and "PICK_FINAL" in log_df.columns:
                    if (log_df["PICK_GROUP"].astype(str).str.strip() == "").any():
                        log_df["PICK_GROUP"] = log_df["PICK_FINAL"].map(_infer_pick_group).fillna(log_df["PICK_GROUP"])

                if "SIM_LEAGUE_GOALS_AVG_10" in log_df.columns and "TARGET_LEAGUE_GOALS_AVG_10" in log_df.columns:
                    if log_df["SIM_LEAGUE_GOALS_AVG_10"].isna().all():
                        log_df["SIM_LEAGUE_GOALS_AVG_10"] = log_df["TARGET_LEAGUE_GOALS_AVG_10"]
                if "SIM_LEAGUE_GOALS_AVG_50" in log_df.columns and "TARGET_LEAGUE_GOALS_AVG_50" in log_df.columns:
                    if log_df["SIM_LEAGUE_GOALS_AVG_50"].isna().all():
                        log_df["SIM_LEAGUE_GOALS_AVG_50"] = log_df["TARGET_LEAGUE_GOALS_AVG_50"]

            log_schema_cols = [
                "Match_ID", "Date", "League", "HomeTeam", "AwayTeam",
                "PICK_FINAL", "PICK_GROUP", "PICK_ODD", "PICK_PROB",
                "PICK_PROB_EFF", "PICK_PROB_ADJ",
                "PICK_BASE", "PICK_SWITCHED", "SWITCH_REASON",
                "PICK_SCORE_TOTAL", "PICK_MIS_BONUS", "PICK_SIM_BONUS",
                "MARKET_WEIGHT", "BONUS_TOTAL", "MW_IMPLIED",
                "TRUST_PCT", "SIM_QUALITY", "EFFECTIVE_N",
                "HAS_CANDS", "CANDS_N", "CANDS_REASON", "FINAL_MKT",
                "SEL_SCORE", "SEL_ALPHA", "SEL_SIM", "SEL_P_ADJ",
                "SEL_MISPRICING", "SEL_MIS_BONUS",
                "POLICY_FLAG", "POLICY_RULE",
                "P_OVER25_ADJ", "P_BTTSY_ADJ",
                "TARGET_LEAGUE_GOALS_AVG_10", "TARGET_LEAGUE_GOALS_AVG_50",
                "SIM_LEAGUE_GOALS_AVG_10", "SIM_LEAGUE_GOALS_AVG_50",
                "xG_Home", "xG_Away", "xG_Total",
                "TOP1_MKT", "TOP1_SCORE", "TOP1_PROB_EFF",
                "TOP2_MKT", "TOP2_SCORE", "TOP2_PROB_EFF",
                "TOP3_MKT", "TOP3_SCORE", "TOP3_PROB_EFF",
                "RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU",
                "RES_RESULT_BTTS", "RES_UPDATED_AT",
            ]
            if not log_df.empty:
                missing_cols = [c for c in log_schema_cols if c not in log_df.columns]
                if missing_cols:
                    for c in missing_cols:
                        log_df[c] = np.nan
                    log_df.to_csv(log_path, index=False)
                # Backfill policy flags/rules with current policy logic
                if "POLICY_FLAG" in log_df.columns and "POLICY_RULE" in log_df.columns:
                    try:
                        policy_ctx = _load_policy_context(log_path)
                        pf = []
                        pr = []
                        for _, row in log_df.iterrows():
                            flag, rule = _policy_flag(row, policy_ctx)
                            pf.append(int(flag))
                            pr.append(rule)
                        log_df["POLICY_FLAG"] = pf
                        log_df["POLICY_RULE"] = pr
                        log_df.to_csv(log_path, index=False)
                    except Exception:
                        pass

            # --- Controls: restore results / recalc RES / recalc X / rebuild from backtest ---
            restore_col, res_col, x_col, rebuild_col = st.columns(4)
            if restore_col.button("Yedekten RES_ geri yukle", key="btn_restore_res"):
                try:
                    base_dir = _get_base_dir()
                    backups = sorted(
                        glob.glob(os.path.join(base_dir, "pred_results_log_backup_*.csv")),
                        key=lambda p: os.path.getmtime(p),
                        reverse=True,
                    )
                    if not backups:
                        st.warning("Yedek bulunamadi.")
                    else:
                        backup_path = backups[0]
                        backup_df = pd.read_csv(backup_path)
                        if "Match_ID" not in backup_df.columns:
                            st.error("Yedek dosyada Match_ID yok.")
                        else:
                            if "Match_ID" not in log_df.columns:
                                log_df["Match_ID"] = (
                                    log_df.get("Date", "").astype(str)
                                    + "|" + log_df.get("League", "").astype(str)
                                    + "|" + log_df.get("HomeTeam", "").astype(str)
                                    + "|" + log_df.get("AwayTeam", "").astype(str)
                                )
                            bmap = backup_df.set_index("Match_ID")
                            fill_cols = [
                                "RES_FTHG",
                                "RES_FTAG",
                                "RES_RESULT_1X2",
                                "RES_RESULT_OU",
                                "RES_RESULT_BTTS",
                                "RES_UPDATED_AT",
                            ]
                            for col in fill_cols:
                                if col not in log_df.columns:
                                    log_df[col] = np.nan
                                if col in bmap.columns:
                                    log_df[col] = log_df[col].where(
                                        log_df[col].notna(), log_df["Match_ID"].map(bmap[col])
                                    )
                            log_df.to_csv(log_path, index=False)
                            try:
                                log_df = pd.read_csv(log_path)
                            except Exception:
                                pass
                            st.success(f"Yedekten dolduruldu: {os.path.basename(backup_path)}")
                except Exception as e:
                    st.error(f"Yedek geri yukleme hatasi: {e}")

            if res_col.button("RES_ yeniden hesapla (skorlardan)", key="btn_recalc_res"):
                try:
                    if log_df.empty:
                        st.warning("Log bos.")
                    else:
                        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = os.path.join(_get_base_dir(), f"pred_results_log_backup_{ts}.csv")
                        log_df.to_csv(backup_path, index=False)

                        def _is_missing(v):
                            if v is None:
                                return True
                            if isinstance(v, str) and v.strip().lower() in ("", "nan", "none"):
                                return True
                            try:
                                return bool(pd.isna(v))
                            except Exception:
                                return False

                        if "RES_FTHG" in log_df.columns and "RES_FTAG" in log_df.columns:
                            hg = pd.to_numeric(log_df["RES_FTHG"], errors="coerce")
                            ag = pd.to_numeric(log_df["RES_FTAG"], errors="coerce")
                            valid = hg.notna() & ag.notna()
                            if valid.any():
                                if "RES_RESULT_1X2" not in log_df.columns:
                                    log_df["RES_RESULT_1X2"] = np.nan
                                if "RES_RESULT_OU" not in log_df.columns:
                                    log_df["RES_RESULT_OU"] = np.nan
                                if "RES_RESULT_BTTS" not in log_df.columns:
                                    log_df["RES_RESULT_BTTS"] = np.nan

                                res_1x2 = pd.Series(
                                    np.where(hg > ag, "HOME", np.where(hg < ag, "AWAY", "DRAW")),
                                    index=log_df.index,
                                    dtype="object",
                                )
                                tg = hg + ag
                                res_ou = pd.Series(
                                    np.where(tg >= 3, "Over2.5", "Under2.5"),
                                    index=log_df.index,
                                    dtype="object",
                                )
                                res_btts = pd.Series(
                                    np.where((hg > 0) & (ag > 0), "BTTS Yes", "BTTS No"),
                                    index=log_df.index,
                                    dtype="object",
                                )

                                mask_1x2 = valid & log_df["RES_RESULT_1X2"].map(_is_missing)
                                mask_ou = valid & log_df["RES_RESULT_OU"].map(_is_missing)
                                mask_btts = valid & log_df["RES_RESULT_BTTS"].map(_is_missing)
                                log_df.loc[mask_1x2, "RES_RESULT_1X2"] = res_1x2[mask_1x2]
                                log_df.loc[mask_ou, "RES_RESULT_OU"] = res_ou[mask_ou]
                                log_df.loc[mask_btts, "RES_RESULT_BTTS"] = res_btts[mask_btts]
                                log_df["RES_UPDATED_AT"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_df.to_csv(log_path, index=False)
                                try:
                                    log_df = pd.read_csv(log_path)
                                except Exception:
                                    pass
                                st.success("RES_ kolonlari yeniden hesaplandi.")
                            else:
                                st.info("RES_FTHG/RES_FTAG dolu satir yok.")
                        else:
                            st.warning("RES_FTHG/RES_FTAG kolonlari yok.")
                except Exception as e:
                    st.error(f"RES_ yeniden hesaplama hatasi: {e}")

            if x_col.button("Tum logda X yeniden hesapla (yalniz PICK_FINAL)", key="btn_recalc_x"):
                try:
                    if log_df.empty:
                        st.warning("Log bos.")
                    else:
                        def _prob01(v):
                            try:
                                f = float(v)
                                if not np.isfinite(f):
                                    return np.nan
                                return f / 100.0 if f > 1.0 else f
                            except Exception:
                                return np.nan

                        def _fnum(v):
                            try:
                                f = float(v)
                                return f if np.isfinite(f) else np.nan
                            except Exception:
                                return np.nan

                        def _draw_odd(row_obj):
                            odd = _fnum(row_obj.get("DrawOdd"))
                            if not np.isfinite(odd):
                                odd = _fnum(row_obj.get("Odds_Open_Draw"))
                            if not np.isfinite(odd):
                                sim_imp = _prob01(row_obj.get("SIM_IMP_DRAW"))
                                if np.isfinite(sim_imp) and sim_imp > 0:
                                    odd = 1.0 / sim_imp
                            return odd

                        p_draw = log_df.get("Sim_Draw%", pd.Series(index=log_df.index, dtype=float))
                        p_draw = p_draw.apply(_prob01)
                        if p_draw.isna().all() and "SIM_IMP_DRAW" in log_df.columns:
                            p_draw = log_df["SIM_IMP_DRAW"].apply(_prob01)
                        if p_draw.isna().all() and "Poisson_Draw_Pct" in log_df.columns:
                            p_draw = log_df["Poisson_Draw_Pct"].apply(_prob01)

                        p_over = log_df.get("P_OVER25_ADJ", pd.Series(index=log_df.index, dtype=float))
                        p_over = p_over.apply(_prob01)
                        if p_over.isna().all() and "Sim_Over25%" in log_df.columns:
                            p_over = log_df["Sim_Over25%"].apply(_prob01)

                        draw_odd = log_df.apply(_draw_odd, axis=1)

                        x_min_draw = 0.30
                        x_min_draw_odd = 3.0
                        x_max_over = 0.58
                        gate = (
                            p_draw.notna()
                            & draw_odd.notna()
                            & p_over.notna()
                            & (p_draw >= x_min_draw)
                            & (draw_odd >= x_min_draw_odd)
                            & (p_over <= x_max_over)
                        )
                        updated = int(gate.sum())
                        log_df.loc[gate, "PICK_FINAL"] = "X"
                        log_df.to_csv(log_path, index=False)
                        try:
                            log_df = pd.read_csv(log_path)
                        except Exception:
                            pass
                        st.success(f"X yeniden hesaplandi: {updated} satir guncellendi.")
                except Exception as e:
                    st.error(f"X yeniden hesaplama hatasi: {e}")

            if rebuild_col.button("Backtest'ten Logu Yeniden Uret (tam)", key="btn_rebuild_from_bt"):
                try:
                    base_dir = _get_base_dir()
                    backtest_path = os.path.join(base_dir, "Backtest-Out", "backtest_predictions.csv")
                    if not os.path.exists(backtest_path):
                        st.warning("Backtest-Out/backtest_predictions.csv bulunamadi.")
                    else:
                        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = os.path.join(base_dir, f"pred_results_log_backup_{ts}.csv")
                        if not log_df.empty:
                            log_df.to_csv(backup_path, index=False)
                        back_df = pd.read_csv(backtest_path)
                        for c in log_schema_cols:
                            if c not in back_df.columns:
                                back_df[c] = np.nan
                        back_df = back_df[log_schema_cols].copy()
                        for c in ["RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS"]:
                            if c not in back_df.columns:
                                back_df[c] = np.nan
                        if not log_df.empty and "Match_ID" in log_df.columns:
                            backup_map = log_df.set_index("Match_ID")
                            for c in ["RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS", "RES_UPDATED_AT"]:
                                if c in backup_map.columns:
                                    back_df[c] = back_df[c].where(
                                        back_df[c].notna(), back_df["Match_ID"].map(backup_map[c])
                                    )
                        back_df.to_csv(log_path, index=False)
                        try:
                            log_df = pd.read_csv(log_path)
                        except Exception:
                            pass
                        st.success("Backtest logu yenilendi (tam, merge ile).")
                except Exception as e:
                    st.error(f"Backtest log yenileme hatasi: {e}")

            # Results import
            results_upload = st.file_uploader(
                "Results import (Date, League, HomeTeam, AwayTeam, FTHG, FTAG veya Match_ID)",
                type="csv",
                key="results_import_csv",
            )
            if results_upload is not None and st.button("Results import uygula", key="btn_results_import_apply"):
                try:
                    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = os.path.join(_get_base_dir(), f"pred_results_log_backup_{ts}.csv")
                    if not log_df.empty:
                        log_df.to_csv(backup_path, index=False)
                    imp = pd.read_csv(results_upload)
                    if "Match_ID" not in imp.columns:
                        need = {"Date", "League", "HomeTeam", "AwayTeam"}
                        if not need.issubset(imp.columns):
                            st.error("Match_ID yok ve Date/League/HomeTeam/AwayTeam eksik.")
                            imp = None
                        else:
                            tmp = imp.copy()
                            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
                            tmp["Match_ID"] = tmp.apply(DataEngine.make_match_id, axis=1)
                            imp = tmp
                    if imp is not None and "Match_ID" in imp.columns:
                        fthg_col = "FTHG" if "FTHG" in imp.columns else "FT_Score_Home" if "FT_Score_Home" in imp.columns else None
                        ftag_col = "FTAG" if "FTAG" in imp.columns else "FT_Score_Away" if "FT_Score_Away" in imp.columns else None
                        if fthg_col is None or ftag_col is None:
                            st.error("FTHG/FTAG kolonlari bulunamadi.")
                        else:
                            imp["RES_FTHG"] = pd.to_numeric(imp[fthg_col], errors="coerce")
                            imp["RES_FTAG"] = pd.to_numeric(imp[ftag_col], errors="coerce")
                            imp = imp.dropna(subset=["RES_FTHG", "RES_FTAG"])
                            if imp.empty:
                                st.warning("Import dosyasinda skor yok.")
                            else:
                                if "Match_ID" not in log_df.columns:
                                    log_df["Match_ID"] = (
                                        log_df.get("Date", "").astype(str)
                                        + "|" + log_df.get("League", "").astype(str)
                                        + "|" + log_df.get("HomeTeam", "").astype(str)
                                        + "|" + log_df.get("AwayTeam", "").astype(str)
                                    )
                                idx = imp.set_index("Match_ID")
                                for c in ["RES_FTHG", "RES_FTAG"]:
                                    log_df[c] = log_df["Match_ID"].map(idx[c]).fillna(log_df.get(c, np.nan))
                                hg = pd.to_numeric(log_df["RES_FTHG"], errors="coerce")
                                ag = pd.to_numeric(log_df["RES_FTAG"], errors="coerce")
                                valid = hg.notna() & ag.notna()
                                log_df["RES_RESULT_1X2"] = np.where(hg > ag, "HOME", np.where(hg < ag, "AWAY", "DRAW"))
                                tg = hg + ag
                                log_df["RES_RESULT_OU"] = np.where(tg >= 3, "Over2.5", "Under2.5")
                                log_df["RES_RESULT_BTTS"] = np.where((hg > 0) & (ag > 0), "BTTS Yes", "BTTS No")
                                log_df.loc[~valid, ["RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS"]] = np.nan
                                log_df["RES_UPDATED_AT"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_df.to_csv(log_path, index=False)
                                try:
                                    log_df = pd.read_csv(log_path)
                                except Exception:
                                    pass
                                st.success("Results import uygulandi.")
                except Exception as e:
                    st.error(f"Results import hatasi: {e}")

            # Log recompute (optional full recompute)
            full_recompute = st.checkbox(
                "Logu tamamen yeniden hesapla (yavas, tum satirlar)",
                value=False,
                key="chk_full_recompute",
            )
            if st.button("Logu yeniden hesapla", key="btn_log_recompute"):
                try:
                    if log_df.empty:
                        st.warning("Log bos.")
                    else:
                        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = os.path.join(_get_base_dir(), f"pred_results_log_backup_{ts}.csv")
                        log_df.to_csv(backup_path, index=False)

                        if full_recompute:
                            params = debug.get("chosen_params", {}) if isinstance(debug, dict) else {}
                            new_pick = []
                            new_group = []
                            new_switch = []
                            for _, row in log_df.iterrows():
                                best = pick_best_market_synth(row, params=params, debug_context=debug)
                                if isinstance(best, dict):
                                    new_pick.append(best.get("market", row.get("PICK_FINAL", "")))
                                    new_group.append(best.get("final_group", best.get("group", row.get("PICK_GROUP", ""))))
                                    new_switch.append(best.get("switch_reason", ""))
                                else:
                                    new_pick.append(row.get("PICK_FINAL", ""))
                                    new_group.append(row.get("PICK_GROUP", ""))
                                    new_switch.append("")
                            log_df["PICK_FINAL"] = new_pick
                            log_df["PICK_GROUP"] = new_group
                            if "SWITCH_REASON" in log_df.columns:
                                log_df["SWITCH_REASON"] = new_switch
                            log_df.to_csv(log_path, index=False)
                            try:
                                log_df = pd.read_csv(log_path)
                            except Exception:
                                pass
                            st.success("Log tamamen yeniden hesaplandi.")
                        else:
                            st.info("Hizli mod: otomatik log senkronu kullaniliyor (full recompute kapali).")
                except Exception as e:
                    st.error(f"Log yeniden hesaplama hatasi: {e}")

            # Auto-sync current predictions into log (by Match_ID)
            is_editing_results = bool(st.session_state.get("_results_editing", False))
            if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
                pred_cols = [
                    "Match_ID", "Date", "League", "HomeTeam", "AwayTeam",
                    # pick snapshot
                    "PICK_FINAL", "PICK_GROUP", "PICK_ODD", "PICK_PROB",
                    "PICK_PROB_EFF", "PICK_PROB_ADJ",
                    "PICK_BASE", "PICK_SWITCHED", "SWITCH_REASON",
                    "PICK_SCORE_TOTAL", "PICK_MIS_BONUS", "PICK_SIM_BONUS",
                    "MARKET_WEIGHT", "BONUS_TOTAL", "MW_IMPLIED",
                    # evidence/quality
                    "TRUST_PCT", "SIM_QUALITY", "EFFECTIVE_N",
                    "HAS_CANDS", "CANDS_N", "CANDS_REASON", "FINAL_MKT",
                    # selected candidate details
                    "SEL_SCORE", "SEL_ALPHA", "SEL_SIM", "SEL_P_ADJ",
                    "SEL_MISPRICING", "SEL_MIS_BONUS",
                    "POLICY_FLAG", "POLICY_RULE",
                    # signal columns for OU/league tests
                    "P_OVER25_ADJ", "P_BTTSY_ADJ",
                    "TARGET_LEAGUE_GOALS_AVG_10", "TARGET_LEAGUE_GOALS_AVG_50",
                    "SIM_LEAGUE_GOALS_AVG_10", "SIM_LEAGUE_GOALS_AVG_50",
                    "xG_Home", "xG_Away", "xG_Total",
                    "TOP1_MKT", "TOP1_SCORE", "TOP1_PROB_EFF",
                    "TOP2_MKT", "TOP2_SCORE", "TOP2_PROB_EFF",
                    "TOP3_MKT", "TOP3_SCORE", "TOP3_PROB_EFF",
                ]
                cur = pred_df.copy()
                if "Match_ID" not in cur.columns:
                    cur["Match_ID"] = (
                        cur.get("Date", "").astype(str)
                        + "|" + cur.get("League", "").astype(str)
                        + "|" + cur.get("HomeTeam", "").astype(str)
                        + "|" + cur.get("AwayTeam", "").astype(str)
                    )
                for c in pred_cols:
                    if c not in cur.columns:
                        cur[c] = np.nan
                if "xG_Total" not in cur.columns and ("xG_Home" in cur.columns and "xG_Away" in cur.columns):
                    cur["xG_Total"] = pd.to_numeric(cur["xG_Home"], errors="coerce") + pd.to_numeric(cur["xG_Away"], errors="coerce")
                if "SIM_LEAGUE_GOALS_AVG_10" in cur.columns and "TARGET_LEAGUE_GOALS_AVG_10" in cur.columns:
                    if cur["SIM_LEAGUE_GOALS_AVG_10"].isna().all() or log_df.get("SIM_LEAGUE_GOALS_AVG_10", pd.Series()).isna().all():
                        cur["SIM_LEAGUE_GOALS_AVG_10"] = cur["TARGET_LEAGUE_GOALS_AVG_10"]
                if "SIM_LEAGUE_GOALS_AVG_50" in cur.columns and "TARGET_LEAGUE_GOALS_AVG_50" in cur.columns:
                    if cur["SIM_LEAGUE_GOALS_AVG_50"].isna().all() or log_df.get("SIM_LEAGUE_GOALS_AVG_50", pd.Series()).isna().all():
                        cur["SIM_LEAGUE_GOALS_AVG_50"] = cur["TARGET_LEAGUE_GOALS_AVG_50"]
                cur = cur[pred_cols].copy()
                cur["Date"] = pd.to_datetime(cur.get("Date"), errors="coerce").dt.strftime("%Y-%m-%d")

                if isinstance(res, pd.DataFrame) and "Match_ID" in res.columns:
                    res_map = res.set_index("Match_ID")
                    for c in pred_cols:
                        if c in res_map.columns and cur[c].isna().all():
                            cur[c] = cur["Match_ID"].map(res_map[c])
                    # xG column name alignment: res uses XG_* while log uses xG_*
                    if "xG_Home" in cur.columns and cur["xG_Home"].isna().all():
                        if "XG_Home" in res_map.columns:
                            cur["xG_Home"] = cur["Match_ID"].map(res_map["XG_Home"])
                    if "xG_Away" in cur.columns and cur["xG_Away"].isna().all():
                        if "XG_Away" in res_map.columns:
                            cur["xG_Away"] = cur["Match_ID"].map(res_map["XG_Away"])
                    if "xG_Total" in cur.columns and cur["xG_Total"].isna().all():
                        if "XG_Total" in res_map.columns:
                            cur["xG_Total"] = cur["Match_ID"].map(res_map["XG_Total"])
                        elif "XG_Home" in res_map.columns and "XG_Away" in res_map.columns:
                            cur["xG_Total"] = (
                                cur["Match_ID"].map(res_map["XG_Home"])
                                + cur["Match_ID"].map(res_map["XG_Away"])
                            )

                pred_sig = None
                try:
                    sig_cols = [c for c in ["Match_ID", "PICK_FINAL", "PICK_GROUP", "PICK_ODD", "PICK_PROB"] if c in cur.columns]
                    if sig_cols:
                        pred_sig = pd.util.hash_pandas_object(cur[sig_cols], index=False).sum()
                except Exception:
                    pred_sig = None
                last_sig = st.session_state.get("_pred_log_sig")

                backfill_cols = [
                    "P_OVER25_ADJ", "P_BTTSY_ADJ",
                    "TARGET_LEAGUE_GOALS_AVG_10", "TARGET_LEAGUE_GOALS_AVG_50",
                    "SIM_LEAGUE_GOALS_AVG_10", "SIM_LEAGUE_GOALS_AVG_50",
                    "xG_Home", "xG_Away", "xG_Total",
                ]
                force_update = False
                if not log_df.empty:
                    for c in backfill_cols:
                        if c in log_df.columns and c in cur.columns:
                            if log_df[c].isna().all() and cur[c].notna().any():
                                force_update = True
                                break

                if (not is_editing_results) or log_df.empty or force_update:
                    pass
                else:
                    pred_sig = last_sig

                if (pred_sig is None) or (pred_sig != last_sig) or log_df.empty or force_update:
                    if log_df.empty:
                        log_df = cur
                    else:
                        if "Match_ID" not in log_df.columns:
                            log_df["Match_ID"] = (
                                log_df.get("Date", "").astype(str)
                                + "|" + log_df.get("League", "").astype(str)
                                + "|" + log_df.get("HomeTeam", "").astype(str)
                                + "|" + log_df.get("AwayTeam", "").astype(str)
                            )
                        log_df = log_df.set_index("Match_ID", drop=False)
                        cur = cur.set_index("Match_ID", drop=False)
                        log_df.update(cur)
                        missing_idx = cur.index.difference(log_df.index)
                        if len(missing_idx) > 0:
                            log_df = pd.concat([log_df, cur.loc[missing_idx]], axis=0)
                        log_df = log_df.reset_index(drop=True)

                    for c in result_cols:
                        if c not in log_df.columns:
                            log_df[c] = np.nan

                    log_df.to_csv(log_path, index=False)
                    if pred_sig is not None:
                        st.session_state["_pred_log_sig"] = pred_sig

            if log_df.empty:
                st.info("Henuz kayitli tahmin sonucu yok.")
            else:
                log_df["Date"] = pd.to_datetime(log_df.get("Date"), errors="coerce").dt.strftime("%Y-%m-%d")
                date_list = sorted([d for d in log_df["Date"].dropna().unique().tolist()])
                selected_date = st.selectbox("Tarih seç", date_list, index=0) if date_list else ""

                view_df = log_df[log_df["Date"] == selected_date].copy() if selected_date else log_df.copy()

                results_ready = False
                if "RES_FTHG" in log_df.columns and "RES_FTAG" in log_df.columns:
                    results_ready = bool(log_df["RES_FTHG"].notna().any() and log_df["RES_FTAG"].notna().any())

                editing_now = False
                if "results_editor" in st.session_state:
                    _draft = st.session_state.get("results_editor")
                    if isinstance(_draft, pd.DataFrame):
                        cols = [c for c in ["RES_FTHG", "RES_FTAG"] if c in _draft.columns and c in view_df.columns]
                        if cols:
                            a = _draft[cols].reset_index(drop=True)
                            b = view_df[cols].reset_index(drop=True)
                            editing_now = not a.equals(b)
                st.session_state["_results_editing"] = editing_now

                stats_ready_flag = bool(st.session_state.get("_results_stats_ready", results_ready))
                stats_enabled = stats_ready_flag and (not editing_now)

                def _norm_pick(val):
                    s = str(val).strip()
                    s = s.split()[0] if s else s
                    return "X" if s.upper() == "MSX" else s

                def _pick_to_result(row):
                    m = _norm_pick(row.get("PICK_FINAL", ""))
                    g = str(row.get("PICK_GROUP", "") or "")
                    if g == "1X2":
                        return {"MS1": "HOME", "X": "DRAW", "MS2": "AWAY"}.get(m, "")
                    if g == "OU":
                        return m if m in ("Over2.5", "Under2.5") else ""
                    if g == "BTTS":
                        return {"BTTS_Yes": "BTTS Yes", "BTTS_No": "BTTS No"}.get(m, "")
                    return ""

                def _build_hit_df(df):
                    if "RES_FTHG" not in df.columns or "RES_FTAG" not in df.columns:
                        return pd.DataFrame()

                    out = df.copy()
                    hg = pd.to_numeric(out["RES_FTHG"], errors="coerce")
                    ag = pd.to_numeric(out["RES_FTAG"], errors="coerce")
                    valid = hg.notna() & ag.notna()
                    if not valid.any():
                        return pd.DataFrame()

                    if "RES_RESULT_1X2" not in out.columns:
                        out["RES_RESULT_1X2"] = np.nan
                    if "RES_RESULT_OU" not in out.columns:
                        out["RES_RESULT_OU"] = np.nan
                    if "RES_RESULT_BTTS" not in out.columns:
                        out["RES_RESULT_BTTS"] = np.nan

                    res_1x2 = pd.Series(
                        np.where(hg > ag, "HOME", np.where(hg < ag, "AWAY", "DRAW")),
                        index=out.index,
                        dtype="object",
                    )
                    mask_1x2 = valid & out["RES_RESULT_1X2"].isna()
                    out.loc[mask_1x2, "RES_RESULT_1X2"] = res_1x2[mask_1x2]
                    tg = hg + ag
                    res_ou = pd.Series(
                        np.where(tg >= 3, "Over2.5", "Under2.5"),
                        index=out.index,
                        dtype="object",
                    )
                    mask_ou = valid & out["RES_RESULT_OU"].isna()
                    out.loc[mask_ou, "RES_RESULT_OU"] = res_ou[mask_ou]
                    res_btts = pd.Series(
                        np.where((hg > 0) & (ag > 0), "BTTS Yes", "BTTS No"),
                        index=out.index,
                        dtype="object",
                    )
                    mask_btts = valid & out["RES_RESULT_BTTS"].isna()
                    out.loc[mask_btts, "RES_RESULT_BTTS"] = res_btts[mask_btts]

                    out = out.dropna(
                        subset=["RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS"]
                    ).copy()
                    if out.empty:
                        return out

                    out["PICK_RESULT_NORM"] = out.apply(_pick_to_result, axis=1)
                    out["HIT"] = False
                    out.loc[out["PICK_GROUP"] == "1X2", "HIT"] = (
                        out["PICK_RESULT_NORM"] == out["RES_RESULT_1X2"]
                    )
                    out.loc[out["PICK_GROUP"] == "OU", "HIT"] = (
                        out["PICK_RESULT_NORM"] == out["RES_RESULT_OU"]
                    )

                    def _norm_btts(v):
                        s = str(v or "").strip()
                        s = s.replace(" ", "_")
                        su = s.upper()
                        if su in ("BTTS_YES", "BTTSYES"):
                            return "BTTS_Yes"
                        if su in ("BTTS_NO", "BTTSNO"):
                            return "BTTS_No"
                        return s

                    pick_btts = out["PICK_RESULT_NORM"].map(_norm_btts)
                    res_btts = out["RES_RESULT_BTTS"].map(_norm_btts)
                    out.loc[out["PICK_GROUP"] == "BTTS", "HIT"] = (pick_btts == res_btts)

                    if "PICK_ODD" in out.columns:
                        odds = pd.to_numeric(out["PICK_ODD"], errors="coerce")
                        stake = pd.Series(
                            np.where(out.get("POLICY_FLAG", 0).fillna(0).astype(int) == 1, 1.0, 0.5),
                            index=out.index,
                        )
                        out["ROI"] = np.where(out["HIT"], (odds - 1.0) * stake, -1.0 * stake)
                    return out

                def _write_stats_files(df_in):
                    try:
                        out_dir = os.path.join(_get_base_dir(), "Backtest-Out")
                        os.makedirs(out_dir, exist_ok=True)
                        hit_df = _build_hit_df(df_in)
                        if hit_df.empty:
                            return

                        hit_df = hit_df.copy()
                        hit_df["PICK_FINAL_NORM"] = hit_df["PICK_FINAL"].replace({"MSX": "X"})
                        market_map = {
                            "MS1": "1X2",
                            "MS2": "1X2",
                            "X": "1X2",
                            "Over2.5": "OU",
                            "Under2.5": "OU",
                            "BTTS_Yes": "BTTS",
                            "BTTS_No": "BTTS",
                        }
                        matrix = (
                            hit_df.groupby(["League", "PICK_FINAL_NORM"])["HIT"]
                            .agg(["count", "mean"])
                            .reset_index()
                            .rename(columns={"mean": "hit"})
                        )
                        matrix = matrix.rename(columns={"PICK_FINAL_NORM": "PICK_FINAL"})
                        matrix["MARKET"] = matrix["PICK_FINAL"].map(market_map).fillna("OTHER")
                        k_shrink = 3
                        global_by_market = (
                            matrix.groupby("MARKET")
                            .apply(lambda g: np.average(g["hit"], weights=g["count"]))
                            .to_dict()
                        )
                        def _adj_hit(row):
                            gh = global_by_market.get(
                                row["MARKET"],
                                np.average(matrix["hit"], weights=matrix["count"]),
                            )
                            wins = row["hit"] * row["count"]
                            return (wins + k_shrink * gh) / (row["count"] + k_shrink)
                        matrix["hit_adj_mkt"] = matrix.apply(_adj_hit, axis=1)
                        matrix.to_csv(os.path.join(out_dir, "league_picktype_matrix.csv"), index=False)

                        if "POLICY_FLAG" in hit_df.columns:
                            pol = hit_df[hit_df["POLICY_FLAG"] == 1]
                        else:
                            pol = pd.DataFrame()
                        if not pol.empty:
                            pol_matrix = (
                                pol.groupby(["League", "PICK_FINAL_NORM"])["HIT"]
                                .agg(["count", "mean"])
                                .reset_index()
                                .rename(columns={"mean": "hit"})
                            )
                        else:
                            pol_matrix = pd.DataFrame(columns=["League", "PICK_FINAL", "count", "hit"])
                        if "PICK_FINAL_NORM" in pol_matrix.columns:
                            pol_matrix = pol_matrix.rename(columns={"PICK_FINAL_NORM": "PICK_FINAL"})
                        if not pol_matrix.empty:
                            pol_matrix["MARKET"] = pol_matrix["PICK_FINAL"].map(market_map).fillna("OTHER")
                            pol_matrix["hit_adj_mkt"] = pol_matrix.apply(_adj_hit, axis=1)
                        pol_matrix.to_csv(os.path.join(out_dir, "league_picktype_matrix_policy.csv"), index=False)

                        bucket_bins = [0, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84, 1.0]
                        bucket_labels = [
                            "0-14%",
                            "15-28%",
                            "29-42%",
                            "43-56%",
                            "57-70%",
                            "71-84%",
                            "85-100%",
                        ]
                        hit_df = hit_df.merge(
                            matrix[["League", "PICK_FINAL", "hit_adj_mkt"]],
                            left_on=["League", "PICK_FINAL_NORM"],
                            right_on=["League", "PICK_FINAL"],
                            how="left",
                        )
                        hit_df["HIT_BUCKET"] = pd.cut(
                            hit_df["hit_adj_mkt"],
                            bins=bucket_bins,
                            labels=bucket_labels,
                            include_lowest=True,
                            right=True,
                        )
                        grp = hit_df.groupby("HIT_BUCKET")["HIT"].agg(["count", "mean"]).reset_index()
                        if "POLICY_FLAG" in hit_df.columns:
                            pol = hit_df[hit_df["POLICY_FLAG"] == 1]
                        else:
                            pol = pd.DataFrame()
                        pol_grp = (
                            pol.groupby("HIT_BUCKET")["HIT"].agg(["count", "mean"]).reset_index()
                            if not pol.empty
                            else pd.DataFrame(columns=["HIT_BUCKET", "count", "mean"])
                        )
                        rows = []
                        for lbl in bucket_labels:
                            g = grp[grp["HIT_BUCKET"] == lbl]
                            p = pol_grp[pol_grp["HIT_BUCKET"] == lbl]
                            rows.append(
                                {
                                    "HIT_BUCKET": lbl,
                                    "count": int(g["count"].iloc[0]) if not g.empty else 0,
                                    "mean": float(g["mean"].iloc[0]) if not g.empty and pd.notna(g["mean"].iloc[0]) else np.nan,
                                    "policy_count": int(p["count"].iloc[0]) if not p.empty else 0,
                                    "policy_mean": float(p["mean"].iloc[0]) if not p.empty and pd.notna(p["mean"].iloc[0]) else np.nan,
                                }
                            )
                        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "policy_bucket_stats.csv"), index=False)
                    except Exception:
                        pass

                if isinstance(log_df, pd.DataFrame) and not log_df.empty:
                    _write_stats_files(log_df)

                if stats_enabled:
                    hit_view = _build_hit_df(view_df)
                    if not hit_view.empty:
                        view_df = view_df.copy()
                        view_df["HIT_ICON"] = ""
                        view_df.loc[hit_view.index, "HIT_ICON"] = np.where(hit_view["HIT"], "O", "X")
                        # ROI: 1 birim bahis (doğruysa odd-1, yanlışsa -1)
                        if "PICK_ODD" in view_df.columns:
                            odds = pd.to_numeric(view_df["PICK_ODD"], errors="coerce")
                            stake = pd.Series(
                                np.where(view_df.get("POLICY_FLAG", 0).fillna(0).astype(int) == 1, 1.0, 0.5),
                                index=view_df.index,
                            )
                            view_df["ROI"] = np.nan
                            view_df.loc[hit_view.index, "ROI"] = np.where(
                                hit_view["HIT"],
                                (odds.loc[hit_view.index] - 1.0) * stake.loc[hit_view.index],
                                -1.0 * stake.loc[hit_view.index],
                            )
                if "POLICY_FLAG" not in view_df.columns:
                    view_df["POLICY_FLAG"] = 0
                view_df["POLICY_ICON"] = np.where(view_df["POLICY_FLAG"].fillna(0) == 1, "✨", "")
                if "SOURCE" not in view_df.columns:
                    view_df["SOURCE"] = "LIVE"
                if "LOCKED" not in view_df.columns:
                    view_df["LOCKED"] = 0

                show_cols = [
                    "Match_ID", "Date", "League", "POLICY_ICON", "HomeTeam", "AwayTeam",
                    "PICK_FINAL", "PICK_GROUP", "PICK_ODD", "PICK_PROB", "HIT_ICON", "ROI",
                    "RES_FTHG", "RES_FTAG",
                    "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS",
                ]
                show_cols = [c for c in show_cols if c in view_df.columns]
                def _mark_results_editing():
                    st.session_state["_results_editing"] = True

                view_live = view_df[view_df["SOURCE"].fillna("LIVE") != "BACKTEST"].copy()
                if not view_live.empty:
                    editable = st.data_editor(
                        view_live[show_cols],
                        use_container_width=True,
                        disabled=[c for c in show_cols if c not in ("RES_FTHG", "RES_FTAG")],
                        key="results_editor",
                        on_change=_mark_results_editing,
                    )
                else:
                    st.info("Live mac bulunamadi (BACKTEST satirlari kilitli).")

                if st.button("Sonuclari Kaydet"):
                    edited = view_live.copy()
                    state_edit = st.session_state.get("results_editor")
                    if isinstance(state_edit, dict) and "edited_rows" in state_edit:
                        for row_idx, changes in state_edit.get("edited_rows", {}).items():
                            try:
                                i = int(row_idx)
                            except Exception:
                                continue
                            if i < 0 or i >= len(edited.index):
                                continue
                            row_key = edited.index[i]
                            for col, val in (changes or {}).items():
                                if col in edited.columns:
                                    edited.at[row_key, col] = val
                    if "RES_FTHG" in edited.columns and "RES_FTAG" in edited.columns:
                        hg = pd.to_numeric(edited["RES_FTHG"], errors="coerce")
                        ag = pd.to_numeric(edited["RES_FTAG"], errors="coerce")
                        valid = hg.notna() & ag.notna()
                        edited["RES_RESULT_1X2"] = np.nan
                        edited.loc[valid, "RES_RESULT_1X2"] = np.where(
                            hg[valid] > ag[valid],
                            "HOME",
                            np.where(hg[valid] < ag[valid], "AWAY", "DRAW"),
                        )
                        tg = hg + ag
                        edited["RES_RESULT_OU"] = np.nan
                        edited.loc[valid, "RES_RESULT_OU"] = np.where(tg[valid] >= 3, "Over2.5", "Under2.5")
                        edited["RES_RESULT_BTTS"] = np.nan
                        edited.loc[valid, "RES_RESULT_BTTS"] = np.where(
                            (hg[valid] > 0) & (ag[valid] > 0), "BTTS Yes", "BTTS No"
                        )
                        edited["RES_UPDATED_AT"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

                    if "Match_ID" in edited.columns and "Match_ID" in log_df.columns:
                        if "SOURCE" not in log_df.columns:
                            log_df["SOURCE"] = "LIVE"
                        if "LOCKED" not in log_df.columns:
                            log_df["LOCKED"] = 0
                        log_df = log_df.set_index("Match_ID", drop=False)
                        edited = edited.set_index("Match_ID", drop=False)
                        log_df.update(edited)
                        log_df = log_df.reset_index(drop=True)
                        log_df.to_csv(log_path, index=False)
                        # Reload to ensure UI reflects persisted data
                        try:
                            log_df = pd.read_csv(log_path)
                        except Exception:
                            pass
                        try:
                            if "POLICY_FLAG" in log_df.columns and "POLICY_RULE" in log_df.columns:
                                policy_ctx = _load_policy_context(log_path)
                                pf = []
                                pr = []
                                for _, row in log_df.iterrows():
                                    flag, rule = _policy_flag(row, policy_ctx)
                                    pf.append(int(flag))
                                    pr.append(rule)
                                log_df["POLICY_FLAG"] = pf
                                log_df["POLICY_RULE"] = pr
                            _write_stats_files(log_df)
                            log_df.to_csv(log_path, index=False)
                            try:
                                log_df = pd.read_csv(log_path)
                            except Exception:
                                pass
                        except Exception as e:
                            st.warning(f"Policy/stat guncelleme hatasi: {e}")
                        st.success("Sonuclar kaydedildi.")
                        st.session_state["_results_editing"] = False
                        st.session_state["_results_stats_ready"] = True
                        if "results_editor" in st.session_state:
                            st.session_state.pop("results_editor", None)
                    else:
                        st.warning("Kaydetme atlandi: Match_ID bulunamadi.")

                res_ready_all = _build_hit_df(log_df) if stats_enabled else pd.DataFrame()
                res_ready_view = _build_hit_df(view_df) if stats_enabled else pd.DataFrame()

                if stats_enabled and not res_ready_all.empty:
                    st.markdown("### Istatistik Ozeti")
                    day_label = selected_date or "Son gun"
                    if selected_date and not res_ready_view.empty:
                        day_df = res_ready_view
                    else:
                        day_df = res_ready_all

                    def _rate(df_in):
                        if df_in is None or df_in.empty:
                            return np.nan, 0
                        total_hits = int(df_in["HIT"].sum())
                        total_cnt = int(len(df_in))
                        return (total_hits / total_cnt) if total_cnt else np.nan, total_cnt

                    day_rate, day_n = _rate(day_df)
                    all_rate, all_n = _rate(res_ready_all)
                    pol_df = res_ready_all[res_ready_all.get("POLICY_FLAG", 0) == 1] if "POLICY_FLAG" in res_ready_all.columns else pd.DataFrame()
                    pol_rate, pol_n = _rate(pol_df)

                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"Gunluk Basari ({day_label})", f"{day_rate:.1%}" if np.isfinite(day_rate) else "n/a", f"n={day_n}")
                    m2.metric("Genel Basari", f"{all_rate:.1%}" if np.isfinite(all_rate) else "n/a", f"n={all_n}")
                    m3.metric("Policy Basari", f"{pol_rate:.1%}" if np.isfinite(pol_rate) else "n/a", f"n={pol_n}")

                    weekly = res_ready_all.copy()
                    weekly["Date"] = pd.to_datetime(weekly.get("Date"), errors="coerce")
                    weekly = weekly[weekly["Date"].notna()].copy()
                    if not weekly.empty:
                        weekly["Week"] = weekly["Date"].dt.to_period("W-MON").dt.start_time
                        weekly_g = weekly.groupby("Week").agg(count=("HIT", "count"), hit=("HIT", "mean")).reset_index()
                        weekly_g = weekly_g.sort_values("Week").tail(12)
                        st.line_chart(weekly_g.set_index("Week")[["hit"]], use_container_width=True)
                        st.caption("Haftalik basari (son 12 hafta)")

                    t1, t2, t3, t4 = st.tabs(["Pick Type", "League Matrix", "Policy Matrix", "Detay Tablolar"])

                    with t1:
                        grp_summary = res_ready_all.groupby("PICK_GROUP").agg(
                            count=("HIT", "count"),
                            sum=("HIT", "sum"),
                            roi=("ROI", "mean"),
                        )
                        if not grp_summary.empty:
                            grp_summary["rate"] = grp_summary["sum"] / grp_summary["count"]
                            st.dataframe(
                                grp_summary.reset_index().style.format({"rate": "{:.1%}", "roi": "{:+.2f}"}),
                                use_container_width=True,
                            )

                    with t2:
                        matrix_path = os.path.join(_get_base_dir(), "Backtest-Out", "league_picktype_matrix.csv")
                        if os.path.exists(matrix_path):
                            matrix_df = pd.read_csv(matrix_path)
                            keep = [c for c in ["League", "PICK_FINAL", "count", "hit", "hit_adj_mkt"] if c in matrix_df.columns]
                            view_m = matrix_df[keep].sort_values(["count"], ascending=False).head(15)
                            st.dataframe(view_m, use_container_width=True)
                        else:
                            st.info("league_picktype_matrix.csv bulunamadi.")

                    with t3:
                        matrix_path = os.path.join(_get_base_dir(), "Backtest-Out", "league_picktype_matrix_policy.csv")
                        if os.path.exists(matrix_path):
                            matrix_df = pd.read_csv(matrix_path)
                            keep = [c for c in ["League", "PICK_FINAL", "count", "hit", "hit_adj_mkt"] if c in matrix_df.columns]
                            view_m = matrix_df[keep].sort_values(["count"], ascending=False).head(15)
                            st.dataframe(view_m, use_container_width=True)
                        else:
                            st.info("league_picktype_matrix_policy.csv bulunamadi.")

                    with t4:
                        pick_order = [
                            "MS1", "X", "MS2",
                            "Over2.5", "Under2.5",
                            "BTTS_Yes", "BTTS_No",
                        ]

                        def _pick_type_summary(df_in):
                            if df_in is None or df_in.empty or "PICK_FINAL" not in df_in.columns:
                                return pd.DataFrame()
                            total_n = float(len(df_in))
                            out = df_in.groupby("PICK_FINAL").agg(
                                count=("HIT", "count"),
                                hit=("HIT", "mean"),
                                roi=("ROI", "mean"),
                                odd_avg=("PICK_ODD", "mean"),
                            ).reset_index()
                            out["coverage"] = out["count"] / total_n if total_n else np.nan
                            out["PICK_FINAL"] = out["PICK_FINAL"].astype(str)
                            out = out.set_index("PICK_FINAL").reindex(pick_order).reset_index()
                            return out

                        if not res_ready_view.empty:
                            st.write("Gunluk pick-type (7 tur) ozeti")
                            daily_pick = _pick_type_summary(res_ready_view)
                            if not daily_pick.empty:
                                st.dataframe(
                                    daily_pick.style.format(
                                        {
                                            "hit": "{:.1%}",
                                            "roi": "{:+.2f}",
                                            "odd_avg": "{:.2f}",
                                            "coverage": "{:.1%}",
                                        }
                                    ),
                                    use_container_width=True,
                                )

                            st.write("Gunluk pick group dagilimi")
                            grp_summary = res_ready_view.groupby("PICK_GROUP").agg(
                                count=("HIT", "count"),
                                sum=("HIT", "sum"),
                                roi=("ROI", "mean"),
                            )
                            if not grp_summary.empty:
                                grp_summary["rate"] = grp_summary["sum"] / grp_summary["count"]
                                st.dataframe(
                                    grp_summary.reset_index().style.format({"rate": "{:.1%}", "roi": "{:+.2f}"}),
                                    use_container_width=True,
                                )
                            if "POLICY_FLAG" in res_ready_view.columns:
                                pol_view = res_ready_view[res_ready_view["POLICY_FLAG"] == 1]
                                st.write("Gunluk policy pick group dagilimi")
                                pol_grp = pol_view.groupby("PICK_GROUP").agg(
                                    count=("HIT", "count"),
                                    sum=("HIT", "sum"),
                                    roi=("ROI", "mean"),
                                )
                                if not pol_grp.empty:
                                    pol_grp["rate"] = pol_grp["sum"] / pol_grp["count"]
                                    st.dataframe(
                                        pol_grp.reset_index().style.format({"rate": "{:.1%}", "roi": "{:+.2f}"}),
                                    use_container_width=True,
                                )

                            if "POLICY_FLAG" in res_ready_view.columns:
                                st.write("Gunluk policy vs non-policy (pick-type)")
                                pol_daily = res_ready_view[res_ready_view["POLICY_FLAG"] == 1]
                                nopol_daily = res_ready_view[res_ready_view["POLICY_FLAG"] == 0]
                                pol_tbl = _pick_type_summary(pol_daily).set_index("PICK_FINAL")
                                np_tbl = _pick_type_summary(nopol_daily).set_index("PICK_FINAL")
                                cmp = pd.DataFrame(index=pick_order)
                                cmp["policy_hit"] = pol_tbl.get("hit")
                                cmp["policy_roi"] = pol_tbl.get("roi")
                                cmp["policy_cov"] = pol_tbl.get("coverage")
                                cmp["nonpol_hit"] = np_tbl.get("hit")
                                cmp["nonpol_roi"] = np_tbl.get("roi")
                                cmp["nonpol_cov"] = np_tbl.get("coverage")
                                st.dataframe(
                                    cmp.reset_index().rename(columns={"index": "PICK_FINAL"}).style.format(
                                        {
                                            "policy_hit": "{:.1%}",
                                            "policy_roi": "{:+.2f}",
                                            "policy_cov": "{:.1%}",
                                            "nonpol_hit": "{:.1%}",
                                            "nonpol_roi": "{:+.2f}",
                                            "nonpol_cov": "{:.1%}",
                                        }
                                    ),
                                    use_container_width=True,
                                )

                        st.write("Genel pick group dagilimi")
                        grp_summary = res_ready_all.groupby("PICK_GROUP").agg(
                            count=("HIT", "count"),
                            sum=("HIT", "sum"),
                            roi=("ROI", "mean"),
                        )
                        if not grp_summary.empty:
                            grp_summary["rate"] = grp_summary["sum"] / grp_summary["count"]
                            st.dataframe(
                                grp_summary.reset_index().style.format({"rate": "{:.1%}", "roi": "{:+.2f}"}),
                                use_container_width=True,
                            )
                        if "POLICY_FLAG" in res_ready_all.columns:
                            pol_all = res_ready_all[res_ready_all["POLICY_FLAG"] == 1]
                            st.write("Genel policy pick group dagilimi")
                            pol_grp = pol_all.groupby("PICK_GROUP").agg(
                                count=("HIT", "count"),
                                sum=("HIT", "sum"),
                                roi=("ROI", "mean"),
                            )
                            if not pol_grp.empty:
                                pol_grp["rate"] = pol_grp["sum"] / pol_grp["count"]
                                st.dataframe(
                                    pol_grp.reset_index().style.format({"rate": "{:.1%}", "roi": "{:+.2f}"}),
                                    use_container_width=True,
                                )

                        st.write("Genel pick-type (7 tur) ozeti")
                        all_pick = _pick_type_summary(res_ready_all)
                        if not all_pick.empty:
                            st.dataframe(
                                all_pick.style.format(
                                    {
                                        "hit": "{:.1%}",
                                        "roi": "{:+.2f}",
                                        "odd_avg": "{:.2f}",
                                        "coverage": "{:.1%}",
                                    }
                                ),
                                use_container_width=True,
                            )

                        if "POLICY_FLAG" in res_ready_all.columns:
                            st.write("Genel policy vs non-policy (pick-type)")
                            pol_all = res_ready_all[res_ready_all["POLICY_FLAG"] == 1]
                            nopol_all = res_ready_all[res_ready_all["POLICY_FLAG"] == 0]
                            pol_tbl = _pick_type_summary(pol_all).set_index("PICK_FINAL")
                            np_tbl = _pick_type_summary(nopol_all).set_index("PICK_FINAL")
                            cmp = pd.DataFrame(index=pick_order)
                            cmp["policy_hit"] = pol_tbl.get("hit")
                            cmp["policy_roi"] = pol_tbl.get("roi")
                            cmp["policy_cov"] = pol_tbl.get("coverage")
                            cmp["nonpol_hit"] = np_tbl.get("hit")
                            cmp["nonpol_roi"] = np_tbl.get("roi")
                            cmp["nonpol_cov"] = np_tbl.get("coverage")
                            st.dataframe(
                                cmp.reset_index().rename(columns={"index": "PICK_FINAL"}).style.format(
                                    {
                                        "policy_hit": "{:.1%}",
                                        "policy_roi": "{:+.2f}",
                                        "policy_cov": "{:.1%}",
                                        "nonpol_hit": "{:.1%}",
                                        "nonpol_roi": "{:+.2f}",
                                        "nonpol_cov": "{:.1%}",
                                    }
                                ),
                                use_container_width=True,
                            )

                        st.write("Haftalik basari trendi (genel vs policy)")
                        weekly = res_ready_all.copy()
                        weekly["Date"] = pd.to_datetime(weekly.get("Date"), errors="coerce")
                        weekly = weekly[weekly["Date"].notna()].copy()
                        if not weekly.empty:
                            weekly["Week"] = weekly["Date"].dt.to_period("W-MON").dt.start_time
                            g_all = weekly.groupby("Week").agg(hit=("HIT", "mean")).reset_index()
                            if "POLICY_FLAG" in weekly.columns:
                                g_pol = (
                                    weekly[weekly["POLICY_FLAG"] == 1]
                                    .groupby("Week")
                                    .agg(policy_hit=("HIT", "mean"))
                                    .reset_index()
                                )
                                g_all = g_all.merge(g_pol, on="Week", how="left")
                            g_all = g_all.sort_values("Week").tail(16)
                            st.line_chart(g_all.set_index("Week"), use_container_width=True)

                        st.write("Ligde yogunluk (Top 10) - genel")
                        league_all = res_ready_all.groupby("League").agg(
                            count=("HIT", "count"),
                            hit=("HIT", "mean"),
                        )
                        if not league_all.empty:
                            league_all = league_all.sort_values("count", ascending=False).head(10).reset_index()
                            st.dataframe(
                                league_all.style.format({"hit": "{:.1%}"}),
                                use_container_width=True,
                            )

                        if "POLICY_FLAG" in res_ready_all.columns:
                            st.write("Ligde yogunluk (Top 10) - policy")
                            league_pol = res_ready_all[res_ready_all["POLICY_FLAG"] == 1].groupby("League").agg(
                                count=("HIT", "count"),
                                hit=("HIT", "mean"),
                            )
                            if not league_pol.empty:
                                league_pol = league_pol.sort_values("count", ascending=False).head(10).reset_index()
                                st.dataframe(
                                    league_pol.style.format({"hit": "{:.1%}"}),
                                    use_container_width=True,
                                )
    else:

        st.warning("Analiz sonucunda gosterilecek mac yok.")



# -----------------------------------------------------------------------------#

# Main App

# -----------------------------------------------------------------------------#

def main():

    st.sidebar.header("1) CSV Yükle")

    past_file = st.sidebar.file_uploader("Geçmis Maçlar (past) CSV", type="csv")

    future_file = st.sidebar.file_uploader("Gelecek Maçlar (future) CSV", type="csv")



    st.sidebar.header("2) KNN Ayarlari")

    k_same = st.sidebar.slider("Same-league neighbors (K_SAME)", 5, 80, 30, 5)

    k_global = st.sidebar.slider("Global neighbors (K_GLOBAL)", 5, 80, 20, 5)

    same_league_mode = st.sidebar.checkbox("Use same-league pool first", value=True)

    min_same_found = st.sidebar.slider("Min same-league matches", 5, 50, 12, 1)

    null_threshold = st.sidebar.slider("Future missing tolerance", 0.0, 1.0, 0.8, 0.05)

    # Varsayilani agresif olmasin: 0.25 daha gerçekçi (senin quantile teshisine uyuyor)

    conf_quality_floor = st.sidebar.slider("CONF quality floor", 0.0, 1.0, 0.25, 0.01)



    st.sidebar.header("3) Feature Weights")

    weight_xg = st.sidebar.slider("xG weight", 1.0, 3.0, 2.0, 0.1)

    weight_elo = st.sidebar.slider("Elo weight", 1.0, 2.0, 1.2, 0.1)

    weight_league_pos = st.sidebar.slider("League Pos weight", 1.0, 2.0, 1.1, 0.1)

    weight_config = {

        "xg": weight_xg,

        "elo": weight_elo,

        "league_pos": weight_league_pos,

        "default": 1.0,

    }



    st.sidebar.header("4) Feature Mode")

    feature_mode = st.sidebar.radio("Feature Selection", ["Auto (all numeric)", "Manual pick"], index=0)



    st.sidebar.header("5) Debug")

    debug_mode = st.sidebar.checkbox("Show full exceptions", value=False)
    st.session_state["DEBUG_MODE"] = debug_mode



    if past_file and future_file:

        try:

            # Önce temp standardize for feature selection

            raw_past_temp = DataEngine.load_csv(past_file)

            raw_future_temp = DataEngine.load_csv(future_file)

            past_temp = DataEngine.standardize(raw_past_temp, is_past=True)

            future_temp = DataEngine.standardize(raw_future_temp, is_past=False)

            common_cols = sorted(set(past_temp.columns).intersection(future_temp.columns))

            possible_features = []

            for col in common_cols:

                if pd.api.types.is_numeric_dtype(past_temp[col]) and pd.api.types.is_numeric_dtype(future_temp[col]):

                    if col not in DataEngine.FEATURE_EXCLUDE and not any(col.startswith(prefix) for prefix in DataEngine.EXCLUDE_PREFIXES):

                        possible_features.append(col)

            if feature_mode == "Manual pick":

                default_selection = [col for col in possible_features if col in ("xG_Home", "xG_Away") or future_temp[col].isna().mean() <= null_threshold][:10]  # örnek default

                manual_features = st.sidebar.multiselect("Select features", possible_features, default=default_selection)

                if "xG_Home" not in manual_features:

                    manual_features.append("xG_Home")

                    st.sidebar.warning("xG_Home zorunlu olarak eklendi.")

                if "xG_Away" not in manual_features:

                    manual_features.append("xG_Away")

                    st.sidebar.warning("xG_Away zorunlu olarak eklendi.")

            else:

                manual_features = None

            params = {

                "k_same": k_same,

                "k_global": k_global,

                "same_league_mode": same_league_mode,

                "min_same_found": min_same_found,

                "null_threshold": null_threshold,

                "weight_config": weight_config,

                "manual_features": manual_features,

                "conf_quality_floor": conf_quality_floor,

            }

            # Persist latest params for backtest sync
            try:
                params_path = os.path.join(_get_base_dir(), "params_latest.json")
                with open(params_path, "w", encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # Cache key olustur

            past_bytes = past_file.getvalue()

            future_bytes = future_file.getvalue()

            key = {

                "past": _sha1_bytes(past_bytes),

                "future": _sha1_bytes(future_bytes),

                "params": _stable_params_key(params),

            }

            # Cache kontrol

            if st.session_state.get("_pipeline_key") != key:

                with st.spinner("Pipeline çalisiyor..."):

                    pred_df, res, knn, past, future, debug = run_pipeline(past_file, future_file, params)

                st.session_state["_pipeline_key"] = key

                st.session_state["_pred_df"] = pred_df

                st.session_state["_res"] = res

                st.session_state["_knn"] = knn

                st.session_state["_past"] = past

                st.session_state["_future"] = future

                st.session_state["_debug"] = debug

            else:

                pred_df = st.session_state["_pred_df"]

                res = st.session_state["_res"]

                knn = st.session_state["_knn"]

                past = st.session_state["_past"]

                future = st.session_state["_future"]

                debug = st.session_state["_debug"]

            st.success("Analiz tamamlandi.")

            render_tabs(pred_df, res, knn, past, future, debug)

        except Exception as exc:

            st.error("Sema/isleme hatasi (fail-fast):")

            if debug_mode:

                st.exception(exc)

            else:

                st.error(str(exc))

    else:

        st.info("Sol menüden Past + Future CSV yükle.")



if __name__ == "__main__":

    main()


