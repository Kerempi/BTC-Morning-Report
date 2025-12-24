import streamlit as st

import pandas as pd

import numpy as np

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler

import hashlib

import json

import math



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

        if (not allow_all_nan) and df[c].isna().all():

            bad.append((c, "all_nan"))

    if bad:

        raise ValueError(f"[{context}] Required columns are missing/all-NaN: {bad}")



def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:

    b2 = b.replace(0, np.nan)

    return a / b2



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

def compute_league_profile(df, W_short=10, W_long=50):
    if df is None or df.empty:
        return {
            "league_regime": np.nan,
            "goals_avg_10": np.nan,
            "goals_avg_50": np.nan,
            "league_regime_score": 0.0,
        }
    df = df.copy()
    if "League" in df.columns and df["League"].nunique() > 1:
        df = df[df["League"] == (df["League"].iloc[0])].copy()
    tg = _get_total_goals_series(df)
    if tg is None:
        return {
            "league_regime": np.nan,
            "goals_avg_10": np.nan,
            "goals_avg_50": np.nan,
            "league_regime_score": 0.0,
        }
    df["TG"] = tg
    league = df["League"].iloc[0] if "League" in df.columns else ""
    sub = df.tail(W_long)
    sub2 = df.tail(W_short)
    goals_avg_50 = sub["TG"].mean()
    goals_avg_10 = sub2["TG"].mean()

    trend = goals_avg_10 - goals_avg_50
    strength = abs(trend)
    conf = min(1, len(sub2) / W_short) * min(1, len(sub) / W_long)
    raw = (strength * conf * 5.0)
    raw = raw if trend > 0 else -raw
    score = max(-1.0, min(1.0, raw / 2.0))
    if not np.isfinite(score):
        score = 0.0

    regime = "NORMAL"
    if score > 0.3:
        regime = "HIGH-TEMPO"
    elif score < -0.3:
        regime = "LOW-TEMPO"

    return {
        "league": league,
        "goals_avg_10": goals_avg_10,
        "goals_avg_50": goals_avg_50,
        "league_regime": regime,
        "league_regime_score": score,
    }

def compute_team_trend(df, team, W_short=10, W_long=50):
    if df is None or df.empty:
        return {"goals_avg_10": np.nan, "goals_avg_50": np.nan, "trend_score": 0.0}
    df = df.copy()
    tg = _get_total_goals_series(df)
    if tg is None:
        return {"goals_avg_10": np.nan, "goals_avg_50": np.nan, "trend_score": 0.0}
    df["TG"] = tg
    if not team:
        return {"goals_avg_10": np.nan, "goals_avg_50": np.nan, "trend_score": 0.0}
    has_home = "HomeTeam" in df.columns
    has_away = "AwayTeam" in df.columns
    if not (has_home or has_away):
        return {"goals_avg_10": np.nan, "goals_avg_50": np.nan, "trend_score": 0.0}
    m = None
    if has_home:
        m = (df["HomeTeam"] == team)
    if has_away:
        m2 = (df["AwayTeam"] == team)
        m = (m2 if m is None else (m | m2))
    team_df = df[m].copy() if m is not None else pd.DataFrame()
    if team_df.empty:
        return {"goals_avg_10": np.nan, "goals_avg_50": np.nan, "trend_score": 0.0}
    s50 = team_df.tail(W_long)["TG"].mean()
    s10 = team_df.tail(W_short)["TG"].mean()
    trend = s10 - s50
    strength = abs(trend)
    conf = min(1, len(team_df.tail(W_short)) / W_short) * min(1, len(team_df.tail(W_long)) / W_long)
    raw_score = strength * conf * 5
    raw_score = raw_score if trend > 0 else -raw_score
    score = max(-1, min(1, raw_score / 2))
    if not np.isfinite(score):
        score = 0.0
    return {"team": team, "goals_avg_10": s10, "goals_avg_50": s50, "trend_score": score}

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

            parsed = df[hcol].apply(parse_form_string)

            if len(parsed) > 0:

                pts, wr, dr, lr, ln = zip(*parsed)

                df[f"{base}_Home_PPGN"] = pts

                df[f"{base}_Home_WinRate"] = wr

                df[f"{base}_Home_DrawRate"] = dr

                df[f"{base}_Home_LossRate"] = lr

                df[f"{base}_Home_Len"] = ln

        if acol in df.columns:

            parsed = df[acol].apply(parse_form_string)

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

            df[c] = _to_num(df[c])



    @staticmethod

    def _encode_form_columns(df: pd.DataFrame):

        form_cols = [c for c in df.columns if c.startswith("Form_")]

        for col in form_cols:

            parsed = [_form_metrics(v) for v in df[col]]

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

                d[dst] = d[src]

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

            d[col] = _to_num(d[col])

        _nonempty_or_fail(d, ["HomeOdd", "DrawOdd", "AwayOdd"], "INGEST(ODDS)", allow_all_nan=False)

        if is_past:

            for col in ["FTHG", "FTAG"]:

                if col not in d.columns:

                    raise ValueError(f"[INGEST(PAST)] Missing score column {col}.")

            d["FTHG"] = _to_num(d["FTHG"])

            d["FTAG"] = _to_num(d["FTAG"])

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

                d[imp_col] = 1.0 / d[odd_col].replace(0, np.nan)

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

            null_ratio = float(future[col].isna().mean())

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

            "XG_BIAS_W": float(w_xg),

            "SCORE_LAM_H": float(lamH) if np.isfinite(lamH) else np.nan,

            "SCORE_LAM_A": float(lamA) if np.isfinite(lamA) else np.nan,

            "SCORE_MODE": mode,

            "SCORE_TOP3": top3,

            "SCORE_RES_RH": float(mu_rh),

            "SCORE_RES_RA": float(mu_ra),

            "SCORE_RES_EFFN": float(eff_n),

            "SCORE_RES_K": float(0.35),

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



def pick_best_market(row: pd.Series):

    markets = [

        ("MS1", "Sim_Home%", "HomeOdd"),

        ("X", "Sim_Draw%", "DrawOdd"),

        ("MS2", "Sim_Away%", "AwayOdd"),

        ("Over2.5", "Sim_Over25%", "Over25Odd"),

        ("Under2.5", "Sim_Under25%", "Under25Odd"),

        ("BTTS_Yes", "Sim_BTTS_Yes%", "BTTSYesOdd"),

        ("BTTS_No", "Sim_BTTS_No%", "BTTSNoOdd"),

    ]



    best = None

    # prob (benzerlik orani) ayniysa kalite ile tie-break yapalim

    row_simq = float(row.get("SIM_QUALITY", 0.0)) if pd.notna(row.get("SIM_QUALITY")) else 0.0



    for label, p_col, o_col in markets:

        prob = row.get(p_col)

        if pd.isna(prob):

            continue



        odd = row.get(o_col)

        ev = None

        if pd.notna(odd) and odd > 1.01:

            ev = prob * odd - 1.0



        # ANA KRITER: prob (benzerlik orani)

        # TIE-BREAK: SIM_QUALITY (row bazli)

        if best is None or (prob > best["prob"]) or (prob == best["prob"] and row_simq > best.get("row_simq", 0.0)):

            best = {

                "market": label,

                "prob": prob,

                "odd": odd,

                "ev": ev,

                "row_simq": row_simq,

            }



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



def pick_best_market_synth(row, return_debug: bool = False):

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

        return pick_best_market(row)



    if isinstance(row, pd.Series):

        row = row.to_dict()



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

        return pick_best_market(pd.Series(row)) if isinstance(row, dict) else pick_best_market(row)



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



    if not top3:

        return pick_best_market(pd.Series(row)) if isinstance(row, dict) else pick_best_market(row)



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

            best_it = top1_row



    out = {

        "market": best_it["market"],

        "prob": float(np.clip(best_it["base_prob_pct"] / 100.0, 0.0, 1.0)),

        "odd": best_it.get("odd", np.nan),

        "synth_prob_pct": float(best_it["synth_prob_pct"]),

        "pick_margin": pick_margin,

    }



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

def build_prediction_table(res: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for _, r in res.iterrows():

        try:

            best = pick_best_market_synth(r)

        except Exception:

            best = pick_best_market(r)

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



        rows.append({

            "Date": r.get("Date"),

            "League": r.get("League"),

            "HomeTeam": r.get("HomeTeam"),

            "AwayTeam": r.get("AwayTeam"),

            "PICK": best.get("market"),

            "PICK_PROB": prob,

            "PICK_ODD": best.get("odd"),

            "PICK_MARGIN": best.get("pick_margin"),

            "TRUST_PCT": trust,

            "BestOfRank": r.get("BestOfRank"),

            "SCORE_LAM_H": r.get("SCORE_LAM_H"),

            "SCORE_LAM_A": r.get("SCORE_LAM_A"),

            "SCORE_MODE": r.get("SCORE_MODE"),

            "SCORE_TOP3": r.get("SCORE_TOP3"),

            "SCENARIO_TITLE": r.get("SCENARIO_TITLE"),

            "SCENARIO_GROUP": r.get("SCENARIO_GROUP"),

        })



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

        for idx, (_, frow) in enumerate(future.iterrows()):

            stats, sim_df = knn.analyze_match(frow)

            st.session_state.setdefault("_sim_cache", {})[frow["Match_ID"]] = sim_df

            rows.append({

                "Future_Index": idx,

                "Match_ID": frow["Match_ID"],

                "Date": frow["Date"],

                "League": frow["League"],

                "HomeTeam": frow["HomeTeam"],

                "AwayTeam": frow["AwayTeam"],

                "HomeOdd": frow.get("HomeOdd", np.nan),

                "DrawOdd": frow.get("DrawOdd", np.nan),

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

                "XG_Home": stats.get("XG_Home", np.nan),

                "XG_Away": stats.get("XG_Away", np.nan),

                "XG_Total": stats.get("XG_Total", np.nan),

                "XG_BIAS_W": stats.get("XG_BIAS_W", np.nan),

                "SCORE_LAM_H": stats.get("SCORE_LAM_H", np.nan),

                "SCORE_LAM_A": stats.get("SCORE_LAM_A", np.nan),

                "SCORE_MODE": stats.get("SCORE_MODE", ""),

                "SCORE_TOP3": stats.get("SCORE_TOP3", ""),

                "SCORE_RES_RH": stats.get("SCORE_RES_RH", np.nan),

                "SCORE_RES_RA": stats.get("SCORE_RES_RA", np.nan),

                "SCORE_RES_EFFN": stats.get("SCORE_RES_EFFN", np.nan),

                "SCORE_RES_K": stats.get("SCORE_RES_K", np.nan),

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

            best = pick_best_market(row)

            if best is None:

                continue

            picks.append({

                "Date": row["Date"],

                "League": row["League"],

                "HomeTeam": row["HomeTeam"],

                "AwayTeam": row["AwayTeam"],

                "PICK_MARKET": best["market"],

                "PICK_LABEL": best["market"],

                "PICK_ODD": best["odd"],

                "PICK_PROB": best["prob"],

                "PICK_EV": best["ev"],

                "SIM_QUALITY": row["SIM_QUALITY"],

                "EFFECTIVE_N": row["EFFECTIVE_N"],

                "CONF": row["CONF"],

                "CONF_REASON": row["CONF_REASON"],

            })

        pred_df = pd.DataFrame(picks)

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

        "Date","League","HomeTeam","AwayTeam",

        "PICK","PICK_ODD","PICK_PROB","TRUST_PCT",

        "Score_Exp","SCORE_MODE","SCORE_TOP3",

        "SCENARIO_TITLE",

        "BestOfRank",

    ]

    PRED_FMT = {"PICK_ODD":"{:.2f}", "PICK_PROB":"{:.1%}", "TRUST_PCT":"{:.1f}", "BestOfRank":"{:.0f}"}



    # Full dump kolon seti (varsa göster)

    show_cols = [

        "Match_ID", "Date", "League", "HomeTeam", "AwayTeam",

        "HomeOdd", "DrawOdd", "AwayOdd",

        "Over25Odd", "Under25Odd", "BTTSYesOdd", "BTTSNoOdd",

        "Sim_Home%", "Sim_Draw%", "Sim_Away%",

        "Sim_Over25%", "Sim_Under25%",

        "Sim_BTTS_Yes%", "Sim_BTTS_No%",

        "Avg_Similarity", "SIM_QUALITY", "EFFECTIVE_N", "TRUST_PCT",

        "same_league_count", "global_count", "BestOfScore",

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

    }

    # ? sadece mevcut kolonlarin formatini uygula

    format_map = {k: v for k, v in format_map.items() if k in show_cols}



    tab1, tab2, tab3 = st.tabs(["📊 Tahmin Tablosu", "🔎 Maç Detayı", "🧩 Şema & Debug"])



    if isinstance(res, pd.DataFrame) and not res.empty:

        with tab1:

            st.subheader("Tahmin Tablosu")

            c1, c2 = st.columns(2)

            min_prob = c1.slider("Min PICK_PROB", 0.0, 1.0, 0.45, 0.01)

            min_trust = c2.slider("Min TRUST_PCT", 0, 100, 60, 1)



            _pred = pred_df.copy() if isinstance(pred_df, pd.DataFrame) else pd.DataFrame()

            # kolon garantisi (disaridan bozuk gelirse bile patlamasin)

            for c in PRED_COLS:

                if c not in _pred.columns:

                    _pred[c] = np.nan



            if not _pred.empty:

                _pred = _pred[_pred["PICK_PROB"].fillna(0) >= float(min_prob)]

                _pred = _pred[_pred["TRUST_PCT"] >= min_trust]



            view = _pred.copy()

            if isinstance(view, pd.DataFrame) and not view.empty:

                if "SCORE_LAM_H" in view.columns and "SCORE_LAM_A" in view.columns:

                    h = pd.to_numeric(view["SCORE_LAM_H"], errors="coerce").round(2)

                    a = pd.to_numeric(view["SCORE_LAM_A"], errors="coerce").round(2)

                    view["Score_Exp"] = h.astype(str) + "-" + a.astype(str)

                else:

                    view["Score_Exp"] = ""
                def _badge_from(row):
                    t = _to_float(row.get("TRUST_PCT"), np.nan)
                    br = _to_float(row.get("BestOfRank"), np.nan)
                    pm = _to_float(row.get("PICK_MARGIN"), np.nan)  # PICK_MARGIN'i okuyalim
                    badge = "⚪"

                    if np.isfinite(t) and np.isfinite(br):
                        if (t >= 70.0) and (br >= 75.0):
                            badge = "🟢"
                        elif (t >= 60.0) and (br >= 65.0):
                            badge = "🟡"
                    elif np.isfinite(t):
                        if t >= 70.0:
                            badge = "🟢"
                        elif t >= 60.0:
                            badge = "🟡"

                    # Yakin karar downgrade'i icin esik 2.9'a cikarildi (2.9 puan alti close-call sayilir)
                    if np.isfinite(pm) and pm < 2.9:
                        if badge == "🟢":
                            badge = "🟡"
                        elif badge == "🟡":
                            badge = "⚪"
                    return badge

                if "PICK" in view.columns:

                    view["PICK"] = view.apply(lambda r: f"{r.get('PICK', '')} {_badge_from(r)}".strip(), axis=1)



            if view.empty:

                st.warning("Tahmin bulunamadi.")

            else:

                st.dataframe(view[PRED_COLS].style.format(PRED_FMT), use_container_width=True)



            with st.expander("Detay Görünüm (Full Dump)", expanded=False):

                FULL_COLS = [

                    "Match_ID","Date","League","HomeTeam","AwayTeam",

                    "PICK","PICK_PROB","PICK_ODD",

                    "Sim_Home%","Sim_Draw%","Sim_Away%","Sim_Over25%","Sim_Under25%","Sim_BTTS_Yes%","Sim_BTTS_No%",

                    "SIM_QUALITY","EFFECTIVE_N","TRUST_PCT"

                ]

                FULL_COLS = [c for c in FULL_COLS if c in res.columns]

                st.dataframe(res[FULL_COLS].style.format(format_map), use_container_width=True)



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

                            best_dbg = pick_best_market_synth(stats, return_debug=True)
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


                    cache_key = frow.get("Match_ID") or f"{frow.get('League')}|{frow.get('HomeTeam')}|{frow.get('AwayTeam')}"



                    if isinstance(sim_df, pd.DataFrame) and not sim_df.empty:



                        cached = _regime_cache.get(cache_key)




                        if isinstance(cached, dict):



                            lp = cached.get("league_profile", {})



                            ht = cached.get("home_trend", {})



                            at = cached.get("away_trend", {})



                        else:



                            lp = compute_league_profile(sim_df)



                            ht = compute_team_trend(sim_df, frow.get("HomeTeam"))



                            at = compute_team_trend(sim_df, frow.get("AwayTeam"))



                            _regime_cache[cache_key] = {



                                "league_profile": lp,



                                "home_trend": ht,



                                "away_trend": at,



                            }



                            if len(_regime_cache) > 500:



                                _regime_cache.pop(next(iter(_regime_cache)))




                        league_goals_avg_10 = lp.get("goals_avg_10", np.nan)



                        league_goals_avg_50 = lp.get("goals_avg_50", np.nan)



                        league_regime = lp.get("league_regime", "n/a")



                        league_regime_score = lp.get("league_regime_score", 0.0)



                        home_trend_score = ht.get("trend_score", 0.0)



                        away_trend_score = at.get("trend_score", 0.0)




                        if not isinstance(league_regime, str) or league_regime.strip() == "" or league_regime.lower() == "nan":



                            league_regime = "n/a"



                        if not np.isfinite(league_regime_score):



                            league_regime_score = 0.0



                        if not np.isfinite(home_trend_score):



                            home_trend_score = 0.0



                        if not np.isfinite(away_trend_score):



                            away_trend_score = 0.0




                        sim_df = sim_df.copy()



                        sim_df["LEAGUE_GOALS_AVG_10"] = league_goals_avg_10



                        sim_df["LEAGUE_GOALS_AVG_50"] = league_goals_avg_50



                        sim_df["LEAGUE_REGIME"] = league_regime



                        sim_df["LEAGUE_REGIME_SCORE"] = league_regime_score



                        sim_df["HOME_TREND_SCORE"] = home_trend_score



                        sim_df["AWAY_TREND_SCORE"] = away_trend_score




                    def _fmt2(v):



                        f = _safe_float(v)



                        return f"{f:.2f}" if np.isfinite(f) else "NA"




                    st.session_state["dbg_league_regime"] = {


                        "LEAGUE_REGIME": league_regime,


                        "LEAGUE_REGIME_SCORE": league_regime_score,


                        "LEAGUE_GOALS_AVG_10": league_goals_avg_10,


                        "LEAGUE_GOALS_AVG_50": league_goals_avg_50,


                        "HOME_TREND_SCORE": home_trend_score,


                        "AWAY_TREND_SCORE": away_trend_score,


                    }



                    st.caption(


                        f"Regime={league_regime} (score={_fmt2(league_regime_score)}) | "


                        f"L10={_fmt2(league_goals_avg_10)} L50={_fmt2(league_goals_avg_50)} | "


                        f"H={_fmt2(home_trend_score)} A={_fmt2(away_trend_score)}"


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
                        "FTHG", "FTAG", "FT_Score", "Result",
                        "xG_Home", "xG_Away", "xG_Total",
                        "TotalGoals", "Similarity_Score",
                    ]
                    if show_advanced_cols:
                        extra_cols = [
                            "LEAGUE_REGIME", "LEAGUE_REGIME_SCORE",
                            "LEAGUE_GOALS_AVG_10", "LEAGUE_GOALS_AVG_50",
                            "HOME_TREND_SCORE", "AWAY_TREND_SCORE",
                        ]
                        sim_cols = sim_cols + extra_cols
                    sim_cols = [c for c in sim_cols if c in sim_df.columns]

                    for c in ["FTHG", "FTAG", "TotalGoals"]:

                        if c in sim_df.columns:

                            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce").round(0).astype("Int64")

                    for c in ["xG_Home", "xG_Away"]:

                        if c in sim_df.columns:

                            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce").round(2)

                    if "xG_Home" in sim_df.columns and "xG_Away" in sim_df.columns:

                        sim_df["xG_Total"] = (sim_df["xG_Home"] + sim_df["xG_Away"]).round(2)

                    if "FTHG" in sim_df.columns and "FTAG" in sim_df.columns:

                        sim_df["FT_Score"] = (

                            sim_df["FTHG"].astype("Int64").astype(str)

                            + "-"

                            + sim_df["FTAG"].astype("Int64").astype(str)

                        )

                    display_count = min(len(sim_df), debug['chosen_params']['k_same'] + debug['chosen_params']['k_global'])

                    st.dataframe(

                        sim_df[sim_cols].head(display_count).style.format({"Similarity_Score": "{:.3f}"}),

                        use_container_width=True,

                    )

        with tab3:

            st.subheader("Sema & Debug")
            dbg_regime = st.session_state.get("dbg_league_regime")
            def _fmt2_dbg(v):
                f = _safe_float(v)
                return f"{f:.2f}" if np.isfinite(f) else "NA"
            if isinstance(dbg_regime, dict):
                league_regime = dbg_regime.get("LEAGUE_REGIME", "n/a")
                if not isinstance(league_regime, str) or league_regime.strip() == "" or league_regime.lower() == "nan":
                    league_regime = "n/a"
                st.write(
                    f"Regime={league_regime} (score={_fmt2_dbg(dbg_regime.get('LEAGUE_REGIME_SCORE'))}) | "
                    f"L10={_fmt2_dbg(dbg_regime.get('LEAGUE_GOALS_AVG_10'))} L50={_fmt2_dbg(dbg_regime.get('LEAGUE_GOALS_AVG_50'))} | "
                    f"H={_fmt2_dbg(dbg_regime.get('HOME_TREND_SCORE'))} A={_fmt2_dbg(dbg_regime.get('AWAY_TREND_SCORE'))}"
                )
                st.caption(
                    f"scores={{regime:{league_regime}, score:{_fmt2_dbg(dbg_regime.get('LEAGUE_REGIME_SCORE'))}, "
                    f"L10:{_fmt2_dbg(dbg_regime.get('LEAGUE_GOALS_AVG_10'))}, L50:{_fmt2_dbg(dbg_regime.get('LEAGUE_GOALS_AVG_50'))}, "
                    f"H:{_fmt2_dbg(dbg_regime.get('HOME_TREND_SCORE'))}, A:{_fmt2_dbg(dbg_regime.get('AWAY_TREND_SCORE'))}}}"
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

    else:

        st.warning("Analiz sonucunda gösterilecek maç yok.")



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

