import numpy as np
import pandas as pd

def ensure_bestofrank(df):
    """Compute BestOfRank robustly even when Prob/Score are missing or degenerate."""
    if df is None or getattr(df, "empty", True):
        return df

    _df = df.copy()

    # Prob fallback
    try:
        prob = pd.to_numeric(_df.get("Prob", np.nan), errors="coerce")
    except Exception:
        prob = pd.Series([np.nan]*len(_df), index=_df.index)

    if prob.notna().sum() == 0:
        if "MARKET_CANON" in _df.columns:
            canon = _df["MARKET_CANON"].astype(str)
            p = pd.Series(np.nan, index=_df.index)

            if "P_Home_Final" in _df.columns:
                p = p.where(~canon.eq("MS1"), pd.to_numeric(_df["P_Home_Final"], errors="coerce"))
            if "P_Draw_Final" in _df.columns:
                p = p.where(~canon.eq("MSX"), pd.to_numeric(_df["P_Draw_Final"], errors="coerce"))
            if "P_Away_Final" in _df.columns:
                p = p.where(~canon.eq("MS2"), pd.to_numeric(_df["P_Away_Final"], errors="coerce"))

            if "P_Over_Final" in _df.columns:
                po = pd.to_numeric(_df["P_Over_Final"], errors="coerce")
                p = p.where(~canon.eq("OU25_OVER"), po)
                p = p.where(~canon.eq("OU25_UNDER"), 1.0 - po)

            if "P_BTTS_Final" in _df.columns:
                pb = pd.to_numeric(_df["P_BTTS_Final"], errors="coerce")
                p = p.where(~canon.eq("BTTS_YES"), pb)
                p = p.where(~canon.eq("BTTS_NO"), 1.0 - pb)

            _df["Prob"] = p.fillna(0.0)
        else:
            _df["Prob"] = 0.0

    # Score fallback
    try:
        score = pd.to_numeric(_df.get("Score", np.nan), errors="coerce")
    except Exception:
        score = pd.Series([np.nan]*len(_df), index=_df.index)

    if score.notna().sum() == 0:
        if "Smart_EV" in _df.columns:
            _df["Score"] = pd.to_numeric(_df["Smart_EV"], errors="coerce").fillna(0.0)
        elif "EV_Calc" in _df.columns:
            _df["Score"] = pd.to_numeric(_df["EV_Calc"], errors="coerce").fillna(0.0)
        else:
            _df["Score"] = pd.to_numeric(_df.get("Prob", 0.0), errors="coerce").fillna(0.0)

    # BestOfRank
    s = pd.to_numeric(_df["Score"], errors="coerce").fillna(0.0)
    p = pd.to_numeric(_df["Prob"], errors="coerce").fillna(0.0)

    z = (s - s.mean())/(s.std(ddof=0)+1e-9) + (p - p.mean())/(p.std(ddof=0)+1e-9)
    _df["BestOfRank"] = z.rank(method="average", ascending=False).astype(int)
    return _df
