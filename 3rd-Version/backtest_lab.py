import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


def _norm_1x2(v):
    s = str(v or "")
    return {"HOME": "MS1", "AWAY": "MS2", "DRAW": "X"}.get(s, s)


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


def _norm_btts(v):
    s = str(v or "")
    return {"BTTS Yes": "BTTS_Yes", "BTTS No": "BTTS_No"}.get(s, s)


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


@st.cache_data(show_spinner=False)
def load_backtest(path):
    df = pd.read_csv(path, low_memory=False)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def compute_hit(df):
    out = df.copy()
    req = ["RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS"]
    if not all(c in out.columns for c in req):
        out["HIT"] = np.nan
        return out
    played = out.dropna(subset=req).copy()
    played["HIT"] = np.nan
    pick_norm = played["PICK_FINAL"].map(_norm_pick)
    played.loc[played["PICK_GROUP"] == "1X2", "HIT"] = (
        pick_norm == played["RES_RESULT_1X2"].map(_norm_1x2)
    )
    played.loc[played["PICK_GROUP"] == "OU", "HIT"] = (
        pick_norm == played["RES_RESULT_OU"]
    )
    played.loc[played["PICK_GROUP"] == "BTTS", "HIT"] = (
        pick_norm.map(_norm_btts) == played["RES_RESULT_BTTS"].map(_norm_btts)
    )
    out["HIT"] = np.nan
    if "Match_ID" in out.columns and "Match_ID" in played.columns:
        hit_map = played.dropna(subset=["Match_ID"]).set_index("Match_ID")["HIT"]
        out["HIT"] = out["Match_ID"].map(hit_map)
    elif len(played) == len(out):
        out.loc[played.index, "HIT"] = played["HIT"].values
    return out


def filter_df(df, date_range, pick_groups, policy_filter, leagues):
    out = df.copy()
    if "Date" in out.columns and date_range:
        start, end = date_range
        out = out[(out["Date"] >= start) & (out["Date"] <= end)]
    if pick_groups:
        out = out[out["PICK_GROUP"].isin(pick_groups)]
    if policy_filter == "Policy only":
        out = out[out.get("POLICY_FLAG", 0).fillna(0).astype(int) == 1]
    elif policy_filter == "Non-policy":
        out = out[out.get("POLICY_FLAG", 0).fillna(0).astype(int) == 0]
    if leagues:
        out = out[out.get("League", "").isin(leagues)]
    return out


def main():
    st.set_page_config(page_title="Backtest Lab", layout="wide")
    st.title("Backtest Lab")

    default_path = os.path.join(os.path.dirname(__file__), "pred_results_log.csv")
    path = st.text_input("Log CSV path", value=default_path)
    if not path or not os.path.exists(path):
        st.warning("Gecerli bir log dosyasi sec.")
        return

    df = load_backtest(path)
    if df.empty:
        st.warning("Dosya bos.")
        return

    df = compute_hit(df)

    # Sidebar filters
    st.sidebar.header("Filters")
    if "Date" in df.columns and df["Date"].notna().any():
        min_d = df["Date"].min().date()
        max_d = df["Date"].max().date()
        date_range = st.sidebar.date_input("Date range", value=(min_d, max_d))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start = datetime.combine(date_range[0], datetime.min.time())
            end = datetime.combine(date_range[1], datetime.max.time())
            date_range = (start, end)
        else:
            date_range = None
    else:
        date_range = None

    groups = sorted([g for g in df.get("PICK_GROUP", pd.Series()).dropna().unique().tolist()])
    pick_groups = st.sidebar.multiselect("Pick groups", groups, default=groups)

    policy_filter = st.sidebar.selectbox("Policy", ["All", "Policy only", "Non-policy"], index=0)

    leagues = sorted([l for l in df.get("League", pd.Series()).dropna().unique().tolist()])
    leagues_sel = st.sidebar.multiselect("League", leagues, default=[])

    filtered = filter_df(df, date_range, pick_groups, policy_filter, leagues_sel)

    tabs = st.tabs(["Diagnosis", "Simulation"])

    with tabs[0]:
        st.subheader("Diagnosis")
        played = filtered[filtered["HIT"].notna()].copy() if "HIT" in filtered.columns else filtered.copy()
        total = len(played)
        hit_rate = float(played["HIT"].mean()) if total and "HIT" in played.columns else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric("Played", f"{total}")
        c2.metric("Hit Rate", f"{hit_rate:.1%}" if np.isfinite(hit_rate) else "n/a")
        if "POLICY_FLAG" in played.columns and "HIT" in played.columns:
            pf = played[played["POLICY_FLAG"].fillna(0).astype(int) == 1]
            pf_hit = float(pf["HIT"].mean()) if not pf.empty else np.nan
            c3.metric("Policy Hit", f"{pf_hit:.1%}" if np.isfinite(pf_hit) else "n/a")
        else:
            c3.metric("Policy Hit", "n/a")

        if "PICK_GROUP" in played.columns and "HIT" in played.columns:
            grp = played.groupby("PICK_GROUP")["HIT"].agg(["count", "mean"]).reset_index()
            grp["rate"] = grp["mean"]
            st.dataframe(grp[["PICK_GROUP", "count", "rate"]].style.format({"rate": "{:.1%}"}), use_container_width=True)

        # Hit vs Miss feature diff
        num_cols = [c for c in played.columns if pd.api.types.is_numeric_dtype(played[c])]
        num_cols = [c for c in num_cols if c not in ("HIT",)]
        hit_df = played[played["HIT"] == True] if "HIT" in played.columns else pd.DataFrame()
        miss_df = played[played["HIT"] == False] if "HIT" in played.columns else pd.DataFrame()
        if not hit_df.empty and not miss_df.empty:
            diffs = []
            for c in num_cols:
                h = _to_num(hit_df[c]).mean()
                m = _to_num(miss_df[c]).mean()
                if np.isfinite(h) and np.isfinite(m):
                    diffs.append((c, h, m, h - m))
            diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            st.markdown("**Hit vs Miss farklari (en yuksek 15)**")
            st.dataframe(
                pd.DataFrame(diffs[:15], columns=["feature", "hit_mean", "miss_mean", "diff"]),
                use_container_width=True,
            )

        # League summary
        if "League" in played.columns and "HIT" in played.columns and not played.empty:
            min_n = st.slider("League min n", 10, 200, 30, 5)
            lg = played.groupby("League")["HIT"].agg(["count", "mean"]).reset_index()
            lg = lg[lg["count"] >= min_n].sort_values("mean", ascending=False)
            st.markdown("**Top Leagues**")
            st.dataframe(lg.head(10).style.format({"mean": "{:.1%}"}), use_container_width=True)
            st.markdown("**Bottom Leagues**")
            st.dataframe(lg.tail(10).style.format({"mean": "{:.1%}"}), use_container_width=True)

    with tabs[1]:
        st.subheader("Scenario Simulation")
        sim_df = filtered[filtered["HIT"].notna()].copy() if "HIT" in filtered.columns else filtered.copy()
        if "HIT" not in sim_df.columns:
            st.info("HIT column missing. Check results columns in backtest file.")
            return
        if sim_df.empty:
            st.info("No played rows in current filter.")
            return

        numeric_cols = [c for c in sim_df.columns if pd.api.types.is_numeric_dtype(sim_df[c])]
        numeric_cols = [c for c in numeric_cols if c not in ("HIT",)]

        rule_count = st.slider("Rule count", 1, 3, 2, 1)
        rules = []
        for i in range(rule_count):
            st.markdown(f"**Rule {i+1}**")
            col = st.selectbox(f"Column {i+1}", numeric_cols, key=f"col_{i}")
            op = st.selectbox(f"Operator {i+1}", [">=", "<="], key=f"op_{i}")
            v = _to_num(sim_df[col])
            min_v = float(v.quantile(0.05)) if v.notna().any() else 0.0
            max_v = float(v.quantile(0.95)) if v.notna().any() else 1.0
            thr = st.slider(f"Threshold {i+1}", min_v, max_v, float(v.median()) if v.notna().any() else 0.0, key=f"thr_{i}")
            rules.append((col, op, thr))

        mask = pd.Series(True, index=sim_df.index)
        for col, op, thr in rules:
            v = _to_num(sim_df[col])
            if op == ">=":
                mask &= v >= thr
            else:
                mask &= v <= thr
        sub = sim_df[mask & mask.notna()].copy()

        base_hit = float(sim_df["HIT"].mean()) if not sim_df.empty else np.nan
        base_n = len(sim_df)
        base_cov = 1.0
        sub_hit = float(sub["HIT"].mean()) if not sub.empty else np.nan
        sub_n = len(sub)
        sub_cov = sub_n / base_n if base_n else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Baseline hit", f"{base_hit:.1%}")
        c2.metric("Scenario hit", f"{sub_hit:.1%}" if np.isfinite(sub_hit) else "n/a")
        c3.metric("Coverage", f"{sub_cov:.1%}")
        c4.metric("Delta", f"{(sub_hit - base_hit):.1%}" if np.isfinite(sub_hit) else "n/a")
        if "PICK_ODD" in sub.columns:
            odd_avg = _to_num(sub["PICK_ODD"]).mean()
            c5.metric("Scenario odd avg", f"{odd_avg:.2f}" if np.isfinite(odd_avg) else "n/a")
        else:
            c5.metric("Scenario odd avg", "n/a")

        st.dataframe(sub.head(50), use_container_width=True)
        csv_bytes = sub.to_csv(index=False).encode("utf-8")
        st.download_button("Download scenario CSV", csv_bytes, "scenario_subset.csv", "text/csv")

        st.divider()
        with st.expander("Auto Scenario Finder (per pick type)", expanded=False):
            pick_vals = sorted([p for p in sim_df.get("PICK_FINAL", pd.Series()).dropna().unique().tolist()])
            if not pick_vals:
                st.info("PICK_FINAL yok, otomatik senaryo bulunamadi.")
                return

            pick_sel = st.selectbox("Pick type", pick_vals)

            base_pick = sim_df[sim_df.get("PICK_FINAL") == pick_sel].copy()
            if base_pick.empty:
                st.info("Seçilen pick tipi için veri yok.")
                return

            base_hit = float(base_pick["HIT"].mean()) if base_pick["HIT"].notna().any() else np.nan
            base_n = len(base_pick)

            cov_min, cov_max = st.slider("Coverage range (within pick type)", 0.01, 0.95, (0.20, 0.40), 0.01)
            hit_min = st.slider("Min hit rate", 0.00, 0.90, 0.60, 0.01)
            odd_min = st.slider("Min odd avg", 1.00, 3.00, 1.50, 0.01)
            min_n = st.slider("Min n", 5, 500, 50, 5)
            max_cols = st.slider("Max columns to scan", 3, 12, 8, 1)
            rule_k = st.slider("Rule combo size", 1, 5, 2, 1)
            max_combos = st.slider("Max combos to evaluate", 100, 3000, 800, 100)

            auto_cols = st.checkbox("Auto pick columns", value=True)
            default_cols = [c for c in ["PICK_SCORE_TOTAL", "PICK_PROB_ADJ", "P_OVER25_ADJ", "P_BTTSY_ADJ", "xG_Total", "xG_Home", "xG_Away", "SCORE_LAM_H", "SCORE_LAM_A", "BestOfRank", "PICK_MARGIN", "MARKET_WEIGHT"] if c in numeric_cols]
            cand_cols = default_cols[:max_cols] if auto_cols else st.multiselect(
                "Columns to scan",
                numeric_cols,
                default=default_cols[:max_cols],
            )
            qs = st.multiselect("Quantiles", [0.5, 0.6, 0.7, 0.8, 0.9], default=[0.6, 0.7, 0.8])

            rows = []
            # precompute rule candidates
            rule_candidates = []
            for col in cand_cols:
                v = _to_num(base_pick[col])
                if v.notna().sum() < min_n:
                    continue
                for q in qs:
                    thr = float(v.quantile(q))
                    for op in (">=", "<="):
                        rule_candidates.append((col, op, q, thr))

            if not rule_candidates:
                st.info("Uygun kolon/quantile bulunamadi.")
                return

            # evaluate combinations (1..rule_k)
            from itertools import combinations

            def _apply_rules(sub_df, rules):
                mask = pd.Series(True, index=sub_df.index)
                for col, op, _q, thr in rules:
                    v = _to_num(sub_df[col])
                    if op == ">=":
                        mask &= v >= thr
                    else:
                        mask &= v <= thr
                return sub_df[mask & mask.notna()]

            combos_checked = 0
            for k in range(1, rule_k + 1):
                for ruleset in combinations(rule_candidates, k):
                    combos_checked += 1
                    if combos_checked > max_combos:
                        break
                    sub2 = _apply_rules(base_pick, ruleset)
                    n = len(sub2)
                    if n < min_n:
                        continue
                    cov = n / base_n if base_n else 0.0
                    if not (cov_min <= cov <= cov_max):
                        continue
                    hit = float(sub2["HIT"].mean()) if sub2["HIT"].notna().any() else np.nan
                    if not np.isfinite(hit) or hit < hit_min:
                        continue
                    odd_avg = float(_to_num(sub2.get("PICK_ODD")).mean()) if "PICK_ODD" in sub2.columns else np.nan
                    if np.isfinite(odd_min) and np.isfinite(odd_avg) and odd_avg < odd_min:
                        continue
                    rule_txt = " & ".join([f"{c} {o} q{int(q*100)}" for c, o, q, _ in ruleset])
                    rows.append(
                        {
                            "rules": rule_txt,
                            "n": n,
                            "coverage": cov,
                            "hit": hit,
                            "delta": hit - base_hit if np.isfinite(base_hit) else np.nan,
                            "odd_avg": odd_avg,
                        }
                    )
                if combos_checked > max_combos:
                    break

            if not rows:
                st.info("Bu kriterlerle senaryo bulunamadi.")
            else:
                out = pd.DataFrame(rows).sort_values(["hit", "coverage"], ascending=[False, False])
                st.dataframe(
                    out.head(30).style.format({"coverage": "{:.1%}", "hit": "{:.1%}", "delta": "{:+.1%}", "odd_avg": "{:.2f}"}),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
