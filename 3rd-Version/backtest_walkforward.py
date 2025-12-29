import argparse
import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import appfinal1 as app


def _get_base_dir():
    try:
        base = os.path.dirname(__file__)
    except Exception:
        base = ""
    return base if base else os.getcwd()


def _ensure_session_state():
    try:
        _ = app.st.session_state
    except Exception:
        class _DummyState(dict):
            pass
        app.st.session_state = _DummyState()


def _parse_date(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def _score_cols(df):
    if {"FT_Score_Home", "FT_Score_Away"}.issubset(df.columns):
        return "FT_Score_Home", "FT_Score_Away"
    if {"FTHG", "FTAG"}.issubset(df.columns):
        return "FTHG", "FTAG"
    return None, None


def _sanitize_for_backtest(df):
    if df is None or df.empty:
        return df
    # Ensure unique columns to avoid df[col] returning a DataFrame
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    def _col(series_or_df):
        if isinstance(series_or_df, pd.DataFrame):
            return series_or_df.iloc[:, 0]
        return series_or_df
    mapping = {
        "HomeOdd": "Odds_Open_Home",
        "DrawOdd": "Odds_Open_Draw",
        "AwayOdd": "Odds_Open_Away",
        "Over25Odd": "Odds_Open_Over25",
        "Under25Odd": "Odds_Open_Under25",
        "BTTSYesOdd": "Odds_Open_BTTS_Yes",
        "BTTSNoOdd": "Odds_Open_BTTS_No",
        "xG_Home": "XG_Home",
        "xG_Away": "XG_Away",
        "xG_Total": "XG_Total",
        "FT_Score_Home": "FTHG",
        "FT_Score_Away": "FTAG",
    }
    for target, source in mapping.items():
        if target not in df.columns or df[target].isna().all():
            if source in df.columns:
                df[target] = _col(df[source])
    return df


def _compute_results(pred, future_std):
    pred = pred.copy()
    if "Match_ID" not in pred.columns or "Match_ID" not in future_std.columns:
        return pred
    res_map = future_std.set_index("Match_ID")
    # bring xG fields for policy rules
    for col in ("xG_Home", "xG_Away", "xG_Total", "XG_Home", "XG_Away", "XG_Total"):
        if col in res_map.columns and col not in pred.columns:
            pred[col] = pred["Match_ID"].map(res_map[col])
    h_col, a_col = _score_cols(future_std)
    if h_col and a_col:
        pred["RES_FTHG"] = pred["Match_ID"].map(res_map[h_col])
        pred["RES_FTAG"] = pred["Match_ID"].map(res_map[a_col])
    hg = pd.to_numeric(pred.get("RES_FTHG"), errors="coerce")
    ag = pd.to_numeric(pred.get("RES_FTAG"), errors="coerce")
    valid = hg.notna() & ag.notna()
    pred["RES_RESULT_1X2"] = pd.Series([pd.NA] * len(pred), dtype="object")
    pred.loc[valid, "RES_RESULT_1X2"] = np.where(
        hg[valid] > ag[valid],
        "HOME",
        np.where(hg[valid] < ag[valid], "AWAY", "DRAW"),
    )
    tg = hg + ag
    pred["RES_RESULT_OU"] = pd.Series([pd.NA] * len(pred), dtype="object")
    pred.loc[valid, "RES_RESULT_OU"] = np.where(
        tg[valid] >= 3, "Over2.5", "Under2.5"
    )
    pred["RES_RESULT_BTTS"] = pd.Series([pd.NA] * len(pred), dtype="object")
    pred.loc[valid, "RES_RESULT_BTTS"] = np.where(
        (hg[valid] > 0) & (ag[valid] > 0), "BTTS Yes", "BTTS No"
    )
    return pred


def _compute_hit(pred):
    out = pred.copy()
    out["HIT"] = pd.Series([pd.NA] * len(out), dtype="boolean")
    if out.empty:
        return out
    def _norm_1x2(v):
        if pd.isna(v):
            return ""
        s = str(v)
        return {"HOME": "MS1", "AWAY": "MS2", "DRAW": "X"}.get(s, s)
    def _norm_btts(v):
        if pd.isna(v):
            return ""
        s = str(v)
        return {"BTTS Yes": "BTTS_Yes", "BTTS No": "BTTS_No"}.get(s, s)
    out.loc[out["PICK_GROUP"] == "1X2", "HIT"] = (
        out["PICK_FINAL"] == out["RES_RESULT_1X2"].map(_norm_1x2)
    )
    out.loc[out["PICK_GROUP"] == "OU", "HIT"] = (
        out["PICK_FINAL"] == out["RES_RESULT_OU"]
    )
    out.loc[out["PICK_GROUP"] == "BTTS", "HIT"] = (
        out["PICK_FINAL"] == out["RES_RESULT_BTTS"].map(_norm_btts)
    )
    return out


def _build_params():
    base_dir = _get_base_dir()
    params_path = os.path.join(base_dir, "params_latest.json")
    if os.path.exists(params_path):
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            if isinstance(params, dict):
                return params
        except Exception:
            pass
    return {}


def _safe_csv_path(path):
    path = os.path.abspath(os.path.normpath(path))
    if os.name == "nt":
        if not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + path
    return path


def _write_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return True
    except OSError:
        try:
            df.to_csv(_safe_csv_path(path), index=False)
            return True
        except OSError:
            return False


def _safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return default
    return {
        "k_same": 30,
        "k_global": 20,
        "same_league_mode": True,
        "min_same_found": 12,
        "null_threshold": 0.8,
        "weight_config": {"xg": 2.0, "elo": 1.2, "league_pos": 1.1, "default": 1.0},
        "manual_features": None,
        "conf_quality_floor": 0.25,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="cleaned_past.csv path")
    parser.add_argument("--outdir", required=True, help="output folder")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--step-days", type=int, default=1, help="day step for low-volume days")
    parser.add_argument("--busy-threshold", type=int, default=30, help="process every day when match count >= this")
    parser.add_argument("--min-past-rows", type=int, default=2000)
    args = parser.parse_args()

    _ensure_session_state()
    os.makedirs(args.outdir, exist_ok=True)
    tmp_dir = os.path.join(args.outdir, "_tmp_backtest")
    os.makedirs(tmp_dir, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    if "Date" not in df.columns:
        raise SystemExit("Date column missing in input.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df = df.sort_values("Date")

    start = _parse_date(args.start_date) if args.start_date else df["Date"].min()
    end = _parse_date(args.end_date) if args.end_date else df["Date"].max()
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()

    params = _build_params()
    all_preds = []
    daily_rows = []

    cur = start
    last_processed = None
    while cur <= end:
        day_df = df[df["Date"].dt.normalize() == cur].copy()
        if day_df.empty:
            cur += timedelta(days=1)
            continue
        is_busy = len(day_df) >= args.busy_threshold
        should_process = is_busy or (last_processed is None) or ((cur - last_processed).days >= args.step_days)
        if not should_process:
            cur += timedelta(days=1)
            continue
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {cur.strftime('%Y-%m-%d')} ({len(day_df)} matches)")
        past_df = df[df["Date"].dt.normalize() < cur].copy()
        if len(past_df) < args.min_past_rows:
            cur += timedelta(days=1)
            continue

        past_df = _sanitize_for_backtest(past_df)
        day_df = _sanitize_for_backtest(day_df)
        # If opening odds are missing for the day, skip to avoid INGEST(ODDS) failure
        odds_cols = ["HomeOdd", "DrawOdd", "AwayOdd"]
        if not all(c in day_df.columns for c in odds_cols) or day_df[odds_cols].isna().all().all():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Skip {cur.strftime('%Y-%m-%d')}: missing opening odds")
            last_processed = cur
            cur += timedelta(days=1)
            continue

        past_path = os.path.join(tmp_dir, f"past_{cur.strftime('%Y%m%d')}.csv")
        fut_path = os.path.join(tmp_dir, f"future_{cur.strftime('%Y%m%d')}.csv")
        past_df.to_csv(past_path, index=False)
        day_df.to_csv(fut_path, index=False)

        try:
            pred_df, res, _knn, _past_std, future_std, _debug = app.run_pipeline(
                past_path, fut_path, params
            )
        except Exception as exc:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error on {cur.strftime('%Y-%m-%d')}: {exc}")
            # write whatever we have so far
            if all_preds:
                _write_csv(
                    pd.concat(all_preds, axis=0, ignore_index=True),
                    os.path.join(args.outdir, "backtest_predictions.csv"),
                )
            if daily_rows:
                _write_csv(
                    pd.DataFrame(daily_rows),
                    os.path.join(args.outdir, "backtest_daily_summary.csv"),
                )
            cur += timedelta(days=args.step_days)
            continue

        if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
            pred_df = _compute_results(pred_df, future_std)
            # Ensure draw-related columns exist for X selection analysis
            if "DrawOdd" not in pred_df.columns:
                def _col(x):
                    return x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
                fut_idx = future_std.set_index("Match_ID")
                if "Odds_Open_Draw" in fut_idx.columns:
                    pred_df["DrawOdd"] = pred_df["Match_ID"].map(_col(fut_idx[["Odds_Open_Draw"]]))
                elif "DrawOdd" in fut_idx.columns:
                    pred_df["DrawOdd"] = pred_df["Match_ID"].map(_col(fut_idx[["DrawOdd"]]))
                else:
                    pred_df["DrawOdd"] = np.nan
            if "SIM_IMP_DRAW" in pred_df.columns and "Sim_Draw%" not in pred_df.columns:
                pred_df["Sim_Draw%"] = pred_df["SIM_IMP_DRAW"]
            if "Sim_Draw%" in pred_df.columns and pred_df["Sim_Draw%"].isna().all() and "SIM_IMP_DRAW" in pred_df.columns:
                pred_df["Sim_Draw%"] = pred_df["SIM_IMP_DRAW"]
            # recompute policy flags/rules with updated xG columns
            try:
                policy_ctx = app._load_policy_context(os.path.join(_get_base_dir(), "pred_results_log.csv"))
            except Exception:
                policy_ctx = {"under_xg_p50": np.nan, "under_leagues": set()}
            flags = []
            rules = []
            for _, row in pred_df.iterrows():
                try:
                    f, r = app._policy_flag(row, policy_ctx)
                except Exception:
                    f, r = 0, ""
                flags.append(int(f))
                rules.append(r)
            pred_df["POLICY_FLAG"] = flags
            pred_df["POLICY_RULE"] = rules
            pred_df = _compute_hit(pred_df)
            all_preds.append(pred_df)
            played = pred_df.dropna(
                subset=["RES_FTHG", "RES_FTAG", "RES_RESULT_1X2", "RES_RESULT_OU", "RES_RESULT_BTTS"]
            )
            if not played.empty:
                daily_rows.append(
                    {
                        "Date": cur.strftime("%Y-%m-%d"),
                        "Count": len(played),
                        "HitRate": _safe_float(pd.to_numeric(played["HIT"], errors="coerce").mean()),
                        "PolicyCount": int((played.get("POLICY_FLAG", 0).fillna(0) == 1).sum()),
                        "PolicyHitRate": _safe_float(
                            pd.to_numeric(
                                played[played.get("POLICY_FLAG", 0).fillna(0) == 1]["HIT"],
                                errors="coerce",
                            ).mean()
                        ) if (played.get("POLICY_FLAG", 0).fillna(0) == 1).any() else np.nan,
                    }
                )

            # incremental save after each day
            _write_csv(
                pd.concat(all_preds, axis=0, ignore_index=True),
                os.path.join(args.outdir, "backtest_predictions.csv"),
            )
            _write_csv(
                pd.DataFrame(daily_rows),
                os.path.join(args.outdir, "backtest_daily_summary.csv"),
            )

        cur += timedelta(days=args.step_days)

    if all_preds:
        out_pred = pd.concat(all_preds, axis=0, ignore_index=True)
        _write_csv(
            out_pred,
            os.path.join(args.outdir, "backtest_predictions.csv"),
        )
    if daily_rows:
        out_daily = pd.DataFrame(daily_rows)
        _write_csv(
            out_daily,
            os.path.join(args.outdir, "backtest_daily_summary.csv"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
