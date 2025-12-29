import argparse
import os
import sys
import pandas as pd


def _norm(s):
    if s is None:
        return ""
    return str(s).strip()

def _norm_id(s):
    s = _norm(s).lower()
    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _build_match_id(df):
    need = ["Date", "League", "HomeTeam", "AwayTeam"]
    if not all(c in df.columns for c in need):
        return None
    date_col = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return (
        date_col.fillna(df["Date"].astype(str).map(_norm))
        + "|"
        + df["League"].astype(str).map(_norm_id)
        + "|"
        + df["HomeTeam"].astype(str).map(_norm_id)
        + "|"
        + df["AwayTeam"].astype(str).map(_norm_id)
    )


def main():
    parser = argparse.ArgumentParser(description="Update pred_results_log.csv with new results.")
    parser.add_argument("--log", required=True, help="Path to pred_results_log.csv")
    parser.add_argument("--updates", required=True, help="Path to updates CSV (Match_ID or Date/League/HomeTeam/AwayTeam).")
    parser.add_argument("--date", help="Optional filter date (YYYY-MM-DD).")
    parser.add_argument("--dry-run", action="store_true", help="Do not write, only report.")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"Log not found: {args.log}")
    if not os.path.exists(args.updates):
        raise SystemExit(f"Updates not found: {args.updates}")

    log_df = pd.read_csv(args.log)
    upd_df = pd.read_csv(args.updates)

    if log_df.empty or upd_df.empty:
        raise SystemExit("Empty log or updates file.")

    if "Match_ID" not in log_df.columns:
        log_df["Match_ID"] = _build_match_id(log_df)

    if "HomeTeam" not in upd_df.columns and "Home_Team" in upd_df.columns:
        upd_df["HomeTeam"] = upd_df["Home_Team"]
    if "AwayTeam" not in upd_df.columns and "Away_Team" in upd_df.columns:
        upd_df["AwayTeam"] = upd_df["Away_Team"]

    if "RES_FTHG" not in upd_df.columns:
        if "FT_Score_Home" in upd_df.columns:
            upd_df["RES_FTHG"] = upd_df["FT_Score_Home"]
        elif "FTHG" in upd_df.columns:
            upd_df["RES_FTHG"] = upd_df["FTHG"]
    if "RES_FTAG" not in upd_df.columns:
        if "FT_Score_Away" in upd_df.columns:
            upd_df["RES_FTAG"] = upd_df["FT_Score_Away"]
        elif "FTAG" in upd_df.columns:
            upd_df["RES_FTAG"] = upd_df["FTAG"]

    if "Match_ID" in upd_df.columns:
        upd_df["Match_ID"] = upd_df["Match_ID"].astype(str).map(_norm)
    else:
        mid = _build_match_id(upd_df)
        if mid is None:
            raise SystemExit("Updates CSV must contain Match_ID or Date/League/HomeTeam/AwayTeam.")
        upd_df["Match_ID"] = mid

    if args.date:
        if "Date" not in upd_df.columns:
            raise SystemExit("--date provided but updates CSV has no Date column.")
        upd_df["_date"] = pd.to_datetime(upd_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        before = len(upd_df)
        upd_df = upd_df[upd_df["_date"] == args.date].copy()
        after = len(upd_df)
        if after == 0:
            raise SystemExit(f"No rows match --date {args.date} (from {before} rows).")

    for col in ["RES_FTHG", "RES_FTAG"]:
        if col not in upd_df.columns:
            raise SystemExit(f"Updates missing column: {col}")

    upd_map = upd_df.set_index("Match_ID")[["RES_FTHG", "RES_FTAG"]]

    updated = 0
    missing = 0
    for i, mid in log_df["Match_ID"].items():
        if mid in upd_map.index:
            log_df.at[i, "RES_FTHG"] = upd_map.at[mid, "RES_FTHG"]
            log_df.at[i, "RES_FTAG"] = upd_map.at[mid, "RES_FTAG"]
            updated += 1
        else:
            missing += 1

    print(f"Updated rows: {updated}")
    print(f"Unmatched rows in updates: {len(upd_map) - updated}")
    print(f"Log rows without update: {missing}")

    if args.dry_run:
        return

    backup = args.log.replace(".csv", "_backup_before_results_update.csv")
    log_df.to_csv(backup, index=False)
    log_df.to_csv(args.log, index=False)
    print(f"Backup written: {backup}")
    print(f"Log updated: {args.log}")


if __name__ == "__main__":
    main()
