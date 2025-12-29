import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# =============================================================================
#  LOCAL SETTINGS
# =============================================================================
INPUT_FILE = r"C:\Users\Kerem\OneDrive - BAN\Desktop\Signalles\2nd-App\3rd-Version\Cleaned-Data\Cleaned-281225.csv"
OUTPUT_DIR = r"C:\Users\Kerem\OneDrive - BAN\Desktop\Signalles\2nd-App\3rd-Version\Cleaned-Data"
# =============================================================================


def excel_col_to_index(col_str):
    col_str = col_str.upper().strip()
    num = 0
    for c in col_str:
        if "A" <= c <= "Z":
            num = num * 26 + (ord(c) - ord("A") + 1)
    return num - 1


# BASIC INFO
idx_date = excel_col_to_index("B")
idx_time = excel_col_to_index("C")
idx_season = excel_col_to_index("D")
idx_home = excel_col_to_index("E")
idx_away = excel_col_to_index("F")
idx_league = excel_col_to_index("G")

# EXTRA BASIC INFO
idx_home_pos = excel_col_to_index("DG")
idx_away_pos = excel_col_to_index("DH")
idx_home_manager_games = excel_col_to_index("DP")
idx_away_manager_games = excel_col_to_index("DQ")

# xG (PRE-MATCH)
idx_xg_home_pre = excel_col_to_index("I")
idx_xg_away_pre = excel_col_to_index("J")

# xG (ACTUAL / AFTERMATCH)
idx_xg_home_actual = excel_col_to_index("AXV")
idx_xg_away_actual = excel_col_to_index("AXW")

# SCORES
idx_ft_score_home = excel_col_to_index("AXP")  # FT - Home Team Score
idx_ft_score_away = excel_col_to_index("AXQ")  # FT - Away Team Score
idx_total_goals = excel_col_to_index("AXR")    # Total Goals

# ELO
idx_home_elo = excel_col_to_index("DJ")
idx_away_elo = excel_col_to_index("DK")

# PRO METRICS
idx_form_tmb_home = excel_col_to_index("BV")
idx_form_tmb_away = excel_col_to_index("BZ")
idx_form_thbh_home = excel_col_to_index("CD")
idx_form_thbh_away = excel_col_to_index("CH")

idx_poisson_home = excel_col_to_index("LL")
idx_poisson_draw = excel_col_to_index("LM")
idx_poisson_away = excel_col_to_index("LN")

idx_odds_moves = excel_col_to_index("CW")

# OPENING ODDS
idx_open_home = excel_col_to_index("AJP")
idx_open_draw = excel_col_to_index("AJQ")
idx_open_away = excel_col_to_index("AJR")
idx_open_o25 = excel_col_to_index("AJY")
idx_open_u25 = excel_col_to_index("AKE")
idx_open_bttsy = excel_col_to_index("AKJ")
idx_open_bttsn = excel_col_to_index("AKK")

# CLOSING ODDS
idx_close_home = excel_col_to_index("ANR")
idx_close_draw = excel_col_to_index("ANS")
idx_close_away = excel_col_to_index("ANT")
idx_close_o25 = excel_col_to_index("AOA")
idx_close_u25 = excel_col_to_index("AOG")
idx_close_bttsy = excel_col_to_index("AOL")
idx_close_bttsn = excel_col_to_index("AOM")


COLUMN_MAPPING = {
    "Date": idx_date,
    "Time": idx_time,
    "Season": idx_season,
    "Home_Team": idx_home,
    "Away_Team": idx_away,
    "League": idx_league,
    "League_Pos_Home": idx_home_pos,
    "League_Pos_Away": idx_away_pos,
    "Manager_Games_Home": idx_home_manager_games,
    "Manager_Games_Away": idx_away_manager_games,
    # xG and scores
    "xG_Home": idx_xg_home_pre,
    "xG_Away": idx_xg_away_pre,
    "xG_Home_Actual": idx_xg_home_actual,
    "xG_Away_Actual": idx_xg_away_actual,
    "FT_Score_Home": idx_ft_score_home,
    "FT_Score_Away": idx_ft_score_away,
    "Total_Goals": idx_total_goals,
    "Elo_Home": idx_home_elo,
    "Elo_Away": idx_away_elo,
    "Form_TMB_Home": idx_form_tmb_home,
    "Form_TMB_Away": idx_form_tmb_away,
    "Form_THBH_Home": idx_form_thbh_home,
    "Form_THBH_Away": idx_form_thbh_away,
    "Poisson_Home_Pct": idx_poisson_home,
    "Poisson_Draw_Pct": idx_poisson_draw,
    "Poisson_Away_Pct": idx_poisson_away,
    "Odds_Moves_Raw": idx_odds_moves,
    "Odds_Open_Home": idx_open_home,
    "Odds_Open_Draw": idx_open_draw,
    "Odds_Open_Away": idx_open_away,
    "Odds_Open_Over25": idx_open_o25,
    "Odds_Open_Under25": idx_open_u25,
    "Odds_Open_BTTS_Yes": idx_open_bttsy,
    "Odds_Open_BTTS_No": idx_open_bttsn,
    "Odds_Close_Home": idx_close_home,
    "Odds_Close_Draw": idx_close_draw,
    "Odds_Close_Away": idx_close_away,
    "Odds_Close_Over25": idx_close_o25,
    "Odds_Close_Under25": idx_close_u25,
    "Odds_Close_BTTS_Yes": idx_close_bttsy,
    "Odds_Close_BTTS_No": idx_close_bttsn,
}


def _parse_date_token(filename):
    # Expect ddmmyy token (e.g., 251225) in filename.
    tokens = re.findall(r"(\d{6})", filename)
    if not tokens:
        return None
    token = tokens[-1]
    try:
        return datetime.strptime(token, "%d%m%y")
    except Exception:
        return None


def clean_football_data(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: file not found: {file_path}")
        return None

    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, header=None, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            continue

    if df is None:
        print("ERROR: failed to read file (format or structure issue).")
        return None

    extracted = {}
    total_cols = df.shape[1]
    for col_name, idx in COLUMN_MAPPING.items():
        if idx < total_cols:
            extracted[col_name] = df.iloc[2:, idx].values
        else:
            extracted[col_name] = np.full(len(df) - 2, np.nan)

    clean_df = pd.DataFrame(extracted)

    if "xG_Home" in clean_df and "xG_Away" in clean_df:
        clean_df["Supremacy_Calc"] = (
            pd.to_numeric(clean_df["xG_Home"], errors="coerce")
            - pd.to_numeric(clean_df["xG_Away"], errors="coerce")
        )

    text_cols = [
        "Date", "Time", "Season", "Home_Team", "Away_Team", "League",
        "Form_TMB_Home", "Form_TMB_Away",
        "Form_THBH_Home", "Form_THBH_Away",
        "Odds_Moves_Raw",
    ]

    for col in text_cols:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).replace("nan", np.nan)

    for col in clean_df.columns:
        if col not in text_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    return clean_df


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: INPUT_FILE not found: {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    token_date = _parse_date_token(os.path.basename(INPUT_FILE))
    if token_date is None:
        print("ERROR: could not parse date token (ddmmyy) from filename.")
        return

    df = clean_football_data(INPUT_FILE)
    if df is None or df.empty:
        print("ERROR: cleaned data is empty.")
        return

    # Normalize Date for splitting
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df["Date_only"] = df["Date"].dt.normalize()

    target_date = token_date.replace(hour=0, minute=0, second=0, microsecond=0)
    past_cutoff = target_date - timedelta(days=1)

    past_df = df[df["Date_only"] < target_date].copy()
    future_df = df[df["Date_only"] >= target_date].copy()

    past_name = f"past_cleaned{past_cutoff.strftime('%d%m%y')}.csv"
    past_path = os.path.join(OUTPUT_DIR, past_name)
    past_df.drop(columns=["Date_only"]).to_csv(past_path, index=False)
    print(f"Saved: {past_path} ({len(past_df)} rows)")

    if future_df.empty:
        print("No future rows found for target date and after.")
        return

    for day, day_df in future_df.groupby("Date_only"):
        day_str = day.strftime("%d%m%y")
        out_name = f"future_cleaned{day_str}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        day_df.drop(columns=["Date_only"]).to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(day_df)} rows)")


if __name__ == "__main__":
    main()
