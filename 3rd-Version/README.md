# Futbol Analist (appfinal1.py) - Usage & Deployment Notes

## Quick start (local)
1) Create a virtual environment (recommended):
   - python -m venv .venv
   - .venv\Scripts\activate
2) Install dependencies:
   - pip install streamlit pandas numpy scikit-learn
3) Run the app:
   - streamlit run appfinal1.py

## Required files (keep these together)
These are needed for the app to show match results and statistics:
- appfinal1.py
- pred_results_log.csv
- params_latest.json
- Backtest-Out/
  - league_picktype_matrix.csv
  - league_picktype_matrix_policy.csv
  - policy_bucket_stats.csv
  - backtest_predictions.csv (optional but used by "Backtest'ten Logu Yeniden Uret (tam)")

If you move files, update paths inside appfinal1.py accordingly. The app expects files relative to its base directory.

## How results are handled
You can enter results in two ways:
1) UI manual edit:
   - "Tahmin Sonuclari" tab -> edit RES_FTHG/RES_FTAG -> "Sonuclari Kaydet"
2) CSV import:
   - Upload a CSV with columns:
     - Match_ID OR (Date, League, HomeTeam, AwayTeam)
     - FTHG, FTAG (or FT_Score_Home, FT_Score_Away)
   - Click "Results import uygula"
The app will update RES_* columns and recompute statistics.

## Server deployment notes
- Streamlit app runs as a single process by default.
- For production, use a process manager (e.g., systemd, pm2) and a reverse proxy (nginx) if needed.
- Consider setting STREAMLIT_SERVER_HEADLESS=true and STREAMLIT_SERVER_PORT.

## Database migration (future)
If you want a web site + DB backend:
- Treat pred_results_log.csv as the single source of truth.
- Create DB tables for:
  - matches (Match_ID, Date, League, HomeTeam, AwayTeam)
  - predictions (PICK_FINAL, PICK_GROUP, odds, probabilities, policy flags)
  - results (RES_FTHG, RES_FTAG, RES_RESULT_1X2, RES_RESULT_OU, RES_RESULT_BTTS)
- Add a data loader that imports from pred_results_log.csv into DB.
- Replace CSV reads/writes with DB queries for scalability.

## Data integrity tips
- Always keep a backup of pred_results_log.csv before bulk updates.
- Avoid overwriting log data without merge logic.
- Verify RES_* fields after imports.
