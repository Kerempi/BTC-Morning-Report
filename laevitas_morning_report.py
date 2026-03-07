import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request

import matplotlib.pyplot as plt

from laevitas_api import laevitas_get

REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "laevitas"
DATA_DIR = Path("data") / "laevitas"
ARCHIVE_CSV = DATA_DIR / "btc_daily_archive.csv"
STRIKE_FLOW_CSV = DATA_DIR / "btc_daily_strike_flow.csv"
INSTRUMENT_FLOW_CSV = DATA_DIR / "btc_daily_instrument_flow.csv"
UTC_NOW = lambda: datetime.now(timezone.utc)
VOL_CONTEXT_EXPIRY = "25SEP26"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=40&interval=daily"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_pages_index(currency: str = "BTC") -> None:
    generated = UTC_NOW().strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{currency} Morning Reports</title>
<style>
body {{ margin:0; font-family: 'Trebuchet MS', Verdana, sans-serif; background:linear-gradient(135deg,#0a1322,#12263e 55%,#1b3554); color:#f5efe3; }}
main {{ max-width:920px; margin:0 auto; padding:42px 20px 60px; }}
h1 {{ font-family:'Palatino Linotype', Georgia, serif; font-size:44px; line-height:1; margin:0 0 12px; }}
p {{ line-height:1.6; color:#d8d0c4; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin-top:24px; }}
.card {{ background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); border-radius:18px; padding:20px; box-shadow:0 14px 30px rgba(0,0,0,0.18); }}
.card h2 {{ margin:0 0 8px; font-size:24px; }}
.button {{ display:inline-block; margin-top:12px; padding:10px 14px; border-radius:999px; color:#081321; background:#f4e3b2; text-decoration:none; font-weight:700; }}
.meta {{ margin-top:26px; font-size:13px; color:#cabfae; }}
</style>
</head>
<body>
<main>
  <h1>{currency} Options Desk Note</h1>
  <p>Automated morning report published via GitHub Pages. Choose the Turkish or English version below.</p>
  <div class='grid'>
    <div class='card'>
      <h2>Turkce</h2>
      <p>Guncel sabah opsiyon raporu, grafikler ve desk note ozeti.</p>
      <a class='button' href='btc_morning_report.html'>Raporu Ac</a>
    </div>
    <div class='card'>
      <h2>English</h2>
      <p>Updated morning options report with charts and desk-note summary.</p>
      <a class='button' href='btc_morning_report_en.html'>Open Report</a>
    </div>
  </div>
  <p class='meta'>Last generated: {generated}</p>
</main>
</body>
</html>"""
    (REPORT_DIR / "index.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / ".nojekyll").write_text("", encoding="utf-8")


def fetch_expiry_oi(market: str, currency: str) -> Dict:
    return laevitas_get(f"/analytics/options/oi_expiry/{market}/{currency}")


def fetch_gex_all(market: str, currency: str) -> Dict:
    return laevitas_get(f"/analytics/options/gex_date_all/{market}/{currency}")


def fetch_oi_strike_all(market: str, currency: str) -> Dict:
    return laevitas_get(f"/analytics/options/oi_strike_all/{market}/{currency}")


def fetch_oi_type(market: str, currency: str) -> Dict:
    return laevitas_get(f"/analytics/options/oi_type/{market}/{currency}")


def fetch_oi_change_summary(currency: str, hours: int) -> Dict:
    return laevitas_get(f"/analytics/options/open_interest_change_summary/{currency}/{hours}")


def fetch_vol_context(currency: str, maturity: str) -> Dict:
    return laevitas_get(f"/analytics/options/model_charts/vol_run/{currency}/{maturity}")


def fetch_atm_iv_ts(market: str, currency: str):
    return laevitas_get(f"/analytics/options/atm_iv_ts/{market}/{currency}")


def fetch_iv_table(market: str, currency: str) -> Dict:
    return laevitas_get(f"/analytics/options/iv_table/{market}/{currency}")


def fetch_top_strategies(currency: str, hours: int):
    return laevitas_get(f"/analytics/options/options_strategy/top_options_strategies/{currency}/{hours}/true")


def fetch_top_instrument_oi_change(market: str, currency: str, hours: int) -> Dict:
    return laevitas_get(f"/analytics/options/top_instrument_oi_change/{market}/{currency}/{hours}")


def fetch_oi_net_change_all(market: str, currency: str, hours: int) -> Dict:
    return laevitas_get(f"/analytics/options/oi_net_change_all/{market}/{currency}/{hours}")


def save_snapshot(name: str, payload: Dict) -> Path:
    stamp = UTC_NOW().strftime("%Y%m%d")
    out = DATA_DIR / f"{name}_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out


def parse_expiry_rows(payload: Dict, market: str) -> List[Dict]:
    rows = payload.get("data", [])
    cleaned: List[Dict] = []
    for row in rows:
        maturity = row.get("maturity")
        if not maturity and row.get("ticker"):
            parts = str(row["ticker"]).split("-", 1)
            if len(parts) == 2:
                maturity = parts[1]
        if not maturity:
            continue
        cleaned.append(
            {
                "market": market.upper(),
                "maturity": maturity,
                "open_interest": float(
                    row.get("open_interest", row.get("notional_c", 0) + row.get("notional_p", 0))
                ),
                "open_interest_notional": float(
                    row.get("open_interest_notional", row.get("c", 0) + row.get("p", 0))
                ),
                "open_interest_change_usd": float(row.get("open_interest_change_usd", 0.0)),
                "open_interest_notional_change": float(row.get("open_interest_notional_change", 0.0)),
            }
        )
    return cleaned


def load_expiry_rows(currency: str, primary_market: str) -> Tuple[List[Dict], str, int]:
    payload = fetch_expiry_oi("aggregate", currency)
    save_snapshot(f"{currency.lower()}_oi_expiry_aggregate", payload)
    rows = parse_expiry_rows(payload, "aggregate")
    if rows:
        return rows, "aggregate", 1
    payload = fetch_expiry_oi(primary_market, currency)
    save_snapshot(f"{currency.lower()}_oi_expiry_{primary_market}", payload)
    rows = parse_expiry_rows(payload, primary_market)
    return rows, primary_market, 2


def aggregate_gex_by_strike(payload: Dict) -> List[Dict]:
    net = defaultdict(float)
    call = defaultdict(float)
    put = defaultdict(float)
    for row in payload.get("data", []):
        strike = int(row["strike"])
        gex = float(row["gex"])
        net[strike] += gex
        if row.get("option_type") == "C":
            call[strike] += gex
        else:
            put[strike] += gex
    return [
        {"strike": strike, "net_gex": net[strike], "call_gex": call[strike], "put_gex": put[strike]}
        for strike in sorted(net)
    ]


def parse_strike_oi_rows(payload: Dict) -> List[Dict]:
    rows = []
    for row in payload.get("data", []):
        strike = int(row["strike"])
        call_notional = float(row.get("notional_c", 0.0))
        put_notional = float(row.get("notional_p", 0.0))
        rows.append(
            {
                "strike": strike,
                "call_oi": float(row.get("c", 0.0)),
                "put_oi": float(row.get("p", 0.0)),
                "call_notional": call_notional,
                "put_notional": put_notional,
                "total_notional": call_notional + put_notional,
            }
        )
    return rows


def parse_type_rows(payload: Dict) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for row in payload.get("data", []):
        option_type = str(row.get("optionType", "")).upper()
        if not option_type:
            continue
        result[option_type] = {
            "open_interest": float(row.get("open_interest", 0.0)),
            "notional": float(row.get("notional", 0.0)),
            "premium": float(row.get("premium", 0.0)),
        }
    return result


def parse_change_rows(payload: Dict) -> List[Dict]:
    rows = []
    for row in payload.get("data", []):
        ticker = str(row.get("ticker", ""))
        maturity = ticker.split("-", 1)[1] if "-" in ticker else ticker
        rows.append(
            {
                "maturity": maturity,
                "open_interest": float(row.get("open_interest", 0.0)),
                "open_interest_notional": float(row.get("open_interest_notional", 0.0)),
                "open_interest_change_usd": float(row.get("open_interest_change_usd", 0.0)),
                "open_interest_notional_change": float(row.get("open_interest_notional_change", 0.0)),
            }
        )
    return rows


def parse_vol_context(payload: Dict) -> Dict[str, float]:
    rows = payload.get("data", [])
    if not rows:
        return {}
    out = {}
    for key, value in rows[0].items():
        out[key] = float(value) if isinstance(value, (int, float)) else value
    return out


def parse_atm_iv_ts(payload: Dict) -> Dict[str, List[Dict[str, float]]]:
    data = payload.get("data", {})
    parsed: Dict[str, List[Dict[str, float]]] = {}
    for label, rows in data.items():
        cleaned = []
        for row in rows:
            maturity = row.get("maturity")
            iv = row.get("iv")
            if not maturity or iv is None:
                continue
            cleaned.append({"maturity": str(maturity), "iv": float(iv)})
        if cleaned:
            parsed[str(label)] = cleaned
    return parsed


def parse_iv_table_rows(payload: Dict) -> List[Dict[str, float]]:
    rows = []
    for row in payload.get("data", []):
        cleaned = {}
        for key, value in row.items():
            cleaned[key] = float(value) if isinstance(value, (int, float)) else value
        if cleaned.get("maturity"):
            rows.append(cleaned)
    return rows


def parse_top_strategy_rows(payload) -> List[Dict[str, float]]:
    source = payload if isinstance(payload, list) else payload.get("data", [])
    rows = []
    for row in source:
        rows.append(
            {
                "strategy": str(row.get("strategy", "")),
                "contracts": float(row.get("contracts", 0.0)),
                "total_notional": float(row.get("total_notional", 0.0)),
                "total_premium_usd": float(row.get("total_premium_usd", 0.0)),
                "net_premium_usd": float(row.get("net_premium_usd", 0.0)),
                "oi_change": float(row.get("oi_change", 0.0)),
                "delta": float(row.get("delta", 0.0)),
                "gamma": float(row.get("gamma", 0.0)),
                "vega": float(row.get("vega", 0.0)),
                "theta": float(row.get("theta", 0.0)),
            }
        )
    return rows


def parse_top_instrument_rows(payload: Dict) -> List[Dict]:
    rows = []
    for row in payload.get("data", []):
        instrument = str(row.get("instrument", ""))
        parts = instrument.split("-")
        maturity = parts[1] if len(parts) > 1 else ""
        strike = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        option_type = parts[3] if len(parts) > 3 else ""
        rows.append(
            {
                "instrument": instrument,
                "maturity": maturity,
                "strike": strike,
                "type": option_type,
                "volume": float(row.get("volume", 0.0)),
                "notional": float(row.get("notional", 0.0)),
                "open_interest": float(row.get("open_interest", 0.0)),
                "oi_change_notional": float(row.get("oi_change_notional", 0.0)),
            }
        )
    return rows


def parse_oi_net_change_all_rows(payload: Dict) -> List[Dict]:
    rows = []
    for row in payload.get("data", []):
        strike = int(row.get("strike", 0))
        c_notional = row.get("c_notional")
        p_notional = row.get("p_notional")
        c_change = float(c_notional) if c_notional is not None else 0.0
        p_change = float(p_notional) if p_notional is not None else 0.0
        rows.append(
            {
                "strike": strike,
                "call_change_notional": c_change,
                "put_change_notional": p_change,
                "net_change_notional": c_change + p_change,
            }
        )
    return rows


def infer_spot(expiry_rows: List[Dict]) -> float:
    nearest = expiry_rows[0]
    notionals = nearest["open_interest_notional"] or 1.0
    return nearest["open_interest"] / notionals


def choose_pin_band(gex_rows: List[Dict], spot: float) -> Tuple[int, int, int]:
    positives = [row for row in gex_rows if row["net_gex"] > 0]
    positives.sort(key=lambda row: row["net_gex"], reverse=True)
    nearby = [row for row in positives if abs(row["strike"] - spot) <= 6000]
    focus = nearby[:6] or positives[:6]
    weighted = sum(row["strike"] * row["net_gex"] for row in focus)
    total = sum(row["net_gex"] for row in focus) or 1.0
    center = int(round(weighted / total / 1000.0) * 1000)
    lower = int(min(row["strike"] for row in focus))
    upper = int(max(row["strike"] for row in focus))
    return lower, center, upper


def fmt_usd(value: float) -> str:
    return f"${value:,.0f}"


def fetch_public_btc_daily_prices() -> List[float]:
    with urllib.request.urlopen(COINGECKO_MARKET_CHART, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    prices = payload.get("prices", [])
    closes = [float(row[1]) for row in prices if isinstance(row, list) and len(row) >= 2]
    if len(closes) < 10:
        raise RuntimeError("Public BTC price history did not contain enough points for realized volatility.")
    return closes


def compute_realized_vols(closes: List[float], windows: List[int]) -> Dict[str, float]:
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    out: Dict[str, float] = {}
    for window in windows:
        sample = returns[-window:]
        if len(sample) < 2:
            continue
        mean = sum(sample) / len(sample)
        variance = sum((value - mean) ** 2 for value in sample) / (len(sample) - 1)
        out[f"rv_{window}d"] = math.sqrt(variance) * math.sqrt(365) * 100
    return out


def iv_lookup(rows: List[Dict[str, float]], label: str) -> Dict[str, float]:
    return {row["maturity"]: float(row["iv"]) for row in rows.get(label, [])}


def build_iv_term_points(iv_table_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    return [
        {
            "maturity": row["maturity"],
            "atm": to_float(row.get("mark_ATM", 0.0)),
            "rr_25d": to_float(row.get("25D_Mark_RR", 0.0)),
            "bf_25d": to_float(row.get("25D_Mark_BF", 0.0)),
            "rr_10d": to_float(row.get("10D_Mark_RR", 0.0)),
            "bf_10d": to_float(row.get("10D_Mark_BF", 0.0)),
        }
        for row in iv_table_rows
    ]


def first_valid_iv_row(rows: List[Dict[str, float]], fallback_atm: float, fallback_maturity: str) -> Dict[str, float]:
    for row in rows:
        if to_float(row.get("atm", 0.0)) > 0:
            return row
    return {
        "maturity": fallback_maturity,
        "atm": fallback_atm,
        "rr_25d": 0.0,
        "bf_25d": 0.0,
        "rr_10d": 0.0,
        "bf_10d": 0.0,
    }


def classify_strategy_bucket(name: str) -> str:
    upper = name.upper()
    if "LONG_PUT" in upper or "BEAR_PUT_SPREAD" in upper or "SHORT_RISK_REVERSAL" in upper:
        return "defensive"
    if "SHORT_PUT" in upper or "LONG_CALL" in upper or "BULL" in upper or "ROLL_UP" in upper:
        return "constructive"
    if "STRADDLE" in upper or "STRANGLE" in upper or "CALENDAR" in upper:
        return "vol"
    if "SHORT_CALL" in upper or "BEAR_CALL_SPREAD" in upper or "ROLL_BACK" in upper:
        return "capped"
    return "mixed"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def archive_fieldnames() -> List[str]:
    return [
        "date",
        "generated_utc",
        "spot",
        "pin_low",
        "pin_center",
        "pin_high",
        "total_net_gex",
        "next_day_maturity",
        "next_day_oi",
        "next_day_oi_change_usd",
        "next_day_share",
        "call_put_notional_ratio",
        "call_put_oi_ratio",
        "atm_iv",
        "front_atm_iv",
        "rv_7d",
        "rv_30d",
        "iv_rv_7d_spread",
        "inventory_net",
        "rr_25d",
        "skew_25d",
        "total_visible_oi",
        "front_total_oi_3d",
        "top_pos_1",
        "top_pos_2",
        "top_neg_1",
        "top_neg_2",
    ]


def strike_flow_fieldnames() -> List[str]:
    return ["date", "strike", "call_change_notional", "put_change_notional", "net_change_notional"]


def instrument_flow_fieldnames() -> List[str]:
    return ["date", "instrument", "maturity", "strike", "type", "volume", "notional", "open_interest", "oi_change_notional"]


def load_archive() -> List[Dict[str, str]]:
    if not ARCHIVE_CSV.exists():
        return []
    with ARCHIVE_CSV.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def upsert_daily_rows(path: Path, fieldnames: List[str], date_key: str, rows: List[Dict[str, object]], unique_cols: List[str]) -> List[Dict[str, str]]:
    existing = load_csv_rows(path)
    incoming = [{k: str(v) for k, v in row.items()} for row in rows]
    def make_key(row: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(row.get(col, "") for col in unique_cols)
    incoming_keys = {make_key(row) for row in incoming}
    kept = [row for row in existing if not (row.get(date_key) == incoming[0].get(date_key, "") and make_key(row) in incoming_keys)] if incoming else existing
    merged = kept + incoming
    merged.sort(key=lambda row: tuple(row.get(col, "") for col in [date_key] + unique_cols))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    return merged


def upsert_archive(row: Dict[str, object]) -> List[Dict[str, str]]:
    existing = load_archive()
    row_s = {k: str(v) for k, v in row.items()}
    rows = [r for r in existing if r.get("date") != row_s["date"]]
    rows.append(row_s)
    rows.sort(key=lambda item: item["date"])
    with ARCHIVE_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=archive_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def has_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def summarize_flow(
    expiry_rows: List[Dict],
    gex_rows: List[Dict],
    strike_rows: List[Dict],
    type_rows: Dict[str, Dict[str, float]],
    change_rows: List[Dict],
    vol_context: Dict[str, float],
    atm_iv_ts: Dict[str, List[Dict[str, float]]],
    iv_table_rows: List[Dict[str, float]],
    strategy_rows: List[Dict[str, float]],
    realized_vols: Dict[str, float],
    top_instrument_rows: List[Dict],
    strike_change_rows: List[Dict],
    expiry_market: str,
    gex_market: str,
) -> Dict[str, object]:
    spot = infer_spot(expiry_rows)
    pin_low, pin_center, pin_high = choose_pin_band(gex_rows, spot)
    total_net_gex = sum(row["net_gex"] for row in gex_rows)
    positive_nodes = sorted(gex_rows, key=lambda row: row["net_gex"], reverse=True)[:5]
    negative_nodes = sorted(gex_rows, key=lambda row: row["net_gex"])[:5]
    nearby_strikes = [row for row in strike_rows if abs(row["strike"] - spot) <= 15000]
    crowded = sorted(nearby_strikes, key=lambda row: row["total_notional"], reverse=True)[:8]
    call_crowded = sorted(nearby_strikes, key=lambda row: row["call_notional"], reverse=True)[:5]
    put_crowded = sorted(nearby_strikes, key=lambda row: row["put_notional"], reverse=True)[:5]

    next_day = expiry_rows[1] if len(expiry_rows) > 1 else expiry_rows[0]
    total_visible_oi = sum(row["open_interest"] for row in expiry_rows)
    next_day_share = next_day["open_interest"] / total_visible_oi if total_visible_oi else 0.0
    front_total_oi_3d = sum(row["open_interest"] for row in expiry_rows[:3])

    call_notional = type_rows.get("C", {}).get("notional", 0.0)
    put_notional = type_rows.get("P", {}).get("notional", 0.0)
    call_oi = type_rows.get("C", {}).get("open_interest", 0.0)
    put_oi = type_rows.get("P", {}).get("open_interest", 0.0)
    notional_ratio = call_notional / put_notional if put_notional else 0.0
    oi_ratio = call_oi / put_oi if put_oi else 0.0

    next_day_change = 0.0
    for row in change_rows:
        if row["maturity"] == next_day["maturity"]:
            next_day_change = row["open_interest_change_usd"]
            break

    rr_25d = float(vol_context.get("rr_25d", 0.0))
    skew_25d = float(vol_context.get("skew_25d", 0.0))
    atm_iv = float(vol_context.get("atm_spot_iv", 0.0))
    iv_term_rows = build_iv_term_points(iv_table_rows)
    today_curve = iv_lookup(atm_iv_ts, "Today")
    ts_front_candidates = list(today_curve.items())
    ts_front_fallback_maturity = ts_front_candidates[0][0] if ts_front_candidates else next_day["maturity"]
    ts_front_fallback_iv = ts_front_candidates[0][1] if ts_front_candidates else atm_iv
    front_iv_row = first_valid_iv_row(iv_term_rows, ts_front_fallback_iv or atm_iv, ts_front_fallback_maturity)
    far_candidates = [row for row in iv_term_rows if to_float(row.get("atm", 0.0)) > 0]
    if far_candidates:
        far_iv_row = far_candidates[min(len(far_candidates) - 1, 5)]
    else:
        far_iv_row = front_iv_row
    front_atm_iv = float(front_iv_row.get("atm", atm_iv))
    far_atm_iv = float(far_iv_row.get("atm", front_atm_iv))
    term_slope = front_atm_iv - far_atm_iv
    front_rr_25d = float(front_iv_row.get("rr_25d", rr_25d))
    front_bf_25d = float(front_iv_row.get("bf_25d", 0.0))
    front_rr_10d = float(front_iv_row.get("rr_10d", 0.0))
    front_bf_10d = float(front_iv_row.get("bf_10d", 0.0))
    rv_7d = float(realized_vols.get("rv_7d", 0.0))
    rv_14d = float(realized_vols.get("rv_14d", 0.0))
    rv_30d = float(realized_vols.get("rv_30d", 0.0))
    iv_rv_7d_spread = front_atm_iv - rv_7d if rv_7d else 0.0
    iv_rv_30d_spread = front_atm_iv - rv_30d if rv_30d else 0.0

    yesterday_curve = iv_lookup(atm_iv_ts, "Yesterday")
    week_curve = iv_lookup(atm_iv_ts, "1 Week Ago")
    common_today_yday = [m for m in today_curve if m in yesterday_curve]
    common_today_week = [m for m in today_curve if m in week_curve]
    ts_front_maturity = common_today_yday[0] if common_today_yday else (iv_term_rows[0]["maturity"] if iv_term_rows else next_day["maturity"])
    ts_day_change = today_curve.get(ts_front_maturity, front_atm_iv) - yesterday_curve.get(ts_front_maturity, today_curve.get(ts_front_maturity, front_atm_iv))
    ts_week_change = today_curve.get(ts_front_maturity, front_atm_iv) - week_curve.get(ts_front_maturity, today_curve.get(ts_front_maturity, front_atm_iv))

    if total_net_gex > 0 and pin_low <= spot <= pin_high:
        base_case = (
            f"Pozitif dealer gamma spotu {pin_low:,.0f}-{pin_high:,.0f} bandinda "
            f"ortalamaya donen bir yapida tutmali; en guclu pin seviyesi {pin_center:,.0f} civari."
        )
    elif total_net_gex > 0:
        base_case = (
            f"Pozitif gamma destekleyici kalmaya devam ediyor, ancak spot ana pin bandinin disinda; "
            f"fiyat yeni bir kabul alani olusturmazsa {pin_center:,.0f} seviyesine dogru cekilme beklenir."
        )
    else:
        base_case = "Net gamma kirilgan. Basit bir pin gununden daha yonlu bir realized volatilite beklenmeli."

    alt_case = (
        f"Fiyat {pin_high:,.0f} uzerinde kabul gorurse, {call_crowded[0]['strike']:,.0f} ve "
        f"{call_crowded[1]['strike']:,.0f} civarindaki call yigilmasi hareketi momentumlu bir yukari uzamaya tasiyabilir."
    )
    invalidation = (
        f"{pin_low:,.0f} altinda kalici bir kirilma, {negative_nodes[0]['strike']:,.0f} ve "
        f"{negative_nodes[1]['strike']:,.0f} civarindaki negatif-gamma ceplerini aciga cikarir."
    )

    dealer = [
        (
            "Dealerlar toplamda hala net long gamma tasiyor; bu nedenle hedge akislarinin "
            "spotun bulundugu bolgede gun ici hareket araligini bastirmasi beklenir."
        )
        if total_net_gex > 0
        else "Dealer tarafinda yonlu devam hareketini bastiracak kadar pozitif gamma yok.",
        (
            f"En yogun pozitif GEX dugumleri {', '.join(f'{row['strike']:,.0f}' for row in positive_nodes[:3])} "
            "seviyelerinde; bu alanlar ani squeeze'ten cok yukari yonlu miknatis etkisi yaratir."
        ),
        (
            f"Asagi yonlu en hassas hedge cepleri {negative_nodes[0]['strike']:,.0f} ve "
            f"{negative_nodes[1]['strike']:,.0f} seviyelerinde kaliyor; bu bolgenin kirilmasi hedge talebini hizlandirabilir."
        ),
    ]

    crowding = [
        (
            f"Call notional hala put notional'in {notional_ratio:.2f} kati, open interest adedi de call lehine "
            f"{oi_ratio:.2f} kat. Bu panic hedge degil, yapici bir konumlanma."
        ),
        (
            f"Spota yakin en buyuk call yigilmalari {', '.join(f'{row['strike']:,.0f}' for row in call_crowded[:3])}; "
            f"en buyuk put yigilmalari ise {', '.join(f'{row['strike']:,.0f}' for row in put_crowded[:3])}."
        ),
        (
            f"Ertesi gun vadesi olan {next_day['maturity']} tum gorunen expiry OI'nin yalnizca {next_day_share:.1%}'i kadar; "
            "bu nedenle spot dogrudan kalabalik strike'lara gitmedikce tek basina tum bandi suruklemesi zor."
        ),
    ]

    pricing = [
        (
            f"Piyasa crash paniginden cok duzenli bir asagi yon korumasi fiyatliyor: "
            f"{vol_context.get('expiry', VOL_CONTEXT_EXPIRY)} vol baglaminda 25-delta risk reversal {rr_25d:.2f}, "
            f"25-delta skew ise {skew_25d:.2f}."
        ),
        f"ATM spot IV {atm_iv:.2f} civarinda; riskin ciddiye alindigini gosteriyor ama tek basina duzensizlik sinyali vermiyor.",
        f"Vade egirisinin on tarafina hala taze risk ekleniyor: ertesi gun vadesine son 24 saatte yaklasik {fmt_usd(next_day_change)} eklendi.",
    ]

    blind_spots = [
        (
            f"Spot {pin_low:,.0f} uzerinde kaldigi surece bant cok sert bir asagi kirilim icin hazir gorunmuyor; "
            "piyasanin ustunde ve etrafinda hala fazla miktarda pozitif gamma birikimi var."
        ),
        (
            f"Ayni anda piyasa temiz bir yukari kirilimi de agresif fiyatlamiyor. "
            f"{call_crowded[0]['strike']:,.0f} ve {call_crowded[1]['strike']:,.0f} civarindaki agir call pozisyonlari, "
            "spot bu seviyeleri hacimle asmadan chase davranisini sinirlayabilir."
        ),
        (
            f"Daha buyuk gizli risk, {negative_nodes[0]['strike']:,.0f} altinda rejim degisimi yasanmasi; "
            "bu bolgede dealer hedging hareketi emmek yerine buyutmeye baslayabilir."
        ),
    ]

    simple_summary = [
        f"Bugun icin ana tablo: fiyatin {pin_low:,.0f}-{pin_high:,.0f} bandinda daha kontrollu kalmasi bekleniyor.",
        f"Yukari tarafta {call_crowded[0]['strike']:,.0f} ve {call_crowded[1]['strike']:,.0f} seviyeleri hem kalabalik hem de dikkat cekici call bolgeleri.",
        f"Asagi tarafta {negative_nodes[0]['strike']:,.0f} ve {negative_nodes[1]['strike']:,.0f} seviyeleri kirilma halinde davranisi degistirebilecek bolgeler.",
        "Kisa vade risk eklenmesi devam ediyor; bu da piyasanin yarina dair ilgisinin canli oldugunu gosteriyor.",
    ]

    top_adds = sorted(top_instrument_rows, key=lambda row: row["oi_change_notional"], reverse=True)[:8]
    top_cuts = sorted(top_instrument_rows, key=lambda row: row["oi_change_notional"])[:8]
    strike_adds = sorted(strike_change_rows, key=lambda row: row["net_change_notional"], reverse=True)[:8]
    strike_cuts = sorted(strike_change_rows, key=lambda row: row["net_change_notional"])[:8]

    add_puts = [row for row in top_adds if row["type"] == "P"]
    add_calls = [row for row in top_adds if row["type"] == "C"]
    cut_puts = [row for row in top_cuts if row["type"] == "P"]
    cut_calls = [row for row in top_cuts if row["type"] == "C"]
    dominant_add = add_puts[0] if add_puts else (add_calls[0] if add_calls else None)
    dominant_cut = cut_calls[0] if cut_calls else (cut_puts[0] if cut_puts else None)

    flow_summary = []
    if dominant_add:
        direction = "put korumasi" if dominant_add["type"] == "P" else "upside call talebi"
        flow_summary.append(
            f"Son 24 saatte en guclu yeni risk {dominant_add['instrument']} uzerinden geldi; bu, {direction} tarafinda aktif yeni konumlanma oldugunu gosteriyor."
        )
    if dominant_cut:
        cut_direction = "call" if dominant_cut["type"] == "C" else "put"
        flow_summary.append(
            f"Ayni anda en belirgin bosalma {dominant_cut['instrument']} tarafinda; bu da eski {cut_direction} riskinin bir kisminin temizlendigini gosteriyor."
        )
    if strike_adds:
        flow_summary.append(
            f"Strike bazli net ekleme en cok {strike_adds[0]['strike']:,.0f} ve {strike_adds[1]['strike']:,.0f} civarinda yogunlasmis; bu bolgeler kisa vadede yeni ilgi merkezi."
        )
    if strike_cuts:
        flow_summary.append(
            f"En sert net bosalma {strike_cuts[0]['strike']:,.0f} ve {strike_cuts[1]['strike']:,.0f} cevresinde; bu da eski pozisyonun oradan cekildigine isaret ediyor."
        )

    put_add_sum = sum(max(0.0, row["oi_change_notional"]) for row in top_instrument_rows if row["type"] == "P")
    call_add_sum = sum(max(0.0, row["oi_change_notional"]) for row in top_instrument_rows if row["type"] == "C")
    call_cut_sum = sum(abs(min(0.0, row["oi_change_notional"])) for row in top_instrument_rows if row["type"] == "C")
    put_cut_sum = sum(abs(min(0.0, row["oi_change_notional"])) for row in top_instrument_rows if row["type"] == "P")
    flow_score = 0
    if put_add_sum > call_add_sum * 1.15:
        flow_score -= 2
    elif call_add_sum > put_add_sum * 1.15:
        flow_score += 2
    if call_cut_sum > put_cut_sum * 1.2:
        flow_score -= 1
    elif put_cut_sum > call_cut_sum * 1.2:
        flow_score += 1
    if strike_adds and strike_adds[0]["strike"] < spot:
        flow_score -= 1
    elif strike_adds and strike_adds[0]["strike"] > spot:
        flow_score += 1
    if flow_score <= -2:
        flow_label = "Defensive / downside hedge akisi"
    elif flow_score >= 2:
        flow_label = "Constructive / upside chase akisi"
    else:
        flow_label = "Mixed / two-way akisi"

    vol_summary = []
    if term_slope >= 2:
        vol_summary.append(
            f"On vade ATM IV ({front_iv_row['maturity']}) {front_atm_iv:.2f}; uzak vade ATM IV ({far_iv_row['maturity']}) {far_atm_iv:.2f}. On taraf pahali ve term structure belirgin backwardation gosteriyor."
        )
    elif term_slope <= -2:
        vol_summary.append(
            f"On vade ATM IV ({front_iv_row['maturity']}) {front_atm_iv:.2f}; uzak vade ATM IV ({far_iv_row['maturity']}) {far_atm_iv:.2f}. Vol egirisi yukari egimli; piyasa yakin riski degil daha kalici belirsizligi fiyatliyor."
        )
    else:
        vol_summary.append(
            f"On vade ATM IV ({front_iv_row['maturity']}) {front_atm_iv:.2f}; uzak vade ATM IV ({far_iv_row['maturity']}) {far_atm_iv:.2f}. Vol egirisi gorece duz, yani yakin vade panik fiyati baskin degil."
        )
    if rv_7d:
        if iv_rv_7d_spread >= 8:
            vol_summary.append(
                f"Front ATM IV, 7 gunluk realized volun {iv_rv_7d_spread:.2f} puan uzerinde. Vol market spotun son gercek hareketine gore primli calisiyor."
            )
        elif iv_rv_7d_spread <= -5:
            vol_summary.append(
                f"Front ATM IV, 7 gunluk realized volun {abs(iv_rv_7d_spread):.2f} puan altinda. Spotun son hareketi opsiyon primlerinden daha agresif; underpriced hareket riski var."
            )
        else:
            vol_summary.append(
                f"Front ATM IV ile 7 gunluk realized vol arasindaki fark {iv_rv_7d_spread:.2f} puan. Vol market ile spot hareketi birbirinden kopuk degil."
            )
    if front_rr_25d <= -5:
        vol_summary.append(
            f"25d RR {front_rr_25d:.2f}; downside putlar call'lara gore anlamli primli. Bu savunmaci tail hedge talebini gosteriyor."
        )
    elif front_rr_25d >= 3:
        vol_summary.append(
            f"25d RR {front_rr_25d:.2f}; upside call talebi normalin uzerinde. Bu call chase veya yukari yon tasima istegini isaret ediyor."
        )
    else:
        vol_summary.append(
            f"25d RR {front_rr_25d:.2f}; skew savunmaci ama panik boyutunda degil. Piyasa asagiyi ciddiye aliyor fakat tek yonlu korku fiyatlamiyor."
        )
    vol_summary.append(
        f"10d RR {front_rr_10d:.2f} ve 25d BF {front_bf_25d:.2f}; bu kombinasyon kisa vadede tail priminin ne kadar yuksek oldugunu gosteriyor."
    )

    strategy_rows = sorted(strategy_rows, key=lambda row: row["total_notional"], reverse=True)
    top_strategies = strategy_rows[:8]
    strategy_totals = defaultdict(float)
    for row in top_strategies:
        strategy_totals[classify_strategy_bucket(row["strategy"])] += row["total_notional"]
    strategy_bias = max(strategy_totals.items(), key=lambda item: item[1])[0] if strategy_totals else "mixed"
    strategy_summary = []
    if top_strategies:
        lead = top_strategies[0]
        strategy_summary.append(
            f"En buyuk strategy baskisi {lead['strategy']} tarafinda; yaklasik {fmt_usd(lead['total_notional'])} notional ve {lead['oi_change']:,.0f} OI degisimi ile tape'i tasiyor."
        )
    if strategy_bias == "constructive":
        strategy_summary.append("Strategy tape genel olarak yapici; short put, long call veya bull spread benzeri yapilar savunmadan daha agir basiyor.")
    elif strategy_bias == "defensive":
        strategy_summary.append("Strategy tape savunmaci; long put ve downside yapilar piyasanin temkinli kaldigini gosteriyor.")
    elif strategy_bias == "vol":
        strategy_summary.append("Strategy tape directional olmaktan cok volatilite alimi/satimi uzerine kurulu; hareketin yonunden ziyade buyuklugu fiyatlanmaya calisiliyor.")
    elif strategy_bias == "capped":
        strategy_summary.append("Strategy tape yukariyi tamamen reddetmiyor ama short call / capped yapilar nedeniyle temiz bir breakout beklentisi de vermiyor.")
    else:
        strategy_summary.append("Strategy tape karisik; tek yonlu bir hikaye yerine birden fazla ihtimal ayni anda trade ediliyor.")
    if any("LONG_STRADDLE" in row["strategy"] or "LONG_STRANGLE" in row["strategy"] for row in top_strategies):
        strategy_summary.append("Long straddle/strangle varligi, bir kisim oyuncunun yon degil realized hareket satin aldigini gosteriyor.")

    final_direction = "range to slight upside"
    if total_net_gex <= 0 or spot < pin_low:
        final_direction = "downside risk"
    elif flow_score >= 2 and strategy_bias in {"constructive", "mixed"} and iv_rv_7d_spread > -2:
        final_direction = "controlled upside"
    elif flow_score <= -2 and front_rr_25d < -4:
        final_direction = "defensive downside"

    final_read = []
    if final_direction == "controlled upside":
        final_read.append(
            f"Birlesik okuma, bugun icin en olasi fiyatlamanin kontrollu yukari egilimli ama pin etkisi altinda oldugunu soyluyor. Dealer gamma hala pozitif, spot ana miknatis bolgesinin icinde ve flow tarafinda call / yapici risk eklemeleri savunmaci akisin onune gecmis durumda."
        )
    elif final_direction == "defensive downside":
        final_read.append(
            f"Birlesik okuma, piyasanin taban senaryosunun savunmaci oldugunu soyluyor. Skew asagi korumayi pahali tutuyor, flow daha cok hedge karakterinde ve spot pin bandinin altina sarkarsa dealer dengesi hareketi buyutebilir."
        )
    elif final_direction == "downside risk":
        final_read.append(
            f"Birlesik okuma, pin mekanizmasinin zayifladigini ve piyasanin daha yonlu bir asagi hareket riski tasidigini gosteriyor. Bu durumda opsiyon piyasasi spota istikrar degil ivme ekleyebilir."
        )
    else:
        final_read.append(
            f"Birlesik okuma, opsiyon piyasasinin halen net bir trend gununden cok kontrollu bir range/pin gunu fiyatladigini gosteriyor. Dealerlar spotu {pin_low:,.0f}-{pin_high:,.0f} bandinda tutmaya calisirken, vol market kisa vade riski tamamen dislamiyor ama panic rejimine de gecmis degil."
        )
    final_read.append(
        f"Vol market tarafinda front ATM IV {front_atm_iv:.2f}, 7 gunluk realized vol {rv_7d:.2f} ve IV-RV spread {iv_rv_7d_spread:.2f}. Bu, opsiyon primlerinin son spot hareketine gore {'primli' if iv_rv_7d_spread > 4 else 'dengeye yakin'} oldugunu gosteriyor."
    )
    final_read.append(
        f"Strategy tape tarafinda baskin ton {strategy_bias}. Bu nedenle bugunun okumasini sadece strike yigilmasi degil, aktif trade edilen stratejilerin ne anlattigi da destekliyor."
    )

    morning_note = []
    morning_note.append(
        f"Bugune girerken opsiyon piyasasi tek yonlu bir breakout degil, once kontrol sonra secici yon genislemesi fiyatliyor. En kritik nokta dealer gamma rejiminin hala pozitif kalmasi; bu da spot {pin_low:,.0f}-{pin_high:,.0f} bandinin icinde kaldigi surece sert intraday genislemelerin hedge akisiyla bastirilma olasiligini yuksek tutuyor. Baska bir deyisle masa acilisinda ilk baz senaryo trend gunu degil, pin/range gunu."
    )
    morning_note.append(
        f"Bu range okumasinin merkezi {pin_center:,.0f}. Cunku hem gamma miknatislari hem de strike bazli acik pozisyon yogunlugu spotu bu eksene geri cekme egiliminde. Yukari tarafta en kalabalik call cepleri {call_crowded[0]['strike']:,.0f} ve {call_crowded[1]['strike']:,.0f}; bunlar ilk asamada hedef kadar direnc de yaratabilir. Asagi tarafta ise {negative_nodes[0]['strike']:,.0f} ve {negative_nodes[1]['strike']:,.0f} seviyeleri rejim degisimi alanlari. Spot bu bolgelerin altina kabul verirse opsiyon piyasasi hareketi emen degil, buyuten tarafa gecebilir."
    )
    morning_note.append(
        f"Vol tarafinda tablo daha incelikli. Front ATM IV {front_atm_iv:.2f}, uzak vade ATM IV {far_atm_iv:.2f} ve term slope {term_slope:+.2f}. Bu egri {'yakina risk primi yuklendiyini' if term_slope > 2 else 'yakinda panik olmadigini ama riskin canli kaldigini' if term_slope > -2 else 'riski daha uzun zamana yaydigini'} soyluyor. Daha onemlisi, 7 gunluk realized vol {rv_7d:.2f} iken front IV ile fark {iv_rv_7d_spread:+.2f}. Yani opsiyon primi {'spotun son hareketine gore pahali' if iv_rv_7d_spread > 6 else 'spotla uyumlu' if iv_rv_7d_spread > -3 else 'spotun gerisinde kaliyor'}; bu ayrim gun icinde vol satmak mi vol kovalamak mi sorusunun temel cevabi."
    )
    morning_note.append(
        f"Skew ve smile verisi halen savunmaci bir alt ton tasiyor. Front 25d risk reversal {front_rr_25d:.2f}, 10d risk reversal {front_rr_10d:.2f} ve 25d butterfly {front_bf_25d:.2f}. Bu kombinasyon piyasada tamamen rahat bir upside chase olmadigini, asagi kuyruk riskinin hala fiyatlandigini gosteriyor. Ancak bu tek basina bearish bir tape demek degil; daha cok, yukari tasinmak istense bile bunun korumasiz yapilmadigini soyluyor."
    )
    morning_note.append(
        f"Akis ve strategy tape bu resmi tamamliyor. Flow skoru {flow_score:+d} ve baskin ton {flow_label.lower()}. Strategy tarafinda one cikan grup {strategy_bias}. Bu, masaya su mesaji veriyor: oyuncular yalnizca eski pozisyonu tasimiyor, aktif olarak yeni risk insa ediyor. Eger yapici call/short put yapilari savunmaci put taleplerini dengeliyorsa piyasa yukariyi tamamen reddetmiyor; ancak short call, capped yapilar veya agir hedge akisi artiyorsa yukari hareketin kalitesi zayif kaliyor."
    )
    morning_note.append(
        f"Sonucta bugun icin operasyonel okuma su: ilk senaryo {pin_low:,.0f}-{pin_high:,.0f} icinde kontrollu fiyatlama, merkezde {pin_center:,.0f}. Yukari uzama olacaksa once {call_crowded[0]['strike']:,.0f} sonra {call_crowded[1]['strike']:,.0f} test edilir, fakat bunun trend gunune donusmesi icin bu seviyelerde kabul ve yeni call eklenmesi gorulmeli. Asagi senaryoda ise {pin_low:,.0f} alti sadece teknik zayiflik degil, opsiyon rejim degisimi anlamina gelir; orada hedge akisinin hizi artar ve spot beklenenden daha hizli asagi tasinabilir."
    )
    morning_note.append(
        "Kisacasi bu raporun bugun icin verdigi masa notu su: piyasa su an kaotik bir kopus degil, kontrol edilen bir fiyat kesfi evresinde. Dealerlar spotu sabitlemeye calisiyor, vol market riski tamamen dusurmuyor, strateji tape ise yon yerine secici ve korunmali risk alindigini anlatiyor. Bu nedenle sabah ilk okuma range/pin; gun icindeki asil soru ise bu denge kirildiginda opsiyon piyasasinin hangi yone ivme ekleyecegi."
    )

    gamma_regime = "positive" if total_net_gex > 0 else "negative"
    gamma_text = (
        f"Gamma {pin_low:,.0f}-{pin_high:,.0f} bandinda pozitif. Spot band icinde kaldigi surece MM hedge akisi hareketi bastirmaya yatkin."
        if total_net_gex > 0
        else f"Gamma kirilgan/negatif. MM hedge akisi hareketi bastirmaktan cok buyutebilir."
    )
    balance_text = f"Balance: {pin_center:,.0f} ({'guclu' if total_net_gex > 0 else 'zayif'})"
    upside_test_1 = call_crowded[0]["strike"]
    upside_test_2 = call_crowded[1]["strike"]
    downside_test_1 = pin_low
    downside_test_2 = negative_nodes[0]["strike"]
    key_takeaways = [
        gamma_text,
        balance_text,
        f"Upside test: {pin_high:,.0f} -> {upside_test_1:,.0f} -> {upside_test_2:,.0f}",
        f"Danger zone yukarida: {pin_high:,.0f} ustu ilk anda guvenilmez olabilir; temiz trend icin {upside_test_1:,.0f} ustunde kabul gerekli.",
        f"Downside test: {downside_test_1:,.0f} -> {downside_test_2:,.0f}",
        f"Asagi kirilimda {downside_test_1:,.0f} alti hedge baskisini buyutebilir; esas hizlanma riski {downside_test_2:,.0f} civarinda.",
    ]

    gamma_score = clamp(total_net_gex * 4, -5, 5)
    flow_proxy_score = clamp(flow_score * 1.25, -5, 5)
    skew_score = clamp((-front_rr_25d) / 2.0, -5, 5)
    vol_risk_score = clamp((-iv_rv_7d_spread) / 4.0, -5, 5)
    strategy_map = {"constructive": 2.0, "defensive": -2.0, "vol": -0.5, "capped": -1.5, "mixed": 0.0}
    strategy_score = strategy_map.get(strategy_bias, 0.0)
    inventory_net = clamp((gamma_score * 0.35) + (flow_proxy_score * 0.2) + (skew_score * 0.15) + (vol_risk_score * 0.15) + (strategy_score * 0.15), -5, 5)
    inventory_label = (
        "MM long gamma / stabilizing"
        if inventory_net >= 1.5
        else "MM neutral to mixed"
        if inventory_net > -1.5
        else "MM short gamma / unstable"
    )
    inventory_rows = [
        {"label": "Gamma rejimi", "score": gamma_score},
        {"label": "Flow baskisi", "score": flow_proxy_score},
        {"label": "Skew savunmasi", "score": skew_score},
        {"label": "IV-RV gerilimi", "score": vol_risk_score},
        {"label": "Strategy tape", "score": strategy_score},
        {"label": "Net proxy", "score": inventory_net},
    ]

    squeeze_trigger = upside_test_1
    breakout_confirm = upside_test_2
    breakdown_trigger = downside_test_1
    acceleration_trigger = downside_test_2
    reject_upside = pin_high
    reject_downside = pin_low
    acceptance_rules = [
        f"Accept above {squeeze_trigger:,.0f}: spot bu seviye ustunde kalir ve yeni call akisiyla desteklenirse yukari test aktif olur.",
        f"Reject above {reject_upside:,.0f}: fiyat bu bolgeyi sadece igneleyip geri donerse ralliye guvenme; pin/range mantigi calisiyor demektir.",
        f"Accept below {breakdown_trigger:,.0f}: spot bu seviye altinda kalirsa downside hedge akisi kuvvetlenir ve {acceleration_trigger:,.0f} ikinci hedef olur.",
        f"Reject below {reject_downside:,.0f}: fiyat altina sarkip hizla geri aliniyorsa asagi kirilim degil stop-hunt / range return okunmali.",
    ]
    playbook = [
        f"Base case: {pin_low:,.0f}-{pin_high:,.0f} icinde pin/range. Ilk beklenen davranis mean reversion ve {pin_center:,.0f} merkezine geri cekilme.",
        f"Bull case: {squeeze_trigger:,.0f} ustunde kabul gorulurse yukari test {breakout_confirm:,.0f} seviyesine uzayabilir. Temiz trend icin yeni call eklenmesi ve flow tonunun iyilesmesi lazim.",
        f"Bear case: {breakdown_trigger:,.0f} alti kabul downside hedge akislarini hizlandirir; {acceleration_trigger:,.0f} civari ikinci hizlanma bolgesi.",
    ]
    takeaways_structured = {
        "gamma": gamma_text,
        "balance": balance_text,
        "upside": f"{pin_high:,.0f} -> {upside_test_1:,.0f} -> {upside_test_2:,.0f}",
        "upside_note": f"Ralliye ilk anda guvenme; temiz acceptance icin en az {upside_test_1:,.0f} ustunde kabul lazim.",
        "downside": f"{downside_test_1:,.0f} -> {downside_test_2:,.0f}",
        "downside_note": f"{downside_test_1:,.0f} alti hedge baskisini artirir; asil hizlanma riski {downside_test_2:,.0f}.",
        "squeeze_trigger": squeeze_trigger,
        "breakout_confirm": breakout_confirm,
        "breakdown_trigger": breakdown_trigger,
        "acceleration_trigger": acceleration_trigger,
        "reject_upside": reject_upside,
        "reject_downside": reject_downside,
    }

    return {
        "spot": spot,
        "pin_low": pin_low,
        "pin_center": pin_center,
        "pin_high": pin_high,
        "base_case": base_case,
        "alt_case": alt_case,
        "invalidation": invalidation,
        "dealer": dealer,
        "crowding": crowding,
        "pricing": pricing,
        "blind_spots": blind_spots,
        "simple_summary": simple_summary,
        "positive_nodes": positive_nodes,
        "negative_nodes": negative_nodes,
        "crowded": crowded,
        "call_crowded": call_crowded,
        "put_crowded": put_crowded,
        "next_day": next_day,
        "next_day_change": next_day_change,
        "next_day_share": next_day_share,
        "type_rows": type_rows,
        "front_changes": change_rows[:6],
        "vol_context": vol_context,
        "expiry_market": expiry_market,
        "gex_market": gex_market,
        "total_visible_oi": total_visible_oi,
        "front_total_oi_3d": front_total_oi_3d,
        "total_net_gex": total_net_gex,
        "notional_ratio": notional_ratio,
        "oi_ratio": oi_ratio,
        "atm_iv": atm_iv,
        "front_atm_iv": front_atm_iv,
        "far_atm_iv": far_atm_iv,
        "term_slope": term_slope,
        "front_rr_25d": front_rr_25d,
        "front_bf_25d": front_bf_25d,
        "front_rr_10d": front_rr_10d,
        "front_bf_10d": front_bf_10d,
        "rv_7d": rv_7d,
        "rv_14d": rv_14d,
        "rv_30d": rv_30d,
        "iv_rv_7d_spread": iv_rv_7d_spread,
        "iv_rv_30d_spread": iv_rv_30d_spread,
        "ts_front_maturity": ts_front_maturity,
        "ts_day_change": ts_day_change,
        "ts_week_change": ts_week_change,
        "rr_25d": rr_25d,
        "skew_25d": skew_25d,
        "top_adds": top_adds,
        "top_cuts": top_cuts,
        "strike_adds": strike_adds,
        "strike_cuts": strike_cuts,
        "flow_summary": flow_summary,
        "flow_label": flow_label,
        "flow_score": flow_score,
        "iv_term_rows": iv_term_rows,
        "vol_summary": vol_summary,
        "strategy_rows": top_strategies,
        "strategy_bias": strategy_bias,
        "strategy_summary": strategy_summary,
        "final_read": final_read,
        "morning_note": morning_note,
        "key_takeaways": key_takeaways,
        "takeaways_structured": takeaways_structured,
        "playbook": playbook,
        "acceptance_rules": acceptance_rules,
        "gamma_regime": gamma_regime,
        "inventory_rows": inventory_rows,
        "inventory_net": inventory_net,
        "inventory_label": inventory_label,
    }


def build_archive_row(summary: Dict[str, object]) -> Dict[str, object]:
    today = UTC_NOW().strftime("%Y-%m-%d")
    return {
        "date": today,
        "generated_utc": UTC_NOW().isoformat(),
        "spot": round(summary["spot"], 2),
        "pin_low": int(summary["pin_low"]),
        "pin_center": int(summary["pin_center"]),
        "pin_high": int(summary["pin_high"]),
        "total_net_gex": round(summary["total_net_gex"], 4),
        "next_day_maturity": summary["next_day"]["maturity"],
        "next_day_oi": round(summary["next_day"]["open_interest"], 2),
        "next_day_oi_change_usd": round(summary["next_day_change"], 2),
        "next_day_share": round(summary["next_day_share"], 6),
        "call_put_notional_ratio": round(summary["notional_ratio"], 4),
        "call_put_oi_ratio": round(summary["oi_ratio"], 4),
        "atm_iv": round(summary["atm_iv"], 2),
        "front_atm_iv": round(summary["front_atm_iv"], 2),
        "rv_7d": round(summary["rv_7d"], 2),
        "rv_30d": round(summary["rv_30d"], 2),
        "iv_rv_7d_spread": round(summary["iv_rv_7d_spread"], 2),
        "inventory_net": round(summary["inventory_net"], 3),
        "rr_25d": round(summary["rr_25d"], 2),
        "skew_25d": round(summary["skew_25d"], 2),
        "total_visible_oi": round(summary["total_visible_oi"], 2),
        "front_total_oi_3d": round(summary["front_total_oi_3d"], 2),
        "top_pos_1": int(summary["positive_nodes"][0]["strike"]),
        "top_pos_2": int(summary["positive_nodes"][1]["strike"]),
        "top_neg_1": int(summary["negative_nodes"][0]["strike"]),
        "top_neg_2": int(summary["negative_nodes"][1]["strike"]),
    }


def build_strike_flow_rows(date_str: str, strike_change_rows: List[Dict]) -> List[Dict[str, object]]:
    return [
        {
            "date": date_str,
            "strike": row["strike"],
            "call_change_notional": round(row["call_change_notional"], 2),
            "put_change_notional": round(row["put_change_notional"], 2),
            "net_change_notional": round(row["net_change_notional"], 2),
        }
        for row in strike_change_rows
    ]


def build_instrument_flow_rows(date_str: str, top_instrument_rows: List[Dict]) -> List[Dict[str, object]]:
    return [
        {
            "date": date_str,
            "instrument": row["instrument"],
            "maturity": row["maturity"],
            "strike": row["strike"] or "",
            "type": row["type"],
            "volume": round(row["volume"], 2),
            "notional": round(row["notional"], 2),
            "open_interest": round(row["open_interest"], 2),
            "oi_change_notional": round(row["oi_change_notional"], 2),
        }
        for row in top_instrument_rows
    ]


def make_expiry_chart(rows: List[Dict], out_path: Path) -> None:
    sample = rows[:8]
    labels = [row["maturity"] for row in sample]
    values = [row["open_interest"] / 1_000_000 for row in sample]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#1f4e79")
    plt.title("BTC Opsiyon Acik Pozisyonu - Vade Bazli")
    plt.ylabel("Acik Pozisyon (milyon USD)")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_gex_chart(rows: List[Dict], spot: float, out_path: Path) -> None:
    window = [row for row in rows if abs(row["strike"] - spot) <= 15000]
    labels = [str(row["strike"]) for row in window]
    values = [row["net_gex"] for row in window]
    colors = ["#2e8b57" if value >= 0 else "#b22222" for value in values]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, values, color=colors)
    plt.title("BTC Net GEX - Strike Bazli")
    plt.ylabel("Net GEX")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_strike_oi_chart(rows: List[Dict], spot: float, out_path: Path) -> None:
    window = [row for row in rows if abs(row["strike"] - spot) <= 15000]
    labels = [str(row["strike"]) for row in window]
    call_values = [row["call_notional"] / 1_000_000 for row in window]
    put_values = [row["put_notional"] / 1_000_000 for row in window]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, call_values, color="#1f4e79", label="Call notional")
    plt.bar(labels, put_values, bottom=call_values, color="#c06c2b", label="Put notional")
    plt.title("BTC Strike Bazli Acik Pozisyon - Spot Cevresi")
    plt.ylabel("Acik Pozisyon (milyon USD)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_archive_chart(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        return
    window = rows[-14:]
    dates = [row["date"] for row in window]
    spot = [to_float(row["spot"]) for row in window]
    next_day_oi = [to_float(row["next_day_oi"]) / 1_000_000 for row in window]
    plt.figure(figsize=(10, 5))
    plt.plot(dates, spot, color="#1f4e79", marker="o", label="Spot")
    plt.plot(dates, next_day_oi, color="#bb7d2a", marker="o", label="Ertesi gun OI (mn USD)")
    plt.title("Gunluk Trend Ozeti")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_flow_change_chart(
    strike_adds: List[Dict], strike_cuts: List[Dict], out_path: Path
) -> None:
    rows = strike_adds[:5] + strike_cuts[:5]
    labels = [str(row["strike"]) for row in rows]
    values = [row["net_change_notional"] / 1_000_000 for row in rows]
    colors = ["#2e8b57" if value >= 0 else "#9f2b2b" for value in values]
    plt.figure(figsize=(11, 5))
    plt.barh(labels, values, color=colors)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("OI Akis Degisimi - En Guclu Strike Eklemeleri / Bosalmalari")
    plt.xlabel("Net OI degisimi (milyon USD)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_flow_heatmap(
    strike_change_rows: List[Dict], spot: float, out_path: Path
) -> None:
    window = [row for row in strike_change_rows if abs(row["strike"] - spot) <= 15000]
    labels = [row["strike"] for row in window]
    matrix = [
        [row["call_change_notional"] / 1_000_000 for row in window],
        [row["put_change_notional"] / 1_000_000 for row in window],
    ]
    plt.figure(figsize=(12, 3.8))
    plt.imshow(matrix, aspect="auto", cmap="RdYlGn")
    plt.yticks([0, 1], ["Call akisi", "Put akisi"])
    plt.xticks(range(len(labels)), [str(label) for label in labels], rotation=45, ha="right")
    plt.colorbar(label="OI degisimi (milyon USD)")
    plt.title("BTC OI Akis Heatmap - Spot Cevresi")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_iv_term_chart(atm_iv_ts: Dict[str, List[Dict[str, float]]], out_path: Path) -> None:
    labels = ["Today", "Yesterday", "1 Week Ago"]
    plt.figure(figsize=(11, 5))
    drawn = False
    for label in labels:
        rows = atm_iv_ts.get(label, [])
        if not rows:
            continue
        plt.plot(
            [row["maturity"] for row in rows],
            [row["iv"] for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
        drawn = True
    if not drawn:
        plt.close()
        return
    plt.title("BTC ATM IV Term Structure")
    plt.ylabel("ATM IV")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_iv_rv_chart(summary: Dict[str, object], out_path: Path) -> None:
    labels = ["Front ATM IV", "RV 7g", "RV 14g", "RV 30g"]
    values = [
        summary["front_atm_iv"],
        summary["rv_7d"],
        summary["rv_14d"],
        summary["rv_30d"],
    ]
    colors = ["#0f4c5c", "#bb7d2a", "#7e9f35", "#5d6d7e"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.title("IV ve Realized Vol Karsilastirmasi")
    plt.ylabel("Vol (%)")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_strategy_chart(rows: List[Dict[str, float]], out_path: Path) -> None:
    sample = rows[:8]
    labels = [row["strategy"].replace("_", " ")[:20] for row in sample]
    values = [row["total_notional"] / 1_000_000 for row in sample]
    colors = ["#2e8b57" if row["net_premium_usd"] < 0 else "#b5651d" for row in sample]
    plt.figure(figsize=(11, 5))
    plt.barh(labels, values, color=colors)
    plt.title("Top Options Strategies - Notional")
    plt.xlabel("Toplam notional (milyon USD)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_mm_inventory_chart(summary: Dict[str, object], out_path: Path) -> None:
    rows = summary["inventory_rows"]
    labels = [row["label"] for row in rows]
    values = [row["score"] for row in rows]
    colors = ["#2e8b57" if value >= 0 else "#9f2b2b" for value in values]
    plt.figure(figsize=(9, 5))
    plt.barh(labels, values, color=colors)
    plt.axvline(0, color="#1b2228", linewidth=1.2)
    plt.axvline(2, color="#c8a646", linewidth=1, linestyle="--")
    plt.axvline(-2, color="#c8a646", linewidth=1, linestyle="--")
    plt.xlim(-5, 5)
    plt.title("Dealer Inventory Proxy")
    plt.xlabel("-5 unstable / short gamma    |    +5 stabilizing / long gamma")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_inventory_trend_chart(rows: List[Dict[str, str]], out_path: Path) -> None:
    if not rows:
        return
    window = rows[-14:]
    dates = [row["date"] for row in window]
    values = [to_float(row.get("inventory_net", "0")) for row in window]
    plt.figure(figsize=(10, 4.8))
    plt.plot(dates, values, color="#0f4c5c", marker="o", linewidth=2)
    plt.axhline(0, color="#1b2228", linewidth=1)
    plt.axhline(2, color="#c8a646", linewidth=1, linestyle="--")
    plt.axhline(-2, color="#c8a646", linewidth=1, linestyle="--")
    plt.ylim(-5, 5)
    plt.title("Dealer Inventory Proxy Trend")
    plt.ylabel("Proxy skor")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_change_section(summary: Dict[str, object], archive_rows: List[Dict[str, str]]) -> Dict[str, object]:
    prev = archive_rows[-2] if len(archive_rows) >= 2 else None
    notes: List[str] = []
    headline = []
    if prev:
        spot_change = summary["spot"] - to_float(prev["spot"])
        iv_change = summary["atm_iv"] - to_float(prev["atm_iv"])
        prev_front_iv = prev.get("front_atm_iv", "")
        front_iv_change = (
            summary["front_atm_iv"] - to_float(prev_front_iv)
            if has_numeric(prev_front_iv)
            else None
        )
        prev_iv_rv = prev.get("iv_rv_7d_spread", "")
        iv_rv_change = (
            summary["iv_rv_7d_spread"] - to_float(prev_iv_rv)
            if has_numeric(prev_iv_rv)
            else None
        )
        next_day_oi_change = summary["next_day"]["open_interest"] - to_float(prev["next_day_oi"])
        gex_change = summary["total_net_gex"] - to_float(prev["total_net_gex"])
        headline.append(f"Spot {spot_change:+,.0f}")
        if front_iv_change is not None:
            headline.append(f"Front IV {front_iv_change:+.2f}")
        if iv_rv_change is not None:
            headline.append(f"IV-RV {iv_rv_change:+.2f}")
        headline.append(f"Net gamma {gex_change:+.3f}")
        headline.append(f"Ertesi gun OI {next_day_oi_change / 1_000_000:+.1f}mn USD")
        notes.append(f"Spot dunden bugune {spot_change:+,.0f} degisti.")
        iv_note = f"ATM spot IV degisimi {iv_change:+.2f} puan;"
        if front_iv_change is not None:
            iv_note += f" front ATM IV degisimi {front_iv_change:+.2f};"
        iv_note += f" net gamma degisimi {gex_change:+.3f}."
        notes.append(iv_note)
        if iv_rv_change is not None:
            notes.append(f"IV-RV spreadi dunden bugune {iv_rv_change:+.2f} puan degisti.")
        notes.append(f"Ertesi gun vadesindeki acik pozisyon {next_day_oi_change / 1_000_000:+.1f} milyon USD degisti.")
        if str(prev.get("top_pos_1")) != str(summary["positive_nodes"][0]["strike"]):
            notes.append(
                f"Ana yukari gamma miknatisi {prev.get('top_pos_1')} seviyesinden "
                f"{summary['positive_nodes'][0]['strike']} seviyesine kaydi."
            )
    else:
        headline.append("Yerel arsiv ilk gun")
        notes.append("Bugun yerel gunluk arsivin ilk baz gunu olarak kaydedildi.")
        notes.append("Yarin itibariyla dunden bugune degisim bolumu anlamli karsilastirma vermeye baslayacak.")
        notes.append("Bugunden itibaren rapor her sabah ayni metrikleri kaydedecek; trend kalitesi her gun artacak.")

    weekly = None
    if len(archive_rows) >= 5:
        week_ref = archive_rows[max(0, len(archive_rows) - 5)]
        weekly = {
            "spot": summary["spot"] - to_float(week_ref["spot"]),
            "iv": summary["atm_iv"] - to_float(week_ref["atm_iv"]),
            "next_day_oi": summary["next_day"]["open_interest"] - to_float(week_ref["next_day_oi"]),
        }
    return {"notes": notes, "weekly": weekly, "headline": headline}


def build_flow_horizon_notes(
    date_str: str, strike_change_rows: List[Dict], top_instrument_rows: List[Dict]
) -> Dict[str, List[str]]:
    strike_history = load_csv_rows(STRIKE_FLOW_CSV)
    inst_history = load_csv_rows(INSTRUMENT_FLOW_CSV)
    notes = {"daily": [], "weekly": [], "monthly": []}

    today_top = sorted(strike_change_rows, key=lambda row: abs(row["net_change_notional"]), reverse=True)[:2]
    if today_top:
        notes["daily"].append(
            f"Bugun en baskin strike akis degisimi {today_top[0]['strike']:,.0f} ve {today_top[1]['strike']:,.0f} cevresinde goruldu."
        )

    past_days = sorted({row["date"] for row in strike_history if row["date"] != date_str})
    if len(past_days) >= 5:
        ref_dates = past_days[-5:]
        weekly_rows = [row for row in strike_history if row["date"] in ref_dates]
        agg = defaultdict(float)
        for row in weekly_rows:
            agg[int(row["strike"])] += to_float(row["net_change_notional"])
        leaders = sorted(agg.items(), key=lambda item: abs(item[1]), reverse=True)[:2]
        if leaders:
            notes["weekly"].append(
                f"Son 5 gunluk yerel kayitta akis en cok {leaders[0][0]:,.0f} ve {leaders[1][0]:,.0f} strike'larinda birikmis."
            )
    else:
        notes["weekly"].append("5 gunluk flow karsilastirmasi icin yeterli yerel veri henuz birikmedi.")

    if len(past_days) >= 20:
        ref_dates = past_days[-20:]
        monthly_rows = [row for row in inst_history if row["date"] in ref_dates]
        call_sum = sum(to_float(row["oi_change_notional"]) for row in monthly_rows if row["type"] == "C")
        put_sum = sum(to_float(row["oi_change_notional"]) for row in monthly_rows if row["type"] == "P")
        if abs(call_sum) >= abs(put_sum):
            notes["monthly"].append("20 gunluk birikimde call tarafi daha baskin.")
        else:
            notes["monthly"].append("20 gunluk birikimde put tarafi daha baskin.")
    else:
        notes["monthly"].append("20 gunluk flow karsilastirmasi icin yeterli yerel veri henuz birikmedi.")
    return notes


def render_list(items: List[str]) -> str:
    return "".join(f"<li>{item}</li>" for item in items)


def render_paragraphs(items: List[str]) -> str:
    return "".join(f"<p>{item}</p>" for item in items)


def render_delta_badges(items: List[str]) -> str:
    badges = []
    for item in items:
        lowered = item.lower()
        klass = "delta-neutral"
        if "+" in item or "artti" in lowered or "yukari" in lowered:
            klass = "delta-up"
        elif "-" in item or "azaldi" in lowered or "asagi" in lowered:
            klass = "delta-down"
        badges.append(f"<span class='delta-badge {klass}'>{item}</span>")
    return "".join(badges)


def tr_to_en_text(text: str) -> str:
    replacements = [
        ("Base case: 68,000-73,000 icinde pin/range. Ilk beklenen davranis mean reversion and 72,000.", "Base case: 68,000-73,000 range with pin behavior. The first expectation is mean reversion back toward 72,000."),
        ("Spot 75,000 ustunde kalir and call akisi desteklerse yukari test aktif.", "Spot holding above 75,000 with confirming call flow activates the upside test."),
        ("Reject above 73,000: fiyat bu bolgeyi sadece igneleyip geri donerse ralliye guvenme; pin/range mantigi calisiyor demektir.", "Reject above 73,000: if price only wicks into the zone and reverses, do not trust the rally; pin/range logic remains in force."),
        ("Reject below 68,000: fiyat altina sarkip hizla geri aliniyorsa asagi kirilim degil stop-hunt / range return okunmali.", "Reject below 68,000: if price breaks below and is reclaimed quickly, read it as a stop-hunt / return-to-range move."),
        ("Bu chart gercek MM inventory degil; gamma, flow, skew, IV-RV and strategy tape uzerinden uretilen bir proxy skorudur.", "This is not true MM inventory; it is a proxy score derived from gamma, flow, skew, IV-RV, and the strategy tape."),
        ("Ertesi gun OI +", "Next-day OI +"),
        ("Ertesi gun OI -", "Next-day OI -"),
        ("68,000A sustained break below ", "68,000. A sustained break below "),
        ("ATM spot IV is is ", "ATM spot IV is "),
        ("Front ATM IV is is ", "Front ATM IV is "),
        ("On vade ATM IV (07MAR26) 53.81; while back-end ATM IV is (20MAR26) 55.65. Vol egirisi gorece duz, yani yakin vade panik fiyati baskin degil.", "Front-end ATM IV (07MAR26) is 53.81, while back-end ATM IV (20MAR26) is 55.65. The vol curve is relatively flat, which means front-end panic pricing is not dominant."),
        ("25d RR -8.55; downside putlar call'lara gore anlamli primli. Bu savunmaci tail hedge talebini gosteriyor.", "25d RR is -8.55; downside puts remain meaningfully richer than calls. That points to defensive tail-hedge demand."),
        ("10d RR -14.92 and 25d BF 1.26; bu kombinasyon kisa vadede tail priminin ne kadar yuksek oldugunu gosteriyor.", "10d RR is -14.92 and 25d BF is 1.26; that combination shows how elevated short-dated tail premium remains."),
        ("Bu grafik son 24 saatte hangi strike'larda yeni risk eklendigini and hangi strike'larda riskin bosaldigini en hizli sekilde gosterir.", "This chart shows where new risk was added and where risk was reduced across strikes over the last 24 hours."),
        ("Heatmap spot cevresinde call and put akisinin tam olarak hangi strike'larda yogunlastigini gosterir.", "The heatmap shows exactly which strikes around spot absorbed the heaviest call and put flow."),
        ("7 gunluk realized vol ", "7-day realized vol "),
        ("IV and RV Karsilastirmasi", "IV and RV Comparison"),
        ("Gunluk ATM IV degisimi", "Daily ATM IV change"),
        ("Haftalik ATM IV degisimi", "Weekly ATM IV change"),
        ("milyon USD", "million USD"),
        ("Base case: 68,000-74,000 icinde pin/range. Ilk beklenen davranis mean reversion and 72,000.", "Base case: 68,000-74,000 range with pin behavior. The first expectation is mean reversion back toward 72,000."),
        ("Bull case: 75,000 acceptance above opens a test toward 80,000 seviyesine uzayabilir. Temiz trend icin yeni call eklenmesi and flow tonunun iyilesmesi lazim.", "Bull case: acceptance above 75,000 opens a test toward 80,000. For a clean trend day, the move still needs fresh call adding and an improvement in flow tone."),
        ("Spot 75,000 ustunde kalir and call akisi desteklerse yukari test aktif.", "Spot holding above 75,000 with confirming call flow activates the upside test."),
        ("Accept above 75,000: spot bu seviye ustunde kalir and yeni call akisiyla desteklenirse yukari test aktif olur.", "Accept above 75,000: if spot holds above the level and fresh call flow confirms, the upside test becomes active."),
        ("Reject above 74,000: fiyat bu bolgeyi sadece igneleyip geri donerse ralliye guvenme; pin/range mantigi calisiyor demektir.", "Reject above 74,000: if price only wicks into the zone and reverses, do not trust the rally; pin/range logic is still active."),
        ("Accept below 68,000: spot bu seviye altinda kalirsa downside hedge akisi kuvvetlenir and 67,000 ikinci hedef olur.", "Accept below 68,000: if spot holds below the level, downside hedge flow should strengthen and 67,000 becomes the next target."),
        ("Bu chart gercek MM inventory degil; gamma, flow, skew, IV-RV and strategy tape uzerinden uretilen bir proxy skorudur.", "This is not true MM inventory; it is a proxy score derived from gamma, flow, skew, IV-RV, and the strategy tape."),
        ("Yerel arsiv buyudukce bu grafik MM/dealer tonunun zaman icinde daha net izlenmesini saglar.", "As the local archive grows, this chart should give a cleaner view of MM/dealer tone through time."),
        ("On the upside, en kalabalik call cepleri 75,000 and 80,000; and those levels can act as resistance as much as targets on first test.", "On the upside, the most crowded call pockets sit at 75,000 and 80,000, and those levels can act as resistance as much as targets on first test."),
        ("On the downside, ise 67,000 and 60,000 are regime-shift levels.", "On the downside, 67,000 and 60,000 are regime-shift levels."),
        ("Front ATM IV is is ", "Front ATM IV is "),
        ("Skew and smile verisi halen savunmaci bir alt ton tasiyor.", "Skew and smile data still carry a defensive undertone."),
        ("Akis and strategy tape bu resmi tamamliyor.", "Flow and the strategy tape complete the picture."),
        ("Flow score -4 and baskin ton defensive / downside hedge akisi.", "Flow score is -4 and the dominant tone is defensive / downside hedge flow."),
        ("Strategy tarafinda one cikan grup constructive.", "The leading strategy bucket is constructive."),
        ("Eger yapici call/short put yapilari savunmaci put taleplerini dengeliyorsa piyasa yukariyi tamamen reddetmiyor; ancak short call, capped yapilar veya agir hedge akisi artiyorsa yukari hareketin kalitesi weak kaliyor.", "If constructive call/short-put structures offset defensive put demand, the market is not rejecting upside outright; but if short-call, capped structures, or heavy hedge flow dominate, the quality of upside weakens."),
        (" and then 80,000 test edilir, fakat bunun trend gunune donusmesi icin bu seviyelerde kabul and yeni call eklenmesi gorulmeli.", " and then 80,000. But for this to become a trend day, those levels must hold with acceptance and fresh call adding."),
        ("On the downside, 68,000 alti sadece teknik weaklik degil, opsiyon rejim degisimi anlamina gelir; orada hedge akisinin hizi artar and spot beklenenden daha hizli asagi tasinabilir.", "On the downside, a break below 68,000 is not just technical weakness; it is an options-regime shift, where hedge flow can accelerate and push spot lower faster than expected."),
        ("Dealerlar spotu sabitlemeye calisiyor, vol market riski tamamen dusurmuyor, strateji tape ise yon yerine secici and korunmali risk alindigini anlatiyor.", "Dealers are trying to stabilize spot, the vol market is not fully dropping risk premia, and the strategy tape says risk is being taken selectively and with protection rather than blindly for direction."),
        ("Spot dunden bugune ", "Spot changed "),
        (" degisti.", " vs yesterday."),
        ("ATM spot IV degisimi ", "ATM spot IV changed "),
        (" puan; net gamma degisimi ", " points; net gamma changed "),
        ("Ertesi gun vadesindeki acik pozisyon ", "Open interest in the next-day expiry changed by "),
        (" milyon USD degisti.", " million USD."),
        ("Bu grafik yerel arsivden beslenir. Ilk gunlerde kisitli olur; her sabah veri geldikce yorum kalitesi artar.", "This chart is driven by the local archive. It will be limited in the first days, then improve as new morning snapshots accumulate."),
        ("Dealer tarafinda yonlu devam hareketini bastiracak kadar pozitif gamma yok.", "There is not enough positive gamma on the dealer side to suppress directional continuation."),
        ("En yogun pozitif GEX dugumleri ", "The heaviest positive GEX nodes sit at "),
        (" seviyelerinde; bu alanlar ani squeeze'ten cok yukari yonlu miknatis etkisi yaratir.", "; these zones are more likely to act as upside magnets than as pure squeeze triggers."),
        ("Asagi yonlu en hassas hedge cepleri ", "The most sensitive downside hedge pockets remain near "),
        (" seviyelerinde kaliyor; bu bolgenin kirilmasi hedge talebini hizlandirabilir.", "; a break there could accelerate hedge demand."),
        ("Call notional hala put notional'in ", "Call notional is still "),
        (" kati, open interest adedi de call lehine ", "x put notional, and open-interest count is "),
        (" kat. Bu panic hedge degil, yapici bir konumlanma.", "x in favor of calls. This is not panic hedging; it is constructive positioning."),
        ("Spota yakin en buyuk call yigilmalari ", "The largest call clusters near spot sit at "),
        ("; en buyuk put yigilmalari ise ", "; the largest put clusters sit at "),
        ("Ertesi gun vadesi olan ", "The next-day expiry "),
        (" tum gorunen expiry OI'nin yalnizca ", " is only "),
        ("'i kadar; bu nedenle spot dogrudan kalabalik strike'lara gitmedikce tek basina tum bandi suruklemesi zor.", " of total visible expiry OI, so on its own it is unlikely to drag the full band unless spot trades directly into crowded strikes."),
        ("Piyasa crash paniginden cok duzenli bir asagi yon korumasi fiyatliyor: ", "The market is pricing orderly downside protection rather than crash panic: "),
        (" vol baglaminda ", " in the "),
        (" 25-delta skew ise ", " context, while 25-delta skew is "),
        ("ATM spot IV ", "ATM spot IV is "),
        (" civarinda; riskin ciddiye alindigini gosteriyor ama tek basina duzensizlik sinyali vermiyor.", "; risk is being taken seriously, but this alone does not signal disorder."),
        ("Expiry egirisinin on tarafina hala taze risk ekleniyor: ertesi gun vadesine son 24 saatte yaklasik ", "Fresh risk is still being added to the front of the expiry curve: roughly "),
        (" eklendi.", " was added to the next-day expiry over the last 24 hours."),
        ("Spot 68,000 uzerinde kaldigi surece bant cok sert bir asagi kirilim icin hazir gorunmuyor; piyasanin ustunde and etrafinda hala fazla miktarda pozitif gamma birikimi var.", "As long as spot holds above 68,000, the band does not look primed for a violent downside break; there is still a large amount of positive gamma above and around the market."),
        ("Ayni anda piyasa temiz bir yukari kirilimi de agresif fiyatlamiyor. ", "At the same time, the market is not aggressively pricing a clean upside breakout. "),
        (" civarindaki agir call pozisyonlari, spot bu seviyeleri hacimle asmadan chase davranisini sinirlayabilir.", " heavy call positioning in that area can limit chase behavior unless spot clears those levels on volume."),
        ("Daha buyuk gizli risk, ", "The larger hidden risk is a regime shift below "),
        (" altinda rejim degisimi yasanmasi; bu bolgede dealer hedging hareketi emmek yerine buyutmeye baslayabilir.", ", where dealer hedging may start amplifying moves rather than absorbing them."),
        ("On vade ATM IV (07MAR26) 49.49; while back-end ATM IV is (20MAR26) 54.97. Vol egirisi yukari egimli; piyasa yakin riski degil daha kalici belirsizligi fiyatliyor.", "Front-end ATM IV (07MAR26) is 49.49, while back-end ATM IV (20MAR26) is 54.97. The vol curve slopes upward, which suggests the market is pricing more persistent uncertainty rather than front-end panic."),
        ("Front ATM IV, 7 gunluk realized volun ", "Front ATM IV is "),
        (" puan altinda. Spotun son hareketi opsiyon primlerinden daha agresif; underpriced hareket riski var.", " points below 7-day realized vol. Spot has moved more aggressively than options premia imply, so there is underpriced-move risk."),
        ("25d RR -7.11; downside putlar call'lara gore anlamli primli. Bu savunmaci tail hedge talebini gosteriyor.", "25d RR is -7.11; downside puts remain meaningfully richer than calls. That points to defensive tail-hedge demand."),
        ("10d RR -14.00 and 25d BF 0.94; bu kombinasyon kisa vadede tail priminin ne kadar yuksek oldugunu gosteriyor.", "10d RR is -14.00 and 25d BF is 0.94; that combination shows how elevated short-dated tail premium remains."),
        ("Laevitas historical RV endpoint'i bu key ile kapali oldugu icin realized vol public BTC gunluk kapanis serisinden hesaplanir.", "The Laevitas historical RV endpoint is not available under this key, so realized vol is computed from public BTC daily close data."),
        ("Defensive / downside hedge akisi", "Defensive / downside hedge flow"),
        ("Son 24 saatte en strong yeni risk ", "Over the last 24 hours, the strongest new risk came through "),
        (" uzerinden geldi; bu, put korumasi tarafinda aktif yeni konumlanma oldugunu gosteriyor.", ", which points to active new positioning in downside put protection."),
        ("Ayni anda en belirgin bosalma ", "At the same time, the clearest unwind came through "),
        (" tarafinda; bu da eski call riskinin bir kisminin temizlendigini gosteriyor.", ", which suggests part of the old call risk was cleared."),
        ("Strike bazli net ekleme en cok ", "The largest strike-level net additions were concentrated around "),
        (" civarinda yogunlasmis; bu bolgeler kisa vadede yeni ilgi merkezi.", ", making those zones the new center of short-dated attention."),
        ("En sert net bosalma ", "The sharpest net reduction came around "),
        (" cevresinde; bu da eski pozisyonun oradan cekildigine isaret ediyor.", ", suggesting old positioning was pulled from that zone."),
        ("Bu grafik son 24 saatte hangi strike'larda yeni risk eklendigini and hangi strike'larda riskin bosaldigini en hizli sekilde gosterir.", "This chart shows, as quickly as possible, where new risk was added and where risk was reduced over the last 24 hours."),
        ("Heatmap spot cevresinde call and put akisinin tam olarak hangi strike'larda yogunlastigini gosterir.", "The heatmap shows exactly which strikes around spot are absorbing the heaviest call and put flow."),
        ("Bugun en baskin strike akis degisimi ", "Today's dominant strike-flow change appeared around "),
        (" cevresinde goruldu.", "."),
        ("5 gunluk flow karsilastirmasi icin yeterli yerel veri henuz birikmedi.", "There is not yet enough local data to compare 5-day flow."),
        ("20 gunluk flow karsilastirmasi icin yeterli yerel veri henuz birikmedi.", "There is not yet enough local data to compare 20-day flow."),
        ("En buyuk strategy baskisi ", "The strongest strategy pressure came from "),
        (" tarafinda; yaklasik ", ", with roughly "),
        (" notional and ", " of notional and "),
        (" OI degisimi ile tape'i tasiyor.", " of OI change driving the tape."),
        ("Strategy tape genel olarak yapici; short put, long call veya bull spread benzeri yapilar savunmadan daha agir basiyor.", "The strategy tape is broadly constructive; short puts, long calls, and bull-spread style structures are outweighing defensive trades."),
        ("Birlesik okuma, pin mekanizmasinin weakladigini and piyasanin daha yonlu bir asagi hareket riski tasidigini gosteriyor. Bu durumda opsiyon piyasasi spota istikrar degil ivme ekleyebilir.", "The integrated read suggests the pin mechanism is weakening and the market carries more directional downside risk. In that regime, the options market is more likely to add momentum than stability to spot."),
        ("Vol market tarafinda front ATM IV ", "On the vol side, front ATM IV is "),
        (" and IV-RV spread ", " and IV-RV spread is "),
        ("Bu, opsiyon primlerinin son spot hareketine gore dengeye yakin oldugunu gosteriyor.", "That suggests options premia are not far from recent spot movement."),
        ("Strategy tape tarafinda baskin ton constructive. Bu nedenle bugunun okumasini sadece strike yigilmasi degil, aktif trade edilen stratejilerin ne anlattigi da destekliyor.", "The dominant tone in the strategy tape is constructive. That means today's read is supported not only by strike crowding but also by what the actively traded strategy mix is saying."),
        ("Net gamma kirilgan. Basit bir pin gununden daha yonlu bir realized volatilite beklenmeli.", "Net gamma is fragile. Expect a more directional realized-volatility session rather than a simple pin day."),
        ("Pozitif dealer gamma spotu ", "Positive dealer gamma should keep spot inside "),
        (" bandinda ortalamaya donen bir yapida tutmali; en guclu pin seviyesi ", " in a mean-reverting structure; the strongest pin sits near "),
        (" civari.", "."),
        ("Pozitif gamma destekleyici kalmaya devam ediyor, ancak spot ana pin bandinin disinda; fiyat yeni bir kabul alani olusturmazsa ", "Positive gamma remains supportive, but spot is trading outside the main pin band; unless price establishes a new area of acceptance, a pullback toward "),
        (" seviyesine dogru cekilme beklenir.", " is likely."),
        ("Fiyat ", "If price accepts above "),
        (" uzerinde kabul gorurse, ", ", the heavy call crowding around "),
        (" ve ", " and "),
        (" civarindaki call yigilmasi hareketi momentumlu bir yukari uzamaya tasiyabilir.", " can turn the move into a more momentum-driven upside extension."),
        (" altinda kalici bir kirilma, ", "A sustained break below "),
        (" civarindaki negatif-gamma ceplerini aciga cikarir.", " would expose negative-gamma pockets in that zone."),
        ("Bugun icin ana tablo: fiyatin ", "Main setup for today: price is more likely to remain controlled inside "),
        (" bandinda daha kontrollu kalmasi bekleniyor.", "."),
        ("Yukari tarafta ", "On the upside, "),
        (" seviyeleri hem kalabalik hem de dikkat cekici call bolgeleri.", " stand out as both crowded and important call-heavy zones."),
        ("Asagi tarafta ", "On the downside, "),
        (" seviyeleri kirilma halinde davranisi degistirebilecek bolgeler.", " are the levels most likely to change the behavior of the session if broken."),
        ("Kisa vade risk eklenmesi devam ediyor; bu da piyasanin yarina dair ilgisinin canli oldugunu gosteriyor.", "Short-dated risk is still being added, which shows live interest into tomorrow's session."),
        ("Base case: ", "Base case: "),
        ("Bull case: ", "Bull case: "),
        ("Bear case: ", "Bear case: "),
        (" icinde pin/range. Ilk beklenen davranis mean reversion ve ", " range with pin behavior. The first expectation is mean reversion back toward "),
        (" merkezine geri cekilme.", "."),
        (" ustunde kabul gorulurse yukari test ", " acceptance above opens a test toward "),
        (" seviyesine uzayabilir. Temiz trend icin yeni call eklenmesi ve flow tonunun iyilesmesi lazim.", ". For a clean trend day, the move still needs fresh call adding and an improvement in flow tone."),
        (" alti kabul downside hedge akislarini hizlandirir; ", " acceptance below should accelerate downside hedge flows; "),
        (" civari ikinci hizlanma bolgesi.", " is the next acceleration zone."),
        ("Gamma kirilgan/negatif. MM hedge akisi hareketi bastirmaktan cok buyutebilir.", "Gamma is fragile/negative. MM hedging is more likely to amplify moves than suppress them."),
        ("Balance: ", "Balance: "),
        ("zayif", "weak"),
        ("guclu", "strong"),
        ("Ralliye ilk anda guvenme; temiz acceptance icin en az ", "Do not trust the first rally impulse; clean acceptance requires at least "),
        (" ustunde kabul lazim.", " to hold above."),
        (" alti hedge baskisini artirir; asil hizlanma riski ", " below increases hedge pressure; the real acceleration risk sits near "),
        (" alti hizla geri aliniyorsa stop-hunt / range return oku.", " loses and is quickly reclaimed, read it as a stop-hunt / return-to-range move."),
        ("Spot ", "Spot "),
        (" ustunde kalir ve call akisi desteklerse yukari test aktif.", " holds above and call flow confirms, the upside test is active."),
        (" civari red gelirse ralliye guvenme; range mantigi suruyor.", " rejects, do not trust the rally; range logic is still in control."),
        (" alti kabul downside hedge akislarini buyutur.", " below should increase downside hedge flow."),
        ("Bugune girerken opsiyon piyasasi tek yonlu bir breakout degil, once kontrol sonra secici yon genislemesi fiyatliyor.", "Coming into today, the options market is not pricing a one-way breakout. It is pricing control first, then selective directional expansion."),
        ("En kritik nokta dealer gamma rejiminin hala pozitif kalmasi; bu da spot ", "The key point is that dealer gamma remains positive, which keeps spot inside "),
        (" bandinin icinde kaldigi surece sert intraday genislemelerin hedge akisiyla bastirilma olasiligini yuksek tutuyor.", " and raises the odds that sharp intraday expansions are dampened by hedging flow as long as spot remains in that band."),
        ("Baska bir deyisle masa acilisinda ilk baz senaryo trend gunu degil, pin/range gunu.", "In other words, the opening base case is not a trend day but a pin/range day."),
        ("Bu range okumasinin merkezi ", "The center of that range read sits at "),
        ("Cunku hem gamma miknatislari hem de strike bazli acik pozisyon yogunlugu spotu bu eksene geri cekme egiliminde.", "because both gamma magnets and strike-level open interest tend to pull spot back toward that axis."),
        (" bunlar ilk asamada hedef kadar direnc de yaratabilir.", " and those levels can act as resistance as much as targets on first test."),
        (" seviyeleri rejim degisimi alanlari. Spot bu bolgelerin altina kabul verirse opsiyon piyasasi hareketi emen degil, buyuten tarafa gecebilir.", " are regime-shift levels. If spot accepts below them, the options market can flip from absorbing the move to amplifying it."),
        ("Vol tarafinda tablo daha incelikli.", "The vol side is more nuanced."),
        ("Front ATM IV ", "Front ATM IV is "),
        (" uzak vade ATM IV ", " while back-end ATM IV is "),
        (" ve term slope ", " and the term slope is "),
        ("Bu egri riski daha uzun zamana yaydigini soyluyor.", "That curve shape says risk is being distributed further out the curve."),
        ("Bu egri yakinda panik olmadigini ama riskin canli kaldigini soyluyor.", "That curve shape says there is no near-dated panic, but risk remains alive."),
        ("Bu egri yakina risk primi yuklendiyini soyluyor.", "That curve shape says the front-end is carrying the main risk premium."),
        ("Daha onemlisi, 7 gunluk realized vol ", "More importantly, 7-day realized vol is "),
        (" iken front IV ile fark ", " while the front-IV gap is "),
        ("Yani opsiyon primi spotun gerisinde kaliyor; bu ayrim gun icinde vol satmak mi vol kovalamak mi sorusunun temel cevabi.", "That means options premium is lagging spot realized movement; this gap matters for deciding whether to fade vol or chase it intraday."),
        ("Yani opsiyon primi spotla uyumlu; bu ayrim gun icinde vol satmak mi vol kovalamak mi sorusunun temel cevabi.", "That means options premium is broadly aligned with spot; this gap matters for deciding whether to fade vol or chase it intraday."),
        ("Yani opsiyon primi spotun son hareketine gore pahali; bu ayrim gun icinde vol satmak mi vol kovalamak mi sorusunun temel cevabi.", "That means options premium is rich versus recent spot movement; this gap matters for deciding whether to fade vol or chase it intraday."),
        ("Skew ve smile verisi halen savunmaci bir alt ton tasiyor.", "Skew and smile data still carry a defensive undertone."),
        ("Bu kombinasyon piyasada tamamen rahat bir upside chase olmadigini, asagi kuyruk riskinin hala fiyatlandigini gosteriyor.", "That combination says the market is not in a carefree upside chase; downside tail risk is still being priced."),
        ("Ancak bu tek basina bearish bir tape demek degil; daha cok, yukari tasinmak istense bile bunun korumasiz yapilmadigini soyluyor.", "That does not automatically create a bearish tape; it simply says any upside risk-taking is still being done with protection."),
        ("Akis ve strategy tape bu resmi tamamliyor.", "Flow and the strategy tape complete the picture."),
        ("Bu, masaya su mesaji veriyor: oyuncular yalnizca eski pozisyonu tasimiyor, aktif olarak yeni risk insa ediyor.", "The desk message is that participants are not merely carrying old positions; they are actively building new risk."),
        ("Eger yapici call/short put yapilari savunmaci put taleplerini dengeliyorsa piyasa yukariyi tamamen reddetmiyor; ancak short call, capped yapilar veya agir hedge akisi artiyorsa yukari hareketin kalitesi zayif kaliyor.", "If constructive call/short-put structures balance defensive put demand, the market is not rejecting upside outright; but if short-call, capped structures, or heavy hedge flow dominate, the quality of upside weakens."),
        ("Sonucta bugun icin operasyonel okuma su: ilk senaryo ", "Operationally, today's first scenario is "),
        (" icinde kontrollu fiyatlama, merkezde ", " with controlled pricing and a center around "),
        ("Yukari uzama olacaksa once ", "If there is upside extension, the market should first test "),
        (" sonra ", " and then "),
        (" test edilir, fakat bunun trend gunune donusmesi icin bu seviyelerde kabul ve yeni call eklenmesi gorulmeli.", ". But for this to become a trend day, those levels must hold with acceptance and fresh call adding."),
        ("Asagi senaryoda ise ", "On the downside, "),
        (" alti sadece teknik zayiflik degil, opsiyon rejim degisimi anlamina gelir; orada hedge akisinin hizi artar ve spot beklenenden daha hizli asagi tasinabilir.", " is not just technical weakness; it is an options-regime shift, where hedge flow can accelerate and push spot lower faster than expected."),
        ("Kisacasi bu raporun bugun icin verdigi masa notu su: piyasa su an kaotik bir kopus degil, kontrol edilen bir fiyat kesfi evresinde.", "In short, the desk note for today is this: the market is not in a chaotic break, but in a controlled price-discovery phase."),
        ("Dealerlar spotu sabitlemeye calisiyor, vol market riski tamamen dusurmuyor, strateji tape ise yon yerine secici ve korunmali risk alindigini anlatiyor.", "Dealers are trying to stabilize spot, the vol market is not fully dropping risk premia, and the strategy tape says risk is being taken selectively and with protection rather than blindly for direction."),
        ("Bu nedenle sabah ilk okuma range/pin; gun icindeki asil soru ise bu denge kirildiginda opsiyon piyasasinin hangi yone ivme ekleyecegi.", "That is why the first morning read is range/pin; the real intraday question is which direction the options market will add momentum to once that balance breaks."),
        ("Sabah Opsiyon Notu", "Morning Options Note"),
        ("Ana senaryo", "Base case"),
        ("Alternatif senaryo", "Alternative case"),
        ("Gecersizlik seviyesi", "Invalidation level"),
        ("Vade kaynagi", "Expiry source"),
        ("GEX kaynagi", "GEX source"),
        ("Public veri", "Public data"),
        ("Uretim zamani", "Generated at"),
        ("Kisa Ozet", "Quick Summary"),
        ("Dunden Bugune", "Yesterday vs Today"),
        ("Gunluk Trend", "Daily Trend"),
        ("Dealer Okumasi", "Dealer Read"),
        ("Kalabaliklasma Okumasi", "Crowding Read"),
        ("Piyasanin Fiyatladigi Sey", "What The Market Is Pricing"),
        ("Piyasanin Iskalayabilecegi Sey", "What The Market May Be Missing"),
        ("Akis Yorumu", "Flow Read"),
        ("Akis Grafigi", "Flow Chart"),
        ("Gunluk / Haftalik / Aylik Flow Notu", "Daily / Weekly / Monthly Flow Note"),
        ("Gunluk:", "Daily:"),
        ("Haftalik:", "Weekly:"),
        ("Aylik:", "Monthly:"),
        ("Vade Yapisi", "Expiry Structure"),
        ("Spot Cevresinde Strike Yigilmasi", "Strike Crowding Around Spot"),
        ("Gamma Haritasi", "Gamma Map"),
        ("Kritik Gamma Dugumleri", "Key Gamma Nodes"),
        ("Pozitif", "Positive"),
        ("Negatif", "Negative"),
        ("Taze Konumlanma", "Fresh Positioning"),
        ("Volatilite Baglami", "Volatility Context"),
        ("Referans vade", "Reference expiry"),
        ("Profesyonel Son Okuma", "Professional Final Read"),
        ("En Guclu Instrument OI Eklemeleri", "Largest Instrument OI Additions"),
        ("En Guclu Instrument OI Bosalmalari", "Largest Instrument OI Reductions"),
        ("Strike Heatmap Benzeri Eklemeler", "Strike-Level OI Additions"),
        ("Strike Heatmap Benzeri Bosalmalar", "Strike-Level OI Reductions"),
        ("Sabah Masa Notu", "Morning Desk Note"),
        ("Yaklasik Spot", "Approx Spot"),
        ("Pin Bandi", "Pin Band"),
        ("Ertesi Gun Vadesi", "Next-Day Expiry"),
        ("Ertesi Gun OI", "Next-Day OI"),
        ("24s OI Eklenmesi", "24h OI Added"),
        ("Call / Put Notional", "Call / Put Notional"),
        ("7g Realized Vol", "7d Realized Vol"),
        ("IV - RV Spread", "IV - RV Spread"),
        ("Flow skoru", "Flow score"),
        ("Baskin ton", "Dominant tone"),
        ("Net ton", "Net tone"),
        ("Squeeze trigger", "Squeeze trigger"),
        ("Breakout confirm", "Breakout confirm"),
        ("Breakdown trigger", "Breakdown trigger"),
        ("Acceleration trigger", "Acceleration trigger"),
        ("Reject above", "Reject above"),
        ("Reject below", "Reject below"),
        ("Acik Pozisyon", "Open Interest"),
        ("Notional BTC", "Notional BTC"),
        ("24s OI Degisimi", "24h OI Change"),
        ("Strike", "Strike"),
        ("Toplam", "Total"),
        ("Call Notional", "Call Notional"),
        ("Put Notional", "Put Notional"),
        ("Net", "Net"),
        ("Call", "Call"),
        ("Put", "Put"),
        ("Vade", "Expiry"),
        ("BTC Degisimi", "BTC Change"),
        ("Forward", "Forward"),
        ("ATM spot IV", "ATM spot IV"),
        ("25d skew", "25d skew"),
        ("25d RR", "25d RR"),
        ("10d fly", "10d fly"),
        ("Instrument", "Instrument"),
        ("OI Degisimi", "OI Change"),
        ("Toplam Notional", "Total Notional"),
        ("Call Degisimi", "Call Change"),
        ("Put Degisimi", "Put Change"),
        ("Base case", "Base case"),
        ("Bull case", "Bull case"),
        ("Bear case", "Bear case"),
        ("Bugun icin ana tablo", "Main view for today"),
        ("Yukari tarafta", "On the upside"),
        ("Asagi tarafta", "On the downside"),
        ("yerel gunluk arsivin ilk baz gunu olarak kaydedildi", "was recorded as the first local archive baseline day"),
        ("Yarin itibariyla", "Starting tomorrow"),
        ("anlamli karsilastirma vermeye baslayacak", "it will begin to provide meaningful comparisons"),
        ("Yerel arsiv ilk gun", "Local archive first day"),
        ("spot", "spot"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def tr_to_en_list(items: List[str]) -> List[str]:
    return [tr_to_en_text(item) for item in items]


def render_table_rows(rows: List[Dict], kind: str) -> str:
    if kind == "expiry":
        return "".join(
            f"<tr><td>{row['maturity']}</td><td>{fmt_usd(row['open_interest'])}</td>"
            f"<td>{row['open_interest_notional']:,.0f}</td><td>{fmt_usd(row['open_interest_change_usd'])}</td></tr>"
            for row in rows[:8]
        )
    if kind == "gex":
        return "".join(
            f"<tr><td>{row['strike']}</td><td>{row['net_gex']:.3f}</td>"
            f"<td>{row['call_gex']:.3f}</td><td>{row['put_gex']:.3f}</td></tr>"
            for row in rows[:5]
        )
    if kind == "crowded":
        return "".join(
            f"<tr><td>{row['strike']}</td><td>{fmt_usd(row['call_notional'])}</td>"
            f"<td>{fmt_usd(row['put_notional'])}</td><td>{fmt_usd(row['total_notional'])}</td></tr>"
            for row in rows[:8]
        )
    if kind == "change":
        return "".join(
            f"<tr><td>{row['maturity']}</td><td>{fmt_usd(row['open_interest_change_usd'])}</td>"
            f"<td>{row['open_interest_notional_change']:,.0f}</td></tr>"
            for row in rows[:6]
        )
    if kind == "instrument":
        return "".join(
            f"<tr><td>{row['instrument']}</td><td>{fmt_usd(row['oi_change_notional'])}</td>"
            f"<td>{fmt_usd(row['notional'])}</td><td>{row['open_interest']:,.0f}</td></tr>"
            for row in rows[:8]
        )
    if kind == "strike_change":
        return "".join(
            f"<tr><td>{row['strike']}</td><td>{fmt_usd(row['call_change_notional'])}</td>"
            f"<td>{fmt_usd(row['put_change_notional'])}</td><td>{fmt_usd(row['net_change_notional'])}</td></tr>"
            for row in rows[:8]
        )
    if kind == "iv_term":
        return "".join(
            f"<tr><td>{row['maturity']}</td><td>{row['atm']:.2f}</td><td>{row['rr_25d']:.2f}</td><td>{row['bf_25d']:.2f}</td></tr>"
            for row in rows[:8]
        )
    if kind == "strategy":
        return "".join(
            f"<tr><td>{row['strategy']}</td><td>{fmt_usd(row['total_notional'])}</td><td>{row['oi_change']:,.0f}</td><td>{row['delta']:.1f}</td></tr>"
            for row in rows[:8]
        )
    return ""


def render_html(
    currency: str,
    summary: Dict[str, object],
    change_section: Dict[str, object],
    flow_horizon_notes: Dict[str, List[str]],
    expiry_rows: List[Dict],
    expiry_chart: Path,
    gex_chart: Path,
    strike_chart: Path,
    archive_chart: Path,
    flow_chart: Path,
    heatmap_chart: Path,
    iv_term_chart: Path,
    iv_rv_chart: Path,
    strategy_chart: Path,
    inventory_chart: Path,
    inventory_trend_chart: Path,
    api_calls: int,
    public_calls: int,
) -> str:
    generated = UTC_NOW().isoformat()
    type_rows = summary["type_rows"]
    vol_context = summary["vol_context"]
    weekly_block = ""
    if change_section.get("weekly"):
        weekly = change_section["weekly"]
        weekly_block = (
            f"<p><strong>5 gunluk referans:</strong> Spot {weekly['spot']:+,.0f}, "
            f"ATM IV {weekly['iv']:+.2f}, ertesi gun OI {weekly['next_day_oi'] / 1_000_000:+.1f} milyon USD.</p>"
        )
    return f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>{currency} Sabah Opsiyon Notu</title>
<style>
:root {{ --bg:#efe6d3; --paper:#f8f2e7; --paper-2:#fffdf8; --ink:#191713; --muted:#6b6459; --line:#d6c7ae; --line-2:#eadfcf; --accent:#123b52; --gold:#b8903d; --up:#2f7d51; --down:#a23c31; --navy:#071126; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:
  radial-gradient(circle at 0% 0%, rgba(255,255,255,0.65), transparent 32%),
  linear-gradient(135deg, #f7f2e9 0%, #ecdfca 52%, #e7d7be 100%);
  font-family:'Trebuchet MS', Verdana, sans-serif; }}
main {{ max-width:1240px; margin:0 auto; padding:34px 24px 48px; }}
section {{ background:rgba(255,253,248,0.92); border:1px solid var(--line-2); border-radius:24px; padding:26px; margin-bottom:20px; box-shadow:0 18px 48px rgba(37,29,15,0.08); backdrop-filter:blur(6px); }}
h1,h2,h3 {{ margin:0 0 12px; font-family:'Palatino Linotype', 'Book Antiqua', Georgia, serif; font-weight:700; letter-spacing:-0.02em; }}
h1 {{ font-size:46px; line-height:0.98; max-width:none; }}
h2 {{ font-size:28px; }}
h3 {{ font-size:18px; }}
p {{ margin:0 0 12px; line-height:1.68; color:#201c17; }}
.hero {{ position:relative; overflow:hidden; padding:34px; background:
  linear-gradient(145deg, rgba(8,24,46,0.95), rgba(18,43,67,0.92)),
  linear-gradient(180deg, #10273f, #0a1626); color:#f6f0e6; border-color:#304660; }}
.hero::after {{ content:''; position:absolute; inset:auto -6% -48% auto; width:340px; height:340px; border-radius:50%; background:radial-gradient(circle, rgba(184,144,61,0.28), rgba(184,144,61,0.02) 65%, transparent 72%); pointer-events:none; }}
.hero p, .hero small {{ color:#f0e7d8; max-width:74ch; position:relative; z-index:1; }}
.hero strong {{ color:#ffffff; }}
.summary-strip, .playbook-strip {{ background:linear-gradient(180deg, rgba(255,252,246,0.98), rgba(246,238,225,0.92)); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
.metric {{ background:linear-gradient(180deg, #fffaf0, #f6edde); border:1px solid #e7d9bf; border-radius:18px; padding:16px; box-shadow:inset 0 1px 0 rgba(255,255,255,0.75); }}
.metric .label {{ font-size:11px; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted); }}
.metric .value {{ font-family:'Palatino Linotype', 'Book Antiqua', Georgia, serif; font-size:31px; margin-top:8px; color:#10253a; }}
.two {{ display:grid; grid-template-columns:1.18fr 1fr; gap:20px; align-items:start; }}
ul {{ margin:0; padding-left:18px; }}
li {{ margin:0 0 9px; line-height:1.55; }}
img {{ width:100%; border-radius:16px; border:1px solid #eadbc2; background:#fff; box-shadow:0 10px 28px rgba(25,20,14,0.06); }}
table {{ width:100%; border-collapse:collapse; font-size:14px; background:rgba(255,255,255,0.65); border-radius:14px; overflow:hidden; }}
th,td {{ padding:10px 8px; text-align:left; border-bottom:1px solid #ecdfc8; }}
th {{ font-size:11px; letter-spacing:0.1em; text-transform:uppercase; color:#6f6558; background:#f8f1e4; }}
small {{ opacity:0.82; font-size:12px; }}
.badge {{ display:inline-block; padding:7px 11px; border-radius:999px; background:rgba(255,244,219,0.16); color:#fff4df; border:1px solid rgba(255,234,187,0.22); font-size:11px; letter-spacing:0.08em; text-transform:uppercase; margin:0 8px 8px 0; }}
.takeaway-shell {{ background:
  linear-gradient(180deg, #071126, #0b1534 68%, #101d45);
  color:#f4f0e6; border-radius:24px; padding:28px; border:1px solid #243564; box-shadow:inset 0 1px 0 rgba(255,255,255,0.04); }}
.takeaway-card {{ background:linear-gradient(180deg, #f3efe6, #f8f4ed); color:#161b20; border-radius:12px; padding:22px; border:1px solid #c7cddd; box-shadow:0 14px 34px rgba(5,10,20,0.22); }}
.takeaway-card h2 {{ text-align:center; margin-bottom:20px; font-size:24px; }}
.takeaway-card p {{ margin:0 0 12px; line-height:1.48; }}
.takeaway-strong {{ font-weight:700; }}
.takeaway-red {{ color:#c93c32; font-weight:700; }}
.takeaway-green {{ color:#53b95d; font-weight:700; }}
.takeaway-box {{ border:3px solid #d92a1d; padding:8px 10px; margin:12px 0 14px; background:#fff4f1; box-shadow:inset 0 1px 0 rgba(255,255,255,0.55); }}
.signal-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:14px; }}
.signal-card {{ border-radius:14px; padding:14px; border:1px solid #d9d0bd; font-size:14px; line-height:1.45; box-shadow:0 6px 18px rgba(30,22,10,0.06); }}
.signal-card strong {{ display:block; margin-bottom:6px; font-size:11px; letter-spacing:0.12em; }}
.accept-card {{ background:linear-gradient(180deg, #edf8f0, #e6f2e9); border-color:#98c8a5; }}
.reject-card {{ background:linear-gradient(180deg, #fff1ef, #fce8e5); border-color:#e0a7a0; }}
.delta-row {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }}
.delta-badge {{ display:inline-block; padding:9px 12px; border-radius:999px; font-size:12px; letter-spacing:0.02em; border:1px solid transparent; font-weight:700; }}
.delta-up {{ background:#edf8f0; color:#21623a; border-color:#9dcca9; }}
.delta-down {{ background:#fff1ef; color:#8f2e28; border-color:#e3a7a0; }}
.delta-neutral {{ background:#f3efe6; color:#574f42; border-color:#d8ceb8; }}
@media (max-width: 900px) {{
  main {{ padding:20px 14px 34px; }}
  .two, .signal-grid {{ grid-template-columns:1fr; }}
  section {{ padding:18px; border-radius:18px; }}
  h1 {{ font-size:34px; max-width:none; }}
  h2 {{ font-size:24px; }}
  .metric .value {{ font-size:27px; }}
}}
</style>
</head>
<body>
<main>
  <section class='hero'>
    <h1>{currency} Sabah Opsiyon Notu</h1>
    <p><strong>Ana senaryo:</strong> {summary['base_case']}</p>
    <p><strong>Alternatif senaryo:</strong> {summary['alt_case']}</p>
    <p><strong>Gecersizlik seviyesi:</strong> {summary['invalidation']}</p>
    <p><span class='badge'>Vade kaynagi: {summary['expiry_market']}</span><span class='badge'>GEX kaynagi: {summary['gex_market']}</span><span class='badge'>Laevitas API: {api_calls}</span><span class='badge'>Public veri: {public_calls}</span></p>
    <small>Uretim zamani {generated}</small>
  </section>
  <section class='summary-strip'>
    <h2>Kisa Ozet</h2>
    <ul>{render_list(summary['simple_summary'])}</ul>
  </section>
  <section class='playbook-strip'>
    <h2>Today's Playbook</h2>
    <ul>{render_list(summary['playbook'])}</ul>
    <p><strong>Yesterday vs Today:</strong></p>
    <div class='delta-row'>{render_delta_badges(change_section['headline'])}</div>
  </section>
  <section class='two'>
    <div class='takeaway-shell'>
      <div class='takeaway-card'>
        <h2>KEY LEVELS + TAKEAWAYS</h2>
        <p class='takeaway-red'>{summary['takeaways_structured']['gamma']}</p>
        <p class='takeaway-strong'>{summary['takeaways_structured']['balance']}</p>
        <p class='takeaway-strong'>UPSIDE TEST: {summary['takeaways_structured']['upside']}</p>
        <div class='takeaway-box'>{summary['takeaways_structured']['upside_note']}</div>
        <p><strong>DOWNSIDE TEST:</strong> {summary['takeaways_structured']['downside']}</p>
        <p>{summary['takeaways_structured']['downside_note']}</p>
        <div class='signal-grid'>
          <div class='signal-card accept-card'>
            <strong>ACCEPT ABOVE</strong>
            Spot {summary['takeaways_structured']['squeeze_trigger']:,.0f} ustunde kalir ve call akisi desteklerse yukari test aktif.
          </div>
          <div class='signal-card reject-card'>
            <strong>REJECT ABOVE</strong>
            {summary['takeaways_structured']['reject_upside']:,.0f} civari red gelirse ralliye guvenme; range mantigi suruyor.
          </div>
          <div class='signal-card accept-card'>
            <strong>ACCEPT BELOW</strong>
            {summary['takeaways_structured']['breakdown_trigger']:,.0f} alti kabul downside hedge akislarini buyutur.
          </div>
          <div class='signal-card reject-card'>
            <strong>REJECT BELOW</strong>
            {summary['takeaways_structured']['reject_downside']:,.0f} alti hizla geri aliniyorsa stop-hunt / range return oku.
          </div>
        </div>
      </div>
    </div>
    <div>
      <h2>Dealer Inventory Proxy</h2>
      <img src='laevitas/{inventory_chart.name}' alt='Dealer inventory proxy chart'>
      <p><strong>Net ton:</strong> {summary['inventory_label']} ({summary['inventory_net']:+.2f})</p>
      <p>Bu chart gercek MM inventory degil; gamma, flow, skew, IV-RV ve strategy tape uzerinden uretilen bir proxy skorudur.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Inventory Trend</h2>
      <img src='laevitas/{inventory_trend_chart.name}' alt='Dealer inventory proxy trend'>
      <p>Yerel arsiv buyudukce bu grafik MM/dealer tonunun zaman icinde daha net izlenmesini saglar.</p>
    </div>
    <div>
      <h2>Trigger Map</h2>
      <p><strong>Squeeze trigger:</strong> {summary['takeaways_structured']['squeeze_trigger']:,.0f}</p>
      <p><strong>Breakout confirm:</strong> {summary['takeaways_structured']['breakout_confirm']:,.0f}</p>
      <p><strong>Breakdown trigger:</strong> {summary['takeaways_structured']['breakdown_trigger']:,.0f}</p>
      <p><strong>Acceleration trigger:</strong> {summary['takeaways_structured']['acceleration_trigger']:,.0f}</p>
      <p><strong>Reject above:</strong> {summary['takeaways_structured']['reject_upside']:,.0f}</p>
      <p><strong>Reject below:</strong> {summary['takeaways_structured']['reject_downside']:,.0f}</p>
      <ul>{render_list(summary['acceptance_rules'])}</ul>
    </div>
  </section>
  <section>
    <h2>Sabah Masa Notu</h2>
    {render_paragraphs(summary['morning_note'])}
  </section>
  <section>
    <div class='grid'>
      <div class='metric'><div class='label'>Yaklasik Spot</div><div class='value'>{summary['spot']:,.0f}</div></div>
      <div class='metric'><div class='label'>Pin Bandi</div><div class='value'>{summary['pin_low']:,.0f} - {summary['pin_high']:,.0f}</div></div>
      <div class='metric'><div class='label'>Ertesi Gun Vadesi</div><div class='value'>{summary['next_day']['maturity']}</div></div>
      <div class='metric'><div class='label'>Ertesi Gun OI</div><div class='value'>{fmt_usd(summary['next_day']['open_interest'])}</div></div>
      <div class='metric'><div class='label'>24s OI Eklenmesi</div><div class='value'>{fmt_usd(summary['next_day_change'])}</div></div>
      <div class='metric'><div class='label'>Call / Put Notional</div><div class='value'>{type_rows.get('C',{}).get('notional',0)/max(type_rows.get('P',{}).get('notional',1),1):.2f}x</div></div>
      <div class='metric'><div class='label'>Front ATM IV</div><div class='value'>{summary['front_atm_iv']:.2f}</div></div>
      <div class='metric'><div class='label'>7g Realized Vol</div><div class='value'>{summary['rv_7d']:.2f}</div></div>
      <div class='metric'><div class='label'>IV - RV Spread</div><div class='value'>{summary['iv_rv_7d_spread']:+.2f}</div></div>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Dunden Bugune</h2>
      <ul>{render_list(change_section['notes'])}</ul>
      {weekly_block}
    </div>
    <div>
      <h2>Gunluk Trend</h2>
      <img src='laevitas/{archive_chart.name}' alt='Gunluk trend'>
      <p>Bu grafik yerel arsivden beslenir. Ilk gunlerde kisitli olur; her sabah veri geldikce yorum kalitesi artar.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Dealer Okumasi</h2>
      <ul>{render_list(summary['dealer'])}</ul>
    </div>
    <div>
      <h2>Kalabaliklasma Okumasi</h2>
      <ul>{render_list(summary['crowding'])}</ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Piyasanin Fiyatladigi Sey</h2>
      <ul>{render_list(summary['pricing'])}</ul>
    </div>
    <div>
      <h2>Piyasanin Iskalayabilecegi Sey</h2>
      <ul>{render_list(summary['blind_spots'])}</ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Vol Market Intelligence</h2>
      <ul>{render_list(summary['vol_summary'])}</ul>
      <p><strong>Front maturity:</strong> {summary['ts_front_maturity']}</p>
      <p><strong>Gunluk ATM IV degisimi:</strong> {summary['ts_day_change']:+.2f}</p>
      <p><strong>Haftalik ATM IV degisimi:</strong> {summary['ts_week_change']:+.2f}</p>
    </div>
    <div>
      <h2>IV ve RV Karsilastirmasi</h2>
      <img src='laevitas/{iv_rv_chart.name}' alt='IV RV chart'>
      <p>Laevitas historical RV endpoint'i bu key ile kapali oldugu icin realized vol public BTC gunluk kapanis serisinden hesaplanir.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Akis Yorumu</h2>
      <p><strong>Flow skoru:</strong> {summary['flow_label']} ({summary['flow_score']:+d})</p>
      <ul>{render_list(summary['flow_summary'])}</ul>
    </div>
    <div>
      <h2>Akis Grafigi</h2>
      <img src='laevitas/{flow_chart.name}' alt='Akis degisimi grafigi'>
      <p>Bu grafik son 24 saatte hangi strike'larda yeni risk eklendigini ve hangi strike'larda riskin bosaldigini en hizli sekilde gosterir.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Flow Heatmap</h2>
      <img src='laevitas/{heatmap_chart.name}' alt='Flow heatmap'>
      <p>Heatmap spot cevresinde call ve put akisinin tam olarak hangi strike'larda yogunlastigini gosterir.</p>
    </div>
    <div>
      <h2>Gunluk / Haftalik / Aylik Flow Notu</h2>
      <p><strong>Gunluk:</strong></p>
      <ul>{render_list(flow_horizon_notes['daily'])}</ul>
      <p><strong>Haftalik:</strong></p>
      <ul>{render_list(flow_horizon_notes['weekly'])}</ul>
      <p><strong>Aylik:</strong></p>
      <ul>{render_list(flow_horizon_notes['monthly'])}</ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>ATM IV Term Structure</h2>
      <img src='laevitas/{iv_term_chart.name}' alt='ATM IV term structure'>
      <table><tr><th>Vade</th><th>ATM IV</th><th>25d RR</th><th>25d BF</th></tr>{render_table_rows(summary['iv_term_rows'], 'iv_term')}</table>
    </div>
    <div>
      <h2>Strategy Tape</h2>
      <img src='laevitas/{strategy_chart.name}' alt='Strategy chart'>
      <p><strong>Baskin ton:</strong> {summary['strategy_bias']}</p>
      <ul>{render_list(summary['strategy_summary'])}</ul>
      <table><tr><th>Strategy</th><th>Notional</th><th>OI Degisimi</th><th>Delta</th></tr>{render_table_rows(summary['strategy_rows'], 'strategy')}</table>
    </div>
  </section>
  <section>
    <h2>Vade Yapisi</h2>
    <img src='laevitas/{expiry_chart.name}' alt='Expiry OI chart'>
    <table><tr><th>Vade</th><th>Acik Pozisyon</th><th>Notional BTC</th><th>24s OI Degisimi</th></tr>{render_table_rows(expiry_rows, 'expiry')}</table>
  </section>
  <section>
    <h2>Spot Cevresinde Strike Yigilmasi</h2>
    <img src='laevitas/{strike_chart.name}' alt='Strike OI chart'>
    <table><tr><th>Strike</th><th>Call Notional</th><th>Put Notional</th><th>Toplam</th></tr>{render_table_rows(summary['crowded'], 'crowded')}</table>
  </section>
  <section class='two'>
    <div>
      <h2>Gamma Haritasi</h2>
      <img src='laevitas/{gex_chart.name}' alt='GEX chart'>
    </div>
    <div>
      <h2>Kritik Gamma Dugumleri</h2>
      <h3>Pozitif</h3>
      <table><tr><th>Strike</th><th>Net</th><th>Call</th><th>Put</th></tr>{render_table_rows(summary['positive_nodes'], 'gex')}</table>
      <h3>Negatif</h3>
      <table><tr><th>Strike</th><th>Net</th><th>Call</th><th>Put</th></tr>{render_table_rows(summary['negative_nodes'], 'gex')}</table>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Taze Konumlanma</h2>
      <table><tr><th>Vade</th><th>24s OI Degisimi</th><th>BTC Degisimi</th></tr>{render_table_rows(summary['front_changes'], 'change')}</table>
    </div>
    <div>
      <h2>Volatilite Baglami</h2>
      <p><strong>Referans vade:</strong> {vol_context.get('expiry', VOL_CONTEXT_EXPIRY)}</p>
      <p><strong>Forward:</strong> {vol_context.get('forward', 0):,.2f}</p>
      <p><strong>ATM spot IV:</strong> {vol_context.get('atm_spot_iv', 0):.2f}</p>
      <p><strong>25d skew:</strong> {vol_context.get('skew_25d', 0):.2f}</p>
      <p><strong>25d RR:</strong> {vol_context.get('rr_25d', 0):.2f}</p>
      <p><strong>10d fly:</strong> {vol_context.get('fly_10d', 0):.2f}</p>
    </div>
  </section>
  <section>
    <h2>Profesyonel Son Okuma</h2>
    <ul>{render_list(summary['final_read'])}</ul>
  </section>
  <section class='two'>
    <div>
      <h2>En Guclu Instrument OI Eklemeleri</h2>
      <table><tr><th>Instrument</th><th>OI Degisimi</th><th>Toplam Notional</th><th>Acik Pozisyon</th></tr>{render_table_rows(summary['top_adds'], 'instrument')}</table>
    </div>
    <div>
      <h2>En Guclu Instrument OI Bosalmalari</h2>
      <table><tr><th>Instrument</th><th>OI Degisimi</th><th>Toplam Notional</th><th>Acik Pozisyon</th></tr>{render_table_rows(summary['top_cuts'], 'instrument')}</table>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Strike Heatmap Benzeri Eklemeler</h2>
      <table><tr><th>Strike</th><th>Call Degisimi</th><th>Put Degisimi</th><th>Net</th></tr>{render_table_rows(summary['strike_adds'], 'strike_change')}</table>
    </div>
    <div>
      <h2>Strike Heatmap Benzeri Bosalmalar</h2>
      <table><tr><th>Strike</th><th>Call Degisimi</th><th>Put Degisimi</th><th>Net</th></tr>{render_table_rows(summary['strike_cuts'], 'strike_change')}</table>
    </div>
  </section>
</main>
</body>
</html>"""


def render_html_en(
    currency: str,
    summary: Dict[str, object],
    change_section: Dict[str, object],
    flow_horizon_notes: Dict[str, List[str]],
    expiry_rows: List[Dict],
    expiry_chart: Path,
    gex_chart: Path,
    strike_chart: Path,
    archive_chart: Path,
    flow_chart: Path,
    heatmap_chart: Path,
    iv_term_chart: Path,
    iv_rv_chart: Path,
    strategy_chart: Path,
    inventory_chart: Path,
    inventory_trend_chart: Path,
    api_calls: int,
    public_calls: int,
) -> str:
    summary_en = dict(summary)
    for key in [
        "simple_summary",
        "playbook",
        "acceptance_rules",
        "morning_note",
        "dealer",
        "crowding",
        "pricing",
        "blind_spots",
        "vol_summary",
        "flow_summary",
        "strategy_summary",
        "final_read",
    ]:
        summary_en[key] = tr_to_en_list(summary.get(key, []))

    summary_en["base_case"] = tr_to_en_text(summary["base_case"])
    summary_en["alt_case"] = tr_to_en_text(summary["alt_case"])
    summary_en["invalidation"] = tr_to_en_text(summary["invalidation"])
    summary_en["flow_label"] = tr_to_en_text(summary["flow_label"])
    summary_en["strategy_bias"] = tr_to_en_text(summary["strategy_bias"])
    summary_en["inventory_label"] = tr_to_en_text(summary["inventory_label"])

    takeaways_structured = dict(summary["takeaways_structured"])
    for key in ["gamma", "balance", "upside_note", "downside_note"]:
        takeaways_structured[key] = tr_to_en_text(takeaways_structured[key])
    summary_en["takeaways_structured"] = takeaways_structured

    change_section_en = {
        "notes": tr_to_en_list(change_section.get("notes", [])),
        "headline": tr_to_en_list(change_section.get("headline", [])),
        "weekly": change_section.get("weekly"),
    }
    flow_horizon_notes_en = {key: tr_to_en_list(value) for key, value in flow_horizon_notes.items()}

    html = render_html(
        currency,
        summary_en,
        change_section_en,
        flow_horizon_notes_en,
        expiry_rows,
        expiry_chart,
        gex_chart,
        strike_chart,
        archive_chart,
        flow_chart,
        heatmap_chart,
        iv_term_chart,
        iv_rv_chart,
        strategy_chart,
        inventory_chart,
        inventory_trend_chart,
        api_calls,
        public_calls,
    )
    return tr_to_en_text(html)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a trend-aware morning options report from Laevitas.")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--market", default="deribit")
    args = parser.parse_args()

    ensure_dirs()
    expiry_rows, expiry_market, expiry_calls = load_expiry_rows(args.currency, args.market)
    gex_payload = fetch_gex_all(args.market, args.currency)
    save_snapshot(f"{args.currency.lower()}_gex_all_{args.market}", gex_payload)
    gex_rows = aggregate_gex_by_strike(gex_payload)
    strike_payload = fetch_oi_strike_all(args.market, args.currency)
    save_snapshot(f"{args.currency.lower()}_oi_strike_all_{args.market}", strike_payload)
    strike_rows = parse_strike_oi_rows(strike_payload)
    type_payload = fetch_oi_type(args.market, args.currency)
    save_snapshot(f"{args.currency.lower()}_oi_type_{args.market}", type_payload)
    type_rows = parse_type_rows(type_payload)
    change_payload = fetch_oi_change_summary(args.currency, 24)
    save_snapshot(f"{args.currency.lower()}_oi_change_summary_24h", change_payload)
    change_rows = parse_change_rows(change_payload)
    vol_payload = fetch_vol_context(args.currency, VOL_CONTEXT_EXPIRY)
    save_snapshot(f"{args.currency.lower()}_vol_context_{VOL_CONTEXT_EXPIRY.lower()}", vol_payload)
    vol_context = parse_vol_context(vol_payload)
    atm_iv_ts_payload = fetch_atm_iv_ts(args.market, args.currency)
    save_snapshot(f"{args.currency.lower()}_atm_iv_ts_{args.market}", atm_iv_ts_payload)
    atm_iv_ts = parse_atm_iv_ts(atm_iv_ts_payload)
    iv_table_payload = fetch_iv_table(args.market, args.currency)
    save_snapshot(f"{args.currency.lower()}_iv_table_{args.market}", iv_table_payload)
    iv_table_rows = parse_iv_table_rows(iv_table_payload)
    strategy_payload = fetch_top_strategies(args.currency, 24)
    save_snapshot(f"{args.currency.lower()}_top_strategies_24h", strategy_payload if isinstance(strategy_payload, dict) else {"data": strategy_payload})
    strategy_rows = parse_top_strategy_rows(strategy_payload)
    top_instrument_payload = fetch_top_instrument_oi_change(args.market, args.currency, 24)
    save_snapshot(f"{args.currency.lower()}_top_instrument_oi_change_{args.market}_24h", top_instrument_payload)
    top_instrument_rows = parse_top_instrument_rows(top_instrument_payload)
    strike_change_payload = fetch_oi_net_change_all(args.market, args.currency, 24)
    save_snapshot(f"{args.currency.lower()}_oi_net_change_all_{args.market}_24h", strike_change_payload)
    strike_change_rows = parse_oi_net_change_all_rows(strike_change_payload)
    realized_vols = compute_realized_vols(fetch_public_btc_daily_prices(), [7, 14, 30])

    if not expiry_rows or not gex_rows or not strike_rows or not type_rows or not change_rows:
        raise RuntimeError("Laevitas response did not contain enough data to build the report.")

    summary = summarize_flow(
        expiry_rows,
        gex_rows,
        strike_rows,
        type_rows,
        change_rows,
        vol_context,
        atm_iv_ts,
        iv_table_rows,
        strategy_rows,
        realized_vols,
        top_instrument_rows,
        strike_change_rows,
        expiry_market,
        args.market,
    )
    today = UTC_NOW().strftime("%Y-%m-%d")
    archive_rows = upsert_archive(build_archive_row(summary))
    upsert_daily_rows(
        STRIKE_FLOW_CSV,
        strike_flow_fieldnames(),
        "date",
        build_strike_flow_rows(today, strike_change_rows),
        ["strike"],
    )
    upsert_daily_rows(
        INSTRUMENT_FLOW_CSV,
        instrument_flow_fieldnames(),
        "date",
        build_instrument_flow_rows(today, top_instrument_rows),
        ["instrument"],
    )
    change_section = build_change_section(summary, archive_rows)
    flow_horizon_notes = build_flow_horizon_notes(today, strike_change_rows, top_instrument_rows)

    expiry_chart = CHART_DIR / f"{args.currency.lower()}_expiry_oi.png"
    gex_chart = CHART_DIR / f"{args.currency.lower()}_gex.png"
    strike_chart = CHART_DIR / f"{args.currency.lower()}_strike_oi.png"
    archive_chart = CHART_DIR / f"{args.currency.lower()}_archive_trend.png"
    flow_chart = CHART_DIR / f"{args.currency.lower()}_flow_change.png"
    heatmap_chart = CHART_DIR / f"{args.currency.lower()}_flow_heatmap.png"
    iv_term_chart = CHART_DIR / f"{args.currency.lower()}_iv_term_structure.png"
    iv_rv_chart = CHART_DIR / f"{args.currency.lower()}_iv_rv.png"
    strategy_chart = CHART_DIR / f"{args.currency.lower()}_top_strategies.png"
    inventory_chart = CHART_DIR / f"{args.currency.lower()}_inventory_proxy.png"
    inventory_trend_chart = CHART_DIR / f"{args.currency.lower()}_inventory_proxy_trend.png"
    make_expiry_chart(expiry_rows, expiry_chart)
    make_gex_chart(gex_rows, summary["spot"], gex_chart)
    make_strike_oi_chart(strike_rows, summary["spot"], strike_chart)
    make_archive_chart(archive_rows, archive_chart)
    make_flow_change_chart(summary["strike_adds"], summary["strike_cuts"], flow_chart)
    make_flow_heatmap(strike_change_rows, summary["spot"], heatmap_chart)
    make_iv_term_chart(atm_iv_ts, iv_term_chart)
    make_iv_rv_chart(summary, iv_rv_chart)
    make_strategy_chart(summary["strategy_rows"], strategy_chart)
    make_mm_inventory_chart(summary, inventory_chart)
    make_inventory_trend_chart(archive_rows, inventory_trend_chart)

    api_calls = expiry_calls + 10
    public_calls = 1
    html = render_html(
        args.currency,
        summary,
        change_section,
        flow_horizon_notes,
        expiry_rows,
        expiry_chart,
        gex_chart,
        strike_chart,
        archive_chart,
        flow_chart,
        heatmap_chart,
        iv_term_chart,
        iv_rv_chart,
        strategy_chart,
        inventory_chart,
        inventory_trend_chart,
        api_calls,
        public_calls,
    )
    out_html = REPORT_DIR / f"{args.currency.lower()}_morning_report.html"
    out_html.write_text(html, encoding="utf-8")
    html_en = render_html_en(
        args.currency,
        summary,
        change_section,
        flow_horizon_notes,
        expiry_rows,
        expiry_chart,
        gex_chart,
        strike_chart,
        archive_chart,
        flow_chart,
        heatmap_chart,
        iv_term_chart,
        iv_rv_chart,
        strategy_chart,
        inventory_chart,
        inventory_trend_chart,
        api_calls,
        public_calls,
    )
    out_html_en = REPORT_DIR / f"{args.currency.lower()}_morning_report_en.html"
    out_html_en.write_text(html_en, encoding="utf-8")
    write_pages_index(args.currency)
    print(f"report={out_html}")
    print(f"report_en={out_html_en}")
    print(f"expiry_chart={expiry_chart}")
    print(f"gex_chart={gex_chart}")
    print(f"strike_chart={strike_chart}")
    print(f"archive_chart={archive_chart}")
    print(f"flow_chart={flow_chart}")
    print(f"heatmap_chart={heatmap_chart}")
    print(f"iv_term_chart={iv_term_chart}")
    print(f"iv_rv_chart={iv_rv_chart}")
    print(f"strategy_chart={strategy_chart}")
    print(f"inventory_chart={inventory_chart}")
    print(f"inventory_trend_chart={inventory_trend_chart}")
    print(f"archive_rows={len(archive_rows)}")
    print(f"api_calls={api_calls}")
    print(f"public_calls={public_calls}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"fatal_error={exc.__class__.__name__}: {exc}")
        raise
