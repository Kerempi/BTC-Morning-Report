import argparse
import csv
import json
import math
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from laevitas_api import laevitas_get, laevitas_graphql

REPORT_DIR = Path("reports")
CHART_DIR = REPORT_DIR / "laevitas"
DATA_DIR = Path("data") / "laevitas"
ARCHIVE_CSV = DATA_DIR / "btc_daily_archive.csv"
STRIKE_FLOW_CSV = DATA_DIR / "btc_daily_strike_flow.csv"
INSTRUMENT_FLOW_CSV = DATA_DIR / "btc_daily_instrument_flow.csv"
UTC_NOW = lambda: datetime.now(timezone.utc)
VOL_CONTEXT_EXPIRY = "25SEP26"
COINGECKO_MARKET_CHART = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=900&interval=daily"
BINANCE_BTC_DAILY = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000"
DEFAULT_BTC_POSITION_MAP_LEVELS = {
    "tail_hedge": 20000.0,
    "put_hedge_low": 55000.0,
    "put_hedge_high": 60000.0,
    "magnet_strike": 75000.0,
    "call_cluster_high": 85000.0,
    "speculation": 100000.0,
}
MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def loc(lang: str, tr: str, en: str) -> str:
    return en if lang == "en" else tr


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_maturity_date(maturity: str):
    match = re.fullmatch(r"(\d{1,2})([A-Z]{3})(\d{2})", str(maturity).upper())
    if not match:
        return None
    day = int(match.group(1))
    month = MONTH_MAP.get(match.group(2))
    year = 2000 + int(match.group(3))
    if not month:
        return None
    try:
        return datetime(year, month, day, tzinfo=timezone.utc).date()
    except ValueError:
        return None


def is_quarterly_maturity(maturity: str) -> bool:
    dt = parse_maturity_date(maturity)
    if not dt:
        return False
    return dt.month in (3, 6, 9, 12) and dt.day >= 15


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
  <p>Automated morning report published via GitHub Pages.</p>
  <div class='grid'>
    <div class='card'>
      <h2>Turkce</h2>
      <p>Guncel sabah opsiyon raporu, grafikler ve desk note ozeti.</p>
      <a class='button' href='btc_morning_report.html'>Raporu Ac</a>
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


def fetch_gex_term_structure(currency: str) -> Dict:
    return laevitas_graphql(f'{{ getGexTermStructure(currency:"{currency.upper()}") }}')


def save_snapshot(name: str, payload: Dict) -> Path:
    stamp = UTC_NOW().strftime("%Y%m%d")
    out = DATA_DIR / f"{name}_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out


def payload_has_nonempty_data(payload: Dict) -> bool:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return bool(data)
    if isinstance(data, dict):
        return bool(data)
    return data not in (None, "")


def retry_empty_payload(fetch_fn, parse_fn, retries: int = 1, sleep_seconds: float = 2.0):
    payload = fetch_fn()
    rows = parse_fn(payload)
    attempts = 0
    while not rows and attempts < retries:
        attempts += 1
        time.sleep(sleep_seconds)
        payload = fetch_fn()
        rows = parse_fn(payload)
    return payload, rows, attempts


def extract_snapshot_date(snapshot_path: Path) -> str:
    match = re.search(r"_(\d{8})\.json$", snapshot_path.name)
    if not match:
        return snapshot_path.stem
    stamp = match.group(1)
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


def load_latest_nonempty_snapshot(name: str) -> Tuple[Optional[Dict], Optional[Path]]:
    candidates = sorted(DATA_DIR.glob(f"{name}_*.json"), reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload_has_nonempty_data(payload):
            return payload, candidate
    return None, None


def load_previous_nonempty_snapshot(name: str, before_date_text: str) -> Tuple[Optional[Dict], Optional[Path]]:
    before_stamp = before_date_text.replace("-", "")
    candidates = sorted(DATA_DIR.glob(f"{name}_*.json"), reverse=True)
    for candidate in candidates:
        match = re.search(r"_(\d{8})\.json$", candidate.name)
        if not match:
            continue
        if match.group(1) >= before_stamp:
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload_has_nonempty_data(payload):
            return payload, candidate
    return None, None


def load_snapshot_for_date(name: str, date_text: str) -> Tuple[Optional[Dict], Optional[Path]]:
    stamp = date_text.replace("-", "")
    candidate = DATA_DIR / f"{name}_{stamp}.json"
    if not candidate.exists():
        return None, None
    try:
        return json.loads(candidate.read_text(encoding="utf-8")), candidate
    except Exception:
        return None, None


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
                "call_btc_oi": float(row.get("c", 0.0)),
                "put_btc_oi": float(row.get("p", 0.0)),
                "call_notional": float(row.get("notional_c", 0.0)),
                "put_notional": float(row.get("notional_p", 0.0)),
                "call_premium": float(row.get("premium_c", 0.0)),
                "put_premium": float(row.get("premium_p", 0.0)),
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


def parse_gex_term_structure_rows(payload: Dict) -> List[Dict]:
    root = payload.get("data", {}).get("getGexTermStructure", {}) if isinstance(payload, dict) else {}
    structure = root.get("term_structure", {}) if isinstance(root, dict) else {}
    rows: List[Dict] = []
    if not isinstance(structure, dict):
        return rows
    for maturity, value in structure.items():
        dt = parse_maturity_date(str(maturity))
        if dt is None:
            continue
        rows.append(
            {
                "maturity": str(maturity),
                "date": dt,
                "gex": float(value or 0.0),
            }
        )
    rows.sort(key=lambda row: row["date"])
    return rows


def parse_gex_term_structure_meta(payload: Dict) -> Dict[str, float]:
    root = payload.get("data", {}).get("getGexTermStructure", {}) if isinstance(payload, dict) else {}
    if not isinstance(root, dict):
        return {}
    meta: Dict[str, float] = {}
    try:
        meta["total_gex"] = float(root.get("total_gex", 0.0) or 0.0)
    except Exception:
        pass
    try:
        meta["spot"] = float(root.get("spot", 0.0) or 0.0)
    except Exception:
        pass
    return meta


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


def compute_max_pain(strike_rows: List[Dict]) -> float:
    strikes = sorted({float(row.get("strike", 0.0)) for row in strike_rows if row.get("strike") is not None})
    if not strikes:
        return 0.0

    best_settle = strikes[0]
    best_loss = float("inf")
    for settle in strikes:
        total_loss = 0.0
        for row in strike_rows:
            strike = float(row.get("strike", 0.0))
            call_oi = float(row.get("call_oi", 0.0))
            put_oi = float(row.get("put_oi", 0.0))
            total_loss += call_oi * max(0.0, settle - strike)
            total_loss += put_oi * max(0.0, strike - settle)
        if total_loss < best_loss:
            best_loss = total_loss
            best_settle = settle

    return best_settle


def fmt_usd(value: float) -> str:
    return f"${value:,.0f}"


def fetch_public_btc_daily_series() -> List[Dict[str, object]]:
    series: List[Dict[str, object]] = []
    try:
        with urllib.request.urlopen(COINGECKO_MARKET_CHART, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        prices = payload.get("prices", [])
        series = [
            {
                "ts": datetime.fromtimestamp(float(row[0]) / 1000, tz=timezone.utc),
                "close": float(row[1]),
            }
            for row in prices
            if isinstance(row, list) and len(row) >= 2
        ]
    except Exception:
        with urllib.request.urlopen(BINANCE_BTC_DAILY, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        series = [
            {
                "ts": datetime.fromtimestamp(float(row[0]) / 1000, tz=timezone.utc),
                "close": float(row[4]),
            }
            for row in payload
            if isinstance(row, list) and len(row) >= 5
        ]
    if len(series) < 10:
        raise RuntimeError("Public BTC price history did not contain enough points for charting.")
    return series


def fetch_public_btc_daily_prices() -> List[float]:
    series = fetch_public_btc_daily_series()
    closes = [float(row["close"]) for row in series]
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


def compute_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    for i in range(len(closes) - period, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger_stats(closes: List[float], window: int = 20) -> Dict[str, float]:
    if len(closes) < window:
        last = closes[-1] if closes else 0.0
        return {
            "sma": last,
            "upper": last,
            "lower": last,
            "zscore": 0.0,
            "band_pct": 0.5,
        }
    sample = closes[-window:]
    sma = sum(sample) / window
    variance = sum((value - sma) ** 2 for value in sample) / window
    std = math.sqrt(variance)
    upper = sma + 2 * std
    lower = sma - 2 * std
    last = sample[-1]
    zscore = (last - sma) / std if std else 0.0
    band_pct = (last - lower) / (upper - lower) if upper > lower else 0.5
    return {
        "sma": sma,
        "upper": upper,
        "lower": lower,
        "zscore": zscore,
        "band_pct": band_pct,
    }


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


def round_strike_level(value: float, step: int = 1000) -> float:
    if step <= 0:
        return float(value)
    return float(int(round(value / step) * step))


def bucket_strength(value: float) -> str:
    if value >= 0.75:
        return "strong"
    if value >= 0.4:
        return "medium"
    return "weak"


def build_chain_pressure_rows(
    strike_rows: List[Dict],
    gex_rows: List[Dict],
    strike_change_rows: List[Dict],
    anchor_spot: float,
    window_size: float = 15000.0,
) -> List[Dict[str, float]]:
    nearby_strikes = [row for row in strike_rows if abs(float(row["strike"]) - anchor_spot) <= window_size]
    gex_lookup = {float(row["strike"]): float(row["net_gex"]) for row in gex_rows}
    strike_change_lookup = {float(row["strike"]): float(row["net_change_notional"]) for row in strike_change_rows}
    chain_rows: List[Dict[str, float]] = []
    for row in sorted(nearby_strikes, key=lambda item: float(item["strike"])):
        strike = float(row["strike"])
        call_notional_mn = float(row["call_notional"]) / 1_000_000
        put_notional_mn = float(row["put_notional"]) / 1_000_000
        net_notional_mn = call_notional_mn - put_notional_mn
        net_flow_mn = strike_change_lookup.get(strike, 0.0) / 1_000_000
        net_gex = gex_lookup.get(strike, 0.0)
        upside_pressure = max(call_notional_mn, 0.0) + max(net_flow_mn, 0.0) + max(net_gex, 0.0) * 18
        downside_pressure = max(put_notional_mn, 0.0) + max(-net_flow_mn, 0.0) + max(-net_gex, 0.0) * 18
        chain_rows.append(
            {
                "strike": strike,
                "call_notional": float(row["call_notional"]),
                "put_notional": float(row["put_notional"]),
                "net_notional_mn": net_notional_mn,
                "net_flow_mn": net_flow_mn,
                "net_gex": net_gex,
                "upside_pressure": upside_pressure,
                "downside_pressure": downside_pressure,
            }
        )
    return chain_rows


def build_dealer_pressure_section(
    chain_rows: List[Dict[str, float]],
    previous_chain_rows: Optional[List[Dict[str, float]]] = None,
    previous_date: Optional[str] = None,
    current_date: Optional[str] = None,
) -> Dict[str, object]:
    if not chain_rows:
        return {
            "delta_rows": [],
            "comparison_label": "",
            "badges": [],
            "summary": "Dealer pressure proxy verisi yok.",
            "notes": ["Bu bolum strike OI, GEX ve 24s strike akisindan turetilen proxy baski skoruna dayanir."],
        }

    notes = [
        "Bu harita gercek dealer inventory degil; strike OI, GEX ve 24s strike akisindan uretilen proxy baski skorudur.",
        "Ust harita yapisal stok agirligini gosterir; bu nedenle ayni strike'lar birden fazla gun boyunca benzer gorunebilir.",
    ]
    prev_map = {float(row["strike"]): row for row in (previous_chain_rows or [])}
    delta_rows: List[Dict[str, float]] = []
    for row in chain_rows:
        strike = float(row["strike"])
        prev_row = prev_map.get(strike, {})
        upside_delta = float(row["upside_pressure"]) - float(prev_row.get("upside_pressure", 0.0) or 0.0)
        downside_delta = float(row["downside_pressure"]) - float(prev_row.get("downside_pressure", 0.0) or 0.0)
        delta_rows.append(
            {
                "strike": strike,
                "upside_delta": upside_delta,
                "downside_delta": downside_delta,
                "net_delta": upside_delta - downside_delta,
            }
        )

    comparison_label = f"{previous_date} -> {current_date}" if previous_date and current_date else ""
    badges: List[str] = []
    summary = "Proxy harita stok agirlikli oldugu icin gunluk degisimler bazen yuzeyde yavas gorunur."
    if previous_chain_rows and delta_rows:
        strongest_up = max(delta_rows, key=lambda item: float(item["upside_delta"]))
        strongest_down = max(delta_rows, key=lambda item: float(item["downside_delta"]))
        strongest_net = max(delta_rows, key=lambda item: abs(float(item["net_delta"])))
        total_up_delta = sum(float(item["upside_delta"]) for item in delta_rows)
        total_down_delta = sum(float(item["downside_delta"]) for item in delta_rows)
        badges = [
            f"Yukari baski {'artti' if total_up_delta >= 0 else 'azaldi'}: {total_up_delta:+.1f}",
            f"Asagi baski {'artti' if total_down_delta >= 0 else 'azaldi'}: {total_down_delta:+.1f}",
        ]
        if float(strongest_up["upside_delta"]) > 0:
            badges.append(f"Yukari build: {strongest_up['strike']:,.0f} {float(strongest_up['upside_delta']):+.1f}")
        if float(strongest_down["downside_delta"]) > 0:
            badges.append(f"Asagi build: {strongest_down['strike']:,.0f} {float(strongest_down['downside_delta']):+.1f}")
        if float(strongest_net["net_delta"]) > 0:
            summary = (
                f"Dunden bugune en net yukari kayma {strongest_net['strike']:,.0f} cevresinde; "
                "ana harita benzer kalsa bile taze baski dagilimi yukari yone kaydi."
            )
        elif float(strongest_net["net_delta"]) < 0:
            summary = (
                f"Dunden bugune en net asagi kayma {strongest_net['strike']:,.0f} cevresinde; "
                "ana harita benzer kalsa bile taze baski dagilimi savunmaci tarafa kaydi."
            )
        notes.append(
            "Alt delta chart'i ayni strike'larda bugun dunden daha fazla yukari mi asagi mi baski biriktigini gosterir."
        )

    return {
        "delta_rows": delta_rows,
        "comparison_label": comparison_label,
        "badges": badges,
        "summary": summary,
        "notes": notes,
    }


def build_proxy_hedge_impact_section(summary: Dict[str, object]) -> Dict[str, object]:
    chain_rows = summary.get("chain_rows", [])
    spot = float(summary.get("spot", 0.0) or 0.0)
    if not chain_rows or spot <= 0:
        return {
            "grid_spots": [],
            "grid_scores": [],
            "current_score": 0.0,
            "regime": "Veri yok",
            "regime_class": "delta-neutral",
            "band_label": "Belirsiz",
            "support_spot": 0.0,
            "fragile_spot": 0.0,
            "headline": "Proxy hedge-impact modeli icin yeterli veri yok.",
            "summary": "Bu bolum true dealer inventory degil; veri yoksa chart uretilmez.",
            "badges": [],
            "notes": [
                "Bu bolum signed dealer pozisyonu kullanmaz; strike OI, proxy pressure ve GEX'ten turetilen senaryo bazli bir hedge-impact modelidir.",
            ],
        }

    grid_spots = [spot - 15000 + 500 * i for i in range(61)]
    grid_scores: List[float] = []
    local_gamma_values: List[float] = []
    scale = 2400.0
    for grid_spot in grid_spots:
        net_pressure = 0.0
        local_gamma = 0.0
        for row in chain_rows:
            strike = float(row["strike"])
            distance = abs(grid_spot - strike)
            weight = math.exp(-(distance / scale) ** 2)
            net_pressure += (float(row["upside_pressure"]) - float(row["downside_pressure"])) * weight
            local_gamma += float(row["net_gex"]) * weight
        score = local_gamma * 850.0 + net_pressure * 0.12
        grid_scores.append(score)
        local_gamma_values.append(local_gamma)

    nearest_idx = min(range(len(grid_spots)), key=lambda idx: abs(grid_spots[idx] - spot))
    current_score = float(grid_scores[nearest_idx])
    current_gamma = float(local_gamma_values[nearest_idx])
    support_idx = max(range(len(grid_scores)), key=lambda idx: grid_scores[idx])
    fragile_idx = min(range(len(grid_scores)), key=lambda idx: grid_scores[idx])

    left_idx = nearest_idx
    while left_idx > 0 and grid_scores[left_idx - 1] > 0:
        left_idx -= 1
    right_idx = nearest_idx
    while right_idx < len(grid_scores) - 1 and grid_scores[right_idx + 1] > 0:
        right_idx += 1
    if grid_scores[nearest_idx] > 0:
        band_label = f"{grid_spots[left_idx]:,.0f}-{grid_spots[right_idx]:,.0f}"
    else:
        band_label = "Aktif long-gamma pocket disinda"

    if current_score >= 40:
        regime = "Dengeleyici / long-gamma-benzeri"
        regime_class = "delta-up"
        headline = (
            f"Proxy model, spotun {band_label} civarinda daha dengeleyici bir hedge rejimine yaklastigini soyluyor; "
            "hareketler emilmeye daha yatkin olabilir."
        )
    elif current_score <= -40:
        regime = "Kirilgan / short-gamma-benzeri"
        regime_class = "delta-down"
        headline = (
            "Proxy model, spotun hala kirilgan bir pocket'ta oldugunu soyluyor; "
            "hareketler emilmekten cok buyumeye daha yatkin olabilir."
        )
    else:
        regime = "Gecis / karisik"
        regime_class = "delta-neutral"
        headline = (
            "Proxy model, spotun temiz bir long-gamma ya da short-gamma pocket'inda olmadigini; "
            "rejimin daha gecis karakteri tasidigini soyluyor."
        )

    if current_gamma > 0:
        summary_line = (
            f"Spot cevresindeki yerel gamma tonu pozitif ({current_gamma:+.2f}); bu, idealize edilmis hedge etkisinin "
            "hareketi bastirma yonune kaydigini ima ediyor."
        )
    elif current_gamma < 0:
        summary_line = (
            f"Spot cevresindeki yerel gamma tonu negatif ({current_gamma:+.2f}); bu, idealize edilmis hedge etkisinin "
            "hareketi buyutme yonune kayabildigini ima ediyor."
        )
    else:
        summary_line = "Spot cevresindeki yerel gamma tonu notr; hedge etkisi daha cok akis ve strike yogunluguna bagli."

    badges = [
        f"Mevcut proxy skor: {current_score:+.0f}",
        f"Rejim: {regime}",
        f"Dengeleyici pocket: {grid_spots[support_idx]:,.0f}",
        f"Kirilgan pocket: {grid_spots[fragile_idx]:,.0f}",
    ]
    if band_label != "Aktif long-gamma pocket disinda":
        badges.insert(2, f"Long-gamma alani: {band_label}")

    notes = [
        "Bu bolum true dealer position degil; strike OI, proxy pressure ve GEX'ten turetilen senaryo bazli bir hedge-impact egzersizidir.",
        "Pozitif skor, dealerlarin hareketi emmeye daha yakin oldugu long-gamma-benzeri bir pocket'i; negatif skor ise daha kirilgan short-gamma-benzeri davranisi ima eder.",
        f"En guclu stabilizing pocket {grid_spots[support_idx]:,.0f} civarinda, en kirilgan pocket ise {grid_spots[fragile_idx]:,.0f} civarinda gorunuyor.",
        summary_line,
    ]
    return {
        "grid_spots": grid_spots,
        "grid_scores": grid_scores,
        "spot": spot,
        "current_score": current_score,
        "regime": regime,
        "regime_class": regime_class,
        "band_label": band_label,
        "support_spot": grid_spots[support_idx],
        "fragile_spot": grid_spots[fragile_idx],
        "headline": headline,
        "summary": summary_line,
        "badges": badges,
        "notes": notes,
    }


def enrich_proxy_hedge_impact_history(section: Dict[str, object], archive_rows: List[Dict[str, str]]) -> Dict[str, object]:
    enriched = dict(section)
    history = [row for row in archive_rows if has_numeric(row.get("proxy_hedge_score", ""))]
    trend_rows = history[-14:]
    trend_dates = [row["date"] for row in trend_rows]
    trend_scores = [to_float(row.get("proxy_hedge_score", "0")) for row in trend_rows]
    enriched["trend_dates"] = trend_dates
    enriched["trend_scores"] = trend_scores
    enriched["history_label"] = ""
    enriched["history_badges"] = []
    enriched["history_summary"] = "Yerel arsiv buyudukce proxy hedge-impact rejiminin gunler icindeki kaymasi burada izlenir."
    enriched["history_notes"] = [
        "Bu trend true dealer pozisyonu degil; gunluk snapshot'lardan uretilen proxy hedge-impact skorunun zaman serisini gosterir.",
    ]
    if len(history) >= 2:
        prev_row = history[-2]
        prev_score = to_float(prev_row.get("proxy_hedge_score", "0"))
        prev_support = to_float(prev_row.get("proxy_support_spot", "0"))
        prev_fragile = to_float(prev_row.get("proxy_fragile_spot", "0"))
        score_delta = float(section.get("current_score", 0.0) or 0.0) - prev_score
        support_delta = float(section.get("support_spot", 0.0) or 0.0) - prev_support
        fragile_delta = float(section.get("fragile_spot", 0.0) or 0.0) - prev_fragile
        enriched["history_label"] = f"{prev_row['date']} -> {history[-1]['date']}"
        enriched["history_badges"] = [
            f"Skor degisimi: {score_delta:+.0f}",
            f"Dengeleyici pocket kaymasi: {support_delta:+.0f}",
            f"Kirilgan pocket kaymasi: {fragile_delta:+.0f}",
        ]
        if score_delta >= 75:
            enriched["history_summary"] = (
                "Proxy hedge-impact skoru dunden bugune belirgin guclendi; piyasa daha dengeleyici bir pocket'e kayiyor olabilir."
            )
        elif score_delta <= -75:
            enriched["history_summary"] = (
                "Proxy hedge-impact skoru dunden bugune zayifladi; hedge etkisi daha kirilgan bir rejime kayiyor olabilir."
            )
        else:
            enriched["history_summary"] = (
                "Proxy hedge-impact skoru dunden bugune sinirli degisti; temel rejim ayni kalsa da pocket seviyeleri hafif yer degistirmis olabilir."
            )
        enriched["history_notes"].append(
            f"Son gunluk kaymada dengeleyici pocket {support_delta:+.0f}, kirilgan pocket ise {fragile_delta:+.0f} puan yer degistirdi."
        )
    return enriched


def first_valid_iv_row(rows: List[Dict[str, float]], fallback_atm: float, fallback_maturity: str) -> Dict[str, float]:
    for row in rows:
        if to_float(row.get("atm", 0.0)) > 0 and (
            abs(to_float(row.get("rr_25d", 0.0))) > 0
            or abs(to_float(row.get("bf_25d", 0.0))) > 0
            or abs(to_float(row.get("rr_10d", 0.0))) > 0
            or abs(to_float(row.get("bf_10d", 0.0))) > 0
        ):
            return row
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


def compute_behavioral_score_from_snapshots(
    spot: float,
    iv_rv_7d_spread: float,
    iv_table_rows: List[Dict[str, float]],
    top_instrument_rows: List[Dict],
    strike_change_rows: List[Dict],
) -> Optional[int]:
    if not top_instrument_rows or not strike_change_rows or not iv_table_rows:
        return None

    iv_term_rows = build_iv_term_points(iv_table_rows)
    front_iv_row = first_valid_iv_row(iv_term_rows, 0.0, "")
    front_rr_25d = float(front_iv_row.get("rr_25d", 0.0))

    top_adds = sorted(top_instrument_rows, key=lambda row: row["oi_change_notional"], reverse=True)[:8]
    add_puts = [row for row in top_adds if row["type"] == "P"]
    add_calls = [row for row in top_adds if row["type"] == "C"]
    dominant_add = add_puts[0] if add_puts else (add_calls[0] if add_calls else None)

    put_add_sum = sum(max(0.0, row["oi_change_notional"]) for row in top_instrument_rows if row["type"] == "P")
    call_add_sum = sum(max(0.0, row["oi_change_notional"]) for row in top_instrument_rows if row["type"] == "C")

    crowd_chase_raw = 0
    if put_add_sum > call_add_sum * 1.2:
        crowd_chase_raw -= 2
    elif call_add_sum > put_add_sum * 1.2:
        crowd_chase_raw += 2

    if dominant_add and dominant_add.get("strike"):
        add_strike = float(dominant_add["strike"])
        if dominant_add["type"] == "P" and add_strike < spot:
            crowd_chase_raw -= 2
        elif dominant_add["type"] == "C" and add_strike > spot:
            crowd_chase_raw += 2

    if iv_rv_7d_spread >= 8:
        crowd_chase_raw += 1 if call_add_sum >= put_add_sum else -1

    if front_rr_25d <= -6:
        crowd_chase_raw -= 1
    elif front_rr_25d >= 4:
        crowd_chase_raw += 1

    return int(clamp(crowd_chase_raw, -5, 5))


def archive_fieldnames() -> List[str]:
    return [
        "date",
        "generated_utc",
        "spot",
        "proxy_hedge_score",
        "proxy_support_spot",
        "proxy_fragile_spot",
        "pin_low",
        "pin_center",
        "pin_high",
        "max_pain",
        "spot_max_pain_gap",
        "total_net_gex",
        "next_day_maturity",
        "next_day_oi",
        "next_day_oi_change_usd",
        "next_day_share",
        "call_put_notional_ratio",
        "call_put_oi_ratio",
        "atm_iv",
        "front_atm_iv",
        "normalized_25d_skew_ratio",
        "rv_7d",
        "rv_30d",
        "iv_rv_7d_spread",
        "inventory_net",
        "crowd_chase_score",
        "rr_25d",
        "skew_25d",
        "tail_hedge",
        "put_hedge_low",
        "put_hedge_high",
        "magnet_strike",
        "call_cluster_high",
        "speculation",
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


def rewrite_archive(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows.sort(key=lambda item: item["date"])
    with ARCHIVE_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=archive_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def backfill_archive_metrics(rows: List[Dict[str, str]], currency: str, market: str) -> List[Dict[str, str]]:
    changed = False
    expiry_aggregate_snapshot_name = f"{currency.lower()}_oi_expiry_aggregate"
    expiry_market_snapshot_name = f"{currency.lower()}_oi_expiry_{market}"
    gex_snapshot_name = f"{currency.lower()}_gex_all_{market}"
    strike_snapshot_name = f"{currency.lower()}_oi_strike_all_{market}"
    top_instrument_snapshot_name = f"{currency.lower()}_top_instrument_oi_change_{market}_24h"
    strike_change_snapshot_name = f"{currency.lower()}_oi_net_change_all_{market}_24h"
    iv_table_snapshot_name = f"{currency.lower()}_iv_table_{market}"
    for row in rows:
        needs_pain = not has_numeric(row.get("max_pain", "")) or not has_numeric(row.get("spot_max_pain_gap", ""))
        needs_behavioral = not has_numeric(row.get("crowd_chase_score", ""))
        if not needs_pain:
            pass
        else:
            payload, _ = load_snapshot_for_date(strike_snapshot_name, row["date"])
            if payload:
                strike_rows = parse_strike_oi_rows(payload)
                if strike_rows:
                    max_pain = compute_max_pain(strike_rows)
                    spot = to_float(row.get("spot", "0"))
                    row["max_pain"] = f"{max_pain:.0f}"
                    row["spot_max_pain_gap"] = f"{(spot - max_pain):.2f}"
                    changed = True
        if needs_behavioral:
            top_payload, _ = load_snapshot_for_date(top_instrument_snapshot_name, row["date"])
            strike_payload, _ = load_snapshot_for_date(strike_change_snapshot_name, row["date"])
            iv_payload, _ = load_snapshot_for_date(iv_table_snapshot_name, row["date"])
            if top_payload and strike_payload and iv_payload:
                top_rows = parse_top_instrument_rows(top_payload)
                strike_rows = parse_oi_net_change_all_rows(strike_payload)
                iv_rows = parse_iv_table_rows(iv_payload)
                score = compute_behavioral_score_from_snapshots(
                    to_float(row.get("spot", "0")),
                    to_float(row.get("iv_rv_7d_spread", "0")),
                    iv_rows,
                    top_rows,
                    strike_rows,
                )
                if score is not None:
                    row["crowd_chase_score"] = str(int(score))
                    changed = True
        needs_proxy_hedge = not has_numeric(row.get("proxy_hedge_score", ""))
        if needs_proxy_hedge:
            expiry_payload, _ = load_snapshot_for_date(expiry_aggregate_snapshot_name, row["date"])
            expiry_rows = parse_expiry_rows(expiry_payload, "aggregate") if expiry_payload else []
            if not expiry_rows:
                expiry_payload, _ = load_snapshot_for_date(expiry_market_snapshot_name, row["date"])
                expiry_rows = parse_expiry_rows(expiry_payload, market) if expiry_payload else []
            gex_payload, _ = load_snapshot_for_date(gex_snapshot_name, row["date"])
            strike_payload, _ = load_snapshot_for_date(strike_snapshot_name, row["date"])
            strike_change_payload, _ = load_snapshot_for_date(strike_change_snapshot_name, row["date"])
            if expiry_rows and gex_payload and strike_payload and strike_change_payload:
                gex_rows = aggregate_gex_by_strike(gex_payload)
                strike_rows = parse_strike_oi_rows(strike_payload)
                strike_change_rows = parse_oi_net_change_all_rows(strike_change_payload)
                if gex_rows and strike_rows:
                    spot = infer_spot(expiry_rows)
                    chain_rows = build_chain_pressure_rows(strike_rows, gex_rows, strike_change_rows, spot)
                    proxy_section = build_proxy_hedge_impact_section({"chain_rows": chain_rows, "spot": spot})
                    row["proxy_hedge_score"] = f"{float(proxy_section.get('current_score', 0.0) or 0.0):.2f}"
                    row["proxy_support_spot"] = f"{float(proxy_section.get('support_spot', 0.0) or 0.0):.2f}"
                    row["proxy_fragile_spot"] = f"{float(proxy_section.get('fragile_spot', 0.0) or 0.0):.2f}"
                    changed = True
    return rewrite_archive(rows) if changed else rows


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


def percentile_nearest(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = min(max(q, 0.0), 1.0)
    idx = int(round(q * (len(ordered) - 1)))
    return ordered[idx]


def compute_iv_regime_stats(current_iv: float, archive_rows: List[Dict[str, str]]) -> Dict[str, float]:
    history = [to_float(row.get("front_atm_iv", "")) for row in archive_rows if has_numeric(row.get("front_atm_iv", ""))]
    if current_iv > 0:
        history.append(current_iv)
    history = [value for value in history if value > 0]
    if not history:
        return {"sample_size": 0.0, "iv_percentile": 0.0, "iv_rank": 0.0, "adj_iv_rank": 0.0}

    sample_size = len(history)
    lower_count = sum(1 for value in history if value < current_iv)
    iv_percentile = (lower_count / sample_size) * 100 if sample_size else 0.0

    low = min(history)
    high = max(history)
    iv_rank = ((current_iv - low) / (high - low) * 100) if high > low else 0.0

    p05 = percentile_nearest(history, 0.05)
    p95 = percentile_nearest(history, 0.95)
    clipped_low = p05
    clipped_high = p95 if p95 > p05 else high
    adj_iv_rank = ((current_iv - clipped_low) / (clipped_high - clipped_low) * 100) if clipped_high > clipped_low else iv_rank

    return {
        "sample_size": float(sample_size),
        "iv_percentile": max(0.0, min(100.0, iv_percentile)),
        "iv_rank": max(0.0, min(100.0, iv_rank)),
        "adj_iv_rank": max(0.0, min(100.0, adj_iv_rank)),
    }


def compute_pin_proximity_score(spot: float, pin_low: float, pin_center: float, pin_high: float) -> float:
    half_band = max((pin_high - pin_low) / 2.0, 1.0)
    return clamp(1.0 - (abs(spot - pin_center) / half_band), 0.0, 1.0)


def compute_passive_support_score(
    spot: float,
    pin_low: float,
    pin_center: float,
    pin_high: float,
    total_net_gex: float,
    next_day_share: float,
    front_total_oi_3d: float,
    total_visible_oi: float,
) -> float:
    gamma_component = clamp(max(total_net_gex, 0.0) / 2.0, 0.0, 1.0)
    near_expiry_share = (front_total_oi_3d / total_visible_oi) if total_visible_oi else next_day_share
    near_component = clamp(near_expiry_share / 0.12, 0.0, 1.0)
    pin_component = compute_pin_proximity_score(spot, pin_low, pin_center, pin_high)
    return round(100.0 * (0.45 * gamma_component + 0.35 * near_component + 0.20 * pin_component), 1)


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
    max_pain = compute_max_pain(strike_rows)
    spot_max_pain_gap = spot - max_pain
    total_net_gex = sum(row["net_gex"] for row in gex_rows)
    positive_nodes = sorted(gex_rows, key=lambda row: row["net_gex"], reverse=True)[:5]
    negative_nodes = sorted(gex_rows, key=lambda row: row["net_gex"])[:5]
    nearby_strikes = [row for row in strike_rows if abs(row["strike"] - spot) <= 15000]
    crowded = sorted(nearby_strikes, key=lambda row: row["total_notional"], reverse=True)[:8]
    call_crowded = sorted(nearby_strikes, key=lambda row: row["call_notional"], reverse=True)[:5]
    put_crowded = sorted(nearby_strikes, key=lambda row: row["put_notional"], reverse=True)[:5]
    gex_lookup = {float(row["strike"]): float(row["net_gex"]) for row in gex_rows}
    downside_zone_floor = max(round_strike_level(spot - 25000.0), round_strike_level(spot * 0.7))
    downside_put_candidates = sorted(
        [
            row for row in strike_rows
            if downside_zone_floor <= row["strike"] < spot and row["put_notional"] > 0
        ],
        key=lambda row: row["put_notional"],
        reverse=True,
    )
    upside_call_candidates = sorted(
        [row for row in strike_rows if row["strike"] >= spot and row["call_notional"] > 0],
        key=lambda row: row["call_notional"],
        reverse=True,
    )
    put_zone_rows = downside_put_candidates[:3]
    put_zone_strikes = sorted({round_strike_level(row["strike"]) for row in put_zone_rows}) if put_zone_rows else []
    put_hedge_low = put_zone_strikes[0] if put_zone_strikes else DEFAULT_BTC_POSITION_MAP_LEVELS["put_hedge_low"]
    put_hedge_high = put_zone_strikes[-1] if put_zone_strikes else DEFAULT_BTC_POSITION_MAP_LEVELS["put_hedge_high"]
    if put_hedge_low == put_hedge_high:
        put_hedge_low = max(0.0, put_hedge_high - 5000.0)

    def magnet_score(row: Dict[str, float]) -> float:
        strike = float(row["strike"])
        total_notional = float(row["total_notional"])
        if total_notional <= 0:
            return 0.0
        call_notional = float(row["call_notional"])
        put_notional = float(row["put_notional"])
        balance_ratio = max(0.0, 1.0 - abs(call_notional - put_notional) / total_notional)
        pin_proximity = max(0.0, 1.0 - min(abs(strike - pin_center), 12000.0) / 12000.0)
        spot_proximity = max(0.0, 1.0 - min(abs(strike - spot), 15000.0) / 15000.0)
        positive_gex = max(gex_lookup.get(float(strike), 0.0), 0.0)
        return (
            total_notional * (0.30 + 0.70 * balance_ratio) * (0.40 + 0.60 * pin_proximity) * (0.55 + 0.45 * spot_proximity)
            + positive_gex * 50_000_000
        )

    magnet_candidate_pool = sorted(
        [row for row in nearby_strikes if row["total_notional"] > 0],
        key=magnet_score,
        reverse=True,
    )
    magnet_strike = (
        round_strike_level(magnet_candidate_pool[0]["strike"])
        if magnet_candidate_pool
        else round_strike_level(pin_center or max_pain)
    )
    if magnet_strike < put_hedge_high:
        upper_balance_pool = [
            row for row in magnet_candidate_pool
            if float(row["strike"]) >= max(put_hedge_high, spot - 5000.0)
        ]
        if upper_balance_pool:
            magnet_strike = round_strike_level(upper_balance_pool[0]["strike"])
        else:
            magnet_strike = max(round_strike_level(pin_center or spot), put_hedge_high)

    call_cluster_ceiling = max(magnet_strike + 20000.0, round_strike_level(spot * 1.25))
    call_cluster_pool = [row for row in upside_call_candidates if row["strike"] <= call_cluster_ceiling]
    call_cluster_rows = (call_cluster_pool or upside_call_candidates)[:4]
    call_cluster_high = (
        max(round_strike_level(row["strike"]) for row in call_cluster_rows)
        if call_cluster_rows
        else DEFAULT_BTC_POSITION_MAP_LEVELS["call_cluster_high"]
    )
    if call_cluster_high < magnet_strike:
        call_cluster_high = magnet_strike

    speculation_ceiling = max(call_cluster_high + 20000.0, round_strike_level(spot * 1.45))
    speculation_rows = [
        row for row in upside_call_candidates
        if call_cluster_high < row["strike"] <= speculation_ceiling
    ]
    speculation = (
        max(round_strike_level(row["strike"]) for row in speculation_rows[:5])
        if speculation_rows
        else round_strike_level(max(call_cluster_high + 15000.0, spot * 1.45))
    )

    tail_floor = max(5000.0, round_strike_level(spot * 0.45))
    tail_rows = sorted(
        [
            row for row in strike_rows
            if tail_floor <= row["strike"] < put_hedge_low and row["put_notional"] > 0
        ],
        key=lambda row: row["put_notional"],
        reverse=True,
    )
    tail_hedge = (
        min(round_strike_level(row["strike"]) for row in tail_rows[:3])
        if tail_rows
        else max(5000.0, put_hedge_low - 10000.0)
    )
    clean_position_map_levels = {
        "tail_hedge": tail_hedge,
        "put_hedge_low": put_hedge_low,
        "put_hedge_high": put_hedge_high,
        "magnet_strike": magnet_strike,
        "call_cluster_high": call_cluster_high,
        "speculation": speculation,
    }
    put_zone_profile_rows = [
        row for row in strike_rows
        if put_hedge_low <= row["strike"] <= put_hedge_high and row["put_notional"] > 0
    ]
    call_zone_profile_rows = [
        row for row in strike_rows
        if magnet_strike <= row["strike"] <= call_cluster_high and row["call_notional"] > 0
    ]
    max_put_zone_notional = max((row["put_notional"] for row in put_zone_profile_rows), default=1.0)
    max_call_zone_notional = max((row["call_notional"] for row in call_zone_profile_rows), default=1.0)
    tail_profile_rows = [
        row for row in strike_rows
        if tail_hedge <= row["strike"] < put_hedge_low and row["put_notional"] > 0
    ]
    speculation_profile_rows = [
        row for row in strike_rows
        if call_cluster_high < row["strike"] <= speculation and row["call_notional"] > 0
    ]
    max_tail_notional = max((row["put_notional"] for row in tail_profile_rows), default=1.0)
    max_spec_notional = max((row["call_notional"] for row in speculation_profile_rows), default=1.0)
    magnet_total_notional = next(
        (
            row["total_notional"] + max(gex_lookup.get(float(row["strike"]), 0.0), 0.0) * 100_000_000
            for row in strike_rows
            if round_strike_level(row["strike"]) == magnet_strike
        ),
        0.0,
    )
    clean_position_map_profile = {
        "tail_zone": [
            {
                "strike": round_strike_level(row["strike"]),
                "strength": max(0.08, min(1.0, row["put_notional"] / max_tail_notional)),
                "bucket": bucket_strength(max(0.08, min(1.0, row["put_notional"] / max_tail_notional))),
            }
            for row in sorted(tail_profile_rows, key=lambda item: item["put_notional"], reverse=True)[:4]
        ],
        "put_zone": [
            {
                "strike": round_strike_level(row["strike"]),
                "strength": max(0.08, min(1.0, row["put_notional"] / max_put_zone_notional)),
                "bucket": bucket_strength(max(0.08, min(1.0, row["put_notional"] / max_put_zone_notional))),
            }
            for row in sorted(put_zone_profile_rows, key=lambda item: item["put_notional"], reverse=True)[:4]
        ],
        "call_zone": [
            {
                "strike": round_strike_level(row["strike"]),
                "strength": max(0.08, min(1.0, row["call_notional"] / max_call_zone_notional)),
                "bucket": bucket_strength(max(0.08, min(1.0, row["call_notional"] / max_call_zone_notional))),
            }
            for row in sorted(call_zone_profile_rows, key=lambda item: item["call_notional"], reverse=True)[:4]
        ],
        "speculation_zone": [
            {
                "strike": round_strike_level(row["strike"]),
                "strength": max(0.08, min(1.0, row["call_notional"] / max_spec_notional)),
                "bucket": bucket_strength(max(0.08, min(1.0, row["call_notional"] / max_spec_notional))),
            }
            for row in sorted(speculation_profile_rows, key=lambda item: item["call_notional"], reverse=True)[:4]
        ],
        "magnet_strength": max(
            0.2,
            min(
                1.0,
                magnet_total_notional / max((row["total_notional"] for row in nearby_strikes), default=1.0),
            ),
        ),
        "tail_anchor": (
            round_strike_level(max(tail_profile_rows, key=lambda row: row["put_notional"])["strike"])
            if tail_profile_rows
            else tail_hedge
        ),
        "put_anchor": (
            round_strike_level(max(put_zone_profile_rows, key=lambda row: row["put_notional"])["strike"])
            if put_zone_profile_rows
            else put_hedge_high
        ),
        "call_anchor": (
            round_strike_level(max(call_zone_profile_rows, key=lambda row: row["call_notional"])["strike"])
            if call_zone_profile_rows
            else call_cluster_high
        ),
        "spec_anchor": (
            round_strike_level(max(speculation_profile_rows, key=lambda row: row["call_notional"])["strike"])
            if speculation_profile_rows
            else speculation
        ),
    }
    chain_rows = build_chain_pressure_rows(strike_rows, gex_rows, strike_change_rows, spot)
    chain_focus = sorted(chain_rows, key=lambda row: abs(row["net_flow_mn"]) + abs(row["net_gex"]) + abs(row["net_notional_mn"]), reverse=True)[:14]
    upside_candidates = [
        row for row in chain_rows
        if row["net_notional_mn"] > 0 and (row["net_gex"] > 0 or row["net_flow_mn"] > 0)
    ]
    downside_candidates = [
        row for row in chain_rows
        if row["net_notional_mn"] < 0 and (row["net_gex"] < 0 or row["net_flow_mn"] >= 0)
    ]
    upside_focus = sorted(upside_candidates or chain_rows, key=lambda row: row["upside_pressure"], reverse=True)[:3]
    downside_focus = sorted(downside_candidates or chain_rows, key=lambda row: row["downside_pressure"], reverse=True)[:3]
    strongest_upside = ", ".join(f"{row['strike']:,.0f}" for row in upside_focus if row["upside_pressure"] > 0)
    strongest_downside = ", ".join(f"{row['strike']:,.0f}" for row in downside_focus if row["downside_pressure"] > 0)
    strongest_flow_row = max(chain_rows, key=lambda row: abs(row["net_flow_mn"])) if chain_rows else None
    mm_map_read = []
    if strongest_upside:
        mm_map_read.append(
            f"En guclu yukari merkezler {strongest_upside}; bu strike'larda call agirligi, pozitif gamma ve/veya taze akis birlikte yogunlasiyor."
        )
    if strongest_downside:
        mm_map_read.append(
            f"En guclu asagi hedge cepleri {strongest_downside}; bu alanlar downside korumasinin en yogun kaldigi bolgeler."
        )
    if strongest_flow_row:
        flow_side = "yukari" if strongest_flow_row["net_flow_mn"] >= 0 else "asagi"
        mm_map_read.append(
            f"Son 24 saatte en belirgin zincir kaymasi {strongest_flow_row['strike']:,.0f} strike'inda {flow_side} yone oldu ({strongest_flow_row['net_flow_mn']:+.1f}mn); bu seviye kisa vadede yeni ilgi merkezi gibi okunmali."
        )

    dated_expiry_rows = []
    for row in expiry_rows:
        dt = parse_maturity_date(row["maturity"])
        if dt is not None:
            dated_expiry_rows.append((dt, row))
    dated_expiry_rows.sort(key=lambda item: item[0])
    front_expiry = dated_expiry_rows[0][1] if dated_expiry_rows else expiry_rows[0]

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
    normalized_25d_skew_ratio = ((-front_rr_25d) / front_atm_iv) if front_atm_iv else 0.0
    front_rr_10d = float(front_iv_row.get("rr_10d", 0.0))
    front_bf_10d = float(front_iv_row.get("bf_10d", 0.0))
    rv_7d = float(realized_vols.get("rv_7d", 0.0))
    rv_14d = float(realized_vols.get("rv_14d", 0.0))
    rv_30d = float(realized_vols.get("rv_30d", 0.0))
    iv_rv_7d_spread = front_atm_iv - rv_7d if rv_7d else 0.0

    front_call_notional = float(front_expiry.get("call_notional", 0.0))
    front_put_notional = float(front_expiry.get("put_notional", 0.0))
    front_call_btc_oi = float(front_expiry.get("call_btc_oi", 0.0))
    front_put_btc_oi = float(front_expiry.get("put_btc_oi", 0.0))
    front_call_premium = float(front_expiry.get("call_premium", 0.0))
    front_put_premium = float(front_expiry.get("put_premium", 0.0))
    front_total_notional = front_call_notional + front_put_notional
    front_total_btc_oi = front_call_btc_oi + front_put_btc_oi
    front_put_notional_share = (front_put_notional / front_total_notional) if front_total_notional else 0.0
    front_put_premium_share = (front_put_premium / (front_call_premium + front_put_premium)) if (front_call_premium + front_put_premium) else 0.0
    front_put_oi_share = (front_put_btc_oi / front_total_btc_oi) if front_total_btc_oi else 0.0
    front_positioning_verdict = "On vade karisik."
    if front_put_notional_share >= 0.58:
        front_positioning_verdict = "On vade pozisyonlamasi put-agir."
    elif front_put_notional_share <= 0.42:
        front_positioning_verdict = "On vade pozisyonlamasi call-agir."

    front_premium_verdict = "Premium dagilimi dengeliye yakin."
    if front_put_premium_share >= 0.58:
        front_premium_verdict = "Premium okuması savunmaci; odemenin buyuk kismi put tarafinda."
    elif front_put_premium_share <= 0.42:
        front_premium_verdict = "Premium okuması yapici; odemenin buyuk kismi call tarafinda."

    if front_put_premium_share >= 0.58 and spot >= pin_low:
        front_post_expiry_implication = "Expiry sonrasi destek korunursa hedge primi cozulerek kisa vadeli rahatlama alani acilabilir."
    elif front_put_notional_share >= 0.58 and spot < pin_low:
        front_post_expiry_implication = "Spot zayif kalirsa expiry sonrasi bile downside baskisi hemen kaybolmayabilir."
    elif front_put_notional_share <= 0.42 and spot >= pin_center:
        front_post_expiry_implication = "Call-agir on vade ve yukarida kalan spot, expiry sonrasi yukari devam potansiyelini destekler."
    else:
        front_post_expiry_implication = "Expiry kapanisi tek basina yon vermez; asil okuma spotun kilit seviyelere gore nerede tutuldugudur."

    if front_put_notional_share >= 0.58 or front_put_premium_share >= 0.58:
        front_expiry_verdict = "On vade put-agir ve koruma maliyeti belirgin."
    elif front_put_notional_share <= 0.42 and front_put_premium_share <= 0.42:
        front_expiry_verdict = "On vade call-agir; upside ilgisi daha baskin."
    else:
        front_expiry_verdict = "On vade karisik; call/put dagilimi dengeliye yakin."
    front_expiry_notes = [
        f"En yakin vade {front_expiry['maturity']}; toplam notional {fmt_usd(front_total_notional)} ve toplam OI yaklasik {front_total_btc_oi:,.0f} BTC.",
        f"Call OI {front_call_btc_oi:,.0f} BTC, put OI {front_put_btc_oi:,.0f} BTC; put OI payi %{front_put_oi_share * 100:.1f}.",
        f"Call notional {fmt_usd(front_call_notional)}, put notional {fmt_usd(front_put_notional)}; put notional payi %{front_put_notional_share * 100:.1f}.",
        f"Call premium {fmt_usd(front_call_premium)}, put premium {fmt_usd(front_put_premium)}; put premium payi %{front_put_premium_share * 100:.1f}.",
    ]
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

    upside_acceptance_candidates = sorted(
        {
            float(row["strike"])
            for row in (call_crowded + positive_nodes)
            if float(row["strike"]) > float(pin_high)
        }
    )
    upside_acceptance_ref = (
        upside_acceptance_candidates[0]
        if upside_acceptance_candidates
        else float(call_crowded[0]["strike"])
    )

    behavioral_components = []
    crowd_chase_raw = 0
    put_call_imbalance_score = 0
    strike_distance_score = 0
    premium_stress_score = 0
    skew_stress_score = 0

    if put_add_sum > call_add_sum * 1.2:
        put_call_imbalance_score = -2
    elif call_add_sum > put_add_sum * 1.2:
        put_call_imbalance_score = 2
    crowd_chase_raw += put_call_imbalance_score
    behavioral_components.append({
        "label": "Put/Call dengesizligi",
        "score": put_call_imbalance_score,
        "note": (
            "Put eklenmesi call eklenmesinden belirgin guclu." if put_call_imbalance_score < 0 else
            "Call eklenmesi put eklenmesinden belirgin guclu." if put_call_imbalance_score > 0 else
            "Put ve call eklenmesi birbirine yakin."
        ),
    })

    if dominant_add and dominant_add.get('strike'):
        add_strike = float(dominant_add['strike'])
        if dominant_add['type'] == 'P' and add_strike < spot:
            strike_distance_score = -2
        elif dominant_add['type'] == 'C' and add_strike > spot:
            strike_distance_score = 2
    crowd_chase_raw += strike_distance_score
    behavioral_components.append({
        "label": "Spota gore strike konumu",
        "score": strike_distance_score,
        "note": (
            "Baskin yeni risk spotun altindaki put strike'larinda birikiyor." if strike_distance_score < 0 else
            "Baskin yeni risk spotun ustundeki call strike'larinda birikiyor." if strike_distance_score > 0 else
            "Baskin yeni risk spota gore asiri kovalanan bir strike'ta degil."
        ),
    })

    if iv_rv_7d_spread >= 8:
        premium_stress_score = 1 if call_add_sum >= put_add_sum else -1
    crowd_chase_raw += premium_stress_score
    behavioral_components.append({
        "label": "Premium stresi",
        "score": premium_stress_score,
        "note": (
            "Opsiyon primi pahali ve put tarafi panige daha yakin." if premium_stress_score < 0 else
            "Opsiyon primi pahali ve call tarafi kovalamaya daha yakin." if premium_stress_score > 0 else
            "IV-RV farki bu sinyali asiri guclendirmiyor."
        ),
    })

    if front_rr_25d <= -6:
        skew_stress_score = -1
    elif front_rr_25d >= 4:
        skew_stress_score = 1
    crowd_chase_raw += skew_stress_score
    behavioral_components.append({
        "label": "Skew stresi",
        "score": skew_stress_score,
        "note": (
            "Skew savunmaci ve downside kuyruk riskini pahali tutuyor." if skew_stress_score < 0 else
            "Skew yukari call talebini destekliyor." if skew_stress_score > 0 else
            "Skew tarafinda asiri tek yonlu baski yok."
        ),
    })

    strategy_rows = sorted(strategy_rows, key=lambda row: row["total_notional"], reverse=True)
    top_strategies = strategy_rows[:8]
    strategy_totals = defaultdict(float)
    for row in top_strategies:
        strategy_totals[classify_strategy_bucket(row["strategy"])] += row["total_notional"]
    strategy_bias = max(strategy_totals.items(), key=lambda item: item[1])[0] if strategy_totals else "mixed"

    crowd_chase_score = int(clamp(crowd_chase_raw, -5, 5))

    if crowd_chase_score <= -3:
        behavioral_label = "Retail-benzeri gec put panigi"
        fade_or_follow = "Spot kirilim seviyesinin altinda kabul gormeden asagi panigi fade et"
        predictive_read = (
            f"Kisa vadeli put talebi dususun arkasindan geliyor gibi duruyor. Bu akisin buyuk kismi gec hedge ise, {pin_low:,.0f} alti kalici acceptance gelmeden asagi follow-through sinyali tek basina guclu sayilmaz."
        )
        behavioral_confidence = "Orta"
    elif crowd_chase_score >= 3:
        behavioral_label = "Retail-benzeri gec call kovalamasi"
        fade_or_follow = "Spot squeeze seviyesi ustunde kabul gormeden yukari kovalamayi fade et"
        predictive_read = (
            f"Yukari call akisi fiyatin arkasindan kosuyor olabilir. {upside_acceptance_ref:,.0f} ustunde kalici acceptance ve yeni call eklenmesi gormeden bu akis daha cok gec momentum alimi gibi okunur."
        )
        behavioral_confidence = "Orta"
    elif flow_score <= -2 and front_rr_25d <= -4:
        behavioral_label = "Smart-money-benzeri savunmaci akis"
        fade_or_follow = "Kirilim seviyeleri gecerli kaldikca savunmaci konumlanmayi takip et"
        predictive_read = (
            f"Bu akis tamamen duygusal gorunmuyor; skew, flow ve strike dagilimi savunmaci hedge'in daha bilgili olabilecegini soyluyor. {pin_low:,.0f} alti kabul bu okumayi guclendirir."
        )
        behavioral_confidence = "Yuksek"
    elif flow_score >= 2 and strategy_bias in {'constructive', 'mixed'}:
        behavioral_label = "Smart-money-benzeri yapici akis"
        fade_or_follow = "Yukarida sadece temiz acceptance ile takip et"
        predictive_read = (
            f"Call akisi tek basina euforia gibi gorunmuyor; yine de predictive guc kazanmasi icin {upside_acceptance_ref:,.0f} ustunde acceptance ve devam eden call eklenmesi gerekli."
        )
        behavioral_confidence = "Orta"
    else:
        behavioral_label = "Notr / karisik akis"
        fade_or_follow = "Temiz bir davranissal edge yok"
        predictive_read = (
            "Akis karmasik. Bu nedenle davranissal tarafta guclu bir fade ya da follow sinyali yok; seviye teyidi sinyalin kendisinden daha onemli."
        )
        behavioral_confidence = "Dusuk"

    behavioral_bullets = [
        f"Kalabalik kovalama skoru {crowd_chase_score:+d}: {behavioral_label}.",
        f"Fade mi Follow mu?: {fade_or_follow}.",
        predictive_read,
    ]
    behavioral_why = [
        f"{item['label']} ({item['score']:+d}): {item['note']}" for item in behavioral_components
    ]

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

    strategy_lookup = {row["strategy"]: row for row in top_strategies}
    strategy_flow_headline = "Strategy tape karisik; net tek yonlu aksiyon yerine hedge, cap ve secici yukari risk ayni anda gorunuyor."
    strategy_flow_interpretation: List[str] = []
    strategy_flow_flags: List[str] = []

    put_calendar = strategy_lookup.get("PUT_CALENDAR")
    long_put = strategy_lookup.get("LONG_PUT")
    short_call = strategy_lookup.get("SHORT_CALL")
    long_call = strategy_lookup.get("LONG_CALL")
    short_put = strategy_lookup.get("SHORT_PUT")
    bull_put = strategy_lookup.get("BULL_PUT_SPREAD")
    ratio_call = strategy_lookup.get("SHORT_RATIO_CALL_SPREAD")

    if put_calendar and put_calendar["oi_change"] > 0:
        strategy_flow_flags.append("roll")
        strategy_flow_interpretation.append(
            f"PUT_CALENDAR {fmt_usd(put_calendar['total_notional'])} ve {put_calendar['oi_change']:,.0f} OI artisi ile one cikiyor; bu, downside korumanin sadece kapatilmasindan cok vade boyunca tasindigi / uzatildigi bir tape'e daha yakin."
        )
    if long_put and long_put["oi_change"] > 0:
        strategy_flow_flags.append("fresh_hedge")
        strategy_flow_interpretation.append(
            f"LONG_PUT tarafinda {fmt_usd(long_put['total_notional'])} ve {long_put['oi_change']:,.0f} OI artisi var; bu, asagi yonlu korumanin taze hedge olarak eklendigini gosteriyor."
        )
    if short_call:
        if short_call["oi_change"] < 0:
            strategy_flow_flags.append("unwind_cap")
            strategy_flow_interpretation.append(
                f"SHORT_CALL halen buyuk ({fmt_usd(short_call['total_notional'])}) ama OI degisimi {short_call['oi_change']:,.0f}; bu, mevcut upside cap'lerin bir kisminin kapatildigina / gevsetildigine isaret ediyor."
            )
        elif short_call["oi_change"] > 0:
            strategy_flow_flags.append("cap")
            strategy_flow_interpretation.append(
                f"SHORT_CALL OI artisi {short_call['oi_change']:,.0f}; bu, yukari hareketin call satarak cap'lendigini ve temiz breakout beklentisinin sinirli kaldigini gosteriyor."
            )
    if long_call and long_call["oi_change"] > 0:
        strategy_flow_flags.append("re_risk")
        strategy_flow_interpretation.append(
            f"LONG_CALL OI artisi {long_call['oi_change']:,.0f}; tape'in bir parcasi yukari risk tekrar insa ediyor, ancak bu tek basina euforia okumasi degil."
        )
    if short_put:
        if short_put["oi_change"] < 0:
            strategy_flow_flags.append("less_put_sell")
            strategy_flow_interpretation.append(
                f"SHORT_PUT OI degisimi {short_put['oi_change']:,.0f}; put satarak risk alma istegi artmiyor, aksine bir miktar geri cekiliyor."
            )
        elif short_put["oi_change"] > 0:
            strategy_flow_flags.append("put_sell")
            strategy_flow_interpretation.append(
                f"SHORT_PUT OI artisi {short_put['oi_change']:,.0f}; oyuncularin bir kismi downside'i satip carry toplamaya devam ediyor."
            )
    if bull_put and bull_put["oi_change"] > 0:
        strategy_flow_flags.append("defined_bull")
        strategy_flow_interpretation.append(
            f"BULL_PUT_SPREAD OI artisi {bull_put['oi_change']:,.0f}; yonlu iyimserlik var ama sinirli riskli / defined-risk yapilar tercih ediliyor."
        )
    if ratio_call and ratio_call["oi_change"] > 0:
        strategy_flow_flags.append("capped_upside")
        strategy_flow_interpretation.append(
            f"SHORT_RATIO_CALL_SPREAD OI artisi {ratio_call['oi_change']:,.0f}; yukari senaryo tamamen reddedilmiyor ama extension daha capped / secici yapilarla trade ediliyor."
        )

    if "roll" in strategy_flow_flags and "fresh_hedge" in strategy_flow_flags:
        strategy_flow_headline = "Tape, downside korumanin hem tazelendigini hem de daha ileri vadeye tasindigini gosteriyor."
    elif "re_risk" in strategy_flow_flags and "unwind_cap" in strategy_flow_flags:
        strategy_flow_headline = "Tape, yukari tarafta cap'lerin bir miktar gevsedigini ve secici re-risking oldugunu gosteriyor."
    elif "cap" in strategy_flow_flags and "fresh_hedge" in strategy_flow_flags:
        strategy_flow_headline = "Tape ayni anda hem downside hedge hem de upside cap anlatiyor; yon var ama rahat degil."
    elif "defined_bull" in strategy_flow_flags and "re_risk" in strategy_flow_flags:
        strategy_flow_headline = "Tape net euforia degil; defined-risk yapilarla kontrollu yukari risk alimi var."
    elif "fresh_hedge" in strategy_flow_flags:
        strategy_flow_headline = "Tape'in baskin mesaji taze downside hedge."
    elif "re_risk" in strategy_flow_flags:
        strategy_flow_headline = "Tape'in baskin mesaji secici yukari risk yeniden insasi."

    strategy_verdict = "Karisik"
    dominant_action = "Karisik aksiyon"
    trading_implication = "Tape tek basina trend empoze etmiyor; seviye teyidi olmadan yonu buyutme."
    if "roll" in strategy_flow_flags and "fresh_hedge" in strategy_flow_flags:
        strategy_verdict = "Karisik ama savunmaci egilimli"
        strategy_verdict_class = "delta-down"
        dominant_action = "Taze downside hedge + hedge roll-forward"
        trading_implication = "Bu tape outright panic degil ama rahat risk-on da degil; yukariyi secici, asagiyi ise korunmaci oku."
    elif "cap" in strategy_flow_flags and "fresh_hedge" in strategy_flow_flags:
        strategy_verdict = "Savunmaci"
        strategy_verdict_class = "delta-down"
        dominant_action = "Taze hedge + upside cap"
        trading_implication = "Breakout kalitesi zayiflar; yukari extension gorursen bile cap riski nedeniyle acceptance ara."
    elif "re_risk" in strategy_flow_flags and "unwind_cap" in strategy_flow_flags:
        strategy_verdict = "Karisik ama yapici egilimli"
        strategy_verdict_class = "delta-neutral"
        dominant_action = "Cap unwind + secici re-risking"
        trading_implication = "Yukari ihtimal masada ama ancak acceptance ile guclenir; tape tek basina yeterli degil."
    elif "defined_bull" in strategy_flow_flags and "re_risk" in strategy_flow_flags:
        strategy_verdict = "Yapici"
        strategy_verdict_class = "delta-up"
        dominant_action = "Defined-risk bullish re-risking"
        trading_implication = "Oyuncular yukariyi kovaliyor degil, kontrollu sekilde insa ediyor; bu bullish ama euforik degil."
    elif "fresh_hedge" in strategy_flow_flags:
        strategy_verdict = "Savunmaci"
        strategy_verdict_class = "delta-down"
        dominant_action = "Taze downside hedge"
        trading_implication = "Tape korunma tarafina yatkin; asagi senaryoyu yok saymak icin veri yok."
    elif "re_risk" in strategy_flow_flags:
        strategy_verdict = "Yapici"
        strategy_verdict_class = "delta-up"
        dominant_action = "Secici upside re-risking"
        trading_implication = "Yukari hikaye terk edilmemis, ama follow-through icin seviye teyidi hala gerekli."
    else:
        strategy_verdict_class = "delta-neutral"

    max_strategy_notional = max((row["total_notional"] for row in top_strategies), default=1.0)
    strategy_action_rows: List[Dict[str, object]] = []
    if put_calendar and put_calendar["oi_change"] > 0:
        strategy_action_rows.append({
            "label": "Roll / uzatma",
            "score": min(1.0, put_calendar["total_notional"] / max_strategy_notional) * 100.0,
            "value_label": f"{fmt_usd(put_calendar['total_notional'])}",
            "kind": "roll",
        })
    if long_put and long_put["oi_change"] > 0:
        strategy_action_rows.append({
            "label": "Taze hedge",
            "score": min(1.0, long_put["total_notional"] / max_strategy_notional) * 100.0,
            "value_label": f"{fmt_usd(long_put['total_notional'])}",
            "kind": "hedge",
        })
    if short_call and abs(short_call["oi_change"]) > 0:
        strategy_action_rows.append({
            "label": "Cap / overwrite",
            "score": min(1.0, short_call["total_notional"] / max_strategy_notional) * 100.0,
            "value_label": f"{fmt_usd(short_call['total_notional'])}",
            "kind": "cap" if short_call["oi_change"] > 0 else "unwind",
        })
    if long_call and long_call["oi_change"] > 0:
        strategy_action_rows.append({
            "label": "Yukari risk yeniden insasi",
            "score": min(1.0, long_call["total_notional"] / max_strategy_notional) * 100.0,
            "value_label": f"{fmt_usd(long_call['total_notional'])}",
            "kind": "rerisk",
        })
    if bull_put and bull_put["oi_change"] > 0:
        strategy_action_rows.append({
            "label": "Defined-risk bullish",
            "score": min(1.0, bull_put["total_notional"] / max_strategy_notional) * 100.0,
            "value_label": f"{fmt_usd(bull_put['total_notional'])}",
            "kind": "defined_bull",
        })
    strategy_action_rows = sorted(strategy_action_rows, key=lambda row: row["score"], reverse=True)

    roll_transfer_headline = "Mevcut strategy endpoint'i leg-level veri vermedigi icin bu bolum tahmini transfer okumasidir."
    roll_transfer_notes: List[str] = []
    roll_transfer_rows: List[Dict[str, str]] = []
    add_put_instruments = sorted([row for row in top_adds if row["type"] == "P" and row.get("strike")], key=lambda row: row["oi_change_notional"], reverse=True)
    cut_put_instruments = sorted([row for row in top_cuts if row["type"] == "P" and row.get("strike")], key=lambda row: row["oi_change_notional"])
    add_call_instruments = sorted([row for row in top_adds if row["type"] == "C" and row.get("strike")], key=lambda row: row["oi_change_notional"], reverse=True)
    cut_call_instruments = sorted([row for row in top_cuts if row["type"] == "C" and row.get("strike")], key=lambda row: row["oi_change_notional"])

    if add_put_instruments and cut_put_instruments:
        add_put = add_put_instruments[0]
        cut_put = cut_put_instruments[0]
        if add_put["maturity"] != cut_put["maturity"]:
            roll_transfer_notes.append(
                f"Put tarafinda en guclu ekleme {add_put['instrument']}, en guclu bosalma {cut_put['instrument']}; bu, korumanin {cut_put['maturity']} vadesinden {add_put['maturity']} vadesine tasiniyor olabilecegini dusunduruyor."
            )
            roll_transfer_rows.append({
                "side": "Put",
                "source": f"{cut_put['maturity']} {cut_put['strike']:,.0f}P",
                "target": f"{add_put['maturity']} {add_put['strike']:,.0f}P",
                "caption": "Koruma vade boyunca tasiniyor olabilir",
            })
        if add_put["strike"] != cut_put["strike"]:
            strike_dir = "asagiya" if add_put["strike"] < cut_put["strike"] else "yukariya"
            roll_transfer_notes.append(
                f"Put strike transferi tahminen {cut_put['strike']:,.0f} -> {add_put['strike']:,.0f} yonunde {strike_dir}; bu da hedge seviyesinin yeniden konumlandigini ima eder."
            )

    if add_call_instruments and cut_call_instruments:
        add_call = add_call_instruments[0]
        cut_call = cut_call_instruments[0]
        if add_call["maturity"] != cut_call["maturity"]:
            roll_transfer_notes.append(
                f"Call tarafinda en guclu ekleme {add_call['instrument']}, en guclu bosalma {cut_call['instrument']}; bu, upside pozisyonunun {cut_call['maturity']} vadesinden {add_call['maturity']} vadesine aktariliyor olabilecegini gosteriyor."
            )
            roll_transfer_rows.append({
                "side": "Call",
                "source": f"{cut_call['maturity']} {cut_call['strike']:,.0f}C",
                "target": f"{add_call['maturity']} {add_call['strike']:,.0f}C",
                "caption": "Upside pozisyonu daha yakin / farkli vadeye tasiniyor olabilir",
            })
        if add_call["strike"] != cut_call["strike"]:
            strike_dir = "yukariya" if add_call["strike"] > cut_call["strike"] else "asagiya"
            roll_transfer_notes.append(
                f"Call strike transferi tahminen {cut_call['strike']:,.0f} -> {add_call['strike']:,.0f} yonunde {strike_dir}; bu, upside target/cap seviyesinin yer degistirdigine isaret edebilir."
            )

    if strike_adds and strike_cuts and strike_adds[0]["strike"] != strike_cuts[0]["strike"]:
        direction = "yukariya" if strike_adds[0]["strike"] > strike_cuts[0]["strike"] else "asagiya"
        roll_transfer_notes.append(
            f"Net strike akis merkezi {strike_cuts[0]['strike']:,.0f}'dan {strike_adds[0]['strike']:,.0f}'a {direction} kayiyor; bu da yeni ilginin hangi strike'a toplandigini gosteriyor."
        )
    if not roll_transfer_notes:
        roll_transfer_notes.append("Bugunku veri olasi roll/transfer seviyelerini guvenle ayristirmiyor; tape daha cok genel hedge ve secici yeniden konumlanma tonu veriyor.")

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
    upside_candidates = sorted(
        {
            float(row["strike"])
            for row in (call_crowded + positive_nodes)
            if float(row["strike"]) > float(pin_high)
        }
    )
    if not upside_candidates:
        upside_candidates = sorted({float(row["strike"]) for row in call_crowded}, reverse=True)
    upside_test_1 = upside_candidates[0]
    upside_test_2 = upside_candidates[1] if len(upside_candidates) > 1 else upside_test_1
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

    if inventory_net <= -1.0 or (flow_score < 0 and total_net_gex <= 0):
        directional_bias = "Notrden hafif negatife"
        directional_note = (
            "Yukari senaryo kosullu ve teyit istiyor; asagi taraf ise kirilma halinde daha kolay hizlanabilir."
        )
    elif inventory_net >= 1.5 and flow_score >= 1 and total_net_gex > 0:
        directional_bias = "Notrden hafif pozitife"
        directional_note = (
            "Asagi hareketler hedge akisiyla daha kolay bastirilabilir; temiz yukari devam icin yine de acceptance gerekli."
        )
    else:
        directional_bias = "Yon olarak notr ama kirilgan"
        directional_note = (
            "Piyasa net trend fiyatlamiyor; ancak simdilik yukari senaryo teyit isterken asagi taraf daha az teyitle calisabilir."
        )

    front_expected_first_effect = (
        f"Baz senaryo hafif relief; expiry sonrasi put primi cozulerek ilk tepki yukari olabilir, ancak bunun kalici bullish devama donmesi otomatik degil."
    )
    front_confirmation_needed = (
        f"Teyit sarti: spotun once {pin_center:,.0f}, ideal olarak {magnet_strike:,.0f} ustunde acceptance gormesi ve call akisinin dagilmamasi."
    )
    front_negative_trigger = (
        f"Negatif tetik: {pin_low:,.0f} alti kalicilik hedge cozulmesi yerine yeniden savunma ve downside baskisi uretebilir."
    )
    if front_put_premium_share >= 0.58 and (inventory_net <= -1.0 or spot < pin_low):
        front_expected_first_effect = (
            "Baz senaryo relief degil, savunmaci tonun surmesi; expiry gecse bile downside baskisi hemen kaybolmayabilir."
        )
        front_confirmation_needed = (
            f"Tersine donus teyidi icin spotun en az {pin_low:,.0f} ustune geri yerlesmesi ve flow tonunun savunmadan karisiga donmesi gerekir."
        )
        front_negative_trigger = (
            f"Negatif tetik: {pin_low:,.0f} alti yeni kabul, expiry sonrasi bile hedge baskisini hizlandirabilir."
        )
    elif front_put_notional_share <= 0.42 and front_put_premium_share <= 0.42 and inventory_net >= 1.0 and spot >= pin_center:
        front_expected_first_effect = (
            "Baz senaryo yukari devam; front expiry call-agir kaldigi icin expiry sonrasi cap/hedge temizlenmesi yukari uzamayi destekleyebilir."
        )
        front_confirmation_needed = (
            f"Teyit sarti: {magnet_strike:,.0f} ustunde kalicilik ve {pin_high:,.0f} civarinda reject yerine acceptance."
        )
        front_negative_trigger = (
            f"Negatif tetik: spot yeniden {pin_center:,.0f} altina inerse bu okuma relief'ten oteye tasinmaz."
        )
    elif front_put_premium_share >= 0.58:
        front_expected_first_effect = (
            "Baz senaryo ilk anda relief denemesi; ancak premium savunmaci kaldigi icin bu pencere teyitsiz kalirsa hareket kolayca tekrar range/savunma moduna donebilir."
        )
        front_confirmation_needed = (
            f"Teyit sarti: spotun {pin_center:,.0f} ustunde kalmasi ve yukari testlerde yeni savunmaci akisin geri gelmemesi."
        )
        front_negative_trigger = (
            f"Negatif tetik: {pin_low:,.0f} alti kabul, expiry rahatlamasi beklemek yerine savunmaci tonun dogru okundugunu teyit eder."
        )

    asymmetry_note = (
        f"Yukari tarafta temiz acceptance icin en az {upside_test_1:,.0f} ustu gerekir; asagi tarafta ise {downside_test_1:,.0f} alti kirilma tek basina daha hizli hedge akisina donebilir."
    )
    change_mind_note = (
        f"Bu gorusum {upside_test_1:,.0f} ustunde kalici acceptance ve belirgin call eklenmesiyle daha pozitif; {downside_test_1:,.0f} alti kabul ile daha negatif olur."
    )
    if total_net_gex > 0 and not (pin_low <= spot <= pin_high):
        desk_regime = (
            "Kontrollu ama kirilgan. Genel gamma yapisi tampon uretmeye devam ediyor, ancak on vadedeki zayiflama nedeniyle denge bozulursa hareket hizlanabilir."
        )
    elif total_net_gex > 0:
        desk_regime = (
            "Kontrollu. Genel gamma yapisi dengeleyici kalmaya devam ediyor ve fiyat davranisi simdilik daha cok pin/range karakteri tasiyor."
        )
    else:
        desk_regime = (
            "Kirilgan. Gamma yapisi artik hareketi bastirmaktan cok buyutmeye daha yakin, bu yuzden fiyat davranisi daha yonlu olabilir."
        )
    desk_asymmetry = (
        f"Asagi yon daha kolay hizlanabilir. Yukari devam icin {upside_test_1:,.0f} ustu acceptance gerekirken, {downside_test_1:,.0f} alti kirilma hedge akislarini daha hizli buyutebilir."
    )
    desk_upside_confirmation = (
        f"{upside_test_1:,.0f} ustunde kalici acceptance ve belirgin yeni call eklenmesi."
    )
    desk_downside_trigger = (
        f"{downside_test_1:,.0f} alti kabul. Bu bolgenin altinda hareket emilen degil buyuyen tarafa donebilir."
    )
    if inventory_net <= -1.0 or (flow_score < 0 and total_net_gex <= 0):
        desk_one_liner = (
            "Opsiyon yapisi tamamen bearish degil, ama mevcut dagilim kisa vadede asagi kirilmanin yukari teyitten daha kolay calisabilecegini soyluyor."
        )
    elif inventory_net >= 1.5 and total_net_gex > 0:
        desk_one_liner = (
            "Opsiyon yapisi yapici kalmaya devam ediyor, ancak temiz yukari devam icin acceptance olmadan momentumun kalici oldugu soylenemez."
        )
    else:
        desk_one_liner = (
            "Opsiyon yapisi net trend gununden cok kontrollu fiyatlama anlatiyor; kirilma olursa asimetrinin hangi yone daha kolay calistigi asil belirleyici olacak."
        )
    if total_net_gex > 0 and not (pin_low <= spot <= pin_high):
        desk_regime_badge = "Rejim: Kontrollu ama kirilgan"
    elif total_net_gex > 0:
        desk_regime_badge = "Rejim: Kontrollu"
    else:
        desk_regime_badge = "Rejim: Kirilgan"
    desk_badges = [
        desk_regime_badge,
        f"Gecis noktasi: {magnet_strike:,.0f}",
        f"Yukari teyit: {upside_test_1:,.0f}+",
        f"Asagi tetik: {downside_test_1:,.0f}-",
    ]
    if "Asagi yon" in desk_asymmetry:
        desk_badges.insert(1, "Asimetri: Asagi hizlanma")
    else:
        desk_badges.insert(1, "Asimetri: Dengeli")

    squeeze_trigger = upside_test_1
    breakout_confirm = upside_test_2
    breakdown_trigger = downside_test_1
    acceleration_trigger = downside_test_2
    reject_upside = pin_high
    reject_downside = pin_low
    acceptance_rules = [
        f"Accept above {squeeze_trigger:,.0f}: spot bu seviye ustunde kalir ve yeni call akisiyla desteklenirse yukari test aktif olur.",
        f"Reject above {reject_upside:,.0f}: fiyat bu bolgeyi sadece igneleyip geri donerse ralliye guvenme; pin/range mantigi calisiyor demektir.",
        f"Accept below {pin_low:,.0f}: spot bu seviye altinda kalirsa downside hedge akisi kuvvetlenir ve {acceleration_trigger:,.0f} ikinci hedef olur.",
        f"Reject below {reject_downside:,.0f}: fiyat altina sarkip hizla geri aliniyorsa asagi kirilim degil stop-hunt / range return okunmali.",
    ]
    playbook = [
        f"Base case: {pin_low:,.0f}-{pin_high:,.0f} icinde pin/range. Ilk beklenen davranis mean reversion ve {pin_center:,.0f} merkezine geri cekilme.",
        f"Bull case: {squeeze_trigger:,.0f} ustunde kabul gorulurse yukari test {breakout_confirm:,.0f} seviyesine uzayabilir. Bu senaryo kosullu; temiz trend icin yeni call eklenmesi ve flow tonunun belirgin iyilesmesi lazim.",
        f"Bear case: {pin_low:,.0f} alti kabul downside hedge akislarini hizlandirir; {acceleration_trigger:,.0f} civari ikinci hizlanma bolgesi. Asagi taraf simdilik daha az teyitle aktive olabilir.",
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
    gamma_map_read = build_gamma_map_read(
        {
            "spot": spot,
            "pin_low": pin_low,
            "pin_center": pin_center,
            "pin_high": pin_high,
            "total_net_gex": total_net_gex,
            "positive_nodes": positive_nodes,
            "negative_nodes": negative_nodes,
        }
    )

    return {
        "spot": spot,
        "pin_low": pin_low,
        "pin_center": pin_center,
        "pin_high": pin_high,
        "max_pain": max_pain,
        "spot_max_pain_gap": spot_max_pain_gap,
        "base_case": base_case,
        "alt_case": alt_case,
        "invalidation": invalidation,
        "dealer": dealer,
        "crowding": crowding,
        "pricing": pricing,
        "blind_spots": blind_spots,
        "simple_summary": simple_summary,
        "gamma_map_read": gamma_map_read,
        "positive_nodes": positive_nodes,
        "negative_nodes": negative_nodes,
        "crowded": crowded,
        "call_crowded": call_crowded,
        "put_crowded": put_crowded,
        "clean_position_map_levels": clean_position_map_levels,
        "clean_position_map_profile": clean_position_map_profile,
        "chain_rows": chain_rows,
        "chain_focus": chain_focus,
        "mm_map_read": mm_map_read,
        "next_day": next_day,
        "front_expiry": front_expiry,
        "front_expiry_verdict": front_expiry_verdict,
        "front_positioning_verdict": front_positioning_verdict,
        "front_premium_verdict": front_premium_verdict,
        "front_post_expiry_implication": front_post_expiry_implication,
        "front_expected_first_effect": front_expected_first_effect,
        "front_confirmation_needed": front_confirmation_needed,
        "front_negative_trigger": front_negative_trigger,
        "front_expiry_notes": front_expiry_notes,
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
        "normalized_25d_skew_ratio": normalized_25d_skew_ratio,
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
        "crowd_chase_score": crowd_chase_score,
        "behavioral_label": behavioral_label,
        "fade_or_follow": fade_or_follow,
        "predictive_read": predictive_read,
        "behavioral_confidence": behavioral_confidence,
        "behavioral_bullets": behavioral_bullets,
        "behavioral_components": behavioral_components,
        "behavioral_why": behavioral_why,
        "iv_term_rows": iv_term_rows,
        "vol_summary": vol_summary,
        "strategy_rows": top_strategies,
        "strategy_bias": strategy_bias,
        "strategy_summary": strategy_summary,
        "strategy_flow_headline": strategy_flow_headline,
        "strategy_flow_interpretation": strategy_flow_interpretation,
        "strategy_verdict": strategy_verdict,
        "strategy_verdict_class": strategy_verdict_class,
        "strategy_dominant_action": dominant_action,
        "strategy_trading_implication": trading_implication,
        "strategy_action_rows": strategy_action_rows,
        "roll_transfer_headline": roll_transfer_headline,
        "roll_transfer_notes": roll_transfer_notes,
        "roll_transfer_rows": roll_transfer_rows,
        "final_read": final_read,
        "morning_note": morning_note,
        "key_takeaways": key_takeaways,
        "takeaways_structured": takeaways_structured,
        "playbook": playbook,
        "acceptance_rules": acceptance_rules,
        "gamma_regime": gamma_regime,
        "inventory_rows": inventory_rows,
        "inventory_net": inventory_net,
        "crowd_chase_score": crowd_chase_score,
        "inventory_label": inventory_label,
        "directional_bias": directional_bias,
        "directional_note": directional_note,
        "asymmetry_note": asymmetry_note,
        "change_mind_note": change_mind_note,
        "desk_regime": desk_regime,
        "desk_asymmetry": desk_asymmetry,
        "desk_upside_confirmation": desk_upside_confirmation,
        "desk_downside_trigger": desk_downside_trigger,
        "desk_one_liner": desk_one_liner,
        "desk_badges": desk_badges,
    }


def build_archive_row(summary: Dict[str, object], proxy_hedge_impact_section: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    today = UTC_NOW().strftime("%Y-%m-%d")
    proxy_hedge_impact_section = proxy_hedge_impact_section or {}
    return {
        "date": today,
        "generated_utc": UTC_NOW().isoformat(),
        "spot": round(summary["spot"], 2),
        "proxy_hedge_score": round(float(proxy_hedge_impact_section.get("current_score", 0.0) or 0.0), 2),
        "proxy_support_spot": round(float(proxy_hedge_impact_section.get("support_spot", 0.0) or 0.0), 2),
        "proxy_fragile_spot": round(float(proxy_hedge_impact_section.get("fragile_spot", 0.0) or 0.0), 2),
        "pin_low": int(summary["pin_low"]),
        "pin_center": int(summary["pin_center"]),
        "pin_high": int(summary["pin_high"]),
        "max_pain": int(summary["max_pain"]),
        "spot_max_pain_gap": round(summary["spot_max_pain_gap"], 2),
        "total_net_gex": round(summary["total_net_gex"], 4),
        "next_day_maturity": summary["next_day"]["maturity"],
        "next_day_oi": round(summary["next_day"]["open_interest"], 2),
        "next_day_oi_change_usd": round(summary["next_day_change"], 2),
        "next_day_share": round(summary["next_day_share"], 6),
        "call_put_notional_ratio": round(summary["notional_ratio"], 4),
        "call_put_oi_ratio": round(summary["oi_ratio"], 4),
        "atm_iv": round(summary["atm_iv"], 2),
        "front_atm_iv": round(summary["front_atm_iv"], 2),
        "normalized_25d_skew_ratio": round(summary["normalized_25d_skew_ratio"], 4),
        "rv_7d": round(summary["rv_7d"], 2),
        "rv_30d": round(summary["rv_30d"], 2),
        "iv_rv_7d_spread": round(summary["iv_rv_7d_spread"], 2),
        "inventory_net": round(summary["inventory_net"], 3),
        "crowd_chase_score": int(summary["crowd_chase_score"]),
        "rr_25d": round(summary["rr_25d"], 2),
        "skew_25d": round(summary["skew_25d"], 2),
        "tail_hedge": int(summary["clean_position_map_levels"]["tail_hedge"]),
        "put_hedge_low": int(summary["clean_position_map_levels"]["put_hedge_low"]),
        "put_hedge_high": int(summary["clean_position_map_levels"]["put_hedge_high"]),
        "magnet_strike": int(summary["clean_position_map_levels"]["magnet_strike"]),
        "call_cluster_high": int(summary["clean_position_map_levels"]["call_cluster_high"]),
        "speculation": int(summary["clean_position_map_levels"]["speculation"]),
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


def make_expiry_chart(rows: List[Dict], out_path: Path, lang: str = "tr") -> None:
    sample = sorted(rows, key=lambda row: row.get("open_interest", 0.0), reverse=True)[:8]
    labels = [row["maturity"] for row in sample]
    values = [row["open_interest"] / 1_000_000 for row in sample]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color="#1f4e79")
    plt.title(loc(lang, "BTC Opsiyon Acik Pozisyonu - Vade Bazli", "BTC Open Interest by Expiry"))
    plt.ylabel(loc(lang, "Acik Pozisyon (milyon USD)", "Open Interest (USD mn)"))
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_gex_chart(rows: List[Dict], spot: float, out_path: Path, lang: str = "tr") -> None:
    window = [row for row in rows if abs(row["strike"] - spot) <= 15000]
    labels = [str(row["strike"]) for row in window]
    values = [row["net_gex"] for row in window]
    colors = ["#2e8b57" if value >= 0 else "#b22222" for value in values]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, values, color=colors)
    plt.title(loc(lang, "BTC Net GEX - Strike Bazli", "BTC Net GEX by Strike"))
    plt.ylabel("Net GEX")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_strike_oi_chart(rows: List[Dict], spot: float, out_path: Path, lang: str = "tr") -> None:
    window = [row for row in rows if abs(row["strike"] - spot) <= 15000]
    labels = [str(row["strike"]) for row in window]
    call_values = [row["call_notional"] / 1_000_000 for row in window]
    put_values = [row["put_notional"] / 1_000_000 for row in window]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, call_values, color="#1f4e79", label="Call notional")
    plt.bar(labels, put_values, bottom=call_values, color="#c06c2b", label="Put notional")
    plt.title(loc(lang, "BTC Strike Bazli Acik Pozisyon - Spot Cevresi", "BTC Strike Open Interest Around Spot"))
    plt.ylabel(loc(lang, "Acik Pozisyon (milyon USD)", "Open Interest (USD mn)"))
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*constrained_layout not applied.*",
            category=UserWarning,
        )
        plt.savefig(out_path, dpi=160)
    plt.close()


def make_mm_position_map_chart(summary: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    spot = float(summary["spot"])
    strike_rows = summary.get("chain_rows", [])
    if not strike_rows:
        return
    window = [row for row in strike_rows if abs(row["strike"] - spot) <= 15000]
    if not window:
        return

    labels = [str(int(row["strike"])) for row in window]
    left_values = [-float(row["downside_pressure"]) for row in window]
    right_values = [float(row["upside_pressure"]) for row in window]

    plt.figure(figsize=(12, 7))
    plt.barh(labels, left_values, color="#c6a33a", alpha=0.92)
    plt.barh(labels, right_values, color="#7fb0f0", alpha=0.95)
    plt.axvline(0, color="#f4f0e6", linewidth=1.2)
    plt.title("MM Position Map v1")
    plt.xlabel(loc(lang, "Proxy pressure score", "Proxy pressure score"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_dealer_pressure_delta_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = section.get("delta_rows", [])
    if not rows:
        return

    labels = [str(int(row["strike"])) for row in rows]
    upside_values = [float(row["upside_delta"]) for row in rows]
    downside_values = [-float(row["downside_delta"]) for row in rows]

    fig_height = max(7.2, min(14.0, len(rows) * 0.34))
    fig, ax = plt.subplots(figsize=(12.4, fig_height))
    y_pos = list(range(len(rows)))
    ax.barh(y_pos, downside_values, color="#c0392b", alpha=0.88, label=loc(lang, "Asagi delta", "Downside delta"))
    ax.barh(y_pos, upside_values, color="#2f9e44", alpha=0.88, label=loc(lang, "Yukari delta", "Upside delta"))
    ax.axvline(0, color="#334155", linewidth=1.1)
    ax.set_yticks(y_pos)
    sparse_labels = [label if (int(float(label)) % 1000 == 0) else "" for label in labels]
    ax.set_yticklabels(sparse_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_title(loc(lang, "Dealer Pressure Delta (Gunluk)", "Dealer Pressure Delta (Daily)"))
    ax.set_xlabel(loc(lang, "Solda asagi, sagda yukari delta", "Left: downside delta, right: upside delta"))
    ax.grid(axis="x", alpha=0.18)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.10)
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_proxy_hedge_impact_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    grid_spots = section.get("grid_spots", [])
    grid_scores = section.get("grid_scores", [])
    if not grid_spots or not grid_scores:
        return

    current_score = float(section.get("current_score", 0.0) or 0.0)
    spot = float(section.get("spot", 0.0) or 0.0)
    support_spot = float(section.get("support_spot", grid_spots[0]))
    fragile_spot = float(section.get("fragile_spot", grid_spots[0]))
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    ax.plot(grid_spots, grid_scores, color="#123b52", linewidth=2.2)
    ax.fill_between(grid_spots, grid_scores, 0, where=[value >= 0 for value in grid_scores], color="#79b87d", alpha=0.30)
    ax.fill_between(grid_spots, grid_scores, 0, where=[value < 0 for value in grid_scores], color="#c96a5c", alpha=0.30)
    ax.axhline(0, color="#4b5563", linewidth=1.1)
    current_idx = min(range(len(grid_spots)), key=lambda idx: abs(grid_spots[idx] - spot)) if spot else min(range(len(grid_spots)), key=lambda idx: abs(grid_scores[idx] - current_score))
    current_spot = float(grid_spots[current_idx])
    ax.axvline(current_spot, color="#8f2e28" if current_score < 0 else "#21623a", linestyle="--", linewidth=1.4)
    ax.scatter([current_spot], [current_score], color="#8f2e28" if current_score < 0 else "#21623a", s=44, zorder=3)
    ax.scatter([support_spot], [max(grid_scores)], color="#21623a", s=28, zorder=3)
    ax.scatter([fragile_spot], [min(grid_scores)], color="#8f2e28", s=28, zorder=3)
    ax.text(support_spot, max(grid_scores), loc(lang, " stabilizing", " stabilizing"), va="bottom", ha="left", fontsize=8, color="#21623a")
    ax.text(fragile_spot, min(grid_scores), loc(lang, " fragile", " fragile"), va="top", ha="left", fontsize=8, color="#8f2e28")
    ax.set_title(loc(lang, "Proxy Hedge Impact Surface", "Proxy Hedge Impact Surface"))
    ax.set_xlabel(loc(lang, "Hipotetik spot seviyesi", "Hypothetical spot level"))
    ax.set_ylabel(loc(lang, "Proxy hedge-impact skoru", "Proxy hedge-impact score"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{int(value):,}"))
    ax.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_proxy_hedge_impact_trend_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    dates = section.get("trend_dates", [])
    scores = section.get("trend_scores", [])
    if not dates or not scores:
        return
    plt.figure(figsize=(9.6, 4.4))
    plt.plot(dates, scores, color="#0f4c5c", marker="o", linewidth=2.2)
    plt.axhline(0, color="#1b2228", linewidth=1)
    plt.axhline(75, color="#2f9e44", linewidth=1, linestyle="--", alpha=0.75)
    plt.axhline(-75, color="#c0392b", linewidth=1, linestyle="--", alpha=0.75)
    plt.title(loc(lang, "Proxy Hedge-Impact Trendi", "Proxy Hedge-Impact Trend"))
    plt.ylabel(loc(lang, "Proxy skor", "Proxy score"))
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_clean_options_position_map_chart(
    price_series: List[Dict[str, object]], out_path: Path, levels: Dict[str, float], profile: Dict[str, object], lang: str = "tr"
) -> None:
    if not price_series:
        return
    dates = [row["ts"] for row in price_series]
    closes = [float(row["close"]) for row in price_series]
    last_date = dates[-1]
    label_x = last_date - timedelta(days=35)

    tail_hedge = levels["tail_hedge"]
    put_low = levels["put_hedge_low"]
    put_high = levels["put_hedge_high"]
    magnet = levels["magnet_strike"]
    call_high = levels["call_cluster_high"]
    speculation = levels["speculation"]

    fig, ax = plt.subplots(figsize=(13.6, 8.2), facecolor="#0b0d12")
    ax.set_facecolor("#0b0d12")
    ax.plot(dates, closes, color="#60e5e5", linewidth=1.1, alpha=0.95)

    def draw_bucket_bands(rows, lower, upper, color_map, half_height):
        for row in rows:
            strike = float(row["strike"])
            bucket = row.get("bucket", "weak")
            color, alpha = color_map.get(bucket, color_map["weak"])
            ax.axhspan(
                max(lower, strike - half_height),
                min(upper, strike + half_height),
                color=color,
                alpha=alpha,
            )

    ax.axhspan(put_low, put_high, color="#801b1b", alpha=0.10)
    ax.axhspan(tail_hedge, put_low, color="#4e0d0d", alpha=0.05)
    draw_bucket_bands(
        profile.get("tail_zone", []),
        tail_hedge,
        put_low,
        {
            "strong": ("#a11d1d", 0.55),
            "medium": ("#7f1515", 0.36),
            "weak": ("#5f1010", 0.20),
        },
        1600,
    )
    draw_bucket_bands(
        profile.get("put_zone", []),
        put_low,
        put_high,
        {
            "strong": ("#ff3636", 0.62),
            "medium": ("#c72626", 0.42),
            "weak": ("#8f1f1f", 0.22),
        },
        1800,
    )
    ax.axhline(put_high, color="#ff4b4b", linewidth=2.0)
    ax.axhline(put_low, color="#ff4b4b", linewidth=2.0)
    magnet_strength = float(profile.get("magnet_strength", 0.4))
    ax.axhline(magnet, color="#ffb000", linewidth=1.8 + 1.2 * magnet_strength, alpha=0.85 + 0.15 * magnet_strength)
    ax.axhspan(magnet, call_high, color="#1c6b2b", alpha=0.07)
    draw_bucket_bands(
        profile.get("call_zone", []),
        magnet,
        call_high,
        {
            "strong": ("#49df6f", 0.56),
            "medium": ("#2e9b49", 0.36),
            "weak": ("#226f35", 0.20),
        },
        1800,
    )
    ax.axhline(call_high, color="#4fd26b", linewidth=2.0)
    ax.axhspan(call_high, speculation, color="#0d2f69", alpha=0.05)
    draw_bucket_bands(
        profile.get("speculation_zone", []),
        call_high,
        speculation,
        {
            "strong": ("#4a7fff", 0.56),
            "medium": ("#2563ff", 0.34),
            "weak": ("#1847b8", 0.18),
        },
        1500,
    )
    ax.axhline(speculation, color="#2563ff", linewidth=2.0)
    ax.axhline(tail_hedge, color="#9aa0a6", linewidth=1.4, alpha=0.8)

    label_style = dict(fontsize=10, va="bottom", fontweight="medium")
    ax.text(label_x, put_high + 600, f"{int(put_high)}-{int(put_low)}  {loc(lang, 'Put Hedge Zone', 'Put Hedge Zone')}", color="#ff5a5a", **label_style)
    ax.text(label_x, magnet + 550, f"{int(magnet)}  {loc(lang, 'Magnet Strike', 'Magnet Strike')}", color="#ffc331", fontsize=10.5, va="bottom", fontweight="bold")
    ax.text(label_x, call_high + 600, f"{int(call_high)}  {loc(lang, 'Call Positioning', 'Call Positioning')}", color="#5be07a", **label_style)
    ax.text(label_x, speculation + 600, f"{int(speculation)}  {loc(lang, 'Speculation Calls', 'Speculation Calls')}", color="#4d7cff", **label_style)
    ax.text(label_x, tail_hedge + 550, f"{int(tail_hedge)}  {loc(lang, 'Tail Hedge', 'Tail Hedge')}", color="#c2c7ce", **label_style)

    ax.set_title(
        loc(lang, "BTC Options Position Map (Clean)", "BTC Options Position Map (Clean)"),
        color="#f3f4f6",
        loc="left",
        fontsize=18,
        pad=12,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.97,
        f"Tail Hedge {int(tail_hedge):,} | Put Hedge {int(put_low):,}-{int(put_high):,} | Magnet {int(magnet):,} | Call Cluster {int(call_high):,} | Speculation {int(speculation):,}",
        transform=ax.transAxes,
        color="#cfd7df",
        fontsize=10,
        va="top",
    )
    ax.text(
        0.99,
        0.97,
        loc(lang, "Strong / Medium / Weak intensity bands", "Strong / Medium / Weak intensity bands"),
        transform=ax.transAxes,
        color="#97a3b3",
        fontsize=9,
        va="top",
        ha="right",
    )
    ax.grid(color="#1b2230", alpha=0.35, linewidth=0.8)
    ax.tick_params(colors="#cfd7df", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#3a4250")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{int(value):,}"))
    ax.set_ylim(min(min(closes), tail_hedge) * 0.82, max(max(closes), speculation) * 1.08)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close()


def make_archive_chart(rows: List[Dict[str, str]], out_path: Path, lang: str = "tr") -> None:
    if not rows:
        return
    window = rows[-14:]
    dates = [row["date"] for row in window]
    spot = [to_float(row["spot"]) for row in window]
    next_day_oi = [to_float(row["next_day_oi"]) / 1_000_000 for row in window]
    plt.figure(figsize=(10, 5))
    plt.plot(dates, spot, color="#1f4e79", marker="o", label="Spot")
    plt.plot(dates, next_day_oi, color="#bb7d2a", marker="o", label=loc(lang, "Ertesi gun OI (mn USD)", "Next-day OI (USD mn)"))
    plt.title(loc(lang, "Gunluk Trend Ozeti", "Daily Trend Summary"))
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_flow_change_chart(
    strike_adds: List[Dict], strike_cuts: List[Dict], out_path: Path, lang: str = "tr"
) -> None:
    rows = strike_adds[:5] + strike_cuts[:5]
    labels = [str(row["strike"]) for row in rows]
    values = [row["net_change_notional"] / 1_000_000 for row in rows]
    colors = ["#2e8b57" if value >= 0 else "#9f2b2b" for value in values]
    plt.figure(figsize=(11, 5))
    plt.barh(labels, values, color=colors)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title(loc(lang, "OI Akis Degisimi - En Guclu Strike Eklemeleri / Bosalmalari", "OI Flow Change - Strongest Strike Adds / Cuts"))
    plt.xlabel(loc(lang, "Net OI degisimi (milyon USD)", "Net OI change (USD mn)"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_flow_heatmap(
    strike_change_rows: List[Dict], spot: float, out_path: Path, lang: str = "tr"
) -> None:
    window = [row for row in strike_change_rows if abs(row["strike"] - spot) <= 15000]
    labels = [row["strike"] for row in window]
    matrix = [
        [row["call_change_notional"] / 1_000_000 for row in window],
        [row["put_change_notional"] / 1_000_000 for row in window],
    ]
    plt.figure(figsize=(12, 3.8))
    plt.imshow(matrix, aspect="auto", cmap="RdYlGn")
    plt.yticks([0, 1], [loc(lang, "Call akisi", "Call flow"), loc(lang, "Put akisi", "Put flow")])
    plt.xticks(range(len(labels)), [str(label) for label in labels], rotation=45, ha="right")
    plt.colorbar(label=loc(lang, "OI degisimi (milyon USD)", "OI change (USD mn)"))
    plt.title(loc(lang, "BTC OI Akis Heatmap - Spot Cevresi", "BTC OI Flow Heatmap Around Spot"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_iv_term_chart(atm_iv_ts: Dict[str, List[Dict[str, float]]], out_path: Path, lang: str = "tr") -> None:
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
    plt.title(loc(lang, "BTC ATM IV Term Structure", "BTC ATM IV Term Structure"))
    plt.ylabel("ATM IV")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_iv_rv_chart(summary: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
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
    plt.title(loc(lang, "IV ve Realized Vol Karsilastirmasi", "IV vs Realized Vol"))
    plt.ylabel("Vol (%)")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_normalized_skew_chart(rows: List[Dict[str, str]], out_path: Path, lang: str = "tr") -> None:
    if not rows:
        return
    filtered = [row for row in rows[-30:] if has_numeric(row.get("normalized_25d_skew_ratio", ""))]
    if not filtered:
        return
    dates = [row["date"] for row in filtered]
    values = [to_float(row.get("normalized_25d_skew_ratio", "0")) for row in filtered]
    plt.figure(figsize=(10, 4.8))
    plt.plot(dates, values, color="#8b5a2b", marker="o", linewidth=2)
    plt.axhline(0, color="#1b2228", linewidth=1)
    plt.title(loc(lang, "Normalize Edilmis 25D Skew", "Normalized 25D Skew"))
    plt.ylabel("(Put vol - Call vol) / ATM IV")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_strategy_chart(rows: List[Dict[str, float]], out_path: Path, lang: str = "tr") -> None:
    sample = rows[:8]
    labels = [row["strategy"].replace("_", " ")[:20] for row in sample]
    values = [row["total_notional"] / 1_000_000 for row in sample]
    colors = ["#2e8b57" if row["net_premium_usd"] < 0 else "#b5651d" for row in sample]
    plt.figure(figsize=(11, 5))
    plt.barh(labels, values, color=colors)
    plt.title("Top Options Strategies - Notional")
    plt.xlabel(loc(lang, "Toplam notional (milyon USD)", "Total notional (USD mn)"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_mm_inventory_chart(summary: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = summary["inventory_rows"]
    inventory_label_map_en = {
        "Gamma rejimi": "Gamma regime",
        "Flow baskisi": "Flow pressure",
        "Skew savunmasi": "Skew defense",
        "IV-RV gerilimi": "IV-RV tension",
        "Strategy tape": "Strategy tape",
        "Net proxy": "Net proxy",
    }
    labels = [
        inventory_label_map_en.get(row["label"], tr_to_en_text(row["label"])) if lang == "en" else row["label"]
        for row in rows
    ]
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


def make_inventory_trend_chart(rows: List[Dict[str, str]], out_path: Path, lang: str = "tr") -> None:
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
    plt.title(loc(lang, "Dealer Inventory Proxy Trendi", "Dealer Inventory Proxy Trend"))
    plt.ylabel(loc(lang, "Proxy skor", "Proxy score"))
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_behavioral_trend_chart(rows: List[Dict[str, str]], out_path: Path, lang: str = "tr") -> None:
    if not rows:
        return
    window = rows[-14:]
    filtered = [row for row in window if has_numeric(row.get("crowd_chase_score", ""))]
    if not filtered:
        return
    dates = [row["date"] for row in filtered]
    values = [to_float(row.get("crowd_chase_score", "0")) for row in filtered]
    plt.figure(figsize=(10, 4.8))
    plt.plot(dates, values, color="#7a3b2e", marker="o", linewidth=2)
    plt.axhline(0, color="#1b2228", linewidth=1)
    plt.axhline(3, color="#c8a646", linewidth=1, linestyle="--")
    plt.axhline(-3, color="#c8a646", linewidth=1, linestyle="--")
    plt.ylim(-5, 5)
    plt.title(loc(lang, "Davranissal Akis Skoru Trendi", "Behavioral Flow Score Trend"))
    plt.ylabel(loc(lang, "Skor", "Score"))
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_behavioral_components_chart(summary: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = summary.get("behavioral_components", [])
    if not rows:
        return
    behavioral_label_map_en = {
        "Put/Call dengesizligi": "Put/Call imbalance",
        "Spota gore strike konumu": "Strike location vs spot",
        "Premium stresi": "Premium stress",
        "Skew stresi": "Skew stress",
    }
    labels = [
        behavioral_label_map_en.get(row["label"], tr_to_en_text(row["label"])) if lang == "en" else row["label"]
        for row in rows
    ]
    values = [row["score"] for row in rows]
    display_values = [value if abs(value) > 1e-9 else 0.06 for value in values]
    colors = ["#2e8b57" if value > 0 else "#9f2b2b" if value < 0 else "#8d877c" for value in values]
    plt.figure(figsize=(8.4, 4.6))
    bars = plt.barh(labels, display_values, color=colors)
    plt.axvline(0, color="#1b2228", linewidth=1)
    plt.xlim(-2.5, 2.5)
    plt.title(loc(lang, "Davranissal Bilesenler", "Behavioral Components"))
    plt.xlabel(loc(lang, "Skor katkisi", "Score contribution"))
    for bar, value in zip(bars, values):
        x = value if abs(value) > 1e-9 else 0.08
        ha = "left" if x >= 0 else "right"
        offset = 0.06 if x >= 0 else -0.06
        plt.text(x + offset, bar.get_y() + bar.get_height() / 2, f"{value:+.0f}", va="center", ha=ha, fontsize=8, color="#3a342b")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()




def build_pain_section(summary: Dict[str, object], archive_rows: List[Dict[str, str]]) -> Dict[str, object]:
    prev = archive_rows[-2] if len(archive_rows) >= 2 else None
    pain_change = 0.0
    gap_change = 0.0
    if prev:
        pain_change = summary["max_pain"] - to_float(prev.get("max_pain", "0"))
        gap_change = summary["spot_max_pain_gap"] - to_float(prev.get("spot_max_pain_gap", "0"))
    week_ref = archive_rows[max(0, len(archive_rows) - 5)] if len(archive_rows) >= 5 else None
    week_change = summary["max_pain"] - to_float(week_ref.get("max_pain", "0")) if week_ref else 0.0
    notes = [
        f"Bugunun aggregate max pain seviyesi {summary['max_pain']:,.0f}.",
        f"Spot ile max pain arasi fark {summary['spot_max_pain_gap']:+,.0f} puan.",
    ]
    if prev:
        notes.append(f"Dunden bugune pain kaymasi {pain_change:+,.0f} puan; spot-pain fark degisimi {gap_change:+,.0f} puan.")
    if week_ref:
        notes.append(f"5 gunluk pain kaymasi {week_change:+,.0f} puan.")
    return {"pain_change": pain_change, "gap_change": gap_change, "week_change": week_change, "notes": notes}


def build_vanna_charm_proxy_section(summary: Dict[str, object], expiry_rows: List[Dict]) -> Dict[str, object]:
    today = UTC_NOW().date()
    total_oi = float(summary.get("total_visible_oi", 0.0))
    dated_rows = []
    for row in expiry_rows:
        dt = parse_maturity_date(row.get("maturity", ""))
        if not dt:
            continue
        dte = (dt - today).days
        if dte < 0:
            continue
        dated_rows.append({**row, "dte": dte, "is_quarterly": is_quarterly_maturity(row.get("maturity", ""))})

    oi_3d = sum(row["open_interest"] for row in dated_rows if row["dte"] <= 3)
    oi_7d = sum(row["open_interest"] for row in dated_rows if row["dte"] <= 7)
    quarterly_7d = sum(row["open_interest"] for row in dated_rows if row["dte"] <= 7 and row["is_quarterly"])
    near_3d_share = (oi_3d / total_oi) if total_oi else 0.0
    near_7d_share = (oi_7d / total_oi) if total_oi else 0.0
    quarterly_7d_share = (quarterly_7d / total_oi) if total_oi else 0.0

    front_dt = parse_maturity_date(summary["next_day"]["maturity"])
    front_dte = max(0, (front_dt - today).days) if front_dt else 0

    pin_width = max(float(summary["pin_high"]) - float(summary["pin_low"]), 5000.0)
    pin_proximity = max(0.0, 1.0 - (abs(float(summary["spot"]) - float(summary["pin_center"])) / pin_width))
    gamma_support = max(0.0, min(1.0, float(summary["inventory_net"]) / 4.0))
    vol_premium = max(0.0, min(1.0, float(summary["iv_rv_7d_spread"]) / 20.0))
    term_flatness = max(0.0, 1.0 - min(abs(float(summary["term_slope"])) / 8.0, 1.0))
    skew_drag = max(0.0, min(1.0, abs(float(summary["front_rr_25d"])) / 12.0))

    charm_proxy = (
        0.45 * min(1.0, near_3d_share / 0.18)
        + 0.20 * min(1.0, near_7d_share / 0.30)
        + 0.20 * gamma_support
        + 0.15 * pin_proximity
    ) * 100.0
    vanna_proxy = (
        0.35 * vol_premium
        + 0.25 * gamma_support
        + 0.20 * pin_proximity
        + 0.20 * term_flatness
    ) * 100.0
    compression_proxy = (
        0.40 * (charm_proxy / 100.0)
        + 0.35 * (vanna_proxy / 100.0)
        + 0.15 * pin_proximity
        - 0.10 * skew_drag
    ) * 100.0
    compression_proxy = max(0.0, min(100.0, compression_proxy))

    if compression_proxy >= 67:
        regime = "Yuksek compression / pinning baskisi"
    elif compression_proxy >= 45:
        regime = "Orta compression / kontrollu fiyatlama"
    else:
        regime = "Dusuk compression / daha serbest fiyat kesfi"

    if compression_proxy >= 70 and front_dte <= 3:
        takeaway_badge = "Peak into expiry"
        takeaway_class = "delta-up"
        takeaway = "Decay flow etkisi guclu; expiry'ye kadar vol compression ve pinning baskisi baz senaryo."
    elif compression_proxy >= 50 and front_dte <= 5:
        takeaway_badge = "Moderate"
        takeaway_class = "delta-neutral"
        takeaway = "Decay flow etkisi orta; expiry'ye kadar kontrollu fiyatlama ve vol compression hala olasi."
    elif compression_proxy >= 35:
        takeaway_badge = "Weak-Moderate"
        takeaway_class = "delta-neutral"
        takeaway = "Decay flow etkisi zayif-orta; tek basina pinning beklemek icin yeterli degil, seviye teyidi gerekli."
    else:
        takeaway_badge = "Weak"
        takeaway_class = "delta-down"
        takeaway = "Decay flow etkisi zayif; bu bolum tek basina vol compression beklemek icin yeterli sinyal vermiyor."

    quarterly_note = (
        f"Yakin 7 gundeki quarterly vade yogunlugu toplam OI'nin %{quarterly_7d_share * 100:.1f}'i."
        if quarterly_7d_share > 0
        else "Yakin 7 gunde belirgin quarterly expiry yogunlugu yok."
    )
    notes = [
        "Bu bolum dogrudan vanna/charm greeklerini hesaplamaz; yakin vade OI yogunlugu, pozitif gamma, pin merkezine yakinlik ve IV priminden turetilen bir proxy'dir.",
        f"Front expiry {summary['next_day']['maturity']} ve kalan sure yaklasik {front_dte} gun. 3 gun icinde vadesi dolacak OI payi %{near_3d_share * 100:.1f}, 7 gun icinde %{near_7d_share * 100:.1f}.",
        quarterly_note,
        f"Charm proxy {charm_proxy:.0f}: yakin vade OI yogunlugu {'yuksek' if charm_proxy >= 67 else 'orta' if charm_proxy >= 45 else 'dusuk'} ve dealer gamma ile birlikte spotta zaman-decay kaynakli pin baskisi {'artabilir' if charm_proxy >= 55 else 'sinirli'}.",
        f"Vanna proxy {vanna_proxy:.0f}: front IV'nin realized vola gore primi ({summary['iv_rv_7d_spread']:+.2f}) ve gamma tonu, vol geri cekilirse hedge akislarinin spota dengeleyici etki verme potansiyelini {'guclendiriyor' if vanna_proxy >= 55 else 'sinirli tutuyor'}.",
        f"Net okuma {regime}. Spot pin merkezine yakinlik skoru %{pin_proximity * 100:.0f}, gamma destegi %{gamma_support * 100:.0f}.",
    ]
    checklist = [
        f"Spot {summary['pin_low']:,.0f}-{summary['pin_high']:,.0f} bandinda kaldikca compression proxy daha anlamli calisir.",
        f"{summary['clean_position_map_levels']['magnet_strike']:,.0f} ustu acceptance gelirse pin baskisi zayiflar, proxy'nin yonlendiriciligi duser.",
        f"IV-RV spread hizla kapanirsa vanna bileseni zayiflar; {summary['iv_rv_7d_spread']:+.2f} seviyesinden geriye bak.",
    ]
    components = [
        ("Yakin 3g OI", near_3d_share * 100.0),
        ("7g expiry OI", near_7d_share * 100.0),
        ("Quarterly 7g", quarterly_7d_share * 100.0),
        ("Gamma destegi", gamma_support * 100.0),
        ("Pin yakinligi", pin_proximity * 100.0),
        ("IV primi", vol_premium * 100.0),
    ]
    return {
        "front_dte": front_dte,
        "near_3d_share": near_3d_share,
        "near_7d_share": near_7d_share,
        "quarterly_7d_share": quarterly_7d_share,
        "charm_proxy": charm_proxy,
        "vanna_proxy": vanna_proxy,
        "compression_proxy": compression_proxy,
        "regime": regime,
        "takeaway_badge": takeaway_badge,
        "takeaway_class": takeaway_class,
        "takeaway": takeaway,
        "notes": notes,
        "checklist": checklist,
        "components": components,
    }


def make_vanna_charm_proxy_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    components = section.get("components", [])
    if not components:
        return
    vanna_label_map_en = {
        "Yakin 3g OI": "Near 3d OI",
        "7g expiry OI": "7d expiry OI",
        "Quarterly 7g": "Quarterly 7d",
        "Gamma destegi": "Gamma support",
        "Pin yakinligi": "Pin proximity",
        "IV primi": "IV premium",
    }
    labels = [
        vanna_label_map_en.get(label, tr_to_en_text(label)) if lang == "en" else label
        for label, _ in components
    ]
    values = [value for _, value in components]
    colors = []
    for value in values:
        if value >= 67:
            colors.append("#2e8b57")
        elif value >= 45:
            colors.append("#c8a646")
        else:
            colors.append("#6f7a84")
    plt.figure(figsize=(8.6, 4.8))
    bars = plt.barh(labels, values, color=colors)
    plt.xlim(0, 100)
    plt.axvline(45, color="#c8a646", linestyle="--", linewidth=1)
    plt.axvline(67, color="#2e8b57", linestyle="--", linewidth=1)
    plt.title(loc(lang, "Vanna / Charm Proxy Bilesenleri", "Vanna / Charm Proxy Components"))
    plt.xlabel(loc(lang, "Skor / pay (%)", "Score / share (%)"))
    for bar, value in zip(bars, values):
        plt.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center", ha="left", fontsize=8, color="#3a342b")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_strategy_action_chart(summary: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = summary.get("strategy_action_rows", [])
    if not rows:
        return
    strategy_action_label_map_en = {
        "Roll / uzatma": "Roll / extend",
        "Taze hedge": "Fresh hedge",
        "Cap / overwrite": "Cap / overwrite",
        "Yukari risk yeniden insasi": "Upside risk rebuild",
        "Defined-risk bullish": "Defined-risk bullish",
        "Unwind / gevseme": "Unwind / easing",
    }
    labels = [
        strategy_action_label_map_en.get(row["label"], tr_to_en_text(row["label"])) if lang == "en" else row["label"]
        for row in rows
    ]
    values = [row["score"] for row in rows]
    colors = []
    for row in rows:
        if row["kind"] in {"hedge", "roll", "cap"}:
            colors.append("#9f2b2b")
        elif row["kind"] in {"rerisk", "defined_bull"}:
            colors.append("#2e8b57")
        else:
            colors.append("#8d877c")
    plt.figure(figsize=(8.4, 4.8))
    bars = plt.barh(labels, values, color=colors)
    plt.title("Strategy Action Map")
    plt.xlabel(loc(lang, "Goreli etki skoru", "Relative impact score"))
    plt.xlim(0, max(values + [1.0]) * 1.18)
    for bar, row in zip(bars, rows):
        plt.text(bar.get_width() + max(values) * 0.03, bar.get_y() + bar.get_height() / 2, row["value_label"], va="center", ha="left", fontsize=8, color="#3a342b")
    plt.text(0.98, 0.04, loc(lang, "Kirmizi: savunmaci / cap | Yesil: yapici", "Red: defensive / cap | Green: constructive"), transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=8, color="#6b6459")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_roll_transfer_chart(summary: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = summary.get("roll_transfer_rows", [])
    if not rows:
        return
    roll_caption_map_en = {
        "Koruma vade boyunca tasiniyor olabilir": "Protection may be moving further out the curve",
        "Upside pozisyonu daha yakin / farkli vadeye tasiniyor olabilir": "Upside positioning may be moving to a nearer / different expiry",
    }
    fig_h = 3.6 if len(rows) == 1 else max(4.2, 2.4 + 1.45 * len(rows))
    fig, ax = plt.subplots(figsize=(10.2, fig_h))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0f1728")
    ax.axis("off")
    y_positions = list(range(len(rows), 0, -1))
    for y in y_positions:
        ax.hlines(y, 0.11, 0.89, color="#21314d", linewidth=1, alpha=0.55)
    for y, row in zip(y_positions, rows):
        color = "#9f2b2b" if row["side"] == "Put" else "#2e8b57"
        badge_face = "#3a1620" if row["side"] == "Put" else "#123626"
        badge_edge = "#df5a5a" if row["side"] == "Put" else "#5bd08a"
        ax.text(
            0.08, y, (tr_to_en_text(row["side"]) if lang == "en" else row["side"]).upper(),
            va="center", ha="center", fontsize=10, fontweight="bold", color="#f7efe1",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_face, edgecolor=badge_edge, linewidth=1.2)
        )
        ax.scatter([0.22], [y], s=360, color=color, alpha=0.95, edgecolors="#f5efe3", linewidths=0.8)
        ax.scatter([0.80], [y], s=360, color=color, alpha=0.45, edgecolors="#f5efe3", linewidths=0.8)
        ax.annotate("", xy=(0.75, y), xytext=(0.27, y), arrowprops=dict(arrowstyle="->", lw=3.0, color=color))
        ax.text(
            0.22, y + 0.23, row["source"], va="bottom", ha="center", fontsize=9.5, color="#f5efe3",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#152238", edgecolor="#344766", linewidth=1.0)
        )
        ax.text(
            0.80, y + 0.23, row["target"], va="bottom", ha="center", fontsize=9.5, color="#f5efe3",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#152238", edgecolor="#344766", linewidth=1.0)
        )
        caption = row["caption"]
        if lang == "en":
            caption = roll_caption_map_en.get(caption, tr_to_en_text(caption))
        ax.text(0.51, y - 0.28, caption, va="top", ha="center", fontsize=8.5, color="#aeb8c7")
    header_y = len(rows) + (0.42 if len(rows) == 1 else 0.58)
    ax.text(0.22, header_y, loc(lang, "Kaynak", "Source"), ha="center", va="bottom", fontsize=10, color="#aeb8c7")
    ax.text(0.80, header_y, loc(lang, "Hedef", "Target"), ha="center", va="bottom", fontsize=10, color="#aeb8c7")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.6, len(rows) + (0.75 if len(rows) == 1 else 1.0))
    plt.title(loc(lang, "Tahmini Roll / Transfer Haritasi", "Estimated Roll / Transfer Map"), color="#f5efe3", pad=16, fontsize=16, fontweight="bold")
    if len(rows) == 1:
        ax.text(0.5, 0.88, loc(lang, "Bugun tek baskin transfer infer edildi", "A single dominant transfer was inferred today"), transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#94a0b2")
    ax.text(0.98, 0.03, loc(lang, "Kirmizi: hedge / savunma | Yesil: call yeniden konumlanma", "Red: hedge / defense | Green: call repositioning"), transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#94a0b2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_layered_trade_check_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = section.get("rows", [])
    if not rows:
        return
    labels = [str(row["label"]) for row in rows]
    long_scores = [int(row.get("long", 0)) for row in rows]
    short_scores = [-int(row.get("short", 0)) for row in rows]
    ypos = list(range(len(labels)))
    verdict = str(section.get("verdict", ""))
    long_total = int(section.get("long_score", 0))
    short_total = int(section.get("short_score", 0))

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.barh(ypos, short_scores, color="#b03a2e", alpha=0.88, label=loc(lang, "Short teyit", "Short confirmation"))
    ax.barh(ypos, long_scores, color="#2e8b57", alpha=0.88, label=loc(lang, "Long teyit", "Long confirmation"))
    ax.axvline(0, color="#2f2f2f", linewidth=1.2)
    ax.axvspan(-4.5, -2.5, color="#f6ddd7", alpha=0.45, zorder=0)
    ax.axvspan(2.5, 4.5, color="#ddefe3", alpha=0.45, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(loc(lang, "Katman skoru", "Layer score"))
    ax.set_title(loc(lang, "Katmanli Teyit Matrisi", "Layered Confirmation Matrix"))
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right")

    for y, value in enumerate(short_scores):
        if value:
            ax.text(value - 0.05, y, f"{value}", va="center", ha="right", fontsize=9, color="#5b1c15")
    for y, value in enumerate(long_scores):
        if value:
            ax.text(value + 0.05, y, f"+{value}", va="center", ha="left", fontsize=9, color="#1f5a37")

    limit = max(1, max(max(long_scores), max(abs(v) for v in short_scores))) + 0.5
    ax.set_xlim(-limit, limit)
    fig.text(
        0.5,
        0.98,
        verdict,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#1f1f1f",
    )
    fig.text(
        0.5,
        0.94,
        loc(
            lang,
            f"Long teyit {long_total} | Short teyit {short_total}",
            f"Long confirmation {long_total} | Short confirmation {short_total}",
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#4d4d4d",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_clean_position_map_notes(summary: Dict[str, object], levels: Dict[str, float]) -> List[str]:
    spot = float(summary["spot"])
    put_low = levels["put_hedge_low"]
    put_high = levels["put_hedge_high"]
    magnet = levels["magnet_strike"]
    call_high = levels["call_cluster_high"]
    speculation = levels["speculation"]
    tail = levels["tail_hedge"]
    profile = summary.get("clean_position_map_profile", {})
    tail_anchor = float(profile.get("tail_anchor", tail))
    put_anchor = float(profile.get("put_anchor", put_high))
    call_anchor = float(profile.get("call_anchor", call_high))
    spec_anchor = float(profile.get("spec_anchor", speculation))

    notes = [
        f"Put hedge bolgesi {put_low:,.0f}-{put_high:,.0f}; bu alan asagi yonlu korumanin toplandigi ana zone gibi okunmali. En guclu put cebi yaklasik {put_anchor:,.0f}.",
        f"Magnet strike {magnet:,.0f}; bu seviye en buyuk notional dugumu degil, daha cok denge / pin karakteri tasiyan strike olarak secilir. Spot bu seviyeye yaklastikca pin / geri cekilme etkisi daha anlamli hale gelebilir.",
        f"Call positioning bandi {magnet:,.0f}-{call_high:,.0f}; bu zonun ust tarafinda acceptance gelirse yukari uzama alani acilir. En guclu call yogunlugu {call_anchor:,.0f} cevresinde.",
    ]
    if put_anchor < magnet and put_anchor >= put_low:
        notes.append(
            f"Asagida {put_anchor:,.0f} civarinda daha guclu bir hedge duvari var; bu seviye magnet degil, savunma / protection anchor gibi okunmali."
        )
    if spot < put_high:
        notes.append(
            f"Spot su an put hedge zonuna yakin ({spot:,.0f}); bu da piyasanin hala savunmaci tarafta denge aradigini gosteriyor."
        )
    elif spot < magnet:
        notes.append(
            f"Spot {magnet:,.0f} altinda; yukari hareketin daha temiz pozitif okumaya donmesi icin once magnet strike geri alinmali."
        )
    elif spot <= call_high:
        notes.append(
            f"Spot call positioning bandinin icinde; bu alan trend kadar direncli / yapiskan fiyat davranisi da uretebilir."
        )
    else:
        notes.append(
            f"Spot {call_high:,.0f} ustunde; call positioning ust bolgesine tasinmis durumda, bundan sonra {speculation:,.0f} daha anlamli hedef olur."
        )
    notes.append(f"Speculation calls bandi {call_high:,.0f}-{speculation:,.0f}; bu alan daha uzak OTM yukari beklentiyi gosterir. En guclu spekulatif call strike'i {spec_anchor:,.0f}.")
    notes.append(f"Tail hedge {tail:,.0f}; bu seviye gunluk trade seviyesi degil, daha cok uzak kuyruk risk referansi. En guclu uzak kuyruk putu {tail_anchor:,.0f}.")
    return notes


def build_gamma_map_read(summary: Dict[str, object]) -> Dict[str, str]:
    spot = float(summary["spot"])
    pin_low = float(summary["pin_low"])
    pin_center = float(summary["pin_center"])
    pin_high = float(summary["pin_high"])
    total_net_gex = float(summary["total_net_gex"])
    positive_nodes = summary["positive_nodes"]
    negative_nodes = summary["negative_nodes"]

    top_positive = float(positive_nodes[0]["strike"]) if positive_nodes else pin_high
    second_positive = float(positive_nodes[1]["strike"]) if len(positive_nodes) > 1 else top_positive
    top_negative = float(negative_nodes[0]["strike"]) if negative_nodes else pin_low
    second_negative = float(negative_nodes[1]["strike"]) if len(negative_nodes) > 1 else top_negative

    if spot < pin_low:
        current_zone = f"Spot su an pozitif gamma denge bandinin altinda; {pin_low:,.0f} alti artik gecis / zayiflama bolgesi."
        current_behavior = f"Bu bolgede hareket emilmek yerine daha yonlu ve kirilgan olmaya baslar; {top_negative:,.0f} ve {second_negative:,.0f} civarindaki negatif gamma cepleri daha kolay aktive olabilir."
    elif spot <= pin_high:
        current_zone = f"Spot su an {pin_low:,.0f}-{pin_high:,.0f} pozitif gamma denge bandinin icinde."
        current_behavior = f"Bu rejimde temel davranis mean reversion, pin ve hareket bastirma. En dogal merkez {pin_center:,.0f}; ilk refleks genelde bu eksene geri cekilme olur."
    else:
        current_zone = f"Spot su an pozitif gamma bandinin ust tarafinda; ilk yukari acceptance alani {pin_high:,.0f} ustunde."
        current_behavior = f"Bu bolgede yukari devam denenebilir ama ilk call / gamma surtunmesi {top_positive:,.0f} ve {second_positive:,.0f} ceplerinde gelir; hareket temiz trendden cok yapiskan extension da uretebilir."

    positive_behavior = f"Pozitif gamma tarafinda ({pin_low:,.0f}-{top_positive:,.0f}) fiyat davranisi daha kontrollu, geri cekilmeye yatkin ve hedge akisiyla bastirilmaya acik."
    transition_zone = f"Gecis bolgesi {pin_low:,.0f} civari. Spot bu seviyenin ustunde kaldikca pin mantigi korunur; alti kabul gorurse piyasa hareketi emen degil, buyuten tarafa gecmeye baslar."
    negative_behavior = f"Negatif gamma / hedge hizlanma bolgeleri asagida {top_negative:,.0f} ve {second_negative:,.0f}. Bu cepler acilirse dealer akisi downside'i buyutebilir."
    if spot < pin_low:
        execution_short = f"Short aramasi icin ana bolge {pin_low:,.0f} alti acceptance. Yani fiyat sadece sarkmakla kalmayip bu seviye altinda tutulmali; o durumda {top_negative:,.0f} ve {second_negative:,.0f} ceplerine dogru hizlanma daha olasi."
        execution_long = f"Long ancak {pin_low:,.0f} geri alinir ve spot tekrar denge bandina yerlesirse aranir; aksi halde yukari denemeler daha cok relief / counter-trend kalir."
        execution_no_trade = f"{pin_low:,.0f} alti reject olup hizli geri alim gelirse bu bolge stop-hunt olabilir; o durumda short kovalamak yerine yeniden acceptance beklenmeli."
    elif spot <= pin_high:
        execution_short = f"Band icinde short ancak {pin_high:,.0f} civari reject veya {pin_low:,.0f} alti acceptance ile aranir. Band ortasinda yone kovalamak zayif trade olur."
        execution_long = f"Band icinde long, {pin_low:,.0f}-{pin_center:,.0f} ustunde tutunma ya da {pin_high:,.0f} ustu acceptance ile daha temiz hale gelir."
        execution_no_trade = f"Spot {pin_low:,.0f}-{pin_high:,.0f} araliginda sikişiyorsa ve akis karisiksa en zayif trade ortadan zorla yone girmektir."
    else:
        execution_short = f"Spot bandin ustundeyken short ancak {pin_high:,.0f} altina geri donus ve reject ile aranir; aksi halde pozitif gamma tarafinda erken fade etmek zayif kalabilir."
        execution_long = f"Long senaryosu spot {pin_high:,.0f} ustunde kabul gordukce ve {top_positive:,.0f} / {second_positive:,.0f} ceplerine tasinma surdukce daha temizdir."
        execution_no_trade = f"Spot bandin ustunde kabul goruyorsa sadece ilk extension'a fade etmek yerine call/gamma surtunmesinin gercekten devreye girdigini gormek gerekir."

    if total_net_gex > 0 and pin_low <= spot <= pin_high:
        base_case = f"Base case: gamma rejimi destekleyici. Spot {pin_low:,.0f} ustunde ve denge bandinin icinde kaldikca ana beklenti kontrollu trade, pin ve yukari denemelerde once {top_positive:,.0f} civarinda surtunme."
        regime_badge = "Pin / Stabilizing"
        regime_class = "delta-up"
    elif pin_low <= spot <= pin_high:
        base_case = f"Base case: net gamma destegi zayif ama spot halen denge bandinin icinde. Rejim tamamen bozulmus degil; fiyat daha kirilgan, fakat ilk okuma hala kontrollu / range benzeri davranis."
        regime_badge = "Transition / Fragile"
        regime_class = "delta-neutral"
    elif total_net_gex > 0 and spot >= pin_low:
        base_case = f"Base case: gamma hala tamamen bozulmus degil ama spot ana denge bandinin disinda. {pin_center:,.0f} civarina geri cekilme ve yeni acceptance arayisi daha olasi."
        regime_badge = "Transition / Fragile"
        regime_class = "delta-neutral"
    elif total_net_gex > 0:
        base_case = f"Base case: toplam gamma tamamen bozulmus degil ama spot zayif tarafta. {pin_low:,.0f} geri alinmadan pozitif gamma konforu geri gelmez."
        regime_badge = "Transition / Fragile"
        regime_class = "delta-neutral"
    else:
        base_case = f"Base case: net gamma destegi zayif ve spot artik denge bandinin altinda. Davranis daha yonlu ve tetik-bagli okunmali; {top_negative:,.0f} ve {second_negative:,.0f} cepleri daha hizli aktive olabilir."
        regime_badge = "Negative Gamma Risk Active"
        regime_class = "delta-down"


    return {
        "current_zone": current_zone,
        "current_behavior": current_behavior,
        "positive_behavior": positive_behavior,
        "transition_zone": transition_zone,
        "negative_behavior": negative_behavior,
        "execution_short": execution_short,
        "execution_long": execution_long,
        "execution_no_trade": execution_no_trade,
        "base_case": base_case,
        "regime_badge": regime_badge,
        "regime_class": regime_class,
    }


def build_clean_position_map_section(summary: Dict[str, object], archive_rows: List[Dict[str, str]]) -> Dict[str, object]:
    levels = summary["clean_position_map_levels"]
    spot = float(summary["spot"])
    prev = archive_rows[-2] if len(archive_rows) >= 2 else None
    prev_has_levels = bool(
        prev
        and has_numeric(prev.get("magnet_strike", ""))
        and has_numeric(prev.get("put_hedge_low", ""))
        and has_numeric(prev.get("call_cluster_high", ""))
    )
    deltas = {
        "tail_hedge": 0.0,
        "put_hedge_low": 0.0,
        "put_hedge_high": 0.0,
        "magnet_strike": 0.0,
        "call_cluster_high": 0.0,
        "speculation": 0.0,
    }
    notes: List[str] = []
    if prev and prev_has_levels:
        for key in deltas:
            deltas[key] = levels[key] - to_float(prev.get(key, "0"))
        if deltas["magnet_strike"]:
            notes.append(f"Magnet strike dunden bugune {deltas['magnet_strike']:+,.0f} puan kaydi.")
        else:
            notes.append(f"Magnet strike degismedi ve {levels['magnet_strike']:,.0f} seviyesinde kaldi.")
        if deltas["put_hedge_low"] or deltas["put_hedge_high"]:
            notes.append(
                f"Put hedge zone {levels['put_hedge_low']:,.0f}-{levels['put_hedge_high']:,.0f} bandina kaydi ({deltas['put_hedge_low']:+,.0f} / {deltas['put_hedge_high']:+,.0f})."
            )
        else:
            notes.append(
                f"Put hedge zone degismedi ve {levels['put_hedge_low']:,.0f}-{levels['put_hedge_high']:,.0f} bandinda kaldi."
            )
        if deltas["call_cluster_high"] or deltas["speculation"]:
            notes.append(
                f"Yukari call cluster / speculation tavanlari {levels['call_cluster_high']:,.0f} ve {levels['speculation']:,.0f}; gunluk kayma {deltas['call_cluster_high']:+,.0f} / {deltas['speculation']:+,.0f}."
            )
    else:
        notes.append("Position map seviyeleri bugun ilk kez yerel arsive yazildi; yarindan itibaren gunluk kayma notlari anlamli hale gelecek.")
    scenarios = [
        f"Magnet strike {levels['magnet_strike']:,.0f} geri alinir ve ustunde acceptance gelirse, piyasa savunmaci rejimden cikarak call positioning bandina dogru uzama sansi kazanir.",
        f"Fiyat put hedge zone {levels['put_hedge_low']:,.0f}-{levels['put_hedge_high']:,.0f} bandina girerse asil soru dokunmak degil, bu bolgede kabul gorup gormedigidir; kabul olursa hedge baskisi buyur, reject olursa tepki olasiligi artar.",
        f"Call positioning bandi {levels['magnet_strike']:,.0f}-{levels['call_cluster_high']:,.0f} icine gecis tek basina trend teyidi degildir; bu alan hedef kadar yapiskan / direncli fiyatlama da uretebilir. Temiz pozitif senaryo icin zone icinde kalicilik gerekir.",
        f"Tail hedge {levels['tail_hedge']:,.0f} gunluk trade seviyesi degil; daha cok normal hedge alaninin da kirildigi, kuyruk riskin gercekten fiyatlanmaya basladigi uzak stres referansidir.",
    ]
    if spot < levels["magnet_strike"]:
        scenarios.insert(
            0,
            f"Spot su an magnet strike'in altinda ({spot:,.0f}); bu nedenle ilk rejim degisimi sinyali {levels['magnet_strike']:,.0f} ustunde acceptance olur."
        )
    elif spot <= levels["call_cluster_high"]:
        scenarios.insert(
            0,
            f"Spot su an call positioning bandinin icinde ({spot:,.0f}); burada asıl soru devam mı yoksa yapiskan range mi uretilecegi."
        )
    else:
        scenarios.insert(
            0,
            f"Spot call positioning bandinin ustunde ({spot:,.0f}); bu durumda yukari extension fiyatlaniyor olabilir, ama speculation alanina yaklastikca geri verme riski de artar."
        )
    return {"levels": levels, "deltas": deltas, "notes": notes, "scenarios": scenarios}


def build_position_map_drift_section(summary: Dict[str, object], archive_rows: List[Dict[str, str]]) -> Dict[str, object]:
    filtered = [
        row for row in archive_rows[-20:]
        if has_numeric(row.get("magnet_strike", ""))
        and has_numeric(row.get("put_hedge_low", ""))
        and has_numeric(row.get("put_hedge_high", ""))
        and has_numeric(row.get("call_cluster_high", ""))
        and has_numeric(row.get("speculation", ""))
        and has_numeric(row.get("tail_hedge", ""))
        and has_numeric(row.get("spot", ""))
    ]
    if len(filtered) < 2:
        return {
            "headline": "Position map drift icin yeterli tarih yok.",
            "notes": ["Gunluk drift okumasini guvenli yapmak icin en az iki tarihli seviye gerekir."],
            "delta_metrics": [],
        }

    prev = filtered[-2]
    curr = filtered[-1]
    spot = float(summary["spot"])

    prev_spot = to_float(prev.get("spot", "0"))
    prev_magnet = to_float(prev["magnet_strike"])
    curr_magnet = to_float(curr["magnet_strike"])
    prev_put_low = to_float(prev["put_hedge_low"])
    prev_put_high = to_float(prev["put_hedge_high"])
    curr_put_low = to_float(curr["put_hedge_low"])
    curr_put_high = to_float(curr["put_hedge_high"])
    prev_call = to_float(prev["call_cluster_high"])
    curr_call = to_float(curr["call_cluster_high"])
    prev_spec = to_float(prev["speculation"])
    curr_spec = to_float(curr["speculation"])
    prev_tail = to_float(prev["tail_hedge"])
    curr_tail = to_float(curr["tail_hedge"])

    prev_put_width = prev_put_high - prev_put_low
    curr_put_width = curr_put_high - curr_put_low
    prev_put_gap = max(0.0, prev_spot - prev_put_high)
    curr_put_gap = max(0.0, spot - curr_put_high)
    prev_call_gap = max(0.0, prev_call - prev_spot)
    curr_call_gap = max(0.0, curr_call - spot)

    magnet_delta = curr_magnet - prev_magnet
    put_shift = (curr_put_low + curr_put_high) / 2 - (prev_put_low + prev_put_high) / 2
    put_width_delta = curr_put_width - prev_put_width
    call_delta = curr_call - prev_call
    spec_delta = curr_spec - prev_spec
    tail_delta = curr_tail - prev_tail

    headline_parts: List[str] = []
    if magnet_delta > 0:
        headline_parts.append("magnet yukari")
    elif magnet_delta < 0:
        headline_parts.append("magnet asagi")
    else:
        headline_parts.append("magnet sabit")
    if put_shift > 0:
        headline_parts.append("put savunma yaklasti")
    elif put_shift < 0:
        headline_parts.append("put savunma asagi kaydi")
    else:
        headline_parts.append("put zone sabit")
    if call_delta < 0:
        headline_parts.append("call cluster yaklasti")
    elif call_delta > 0:
        headline_parts.append("call cluster uzaklasti")
    else:
        headline_parts.append("call cluster sabit")
    headline = " | ".join(headline_parts)

    notes: List[str] = []
    if magnet_delta > 0:
        notes.append(f"Magnet strike {magnet_delta:+,.0f} puan yukari kaydi; denge / pin merkezi yukari tasiniyor.")
    elif magnet_delta < 0:
        notes.append(f"Magnet strike {magnet_delta:+,.0f} puan asagi kaydi; denge merkezi zayifladi.")
    else:
        notes.append(f"Magnet strike degismedi ve {curr_magnet:,.0f} seviyesinde kaldi.")

    if put_shift > 0:
        notes.append(f"Put hedge zone {put_shift:+,.0f} puan yukari kaydi; downside koruma spota daha yakinlasarak guclendi.")
    elif put_shift < 0:
        notes.append(f"Put hedge zone {put_shift:+,.0f} puan asagi kaydi; koruma daha derin strike'lara itildi.")
    else:
        notes.append(f"Put hedge zone merkezi ayni kaldi ({curr_put_low:,.0f}-{curr_put_high:,.0f}).")
    if put_width_delta > 0:
        notes.append(f"Put hedge bandi {put_width_delta:+,.0f} puan genisledi; koruma daha yaygin strike'lara dagiliyor.")
    elif put_width_delta < 0:
        notes.append(f"Put hedge bandi {put_width_delta:+,.0f} puan daraldi; downside ilgi daha dar bir cepte yogunlasiyor.")
    if curr_put_gap < prev_put_gap:
        notes.append(f"Put hedge zone spota yaklasti ({prev_put_gap:,.0f} -> {curr_put_gap:,.0f}); asagi savunma daha ilgili.")
    elif curr_put_gap > prev_put_gap:
        notes.append(f"Put hedge zone spotdan uzaklasti ({prev_put_gap:,.0f} -> {curr_put_gap:,.0f}); asagi savunma daha derinde kaliyor.")

    if call_delta < 0:
        notes.append(f"Call cluster {call_delta:+,.0f} puan asagi geldi; yakin upside positioning gucleniyor.")
    elif call_delta > 0:
        notes.append(f"Call cluster {call_delta:+,.0f} puan yukari tasindi; yukari hedef daha extension-tipi hale geliyor.")
    else:
        notes.append(f"Call cluster degismedi ve {curr_call:,.0f} seviyesinde kaldi.")
    if curr_call_gap < prev_call_gap:
        notes.append(f"Call cluster spota yaklasti ({prev_call_gap:,.0f} -> {curr_call_gap:,.0f}); yakin upside ilgisi artti.")
    elif curr_call_gap > prev_call_gap:
        notes.append(f"Call cluster spotdan uzaklasti ({prev_call_gap:,.0f} -> {curr_call_gap:,.0f}); yakin upside ilgisi zayifladi.")

    if spec_delta < 0:
        notes.append(f"Speculation tavani {spec_delta:+,.0f} puan asagi indi; uzak call iyimserligi daha yakin strike'lara cekiliyor.")
    elif spec_delta > 0:
        notes.append(f"Speculation tavani {spec_delta:+,.0f} puan yukari gitti; uzak OTM upside hala yasiyor.")
    else:
        notes.append(f"Speculation seviyesi sabit ve {curr_spec:,.0f}.")

    if tail_delta:
        notes.append(f"Tail hedge seviyesi {tail_delta:+,.0f} puan kaydi; uzak kuyruk risk referansi guncellendi.")
    else:
        notes.append(f"Tail hedge seviyesi degismedi ve {curr_tail:,.0f} olarak kaldi.")

    delta_metrics = [
        {"label": "Magnet", "value": magnet_delta, "kind": "neutral"},
        {"label": "Put merkez", "value": put_shift, "kind": "down"},
        {"label": "Put genislik", "value": put_width_delta, "kind": "down"},
        {"label": "Call cluster", "value": call_delta, "kind": "up"},
        {"label": "Speculation", "value": spec_delta, "kind": "up"},
        {"label": "Tail hedge", "value": tail_delta, "kind": "tail"},
        {"label": "Put spot mesafe", "value": prev_put_gap - curr_put_gap, "kind": "down"},
        {"label": "Call spot mesafe", "value": prev_call_gap - curr_call_gap, "kind": "up"},
    ]

    simple_notes = [
        notes[0],
        notes[1],
        notes[3] if len(notes) > 3 else "",
        notes[4] if len(notes) > 4 else "",
        notes[5] if len(notes) > 5 else "",
        notes[7] if len(notes) > 7 else "",
    ]
    simple_notes = [n for n in simple_notes if n]

    return {"headline": headline, "notes": simple_notes, "delta_metrics": delta_metrics}


def build_gamma_support_monitor_section(
    summary: Dict[str, object],
    archive_rows: List[Dict[str, str]],
    expiry_rows: List[Dict],
    currency: str = "BTC",
    primary_market: str = "deribit",
) -> Dict[str, object]:
    history_rows = [
        row for row in archive_rows[-30:]
        if has_numeric(row.get("spot", ""))
        and has_numeric(row.get("pin_low", ""))
        and has_numeric(row.get("pin_center", ""))
        and has_numeric(row.get("pin_high", ""))
        and has_numeric(row.get("total_net_gex", ""))
        and has_numeric(row.get("next_day_share", ""))
        and has_numeric(row.get("front_total_oi_3d", ""))
        and has_numeric(row.get("total_visible_oi", ""))
    ]
    history_points: List[Dict[str, object]] = []
    for row in history_rows:
        history_points.append(
            {
                "date": row["date"],
                "score": compute_passive_support_score(
                    to_float(row.get("spot", "0")),
                    to_float(row.get("pin_low", "0")),
                    to_float(row.get("pin_center", "0")),
                    to_float(row.get("pin_high", "0")),
                    to_float(row.get("total_net_gex", "0")),
                    to_float(row.get("next_day_share", "0")),
                    to_float(row.get("front_total_oi_3d", "0")),
                    to_float(row.get("total_visible_oi", "0")),
                ),
            }
        )

    current_score = compute_passive_support_score(
        float(summary["spot"]),
        float(summary["pin_low"]),
        float(summary["pin_center"]),
        float(summary["pin_high"]),
        float(summary["total_net_gex"]),
        float(summary["next_day_share"]),
        float(summary["front_total_oi_3d"]),
        float(summary["total_visible_oi"]),
    )
    prev_score = float(history_points[-2]["score"]) if len(history_points) >= 2 else current_score
    week_score = float(history_points[-6]["score"]) if len(history_points) >= 6 else prev_score

    if current_score >= 70:
        support_label = "Pasif destek guclu"
        support_class = "delta-up"
    elif current_score >= 50:
        support_label = "Pasif destek aktif"
        support_class = "delta-neutral"
    elif current_score >= 30:
        support_label = "Pasif destek inceliyor"
        support_class = "delta-neutral"
    else:
        support_label = "Pasif destek zayif"
        support_class = "delta-down"

    today = UTC_NOW().date()
    total_visible_oi = sum(float(row.get("open_interest", 0.0)) for row in expiry_rows) or 1.0
    month_buckets: Dict[str, Dict[str, float]] = {}
    next_14d_share = 0.0
    next_30d_share = 0.0
    next_60d_share = 0.0
    for row in expiry_rows:
        maturity = str(row.get("maturity", ""))
        dt = parse_maturity_date(maturity)
        if dt is None or dt < today:
            continue
        dte = (dt - today).days
        share = float(row.get("open_interest", 0.0)) / total_visible_oi
        if dte <= 14:
            next_14d_share += share
        if dte <= 30:
            next_30d_share += share
        if dte <= 60:
            next_60d_share += share
        month_key = dt.strftime("%b%y").upper()
        proximity_weight = max(0.15, 1.0 - (dte / 120.0))
        bucket = month_buckets.setdefault(month_key, {"share": 0.0, "support": 0.0})
        bucket["share"] += share
        bucket["support"] += share * proximity_weight

    forward_labels = list(month_buckets.keys())[:6]
    forward_share_values = [month_buckets[label]["share"] * 100.0 for label in forward_labels]
    forward_support_values = [month_buckets[label]["support"] * 100.0 for label in forward_labels]
    strongest_month = forward_labels[forward_support_values.index(max(forward_support_values))] if forward_labels else "-"
    weakest_month = forward_labels[forward_support_values.index(min(forward_support_values))] if forward_labels else "-"
    strongest_value = max(forward_support_values) if forward_support_values else 0.0
    weakest_value = min(forward_support_values) if forward_support_values else 0.0

    history_dates = [str(point["date"]) for point in history_points[-14:]]
    forward_history_matrix: List[List[float]] = []
    for month_label in forward_labels:
        row_values: List[float] = []
        for date_text in history_dates:
            payload, _ = load_snapshot_for_date(f"{currency.lower()}_oi_expiry_aggregate", date_text)
            daily_rows = parse_expiry_rows(payload or {}, "aggregate") if payload else []
            if not daily_rows:
                payload, _ = load_snapshot_for_date(f"{currency.lower()}_oi_expiry_{primary_market}", date_text)
                daily_rows = parse_expiry_rows(payload or {}, primary_market) if payload else []
            if not daily_rows:
                row_values.append(0.0)
                continue
            daily_today = datetime.strptime(date_text, "%Y-%m-%d").date()
            daily_total_oi = sum(float(r.get("open_interest", 0.0)) for r in daily_rows) or 1.0
            support_value = 0.0
            for r in daily_rows:
                maturity = str(r.get("maturity", ""))
                dt = parse_maturity_date(maturity)
                if dt is None or dt < daily_today:
                    continue
                if dt.strftime("%b%y").upper() != month_label:
                    continue
                dte = (dt - daily_today).days
                share = float(r.get("open_interest", 0.0)) / daily_total_oi
                proximity_weight = max(0.15, 1.0 - (dte / 120.0))
                support_value += share * proximity_weight * 100.0
            row_values.append(support_value)
        forward_history_matrix.append(row_values)
    forward_delta_7d = []
    for row_values in forward_history_matrix:
        if len(row_values) >= 7:
            forward_delta_7d.append(row_values[-1] - row_values[-7])
        elif len(row_values) >= 2:
            forward_delta_7d.append(row_values[-1] - row_values[0])
        else:
            forward_delta_7d.append(0.0)

    if next_14d_share >= 0.12:
        headline = "Pasif destek on tarafta yogun; yakin vadeler hala aktif tampon uretiyor."
    elif next_30d_share >= 0.18:
        headline = "Pasif destek tamamen bitmis degil ama daha dis vadeye tasiniyor."
    else:
        headline = "Yakin vade gamma tamponu inceliyor; pasif destek daha secici hale geliyor."

    if next_14d_share >= 0.12 and current_score >= 50:
        forward_verdict = f"Beklenen sey: pasif destek yakin vadede hala aktif; en yogun pencere {strongest_month} boyunca pin / mean-reversion davranisi daha kolay surer."
    elif next_30d_share >= 0.18:
        forward_verdict = f"Beklenen sey: destek hemen bitmiyor ama takvime yayiliyor; en yogun pencere {strongest_month}, sonrasinda tampon daha secici hale geliyor."
    else:
        forward_verdict = f"Beklenen sey: yakin vade pasif destek zayif; {strongest_month} disinda belirgin bir tampon yoksa fiyat daha yonlu davranabilir."

    notes = [
        f"Bugunku pasif destek skoru {current_score:.0f}/100; dunden bugune {current_score - prev_score:+.0f}, son 5 baz gune gore {current_score - week_score:+.0f}.",
        f"Onumuzdeki 14 gunde gorunen expiry OI'nin %{next_14d_share * 100:.1f}'i, 30 gunde %{next_30d_share * 100:.1f}'i, 60 gunde %{next_60d_share * 100:.1f}'i birikiyor.",
        f"Takvimde en yogun pasif destek penceresi {strongest_month} (skor {strongest_value:.1f}); en ince pencere {weakest_month} (skor {weakest_value:.1f}).",
    ]
    if current_score >= 50 and next_14d_share >= 0.10:
        notes.append("Bu kombinasyon, pin / mean-reversion davranisinin hemen bitmedigini; yakin vade expirylere kadar pasif destegin tasinabildigini soyler.")
    elif current_score < 35 and next_14d_share < 0.08:
        notes.append("Hem tarihsel destek skoru hem de yakin-vade expiry yogunlugu zayif; pasif destek penceresi inceliyor ve fiyat daha serbest davranabilir.")
    else:
        notes.append("Yapi tamamen bosalmis degil, ama destek artik tek bir gunluk expiry duvarindan cok dagilan bir takvim etkisiyle tasiniyor.")

    return {
        "headline": headline,
        "support_label": support_label,
        "support_class": support_class,
        "current_score": current_score,
        "score_delta_day": current_score - prev_score,
        "score_delta_week": current_score - week_score,
        "next_14d_share": next_14d_share * 100.0,
        "next_30d_share": next_30d_share * 100.0,
        "next_60d_share": next_60d_share * 100.0,
        "strongest_month": strongest_month,
        "weakest_month": weakest_month,
        "forward_verdict": forward_verdict,
        "history_points": history_points,
        "forward_labels": forward_labels,
        "forward_share_values": forward_share_values,
        "forward_support_values": forward_support_values,
        "forward_history_dates": history_dates,
        "forward_history_matrix": forward_history_matrix,
        "forward_delta_7d": forward_delta_7d,
        "notes": notes,
    }


def build_gex_term_structure_section(
    summary: Dict[str, object],
    term_rows: List[Dict],
    term_meta: Optional[Dict[str, float]] = None,
    previous_term_rows: Optional[List[Dict]] = None,
    previous_term_meta: Optional[Dict[str, float]] = None,
    previous_snapshot_date: Optional[str] = None,
) -> Dict[str, object]:
    if not term_rows:
        return {
            "rows": [],
            "spot": float(summary["spot"]),
            "front_maturity": "-",
            "front_gex": 0.0,
            "total_gex": 0.0,
            "strongest_positive": None,
            "strongest_negative": None,
            "front_rolloff_risk": "Veri yok",
            "next_support_expiry": "Veri yok",
            "gamma_vacuum_window": "Veri yok",
            "headline": "Expiry bazli GEX term structure verisi yok.",
            "notes": ["Bu bolum icin canli expiry-bazli GEX verisi alinmadi."],
            "delta_rows": [],
            "delta_badges": [],
            "delta_summary": "Dunden bugune karsilastirma icin onceki snapshot yok.",
            "delta_notes": ["Bu bolum en az iki gunluk expiry-bazli GEX snapshot biriktikce dolacak."],
            "comparison_label": "",
        }

    today = UTC_NOW().date()
    future_rows = [row for row in term_rows if row["date"] >= today]
    if not future_rows:
        future_rows = term_rows[:]

    reported_total_gex = 0.0
    if term_meta:
        try:
            reported_total_gex = float(term_meta.get("total_gex", 0.0) or 0.0)
        except Exception:
            reported_total_gex = 0.0
    total_gex = reported_total_gex if reported_total_gex else sum(float(row["gex"]) for row in future_rows)
    front = future_rows[0]
    strongest_positive = max(future_rows, key=lambda row: float(row["gex"]))
    negative_rows = [row for row in future_rows if float(row["gex"]) < 0]
    strongest_negative = min(negative_rows, key=lambda row: float(row["gex"])) if negative_rows else None

    top_positive = sorted(
        [row for row in future_rows if float(row["gex"]) > 0],
        key=lambda row: float(row["gex"]),
        reverse=True,
    )[:3]
    top_negative = sorted(
        [row for row in future_rows if float(row["gex"]) < 0],
        key=lambda row: float(row["gex"]),
    )[:3]
    later_positive = [row for row in future_rows[1:] if float(row["gex"]) > 0]
    next_support = later_positive[0] if later_positive else None
    front_share = (float(front["gex"]) / total_gex) if total_gex > 0 else 0.0
    next_positive_gap_days = (next_support["date"] - front["date"]).days if next_support else None
    nearby_negative = [row for row in future_rows[1:4] if float(row["gex"]) < 0]

    front_sign = "pozitif" if float(front["gex"]) > 0 else "negatif" if float(front["gex"]) < 0 else "nötr"
    if float(front["gex"]) > 0 and strongest_negative:
        headline = (
            f"On vade {front['maturity']} {front_sign} gamma tasiyor; pasif destek var ama "
            f"{strongest_negative['maturity']} vadesinde negatif gamma cebi duruyor."
        )
    elif float(front["gex"]) > 0:
        headline = f"On vade {front['maturity']} pozitif gamma tasiyor; pasif destek hemen bosalmiyor."
    elif strongest_positive:
        headline = (
            f"On vade {front['maturity']} destekleyici degil; bir sonraki anlamli gamma destegi "
            f"{strongest_positive['maturity']} vadesinde."
        )
    else:
        headline = f"On vade {front['maturity']} gamma desteği zayif; takvimde belirgin tampon görünmüyor."

    notes = [
        f"En guclu pozitif gamma vadesi {strongest_positive['maturity']} ({float(strongest_positive['gex']):,.0f}); pasif destek en cok burada yogunlasiyor."
        if strongest_positive
        else "Pozitif gamma taşıyan anlamlı bir expiry görünmüyor.",
        (
            f"En riskli negatif gamma vadesi {strongest_negative['maturity']} ({float(strongest_negative['gex']):,.0f}); bu pencere fiyat davranisini daha yonlu yapabilir."
            if strongest_negative
            else "Gorunen vadelerde anlamli negatif gamma cebi yok."
        ),
        (
            f"On vade {front['maturity']} {front_sign} gamma ile {float(front['gex']):,.0f}; bu expiry gectikce pasif desteğin ilk nereye kayacagi term structure ile okunabilir."
        ),
    ]
    if float(front["gex"]) > 0 and strongest_positive and strongest_positive["maturity"] != front["maturity"]:
        notes.append(
            f"Front expiry bosalirken destek tamamen kaybolmak yerine {strongest_positive['maturity']} tarafina kayabilir."
        )
    elif float(front["gex"]) <= 0:
        notes.append(
            "Front expiry tek basina pasif destek vermiyor; relief yerine daha kirilgan fiyat davranisi beklenmeli."
        )

    if float(front["gex"]) <= 0:
        front_rolloff_risk = "Yuksek: on vade destekleyici degil; expiry gecisi pasif destek eklemeyecek."
    elif front_share >= 0.30 and (not next_support or (next_positive_gap_days is not None and next_positive_gap_days > 10)):
        front_rolloff_risk = "Yuksek: on vade destegin buyuk kismini tasiyor; bu pencere bosalirken tampon hizla inceliyor olabilir."
    elif front_share >= 0.18:
        front_rolloff_risk = "Orta: on vade anlamli destek tasiyor ama takvimde arkadan gelen ikinci bir tampon var."
    else:
        front_rolloff_risk = "Dusuk-Orta: destek tek expiry'ye yigilmamis; roll-off daha yumusak olabilir."

    if next_support:
        next_support_expiry = (
            f"{next_support['maturity']}: sonraki pozitif gamma tamponu burada; front bosalirken destek ilk olarak bu vadeye kayabilir."
        )
    elif strongest_positive:
        next_support_expiry = (
            f"{strongest_positive['maturity']}: takvimdeki ana pozitif gamma cebi burada, ama yakin ikinci bir tampon yok."
        )
    else:
        next_support_expiry = "Takvimde anlamli bir sonraki pozitif gamma tamponu gorunmuyor."

    if nearby_negative:
        vacuum_ref = nearby_negative[0]
        gamma_vacuum_window = (
            f"{vacuum_ref['maturity']} civari: on vade gectikten sonra ilk yakin pencere negatif gamma tasiyor; pasif destekten yonlu davranisa gecis riski var."
        )
    elif next_positive_gap_days is not None and next_positive_gap_days > 14:
        gamma_vacuum_window = (
            f"{front['maturity']} sonrasi yaklasik {next_positive_gap_days} gunluk arada belirgin pozitif gamma tamponu zayifliyor olabilir."
        )
    else:
        gamma_vacuum_window = "Yakin takvimde temiz bir gamma vacuum gorunmuyor; destek bir sonraki pozitif vadeye daha duzenli kayabilir."

    if float(front["gex"]) < 0 and strongest_positive:
        summary_sentence = (
            f"Yakin vadede negatif gamma nedeniyle fiyat davranisi daha kirilgan olabilir; "
            f"sonraki pozitif gamma ceplerinde hareketin daha kontrollu hale gelmesi beklenir."
        )
    elif float(front["gex"]) > 0:
        summary_sentence = (
            "Yakin vadede pozitif gamma nedeniyle fiyat davranisinin daha kontrollu kalmasi beklenir; "
            "negatif gamma ceplerine yaklastikca hareket daha yonlu hale gelebilir."
        )
    else:
        summary_sentence = (
            "Gamma dagilimi karisik; yakin vadede fiyat davranisi denge ile kirilganlik arasinda hizla degisebilir."
        )

    comparison_label = ""
    delta_rows: List[Dict[str, object]] = []
    delta_badges: List[str] = []
    delta_notes: List[str] = []
    delta_summary = "Dunden bugune karsilastirma icin onceki snapshot yok."
    if previous_term_rows and previous_snapshot_date:
        previous_future_rows = [row for row in previous_term_rows if row["date"] >= today]
        if not previous_future_rows:
            previous_future_rows = previous_term_rows[:]
        prev_map = {str(row["maturity"]): float(row["gex"]) for row in previous_future_rows}
        current_map = {str(row["maturity"]): float(row["gex"]) for row in future_rows}
        date_map = {str(row["maturity"]): row["date"] for row in future_rows}
        for row in previous_future_rows:
            date_map.setdefault(str(row["maturity"]), row["date"])

        all_maturities = sorted(
            set(current_map) | set(prev_map),
            key=lambda maturity: (date_map.get(maturity, today), maturity),
        )
        for maturity in all_maturities:
            current_gex = current_map.get(maturity, 0.0)
            previous_gex = prev_map.get(maturity, 0.0)
            delta_rows.append(
                {
                    "maturity": maturity,
                    "date": date_map.get(maturity, today),
                    "current_gex": current_gex,
                    "previous_gex": previous_gex,
                    "delta_gex": current_gex - previous_gex,
                }
            )

        if delta_rows:
            total_prev_gex = 0.0
            if previous_term_meta:
                try:
                    total_prev_gex = float(previous_term_meta.get("total_gex", 0.0) or 0.0)
                except Exception:
                    total_prev_gex = 0.0
            if not total_prev_gex:
                total_prev_gex = sum(float(row["gex"]) for row in previous_future_rows)

            total_delta = total_gex - total_prev_gex
            front_prev_gex = prev_map.get(str(front["maturity"]), 0.0)
            front_delta = float(front["gex"]) - front_prev_gex
            strongest_increase = max(delta_rows, key=lambda row: float(row["delta_gex"]))
            strongest_decrease = min(delta_rows, key=lambda row: float(row["delta_gex"]))
            comparison_label = f"{previous_snapshot_date} -> {today.isoformat()}"
            delta_badges = [
                f"Toplam GEX: {total_delta:+,.0f}",
                f"On vade {front['maturity']}: {front_delta:+,.0f}",
            ]
            if float(strongest_increase["delta_gex"]) > 0:
                delta_badges.append(
                    f"En guclenen: {strongest_increase['maturity']} {float(strongest_increase['delta_gex']):+,.0f}"
                )
            if float(strongest_decrease["delta_gex"]) < 0:
                delta_badges.append(
                    f"En zayiflayan: {strongest_decrease['maturity']} {float(strongest_decrease['delta_gex']):+,.0f}"
                )

            delta_notes.append(
                f"Toplam expiry-bazli GEX {previous_snapshot_date} snapshot'ina gore {total_delta:+,.0f} degisti."
            )
            if float(strongest_increase["delta_gex"]) > 0:
                delta_notes.append(
                    f"En belirgin destek artisi {strongest_increase['maturity']} vadesinde goruldu ({float(strongest_increase['delta_gex']):+,.0f})."
                )
            if float(strongest_decrease["delta_gex"]) < 0:
                delta_notes.append(
                    f"En belirgin zayiflama {strongest_decrease['maturity']} vadesinde goruldu ({float(strongest_decrease['delta_gex']):+,.0f})."
                )

            if total_delta > 0 and front_delta >= 0:
                delta_summary = (
                    "Dunden bugune gamma destegi genisledi; ozellikle yakin takvimdeki tampon daha destekleyici hale geldi."
                )
            elif total_delta > 0:
                delta_summary = (
                    "Toplam gamma destegi artsa da yakin vadedeki tampon ayni gucle genislemiyor; destek daha cok arka vadelerde yogunlasiyor."
                )
            elif front_delta < 0:
                delta_summary = (
                    "Dunden bugune on tarafta tampon inceldi; yakin vade davranisi bir gun onceye gore daha kirilgan olabilir."
                )
            else:
                delta_summary = (
                    "Vade yapisinda belirgin yeniden dagilim var; bazi tamponlar guclenirken digerleri zayifladi."
                )

    return {
        "rows": future_rows,
        "spot": float(summary["spot"]),
        "front_maturity": front["maturity"],
        "front_gex": float(front["gex"]),
        "total_gex": total_gex,
        "strongest_positive": strongest_positive,
        "strongest_negative": strongest_negative,
        "front_rolloff_risk": front_rolloff_risk,
        "next_support_expiry": next_support_expiry,
        "gamma_vacuum_window": gamma_vacuum_window,
        "summary_sentence": summary_sentence,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "headline": headline,
        "notes": notes,
        "delta_rows": delta_rows,
        "delta_badges": delta_badges,
        "delta_summary": delta_summary,
        "delta_notes": delta_notes,
        "comparison_label": comparison_label,
    }


def build_iv_regime_section(summary: Dict[str, object], archive_rows: List[Dict[str, str]]) -> Dict[str, object]:
    stats = compute_iv_regime_stats(float(summary["front_atm_iv"]), archive_rows[:-1])
    iv_percentile = stats["iv_percentile"]
    iv_rank = stats["iv_rank"]
    adj_iv_rank = stats["adj_iv_rank"]
    sample_size = int(stats["sample_size"])
    notes = [
        f"Front ATM IV percentile {iv_percentile:.0f}; bugunku implied vol yerel serideki gunlerin yaklasik %{iv_percentile:.0f}'inden daha pahali.",
        f"Front ATM IV rank {iv_rank:.0f}; ham min-max araligina gore vol rejimi {'ust' if iv_rank >= 60 else 'orta-alt'} bolgede.",
        f"Outlier-adjusted rank {adj_iv_rank:.0f}; tekil spike gunlerinin skala etkisini azaltmak icin winsorized rank kullaniyoruz.",
    ]
    if sample_size < 20:
        notes.append(f"Orneklem kisa ({sample_size} gun); bu bolum simdilik yonlendirici, 52 haftalik olgun istatistik degil.")
    if iv_percentile >= 75 and iv_rank <= 55:
        notes.append("Percentile yuksek ama ham rank orta; gecmisteki tekil sert spike gunleri rank'i bastiriyor olabilir. Bu durumda vol sandigindan daha pahali olabilir.")
    elif iv_percentile >= 75 and adj_iv_rank >= 70:
        notes.append("Hem percentile hem duzeltilmis rank yuksek; vol satmak icin zemin guclu, ancak rejim riski ayri kontrol edilmeli.")
    elif iv_percentile <= 25 and iv_rank <= 30:
        notes.append("Hem percentile hem rank dusuk; vol yerel seriye gore ucuz ve vol alici bakisi daha anlamli.")
    else:
        notes.append("Percentile'i alim/satim filtresi, rank'i ise ne kadar ekstrem bolgedeyiz filtresi olarak okuyun.")

    if iv_percentile >= 70 and adj_iv_rank >= 65 and summary["iv_rv_7d_spread"] >= 0:
        vol_bias = "Vol seller bias"
        vol_bias_note = "Implied vol yerel seriye gore pahali bolgede; rejim riski haric ilk egilim vol satmaya daha yakin."
        vol_bias_class = "delta-up"
        checklist = [
            "Percentile yuksek kalmaya devam ediyor mu?",
            "IV-RV spread sifirin ustunde mi?",
            "Rejim degisimi riski vol satmayi bozacak kadar buyuk mu?",
        ]
    elif iv_percentile <= 30 and summary["iv_rv_7d_spread"] <= 0:
        vol_bias = "Vol buyer bias"
        vol_bias_note = "Implied vol yerel seriye gore ucuz; realized hareket IV'yi zorlayabiliyorsa vol alici bakis daha mantikli."
        vol_bias_class = "delta-down"
        checklist = [
            "Percentile dusuk kalmaya devam ediyor mu?",
            "Realized hareket implied volu asiyor mu?",
            "Spotta yeni yonlu/kopmali hareket riski artiyor mu?",
        ]
    else:
        vol_bias = "Neutral / selective vol"
        vol_bias_note = "Vol ne net ucuz ne net pahali; secici olmak ve yon/olay baglamiyla birlikte okumak daha dogru."
        vol_bias_class = "delta-neutral"
        checklist = [
            "Vol tek basina edge vermiyor; yon/olay filtresi ekle.",
            "Percentile ile rank ayni hikayeyi mi anlatiyor?",
            "Stratejiyi tekli vol gorusu yerine secici yapi olarak kur.",
        ]

    return {
        "iv_percentile": iv_percentile,
        "iv_rank": iv_rank,
        "adj_iv_rank": adj_iv_rank,
        "sample_size": sample_size,
        "notes": notes,
        "vol_bias": vol_bias,
        "vol_bias_note": vol_bias_note,
        "vol_bias_class": vol_bias_class,
        "checklist": checklist,
    }


def build_layered_trade_check(
    summary: Dict[str, object],
    archive_rows: List[Dict[str, str]],
    price_series: List[Dict[str, object]],
) -> Dict[str, object]:
    closes = [float(row["close"]) for row in price_series]
    spot = float(summary["spot"])
    rsi14 = compute_rsi(closes, 14)
    bb = compute_bollinger_stats(closes, 20)
    z20 = float(bb["zscore"])
    lower_band = float(bb["lower"])
    upper_band = float(bb["upper"])
    band_pct = float(bb["band_pct"])

    prev = archive_rows[-2] if len(archive_rows) >= 2 else None
    prev_rr = to_float(prev.get("rr_25d", "0")) if prev else float(summary["front_rr_25d"])
    prev_skew_ratio = to_float(prev.get("normalized_25d_skew_ratio", "0")) if prev else float(summary["normalized_25d_skew_ratio"])
    rr_change = float(summary["front_rr_25d"]) - prev_rr
    skew_ratio_change = float(summary["normalized_25d_skew_ratio"]) - prev_skew_ratio
    iv_spread = float(summary["iv_rv_7d_spread"])
    inventory_net = float(summary["inventory_net"])
    pin_low = float(summary["pin_low"])
    pin_center = float(summary["pin_center"])
    pin_high = float(summary["pin_high"])
    gamma_badge = str(summary.get("gamma_map_read", {}).get("regime_badge", ""))

    technical_long = 0
    technical_short = 0
    technical_notes: List[str] = []
    if rsi14 <= 35:
        technical_long += 1
        technical_notes.append(f"RSI(14) {rsi14:.1f}: asiri satima yakin / relief adayi.")
    elif rsi14 >= 65:
        technical_short += 1
        technical_notes.append(f"RSI(14) {rsi14:.1f}: asiri alim tarafi, yukari extension yorulabilir.")
    else:
        technical_notes.append(f"RSI(14) {rsi14:.1f}: teknik ekstrem net degil.")
    if spot <= lower_band or z20 <= -1.5:
        technical_long += 1
        technical_notes.append(f"Spot Bollinger alt banda yakin ({lower_band:,.0f}) / z-score {z20:+.2f}; teknik baski asiriya kaymis olabilir.")
    elif spot >= upper_band or z20 >= 1.5:
        technical_short += 1
        technical_notes.append(f"Spot Bollinger ust banda yakin ({upper_band:,.0f}) / z-score {z20:+.2f}; teknik extension olgunlasiyor olabilir.")
    else:
        technical_notes.append(f"Bollinger/z-score tarafinda fiyat orta bolgede; band ici davranis daha baskin (band %{band_pct*100:.0f}).")

    vol_long = 0
    vol_short = 0
    vol_notes: List[str] = []
    if iv_spread <= 1:
        vol_long += 1
        vol_notes.append(f"IV-RV spread {iv_spread:+.2f}: dususe ragmen vol panik fiyatlamiyor olabilir.")
    elif iv_spread >= 6:
        vol_short += 1
        vol_notes.append(f"IV-RV spread {iv_spread:+.2f}: hareket korkuyla fiyatlaniyor, dusus teyidi daha guclu.")
    else:
        vol_notes.append(f"IV-RV spread {iv_spread:+.2f}: vol tepkisi ara bolgede, tek basina karar vermiyor.")

    skew_long = 0
    skew_short = 0
    skew_notes: List[str] = []
    if rr_change >= 0.5 and skew_ratio_change <= 0:
        skew_long += 1
        skew_notes.append(f"25d RR {rr_change:+.2f} puan rahatliyor; put skew gevsemeye baslamis.")
    elif rr_change <= -0.5 or skew_ratio_change > 0.01:
        skew_short += 1
        skew_notes.append(f"25d RR {rr_change:+.2f} puan daha savunmaci; skew rahatlamiyor.")
    else:
        skew_notes.append(f"Skew degisimi sinirli ({rr_change:+.2f}); koruma talebi net rahatlamis degil.")

    positioning_long = 0
    positioning_short = 0
    positioning_notes: List[str] = []
    if spot >= pin_low and inventory_net >= 0 and gamma_badge != "Negative Gamma Risk Active":
        positioning_long += 1
        positioning_notes.append(f"Spot {pin_low:,.0f} ustunde ve gamma rejimi savunmasiz degil; positioning relief denemesine izin veriyor.")
    elif spot < pin_low or gamma_badge == "Negative Gamma Risk Active":
        positioning_short += 1
        positioning_notes.append(f"Spot denge bandinin altinda / gamma kirilgan; positioning downside devamini daha kolay aktive ediyor.")
    else:
        positioning_notes.append("Positioning filtresi karisik; seviye calisabilir ama temiz teyit yok.")
    if spot >= pin_center and inventory_net > 0:
        positioning_long += 1
        positioning_notes.append(f"Spot {pin_center:,.0f} civarinda/ustunde tutuluyor; denge merkezi tamamen kaybolmus degil.")
    elif spot <= pin_low and inventory_net < 0:
        positioning_short += 1
        positioning_notes.append(f"Spot {pin_low:,.0f} alti ve inventory {inventory_net:+.2f}; savunmaci fiyatlama agirlikli.")

    long_score = technical_long + vol_long + skew_long + positioning_long
    short_score = technical_short + vol_short + skew_short + positioning_short
    if long_score >= 4 and long_score >= short_score + 2:
        verdict = "Relief long adayi guclu"
        verdict_class = "delta-up"
    elif short_score >= 4 and short_score >= long_score + 2:
        verdict = "Downside devam setup'i guclu"
        verdict_class = "delta-down"
    elif long_score >= short_score + 1:
        verdict = "Long setup var ama teyit eksik"
        verdict_class = "delta-neutral"
    elif short_score >= long_score + 1:
        verdict = "Short/devam setup'i var ama teyit eksik"
        verdict_class = "delta-neutral"
    else:
        verdict = "Net setup yok / boyut kucult"
        verdict_class = "delta-neutral"

    if long_score >= 4 and long_score >= short_score + 2:
        bias_label = "Long bias"
        trade_gate = "Trade gate acik"
    elif short_score >= 4 and short_score >= long_score + 2:
        bias_label = "Short bias"
        trade_gate = "Trade gate acik"
    elif long_score > short_score:
        bias_label = "Long adayi"
        trade_gate = "Teyit bekle"
    elif short_score > long_score:
        bias_label = "Short adayi"
        trade_gate = "Teyit bekle"
    else:
        bias_label = "Nötr / karisik"
        trade_gate = "No trade"

    expected_action = (
        "Uc katman ust uste binmedikce agresif boyut alma."
        if abs(long_score - short_score) <= 1
        else "Long tarafinda conviction daha yuksek; yine de seviye teyidi olmadan buyutme."
        if long_score > short_score
        else "Short / defensive taraf daha agir; destek alti kabul gelmeden yine de kovalamak yerine teyit ara."
    )

    rows = [
        {"label": "Teknik ekstrem", "long": technical_long, "short": technical_short, "note": technical_notes[0] if technical_notes else ""},
        {"label": "Vol tepkisi", "long": vol_long, "short": vol_short, "note": vol_notes[0] if vol_notes else ""},
        {"label": "Skew tepkisi", "long": skew_long, "short": skew_short, "note": skew_notes[0] if skew_notes else ""},
        {"label": "Positioning filtresi", "long": positioning_long, "short": positioning_short, "note": positioning_notes[0] if positioning_notes else ""},
    ]
    notes = technical_notes + vol_notes + skew_notes + positioning_notes
    return {
        "rsi14": rsi14,
        "bb_lower": lower_band,
        "bb_upper": upper_band,
        "z20": z20,
        "band_pct": band_pct,
        "long_score": long_score,
        "short_score": short_score,
        "bias_label": bias_label,
        "trade_gate": trade_gate,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "expected_action": expected_action,
        "rows": rows,
        "notes": notes,
    }


def build_layered_trade_check_en(section: Dict[str, object]) -> Dict[str, object]:
    def translate_note(text: str) -> str:
        text = tr_to_en_text(text)
        replacements = [
            (r"RSI\(14\) ([\d\.]+): teknik ekstrem net degil\.", r"RSI(14) \1: no clear technical extreme."),
            (r"RSI\(14\) ([\d\.]+): asiri satima yakin / relief adayi\.", r"RSI(14) \1: approaching oversold / relief candidate."),
            (r"RSI\(14\) ([\d\.]+): asiri alim tarafi, yukari extension yorulabilir\.", r"RSI(14) \1: near overbought, upside extension may be tiring."),
            (r"IV-RV spread ([+-]?[\d\.]+): vol tepkisi ara bolgede, tek basina karar vermiyor\.", r"IV-RV spread \1: the vol reaction sits in the middle zone and does not decide the trade on its own."),
            (r"IV-RV spread ([+-]?[\d\.]+): dususe ragmen vol panik fiyatlamiyor olabilir\.", r"IV-RV spread \1: vol may not be pricing panic despite the drop."),
            (r"IV-RV spread ([+-]?[\d\.]+): hareket korkuyla fiyatlaniyor, dusus teyidi daha guclu\.", r"IV-RV spread \1: the move is being priced with fear, so downside confirmation is stronger."),
            (r"25d RR ([+-]?[\d\.]+) puan rahatliyor; put skew gevsemeye baslamis\.", r"25d RR is easing by \1 points; put skew has started to relax."),
            (r"25d RR ([+-]?[\d\.]+) puan daha savunmaci; skew rahatlamiyor\.", r"25d RR is \1 points more defensive; skew is not relaxing."),
            (r"Skew degisimi sinirli \(([+-]?[\d\.]+)\); koruma talebi net rahatlamis degil\.", r"Skew change is limited (\1); protection demand has not clearly relaxed."),
            (r"Spot denge bandinin altinda / gamma kirilgan; positioning downside devamini daha kolay aktive ediyor\.", r"Spot is below the balance band / gamma is fragile; positioning makes downside continuation easier to activate."),
            (r"Spot ([\d,]+) ustunde ve gamma rejimi savunmasiz degil; positioning relief denemesine izin veriyor\.", r"Spot is above \1 and the gamma regime is not exposed; positioning still allows a relief attempt."),
            (r"Spot ([\d,]+) civarinda/ustunde tutuluyor; denge merkezi tamamen kaybolmus degil\.", r"Spot is being held around / above \1; the balance center is not fully gone."),
            (r"Spot ([\d,]+) alti ve inventory ([+-]?[\d\.]+); savunmaci fiyatlama agirlikli\.", r"Spot is below \1 and inventory is \2; defensive pricing still dominates."),
            (r"Positioning filtresi karisik; seviye calisabilir ama temiz teyit yok\.", r"Positioning filter is mixed; the level may still work, but there is no clean confirmation."),
            (r"Spot Bollinger alt banda yakin \(([\d,]+)\) / z-score ([+-]?[\d\.]+); teknik baski asiriya kaymis olabilir\.", r"Spot is near the Bollinger lower band (\1) / z-score \2; technical pressure may have pushed too far."),
            (r"Spot Bollinger ust banda yakin \(([\d,]+)\) / z-score ([+-]?[\d\.]+); teknik extension olgunlasiyor olabilir\.", r"Spot is near the Bollinger upper band (\1) / z-score \2; technical extension may be maturing."),
            (r"Bollinger/z-score tarafinda fiyat orta bolgede; band ici davranis daha baskin \(band %([\d]+)\)\.", r"Price sits in the middle zone on Bollinger / z-score; in-band behavior is still more dominant (band \1%)."),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        return text

    layered_trade_check_en = {
        **section,
        "verdict": tr_to_en_text(section.get("verdict", "")),
        "expected_action": tr_to_en_text(section.get("expected_action", "")),
        "bias_label": tr_to_en_text(section.get("bias_label", "")),
        "trade_gate": tr_to_en_text(section.get("trade_gate", "")),
        "notes": [translate_note(str(note)) for note in section.get("notes", [])],
        "rows": [
            {
                **row,
                "label": tr_to_en_text(str(row.get("label", ""))),
                "note": translate_note(str(row.get("note", ""))),
            }
            for row in section.get("rows", [])
        ],
    }
    layered_trade_check_en["verdict"] = {
        "Relief long adayi guclu": "Relief long setup is strong",
        "Downside devam setup'i guclu": "Downside continuation setup is strong",
        "Long setup var ama teyit eksik": "There is a long setup, but confirmation is missing",
        "Short/devam setup'i var ama teyit eksik": "There is a short/continuation setup, but confirmation is missing",
        "Net setup yok / boyut kucult": "No clear setup / reduce size",
    }.get(layered_trade_check_en["verdict"], layered_trade_check_en["verdict"])
    layered_trade_check_en["expected_action"] = {
        "Uc katman ust uste binmedikce agresif boyut alma.": "Do not size aggressively unless at least three layers line up.",
        "Long tarafinda conviction daha yuksek; yine de seviye teyidi olmadan buyutme.": "Long-side conviction is higher, but do not add size without level confirmation.",
        "Short / defensive taraf daha agir; destek alti kabul gelmeden yine de kovalamak yerine teyit ara.": "The short / defensive side is stronger, but still wait for acceptance below support instead of chasing.",
    }.get(layered_trade_check_en["expected_action"], layered_trade_check_en["expected_action"])
    layered_trade_check_en["bias_label"] = {
        "Long bias": "Long bias",
        "Short bias": "Short bias",
        "Long adayi": "Long candidate",
        "Short adayi": "Short candidate",
        "Nötr / karisik": "Neutral / mixed",
    }.get(section.get("bias_label", ""), layered_trade_check_en["bias_label"])
    layered_trade_check_en["trade_gate"] = {
        "Trade gate acik": "Trade gate open",
        "Teyit bekle": "Wait for confirmation",
        "No trade": "No trade",
    }.get(section.get("trade_gate", ""), layered_trade_check_en["trade_gate"])
    layer_label_map = {
        "Teknik ekstrem": "Technical extreme",
        "Vol tepkisi": "Vol reaction",
        "Skew tepkisi": "Skew reaction",
        "Positioning filtresi": "Positioning filter",
    }
    layered_trade_check_en["rows"] = [
        {
            **row,
            "label": layer_label_map.get(row["label"], row["label"]),
        }
        for row in layered_trade_check_en["rows"]
    ]
    return layered_trade_check_en


def build_gex_term_structure_section_en(section: Dict[str, object]) -> Dict[str, object]:
    result = dict(section)
    result["headline"] = tr_to_en_text(str(section.get("headline", "")))
    result["notes"] = tr_to_en_list(section.get("notes", []))
    result["front_rolloff_risk"] = tr_to_en_text(str(section.get("front_rolloff_risk", "")))
    result["next_support_expiry"] = tr_to_en_text(str(section.get("next_support_expiry", "")))
    result["gamma_vacuum_window"] = tr_to_en_text(str(section.get("gamma_vacuum_window", "")))
    result["summary_sentence"] = tr_to_en_text(str(section.get("summary_sentence", "")))
    result["summary_sentence"] = result["summary_sentence"].replace(
        "Yakin vadede negatif gamma nedeniyle fiyat davranisi daha kirilgan olabilir; sonraki pozitif gamma ceplerinde hareketin daha kontrollu hale gelmesi beklenir.",
        "Near-dated negative gamma can make price action more fragile; in the next positive-gamma pockets, moves are more likely to become controlled again.",
    )
    result["summary_sentence"] = result["summary_sentence"].replace(
        "Yakin vadede pozitif gamma nedeniyle fiyat davranisinin daha kontrollu kalmasi beklenir; negatif gamma ceplerine yaklastikca hareket daha yonlu hale gelebilir.",
        "Near-dated positive gamma should keep price action more controlled; as spot approaches negative-gamma pockets, moves can become more directional.",
    )
    result["summary_sentence"] = result["summary_sentence"].replace(
        "Gamma dagilimi karisik; yakin vadede fiyat davranisi denge ile kirilganlik arasinda hizla degisebilir.",
        "Gamma is mixed across the curve; near-dated price behavior can shift quickly between balance and fragility.",
    )

    result["front_rolloff_risk"] = re.sub(
        r"Orta: on vade anlamli destek tasiyor ama takvimde arkadan gelen ikinci bir tampon var\.",
        "Moderate: the front expiry carries meaningful support, but there is a second buffer behind it on the calendar.",
        result["front_rolloff_risk"],
    )
    result["front_rolloff_risk"] = re.sub(
        r"Yuksek: on vade destekleyici degil; expiry gecisi pasif destek eklemeyecek\.",
        "High: the front expiry is not supportive; the expiry roll will not add passive support.",
        result["front_rolloff_risk"],
    )
    result["front_rolloff_risk"] = re.sub(
        r"Yuksek: on vade destegin buyuk kismini tasiyor; bu pencere bosalirken tampon hizla inceliyor olabilir\.",
        "High: the front expiry carries a large share of support; the buffer may thin quickly as this window rolls off.",
        result["front_rolloff_risk"],
    )
    result["front_rolloff_risk"] = re.sub(
        r"Dusuk-Orta: destek tek expiry'ye yigilmamis; roll-off daha yumusak olabilir\.",
        "Low-to-moderate: support is not concentrated in a single expiry, so roll-off may be more gradual.",
        result["front_rolloff_risk"],
    )
    result["next_support_expiry"] = re.sub(
        r"([A-Z0-9]+): sonraki pozitif gamma tamponu burada; front bosalirken destek ilk olarak bu vadeye kayabilir\.",
        r"\1: the next positive-gamma buffer sits here; support can rotate here first as the front expiry rolls off.",
        result["next_support_expiry"],
    )
    result["next_support_expiry"] = re.sub(
        r"([A-Z0-9]+): takvimdeki ana pozitif gamma cebi burada, ama yakin ikinci bir tampon yok\.",
        r"\1: this is the main positive-gamma pocket on the calendar, but there is no nearby second buffer.",
        result["next_support_expiry"],
    )
    result["next_support_expiry"] = re.sub(
        r"Takvimde anlamli bir sonraki pozitif gamma tamponu gorunmuyor\.",
        "There is no clearly visible next positive-gamma buffer on the calendar.",
        result["next_support_expiry"],
    )
    result["gamma_vacuum_window"] = re.sub(
        r"([A-Z0-9]+) civari: on vade gectikten sonra ilk yakin pencere negatif gamma tasiyor; pasif destekten yonlu davranisa gecis riski var\.",
        r"Around \1: the first nearby window after the front expiry carries negative gamma, so there is a risk of shifting from passive support to directional behavior.",
        result["gamma_vacuum_window"],
    )
    result["gamma_vacuum_window"] = re.sub(
        r"([A-Z0-9]+) sonrasi yaklasik (\d+) gunluk arada belirgin pozitif gamma tamponu zayifliyor olabilir\.",
        r"After \1, the roughly \2-day gap may see a weaker positive-gamma buffer.",
        result["gamma_vacuum_window"],
    )
    result["gamma_vacuum_window"] = re.sub(
        r"Yakin takvimde temiz bir gamma vacuum gorunmuyor; destek bir sonraki pozitif vadeye daha duzenli kayabilir\.",
        "There is no clean gamma vacuum nearby; support can rotate into the next positive expiry more smoothly.",
        result["gamma_vacuum_window"],
    )
    result["headline"] = re.sub(
        r"On vade ([A-Z0-9]+) pozitif gamma tasiyor; pasif destek var ama ([A-Z0-9]+) vadesinde negatif gamma cebi duruyor\.",
        r"The front expiry \1 carries positive gamma; passive support is present, but there is still a negative-gamma pocket in \2.",
        result["headline"],
    )
    result["headline"] = re.sub(
        r"On vade ([A-Z0-9]+) pozitif gamma tasiyor; pasif destek hemen bosalmiyor\.",
        r"The front expiry \1 carries positive gamma; passive support is not disappearing immediately.",
        result["headline"],
    )
    result["headline"] = re.sub(
        r"On vade ([A-Z0-9]+) destekleyici degil; bir sonraki anlamli gamma destegi ([A-Z0-9]+) vadesinde\.",
        r"The front expiry \1 is not supportive; the next meaningful gamma support sits in \2.",
        result["headline"],
    )
    cleaned_notes = []
    for note in result["notes"]:
        text = str(note)
        text = re.sub(
            r"En guclu pozitif gamma vadesi ([A-Z0-9]+) \(([-\d,]+)\); pasif destek en cok burada yogunlasiyor\.",
            r"The strongest positive-gamma expiry is \1 (\2); passive support is concentrated there most heavily.",
            text,
        )
        text = re.sub(
            r"En riskli negatif gamma vadesi ([A-Z0-9]+) \(([-\d,]+)\); bu pencere fiyat davranisini daha yonlu yapabilir\.",
            r"The riskiest negative-gamma expiry is \1 (\2); that window can make price action more directional.",
            text,
        )
        text = re.sub(
            r"On vade ([A-Z0-9]+) (pozitif|negatif|notr|nötr) gamma ile ([-\d,]+); bu expiry gectikce pasif destegin ilk nereye kayacagi term structure ile okunabilir\.",
            r"The front expiry \1 carries \2 gamma at \3; the term structure shows where passive support is likely to rotate first as this expiry rolls off.",
            text,
        )
        text = text.replace(
            "Front expiry tek basina pasif destek vermiyor; relief yerine daha kirilgan fiyat davranisi beklenmeli.",
            "The front expiry does not provide passive support on its own; instead of relief, more fragile price action should be expected.",
        )
        cleaned_notes.append(text)
    result["notes"] = cleaned_notes
    result["front_rolloff_risk"] = result["front_rolloff_risk"].replace(
        "High: on vade destekleyici degil; expiry gecisi pasif destek eklemeyecek.",
        "High: the front expiry is not supportive; the expiry roll will not add passive support.",
    )
    result["gamma_vacuum_window"] = result["gamma_vacuum_window"].replace(
        "civari: on vade gectikten and then ilk yakin pencere negatif gamma tasiyor; pasif destekten yonlu davranisa gecis riski var.",
        "area: the first nearby window after the front expiry carries negative gamma, so there is a risk of shifting from passive support to directional behavior.",
    )
    return result


def make_iv_regime_chart(rows: List[Dict[str, str]], out_path: Path, lang: str = "tr") -> None:
    filtered = [row for row in rows[-30:] if has_numeric(row.get("front_atm_iv", ""))]
    if len(filtered) < 2:
        return
    dates = [row["date"] for row in filtered]
    percentiles: List[float] = []
    adj_ranks: List[float] = []
    history: List[float] = []
    for row in filtered:
        current = to_float(row.get("front_atm_iv", "0"))
        if current <= 0:
            continue
        stats = compute_iv_regime_stats(current, [{"front_atm_iv": f"{value:.2f}"} for value in history])
        percentiles.append(stats["iv_percentile"])
        adj_ranks.append(stats["adj_iv_rank"])
        history.append(current)
    if not percentiles:
        return
    plot_dates = dates[-len(percentiles):]
    plt.figure(figsize=(10, 4.8))
    plt.plot(plot_dates, percentiles, color="#8b5a2b", marker="o", linewidth=2, label="IV Percentile")
    plt.plot(plot_dates, adj_ranks, color="#0f4c5c", marker="o", linewidth=2, label="Adj. IV Rank")
    plt.axhline(70, color="#c8a646", linewidth=1, linestyle="--")
    plt.axhline(30, color="#c8a646", linewidth=1, linestyle="--")
    plt.ylim(0, 100)
    plt.title(loc(lang, "IV Rejim Trendi", "IV Regime Trend"))
    plt.ylabel(loc(lang, "Skor", "Score"))
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_pain_drift_chart(rows: List[Dict[str, str]], out_path: Path, lang: str = "tr") -> None:
    filtered = [row for row in rows[-30:] if has_numeric(row.get("max_pain", ""))]
    if not filtered:
        return
    dates = [row["date"] for row in filtered]
    pain = [to_float(row.get("max_pain", "0")) for row in filtered]
    spot = [to_float(row.get("spot", "0")) for row in filtered]
    plt.figure(figsize=(10, 4.8))
    plt.plot(dates, pain, color="#8b5a2b", marker="o", linewidth=2, label=loc(lang, "Max pain", "Max pain"))
    plt.plot(dates, spot, color="#0f4c5c", marker="o", linewidth=2, label="Spot")
    plt.title("Pain Drift")
    plt.ylabel(loc(lang, "Seviye", "Level"))
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_position_map_drift_chart(rows: List[Dict[str, str]], out_path: Path, lang: str = "tr") -> None:
    filtered = [
        row for row in rows[-20:]
        if has_numeric(row.get("magnet_strike", ""))
        and has_numeric(row.get("put_hedge_low", ""))
        and has_numeric(row.get("put_hedge_high", ""))
        and has_numeric(row.get("call_cluster_high", ""))
        and has_numeric(row.get("speculation", ""))
        and has_numeric(row.get("tail_hedge", ""))
        and has_numeric(row.get("spot", ""))
    ]
    if len(filtered) < 2:
        return

    filtered = filtered[-14:]
    dates = [row["date"] for row in filtered]
    spot = [to_float(row["spot"]) for row in filtered]
    put_high = [to_float(row["put_hedge_high"]) for row in filtered]
    put_low = [to_float(row["put_hedge_low"]) for row in filtered]
    magnet = [to_float(row["magnet_strike"]) for row in filtered]
    call_cluster = [to_float(row["call_cluster_high"]) for row in filtered]
    speculation = [to_float(row["speculation"]) for row in filtered]

    curr = filtered[-1]
    prev = filtered[-2]
    delta_labels = [loc(lang, "Magnet", "Magnet"), loc(lang, "Put merkez", "Put center"), loc(lang, "Put genislik", "Put width"), loc(lang, "Call cluster", "Call cluster")]
    delta_values = [
        to_float(curr["magnet_strike"]) - to_float(prev["magnet_strike"]),
        ((to_float(curr["put_hedge_low"]) + to_float(curr["put_hedge_high"])) / 2)
        - ((to_float(prev["put_hedge_low"]) + to_float(prev["put_hedge_high"])) / 2),
        (to_float(curr["put_hedge_high"]) - to_float(curr["put_hedge_low"]))
        - (to_float(prev["put_hedge_high"]) - to_float(prev["put_hedge_low"])),
        to_float(curr["call_cluster_high"]) - to_float(prev["call_cluster_high"]),
    ]
    delta_colors = ["#d5bd79" if delta_values[0] == 0 else "#b9770e", "#c0392b", "#922b21", "#1e8449"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.6, 7.4),
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    ax0 = axes[0]
    ax0.fill_between(dates, put_low, put_high, color="#d1495b", alpha=0.18, label=loc(lang, "Put hedge zone", "Put hedge zone"))
    ax0.fill_between(dates, magnet, call_cluster, color="#2f9e44", alpha=0.12, label=loc(lang, "Call positioning", "Call positioning"))
    ax0.fill_between(dates, call_cluster, speculation, color="#3b82f6", alpha=0.08, label=loc(lang, "Speculation", "Speculation"))
    ax0.plot(dates, spot, color="#0f4c5c", marker="o", linewidth=2.6, label=loc(lang, "Spot", "Spot"))
    ax0.plot(dates, magnet, color="#f0b429", linewidth=2.2, label=loc(lang, "Magnet", "Magnet"))
    ax0.plot(dates, call_cluster, color="#2f9e44", linewidth=2.0, label=loc(lang, "Call cluster", "Call cluster"))
    ax0.plot(dates, speculation, color="#3b82f6", linewidth=1.8, label=loc(lang, "Speculation", "Speculation"))
    ax0.plot(dates, put_high, color="#c0392b", linewidth=1.8, linestyle="--", label=loc(lang, "Put hedge ust", "Put hedge upper"))
    ax0.set_title(loc(lang, "Spot ve Position Map Seviyeleri (Son 14 Gun)", "Spot and Position Map Levels (Last 14 Days)"), pad=10)
    ax0.set_ylabel(loc(lang, "Seviye", "Level"))
    ax0.grid(alpha=0.2)
    ax0.legend(fontsize=7.5, ncol=4, loc="upper left")
    ax0.annotate(
        loc(lang, f"Bugun spot {spot[-1]:,.0f}", f"Today's spot {spot[-1]:,.0f}"),
        xy=(dates[-1], spot[-1]),
        xytext=(-36, 10),
        textcoords="offset points",
        fontsize=8,
        color="#0f4c5c",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b8c7cf", alpha=0.9),
    )

    ax1 = axes[1]
    max_abs_delta = max([abs(v) for v in delta_values] + [0.0])
    if max_abs_delta < 100:
        ax1.axis("off")
        ax1.text(0.5, 0.68, loc(lang, "Bugun position-map seviyelerinde", "There is no meaningful"), ha="center", va="center",
                 fontsize=11, fontweight="bold", transform=ax1.transAxes, color="#2b2b2b")
        ax1.text(0.5, 0.48, loc(lang, "anlamli sayisal kayma yok.", "numerical shift in the position map today."), ha="center", va="center",
                 fontsize=11, transform=ax1.transAxes, color="#2b2b2b")
        ax1.text(0.5, 0.24, loc(lang, "Asil degisim spotun bu zone'lara yaklasip\nuzaklasmasinda okunmali.", "The main change should be read through\nspot moving closer to or away from these zones."), ha="center", va="center",
                 fontsize=9.5, transform=ax1.transAxes, color="#666")
    else:
        ax1.barh(delta_labels, delta_values, color=delta_colors, alpha=0.9)
        ax1.axvline(0, color="#444", linewidth=1)
        ax1.set_title(loc(lang, "Bugun Ne Degisti?", "What Changed Today?"), pad=8)
        ax1.set_xlabel(loc(lang, "Dunden bugune puan degisimi", "Point change vs yesterday"))
        ax1.grid(axis="x", alpha=0.2)
        pad = max(500.0, max_abs_delta * 0.18)
        ax1.set_xlim(-max_abs_delta - pad, max_abs_delta + pad)
        for idx, value in enumerate(delta_values):
            if abs(value) < 1:
                ax1.text(0.0, idx, f"{value:+,.0f}", va="center", ha="center", fontsize=8, color="#444")
            else:
                offset = max(80.0, max_abs_delta * 0.04)
                ax1.text(value + (offset if value >= 0 else -offset), idx, f"{value:+,.0f}", va="center",
                         ha="left" if value >= 0 else "right", fontsize=8)
        for label in ax1.get_yticklabels():
            label.set_fontsize(9)

    for label in ax0.get_xticklabels():
        label.set_rotation(40)
        label.set_ha("right")
    ax0.tick_params(axis="x", labelsize=8.5)
    ax1.tick_params(axis="x", labelrotation=0, labelsize=8.5)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.12, hspace=0.42)
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_gamma_support_monitor_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    history_points = section.get("history_points", [])
    forward_labels = section.get("forward_labels", [])
    forward_support_values = section.get("forward_support_values", [])
    forward_share_values = section.get("forward_share_values", [])
    forward_delta_7d = section.get("forward_delta_7d", [])
    if not history_points or not forward_labels:
        return

    dates = [point["date"] for point in history_points[-20:]]
    scores = [float(point["score"]) for point in history_points[-20:]]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.6, 7.6),
        gridspec_kw={"height_ratios": [1.7, 1.25]},
    )

    ax0 = axes[0]
    ax0.plot(dates, scores, color="#145374", linewidth=2.4, marker="o")
    ax0.axhspan(70, 100, color="#dff3e4", alpha=0.55)
    ax0.axhspan(50, 70, color="#fff3cd", alpha=0.55)
    ax0.axhspan(0, 30, color="#f8d7da", alpha=0.45)
    ax0.set_ylim(0, 100)
    ax0.set_title(loc(lang, "Pasif Destek Skoru (Son 20 Gun)", "Passive Support Score (Last 20 Days)"), pad=10)
    ax0.set_ylabel(loc(lang, "Skor", "Score"))
    ax0.grid(alpha=0.2)
    ax0.annotate(
        loc(lang, f"Bugun {scores[-1]:.0f}/100", f"Today {scores[-1]:.0f}/100"),
        xy=(dates[-1], scores[-1]),
        xytext=(-28, 10),
        textcoords="offset points",
        fontsize=8,
        color="#145374",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b8c7cf", alpha=0.9),
    )

    ax1 = axes[1]
    y = np.arange(len(forward_labels))
    colors = ["#16a34a" if delta > 0.5 else "#dc2626" if delta < -0.5 else "#64748b" for delta in forward_delta_7d]
    bars = ax1.barh(y, forward_support_values, color="#3b82f6", alpha=0.85)
    ax1.set_title(loc(lang, "Ay Bazli Destek Ozeti", "Monthly Support Summary"), pad=8)
    ax1.set_ylabel(loc(lang, "Ay penceresi", "Month window"))
    ax1.set_xlabel(loc(lang, "Bugunku destek skoru", "Current support score"))
    ax1.set_yticks(y)
    ax1.set_yticklabels(forward_labels)
    ax1.grid(axis="x", alpha=0.2)
    ax1.invert_yaxis()
    max_support = max(forward_support_values) if forward_support_values else 1.0
    ax1.set_xlim(0, max_support * 1.32)
    for idx, bar in enumerate(bars):
        share_text = f"{forward_share_values[idx]:.1f}%"
        delta = forward_delta_7d[idx] if idx < len(forward_delta_7d) else 0.0
        delta_text = loc(
            lang,
            f"7g {delta:+.1f}",
            f"7d {delta:+.1f}",
        )
        ax1.text(
            bar.get_width() + max_support * 0.03,
            bar.get_y() + bar.get_height() / 2,
            share_text,
            va="center",
            ha="left",
            fontsize=8.5,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.75),
        )
        ax1.text(
            max_support * 1.17,
            bar.get_y() + bar.get_height() / 2,
            delta_text,
            va="center",
            ha="left",
            fontsize=8.5,
            color=colors[idx],
            fontweight="bold",
        )
    ax1.text(
        max_support * 1.17,
        -0.9,
        loc(lang, "Sag kolon: 7 gunde degisim", "Right column: 7d change"),
        fontsize=8,
        color="#475569",
        ha="left",
    )

    for label in ax0.get_xticklabels():
        label.set_rotation(40)
        label.set_ha("right")
    for label in ax1.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.12, hspace=0.38)
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_gex_term_structure_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = section.get("rows", [])
    if not rows:
        return

    labels = [str(row["maturity"]) for row in rows]
    values = [float(row["gex"]) for row in rows]
    colors = ["#2f9e44" if value >= 0 else "#c0392b" for value in values]

    plt.figure(figsize=(11.6, 5.4))
    plt.bar(labels, values, color=colors, alpha=0.9)
    plt.axhline(0, color="#334155", linewidth=1.1)
    plt.title(loc(lang, "GEX Vade Yapisi", "GEX Term Structure by Expiry"))
    plt.ylabel(loc(lang, "Weighted GEX", "Weighted GEX"))
    plt.xticks(rotation=40, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_gex_term_structure_delta_chart(section: Dict[str, object], out_path: Path, lang: str = "tr") -> None:
    rows = section.get("delta_rows", [])
    if not rows:
        return

    labels = [str(row["maturity"]) for row in rows]
    values = [float(row["delta_gex"]) for row in rows]
    colors = ["#2f9e44" if value >= 0 else "#c0392b" for value in values]

    plt.figure(figsize=(11.6, 3.9))
    plt.bar(labels, values, color=colors, alpha=0.9)
    plt.axhline(0, color="#334155", linewidth=1.1)
    plt.title(loc(lang, "Dunden Bugune Vade Bazli GEX Degisimi", "Day-over-Day GEX Change by Expiry"))
    plt.ylabel(loc(lang, "GEX Delta", "GEX Delta"))
    plt.xticks(rotation=40, ha="right")
    plt.grid(axis="y", alpha=0.2)
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
        if "asagi baski artti" in lowered or "downside pressure rose" in lowered or "asagi build" in lowered or "downside build" in lowered:
            klass = "delta-down"
        elif "yukari baski artti" in lowered or "upside pressure rose" in lowered or "yukari build" in lowered or "upside build" in lowered:
            klass = "delta-up"
        elif "asagi baski azaldi" in lowered or "downside pressure fell" in lowered:
            klass = "delta-neutral"
        elif "yukari baski azaldi" in lowered or "upside pressure fell" in lowered:
            klass = "delta-neutral"
        elif "-" in item or "azaldi" in lowered or "asagi" in lowered:
            klass = "delta-down"
        elif "+" in item or "artti" in lowered or "yukari" in lowered:
            klass = "delta-up"
        badges.append(f"<span class='delta-badge {klass}'>{item}</span>")
    return "".join(badges)


def tr_to_en_text(text: str) -> str:
    replacements = [
        ("Base case: 68,000-73,000 icinde pin/range. Ilk beklenen davranis mean reversion and 72,000.", "Base case: 68,000-73,000 range with pin behavior. The first expectation is mean reversion back toward 72,000."),
        ("Spot 75,000 ustunde kalir and call akisi desteklerse yukari test aktif.", "Spot holding above 75,000 with confirming call flow activates the upside test."),
        ("Reject above 73,000: fiyat bu bolgeyi sadece igneleyip geri donerse ralliye guvenme; pin/range mantigi calisiyor demektir.", "Reject above 73,000: if price only wicks into the zone and reverses, do not trust the rally; pin/range logic remains in force."),
        ("Reject below 68,000: fiyat altina sarkip hizla geri aliniyorsa asagi kirilim degil stop-hunt / range return okunmali.", "Reject below 68,000: if price breaks below and is reclaimed quickly, read it as a stop-hunt / return-to-range move."),
        ("Bu chart gercek MM inventory degil; gamma, flow, skew, IV-RV and strategy tape uzerinden uretilen bir proxy skorudur.", "This is not true MM inventory; it is a proxy score derived from gamma, flow, skew, IV-RV, and the strategy tape."),
        ("Bu harita gercek dealer inventory degil; strike OI, GEX ve 24s strike akisindan uretilen proxy baski skorudur.", "This map is not true dealer inventory; it is a proxy pressure score derived from strike OI, GEX, and 24-hour strike flow."),
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
        ("Yonsel egilim", "Directional bias"),
        ("Fikrimi ne degistirir", "What would change my mind"),
        ("Directional bias", "Directional bias"),
        ("Asymmetry", "Asymmetry"),
        ("What would change my mind", "What would change my mind"),
        ("Gunluk Oyun Plani", "Daily Playbook"),
        ("Behavioral Flow Edge", "Behavioral Flow Edge"),
        ("Davranissal Akis Avantaji", "Behavioral Flow Edge"),
        ("Crowd signal", "Crowd signal"),
        ("Kalabalik sinyali", "Crowd signal"),
        ("Fade mi Follow mu?", "Fade or Follow?"),
        ("Confidence", "Confidence"),
        ("Guven duzeyi", "Confidence"),
        ("Bu skor neden boyle cikti:", "Why the score looks like this:"),
        ("Sabah Masa Notu", "Morning Desk Note"),
        ("Yaklasik Spot", "Approx Spot"),
        ("Pin Bandi", "Pin Band"),
        ("Ertesi Gun Vadesi", "Next-Day Expiry"),
        ("Ertesi Gun OI", "Next-Day OI"),
        ("24s OI Eklenmesi", "24h OI Added"),
        ("Position Map Okumasi", "Position Map Read"),
        ("Position Map Senaryolari", "Position Map Scenarios"),
        ("Seviyeler:", "Levels:"),
        ("Bugun Ne Degisti?", "What Changed Today?"),
        ("Karar:", "Takeaway:"),
        ("Kontrol listesi:", "Checklist:"),
        ("Dunden Bugune", "Yesterday vs Today"),
        ("Gunluk Trend", "Daily Trend"),
        ("Dealer Okumasi", "Dealer Read"),
        ("Kalabaliklasma Okumasi", "Crowding Read"),
        ("Piyasanin Fiyatladigi Sey", "What the Market Is Pricing"),
        ("Piyasanin Iskalayabilecegi Sey", "What the Market May Be Missing"),
        ("IV ve RV Karsilastirmasi", "IV vs RV Comparison"),
        ("IV Percentile ve Rank", "IV Percentile and Rank"),
        ("Orneklem", "Sample"),
        ("Normalize Edilmis 25D Skew", "Normalized 25D Skew"),
        ("Mevcut oran", "Current ratio"),
        ("Skew Okumasi", "Skew Read"),
        ("Akis Yorumu", "Flow Read"),
        ("Akis Grafigi", "Flow Chart"),
        ("Gunluk / Haftalik / Aylik Flow Notu", "Daily / Weekly / Monthly Flow Note"),
        ("Gunluk:", "Daily:"),
        ("Haftalik:", "Weekly:"),
        ("Aylik:", "Monthly:"),
        ("Strateji Akis Yorumu", "Strategy Flow Interpretation"),
        ("Strateji Karari", "Strategy Verdict"),
        ("Baskin Aksiyon", "Dominant Action"),
        ("Trade Yansimasi", "Trading Implication"),
        ("Strateji Aksiyon Haritasi", "Strategy Action Map"),
        ("Bu Ne Anlama Geliyor?", "What Does It Mean?"),
        ("Vade Yapisi", "Expiry Structure"),
        ("En Yakin Vade Detayi", "Nearest Expiry Detail"),
        ("Expiry Kapanis Okumasi", "Expiry Close Read"),
        ("Yakin Vade Nasil Okunur?", "How To Read The Front Expiry"),
        ("On Vade Karari", "Front Expiry Verdict"),
        ("Beklenen Ilk Etki", "Expected First Effect"),
        ("Teyit Sarti", "Confirmation Needed"),
        ("Negatif Tetik", "Negative Trigger"),
        ("Pozisyonlama", "Positioning"),
        ("Premium", "Premium"),
        ("Expiry Sonrasi Etki", "Post-Expiry Implication"),
        ("Kisa Ozet", "Quick Summary"),
        ("Gec Put Panigi", "Gec Put Panigi"),
        ("Gec Call Kovalamasi", "Gec Call Kovalamasi"),
        ("Bilgili Savunmaci Hedge", "Bilgili Savunmaci Hedge"),
        ("Bilgili Yukari Konumlanma", "Bilgili Yukari Konumlanma"),
        ("Notr / Karisik Tape", "Notr / Karisik Tape"),
        ("Spot kirilim seviyesinin altinda kabul gormeden asagi panigi fade et", "Spot kirilim seviyesinin altinda kabul gormeden asagi panigi fade et"),
        ("Spot squeeze seviyesi ustunde kabul gormeden yukari kovalamayi fade et", "Spot squeeze seviyesi ustunde kabul gormeden yukari kovalamayi fade et"),
        ("Kirilim seviyeleri gecerli kaldikca savunmaci konumlanmayi takip et", "Kirilim seviyeleri gecerli kaldikca savunmaci konumlanmayi takip et"),
        ("Yukarida sadece temiz acceptance ile takip et", "Yukarida sadece temiz acceptance ile takip et"),
        ("Temiz bir davranissal edge yok", "Temiz bir davranissal edge yok"),
        ("Orta", "Orta"),
        ("Yuksek", "Yuksek"),
        ("Dusuk", "Dusuk"),
        ("Gec Put Panigi", "Late put panic"),
        ("Gec Call Kovalamasi", "Late call chase"),
        ("Bilgili Savunmaci Hedge", "Smart-money defensive hedge"),
        ("Bilgili Yukari Konumlanma", "Smart-money upside positioning"),
        ("Notr / Karisik Tape", "Neutral / mixed tape"),
        ("Spot kirilim seviyesinin altinda kabul gormeden asagi panigi fade et", "Fade downside panic unless spot is accepted below the breakdown trigger"),
        ("Spot squeeze seviyesi ustunde kabul gormeden yukari kovalamayi fade et", "Fade upside chasing unless spot is accepted above the squeeze trigger"),
        ("Kirilim seviyeleri gecerli kaldikca savunmaci konumlanmayi takip et", "Follow defensive positioning while the breakdown levels remain valid"),
        ("Yukarida sadece temiz acceptance ile takip et", "Only follow upside if there is clean acceptance"),
        ("Temiz bir davranissal edge yok", "There is no clean behavioral edge"),
        ("Orta", "Moderate"),
        ("Yuksek", "High"),
        ("Dusuk", "Low"),
        ("Risk notu", "Risk note"),
        ("Ana senaryo", "Base case"),
        ("Yukari tarafta temiz acceptance icin en az 75,000 ustu gerekir; asagi tarafta ise 70,000 alti kirilma tek basina daha hizli hedge akisina donebilir.", "Upside needs clean acceptance above 75,000; on the downside, a break below 70,000 can turn into faster hedge flow by itself."),
        ("Bu gorusum 75,000 ustunde kalici acceptance ve belirgin call eklenmesiyle daha pozitif; 70,000 alti kabul ile daha negatif olur.", "I would turn more constructive on sustained acceptance above 75,000 with visible call adding; I would turn more negative on acceptance below 70,000."),
        ("Alternatif senaryo", "Alternative case"),
        ("Gecersizlik seviyesi", "Invalidation level"),
        ("Notrden hafif negatife", "Neutral to slightly bearish"),
        ("Notrden hafif pozitife", "Neutral to slightly bullish"),
        ("Yon olarak notr ama kirilgan", "Directionally neutral but fragile"),
        ("Yukari senaryo kosullu ve teyit istiyor; asagi taraf ise kirilma halinde daha kolay hizlanabilir.", "Upside is conditional and needs confirmation; downside can accelerate more easily once the structure breaks."),
        ("Asagi hareketler hedge akisiyla daha kolay bastirilabilir; temiz yukari devam icin yine de acceptance gerekli.", "Downside moves can still be absorbed by hedging flows more easily; clean upside continuation still requires acceptance."),
        ("Piyasa net trend fiyatlamiyor; ancak simdilik yukari senaryo teyit isterken asagi taraf daha az teyitle calisabilir.", "The market is not pricing a clear trend; for now, upside requires confirmation while downside can work with less confirmation."),
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
        ("temiz acceptance icin en az ", "clean acceptance requires at least "),
        (" ustu gerekir; asagi tarafta ise ", " above; on the downside, "),
        (" alti kirilma tek basina daha hizli hedge akisina donebilir.", " below can trigger faster hedge flow on its own."),
        ("Call akisi tek basina euforia gibi gorunmuyor; yine de predictive guc kazanmasi icin ", "Call flow does not look euphoric on its own; still, for it to gain predictive power, "),
        (" ustunde acceptance and devam eden call eklenmesi gerekli.", " needs acceptance above and continued call adding."),
        ("Bu bolum dogrudan vanna/charm greeklerini hesaplamaz; yakin vade OI yogunlugu, pozitif gamma, pin merkezine yakinlik and IV priminden turetilen bir proxy'dir.", "This section does not calculate direct vanna/charm greeks; it is a proxy built from near-dated OI concentration, positive gamma, proximity to the pin center, and IV premium."),
        (" and kalan sure yaklasik ", " with roughly "),
        (" gun. 3 gun icinde vadesi dolacak OI payi %", " days remaining. The share of OI expiring within 3 days is "),
        (", 7 gun icinde %", ", and within 7 days it is "),
        (".", "."),
        ("Yakin 7 gundeki quarterly vade yogunlugu toplam OI'nin %", "Quarterly expiry concentration over the next 7 days is "),
        ("'i.", "% of total OI."),
        ("Charm proxy ", "Charm proxy "),
        (": yakin vade OI yogunlugu yuksek and dealer gamma ile birlikte spotta zaman-decay kaynakli pin baskisi artabilir.", ": near-dated OI is elevated and, together with dealer gamma, time decay can increase pin pressure on spot."),
        ("Vanna proxy ", "Vanna proxy "),
        (": front IV'nin realized vola gore primi (", ": front IV premium versus realized vol ("),
        (") and gamma tonu, vol geri cekilirse hedge akislarinin spota dengeleyici etki verme potansiyelini sinirli tutuyor.", ") and the gamma tone suggest that if vol mean-reverts lower, hedge flows may only provide limited stabilizing support."),
        ("Net okuma Orta compression / kontrollu fiyatlama. Spot pin merkezine yakinlik skoru %", "Net read: moderate compression / controlled pricing. Spot-to-pin proximity score is "),
        (", gamma destegi %", ", gamma support is "),
        ("Breakout kalitesi weaklar; yukari extension gorursen bile cap riski nedeniyle acceptance ara.", "Breakout quality weakens; even if you see upside extension, wait for acceptance because cap risk remains."),
        ("Tape ayni anda hem downside hedge hem de upside cap anlatiyor; yon var ama rahat degil.", "The tape is showing both downside hedge demand and upside capping at the same time; there is direction, but it is not comfortable."),
        ("On vade ATM IV (", "Front-end ATM IV ("),
        ("; while back-end ATM IV is (", ", while back-end ATM IV is ("),
        ("). On taraf pahali and term structure belirgin backwardation gosteriyor.", "). The front end is rich and the term structure shows clear backwardation."),
        ("Front ATM IV is is ile 7-day realized vol arasindaki fark ", "The gap between front ATM IV and 7-day realized vol is "),
        (" puan. Vol market ile spot hareketi birbirinden kopuk degil.", " points. The vol market is not materially disconnected from spot movement."),
        ("25d RR ", "25d RR "),
        ("; downside putlar call'lara gore anlamli primli. Bu savunmaci tail hedge talebini gosteriyor.", "; downside puts remain meaningfully richer than calls. That points to defensive tail-hedge demand."),
        ("10d RR ", "10d RR "),
        (" and 25d BF ", " and 25d BF "),
        ("; bu kombinasyon kisa vadede tail priminin ne kadar yuksek oldugunu gosteriyor.", "; that combination shows how elevated short-dated tail premium remains."),
        ("Asimetri:", "Asymmetry:"),
        ("This chart gercek MM inventory degil; gamma, flow, skew, IV-RV and strategy tape uzerinden uretilen bir proxy skorudur.", "This chart is not true MM inventory; it is a proxy score built from gamma, flow, skew, IV-RV, and the strategy tape."),
        (" icinde pin/range. Ilk beklenen davranis mean reversion and ", " range with pin behavior. The first expectation is mean reversion back toward "),
        (" seviyesine uzayabilir. Bu senaryo kosullu; temiz trend icin yeni call eklenmesi and flow tonunun belirgin iyilesmesi lazim.", ". This scenario is conditional; it still needs fresh call adding and a clear improvement in flow tone for a clean trend."),
        ("Asagi taraf simdilik daha az teyitle aktive olabilir.", "Downside can still activate with less confirmation for now."),
        ("fiyat bu bolgeyi sadece igneleyip geri donerse ralliye guvenme; pin/range mantigi calisiyor demektir.", "if price only wicks into the zone and reverses, do not trust the rally; pin/range logic is still in force."),
        ("spot bu seviye altinda kalirsa downside hedge akisi kuvvetlenir and ", "if spot holds below the level, downside hedge flow should strengthen and "),
        (" ikinci hedef olur.", " becomes the next downside target."),
        ("fiyat altina sarkip hizla geri aliniyorsa asagi kirilim degil stop-hunt / range return okunmali.", "if price trades below and is quickly reclaimed, read it as a stop-hunt / return-to-range move."),
        ("Put/Call dengesizligi", "Put/Call imbalance"),
        ("Spota gore strike konumu", "Strike location vs spot"),
        ("Premium stresi", "Premium stress"),
        ("Skew stresi", "Skew stress"),
        ("Call eklenmesi put eklenmesinden belirgin strong.", "Call adding is clearly stronger than put adding."),
        ("Baskin yeni risk spotun altindaki put strike'larinda birikiyor.", "Dominant new risk is clustering in put strikes below spot."),
        ("IV-RV farki bu sinyali asiri guclendirmiyor.", "The IV-RV spread does not materially amplify this signal."),
        ("Skew tarafinda asiri tek yonlu baski yok.", "There is no extreme one-way pressure in skew."),
        ("Kalabalik Kovalama Skoru", "Crowd Chase Score"),
        ("Akis Skoru", "Flow Score"),
        ("Skew Gosterge", "Skew Gauge"),
        ("Bu grafik skorun hangi bilesenlerden geldigini gosterir; yani sinyalin sadece tek bir etiketten ibaret olmadigini aciklar.", "This chart shows which components are driving the score, so the signal is not reduced to a single label."),
        ("Bu grafik kalabaligin gec put panigi mi yoksa gec call kovalamasi mi yaptigini zaman icinde daha net okumayi saglar.", "This chart helps distinguish whether the crowd has been in late put panic or late call chasing over time."),
        ("On the downside, ise ", "On the downside, "),
        ("Front ATM IV is is ", "Front ATM IV is "),
        ("Daha onemlisi, ", "More importantly, "),
        ("and baskin ton constructive / upside chase akisi.", "and the dominant tone is constructive / upside chase flow."),
        (" alti sadece teknik weaklik degil, opsiyon rejim degisimi anlamina gelir; orada hedge akisinin hizi artar and spot beklenenden daha hizli asagi tasinabilir.", " below is not just technical weakness; it signals an options regime shift, where hedge flow can accelerate and push spot lower faster than expected."),
        ("Put hedge bolgesi ", "The put hedge zone is "),
        ("; bu alan asagi yonlu korumanin toplandigi ana zone gibi okunmali. En strong put cebi yaklasik ", "; read this as the main downside protection zone. The strongest put pocket sits around "),
        ("Magnet strike ", "Magnet strike is "),
        ("; spot bu seviyeye yaklastikca pin / geri cekilme etkisi daha anlamli hale gelebilir.", "; as spot approaches this level, pin / pullback effects can matter more."),
        ("Call positioning bandi ", "The call positioning band is "),
        ("; bu zonun ust tarafinda acceptance gelirse yukari uzama alani acilir. En strong call yogunlugu ", "; acceptance near the upper edge of the zone opens room for upside extension. The strongest call concentration sits near "),
        ("Speculation calls bandi ", "The speculation-calls band is "),
        ("; bu alan daha uzak OTM yukari beklentiyi gosterir. En strong spekulatif call strike'i ", "; this zone represents more distant OTM upside expectation. The strongest speculative call strike sits at "),
        ("Spot 75,000 altinda; yukari hareketin daha temiz pozitif okumaya donmesi icin once magnet strike geri alinmali.", "Spot is below 75,000; for the upside read to turn cleaner, price first needs to reclaim the magnet strike."),
        ("Tail hedge ", "Tail hedge is "),
        ("; bu seviye gunluk trade seviyesi degil, daha cok uzak kuyruk risk referansi. En strong uzak kuyruk putu ", "; this is not a day-trading level, but a distant tail-risk reference. The strongest far-tail put sits near "),
        ("Magnet strike degismedi and ", "Magnet strike did not change and remained at "),
        ("Put hedge zone degismedi and ", "The put hedge zone did not change and remained at "),
        (" bandinda kaldi.", "."),
        ("Bu grafik bugunku zone'lari tum gecmise yaymaz; gunluk arsivden magnet, put hedge, call cluster and speculation seviyelerinin nasil kaydigini gosterir.", "This chart does not project today's zones across all history; it shows how magnet, put-hedge, call-cluster, and speculation levels have shifted in the daily archive."),
        ("magnet sabit | put zone sabit | call cluster sabit", "magnet unchanged | put zone unchanged | call cluster unchanged"),
        ("Put hedge zone merkezi ayni kaldi (", "The center of the put hedge zone was unchanged ("),
        ("Call cluster degismedi and ", "The call cluster did not change and remained at "),
        ("Call cluster spota yaklasti (", "The call cluster moved closer to spot ("),
        ("; yakin upside ilgisi artti.", "); near-upside interest increased."),
        ("Speculation seviyesi sabit and ", "The speculation level was unchanged at "),
        ("Moderate compression / kontrollu fiyatlama", "Moderate compression / controlled pricing"),
        ("Decay flow etkisi weak-orta; tek basina pinning beklemek icin yeterli degil, seviye teyidi gerekli.", "Decay-flow pressure is weak to moderate; it is not strong enough on its own to justify a pinning assumption, so level confirmation is still required."),
        ("Net okuma Moderate compression / kontrollu fiyatlama. Spot pin merkezine yakinlik skoru %", "Net read: moderate compression / controlled pricing. Spot-to-pin proximity score is "),
        ("Bu chart gercek greeks degil; yakin vade expiry yogunlugu, quarterly baski, gamma destegi, pin merkezine yakinlik and IV primini tek panelde toplar.", "This chart does not show true greeks; it combines near-dated expiry concentration, quarterly pressure, gamma support, proximity to the pin center, and IV premium in a single panel."),
        ("Spot 71,000-76,000 bandinda kaldikca compression proxy daha anlamli calisir.", "The compression proxy matters more while spot remains inside the 71,000-76,000 band."),
        ("75,000 ustu acceptance gelirse pin baskisi weaklar, proxy'nin yonlendiriciligi duser.", "Acceptance above 75,000 weakens pin pressure and reduces the proxy's directional value."),
        ("IV-RV spread hizla kapanirsa vanna bileseni weaklar; ", "If the IV-RV spread closes quickly, the vanna component weakens; watch "),
        ("Vol egirisi gorece duz, yani yakin vade panik fiyati baskin degil.", "The vol curve is relatively flat, so front-end panic pricing is not dominant."),
        ("; skew savunmaci ama panik boyutunda degil. Piyasa asagiyi ciddiye aliyor fakat tek yonlu korku fiyatlamiyor.", "; skew is defensive but not outright panic. The market is taking downside seriously without pricing one-way fear."),
        ("Long straddle/strangle varligi, bir kisim oyuncunun yon degil realized hareket satin aldigini gosteriyor.", "The presence of long straddles/strangles suggests that some participants are buying realized movement rather than pure direction."),
        ("Savunmaci", "Defensive"),
        ("LONG_PUT tarafinda ", "On the LONG_PUT side, "),
        (" OI artisi var; bu, asagi yonlu korumanin taze hedge olarak eklendigini gosteriyor.", " of OI was added; this points to fresh downside hedging."),
        ("SHORT_CALL OI artisi ", "SHORT_CALL OI increased by "),
        ("; bu, yukari hareketin call satarak cap'lendigini and temiz breakout beklentisinin sinirli kaldigini gosteriyor.", "; this suggests upside is being capped through call selling and that clean breakout conviction remains limited."),
        ("LONG_CALL OI artisi ", "LONG_CALL OI increased by "),
        ("; tape'in bir parcasi yukari risk tekrar insa ediyor, ancak bu tek basina euforia okumasi degil.", "; part of the tape is rebuilding upside risk, but that does not amount to standalone euphoria."),
        ("SHORT_PUT OI artisi ", "SHORT_PUT OI increased by "),
        ("; oyuncularin bir kismi downside'i satip carry toplamaya devam ediyor.", "; some participants are still selling downside and collecting carry."),
        ("Bu grafik strateji isimlerini degil, tape'in hangi aksiyonlarda yogunlastigini gosterir. Soldaki tabloyu hizli okumak icin kullan.", "This chart does not summarize strategy names; it shows which actions dominate the tape. Use it to read the table quickly."),
        ("Mevcut strategy endpoint'i leg-level veri vermedigi icin bu bolum tahmini transfer okumasidir.", "Because the current strategy endpoint does not expose leg-level data, this section is an inferred transfer read."),
        ("Put tarafinda en strong ekleme ", "On the put side, the strongest addition was "),
        (" en strong bosalma ", " while the strongest reduction was "),
        ("; bu, korumanin ", "; this suggests protection may be moving from "),
        (" vadesinden ", " to "),
        (" vadesine tasiniyor olabilecegini dusunduruyor.", "."),
        ("Put strike transferi tahminen ", "The inferred put-strike transfer is "),
        (" yonunde asagiya; bu da hedge seviyesinin yeniden konumlandigini ima eder.", " lower, which implies the hedge level is being repositioned."),
        ("Call tarafinda en strong ekleme ", "On the call side, the strongest addition was "),
        ("; bu, upside pozisyonunun ", "; this suggests upside positioning may be moving from "),
        (" vadesine aktariliyor olabilecegini gosteriyor.", "."),
        ("Call strike transferi tahminen ", "The inferred call-strike transfer is "),
        (" yonunde yukariya; bu, upside target/cap seviyesinin yer degistirdigine isaret edebilir.", " higher, which may indicate that the upside target/cap level has shifted."),
        ("Net strike akis merkezi ", "The center of net strike flow moved from "),
        ("'dan ", " to "),
        ("'a yukariya kayiyor; bu da yeni ilginin hangi strike'a toplandigini gosteriyor.", ", which shows where new attention is concentrating."),
        ("Roll / extend: koruma kapatilmiyor, daha ileri vadeye tasiniyor olabilir.", "Roll / extend: protection may not be closing; it may be moving further out the curve."),
        ("Bu chart mevcut GEX, strike OI and 24s strike akisini birlestiren bir proxy haritadir. Gercek MM inventory degildir; hangi strike'larda yukari/asagi baski yogunlasiyor onu gosterir.", "This chart is a proxy map that combines current GEX, strike OI, and 24h strike flow. It is not true MM inventory; it shows where upside/downside pressure is concentrated across strikes."),
        ("Bu, opsiyon primlerinin son spot hareketine gore primli oldugunu gosteriyor.", "This shows that options premia are rich relative to the latest spot move."),
        ("A sustained break below ", ". A sustained break below "),
        ("Magnet strike is was unchanged at ", "Magnet strike was unchanged at "),
        ("Davranissal bilesenler", "Behavioral components"),
        ("Davranissal akis skoru trendi", "Behavioral flow score trend"),
        ("Gunluk trend", "Daily trend"),
        ("IV rejim trendi", "IV regime trend"),
        ("Normalize edilmis 25D skew", "Normalized 25D skew"),
        ("Akis degisimi grafigi", "Flow change chart"),
        ("Taze hedge + upside cap", "Fresh hedge + upside cap"),
        ("Constructive / upside chase akisi", "Constructive / upside-chase flow"),
        ("Akis Skoru", "Flow Score"),
        ("Davranissal Akis Avantaji", "Behavioral Flow Edge"),
        ("Kalabalik Kovalama Skoru", "Crowd Chase Score"),
        ("Bugun Ne Degisti?", "What Changed Today?"),
        ("Dunden Bugune", "Yesterday vs Today"),
        ("Akis Yorumu", "Flow Read"),
        ("Akis Grafigi", "Flow Chart"),
        ("Strateji Akis Yorumu", "Strategy Flow Read"),
        ("Strateji Karari", "Strategy Verdict"),
        ("Baskin Aksiyon", "Dominant Action"),
        ("Trade Yansimasi", "Trading Implication"),
        ("Strateji Aksiyon Haritasi", "Strategy Action Map"),
        ("Tahmini Roll / Transfer", "Estimated Roll / Transfer"),
        ("Bu Ne Anlama Geliyor?", "What Does It Mean?"),
        ("Vade Yapisi", "Expiry Structure"),
        ("Acik Pozisyon", "Open Interest"),
        ("Call OI (BTC)", "Call OI (BTC)"),
        ("Put OI (BTC)", "Put OI (BTC)"),
        ("Call Notional", "Call Notional"),
        ("Put Notional", "Put Notional"),
        ("OI Degisimi", "OI Change"),
        ("Spot Cevresinde Strike Yigilmasi", "Strike Clustering Around Spot"),
        ("Position Map Okumasi", "Position Map Read"),
        ("Position Map Senaryolari", "Position Map Scenarios"),
        ("Normalize Edilmis 25D Skew", "Normalized 25D Skew"),
        ("Skew Okumasi", "Skew Read"),
        ("Pain Drift", "Pain Drift"),
        ("Gunluk Pain Kaymasi", "Daily Pain Shift"),
        ("5 Gunluk Pain Kaymasi", "5-Day Pain Shift"),
        ("Vade", "Expiry"),
        ("Notional BTC", "Notional BTC"),
        ("Vol tek basina edge vermiyor; yon/olay filtresi ekle.", "Vol alone does not create edge; add a direction/event filter."),
        ("Percentile ile rank ayni hikayeyi mi anlatiyor?", "Do percentile and rank tell the same story?"),
        ("Stratejiyi tekli vol gorusu yerine secici yapi olarak kur.", "Prefer selective structures over a single outright vol view."),
        ("Normalize oran yuksek kaldikca downside korumasi call talebine gore daha pahali kalir.", "As long as the normalized ratio stays elevated, downside protection remains more expensive than call demand."),
        ("Bugunun aggregate max pain seviyesi ", "Today's aggregate max-pain level is "),
        ("Spot ile max pain arasi fark ", "The gap between spot and max pain is "),
        ("Dunden bugune pain kaymasi ", "The daily pain shift is "),
        (" spot-pain fark degisimi ", " and the change in the spot-pain gap is "),
        ("5 gunluk pain kaymasi ", "The 5-day pain shift is "),
        ("Heatmap spot cevresinde call and put akisinin tam olarak hangi strike'larda yogunlastigini gosterir.", "The heatmap shows exactly where call and put flow is concentrated around spot."),
        ("Son 5 gunluk yerel kayitta akis en cok 82,000 and 73,000 strike'larinda birikmis.", "In the local 5-day sample, flow has concentrated most around the 82,000 and 73,000 strikes."),
        ("IV and RV Karsilastirmasi", "IV vs RV Comparison"),
        ("Strateji Flow Read", "Strategy Flow Read"),
        ("Tahmini Roll / Transfer", "Estimated Roll / Transfer"),
        ("Fresh hedge: yeni downside risk aliniyor ya da mevcut long risk korunuyor olabilir.", "Fresh hedge: new downside risk may be getting added, or existing long risk may be getting protected."),
        ("Cap / overwrite: yukari hareket satilarak sinirlanmak isteniyor olabilir.", "Cap / overwrite: upside may be getting sold to limit extension."),
        ("Unwind: onceki cap ya da hedge yapilari gevsetiliyor olabilir.", "Unwind: prior cap or hedge structures may be getting relaxed."),
        ("Defined-risk bullish: oyuncular yukariyi kovalamak yerine spread gibi sinirli riskli yapilari tercih ediyor olabilir.", "Defined-risk bullish: participants may prefer limited-risk structures such as spreads instead of chasing upside outright."),
        ("En strong yukari merkezler ", "The strongest upside centers are "),
        ("; bu strike'larda call agirligi, pozitif gamma ve/veya taze akis birlikte yogunlasiyor.", "; these strikes combine call-heavy positioning, positive gamma, and/or fresh flow."),
        ("En strong asagi hedge cepleri ", "The strongest downside hedge pockets are "),
        ("; bu alanlar downside korumasinin en yogun kaldigi bolgeler.", "; these zones are where downside protection remains most concentrated."),
        ("Son 24 saatte en belirgin zincir kaymasi ", "The clearest chain shift over the last 24 hours occurred at "),
        (" strike'inda yukari yone oldu ", " on the upside "),
        ("; bu seviye kisa vadede yeni ilgi merkezi gibi okunmali.", "; this level should be read as a new short-dated focus area."),
        ("Dealerlar toplamda hala net long gamma tasiyor; bu nedenle hedge akislarinin spotun bulundugu bolgede gun ici hareket araligini bastirmasi beklenir.", "Dealers are still carrying net long gamma overall; that means hedging flow is expected to suppress the intraday range around spot."),
        ("On taraf pahali and term structure belirgin backwardation gosteriyor.", "The front end is rich and the term structure shows clear backwardation."),
        ("Vol ne net ucuz ne net pahali; secici olmak and yon/olay baglamiyla birlikte okumak daha dogru.", "Vol is neither clearly cheap nor clearly expensive; it is better to stay selective and read it alongside direction/event context."),
        ("Percentile'i alim/satim filtresi, rank'i ise ne kadar ekstrem bolgedeyiz filtresi olarak okuyun.", "Use percentile as the buy/sell-vol filter and rank as the measure of how extreme the regime is."),
        ("Because the current strategy endpoint does not expose leg-level data, this section is an inferred transfer read.", "Because the current strategy endpoint does not expose leg-level data, this section is an inferred transfer read."),
        ("Gamma rejimi", "Gamma regime"),
        ("Flow baskisi", "Flow pressure"),
        ("Skew savunmasi", "Skew defense"),
        ("IV-RV gerilimi", "IV-RV tension"),
        ("Yakin 3g OI", "Near 3d OI"),
        ("7g expiry OI", "7d expiry OI"),
        ("Quarterly 7g", "Quarterly 7d"),
        ("Gamma destegi", "Gamma support"),
        ("Pin yakinligi", "Pin proximity"),
        ("IV primi", "IV premium"),
        ("Yukari risk yeniden insasi", "Upside risk rebuild"),
        ("Taze hedge", "Fresh hedge"),
        ("Strateji Flow Read", "Strategy Flow Read"),
        ("The strongest strategy pressure came from ", "The strongest strategy pressure came from "),
        ("Fresh hedge + upside cap", "Fresh hedge + upside cap"),
        ("Koruma vade boyunca tasiniyor olabilir", "Protection may be moving further out the curve"),
        ("Upside pozisyonu daha yakin / farkli vadeye tasiniyor olabilir", "Upside positioning may be moving to a nearer / different expiry"),
        ("Bugun tek baskin transfer infer edildi", "A single dominant transfer was inferred today"),
        ("Kirmizi: hedge / savunma | Yesil: call yeniden konumlanma", "Red: hedge / defense | Green: call repositioning"),
        ("Kirmizi: savunmaci / cap | Yesil: yapici", "Red: defensive / cap | Green: constructive"),
        ("IV primi", "IV premium"),
        ("Pin yakinligi", "Pin proximity"),
        ("Gamma destegi", "Gamma support"),
        ("Yakin 3g OI", "Near 3d OI"),
        ("Quarterly 7g", "Quarterly 7d"),
        ("7g expiry OI", "7d expiry OI"),
        ("Yukari risk yeniden insasi", "Upside risk rebuild"),
        ("Taze hedge", "Fresh hedge"),
        ("Akis ve strategy tape bu resmi tamamliyor.", "Flow and the strategy tape complete the picture."),
        ("Skew ve smile verisi halen savunmaci bir alt ton tasiyor.", "Skew and smile data still carry a defensive undertone."),
        ("Skew savunmaci and downside kuyruk riskini pahali tutuyor.", "Skew is defensive and keeps downside tail risk expensive."),
        ("Bu chart gercek greeks degil; yakin vade expiry yogunlugu, quarterly baski, gamma destegi, pin merkezine yakinlik ve IV primini tek panelde toplar.", "This chart does not show true greeks; it combines near-dated expiry concentration, quarterly pressure, gamma support, proximity to the pin center, and IV premium in one panel."),
        ("Bu grafik strateji isimlerini degil, tape'in hangi aksiyonlarda yogunlastigini gosterir. Soldaki tabloyu hizli okumak icin kullan.", "This chart does not summarize strategy names; it shows which actions dominate the tape. Use it to read the table quickly."),
        ("Bu grafik bugunku zone'lari tum gecmise yaymaz; gunluk arsivden magnet, put hedge, call cluster ve speculation seviyelerinin nasil kaydigini gosterir.", "This chart does not project today's zones across all history; it shows how magnet, put-hedge, call-cluster, and speculation levels have shifted in the daily archive."),
        ("Bu grafik skorun hangi bilesenlerden geldigini gosterir; yani sinyalin sadece tek bir etiketten ibaret olmadigini aciklar.", "This chart shows which components are driving the score, so the signal is not reduced to a single label."),
        ("Bu grafik kalabaligin gec put panigi mi yoksa gec call kovalamasi mi yaptigini zaman icinde daha net okumayi saglar.", "This chart helps read more clearly over time whether the crowd is late-put panicking or late-call chasing."),
        ("Bu grafik son 24 saatte hangi strike'larda yeni risk eklendigini ve hangi strike'larda riskin bosaldigini en hizli sekilde gosterir.", "This chart shows, as quickly as possible, where new risk was added and where risk was reduced across strikes over the last 24 hours."),
        ("Heatmap spot cevresinde call ve put akisinin tam olarak hangi strike'larda yogunlastigini gosterir.", "The heatmap shows exactly where call and put flow is concentrated around spot."),
        ("Bu chart mevcut GEX, strike OI ve 24s strike akisini birlestiren bir proxy haritadir. Gercek MM inventory degildir; hangi strike'larda yukari/asagi baski yogunlasiyor onu gosterir.", "This chart is a proxy map that combines current GEX, strike OI, and 24h strike flow. It is not true MM inventory; it shows where upside/downside pressure is concentrated across strikes."),
        ("Bu chart gercek MM inventory degil; gamma, flow, skew, IV-RV ve strategy tape uzerinden uretilen bir proxy skorudur.", "This chart is not true MM inventory; it is a proxy score built from gamma, flow, skew, IV-RV, and the strategy tape."),
        ("Yerel arsiv buyudukce bu grafik MM/dealer tonunun zaman icinde daha net izlenmesini saglar.", "As the local archive grows, this chart makes the MM/dealer tone easier to track over time."),
        ("Gunluk trend", "Daily trend"),
        ("Gunluk / Haftalik / Aylik Flow Notu", "Daily / Weekly / Monthly Flow Note"),
        ("Gunluk:", "Daily:"),
        ("Haftalik:", "Weekly:"),
        ("Aylik:", "Monthly:"),
        ("IV ve RV Karsilastirmasi", "IV vs RV Comparison"),
        ("Laevitas historical RV endpoint'i bu key ile kapali oldugu icin realized vol public BTC gunluk kapanis serisinden hesaplanir.", "Because the Laevitas historical RV endpoint is unavailable on this key, realized vol is calculated from the public BTC daily close series."),
        ("IV Percentile ve Rank", "IV Percentile and Rank"),
        ("Orneklem", "Sample"),
        ("Skew Gosterge", "Skew Gauge"),
        ("Yaklasik Spot", "Approx. Spot"),
        ("Pin Bandi", "Pin Band"),
        ("Ertesi Gun Vadesi", "Next-Day Expiry"),
        ("Ertesi Gun OI", "Next-Day OI"),
        ("Call OI", "Call OI"),
        ("Put OI", "Put OI"),
        ("Put Premium", "Put Premium"),
        ("24s OI Eklenmesi", "24h OI Add"),
        ("On vade put-agir ve koruma maliyeti belirgin.", "The front expiry is put-heavy and protection costs are clearly dominant."),
        ("On vade call-agir; upside ilgisi daha baskin.", "The front expiry is call-heavy; upside interest is more dominant."),
        ("On vade karisik; call/put dagilimi dengeliye yakin.", "The front expiry is mixed; the call/put split is close to balanced."),
        ("On vade karisik.", "The front expiry is mixed."),
        ("On vade pozisyonlamasi put-agir.", "Front-expiry positioning is put-heavy."),
        ("On vade pozisyonlamasi call-agir.", "Front-expiry positioning is call-heavy."),
        ("Premium dagilimi dengeliye yakin.", "Premium distribution is close to balanced."),
        ("Premium okuması savunmaci; odemenin buyuk kismi put tarafinda.", "Premium is defensive; most of the spend is on the put side."),
        ("Premium okuması yapici; odemenin buyuk kismi call tarafinda.", "Premium is constructive; most of the spend is on the call side."),
        ("Expiry sonrasi destek korunursa hedge primi cozulerek kisa vadeli rahatlama alani acilabilir.", "If support holds after expiry, hedge premium can unwind and open room for a short-term relief move."),
        ("Spot zayif kalirsa expiry sonrasi bile downside baskisi hemen kaybolmayabilir.", "If spot stays weak, downside pressure may not disappear immediately even after expiry."),
        ("Call-agir on vade ve yukarida kalan spot, expiry sonrasi yukari devam potansiyelini destekler.", "A call-heavy front expiry with spot holding higher supports post-expiry upside continuation potential."),
        ("Expiry kapanisi tek basina yon vermez; asil okuma spotun kilit seviyelere gore nerede tutuldugudur.", "Expiry on its own does not determine direction; the real read comes from where spot is held relative to key levels."),
        ("Baz senaryo hafif relief; expiry sonrasi put primi cozulerek ilk tepki yukari olabilir, ancak bunun kalici bullish devama donmesi otomatik degil.", "Base case is mild relief; after expiry, put premium can unwind and allow an initial bounce, but that does not automatically turn into sustained bullish continuation."),
        ("Baz senaryo relief degil, savunmaci tonun surmesi; expiry gecse bile downside baskisi hemen kaybolmayabilir.", "Base case is not relief but persistence of the defensive tone; downside pressure may not disappear immediately even after expiry."),
        ("Baz senaryo yukari devam; front expiry call-agir kaldigi icin expiry sonrasi cap/hedge temizlenmesi yukari uzamayi destekleyebilir.", "Base case is upside continuation; with the front expiry remaining call-heavy, post-expiry cap/hedge clearing can support further upside extension."),
        ("Baz senaryo ilk anda relief denemesi; ancak premium savunmaci kaldigi icin bu pencere teyitsiz kalirsa hareket kolayca tekrar range/savunma moduna donebilir.", "Base case is an initial relief attempt; however, because premium remains defensive, the move can easily fall back into range/defense mode if it is not confirmed."),
        ("On vade put-agir and koruma maliyeti belirgin.", "The front expiry is put-heavy and protection costs are clearly dominant."),
        ("On vade call-agir; upside interest daha baskin.", "The front expiry is call-heavy; upside interest is more dominant."),
        ("On vade karisik; call/put dagilimi close to balanced.", "The front expiry is mixed; the call/put split is close to balanced."),
        ("Gunluk ATM IV degisimi:", "Daily ATM IV change:"),
        ("Haftalik ATM IV degisimi:", "Weekly ATM IV change:"),
        ("Baskin ton:", "Dominant tone:"),
        ("Net ton:", "Net tone:"),
        ("Kontrol listesi:", "Checklist:"),
        ("Mevcut oran:", "Current ratio:"),
        ("Flow skoru:", "Flow score:"),
        ("Front maturity:", "Front maturity:"),
        ("Percentile, bugunku implied volun yerel seriye gore ne kadar sik pahali bolgede oldugunu; rank ise min-max araligi icinde ne kadar yukarida oldugunu gosterir.", "Percentile shows how often today's implied vol sits in an expensive zone versus the local sample; rank shows how high it sits inside the min-max range."),
        ("Bu metrik, 25 delta put vol ile 25 delta call vol farkinin ATM IV'ye oranidir. Oran yukselirse piyasa putlari call'lara gore daha pahali fiyatliyor demektir.", "This metric is the ratio of the 25-delta put vol minus the 25-delta call vol to ATM IV. When the ratio rises, the market is pricing puts richer than calls."),
        ("The center of that range read sits at ", "The center of that range read sits at "),
        (". because both gamma magnets", " because both gamma magnets"),
        ("Invalidation level:</strong> 71,000. .", "Invalidation level:</strong> 71,000."),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    regex_replacements = [
        (r"Bu gorusum ([\d,]+) ustunde kalici acceptance and belirgin call eklenmesiyle daha pozitif; ([\d,]+) alti kabul ile daha negatif olur\.", r"I would turn more constructive on sustained acceptance above \1 with visible call adding; I would turn more negative on acceptance below \2."),
        (r"Gamma ([\d,]+-[\d,]+) bandinda pozitif\. Spot band icinde kaldigi surece MM hedge akisi hareketi bastirmaya yatkin\.", r"Gamma is positive inside the \1 band. As long as spot remains in the band, MM hedging is inclined to suppress moves."),
        (r"Spot ([\d,]+) ustunde kalir and call akisi desteklerse yukari test aktif\.", r"If spot holds above \1 and call flow confirms, the upside test is active."),
        (r"Smart-money-benzeri yapici akis", "Smart-money-style constructive flow"),
        (r"Kalabalik kovalama skoru ([+\-]?\d+): Smart-money-style constructive flow\.", r"Crowd chase score \1: smart-money-style constructive flow."),
        (r"Magnet strike is is ([\d,]+);", r"Magnet strike is \1;"),
        (r"Tail hedge is is ([\d,]+);", r"Tail hedge is \1;"),
        (r"Magnet strike is degismedi and ([\d,]+) seviyesinde kaldi\.", r"Magnet strike was unchanged at \1."),
        (r"The call cluster did not change and remained at ([\d,]+) seviyesinde kaldi\.", r"The call cluster was unchanged at \1."),
        (r"The call cluster moved closer to spot \(([\d,]+) -> ([\d,]+)\)\);", r"The call cluster moved closer to spot (\1 -> \2);"),
        (r"Net okuma Moderate compression / controlled pricing\. Spot pin merkezine yakinlik skoru %([\d.]+), gamma support is ([\d.]+)\.", r"Net read: moderate compression / controlled pricing. Spot-to-pin proximity score is \1%, gamma support is \2."),
        (r"Spot ([\d,]+) uzerinde kaldigi surece bant cok sert bir asagi kirilim icin hazir gorunmuyor; piyasanin ustunde and etrafinda hala fazla miktarda pozitif gamma birikimi var\.", r"As long as spot holds above \1, the band does not look primed for a violent downside break; there is still a large amount of positive gamma above and around the market."),
        (r"On the downside, 60,000 and 70,000 are regime-shift levels\.", r"On the downside, 60,000 and 70,000 remain regime-shift levels."),
        (r"On the LONG_PUT side, \$([0-9,]+) and ([0-9,]+) of OI was added; this points to fresh downside hedging\.", r"On the LONG_PUT side, $\1 of notional and \2 of OI were added; this points to fresh downside hedging."),
        (r"Teyit sarti: spotun once ([\d,]+), ideal olarak ([\d,]+) ustunde acceptance gormesi ve call akisinin dagilmamasi\.", r"Confirmation needed: spot first needs acceptance above \1, ideally above \2, with call flow holding together."),
        (r"Negatif tetik: ([\d,]+) alti kalicilik hedge cozulmesi yerine yeniden savunma ve downside baskisi uretebilir\.", r"Negative trigger: sustained trade below \1 can bring back defense and renewed downside pressure instead of hedge unwind."),
        (r"Tersine donus teyidi icin spotun en az ([\d,]+) ustune geri yerlesmesi ve flow tonunun savunmadan karisiga donmesi gerekir\.", r"For reversal confirmation, spot needs to reclaim at least \1 and flow tone needs to shift from defensive toward mixed."),
        (r"Tersine donus teyidi icin spotun en az ([\d,]+) ustune geri yerlesmesi and flow tonunun savunmadan karisiga donmesi gerekir\.", r"For reversal confirmation, spot needs to reclaim at least \1 and flow tone needs to shift from defensive toward mixed."),
        (r"Negatif tetik: ([\d,]+) alti yeni kabul, expiry sonrasi bile hedge baskisini hizlandirabilir\.", r"Negative trigger: fresh acceptance below \1 can accelerate hedge pressure even after expiry."),
        (r"Negative tetik: ([\d,]+) alti yeni kabul, expiry sonrasi bile hedge baskisini hizlandirabilir\.", r"Negative trigger: fresh acceptance below \1 can accelerate hedge pressure even after expiry."),
        (r"Teyit sarti: ([\d,]+) ustunde kalicilik ve ([\d,]+) civarinda reject yerine acceptance\.", r"Confirmation needed: sustained trade above \1 and acceptance near \2 rather than rejection."),
        (r"Negatif tetik: spot yeniden ([\d,]+) altina inerse bu okuma relief'ten oteye tasinmaz\.", r"If spot falls back below \1, the read should not be upgraded beyond a relief move."),
        (r"Teyit sarti: spotun ([\d,]+) ustunde kalmasi ve yukari testlerde yeni savunmaci akisin geri gelmemesi\.", r"Confirmation needed: spot needs to hold above \1 and fresh defensive flow must not return on upside tests."),
        (r"Negatif tetik: ([\d,]+) alti kabul, expiry rahatlamasi beklemek yerine savunmaci tonun dogru okundugunu teyit eder\.", r"Negative trigger: acceptance below \1 would confirm the defensive read rather than an expiry relief thesis."),
        (r"SHORT_CALL OI increased by ([0-9,]+); this suggests upside is being capped through call selling and that clean breakout conviction remains limited\.", r"SHORT_CALL OI increased by \1; this suggests upside is being capped through call selling and that clean breakout conviction remains limited."),
        (r"LONG_CALL OI increased by ([0-9,]+); part of the tape is rebuilding upside risk, but that does not amount to standalone euphoria\.", r"LONG_CALL OI increased by \1; part of the tape is rebuilding upside risk, but that does not amount to standalone euphoria."),
        (r"SHORT_PUT OI increased by ([0-9,]+); some participants are still selling downside and collecting carry\.", r"SHORT_PUT OI increased by \1; some participants are still selling downside and collecting carry."),
        (r"The put hedge zone is ([\d,]+-[\d,]+); read this as the main downside protection zone\. The strongest put pocket sits around ([\d,]+)\.", r"The put hedge zone is \1; read this as the main downside protection zone. The strongest put pocket sits around \2."),
        (r"The call positioning band is ([\d,]+-[\d,]+); acceptance near the upper edge of the zone opens room for upside extension\. The strongest call concentration sits near ([\d,]+) cevresinde\.", r"The call positioning band is \1; acceptance near the upper edge of the zone opens room for upside extension. The strongest call concentration sits near \2."),
        (r"Spot su an magnet strike'in altinda \(([\d,]+)\); bu nedenle ilk rejim degisimi sinyali ([\d,]+) ustunde acceptance olur\.", r"Spot is currently below the magnet strike (\1); the first regime-change signal would be acceptance above \2."),
        (r"Magnet strike is ([\d,]+) geri alinir and ustunde acceptance gelirse, piyasa savunmaci rejimden cikarak call positioning bandina dogru uzama sansi kazanir\.", r"If magnet strike \1 is reclaimed and accepted above, the market can move out of its defensive regime and extend toward the call-positioning band."),
        (r"If price accepts above put hedge zone ([\d,]+-[\d,]+) bandina girerse asil soru dokunmak degil, bu bolgede kabul gorup gormedigidir; kabul olursa hedge baskisi buyur, reject olursa tepki olasiligi artar\.", r"If price trades into the put-hedge zone \1, the key question is not the touch but whether it is accepted there; acceptance increases hedge pressure, while rejection increases rebound odds."),
        (r"The call positioning band is ([\d,]+-[\d,]+) icine gecis tek basina trend teyidi degildir; bu alan hedef kadar yapiskan / direncli fiyatlama da uretebilir\. Temiz pozitif senaryo icin zone icinde kalicilik gerekir\.", r"A move into the call-positioning band \1 is not trend confirmation on its own; this zone can act as sticky / resistant pricing as much as a target. A clean bullish read still requires sustained acceptance inside the zone."),
        (r"Tail hedge is ([\d,]+) gunluk trade seviyesi degil; daha cok normal hedge alaninin da kirildigi, kuyruk riskin gercekten fiyatlanmaya basladigi uzak stres referansidir\.", r"Tail hedge at \1 is not a day-trading level; it is a distant stress reference where even the normal hedge zone would already be broken and tail risk would be pricing in for real."),
        (r"En yakin vade ([A-Z0-9]+); toplam notional (\$[\d,]+) ve toplam OI yaklasik ([\d,]+) BTC\.", r"The nearest expiry is \1; total notional is \2 and total OI is about \3 BTC."),
        (r"Call OI ([\d,]+) BTC, put OI ([\d,]+) BTC; put OI payi %([\d.]+)\.", r"Call OI is \1 BTC, put OI is \2 BTC; put OI share is \3%."),
        (r"Call notional (\$[\d,]+), put notional (\$[\d,]+); put notional payi %([\d.]+)\.", r"Call notional is \1, put notional is \2; put notional share is \3%."),
        (r"Call premium (\$[\d,]+), put premium (\$[\d,]+); put premium payi %([\d.]+)\.", r"Call premium is \1, put premium is \2; put premium share is \3%."),
        (r"En yakin vade ([A-Z0-9]+); toplam notional (\$[\d,]+) and toplam OI yaklasik ([\d,]+) BTC\.", r"The nearest expiry is \1; total notional is \2 and total OI is about \3 BTC."),
        (r"Bu chart gercek MM inventory degil; gamma, flow, skew, IV-RV and strategy tape uzerinden uretilen bir proxy skorudur\.", r"This chart is not true MM inventory; it is a proxy score built from gamma, flow, skew, IV-RV, and the strategy tape."),
        (r"Bu grafik son 24 saatte hangi strike'larda yeni risk eklendigini and hangi strike'larda riskin bosaldigini en hizli sekilde gosterir\.", r"This chart shows as quickly as possible where new risk was added and where risk was reduced across strikes over the last 24 hours."),
        (r"Gunluk Pain Kaymasi", "Daily Pain Shift"),
        (r"5 Gunluk Pain Kaymasi", "5-Day Pain Shift"),
        (r"watch ([+\-]?\d+\.\d+) seviyesinden geriye bak\.", r"watch from the \1 level."),
        (r"ATM spot IV is is changed ([+\-]?\d+\.\d+) puan; front ATM IV degisimi ([+\-]?\d+\.\d+); net gamma degisimi ([+\-]?\d+\.\d+)\.", r"ATM spot IV changed by \1 points; front ATM IV changed by \2; net gamma changed by \3."),
        (r"IV-RV spreadi dunden bugune ([+\-]?\d+\.\d+) puan vs yesterday\.", r"IV-RV spread changed by \1 points versus yesterday."),
        (r"5 gunluk referans:", "5-day reference:"),
        (r"ertesi gun OI", "next-day OI"),
        (r"Today's aggregate max-pain level is ([\d,]+)\.", r"Today's aggregate max-pain level is \1."),
        (r"The gap between spot and max pain is ([+\-]?\d[\d,]*) puan\.", r"The gap between spot and max pain is \1 points."),
        (r"The daily pain shift is ([+\-]?\d[\d,]*) puan; and the change in the spot-pain gap is ([+\-]?\d[\d,]*) puan\.", r"The daily pain shift is \1 points; the spot-pain gap changed by \2 points."),
        (r"The 5-day pain shift is ([+\-]?\d[\d,]*) puan\.", r"The 5-day pain shift is \1 points."),
        (r"On the LONG_PUT side, \$([0-9,]+) of of notional and ([0-9,]+) of OI were added;", r"On the LONG_PUT side, $\1 of notional and \2 of OI were added;"),
        (r"Front ATM IV is percentile (\d+); bugunku implied vol yerel serideki gunlerin yaklasik %(\d+)'inden daha pahali\.", r"Front ATM IV percentile is \1; today's implied vol is richer than roughly \2% of the local sample."),
        (r"Front ATM IV is rank (\d+); ham min-max araligina gore vol rejimi ust bolgede\.", r"Front ATM IV rank is \1; on a raw min-max basis, the vol regime sits in the upper zone."),
        (r"Outlier-adjusted rank (\d+); tekil spike gunlerinin skala etkisini azaltmak icin winsorized rank kullaniyoruz\.", r"Outlier-adjusted rank is \1; a winsorized rank is used to reduce the scaling impact of single spike days."),
    ]
    for pattern, repl in regex_replacements:
        text = re.sub(pattern, repl, text)
    return text


def tr_to_en_list(items: List[str]) -> List[str]:
    return [tr_to_en_text(item) for item in items]


def render_table_rows(rows: List[Dict], kind: str) -> str:
    if kind == "expiry":
        rows = sorted(rows, key=lambda row: row.get("open_interest", 0.0), reverse=True)
        return "".join(
            f"<tr><td>{row['maturity']}</td><td>{fmt_usd(row['open_interest'])}</td>"
            f"<td>{row.get('call_btc_oi', 0.0):,.0f}</td><td>{row.get('put_btc_oi', 0.0):,.0f}</td>"
            f"<td>{fmt_usd(row.get('call_notional', 0.0))}</td><td>{fmt_usd(row.get('put_notional', 0.0))}</td>"
            f"<td>{fmt_usd(row['open_interest_change_usd'])}</td></tr>"
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
    if kind == "chain":
        return "".join(
            f"<tr><td>{row['strike']:,.0f}</td><td>{fmt_usd(row['call_notional'])}</td><td>{fmt_usd(row['put_notional'])}</td>"
            f"<td>{row['net_notional_mn']:+.1f}mn</td><td>{row['net_flow_mn']:+.1f}mn</td><td>{row['net_gex']:+.2f}</td></tr>"
            for row in rows[:14]
        )
    return ""


def render_html(
    currency: str,
    summary: Dict[str, object],
    change_section: Dict[str, object],
    dealer_pressure_section: Dict[str, object],
    proxy_hedge_impact_section: Dict[str, object],
    clean_position_map_section: Dict[str, object],
    position_map_drift_section: Dict[str, object],
    gamma_support_monitor_section: Dict[str, object],
    gex_term_structure_section: Dict[str, object],
    layered_trade_check: Dict[str, object],
    vanna_charm_section: Dict[str, object],
    iv_regime_section: Dict[str, object],
    flow_horizon_notes: Dict[str, List[str]],
    expiry_rows: List[Dict],
    expiry_chart: Path,
    gex_chart: Path,
    strike_chart: Path,
    archive_chart: Path,
    flow_chart: Path,
    heatmap_chart: Path,
    clean_position_map_chart: Path,
    position_map_drift_chart: Path,
    gamma_support_monitor_chart: Path,
    gex_term_structure_chart: Path,
    gex_term_structure_delta_chart: Path,
    iv_term_chart: Path,
    iv_rv_chart: Path,
    strategy_chart: Path,
    inventory_chart: Path,
    mm_position_map_chart: Path,
    dealer_pressure_delta_chart: Path,
    proxy_hedge_impact_chart: Path,
    proxy_hedge_impact_trend_chart: Path,
    inventory_trend_chart: Path,
    behavioral_trend_chart: Path,
    behavioral_components_chart: Path,
    layered_trade_check_chart: Path,
    vanna_charm_chart: Path,
    strategy_action_chart: Path,
    roll_transfer_chart: Path,
    normalized_skew_chart: Path,
    iv_regime_chart: Path,
    pain_chart: Path,
    pain_section: Dict[str, object],
    api_calls: int,
    public_calls: int,
) -> str:
    generated = UTC_NOW().isoformat()
    asset_version = UTC_NOW().strftime("%Y%m%d%H%M%S")
    type_rows = summary["type_rows"]
    vol_context = summary["vol_context"]
    weekly_block = ""
    if change_section.get("weekly"):
        weekly = change_section["weekly"]
        weekly_block = (
            f"<p><strong>5 gunluk referans:</strong> Spot {weekly['spot']:+,.0f}, "
            f"ATM IV {weekly['iv']:+.2f}, ertesi gun OI {weekly['next_day_oi'] / 1_000_000:+.1f} milyon USD.</p>"
        )
    gex_delta_block = ""
    if gex_term_structure_section.get("delta_rows"):
        gex_delta_block = (
            f"<h3>Dunden Bugune Vade Bazli GEX Degisimi</h3>"
            f"<img src='laevitas/{gex_term_structure_delta_chart.name}?v={asset_version}' alt='Day-over-day GEX delta by expiry'>"
            f"<p class='small-muted'>{gex_term_structure_section['comparison_label']}</p>"
            f"<div class='delta-row'>{render_delta_badges(gex_term_structure_section['delta_badges'])}</div>"
            f"<p><strong>Delta okuma:</strong> {gex_term_structure_section['delta_summary']}</p>"
        )
    atm_iv_ts_note = summary.get("fallback_notes", {}).get("atm_iv_ts", "")
    top_instrument_note = summary.get("fallback_notes", {}).get("top_instrument", "")
    strike_change_note = summary.get("fallback_notes", {}).get("strike_change", "")
    atm_iv_ts_note_html = f"<p><small>{atm_iv_ts_note}</small></p>" if atm_iv_ts_note else ""
    top_instrument_note_html = f"<p><small>{top_instrument_note}</small></p>" if top_instrument_note else ""
    strike_change_note_html = f"<p><small>{strike_change_note}</small></p>" if strike_change_note else ""
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
.small-muted {{ color:var(--muted); font-size:12px; letter-spacing:0.03em; text-transform:uppercase; }}
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
    <p><strong>Bugunun rejimi:</strong> {summary['desk_regime']}</p>
    <p><strong>Risk asimetrisi:</strong> {summary['desk_asymmetry']}</p>
    <p><strong>Yukari teyit:</strong> {summary['desk_upside_confirmation']}</p>
    <p><strong>Asagi tetik:</strong> {summary['desk_downside_trigger']}</p>
    <p><strong>Tek cumlelik masa notu:</strong> {summary['desk_one_liner']}</p>
    <p><span class='badge'>Vade kaynagi: {summary['expiry_market']}</span><span class='badge'>GEX kaynagi: {summary['gex_market']}</span><span class='badge'>Laevitas API: {api_calls}</span><span class='badge'>Public veri: {public_calls}</span></p>
    <small>Uretim zamani {generated}</small>
  </section>
  <section class='summary-strip'>
    <h2>Core Report</h2>
    <p>Bu ust bolum bugunun rejimini, ana seviyeleri ve operasyonel karari hizli okumak icin tutulur. Daha detayli tablolar raporun appendix kismina alinir.</p>
  </section>
  <section class='playbook-strip'>
    <h2>Gunluk Oyun Plani</h2>
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
      <h2>Trigger Map</h2>
      <p><strong>Squeeze trigger:</strong> {summary['takeaways_structured']['squeeze_trigger']:,.0f}</p>
      <p><strong>Breakout confirm:</strong> {summary['takeaways_structured']['breakout_confirm']:,.0f}</p>
      <p><strong>Breakdown trigger:</strong> {summary['takeaways_structured']['breakdown_trigger']:,.0f}</p>
      <p><strong>Acceleration trigger:</strong> {summary['takeaways_structured']['acceleration_trigger']:,.0f}</p>
      <p><strong>Reject above:</strong> {summary['takeaways_structured']['reject_upside']:,.0f}</p>
      <p><strong>Reject below:</strong> {summary['takeaways_structured']['reject_downside']:,.0f}</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Davranissal Akis Avantaji</h2>
      <p><strong>Kalabalik sinyali:</strong> {summary['behavioral_label']}</p>
      <p><strong>Fade mi Follow mu?:</strong> {summary['fade_or_follow']}</p>
      <p><strong>Guven duzeyi:</strong> {summary['behavioral_confidence']}</p>
      <ul>{render_list(summary['behavioral_bullets'])}</ul>
      <p><strong>Bu skor neden boyle cikti:</strong></p>
      <ul>{render_list(summary['behavioral_why'])}</ul>
    </div>
    <div>
      <div class='grid'>
        <div class='metric'><div class='label'>Kalabalik Kovalama Skoru</div><div class='value'>{summary['crowd_chase_score']:+d}</div></div>
        <div class='metric'><div class='label'>Akis Skoru</div><div class='value'>{summary['flow_score']:+d}</div></div>
        <div class='metric'><div class='label'>Skew Gosterge</div><div class='value'>{summary['front_rr_25d']:.2f}</div></div>
        <div class='metric'><div class='label'>IV-RV</div><div class='value'>{summary['iv_rv_7d_spread']:+.2f}</div></div>
      </div>
      <img src='laevitas/{behavioral_components_chart.name}?v={asset_version}' alt='Davranissal bilesenler'>
      <p>Bu grafik skorun hangi bilesenlerden geldigini gosterir; yani sinyalin sadece tek bir etiketten ibaret olmadigini aciklar.</p>
      <img src='laevitas/{behavioral_trend_chart.name}?v={asset_version}' alt='Davranissal akis skoru trendi'>
      <p>Bu grafik kalabaligin gec put panigi mi yoksa gec call kovalamasi mi yaptigini zaman icinde daha net okumayi saglar.</p>
    </div>
  </section>
  <section>
    <h2>Sabah Masa Notu</h2>
    <div class='delta-row'>{render_delta_badges(summary['desk_badges'])}</div>
    <p>{summary['desk_one_liner']}</p>
  </section>
  <section class='two'>
    <div>
      <h2>Katmanli Teyit</h2>
      <p><span class='delta-badge {layered_trade_check["verdict_class"]}'>{layered_trade_check['verdict']}</span></p>
      <div class='grid'>
        <div class='metric'><div class='label'>Bias</div><div class='value'>{layered_trade_check['bias_label']}</div></div>
        <div class='metric'><div class='label'>Trade Gate</div><div class='value'>{layered_trade_check['trade_gate']}</div></div>
        <div class='metric'><div class='label'>Long Skor</div><div class='value'>{layered_trade_check['long_score']}</div></div>
        <div class='metric'><div class='label'>Short Skor</div><div class='value'>{layered_trade_check['short_score']}</div></div>
        <div class='metric'><div class='label'>RSI(14)</div><div class='value'>{layered_trade_check['rsi14']:.1f}</div></div>
        <div class='metric'><div class='label'>20g Z-Score</div><div class='value'>{layered_trade_check['z20']:+.2f}</div></div>
      </div>
      <p><strong>Beklenen aksiyon:</strong> {layered_trade_check['expected_action']}</p>
      <table>
        <tr><th>Katman</th><th>Long</th><th>Short</th><th>Okuma</th></tr>
        {"".join(f"<tr><td>{row['label']}</td><td>{row['long']}</td><td>{row['short']}</td><td>{row['note']}</td></tr>" for row in layered_trade_check['rows'])}
      </table>
    </div>
    <div>
      <img src='laevitas/{layered_trade_check_chart.name}?v={asset_version}' alt='Katmanli teyit matrisi'>
      <h2>Bu Setup Neyi Arar?</h2>
      <ul>{render_list(layered_trade_check['notes'])}</ul>
      <p><strong>Teknik referans:</strong> Bollinger alt bant {layered_trade_check['bb_lower']:,.0f}, ust bant {layered_trade_check['bb_upper']:,.0f}.</p>
      <p>Bu blok tek bir sinyal degil; teknik ekstrem + vol tepkisi + skew davranisi + positioning filtresinin ust uste binip binmedigini kontrol eder.</p>
    </div>
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
      <div class='metric'><div class='label'>Magnet Strike</div><div class='value'>{clean_position_map_section['levels']['magnet_strike']:,.0f}</div></div>
      <div class='metric'><div class='label'>Put Hedge Zone</div><div class='value'>{clean_position_map_section['levels']['put_hedge_low']:,.0f}-{clean_position_map_section['levels']['put_hedge_high']:,.0f}</div></div>
      <div class='metric'><div class='label'>Call Cluster</div><div class='value'>{clean_position_map_section['levels']['call_cluster_high']:,.0f}</div></div>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>BTC Options Position Map (Clean)</h2>
      <img src='laevitas/{clean_position_map_chart.name}?v={asset_version}' alt='BTC Options Position Map Clean'>
    </div>
    <div>
      <h2>Position Map Okumasi</h2>
      <ul>{render_list(summary['clean_position_map_notes'])}</ul>
      <p><strong>Seviyeler:</strong> Tail Hedge {int(summary['clean_position_map_levels']['tail_hedge']):,} | Put Hedge {int(summary['clean_position_map_levels']['put_hedge_low']):,}-{int(summary['clean_position_map_levels']['put_hedge_high']):,} | Magnet {int(summary['clean_position_map_levels']['magnet_strike']):,} | Call Cluster {int(summary['clean_position_map_levels']['call_cluster_high']):,} | Speculation {int(summary['clean_position_map_levels']['speculation']):,}</p>
      <ul>{render_list(clean_position_map_section['notes'])}</ul>
      <p><strong>Bugun ne degisti:</strong> {position_map_drift_section['headline']}</p>
      <ul>{render_list(position_map_drift_section['notes'])}</ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Gamma Destek Monitörü</h2>
      <p><span class='delta-badge {gamma_support_monitor_section["support_class"]}'>{gamma_support_monitor_section['support_label']}</span></p>
      <div class='grid'>
        <div class='metric'><div class='label'>Bugunku Skor</div><div class='value'>{gamma_support_monitor_section['current_score']:.0f}</div></div>
        <div class='metric'><div class='label'>Gunluk Degisim</div><div class='value'>{gamma_support_monitor_section['score_delta_day']:+.0f}</div></div>
        <div class='metric'><div class='label'>14g OI Payi</div><div class='value'>{gamma_support_monitor_section['next_14d_share']:.1f}%</div></div>
        <div class='metric'><div class='label'>30g OI Payi</div><div class='value'>{gamma_support_monitor_section['next_30d_share']:.1f}%</div></div>
      </div>
      <p><strong>Headline:</strong> {gamma_support_monitor_section['headline']}</p>
      <p><strong>Forward Verdict:</strong> {gamma_support_monitor_section['forward_verdict']}</p>
      <ul>{render_list(gamma_support_monitor_section['notes'])}</ul>
    </div>
    <div>
      <img src='laevitas/{gamma_support_monitor_chart.name}?v={asset_version}' alt='Gamma support monitor'>
      <p>Ust panel pasif destek skorunun son gunlerde guclenip zayifladigini, alt panel ise onumuzdeki aylarda destegin hangi vade pencerelerinde yogunlastigini gosterir.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>GEX Vade Yapisi</h2>
      <img src='laevitas/{gex_term_structure_chart.name}?v={asset_version}' alt='GEX term structure by expiry'>
      <p>Bu grafik gercek expiry-bazli GEX dagilimini gosterir; pasif destegin hangi vadelerde tasindigini ve negatif gamma riskinin nerede biriktigini burada goruruz.</p>
      <p><strong>Tek cumlelik okuma:</strong> {gex_term_structure_section['summary_sentence']}</p>
      {gex_delta_block}
    </div>
    <div>
      <h2>Term Structure Okumasi</h2>
      <div class='grid'>
        <div class='metric'><div class='label'>On Vade</div><div class='value'>{gex_term_structure_section['front_maturity']}</div></div>
        <div class='metric'><div class='label'>On Vade GEX</div><div class='value'>{gex_term_structure_section['front_gex']:,.0f}</div></div>
        <div class='metric'><div class='label'>Toplam GEX</div><div class='value'>{gex_term_structure_section['total_gex']:,.0f}</div></div>
        <div class='metric'><div class='label'>Spot</div><div class='value'>{gex_term_structure_section['spot']:,.0f}</div></div>
      </div>
      <p><strong>Headline:</strong> {gex_term_structure_section['headline']}</p>
      <p><strong>Front gamma roll-off risk:</strong> {gex_term_structure_section['front_rolloff_risk']}</p>
      <p><strong>Next support expiry:</strong> {gex_term_structure_section['next_support_expiry']}</p>
      <p><strong>Gamma vacuum window:</strong> {gex_term_structure_section['gamma_vacuum_window']}</p>
      <ul>{render_list(gex_term_structure_section['notes'])}</ul>
      <ul>{render_list(gex_term_structure_section['delta_notes'])}</ul>
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
      <img src='laevitas/{archive_chart.name}?v={asset_version}' alt='Gunluk trend'>
      <p>Bu grafik yerel arsivden beslenir. Ilk gunlerde kisitli olur; her sabah veri geldikce yorum kalitesi artar.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Vol Market Intelligence</h2>
      <ul>{render_list(summary['vol_summary'])}</ul>
      <p><strong>Front maturity:</strong> {summary['ts_front_maturity']}</p>
      <p><strong>Gunluk ATM IV degisimi:</strong> {summary['ts_day_change']:+.2f}</p>
      <p><strong>Haftalik ATM IV degisimi:</strong> {summary['ts_week_change']:+.2f}</p>
      {atm_iv_ts_note_html}
    </div>
    <div>
      <h2>IV ve RV Karsilastirmasi</h2>
      <img src='laevitas/{iv_rv_chart.name}?v={asset_version}' alt='IV RV chart'>
      <p>Laevitas historical RV endpoint'i bu key ile kapali oldugu icin realized vol public BTC gunluk kapanis serisinden hesaplanir.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Akis Yorumu</h2>
      <p><strong>Flow skoru:</strong> {summary['flow_label']} ({summary['flow_score']:+d})</p>
      <ul>{render_list(summary['flow_summary'])}</ul>
      {top_instrument_note_html}
    </div>
    <div>
      <h2>Akis Grafigi</h2>
      <img src='laevitas/{flow_chart.name}?v={asset_version}' alt='Akis degisimi grafigi'>
      <p>Bu grafik son 24 saatte hangi strike'larda yeni risk eklendigini ve hangi strike'larda riskin bosaldigini en hizli sekilde gosterir.</p>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Flow Heatmap</h2>
      <img src='laevitas/{heatmap_chart.name}?v={asset_version}' alt='Flow heatmap'>
      <p>Heatmap spot cevresinde call ve put akisinin tam olarak hangi strike'larda yogunlastigini gosterir.</p>
      {strike_change_note_html}
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
      <img src='laevitas/{iv_term_chart.name}?v={asset_version}' alt='ATM IV term structure'>
      <table><tr><th>Vade</th><th>ATM IV</th><th>25d RR</th><th>25d BF</th></tr>{render_table_rows(summary['iv_term_rows'], 'iv_term')}</table>
    </div>
    <div>
      <h2>Strategy Tape</h2>
      <img src='laevitas/{strategy_chart.name}?v={asset_version}' alt='Strategy chart'>
      <p><strong>Baskin ton:</strong> {summary['strategy_bias']}</p>
      <ul>{render_list(summary['strategy_summary'])}</ul>
      <table><tr><th>Strategy</th><th>Notional</th><th>OI Degisimi</th><th>Delta</th></tr>{render_table_rows(summary['strategy_rows'], 'strategy')}</table>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Strateji Akis Yorumu</h2>
      <p><span class='delta-badge {summary["strategy_verdict_class"]}'>{summary['strategy_verdict']}</span></p>
      <p><strong>Strateji Karari:</strong> {summary['strategy_verdict']}</p>
      <p><strong>Baskin Aksiyon:</strong> {summary['strategy_dominant_action']}</p>
      <p><strong>Trade Yansimasi:</strong> {summary['strategy_trading_implication']}</p>
      <p><strong>Headline:</strong> {summary['strategy_flow_headline']}</p>
      <ul>{render_list(summary['strategy_flow_interpretation'])}</ul>
    </div>
    <div>
      <h2>Strateji Aksiyon Haritasi</h2>
      <img src='laevitas/{strategy_action_chart.name}?v={asset_version}' alt='Strategy action map'>
      <p>Bu grafik strateji isimlerini degil, tape'in hangi aksiyonlarda yogunlastigini gosterir. Soldaki tabloyu hizli okumak icin kullan.</p>
      <img src='laevitas/{roll_transfer_chart.name}?v={asset_version}' alt='Roll transfer map'>
      <p><strong>Tahmini Roll / Transfer:</strong> {summary['roll_transfer_headline']}</p>
      <ul>{render_list(summary['roll_transfer_notes'])}</ul>
      <h3>Bu Ne Anlama Geliyor?</h3>
      <ul>
        <li>Roll / extend: koruma kapatilmiyor, daha ileri vadeye tasiniyor olabilir.</li>
        <li>Fresh hedge: yeni downside risk aliniyor ya da mevcut long risk korunuyor olabilir.</li>
        <li>Cap / overwrite: yukari hareket satilarak sinirlanmak isteniyor olabilir.</li>
        <li>Unwind: onceki cap ya da hedge yapilari gevsetiliyor olabilir.</li>
        <li>Defined-risk bullish: oyuncular yukariyi kovalamak yerine spread gibi sinirli riskli yapilari tercih ediyor olabilir.</li>
      </ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Dealer Pressure Map (Proxy)</h2>
      <img src='laevitas/{mm_position_map_chart.name}?v={asset_version}' alt='MM Position Map v1'>
      <p>Bu chart mevcut GEX, strike OI ve 24s strike akisini birlestiren bir proxy haritadir. Gercek MM inventory degildir; hangi strike'larda yukari/asagi baski yogunlasiyor onu gosterir.</p>
      <img src='laevitas/{dealer_pressure_delta_chart.name}?v={asset_version}' alt='Dealer pressure delta'>
      <p class='small-muted'>{dealer_pressure_section['comparison_label']}</p>
      <div class='delta-row'>{render_delta_badges(dealer_pressure_section['badges'])}</div>
      <p><strong>Gunluk delta okuma:</strong> {dealer_pressure_section['summary']}</p>
      <ul>{render_list(summary['mm_map_read'])}</ul>
      <ul>{render_list(dealer_pressure_section['notes'])}</ul>
    </div>
    <div>
      <h2>Chain Table v1</h2>
      <table><tr><th>Strike</th><th>Call Notional</th><th>Put Notional</th><th>Net OI</th><th>24s Net Flow</th><th>Net GEX</th></tr>{render_table_rows(summary['chain_focus'], 'chain')}</table>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Proxy Hedge-Impact Yuzeyi</h2>
      <img src='laevitas/{proxy_hedge_impact_chart.name}?v={asset_version}' alt='Proxy hedge impact surface'>
      <p>Bu chart true dealer inventory kullanmaz; strike OI, proxy pressure ve GEX ile spot farkli seviyelere giderse hedge etkisinin daha dengeleyici mi daha kirilgan mi calisabilecegini idealize eder.</p>
      <div class='delta-row'>{render_delta_badges(proxy_hedge_impact_section['badges'])}</div>
      <p><strong>Headline:</strong> {proxy_hedge_impact_section['headline']}</p>
      <p><strong>Tek satir:</strong> {proxy_hedge_impact_section['summary']}</p>
      <img src='laevitas/{proxy_hedge_impact_trend_chart.name}?v={asset_version}' alt='Proxy hedge impact trend'>
      <p class='small-muted'>{proxy_hedge_impact_section.get('history_label', '')}</p>
      <div class='delta-row'>{render_delta_badges(proxy_hedge_impact_section.get('history_badges', []))}</div>
      <p><strong>Dunden bugune:</strong> {proxy_hedge_impact_section.get('history_summary', '')}</p>
    </div>
    <div>
      <h2>Nasil Okunur?</h2>
      <p><span class='delta-badge {proxy_hedge_impact_section["regime_class"]}'>{proxy_hedge_impact_section['regime']}</span></p>
      <div class='grid'>
        <div class='metric'><div class='label'>Spot</div><div class='value'>{proxy_hedge_impact_section['spot']:,.0f}</div></div>
        <div class='metric'><div class='label'>Mevcut Skor</div><div class='value'>{proxy_hedge_impact_section['current_score']:+.0f}</div></div>
        <div class='metric'><div class='label'>Stabilizing Pocket</div><div class='value'>{proxy_hedge_impact_section['support_spot']:,.0f}</div></div>
        <div class='metric'><div class='label'>Fragile Pocket</div><div class='value'>{proxy_hedge_impact_section['fragile_spot']:,.0f}</div></div>
      </div>
      <ul>{render_list(proxy_hedge_impact_section['notes'])}</ul>
      <ul>{render_list(proxy_hedge_impact_section.get('history_notes', []))}</ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Gamma Haritasi</h2>
      <img src='laevitas/{gex_chart.name}?v={asset_version}' alt='GEX chart'>
    </div>
    <div>
      <h2>Kritik Gamma Dugumleri</h2>
      <h3>Pozitif</h3>
      <table><tr><th>Strike</th><th>Net</th><th>Call</th><th>Put</th></tr>{render_table_rows(summary['positive_nodes'], 'gex')}</table>
      <h3>Negatif</h3>
      <table><tr><th>Strike</th><th>Net</th><th>Call</th><th>Put</th></tr>{render_table_rows(summary['negative_nodes'], 'gex')}</table>
      <div class='metric' style='margin-top:14px;'>
        <div class='label'>Gamma Rejim Okumasi</div>
        <div class='delta-row'><span class='delta-badge {summary["gamma_map_read"]["regime_class"]}'>{summary["gamma_map_read"]["regime_badge"]}</span></div>
        <p><strong>Spotun Yeri:</strong> {summary['gamma_map_read']['current_zone']}</p>
        <p><strong>Bu Bolgede Davranis:</strong> {summary['gamma_map_read']['current_behavior']}</p>
        <p><strong>Pozitifte Davranis:</strong> {summary['gamma_map_read']['positive_behavior']}</p>
        <p><strong>Gecis Bolgesi:</strong> {summary['gamma_map_read']['transition_zone']}</p>
        <p><strong>Negatifte Davranis:</strong> {summary['gamma_map_read']['negative_behavior']}</p>
        <p><strong>Short Nerde Aranir?:</strong> {summary['gamma_map_read']['execution_short']}</p>
        <p><strong>Long Nerde Aranir?:</strong> {summary['gamma_map_read']['execution_long']}</p>
        <p><strong>No-Trade / Teyitsiz Bolge:</strong> {summary['gamma_map_read']['execution_no_trade']}</p>
        <p><strong>Base Case:</strong> {summary['gamma_map_read']['base_case']}</p>
      </div>
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
  <section class='summary-strip'>
    <h2>Appendix</h2>
    <p>Bu bolum ana karari tasiyan cekirdek sinyallerden daha dusuk oncelikli teyit grafigi ve detay tablolari toplar.</p>
  </section>
  <section class='two'>
    <div>
      <h2>Dealer Inventory Proxy</h2>
      <img src='laevitas/{inventory_chart.name}?v={asset_version}' alt='Dealer inventory proxy chart'>
      <p><strong>Net ton:</strong> {summary['inventory_label']} ({summary['inventory_net']:+.2f})</p>
      <p>Bu chart gercek MM inventory degil; gamma, flow, skew, IV-RV ve strategy tape uzerinden uretilen bir proxy skorudur.</p>
      <img src='laevitas/{inventory_trend_chart.name}?v={asset_version}' alt='Dealer inventory proxy trend'>
      <p>Yerel arsiv buyudukce bu grafik MM/dealer tonunun zaman icinde daha net izlenmesini saglar.</p>
    </div>
    <div>
      <h2>Trigger Rules</h2>
      <ul>{render_list(summary['acceptance_rules'])}</ul>
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
      <h2>Vanna / Charm Proxy</h2>
      <p><span class='delta-badge delta-neutral'>{vanna_charm_section['regime']}</span></p>
      <p><span class='delta-badge {vanna_charm_section["takeaway_class"]}'>{vanna_charm_section['takeaway_badge']}</span></p>
      <p><strong>Karar:</strong> {vanna_charm_section['takeaway']}</p>
      <div class='grid'>
        <div class='metric'><div class='label'>Charm Proxy</div><div class='value'>{vanna_charm_section['charm_proxy']:.0f}</div></div>
        <div class='metric'><div class='label'>Vanna Proxy</div><div class='value'>{vanna_charm_section['vanna_proxy']:.0f}</div></div>
        <div class='metric'><div class='label'>Compression</div><div class='value'>{vanna_charm_section['compression_proxy']:.0f}</div></div>
        <div class='metric'><div class='label'>Front DTE</div><div class='value'>{vanna_charm_section['front_dte']}</div></div>
      </div>
      <ul>{render_list(vanna_charm_section['notes'])}</ul>
    </div>
    <div>
      <img src='laevitas/{vanna_charm_chart.name}?v={asset_version}' alt='Vanna charm proxy'>
      <p>Bu chart gercek greeks degil; yakin vade expiry yogunlugu, quarterly baski, gamma destegi, pin merkezine yakinlik ve IV primini tek panelde toplar.</p>
      <p><strong>Kontrol listesi:</strong></p>
      <ul>{render_list(vanna_charm_section['checklist'])}</ul>
    </div>
  </section>
  <section>
    <h2>Vade Yapisi</h2>
    <img src='laevitas/{expiry_chart.name}?v={asset_version}' alt='Expiry OI chart'>
    <table><tr><th>Vade</th><th>Acik Pozisyon</th><th>Call OI (BTC)</th><th>Put OI (BTC)</th><th>Call Notional</th><th>Put Notional</th><th>24s OI Degisimi</th></tr>{render_table_rows(expiry_rows, 'expiry')}</table>
  </section>
  <section class='two'>
    <div>
      <h2>En Yakin Vade Detayi</h2>
      <p><strong>On Vade Karari:</strong> {summary['front_expiry_verdict']}</p>
      <div class='grid'>
        <div class='metric'><div class='label'>Vade</div><div class='value'>{summary['front_expiry']['maturity']}</div></div>
        <div class='metric'><div class='label'>Call OI</div><div class='value'>{summary['front_expiry'].get('call_btc_oi', 0.0):,.0f}</div></div>
        <div class='metric'><div class='label'>Put OI</div><div class='value'>{summary['front_expiry'].get('put_btc_oi', 0.0):,.0f}</div></div>
        <div class='metric'><div class='label'>Put Premium</div><div class='value'>{fmt_usd(summary['front_expiry'].get('put_premium', 0.0))}</div></div>
      </div>
      <ul>{render_list(summary['front_expiry_notes'])}</ul>
    </div>
    <div>
      <h2>Expiry Kapanis Okumasi</h2>
      <ul>
        <li><strong>Pozisyonlama:</strong> {summary['front_positioning_verdict']}</li>
        <li><strong>Premium:</strong> {summary['front_premium_verdict']}</li>
        <li><strong>Expiry Sonrasi Etki:</strong> {summary['front_post_expiry_implication']}</li>
        <li><strong>Beklenen Ilk Etki:</strong> {summary['front_expected_first_effect']}</li>
        <li><strong>Teyit Sarti:</strong> {summary['front_confirmation_needed']}</li>
        <li><strong>Negatif Tetik:</strong> {summary['front_negative_trigger']}</li>
      </ul>
      <h2>Yakin Vade Nasil Okunur?</h2>
      <ul>
        <li>Call OI, yukari yonlu acik riskin hangi vadede yigildigini gosterir.</li>
        <li>Put OI, asagi koruma veya spekulatif downside talebinin agirligini gosterir.</li>
        <li>Premium tarafi, hangi taraf icin daha cok para odendigini gosterir; bu genelde daha ilgili tarafi ayirmada daha faydalidir.</li>
        <li>Yarin kapanacak vade buyuk ve put-premium agir ise expiry sonrasi temizlenecek hedge etkisi daha onemli olabilir.</li>
      </ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>IV Percentile ve Rank</h2>
      <p><span class='delta-badge {iv_regime_section["vol_bias_class"]}'>{iv_regime_section['vol_bias']}</span></p>
      <p>{iv_regime_section['vol_bias_note']}</p>
      <ul>{render_list(iv_regime_section['notes'])}</ul>
    </div>
    <div>
      <div class='grid'>
        <div class='metric'><div class='label'>IV Percentile</div><div class='value'>{iv_regime_section['iv_percentile']:.0f}</div></div>
        <div class='metric'><div class='label'>IV Rank</div><div class='value'>{iv_regime_section['iv_rank']:.0f}</div></div>
        <div class='metric'><div class='label'>Adj. IV Rank</div><div class='value'>{iv_regime_section['adj_iv_rank']:.0f}</div></div>
        <div class='metric'><div class='label'>Orneklem</div><div class='value'>{iv_regime_section['sample_size']}</div></div>
      </div>
      <img src='laevitas/{iv_regime_chart.name}?v={asset_version}' alt='IV rejim trendi'>
      <p>Percentile, bugunku implied volun yerel seriye gore ne kadar sik pahali bolgede oldugunu; rank ise min-max araligi icinde ne kadar yukarida oldugunu gosterir.</p>
      <p><strong>Vol checklist:</strong></p>
      <ul>{render_list(iv_regime_section['checklist'])}</ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Normalize Edilmis 25D Skew</h2>
      <img src='laevitas/{normalized_skew_chart.name}?v={asset_version}' alt='Normalize edilmis 25D skew'>
      <p><strong>Mevcut oran:</strong> {summary['normalized_25d_skew_ratio']:.3f}</p>
      <p>Bu metrik, 25 delta put vol ile 25 delta call vol farkinin ATM IV'ye oranidir. Oran yukselirse piyasa putlari call'lara gore daha pahali fiyatliyor demektir.</p>
    </div>
    <div>
      <h2>Skew Okumasi</h2>
      <ul><li>Front 25D RR: {summary['front_rr_25d']:.2f}</li><li>Front ATM IV: {summary['front_atm_iv']:.2f}</li><li>Normalize oran yuksek kaldikca downside korumasi call talebine gore daha pahali kalir.</li></ul>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>Pain Drift</h2>
      <img src='laevitas/{pain_chart.name}?v={asset_version}' alt='Pain drift chart'>
      <ul>{render_list(pain_section['notes'])}</ul>
    </div>
    <div>
      <div class='grid'>
        <div class='metric'><div class='label'>Max Pain</div><div class='value'>{summary['max_pain']:,.0f}</div></div>
        <div class='metric'><div class='label'>Spot - Pain</div><div class='value'>{summary['spot_max_pain_gap']:+,.0f}</div></div>
        <div class='metric'><div class='label'>Gunluk Pain Kaymasi</div><div class='value'>{pain_section['pain_change']:+,.0f}</div></div>
        <div class='metric'><div class='label'>5 Gunluk Pain Kaymasi</div><div class='value'>{pain_section['week_change']:+,.0f}</div></div>
      </div>
    </div>
  </section>
  <section class='two'>
    <div>
      <h2>En Guclu Instrument OI Eklemeleri</h2>
      <table><tr><th>Instrument</th><th>OI Degisimi</th><th>Toplam Notional</th><th>Acik Pozisyon</th></tr>{render_table_rows(summary['top_adds'], 'instrument')}</table>
      {top_instrument_note_html}
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
      {strike_change_note_html}
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
    dealer_pressure_section: Dict[str, object],
    proxy_hedge_impact_section: Dict[str, object],
    clean_position_map_section: Dict[str, object],
    position_map_drift_section: Dict[str, object],
    gamma_support_monitor_section: Dict[str, object],
    gex_term_structure_section: Dict[str, object],
    layered_trade_check: Dict[str, object],
    vanna_charm_section: Dict[str, object],
    iv_regime_section: Dict[str, object],
    flow_horizon_notes: Dict[str, List[str]],
    expiry_rows: List[Dict],
    expiry_chart: Path,
    gex_chart: Path,
    strike_chart: Path,
    archive_chart: Path,
    flow_chart: Path,
    heatmap_chart: Path,
    clean_position_map_chart: Path,
    position_map_drift_chart: Path,
    gamma_support_monitor_chart: Path,
    gex_term_structure_chart: Path,
    gex_term_structure_delta_chart: Path,
    iv_term_chart: Path,
    iv_rv_chart: Path,
    strategy_chart: Path,
    inventory_chart: Path,
    mm_position_map_chart: Path,
    dealer_pressure_delta_chart: Path,
    proxy_hedge_impact_chart: Path,
    proxy_hedge_impact_trend_chart: Path,
    inventory_trend_chart: Path,
    behavioral_trend_chart: Path,
    behavioral_components_chart: Path,
    layered_trade_check_chart: Path,
    vanna_charm_chart: Path,
    strategy_action_chart: Path,
    roll_transfer_chart: Path,
    normalized_skew_chart: Path,
    iv_regime_chart: Path,
    pain_chart: Path,
    pain_section: Dict[str, object],
    api_calls: int,
    public_calls: int,
) -> str:
    summary_en = dict(summary)
    summary_en["fallback_notes"] = {key: tr_to_en_text(value) for key, value in summary.get("fallback_notes", {}).items()}
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
        "strategy_flow_interpretation",
        "roll_transfer_notes",
        "final_read",
        "mm_map_read",
        "clean_position_map_notes",
        "behavioral_bullets",
        "behavioral_why",
    ]:
        summary_en[key] = tr_to_en_list(summary.get(key, []))

    summary_en["base_case"] = tr_to_en_text(summary["base_case"])
    summary_en["alt_case"] = tr_to_en_text(summary["alt_case"])
    summary_en["invalidation"] = tr_to_en_text(summary["invalidation"])
    summary_en["flow_label"] = tr_to_en_text(summary["flow_label"])
    summary_en["strategy_bias"] = tr_to_en_text(summary["strategy_bias"])
    summary_en["strategy_flow_headline"] = tr_to_en_text(summary.get("strategy_flow_headline", ""))
    summary_en["strategy_verdict"] = tr_to_en_text(summary.get("strategy_verdict", ""))
    summary_en["strategy_verdict_class"] = summary.get("strategy_verdict_class", "delta-neutral")
    summary_en["strategy_dominant_action"] = tr_to_en_text(summary.get("strategy_dominant_action", ""))
    summary_en["strategy_trading_implication"] = tr_to_en_text(summary.get("strategy_trading_implication", ""))
    summary_en["roll_transfer_headline"] = tr_to_en_text(summary.get("roll_transfer_headline", ""))
    summary_en["inventory_label"] = tr_to_en_text(summary["inventory_label"])
    summary_en["directional_bias"] = tr_to_en_text(summary["directional_bias"])
    summary_en["directional_note"] = tr_to_en_text(summary["directional_note"])
    summary_en["asymmetry_note"] = tr_to_en_text(summary["asymmetry_note"])
    summary_en["change_mind_note"] = tr_to_en_text(summary["change_mind_note"])
    summary_en["desk_regime"] = tr_to_en_text(summary.get("desk_regime", ""))
    summary_en["desk_asymmetry"] = tr_to_en_text(summary.get("desk_asymmetry", ""))
    summary_en["desk_upside_confirmation"] = tr_to_en_text(summary.get("desk_upside_confirmation", ""))
    summary_en["desk_downside_trigger"] = tr_to_en_text(summary.get("desk_downside_trigger", ""))
    summary_en["desk_one_liner"] = tr_to_en_text(summary.get("desk_one_liner", ""))
    desk_badges_en = []
    for item in summary.get("desk_badges", []):
        text = str(item)
        text = text.replace("Rejim: Kontrollu ama kirilgan", "Regime: Controlled but fragile")
        text = text.replace("Rejim: Kontrollu", "Regime: Controlled")
        text = text.replace("Rejim: Kirilgan", "Regime: Fragile")
        text = text.replace("Gecis noktasi:", "Transition point:")
        text = text.replace("Asimetri: Asagi hizlanma", "Asymmetry: Downside acceleration")
        text = text.replace("Asimetri: Dengeli", "Asymmetry: Balanced")
        text = text.replace("Yukari teyit:", "Upside confirmation:")
        text = text.replace("Asagi tetik:", "Downside trigger:")
        desk_badges_en.append(text)
    summary_en["desk_badges"] = desk_badges_en
    summary_en["behavioral_label"] = tr_to_en_text(summary["behavioral_label"])
    summary_en["fade_or_follow"] = tr_to_en_text(summary["fade_or_follow"])
    summary_en["predictive_read"] = tr_to_en_text(summary["predictive_read"])
    summary_en["behavioral_confidence"] = tr_to_en_text(summary["behavioral_confidence"])
    dealer_pressure_section_en = {
        "delta_rows": dealer_pressure_section.get("delta_rows", []),
        "comparison_label": str(dealer_pressure_section.get("comparison_label", "")),
        "badges": [
            tr_to_en_text(str(item))
            .replace("Yukari baski:", "Upside pressure:")
            .replace("Asagi baski:", "Downside pressure:")
            .replace("Yukari build:", "Upside build:")
            .replace("Asagi build:", "Downside build:")
            for item in dealer_pressure_section.get("badges", [])
        ],
        "summary": tr_to_en_text(str(dealer_pressure_section.get("summary", ""))),
        "notes": tr_to_en_list(dealer_pressure_section.get("notes", [])),
    }
    proxy_hedge_impact_section_en = {
        **proxy_hedge_impact_section,
        "regime": tr_to_en_text(str(proxy_hedge_impact_section.get("regime", "")))
        .replace("Dengeleyici / long-gamma-benzeri", "Stabilizing / long-gamma-like")
        .replace("Kirilgan / short-gamma-benzeri", "Fragile / short-gamma-like")
        .replace("Gecis / karisik", "Transition / mixed"),
        "headline": tr_to_en_text(str(proxy_hedge_impact_section.get("headline", ""))),
        "summary": tr_to_en_text(str(proxy_hedge_impact_section.get("summary", ""))),
        "badges": [
            tr_to_en_text(str(item))
            .replace("Mevcut proxy skor:", "Current proxy score:")
            .replace("Rejim:", "Regime:")
            .replace("Dengeleyici pocket:", "Stabilizing pocket:")
            .replace("Kirilgan pocket:", "Fragile pocket:")
            .replace("Long-gamma alani:", "Long-gamma pocket:")
            .replace("Dengeleyici / long-gamma-benzeri", "Stabilizing / long-gamma-like")
            .replace("Kirilgan / short-gamma-benzeri", "Fragile / short-gamma-like")
            .replace("Gecis / karisik", "Transition / mixed")
            for item in proxy_hedge_impact_section.get("badges", [])
        ],
        "notes": tr_to_en_list(proxy_hedge_impact_section.get("notes", [])),
        "history_label": str(proxy_hedge_impact_section.get("history_label", "")),
        "history_badges": [
            tr_to_en_text(str(item))
            .replace("Skor degisimi:", "Score change:")
            .replace("Dengeleyici pocket kaymasi:", "Stabilizing pocket shift:")
            .replace("Kirilgan pocket kaymasi:", "Fragile pocket shift:")
            for item in proxy_hedge_impact_section.get("history_badges", [])
        ],
        "history_summary": tr_to_en_text(str(proxy_hedge_impact_section.get("history_summary", ""))),
        "history_notes": tr_to_en_list(proxy_hedge_impact_section.get("history_notes", [])),
    }
    gamma_map_read_en = {key: tr_to_en_text(value) for key, value in summary.get("gamma_map_read", {}).items()}
    gamma_map_read_en["current_zone"] = re.sub(
        r"Spot su an pozitif gamma denge bandinin altinda; ([\d,]+) alti artik gecis / zayiflama bolgesi\.",
        r"Spot is currently below the positive-gamma balance band; below \1 is now a transition / weakening zone.",
        gamma_map_read_en.get("current_zone", ""),
    )
    gamma_map_read_en["current_zone"] = re.sub(
        r"Spot su an ([\d,]+)-([\d,]+) pozitif gamma denge bandinin icinde\.",
        r"Spot is currently inside the positive-gamma balance band between \1 and \2.",
        gamma_map_read_en.get("current_zone", ""),
    )
    gamma_map_read_en["current_zone"] = re.sub(
        r"Spot su an pozitif gamma bandinin ust tarafinda; ilk yukari acceptance alani ([\d,]+) ustunde\.",
        r"Spot is currently above the positive-gamma band; the first upside-acceptance zone starts above \1.",
        gamma_map_read_en.get("current_zone", ""),
    )
    gamma_map_read_en["current_behavior"] = re.sub(
        r"Bu bolgede hareket emilmek yerine daha yonlu ve kirilgan olmaya baslar; ([\d,]+) ve ([\d,]+) civarindaki negatif gamma cepleri daha kolay aktive olabilir\.",
        r"In this zone, moves are less likely to be absorbed and more likely to turn directional and fragile; the negative-gamma pockets near \1 and \2 can activate more easily.",
        gamma_map_read_en.get("current_behavior", ""),
    )
    gamma_map_read_en["positive_behavior"] = re.sub(
        r"Pozitif gamma tarafinda \(([\d,]+)-([\d,]+)\) fiyat davranisi daha kontrollu, geri cekilmeye yatkin ve hedge akisiyla bastirilmaya acik\.",
        r"In positive gamma (\1-\2), price action is more controlled, more mean-reverting, and more likely to be dampened by hedge flow.",
        gamma_map_read_en.get("positive_behavior", ""),
    )
    gamma_map_read_en["transition_zone"] = re.sub(
        r"Gecis bolgesi ([\d,]+) civari\. Spot bu seviyenin ustunde kaldikca pin mantigi korunur; alti kabul gorurse piyasa hareketi emen degil, buyuten tarafa gecmeye baslar\.",
        r"The transition zone sits around \1. As long as spot stays above it, the pin logic remains intact; acceptance below it starts to shift the market from absorbing moves to amplifying them.",
        gamma_map_read_en.get("transition_zone", ""),
    )
    gamma_map_read_en["negative_behavior"] = re.sub(
        r"Negatif gamma / hedge hizlanma bolgeleri asagida ([\d,]+) ve ([\d,]+)\. Bu cepler acilirse dealer akisi downside'i buyutebilir\.",
        r"The negative-gamma / hedge-acceleration zones sit lower at \1 and \2. If those pockets open up, dealer flow can amplify downside.",
        gamma_map_read_en.get("negative_behavior", ""),
    )
    gamma_map_read_en["execution_short"] = re.sub(
        r"Short aramasi icin ana bolge ([\d,]+) alti acceptance\. Yani fiyat sadece sarkmakla kalmayip bu seviye altinda tutulmali; o durumda ([\d,]+) ve ([\d,]+) ceplerine dogru hizlanma daha olasi\.",
        r"The main short zone is acceptance below \1. Price should not just wick lower; it needs to stay below that level, which would make acceleration toward the \2 and \3 pockets more likely.",
        tr_to_en_text(gamma_map_read_en.get("execution_short", "")),
    )
    gamma_map_read_en["execution_short"] = re.sub(
        r"Band icinde short ancak ([\d,]+) civari reject veya ([\d,]+) alti acceptance ile aranir\. Band ortasinda yone kovalamak zayif trade olur\.",
        r"Inside the band, shorts should mainly be looked for on a rejection near \1 or on acceptance below \2. Chasing direction from the middle of the band is a weak trade.",
        gamma_map_read_en.get("execution_short", ""),
    )
    gamma_map_read_en["execution_short"] = re.sub(
        r"Spot bandin ustundeyken short ancak ([\d,]+) altina geri donus ve reject ile aranir; aksi halde pozitif gamma tarafinda erken fade etmek zayif kalabilir\.",
        r"When spot is above the band, shorts should mainly be looked for on a move back below \1 with rejection; fading too early while positive gamma is still active can be weak.",
        gamma_map_read_en.get("execution_short", ""),
    )
    gamma_map_read_en["execution_long"] = re.sub(
        r"Long ancak ([\d,]+) geri alinir ve spot tekrar denge bandina yerlesirse aranir; aksi halde yukari denemeler daha cok relief / counter-trend kalir\.",
        r"Longs should only be looked for if \1 is reclaimed and spot returns into the balance band; otherwise upside attempts are more likely to remain relief / counter-trend.",
        tr_to_en_text(gamma_map_read_en.get("execution_long", "")),
    )
    gamma_map_read_en["execution_long"] = re.sub(
        r"Band icinde long, ([\d,]+)-([\d,]+) ustunde tutunma ya da ([\d,]+) ustu acceptance ile daha temiz hale gelir\.",
        r"Inside the band, longs become cleaner on holding above \1-\2 or on acceptance above \3.",
        gamma_map_read_en.get("execution_long", ""),
    )
    gamma_map_read_en["execution_long"] = re.sub(
        r"Long senaryosu spot ([\d,]+) ustunde kabul gordukce ve ([\d,]+) / ([\d,]+) ceplerine tasinma surdukce daha temizdir\.",
        r"The long scenario is cleaner as long as spot keeps accepting above \1 and continues rotating toward the \2 / \3 pockets.",
        gamma_map_read_en.get("execution_long", ""),
    )
    gamma_map_read_en["execution_no_trade"] = re.sub(
        r"([\d,]+) alti reject olup hizli geri alim gelirse bu bolge stop-hunt olabilir; o durumda short kovalamak yerine yeniden acceptance beklenmeli\.",
        r"If \1 is rejected and quickly reclaimed, the move may be a stop-hunt; in that case, wait for renewed acceptance instead of chasing shorts.",
        tr_to_en_text(gamma_map_read_en.get("execution_no_trade", "")),
    )
    gamma_map_read_en["execution_no_trade"] = re.sub(
        r"Spot ([\d,]+)-([\d,]+) araliginda sikişiyorsa ve akis karisiksa en zayif trade ortadan zorla yone girmektir\.",
        r"If spot is compressing inside \1-\2 and flow is mixed, the weakest trade is forcing direction from the middle.",
        gamma_map_read_en.get("execution_no_trade", ""),
    )
    gamma_map_read_en["execution_no_trade"] = re.sub(
        r"Spot bandin ustunde kabul goruyorsa sadece ilk extension'a fade etmek yerine call/gamma surtunmesinin gercekten devreye girdigini gormek gerekir\.",
        r"If spot is accepting above the band, do not fade the first extension blindly; wait to see real call / gamma friction show up.",
        gamma_map_read_en.get("execution_no_trade", ""),
    )
    gamma_map_read_en["base_case"] = re.sub(
        r"Base case: toplam gamma desteği zayif\. Spot halen denge bandinin icinde ama rejim kirilgan; fiyat davranisi daha yonlu ve trigger-bagli okunmali\.",
        r"Base case: net gamma support is weak. Spot is still inside the balance band, but the regime is fragile; price should be read as more directional and trigger-dependent.",
        gamma_map_read_en.get("base_case", ""),
    )
    gamma_map_read_en["base_case"] = re.sub(
        r"Base case: gamma rejimi hala destekleyici\. Spot ([\d,]+) ustunde kaldikca ana beklenti kontrollu trade ve yukari denemelerde once ([\d,]+) civarinda surtunme\.",
        r"Base case: the gamma regime is still supportive. As long as spot holds above \1, expect more controlled trade and initial upside friction near \2.",
        gamma_map_read_en.get("base_case", ""),
    )
    gamma_map_read_en["base_case"] = re.sub(
        r"Base case: toplam gamma hala tamamen bozulmus degil ama spot zayif tarafta\. ([\d,]+) geri alinmadan pozitif gamma konforu geri gelmez\.",
        r"Base case: total gamma is not fully broken, but spot remains on the weak side. Positive-gamma comfort does not return unless \1 is reclaimed.",
        gamma_map_read_en.get("base_case", ""),
    )
    gamma_map_read_en["base_case"] = re.sub(
        r"Base case: toplam gamma desteği zayif\. Spot davranisi daha yonlu ve tetik-bagli okunmali\.",
        r"Base case: net gamma support is weak. Spot should be read as more directional and trigger-dependent.",
        gamma_map_read_en.get("base_case", ""),
    )
    gamma_map_read_en["regime_badge"] = {
        "Pin / Stabilizing": "Pin / Stabilizing",
        "Transition / Fragile": "Transition / Fragile",
        "Negative Gamma Risk Active": "Negative Gamma Risk Active",
    }.get(summary.get("gamma_map_read", {}).get("regime_badge", ""), summary.get("gamma_map_read", {}).get("regime_badge", ""))
    gamma_map_read_en["regime_class"] = summary.get("gamma_map_read", {}).get("regime_class", "delta-neutral")
    summary_en["gamma_map_read"] = gamma_map_read_en
    summary_en["front_expected_first_effect"] = tr_to_en_text(summary.get("front_expected_first_effect", ""))
    summary_en["front_confirmation_needed"] = tr_to_en_text(summary.get("front_confirmation_needed", ""))
    summary_en["front_negative_trigger"] = tr_to_en_text(summary.get("front_negative_trigger", ""))
    summary_en["front_confirmation_needed"] = re.sub(
        r"Tersine donus teyidi icin spotun en az ([\d,]+) ustune geri yerlesmesi and flow tonunun savunmadan karisiga donmesi gerekir\.",
        r"For reversal confirmation, spot needs to reclaim at least \1 and flow tone needs to shift from defensive toward mixed.",
        summary_en["front_confirmation_needed"],
    )
    summary_en["front_negative_trigger"] = re.sub(
        r"Negative tetik: ([\d,]+) alti yeni kabul, expiry sonrasi bile hedge baskisini hizlandirabilir\.",
        r"Negative trigger: fresh acceptance below \1 can accelerate hedge pressure even after expiry.",
        summary_en["front_negative_trigger"],
    )

    takeaways_structured = dict(summary["takeaways_structured"])
    for key in ["gamma", "balance", "upside_note", "downside_note"]:
        takeaways_structured[key] = tr_to_en_text(takeaways_structured[key])
    summary_en["takeaways_structured"] = takeaways_structured

    change_section_en = {
        "notes": tr_to_en_list(change_section.get("notes", [])),
        "headline": tr_to_en_list(change_section.get("headline", [])),
        "weekly": change_section.get("weekly"),
    }
    clean_position_map_section_en = {
        **clean_position_map_section,
        "notes": tr_to_en_list(clean_position_map_section.get("notes", [])),
        "scenarios": tr_to_en_list(clean_position_map_section.get("scenarios", [])),
    }
    position_map_drift_section_en = {
        "headline": tr_to_en_text(position_map_drift_section.get("headline", "")),
        "notes": tr_to_en_list(position_map_drift_section.get("notes", [])),
    }
    gamma_support_monitor_section_en = {
        **gamma_support_monitor_section,
        "headline": tr_to_en_text(gamma_support_monitor_section.get("headline", "")),
        "support_label": tr_to_en_text(gamma_support_monitor_section.get("support_label", "")),
        "forward_verdict": tr_to_en_text(gamma_support_monitor_section.get("forward_verdict", "")),
        "notes": tr_to_en_list(gamma_support_monitor_section.get("notes", [])),
    }
    gex_term_structure_section_en = build_gex_term_structure_section_en(gex_term_structure_section)
    gex_term_structure_section_en["delta_badges"] = [
        tr_to_en_text(str(item))
        .replace("Toplam GEX:", "Total GEX:")
        .replace("On vade ", "Front ")
        .replace("En guclenen:", "Strongest build:")
        .replace("En zayiflayan:", "Weakest leg:")
        for item in gex_term_structure_section.get("delta_badges", [])
    ]
    gex_term_structure_section_en["delta_summary"] = tr_to_en_text(str(gex_term_structure_section.get("delta_summary", "")))
    gex_term_structure_section_en["delta_notes"] = tr_to_en_list(gex_term_structure_section.get("delta_notes", []))
    gex_term_structure_section_en["comparison_label"] = str(gex_term_structure_section.get("comparison_label", ""))
    gex_term_structure_section_en["delta_summary"] = gex_term_structure_section_en["delta_summary"].replace(
        "Dunden bugune gamma destegi genisledi; ozellikle yakin takvimdeki tampon daha destekleyici hale geldi.",
        "Gamma support expanded day over day; the near-dated buffer in particular became more supportive.",
    )
    gex_term_structure_section_en["delta_summary"] = gex_term_structure_section_en["delta_summary"].replace(
        "Toplam gamma destegi artsa da yakin vadedeki tampon ayni gucle genislemiyor; destek daha cok arka vadelerde yogunlasiyor.",
        "Total gamma support increased, but the near-dated buffer did not expand with the same strength; support is concentrating further out on the curve.",
    )
    gex_term_structure_section_en["delta_summary"] = gex_term_structure_section_en["delta_summary"].replace(
        "Dunden bugune on tarafta tampon inceldi; yakin vade davranisi bir gun onceye gore daha kirilgan olabilir.",
        "The front-end buffer thinned day over day; near-dated behavior may now be more fragile than it was one day ago.",
    )
    gex_term_structure_section_en["delta_summary"] = gex_term_structure_section_en["delta_summary"].replace(
        "Vade yapisinda belirgin yeniden dagilim var; bazi tamponlar guclenirken digerleri zayifladi.",
        "The term structure is being redistributed meaningfully; some buffers strengthened while others weakened.",
    )
    layered_trade_check_en = build_layered_trade_check_en(layered_trade_check)
    iv_regime_section_en = {
        "notes": tr_to_en_list(iv_regime_section.get("notes", [])),
        "iv_percentile": iv_regime_section.get("iv_percentile"),
        "iv_rank": iv_regime_section.get("iv_rank"),
        "adj_iv_rank": iv_regime_section.get("adj_iv_rank"),
        "sample_size": iv_regime_section.get("sample_size"),
        "vol_bias": tr_to_en_text(iv_regime_section.get("vol_bias", "")),
        "vol_bias_note": tr_to_en_text(iv_regime_section.get("vol_bias_note", "")),
        "vol_bias_class": iv_regime_section.get("vol_bias_class", "delta-neutral"),
        "checklist": tr_to_en_list(iv_regime_section.get("checklist", [])),
    }
    vanna_charm_section_en = dict(vanna_charm_section)
    vanna_charm_section_en["regime"] = tr_to_en_text(vanna_charm_section.get("regime", ""))
    vanna_charm_section_en["takeaway"] = tr_to_en_text(vanna_charm_section.get("takeaway", ""))
    vanna_charm_section_en["takeaway_badge"] = tr_to_en_text(vanna_charm_section.get("takeaway_badge", ""))
    vanna_charm_section_en["notes"] = tr_to_en_list(vanna_charm_section.get("notes", []))
    vanna_charm_section_en["checklist"] = tr_to_en_list(vanna_charm_section.get("checklist", []))
    pain_section_en = {
        **pain_section,
        "notes": tr_to_en_list(pain_section.get("notes", [])),
    }
    flow_horizon_notes_en = {key: tr_to_en_list(value) for key, value in flow_horizon_notes.items()}

    html = render_html(
        currency,
        summary_en,
        change_section_en,
        dealer_pressure_section_en,
        proxy_hedge_impact_section_en,
        clean_position_map_section_en,
        position_map_drift_section_en,
        gamma_support_monitor_section_en,
        gex_term_structure_section_en,
        layered_trade_check_en,
        vanna_charm_section_en,
        iv_regime_section_en,
        flow_horizon_notes_en,
        expiry_rows,
        expiry_chart,
        gex_chart,
        strike_chart,
        archive_chart,
        flow_chart,
        heatmap_chart,
        clean_position_map_chart,
        position_map_drift_chart,
        gamma_support_monitor_chart,
        gex_term_structure_chart,
        gex_term_structure_delta_chart,
        iv_term_chart,
        iv_rv_chart,
        strategy_chart,
        inventory_chart,
        mm_position_map_chart,
        dealer_pressure_delta_chart,
        proxy_hedge_impact_chart,
        proxy_hedge_impact_trend_chart,
        inventory_trend_chart,
        behavioral_trend_chart,
        behavioral_components_chart,
        layered_trade_check_chart,
        vanna_charm_chart,
        strategy_action_chart,
        roll_transfer_chart,
        normalized_skew_chart,
        iv_regime_chart,
        pain_chart,
        pain_section_en,
        api_calls,
        public_calls,
    )
    html = tr_to_en_text(html)
    html = html.replace("Gunluk delta okuma:", "Daily delta read:")
    html = html.replace(
        "Bu chart mevcut GEX, strike OI ve 24s strike akisini birlestiren bir proxy haritadir. Gercek MM inventory degildir; hangi strike'larda yukari/asagi baski yogunlasiyor onu gosterir.",
        "This chart is a proxy map built from current GEX, strike OI, and 24h strike flow. It is not true MM inventory; it shows where upside and downside pressure is concentrated by strike.",
    )
    html = html.replace(
        "Bu harita gercek dealer inventory degil; strike OI, GEX ve 24s strike akisindan uretilen proxy baski skorudur.",
        "This map is not real dealer inventory; it is a proxy pressure score built from strike OI, GEX, and 24h strike flow.",
    )
    html = html.replace(
        "Ust harita yapisal stok agirligini gosterir; bu nedenle ayni strike'lar birden fazla gun boyunca benzer gorunebilir.",
        "The top map reflects structural stock-like positioning, so the same strikes can look similar for several days in a row.",
    )
    html = html.replace(
        "Alt delta chart'i ayni strike'larda bugun dunden daha fazla yukari mi asagi mi baski biriktigini gosterir.",
        "The delta chart below shows whether more upside or downside pressure accumulated at the same strikes versus yesterday.",
    )
    html = html.replace(
        "Proxy harita stok agirlikli oldugu icin gunluk degisimler bazen yuzeyde yavas gorunur.",
        "Because the proxy map is stock-weighted, day-to-day changes can sometimes look slower on the surface.",
    )
    html = html.replace("Proxy Hedge-Impact Yuzeyi", "Proxy Hedge-Impact Surface")
    html = html.replace("Nasil Okunur?", "How To Read It")
    html = html.replace("Tek satir:", "One-line read:")
    html = html.replace("Mevcut Skor", "Current score")
    html = html.replace("Dunden bugune:", "Day over day:")
    html = html.replace("Dengeleyici / long-gamma-benzeri", "Stabilizing / long-gamma-like")
    html = html.replace("Kirilgan / short-gamma-benzeri", "Fragile / short-gamma-like")
    html = html.replace("Gecis / karisik", "Transition / mixed")
    html = html.replace(
        "Bu chart true dealer inventory kullanmaz; strike OI, proxy pressure ve GEX ile spot farkli seviyelere giderse hedge etkisinin daha dengeleyici mi daha kirilgan mi calisabilecegini idealize eder.",
        "This chart does not use true dealer inventory; it idealizes whether hedge impact would behave in a more stabilizing or more fragile way if spot moved across different levels, using strike OI, proxy pressure, and GEX.",
    )
    html = html.replace(
        "Bu chart true dealer inventory kullanmaz; strike OI, proxy pressure and GEX ile spot farkli seviyelere giderse hedge etkisinin daha dengeleyici mi daha kirilgan mi calisabilecegini idealize eder.",
        "This chart does not use true dealer inventory; it idealizes whether hedge impact would behave in a more stabilizing or more fragile way if spot moved across different levels, using strike OI, proxy pressure, and GEX.",
    )
    html = re.sub(
        r"Proxy model, spotun ([\d,]+-[\d,]+) civarinda daha dengeleyici bir hedge rejimine yaklastigini soyluyor; hareketler emilmeye daha yatkin olabilir\.",
        r"The proxy model suggests spot is sitting near a more stabilizing hedge regime inside the \1 band; moves are more likely to be absorbed than amplified.",
        html,
    )
    html = re.sub(
        r"Proxy model, spotun hala kirilgan bir pocket'ta oldugunu soyluyor; hareketler emilmekten cok buyumeye daha yatkin olabilir\.",
        "The proxy model suggests spot is still sitting in a fragile pocket; moves are more likely to amplify than to be absorbed.",
        html,
    )
    html = re.sub(
        r"Proxy model, spotun temiz bir long-gamma ya da short-gamma pocket'inda olmadigini; rejimin daha gecis karakteri tasidigini soyluyor\.",
        "The proxy model suggests spot is not sitting in a clean long-gamma or short-gamma pocket; the regime looks more transitional.",
        html,
    )
    html = re.sub(
        r"Spot cevresindeki yerel gamma tonu pozitif \(([\+\-]?\d+\.\d+)\); bu, idealize edilmis hedge etkisinin hareketi bastirma yonune kaydigini ima ediyor\.",
        r"Local gamma tone around spot is positive (\1); this implies the idealized hedge effect is leaning toward damping moves.",
        html,
    )
    html = re.sub(
        r"Spot cevresindeki yerel gamma tonu negatif \(([\+\-]?\d+\.\d+)\); bu, idealize edilmis hedge etkisinin hareketi buyutme yonune kayabildigini ima ediyor\.",
        r"Local gamma tone around spot is negative (\1); this implies the idealized hedge effect can lean toward amplifying moves.",
        html,
    )
    html = html.replace(
        "Spot cevresindeki yerel gamma tonu notr; hedge etkisi daha cok akis ve strike yogunluguna bagli.",
        "Local gamma tone around spot is neutral; hedge impact depends more on flow and strike concentration.",
    )
    html = html.replace(
        "Yerel arsiv buyudukce proxy hedge-impact rejiminin gunler icindeki kaymasi burada izlenir.",
        "As the local archive grows, this panel tracks how the proxy hedge-impact regime shifts through time.",
    )
    html = html.replace(
        "Proxy hedge-impact skoru dunden bugune belirgin guclendi; piyasa daha dengeleyici bir pocket'e kayiyor olabilir.",
        "The proxy hedge-impact score strengthened meaningfully day over day; the market may be moving toward a more stabilizing pocket.",
    )
    html = html.replace(
        "Proxy hedge-impact skoru dunden bugune zayifladi; hedge etkisi daha kirilgan bir rejime kayiyor olabilir.",
        "The proxy hedge-impact score weakened day over day; hedge impact may be shifting toward a more fragile regime.",
    )
    html = html.replace(
        "Proxy hedge-impact skoru dunden bugune sinirli degisti; temel rejim ayni kalsa da pocket seviyeleri hafif yer degistirmis olabilir.",
        "The proxy hedge-impact score changed only modestly day over day; the core regime is similar, but pocket levels may have shifted slightly.",
    )
    html = html.replace(
        "Proxy hedge-impact skoru day over day weakladi; hedge etkisi daha kirilgan bir rejime kayiyor olabilir.",
        "The proxy hedge-impact score weakened day over day; hedge impact may be shifting toward a more fragile regime.",
    )
    html = html.replace(
        "Proxy hedge-impact skoru day over day belirgin guclendi; piyasa daha dengeleyici bir pocket'e kayiyor olabilir.",
        "The proxy hedge-impact score strengthened meaningfully day over day; the market may be moving toward a more stabilizing pocket.",
    )
    html = html.replace(
        "Proxy hedge-impact skoru day over day sinirli degisti; temel rejim ayni kalsa da pocket seviyeleri hafif yer degistirmis olabilir.",
        "The proxy hedge-impact score changed only modestly day over day; the core regime is similar, but pocket levels may have shifted slightly.",
    )
    html = re.sub(
        r"Proxy hedge-impact skoru day over day weakladi; hedge etkisi daha kirilgan bir rejime kayiyor olabilir\.",
        "The proxy hedge-impact score weakened day over day; hedge impact may be shifting toward a more fragile regime.",
        html,
    )
    html = re.sub(
        r"Proxy hedge-impact skoru day over day belirgin guclendi; piyasa daha dengeleyici bir pocket'e kayiyor olabilir\.",
        "The proxy hedge-impact score strengthened meaningfully day over day; the market may be moving toward a more stabilizing pocket.",
        html,
    )
    html = re.sub(
        r"Proxy hedge-impact skoru day over day sinirli degisti; temel rejim ayni kalsa da pocket seviyeleri hafif yer degistirmis olabilir\.",
        "The proxy hedge-impact score changed only modestly day over day; the core regime is similar, but pocket levels may have shifted slightly.",
        html,
    )
    html = html.replace(
        "Bu bolum true dealer position degil; strike OI, proxy pressure ve GEX'ten turetilen senaryo bazli bir hedge-impact egzersizidir.",
        "This section is not true dealer positioning; it is a scenario-based hedge-impact exercise derived from strike OI, proxy pressure, and GEX.",
    )
    html = html.replace(
        "Bu bolum true dealer position degil; strike OI, proxy pressure and GEX'ten turetilen senaryo bazli bir hedge-impact egzersizidir.",
        "This section is not true dealer positioning; it is a scenario-based hedge-impact exercise derived from strike OI, proxy pressure, and GEX.",
    )
    html = html.replace(
        "Pozitif skor, dealerlarin hareketi emmeye daha yakin oldugu long-gamma-benzeri bir pocket'i; negatif skor ise daha kirilgan short-gamma-benzeri davranisi ima eder.",
        "A positive score implies a long-gamma-like pocket where dealers are more likely to absorb moves; a negative score implies more fragile short-gamma-like behavior.",
    )
    html = html.replace(
        "Positive skor, dealerlarin hareketi emmeye daha yakin oldugu long-gamma-benzeri bir pocket'i; negatif skor ise daha kirilgan short-gamma-benzeri davranisi ima eder.",
        "A positive score implies a long-gamma-like pocket where dealers are more likely to absorb moves; a negative score implies more fragile short-gamma-like behavior.",
    )
    html = re.sub(
        r"En guclu stabilizing pocket ([\d,]+) civarinda, en kirilgan pocket ise ([\d,]+) civarinda gorunuyor\.",
        r"The strongest stabilizing pocket appears near \1, while the most fragile pocket appears near \2.",
        html,
    )
    html = re.sub(
        r"En strong stabilizing pocket ([\d,]+) civarinda, en kirilgan pocket ise ([\d,]+) civarinda gorunuyor\.",
        r"The strongest stabilizing pocket appears near \1, while the most fragile pocket appears near \2.",
        html,
    )
    html = html.replace(
        "Bu trend true dealer pozisyonu degil; gunluk snapshot'lardan uretilen proxy hedge-impact skorunun zaman serisini gosterir.",
        "This trend is not true dealer positioning; it shows the time series of the proxy hedge-impact score built from daily snapshots.",
    )
    html = re.sub(
        r"Son gunluk kaymada dengeleyici pocket ([\+\-]\d+), kirilgan pocket ise ([\+\-]\d+) puan yer degistirdi\.",
        r"On the latest daily move, the stabilizing pocket shifted by \1 points and the fragile pocket shifted by \2 points.",
        html,
    )
    html = html.replace(
        "Proxy model, spotun 67,959-86,459 civarinda daha dengeleyici bir hedge rejimine yaklastigini soyluyor; hareketler emilmeye daha yatkin olabilir.",
        "The proxy model suggests spot is sitting near a more stabilizing hedge regime inside the 67,959-86,459 band; moves are more likely to be absorbed than amplified.",
    )
    html = html.replace(
        "Spot cevresindeki yerel gamma tonu pozitif (+0.83); bu, idealize edilmis hedge etkisinin hareketi bastirma yonune kaydigini ima ediyor.",
        "Local gamma tone around spot is positive (+0.83); this implies the idealized hedge effect is leaning toward damping moves.",
    )
    html = html.replace(
        "Dunden bugune en net yukari kayma ",
        "The clearest day-over-day upside shift was around ",
    )
    html = html.replace(
        "Dunden bugune en net asagi kayma ",
        "The clearest day-over-day downside shift was around ",
    )
    html = html.replace(
        " cevresinde; ana harita benzer kalsa bile taze baski dagilimi yukari yone kaydi.",
        "; even if the main map still looks similar, fresh pressure rotated toward the upside.",
    )
    html = html.replace(
        " cevresinde; ana harita benzer kalsa bile taze baski dagilimi savunmaci tarafa kaydi.",
        "; even if the main map still looks similar, fresh pressure rotated toward the defensive side.",
    )
    html = html.replace("Yukari baski:", "Upside pressure:")
    html = html.replace("Asagi baski:", "Downside pressure:")
    html = html.replace("Yukari build:", "Upside build:")
    html = html.replace("Asagi build:", "Downside build:")
    html = html.replace("Yesterday vs Today Expiry Bazli GEX Degisimi", "Day-over-Day GEX Change by Expiry")
    html = html.replace("Dunden Bugune Vade Bazli GEX Degisimi", "Day-over-Day GEX Change by Expiry")
    html = html.replace("Delta okuma:", "Delta read:")
    html = html.replace("En weaklayan:", "Weakest leg:")
    html = re.sub(
        r"Spot su an pozitif gamma denge bandinin altinda; ([\d,]+) alti artik gecis / weaklama bolgesi\.",
        r"Spot is currently below the positive-gamma balance band; below \1 is now a transition / weakening zone.",
        html,
    )
    html = re.sub(
        r"Bu bolgede hareket emilmek yerine daha yonlu and kirilgan olmaya baslar; ([\d,]+) and ([\d,]+) civarindaki negatif gamma cepleri daha kolay aktive olabilir\.",
        r"In this zone, moves are less likely to be absorbed and more likely to turn directional and fragile; the negative-gamma pockets near \1 and \2 can activate more easily.",
        html,
    )
    html = re.sub(
        r"Positive gamma tarafinda \(([\d,]+)-([\d,]+)\) fiyat davranisi daha kontrollu, geri cekilmeye yatkin and hedge akisiyla bastirilmaya acik\.",
        r"In positive gamma (\1-\2), price action is more controlled, more mean-reverting, and more likely to be dampened by hedge flow.",
        html,
    )
    html = re.sub(
        r"Gecis bolgesi ([\d,]+)\. Spot bu seviyenin ustunde kaldikca pin mantigi korunur; alti kabul gorurse piyasa hareketi emen degil, buyuten tarafa gecmeye baslar\.",
        r"The transition zone sits around \1. As long as spot stays above it, the pin logic remains intact; acceptance below it starts to shift the market from absorbing moves to amplifying them.",
        html,
    )
    html = re.sub(
        r"Negative gamma / hedge hizlanma bolgeleri asagida ([\d,]+) and ([\d,]+)\. Bu cepler acilirse dealer akisi downside'i buyutebilir\.",
        r"The negative-gamma / hedge-acceleration zones sit lower at \1 and \2. If those pockets open up, dealer flow can amplify downside.",
        html,
    )
    html = re.sub(
        r"Base case: toplam gamma desteği weak\. Spot davranisi daha yonlu and tetik-bagli okunmali\.",
        r"Base case: net gamma support is weak. Spot should be read as more directional and trigger-dependent.",
        html,
    )
    html = re.sub(
        r"Bu rejimde temel davranis mean reversion, pin and hareket bastirma\. En dogal merkez ([\d,]+); ilk refleks genelde bu eksene geri cekilme olur\.",
        r"In this regime, the default behavior is mean reversion, pinning, and move suppression. The natural center is \1, and the first reflex is usually a pullback toward that axis.",
        html,
    )
    html = re.sub(
        r"Bu bolgede yukari devam denenebilir ama ilk call / gamma surtunmesi ([\d,]+) and ([\d,]+) ceplerinde gelir; hareket temiz trendden cok yapiskan extension da uretebilir\.",
        r"In this zone, upside continuation can be tested, but the first call / gamma friction appears around \1 and \2; the move can produce sticky extension rather than a clean trend.",
        html,
    )
    html = re.sub(
        r"Base case: net gamma destegi zayif ama spot halen denge bandinin icinde\. Rejim tamamen bozulmus degil; fiyat daha kirilgan, fakat ilk okuma hala kontrollu / range benzeri davranis\.",
        r"Base case: net gamma support is weak, but spot is still inside the balance band. The regime is not fully broken; price is more fragile, but the first read is still controlled / range-like behavior.",
        html,
    )
    html = re.sub(
        r"Base case: gamma hala tamamen bozulmus degil ama spot ana denge bandinin disinda\. ([\d,]+) civarina geri cekilme and yeni acceptance arayisi daha olasi\.",
        r"Base case: gamma is not fully broken, but spot is outside the main balance band. A pullback toward \1 and a search for new acceptance are more likely.",
        html,
    )
    html = re.sub(
        r"Base case: net gamma destegi zayif and spot artik denge bandinin altinda\. Davranis daha yonlu and tetik-bagli okunmali; ([\d,]+) and ([\d,]+) cepleri daha hizli aktive olabilir\.",
        r"Base case: net gamma support is weak and spot is now below the balance band. Behavior should be read as more directional and trigger-driven; the \1 and \2 pockets can activate faster.",
        html,
    )
    html = re.sub(
        r"Call cluster spotdan uzaklasti \(([^\)]+)\); yakin upside ilgisi weakladi\.",
        r"The call cluster moved farther from spot (\1); nearby upside interest weakened.",
        html,
    )
    html = re.sub(
        r"The inferred call-strike transfer is ([\d,]+) -> ([\d,]+) yonunde asagiya; bu, upside target/cap seviyesinin yer degistirdigine isaret edebilir\.",
        r"The inferred call-strike transfer moved lower from \1 to \2, which may indicate a repositioning of the upside target / cap.",
        html,
    )
    html = re.sub(
        r"The clearest chain shift over the last 24 hours occurred at ([\d,]+) strike'inda asagi yone oldu \(([-+\d\.]+mn)\); this level should be read as a new short-dated focus area\.",
        r"The clearest chain shift over the last 24 hours occurred lower at the \1 strike (\2); this level should be read as a new short-dated focus area.",
        html,
    )
    html = re.sub(
        r"Base case: net gamma destegi weak and spot artik denge bandinin altinda\. Davranis daha yonlu and tetik-bagli okunmali; ([\d,]+) and ([\d,]+) cepleri daha hizli aktive olabilir\.",
        r"Base case: net gamma support is weak and spot is now below the balance band. Behavior should be read as more directional and trigger-driven; the \1 and \2 pockets can activate faster.",
        html,
    )
    html = re.sub(
        r"Kalabalik kovalama skoru ([+-]?\d+): Notr / karisik akis\.",
        r"Crowd-chase score \1: neutral / mixed flow.",
        html,
    )
    html = re.sub(
        r"Akis karmasik\. Bu nedenle davranissal tarafta strong bir fade ya da follow sinyali yok; seviye teyidi sinyalin kendisinden daha onemli\.",
        r"Flow is mixed. That means there is no strong fade or follow signal on the behavioral side; level confirmation matters more than the signal itself.",
        html,
    )
    html = re.sub(
        r"On the upside, en kalabalik call cepleri ([\d,]+) and ([\d,]+); and those levels can act as resistance as much as targets on first test\.",
        r"On the upside, the heaviest call pockets sit at \1 and \2; those levels can act as resistance as much as targets on first test.",
        html,
    )
    html = re.sub(
        r"Flow score ([+-]?\d+) and baskin ton mixed / two-way akisi\.",
        r"Flow score \1 with a dominant mixed / two-way flow tone.",
        html,
    )
    html = re.sub(
        r"and then ([\d,]+) test edilir, fakat bunun trend gunune donusmesi icin bu seviyelerde kabul and yeni call eklenmesi gorulmeli\.",
        r"and then test \1, but turning that into a true trend day would still require acceptance at those levels and fresh call adding.",
        html,
    )
    html = re.sub(
        r"Charm proxy (\d+): yakin vade OI yogunlugu dusuk and dealer gamma ile birlikte spotta zaman-decay kaynakli pin baskisi sinirli\.",
        r"Charm proxy \1: near-dated OI concentration is low, so time-decay-driven pin pressure remains limited even with dealer gamma in the background.",
        html,
    )
    html = re.sub(
        r"Net okuma Low compression / daha serbest fiyat kesfi\. Spot pin merkezine yakinlik skoru %(\d+), gamma support is (\d+)\.",
        r"Net read: low compression / freer price discovery. Spot-to-pin proximity score is \1%, gamma support is \2.",
        html,
    )
    html = html.replace("Katmanli Teyit", "Layered Trade Check")
    html = html.replace("Bias", "Bias")
    html = html.replace("Trade Gate", "Trade Gate")
    html = html.replace("Long Skor", "Long Score")
    html = html.replace("Short Skor", "Short Score")
    html = html.replace("20g Z-Score", "20d Z-Score")
    html = html.replace("Short Nerde Aranir?:", "Where To Look For Shorts?:")
    html = html.replace("Long Nerde Aranir?:", "Where To Look For Longs?:")
    html = html.replace("No-Trade / Teyitsiz Bolge:", "No-Trade / Unconfirmed Zone:")
    html = re.sub(
        r"Short aramasi icin ana bolge ([\d,]+) alti acceptance\. Yani fiyat sadece sarkmakla kalmayip bu seviye altinda tutulmali; o durumda ([\d,]+) and ([\d,]+) ceplerine dogru hizlanma daha olasi\.",
        r"The main short zone is acceptance below \1. Price should not just wick lower; it needs to stay below that level, which would make acceleration toward the \2 and \3 pockets more likely.",
        html,
    )
    html = re.sub(
        r"Long ancak ([\d,]+) geri alinir and spot tekrar denge bandina yerlesirse aranir; aksi halde yukari denemeler daha cok relief / counter-trend kalir\.",
        r"Longs should only be looked for if \1 is reclaimed and spot returns into the balance band; otherwise upside attempts are more likely to remain relief / counter-trend.",
        html,
    )
    html = re.sub(
        r"Base case: toplam gamma tamamen bozulmus degil ama spot weak tarafta\. ([\d,]+) geri alinmadan pozitif gamma konforu geri gelmez\.",
        r"Base case: total gamma is not fully broken, but spot remains on the weak side. Positive-gamma comfort does not return unless \1 is reclaimed.",
        html,
    )
    html = html.replace("Beklenen aksiyon:", "Expected action:")
    html = html.replace("Bu Setup Neyi Arar?", "What Is This Looking For?")
    html = html.replace("Teknik referans:", "Technical reference:")
    html = html.replace("Bollinger alt bant", "Bollinger lower band")
    html = html.replace(", ust bant", ", upper band")
    html = html.replace("Gamma Destek MonitÃ¶rÃ¼", "Gamma Support Monitor")
    html = html.replace("Gamma Destek Monitörü", "Gamma Support Monitor")
    html = html.replace("GEX Vade Yapisi", "GEX Term Structure")
    html = html.replace("Term Structure Okumasi", "Term Structure Read")
    html = html.replace("On Vade", "Front Expiry")
    html = html.replace("On Vade GEX", "Front GEX")
    html = html.replace("Toplam GEX", "Total GEX")
    html = html.replace("Bugunku Skor", "Current Score")
    html = html.replace("Gunluk Degisim", "Daily Change")
    html = html.replace("14g OI Payi", "14d OI Share")
    html = html.replace("Yukari baski artti:", "Upside pressure rose:")
    html = html.replace("Yukari baski azaldi:", "Upside pressure fell:")
    html = html.replace("Asagi baski artti:", "Downside pressure rose:")
    html = html.replace("Asagi baski azaldi:", "Downside pressure fell:")
    html = html.replace("30g OI Payi", "30d OI Share")
    html = html.replace("<th>Katman</th>", "<th>Layer</th>")
    html = html.replace("<th>Okuma</th>", "<th>Read</th>")
    html = html.replace("hareket korkuyla fiyatlaniyor, dusus teyidi daha strong.", "the move is being priced with fear, so downside confirmation is stronger.")
    html = html.replace("hareket korkuyla fiyatlaniyor, dusus teyidi daha guclu.", "the move is being priced with fear, so downside confirmation is stronger.")
    html = html.replace("RSI(14) 43.3: teknik ekstrem net degil.", "RSI(14) 43.3: no clear technical extreme.")
    html = html.replace(
        "Bu blok tek bir sinyal degil; teknik ekstrem + vol tepkisi + skew davranisi + positioning filtresinin ust uste binip binmedigini kontrol eder.",
        "This block is not a single signal; it checks whether technical extremes, vol reaction, skew behavior, and the positioning filter line up.",
    )
    html = re.sub(
        r"RSI\(14\) ([\d\.]+): teknik ekstrem net degil\.",
        r"RSI(14) \1: no clear technical extreme.",
        html,
    )
    html = re.sub(
        r"IV-RV spread ([+-]?[\d\.]+): vol tepkisi ara bolgede, tek basina karar vermiyor\.",
        r"IV-RV spread \1: the vol reaction sits in the middle zone and does not decide the trade on its own.",
        html,
    )
    html = re.sub(
        r"IV-RV spread ([+-]?[\d\.]+): hareket korkuyla fiyatlaniyor, dusus teyidi daha strong\.",
        r"IV-RV spread \1: the move is being priced with fear, so downside confirmation is stronger.",
        html,
    )
    html = re.sub(
        r"IV-RV spread ([+-]?[\d\.]+): hareket korkuyla fiyatlaniyor, dusus teyidi daha guclu\.",
        r"IV-RV spread \1: the move is being priced with fear, so downside confirmation is stronger.",
        html,
    )
    html = re.sub(
        r"25d RR ([+-]?[\d\.]+) puan daha savunmaci; skew rahatlamiyor\.",
        r"25d RR is \1 points more defensive; skew is not relaxing.",
        html,
    )
    html = re.sub(
        r"Onumuzdeki 14 gunde gorunen expiry OI'nin %([\d\.]+)'i, 30 gunde %([\d\.]+)'i, 60 gunde %([\d\.]+)'i birikiyor\.",
        r"Of the visible expiry OI, \1% is concentrated within 14 days, \2% within 30 days, and \3% within 60 days.",
        html,
    )
    html = re.sub(
        r"Spot denge bandinin altinda / gamma kirilgan; positioning downside devamini daha kolay aktive ediyor\.",
        r"Spot is below the balance band / gamma is fragile; positioning makes downside continuation easier to activate.",
        html,
    )
    html = re.sub(
        r"On vade ([A-Z0-9]+) negatif gamma ile ([-\d,]+); bu expiry gectikce pasif deste(?:gin|ğin|ÄŸin) ilk nereye kayacagi term structure ile okunabilir\.",
        r"The front expiry \1 carries negative gamma at \2; the term structure shows where passive support is likely to rotate first as this expiry rolls off.",
        html,
    )
    forced_replacements = [
        ("magnet asagi | put zone sabit | call cluster sabit", "magnet lower | put zone unchanged | call cluster unchanged"),
        ("Magnet strike is is -5,000 puan asagi kaydi; denge merkezi weakladi.", "Magnet strike shifted 5,000 points lower; the balance center weakened."),
        ("Call cluster spotdan uzaklasti (21,157 -> 23,447); yakin upside ilgisi weakladi.", "The call cluster moved farther from spot (21,157 -> 23,447); nearby upside interest weakened."),
        ("Vol egirisi yukari egimli; piyasa yakin riski degil daha kalici belirsizligi fiyatliyor.", "The vol curve slopes upward; the market is pricing longer-lasting uncertainty rather than near-dated panic."),
        ("Vol egirisi gorece duz, yani yakin vade panik fiyati baskin degil.", "The vol curve is relatively flat, meaning near-dated panic pricing is not dominant."),
        ("Vol egirisi asagidan yukari diklesiyor; yakin vade stresine gore daha kalici belirsizlik fiyatlaniyor.", "The vol curve steepens upward; the market is pricing more persistent uncertainty than immediate front-end stress."),
        ("The next-day expiry 03APR26 is only 7.1% of total visible expiry OI, so on its own it is unlikely to drag the full band unless spot trades directly into crowded strikes.", "The next-day expiry 03APR26 is only 7.1% of total visible expiry OI, so on its own it is unlikely to drag the full band unless spot trades directly into crowded strikes."),
        ("The next-day expiry 10APR26 is only 7.1% of total visible expiry OI, so on its own it is unlikely to drag the full band unless spot trades directly into crowded strikes.", "The next-day expiry 10APR26 is only 7.1% of total visible expiry OI, so on its own it is unlikely to drag the full band unless spot trades directly into crowded strikes."),
        ("Ana yukari gamma miknatisi 75000 seviyesinden 72000 seviyesine kaydi.", "The main upside gamma magnet shifted from 75,000 to 72,000."),
        ("The inferred call-strike transfer is 78,000 -> 76,000 yonunde asagiya; bu, upside target/cap seviyesinin yer degistirdigine isaret edebilir.", "The inferred call-strike transfer moved lower from 78,000 to 76,000, which may indicate a repositioning of the upside target / cap."),
        ("The center of net strike flow moved from 75,000 to 63,000'a asagiya kayiyor; bu da yeni ilginin hangi strike'a toplandigini gosteriyor.", "The center of net strike flow moved lower from 75,000 to 63,000, showing where new interest is concentrating."),
        ("Teyit sarti: spotun 71,000 ustunde kalmasi and yukari testlerde yeni savunmaci akisin geri gelmemesi.", "Confirmation needed: spot needs to hold above 71,000 and fresh defensive flow must not return on upside tests."),
        ("Negative tetik: 66,500 alti kabul, expiry rahatlamasi beklemek yerine savunmaci tonun dogru okundugunu teyit eder.", "Negative trigger: acceptance below 66,500 would confirm the defensive read rather than an expiry-relief thesis."),
        ("Yukari senaryo kosullu and teyit istiyor; asagi taraf ise kirilma halinde daha kolay hizlanabilir.", "The upside case is conditional and still needs confirmation; downside can accelerate more easily on a break."),
        ("66,500. . A sustained break below 65,000 and 60,000 would expose negative-gamma pockets in that zone.", "66,500. A sustained break below 65,000 and 60,000 would expose negative-gamma pockets in that zone."),
        ("Accept above 80,000: spot bu seviye ustunde kalir and yeni call akisiyla desteklenirse yukari test aktif olur.", "Accept above 80,000: if spot holds above the level and fresh call flow confirms, the upside test becomes active."),
        ("Kalabalik kovalama skoru -5: Retail-benzeri gec put panigi.", "Crowd-chase score -5: retail-style late put panic."),
        ("Kisa vadeli put talebi dususun arkasindan geliyor gibi duruyor. Bu akisin buyuk kismi gec hedge ise, 66,500 alti kalici acceptance gelmeden asagi follow-through sinyali tek basina strong sayilmaz.", "Short-dated put demand appears to be arriving after the drop. If much of this flow is late hedging, downside follow-through should not be treated as strong on its own unless there is sustained acceptance below 66,500."),
        ("Davranissal Akis Avantaji", "Behavioral Flow Edge"),
        ("Ertesi Gun Vadesi", "Next-Day Expiry"),
        ("Seviyeler:", "Levels:"),
        ("Bugun Ne Degisti?", "What Changed Today?"),
        ("Dunden Bugune", "From Yesterday to Today"),
        ("Vade Yapisi", "Expiry Structure"),
        ("En Yakin Vade Detayi", "Nearest Expiry Detail"),
        ("On Vade Karari:", "Front Expiry Verdict:"),
        ("Teyit Sarti:", "Confirmation Needed:"),
        ("Negatif Tetik:", "Negative Trigger:"),
        ("Bugunun rejimi:", "Today's regime:"),
        ("Risk asimetrisi:", "Risk asymmetry:"),
        ("Yukari teyit:", "Upside confirmation:"),
        ("Asagi tetik:", "Downside trigger:"),
        ("Tek cumlelik masa notu:", "One-line desk note:"),
        ("Yakin Vade Nasil Okunur?", "How Should The Front Expiry Be Read?"),
        ("Laevitas historical RV endpoint'i bu key ile kapali oldugu icin realized vol public BTC gunluk kapanis serisinden hesaplanir.", "Because the Laevitas historical RV endpoint is not available under this key, realized vol is calculated from the public BTC daily close series."),
        ("Normalize oran yuksek kaldikca downside korumasi call talebine gore daha pahali kalir.", "As the normalized ratio stays high, downside protection remains more expensive than call demand."),
        ("Bu chart mevcut GEX, strike OI ve 24s strike akisini birlestiren bir proxy haritadir. Gercek MM inventory degildir; hangi strike'larda yukari/asagi baski yogunlasiyor onu gosterir.", "This chart is a proxy map that combines current GEX, strike OI, and 24h strike flow. It is not true MM inventory; it shows where upside/downside pressure is concentrated by strike."),
        ("Bu grafik bugunku zone'lari tum gecmise yaymaz; gunluk arsivden magnet, put hedge, call cluster ve speculation seviyelerinin nasil kaydigini gosterir.", "This chart does not project today's zones across the full history; it shows how magnet, put hedge, call cluster, and speculation levels have shifted in the daily archive."),
        ("Implied vol yerel seriye gore ucuz; realized hareket IV'yi zorlayabiliyorsa vol alici bakis daha mantikli.", "Implied vol looks cheap versus the local sample; if realized movement can keep pressuring IV, a vol-buyer stance makes more sense."),
        ("Percentile dusuk kalmaya devam ediyor mu?", "Is percentile staying low?"),
        ("Realized hareket implied volu asiyor mu?", "Is realized movement running above implied vol?"),
        ("Spotta yeni yonlu/kopmali hareket riski artiyor mu?", "Is the risk of a fresh directional / breakout move increasing in spot?"),
        ("Kalabalik kovalama skoru -1: Notr / karisik akis.", "Crowd-chase score -1: neutral / mixed flow."),
        ("Kalabalik kovalama skoru -3: Retail-benzeri gec put panigi.", "Crowd-chase score -3: late retail-style put panic."),
        ("Retail-benzeri gec put panigi", "Late retail-style put panic"),
        ("Kisa vadeli put talebi dususun arkasindan geliyor gibi duruyor. Bu akisin buyuk kismi gec hedge ise, 70,000 alti kalici acceptance gelmeden asagi follow-through sinyali tek basina strong sayilmaz.", "Short-dated put demand looks like it is arriving after the drop. If most of this flow is late hedging, downside follow-through should not be treated as strong on its own unless there is sustained acceptance below 70,000."),
        ("Akis karmasik. Bu nedenle davranissal tarafta strong bir fade ya da follow sinyali yok; seviye teyidi sinyalin kendisinden daha onemli.", "Flow is mixed. That means there is no strong fade or follow signal on the behavioral side; level confirmation matters more than the signal itself."),
        ("Flow score:</strong> Mixed / two-way akisi (+1)", "Flow score:</strong> Mixed / two-way flow (+1)"),
        ("Flow score:</strong> Mixed / two-way akisi (+0)", "Flow score:</strong> Mixed / two-way flow (+0)"),
        ("Flow score:</strong> Mixed / two-way akisi (-1)", "Flow score:</strong> Mixed / two-way flow (-1)"),
        ("Put/Call imbalance (+0): Put ve call eklenmesi birbirine yakin.", "Put/Call imbalance (+0): put and call adding are close to balanced."),
        ("Put/Call imbalance (+0): Put and call eklenmesi birbirine yakin.", "Put/Call imbalance (+0): put and call adding are close to balanced."),
        ("Put/Call imbalance (-2): Put eklenmesi call eklenmesinden belirgin strong.", "Put/Call imbalance (-2): put adding is clearly stronger than call adding."),
        ("Premium stress (-1): Opsiyon primi pahali and put tarafi panige daha yakin.", "Premium stress (-1): option premium is rich and the put side is closer to panic."),
        ("On the upside, en kalabalik call cepleri 80,000 and 75,000; and those levels can act as resistance as much as targets on first test.", "On the upside, the heaviest call pockets sit at 80,000 and 75,000; those levels can act as resistance as much as targets on first test."),
        ("Flow score -2 and baskin ton defensive / downside hedge akisi.", "Flow score -2 with a dominant defensive / downside-hedge tone."),
        ("then 75,000 test edilir, fakat bunun trend gunune donusmesi icin bu seviyelerde kabul and yeni call eklenmesi gorulmeli.", "and then test 75,000, but turning that into a true trend day would still require acceptance at those levels and fresh call adding."),
        ("Magnet strike is 70,000; bu seviye en buyuk notional dugumu degil, daha cok denge / pin karakteri tasiyan strike olarak secilir. Spot bu seviyeye yaklastikca pin / geri cekilme etkisi daha anlamli hale gelebilir.", "Magnet strike is 70,000; this is not simply the largest notional node, but the strike with the cleaner balance / pin character. As spot approaches it, pin / pullback behavior becomes more relevant."),
        ("Asagida 60,000 civarinda daha strong bir hedge duvari var; bu seviye magnet degil, savunma / protection anchor gibi okunmali.", "Lower down, there is a stronger hedge wall around 60,000; this should be read as a defense / protection anchor, not as the magnet."),
        ("Spot 70,000 altinda; yukari hareketin daha temiz pozitif okumaya donmesi icin once magnet strike geri alinmali.", "Spot is below 70,000; for the upside read to turn cleaner, price first needs to reclaim the magnet strike."),
        ("Magnet strike is is dunden bugune -5,000 puan kaydi.", "Magnet strike shifted -5,000 points day over day."),
        ("Charm proxy 19: yakin vade OI yogunlugu dusuk and dealer gamma ile birlikte spotta zaman-decay kaynakli pin baskisi sinirli.", "Charm proxy 19: near-dated OI concentration is low, so time-decay-driven pin pressure remains limited even with dealer gamma in the background."),
        ("Net okuma Low compression / daha serbest fiyat kesfi. Spot pin merkezine yakinlik skoru %16, gamma support is 0.", "Net read: low compression / freer price discovery. Spot-to-pin proximity is 16% and gamma support is effectively zero."),
        ("Dusuk compression / daha serbest fiyat kesfi", "Low compression / freer price discovery"),
        ("Yakin 7 gunde belirgin quarterly expiry yogunlugu yok.", "There is no meaningful quarterly expiry concentration over the next 7 days."),
        ("Spot 70,000-75,000 bandinda kaldikca compression proxy daha anlamli calisir.", "The compression proxy matters more while spot remains inside the 70,000-75,000 band."),
        ("Pasif destek guclu", "Passive support strong"),
        ("Pasif destek aktif", "Passive support active"),
        ("Pasif destek inceliyor", "Passive support thinning"),
        ("Pasif destek zayif", "Passive support weak"),
        ("Pasif destek weak", "Passive support weak"),
        ("Bu grafik gercek expiry-bazli GEX dagilimini gosterir; pasif destegin hangi vadelerde tasindigini and negatif gamma riskinin nerede biriktigini burada goruruz.", "This chart shows the true expiry-by-expiry GEX distribution; it shows which maturities are carrying passive support and where negative-gamma risk is building."),
        ("En strong pozitif gamma vadesi 10APR26 (1,307,746); pasif destek en cok burada yogunlasiyor.", "The strongest positive-gamma expiry is 10APR26 (1,307,746); passive support is concentrated there most heavily."),
        ("On vade 7APR26 negatif gamma ile -147,647; bu expiry gectikce pasif destegin ilk nereye kayacagi term structure ile okunabilir.", "The front expiry 7APR26 carries negative gamma at -147,647; the term structure shows where passive support is likely to rotate first as this expiry rolls off."),
        ("On vade 7APR26 negatif gamma ile -147,647; bu expiry gectikce pasif desteÄŸin ilk nereye kayacagi term structure ile okunabilir.", "The front expiry 7APR26 carries negative gamma at -147,647; the term structure shows where passive support is likely to rotate first as this expiry rolls off."),
        ("Pasif destek on tarafta yogun; yakin vadeler hala aktif tampon uretiyor.", "Passive support is concentrated near the front; nearby expiries are still providing an active buffer."),
        ("Pasif destek tamamen bitmis degil ama daha dis vadeye tasiniyor.", "Passive support is not fully gone, but it is shifting farther out on the calendar."),
        ("Beklenen sey: pasif destek yakin vadede hala aktif; en yogun pencere ", "Expected path: passive support remains active in the near-dated window; the densest window is "),
        (" boyunca pin / mean-reversion davranisi daha kolay surer.", ", where pin / mean-reversion behavior can persist more easily."),
        ("Beklenen sey: destek hemen bitmiyor ama takvime yayiliyor; en yogun pencere ", "Expected path: support does not disappear immediately, but it is spreading out on the calendar; the densest window is "),
        (", sonrasinda tampon daha secici hale geliyor.", ", after which the buffer becomes more selective."),
        ("Beklenen sey: yakin vade pasif destek zayif; ", "Expected path: near-dated passive support is weak; "),
        (" disinda belirgin bir tampon yoksa fiyat daha yonlu davranabilir.", " aside, if no clear buffer builds, price can behave more directionally."),
        ("Ust panel pasif destek skorunun son gunlerde guclenip weakladigini, alt panel ise onumuzdeki aylarda destegin hangi vade pencerelerinde yogunlastigini gosterir.", "The top panel shows whether passive support has strengthened or weakened recently; the bottom panel shows which future expiry windows concentrate that support across the coming months."),
        ("Bugunku pasif destek skoru", "Today's passive support score"),
        ("Of the visible expiry OI, %", "Of the visible expiry OI, "),
        ("'i, within 30 days %", "% is concentrated within 14 days, "),
        ("'i, within 60 days %", "% within 30 days, "),
        ("'i is concentrated.", "% within 60 days."),
        ("dunden bugune", "day over day"),
        ("son 5 baz gune gore", "versus the last 5-session baseline"),
        ("Onumuzdeki 14 gunde gorunen expiry OI'nin", "Of the visible expiry OI,"),
        ("30 gunde", "within 30 days"),
        ("60 gunde", "within 60 days"),
        ("birikiyor.", "is concentrated."),
        ("Takvimde en yogun pasif destek penceresi", "The densest passive-support window on the calendar is"),
        ("en ince pencere", "while the thinnest window is"),
        ("(skor ", "(score "),
        ("Yapi tamamen bosalmis degil, ama destek artik tek bir gunluk expiry duvarindan cok dagilan bir takvim etkisiyle tasiniyor.", "The structure is not fully emptied, but support is now carried more by a distributed calendar effect than by a single one-day expiry wall."),
        ("Ust panel pasif destek skorunun son gunlerde guclenip zayifladigini, alt panel ise onumuzdeki aylarda destegin hangi vade pencerelerinde yogunlastigini gosterir.", "The top panel shows whether passive support has strengthened or weakened recently; the bottom panel shows which future expiry windows carry that support across the coming months."),
        ("Low compression / daha serbest fiyat kesfi", "Low compression / freer price discovery"),
        ("Spot 66,500-72,000 bandinda kaldikca compression proxy daha anlamli calisir.", "The compression proxy matters more while spot remains inside the 66,500-72,000 band."),
        ("70,000 ustu acceptance gelirse pin baskisi weaklar, proxy'nin yonlendiriciligi duser.", "Acceptance above 70,000 weakens pin pressure and reduces the proxy's directional value."),
        ("Son 5 gunluk yerel kayitta akis en cok 100,000 and 83,000 strike'larinda birikmis.", "Over the last 5 local sessions, flow has accumulated most heavily at the 100,000 and 83,000 strikes."),
        ("20 gunluk birikimde call tarafi daha baskin.", "The 20-day accumulation is more call-dominant."),
        ("20 gunluk birikimde put tarafi daha baskin.", "The 20-day accumulation is more put-dominant."),
        ("Call OI, yukari yonlu acik riskin hangi vadede yigildigini gosterir.", "Call OI shows which expiry is carrying the open upside risk."),
        ("Put OI, asagi koruma veya spekulatif downside talebinin agirligini gosterir.", "Put OI shows the weight of downside protection or speculative downside demand."),
        ("Premium tarafi, hangi taraf icin daha cok para odendigini gosterir; bu genelde daha ilgili tarafi ayirmada daha faydalidir.", "The premium split shows where more money was paid; this is often the cleaner way to identify the side that mattered most."),
        ("Yarin kapanacak vade buyuk ve put-premium agir ise expiry sonrasi temizlenecek hedge etkisi daha onemli olabilir.", "If the expiry that closes tomorrow is large and put-premium heavy, the post-expiry hedge unwind effect can matter more."),
        (" puan uzerinde. Vol market spotun son gercek hareketine gore primli calisiyor.", " points above. The vol market is trading rich relative to spot's recent realized move."),
        ("Kontrollu ama kirilgan. Genel gamma yapisi tampon uretmeye devam ediyor, ancak on vadedeki zayiflama nedeniyle denge bozulursa hareket hizlanabilir.", "Controlled but fragile. The broader gamma structure is still providing a buffer, but weakness in the front expiry means moves can accelerate if balance breaks."),
        ("Kontrollu ama kirilgan. Genel gamma yapisi tampon uretmeye devam ediyor, ancak on vadedeki weaklama nedeniyle denge bozulursa hareket hizlanabilir.", "Controlled but fragile. The broader gamma structure is still providing a buffer, but weakness in the front expiry means moves can accelerate if balance breaks."),
        ("Kontrollu. Genel gamma yapisi dengeleyici kalmaya devam ediyor ve fiyat davranisi simdilik daha cok pin/range karakteri tasiyor.", "Controlled. The broader gamma structure remains stabilizing, so price action still carries more of a pin/range character for now."),
        ("Kirilgan. Gamma yapisi artik hareketi bastirmaktan cok buyutmeye daha yakin, bu yuzden fiyat davranisi daha yonlu olabilir.", "Fragile. The gamma structure is now closer to amplifying moves than suppressing them, so price action can become more directional."),
        ("80,000 ustunde kalici acceptance ve belirgin yeni call eklenmesi.", "Sustained acceptance above 80,000 with visible fresh call adding."),
        ("69,000 alti kabul. Bu bolgenin altinda hareket emilen degil buyuyen tarafa donebilir.", "Acceptance below 69,000. Under that zone, moves can shift from being absorbed to being amplified."),
        ("Asagi yon daha kolay hizlanabilir. Yukari devam icin ", "Downside can accelerate more easily. Upside continuation requires "),
        (" ustu acceptance gerekirken, ", " above for clean acceptance, while a break below "),
        (" alti kirilma hedge akislarini daha hizli buyutebilir.", " can expand hedge flows more quickly."),
        (" ustunde kalici acceptance ve belirgin yeni call eklenmesi.", " sustained acceptance above with visible fresh call adding."),
        (" ustunde kalici acceptance ve belirgin yeni call eklenmesi.", " sustained above with visible fresh call adding."),
        (" alti kabul. Bu bolgenin altinda hareket emilen degil buyuyen tarafa donebilir.", " acceptance below. Under that zone, moves can shift from being absorbed to being amplified."),
        (" alti kabul. Bu bolgenin altinda hareket emilen degil buyuyen tarafa donebilir.", " acceptance below. Under that zone, moves can shift from being absorbed to being amplified."),
        ("Opsiyon yapisi tamamen bearish degil, ama mevcut dagilim kisa vadede asagi kirilmanin yukari teyitten daha kolay calisabilecegini soyluyor.", "The options structure is not outright bearish, but the current distribution suggests downside breaks can work more easily than upside confirmation in the near term."),
        ("Opsiyon yapisi yapici kalmaya devam ediyor, ancak temiz yukari devam icin acceptance olmadan momentumun kalici oldugu soylenemez.", "The options structure remains constructive, but without acceptance it is still hard to call upside momentum durable."),
        ("Opsiyon yapisi net trend gununden cok kontrollu fiyatlama anlatiyor; kirilma olursa asimetrinin hangi yone daha kolay calistigi asil belirleyici olacak.", "The options structure points more to controlled pricing than a clear trend day; if a break comes, the key question will be which side the asymmetry favors."),
        ("Tersine donus teyidi icin spotun en az 70,500 ustune geri yerlesmesi and flow tonunun savunmadan karisiga donmesi gerekir.", "For reversal confirmation, spot needs to reclaim at least 70,500 and flow tone needs to shift from defensive toward mixed."),
        ("Negative tetik: 70,500 alti yeni kabul, expiry sonrasi bile hedge baskisini hizlandirabilir.", "Negative trigger: fresh acceptance below 70,500 can accelerate hedge pressure even after expiry."),
        ("BULL_PUT_SPREAD OI artisi 561; yonlu iyimserlik var ama sinirli riskli / defined-risk yapilar tercih ediliyor.", "BULL_PUT_SPREAD OI increased by 561; there is directional optimism, but limited-risk / defined-risk structures are preferred."),
        ("The center of net strike flow moved from 85,000 to 71,000'a asagiya kayiyor; bu da yeni ilginin hangi strike'a toplandigini gosteriyor.", "The center of net strike flow moved lower from 85,000 to 71,000, showing where new interest is concentrating."),
        ("Yarin kapanacak vade buyuk and put-premium agir ise expiry sonrasi temizlenecek hedge etkisi daha onemli olabilir.", "If the expiry that closes tomorrow is large and put-premium heavy, the post-expiry hedge unwind effect can matter more."),
        ("Tersine donus teyidi icin spotun en az 70,500 ustune geri yerlesmesi and flow tonunun savunmadan karisiga donmesi gerekir.", "For reversal confirmation, spot needs to reclaim at least 70,500 and flow tone needs to shift from defensive toward mixed."),
        ("Negative tetik: 70,500 alti yeni kabul, expiry sonrasi bile hedge baskisini hizlandirabilir.", "Negative trigger: fresh acceptance below 70,500 can accelerate hedge pressure even after expiry."),
        ("Gamma Haritasi", "Gamma Map"),
        ("Kritik Gamma Dugumleri", "Key Gamma Nodes"),
        ("Gamma Rejim Okumasi", "Gamma Regime Read"),
        ("Spotun Yeri:", "Spot Location:"),
        ("Bu Bolgede Davranis:", "Behavior In This Zone:"),
        ("Pozitifte Davranis:", "Behavior In Positive Gamma:"),
        ("Gecis Bolgesi:", "Transition Zone:"),
        ("Negatifte Davranis:", "Behavior In Negative Gamma:"),
        ("Positivete Davranis:", "Behavior In Positive Gamma:"),
        ("Negativete Davranis:", "Behavior In Negative Gamma:"),
        ("Pozitif", "Positive"),
        ("Negatif", "Negative"),
        ("Taze Konumlanma", "Fresh Positioning"),
        ("Volatilite Baglami", "Volatility Context"),
        ("Profesyonel Son Okuma", "Professional Final Read"),
        ("En Guclu Instrument OI Eklemeleri", "Top Instrument OI Adds"),
        ("En Guclu Instrument OI Bosalmalari", "Top Instrument OI Cuts"),
        ("Strike Heatmap Benzeri Eklemeler", "Strike Heatmap-style Adds"),
        ("Strike Heatmap Benzeri Bosalmalar", "Strike Heatmap-style Cuts"),
        ("Vade", "Expiry"),
        ("OI Degisimi", "OI Change"),
        ("Toplam Notional", "Total Notional"),
        ("Acik Pozisyon", "Open Interest"),
        ("Call Degisimi", "Call Change"),
        ("Put Degisimi", "Put Change"),
        ("24s OI Degisimi", "24h OI Change"),
        ("BTC Degisimi", "BTC Change"),
        ("Referans vade:", "Reference expiry:"),
    ]
    for src, dst in forced_replacements:
        html = html.replace(src, dst)
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a trend-aware morning options report from Laevitas.")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--market", default="deribit")
    args = parser.parse_args()

    ensure_dirs()
    fallback_notes = {}

    gex_name = f"{args.currency.lower()}_gex_all_{args.market}"
    expiry_rows, expiry_market, expiry_calls = load_expiry_rows(args.currency, args.market)
    if not expiry_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(f"{args.currency.lower()}_oi_expiry_aggregate")
        if fallback_payload:
            expiry_rows = parse_expiry_rows(fallback_payload, "aggregate")
            expiry_market = "aggregate"
            fallback_notes["expiry_oi"] = (
                f"Bugun canli expiry OI verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu aggregate snapshot kullanildi."
            )
        if not expiry_rows:
            fallback_payload, fallback_path = load_latest_nonempty_snapshot(f"{args.currency.lower()}_oi_expiry_{args.market}")
            if fallback_payload:
                expiry_rows = parse_expiry_rows(fallback_payload, args.market)
                expiry_market = args.market
                fallback_notes["expiry_oi"] = (
                    f"Bugun canli expiry OI verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu {args.market} snapshot kullanildi."
                )

    gex_payload = fetch_gex_all(args.market, args.currency)
    save_snapshot(gex_name, gex_payload)
    gex_rows = aggregate_gex_by_strike(gex_payload)
    if not gex_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(gex_name)
        if fallback_payload:
            gex_rows = aggregate_gex_by_strike(fallback_payload)
            fallback_notes['gex'] = f"Bugun canli GEX verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."

    gex_term_name = f"{args.currency.lower()}_gex_term_structure"
    gex_term_payload = {}
    gex_term_rows: List[Dict] = []
    gex_term_meta: Dict[str, float] = {}
    gex_term_snapshot_date = UTC_NOW().date().isoformat()
    try:
        gex_term_payload = fetch_gex_term_structure(args.currency)
        save_snapshot(gex_term_name, gex_term_payload)
        gex_term_rows = parse_gex_term_structure_rows(gex_term_payload)
        gex_term_meta = parse_gex_term_structure_meta(gex_term_payload)
    except Exception as exc:
        fallback_notes["gex_term_fetch"] = f"Expiry bazli GEX term structure endpoint'i hata verdi: {type(exc).__name__}."
    if not gex_term_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(gex_term_name)
        if fallback_payload:
            gex_term_rows = parse_gex_term_structure_rows(fallback_payload)
            gex_term_meta = parse_gex_term_structure_meta(fallback_payload)
            gex_term_snapshot_date = extract_snapshot_date(fallback_path)
            fallback_notes["gex_term"] = (
                f"Bugun canli expiry-bazli GEX verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."
            )
    previous_gex_term_payload, previous_gex_term_path = load_previous_nonempty_snapshot(gex_term_name, gex_term_snapshot_date)
    previous_gex_term_rows = parse_gex_term_structure_rows(previous_gex_term_payload) if previous_gex_term_payload else []
    previous_gex_term_meta = parse_gex_term_structure_meta(previous_gex_term_payload) if previous_gex_term_payload else {}
    previous_gex_term_date = extract_snapshot_date(previous_gex_term_path) if previous_gex_term_path else None

    strike_name = f"{args.currency.lower()}_oi_strike_all_{args.market}"
    strike_payload = fetch_oi_strike_all(args.market, args.currency)
    save_snapshot(strike_name, strike_payload)
    strike_rows = parse_strike_oi_rows(strike_payload)
    if not strike_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(strike_name)
        if fallback_payload:
            strike_rows = parse_strike_oi_rows(fallback_payload)
            fallback_notes['strike_oi'] = f"Bugun canli strike OI verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."

    type_name = f"{args.currency.lower()}_oi_type_{args.market}"
    type_payload = fetch_oi_type(args.market, args.currency)
    save_snapshot(type_name, type_payload)
    type_rows = parse_type_rows(type_payload)
    if not type_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(type_name)
        if fallback_payload:
            type_rows = parse_type_rows(fallback_payload)
            fallback_notes['oi_type'] = f"Bugun canli call/put split verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."

    change_name = f"{args.currency.lower()}_oi_change_summary_24h"
    change_payload = fetch_oi_change_summary(args.currency, 24)
    save_snapshot(change_name, change_payload)
    change_rows = parse_change_rows(change_payload)
    if not change_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(change_name)
        if fallback_payload:
            change_rows = parse_change_rows(fallback_payload)
            fallback_notes['oi_change'] = f"Bugun canli OI degisim ozeti bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."

    vol_name = f"{args.currency.lower()}_vol_context_{VOL_CONTEXT_EXPIRY.lower()}"
    vol_payload = fetch_vol_context(args.currency, VOL_CONTEXT_EXPIRY)
    save_snapshot(vol_name, vol_payload)
    vol_context = parse_vol_context(vol_payload)
    if not vol_context:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(vol_name)
        if fallback_payload:
            vol_context = parse_vol_context(fallback_payload)
            fallback_notes['vol_context'] = f"Bugun canli vol context verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."

    atm_iv_ts_name = f"{args.currency.lower()}_atm_iv_ts_{args.market}"
    atm_iv_ts_payload = fetch_atm_iv_ts(args.market, args.currency)
    save_snapshot(atm_iv_ts_name, atm_iv_ts_payload)
    atm_iv_ts = parse_atm_iv_ts(atm_iv_ts_payload)
    if not atm_iv_ts:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(atm_iv_ts_name)
        if fallback_payload:
            atm_iv_ts = parse_atm_iv_ts(fallback_payload)
            fallback_notes['atm_iv_ts'] = (
                f"Bugun canli ATM IV zaman serisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."
            )
    iv_table_name = f"{args.currency.lower()}_iv_table_{args.market}"
    iv_table_payload = fetch_iv_table(args.market, args.currency)
    save_snapshot(iv_table_name, iv_table_payload)
    iv_table_rows = parse_iv_table_rows(iv_table_payload)
    if not iv_table_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(iv_table_name)
        if fallback_payload:
            iv_table_rows = parse_iv_table_rows(fallback_payload)
            fallback_notes['iv_table'] = f"Bugun canli IV table verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."

    strategy_name = f"{args.currency.lower()}_top_strategies_24h"
    strategy_payload = None
    strategy_rows = []
    try:
        strategy_payload = fetch_top_strategies(args.currency, 24)
        save_snapshot(strategy_name, strategy_payload if isinstance(strategy_payload, dict) else {"data": strategy_payload})
        strategy_rows = parse_top_strategy_rows(strategy_payload)
    except Exception as exc:
        fallback_notes['strategy_fetch'] = f"Canli strategy tape endpoint'i hata verdi: {type(exc).__name__}."
    if not strategy_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(strategy_name)
        if fallback_payload:
            strategy_rows = parse_top_strategy_rows(fallback_payload)
            fallback_notes['strategy'] = f"Bugun canli strategy tape verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."
    top_instrument_name = f"{args.currency.lower()}_top_instrument_oi_change_{args.market}_24h"
    top_instrument_payload, top_instrument_rows, top_instrument_retry_count = retry_empty_payload(
        lambda: fetch_top_instrument_oi_change(args.market, args.currency, 24),
        parse_top_instrument_rows,
        retries=1,
        sleep_seconds=2.0,
    )
    save_snapshot(top_instrument_name, top_instrument_payload)
    if not top_instrument_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(top_instrument_name)
        if fallback_payload:
            top_instrument_rows = parse_top_instrument_rows(fallback_payload)
            fallback_notes['top_instrument'] = (
                f"Bugun canli instrument flow verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."
            )
    elif top_instrument_retry_count:
        fallback_notes['top_instrument_retry'] = (
            f"Instrument flow ilk denemede bos geldi; {top_instrument_retry_count} ek deneme sonrasi canli veri alindi."
        )

    strike_change_name = f"{args.currency.lower()}_oi_net_change_all_{args.market}_24h"
    strike_change_payload, strike_change_rows, strike_change_retry_count = retry_empty_payload(
        lambda: fetch_oi_net_change_all(args.market, args.currency, 24),
        parse_oi_net_change_all_rows,
        retries=1,
        sleep_seconds=2.0,
    )
    save_snapshot(strike_change_name, strike_change_payload)
    if not strike_change_rows:
        fallback_payload, fallback_path = load_latest_nonempty_snapshot(strike_change_name)
        if fallback_payload:
            strike_change_rows = parse_oi_net_change_all_rows(fallback_payload)
            fallback_notes['strike_change'] = (
                f"Bugun canli strike flow verisi bos geldi; {extract_snapshot_date(fallback_path)} tarihli son dolu snapshot kullanildi."
            )
    elif strike_change_retry_count:
        fallback_notes['strike_change_retry'] = (
            f"Strike flow ilk denemede bos geldi; {strike_change_retry_count} ek deneme sonrasi canli veri alindi."
        )
    price_series = fetch_public_btc_daily_series()
    realized_vols = compute_realized_vols([float(row["close"]) for row in price_series], [7, 14, 30])
    current_snapshot_date = UTC_NOW().date().isoformat()

    previous_strike_payload, previous_strike_path = load_previous_nonempty_snapshot(strike_name, current_snapshot_date)
    previous_strike_rows = parse_strike_oi_rows(previous_strike_payload) if previous_strike_payload else []
    previous_snapshot_date = extract_snapshot_date(previous_strike_path) if previous_strike_path else None
    previous_gex_rows: List[Dict] = []
    previous_strike_change_rows: List[Dict] = []
    if previous_snapshot_date:
        previous_gex_payload, _ = load_snapshot_for_date(gex_name, previous_snapshot_date)
        if previous_gex_payload:
            previous_gex_rows = aggregate_gex_by_strike(previous_gex_payload)
        previous_strike_change_payload, _ = load_snapshot_for_date(strike_change_name, previous_snapshot_date)
        if previous_strike_change_payload:
            previous_strike_change_rows = parse_oi_net_change_all_rows(previous_strike_change_payload)

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
    summary["clean_position_map_notes"] = build_clean_position_map_notes(summary, summary["clean_position_map_levels"])
    summary['fallback_notes'] = fallback_notes
    previous_chain_rows = build_chain_pressure_rows(previous_strike_rows, previous_gex_rows, previous_strike_change_rows, summary["spot"])
    dealer_pressure_section = build_dealer_pressure_section(
        summary.get("chain_rows", []),
        previous_chain_rows=previous_chain_rows,
        previous_date=previous_snapshot_date,
        current_date=current_snapshot_date,
    )
    proxy_hedge_impact_section = build_proxy_hedge_impact_section(summary)
    today = UTC_NOW().strftime("%Y-%m-%d")
    archive_rows = upsert_archive(build_archive_row(summary, proxy_hedge_impact_section))
    archive_rows = backfill_archive_metrics(archive_rows, args.currency, args.market)
    proxy_hedge_impact_section = enrich_proxy_hedge_impact_history(proxy_hedge_impact_section, archive_rows)
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
    clean_position_map_section = build_clean_position_map_section(summary, archive_rows)
    position_map_drift_section = build_position_map_drift_section(summary, archive_rows)
    gamma_support_monitor_section = build_gamma_support_monitor_section(summary, archive_rows, expiry_rows)
    gex_term_structure_section = build_gex_term_structure_section(
        summary,
        gex_term_rows,
        gex_term_meta,
        previous_term_rows=previous_gex_term_rows,
        previous_term_meta=previous_gex_term_meta,
        previous_snapshot_date=previous_gex_term_date,
    )
    layered_trade_check = build_layered_trade_check(summary, archive_rows, price_series)
    layered_trade_check_en = build_layered_trade_check_en(layered_trade_check)
    vanna_charm_section = build_vanna_charm_proxy_section(summary, expiry_rows)
    iv_regime_section = build_iv_regime_section(summary, archive_rows)
    pain_section = build_pain_section(summary, archive_rows)
    flow_horizon_notes = build_flow_horizon_notes(today, strike_change_rows, top_instrument_rows)

    expiry_chart = CHART_DIR / f"{args.currency.lower()}_expiry_oi.png"
    gex_chart = CHART_DIR / f"{args.currency.lower()}_gex.png"
    strike_chart = CHART_DIR / f"{args.currency.lower()}_strike_oi.png"
    archive_chart = CHART_DIR / f"{args.currency.lower()}_archive_trend.png"
    flow_chart = CHART_DIR / f"{args.currency.lower()}_flow_change.png"
    heatmap_chart = CHART_DIR / f"{args.currency.lower()}_flow_heatmap.png"
    clean_position_map_chart = CHART_DIR / f"{args.currency.lower()}_options_position_map_clean.png"
    position_map_drift_chart = CHART_DIR / f"{args.currency.lower()}_position_map_drift.png"
    gamma_support_monitor_chart = CHART_DIR / f"{args.currency.lower()}_gamma_support_monitor.png"
    gex_term_structure_chart = CHART_DIR / f"{args.currency.lower()}_gex_term_structure.png"
    gex_term_structure_delta_chart = CHART_DIR / f"{args.currency.lower()}_gex_term_structure_delta.png"
    iv_term_chart = CHART_DIR / f"{args.currency.lower()}_iv_term_structure.png"
    iv_rv_chart = CHART_DIR / f"{args.currency.lower()}_iv_rv.png"
    strategy_chart = CHART_DIR / f"{args.currency.lower()}_top_strategies.png"
    inventory_chart = CHART_DIR / f"{args.currency.lower()}_inventory_proxy.png"
    mm_position_map_chart = CHART_DIR / f"{args.currency.lower()}_mm_position_map.png"
    dealer_pressure_delta_chart = CHART_DIR / f"{args.currency.lower()}_dealer_pressure_delta.png"
    proxy_hedge_impact_chart = CHART_DIR / f"{args.currency.lower()}_proxy_hedge_impact.png"
    proxy_hedge_impact_trend_chart = CHART_DIR / f"{args.currency.lower()}_proxy_hedge_impact_trend.png"
    inventory_trend_chart = CHART_DIR / f"{args.currency.lower()}_inventory_proxy_trend.png"
    behavioral_trend_chart = CHART_DIR / f"{args.currency.lower()}_behavioral_trend.png"
    behavioral_components_chart = CHART_DIR / f"{args.currency.lower()}_behavioral_components.png"
    layered_trade_check_chart = CHART_DIR / f"{args.currency.lower()}_layered_trade_check.png"
    vanna_charm_chart = CHART_DIR / f"{args.currency.lower()}_vanna_charm_proxy.png"
    strategy_action_chart = CHART_DIR / f"{args.currency.lower()}_strategy_action_map.png"
    roll_transfer_chart = CHART_DIR / f"{args.currency.lower()}_roll_transfer_map.png"
    normalized_skew_chart = CHART_DIR / f"{args.currency.lower()}_normalized_skew.png"
    iv_regime_chart = CHART_DIR / f"{args.currency.lower()}_iv_regime.png"
    pain_chart = CHART_DIR / f"{args.currency.lower()}_pain_drift.png"
    expiry_chart_en = CHART_DIR / f"{args.currency.lower()}_expiry_oi_en.png"
    gex_chart_en = CHART_DIR / f"{args.currency.lower()}_gex_en.png"
    strike_chart_en = CHART_DIR / f"{args.currency.lower()}_strike_oi_en.png"
    archive_chart_en = CHART_DIR / f"{args.currency.lower()}_archive_trend_en.png"
    flow_chart_en = CHART_DIR / f"{args.currency.lower()}_flow_change_en.png"
    heatmap_chart_en = CHART_DIR / f"{args.currency.lower()}_flow_heatmap_en.png"
    clean_position_map_chart_en = CHART_DIR / f"{args.currency.lower()}_options_position_map_clean_en.png"
    position_map_drift_chart_en = CHART_DIR / f"{args.currency.lower()}_position_map_drift_en.png"
    gamma_support_monitor_chart_en = CHART_DIR / f"{args.currency.lower()}_gamma_support_monitor_en.png"
    gex_term_structure_chart_en = CHART_DIR / f"{args.currency.lower()}_gex_term_structure_en.png"
    gex_term_structure_delta_chart_en = CHART_DIR / f"{args.currency.lower()}_gex_term_structure_delta_en.png"
    iv_term_chart_en = CHART_DIR / f"{args.currency.lower()}_iv_term_structure_en.png"
    iv_rv_chart_en = CHART_DIR / f"{args.currency.lower()}_iv_rv_en.png"
    strategy_chart_en = CHART_DIR / f"{args.currency.lower()}_top_strategies_en.png"
    inventory_chart_en = CHART_DIR / f"{args.currency.lower()}_inventory_proxy_en.png"
    mm_position_map_chart_en = CHART_DIR / f"{args.currency.lower()}_mm_position_map_en.png"
    dealer_pressure_delta_chart_en = CHART_DIR / f"{args.currency.lower()}_dealer_pressure_delta_en.png"
    proxy_hedge_impact_chart_en = CHART_DIR / f"{args.currency.lower()}_proxy_hedge_impact_en.png"
    proxy_hedge_impact_trend_chart_en = CHART_DIR / f"{args.currency.lower()}_proxy_hedge_impact_trend_en.png"
    inventory_trend_chart_en = CHART_DIR / f"{args.currency.lower()}_inventory_proxy_trend_en.png"
    behavioral_trend_chart_en = CHART_DIR / f"{args.currency.lower()}_behavioral_trend_en.png"
    behavioral_components_chart_en = CHART_DIR / f"{args.currency.lower()}_behavioral_components_en.png"
    layered_trade_check_chart_en = CHART_DIR / f"{args.currency.lower()}_layered_trade_check_en.png"
    vanna_charm_chart_en = CHART_DIR / f"{args.currency.lower()}_vanna_charm_proxy_en.png"
    strategy_action_chart_en = CHART_DIR / f"{args.currency.lower()}_strategy_action_map_en.png"
    roll_transfer_chart_en = CHART_DIR / f"{args.currency.lower()}_roll_transfer_map_en.png"
    normalized_skew_chart_en = CHART_DIR / f"{args.currency.lower()}_normalized_skew_en.png"
    iv_regime_chart_en = CHART_DIR / f"{args.currency.lower()}_iv_regime_en.png"
    pain_chart_en = CHART_DIR / f"{args.currency.lower()}_pain_drift_en.png"
    make_expiry_chart(expiry_rows, expiry_chart)
    make_expiry_chart(expiry_rows, expiry_chart_en, lang="en")
    make_gex_chart(gex_rows, summary["spot"], gex_chart)
    make_gex_chart(gex_rows, summary["spot"], gex_chart_en, lang="en")
    make_strike_oi_chart(strike_rows, summary["spot"], strike_chart)
    make_strike_oi_chart(strike_rows, summary["spot"], strike_chart_en, lang="en")
    make_archive_chart(archive_rows, archive_chart)
    make_archive_chart(archive_rows, archive_chart_en, lang="en")
    make_flow_change_chart(summary["strike_adds"], summary["strike_cuts"], flow_chart)
    make_flow_change_chart(summary["strike_adds"], summary["strike_cuts"], flow_chart_en, lang="en")
    make_flow_heatmap(strike_change_rows, summary["spot"], heatmap_chart)
    make_flow_heatmap(strike_change_rows, summary["spot"], heatmap_chart_en, lang="en")
    make_clean_options_position_map_chart(
        price_series,
        clean_position_map_chart,
        summary["clean_position_map_levels"],
        summary["clean_position_map_profile"],
    )
    make_clean_options_position_map_chart(
        price_series,
        clean_position_map_chart_en,
        summary["clean_position_map_levels"],
        summary["clean_position_map_profile"],
        lang="en",
    )
    make_position_map_drift_chart(archive_rows, position_map_drift_chart)
    make_position_map_drift_chart(archive_rows, position_map_drift_chart_en, lang="en")
    make_gamma_support_monitor_chart(gamma_support_monitor_section, gamma_support_monitor_chart)
    make_gamma_support_monitor_chart(gamma_support_monitor_section, gamma_support_monitor_chart_en, lang="en")
    make_gex_term_structure_chart(gex_term_structure_section, gex_term_structure_chart)
    make_gex_term_structure_chart(gex_term_structure_section, gex_term_structure_chart_en, lang="en")
    make_gex_term_structure_delta_chart(gex_term_structure_section, gex_term_structure_delta_chart)
    make_gex_term_structure_delta_chart(gex_term_structure_section, gex_term_structure_delta_chart_en, lang="en")
    make_iv_term_chart(atm_iv_ts, iv_term_chart)
    make_iv_term_chart(atm_iv_ts, iv_term_chart_en, lang="en")
    make_iv_rv_chart(summary, iv_rv_chart)
    make_iv_rv_chart(summary, iv_rv_chart_en, lang="en")
    make_strategy_chart(summary["strategy_rows"], strategy_chart)
    make_strategy_chart(summary["strategy_rows"], strategy_chart_en, lang="en")
    make_mm_inventory_chart(summary, inventory_chart)
    make_mm_inventory_chart(summary, inventory_chart_en, lang="en")
    make_mm_position_map_chart(summary, mm_position_map_chart)
    make_mm_position_map_chart(summary, mm_position_map_chart_en, lang="en")
    make_dealer_pressure_delta_chart(dealer_pressure_section, dealer_pressure_delta_chart)
    make_dealer_pressure_delta_chart(dealer_pressure_section, dealer_pressure_delta_chart_en, lang="en")
    make_proxy_hedge_impact_chart(proxy_hedge_impact_section, proxy_hedge_impact_chart)
    make_proxy_hedge_impact_chart(proxy_hedge_impact_section, proxy_hedge_impact_chart_en, lang="en")
    make_proxy_hedge_impact_trend_chart(proxy_hedge_impact_section, proxy_hedge_impact_trend_chart)
    make_proxy_hedge_impact_trend_chart(proxy_hedge_impact_section, proxy_hedge_impact_trend_chart_en, lang="en")
    make_inventory_trend_chart(archive_rows, inventory_trend_chart)
    make_inventory_trend_chart(archive_rows, inventory_trend_chart_en, lang="en")
    make_behavioral_trend_chart(archive_rows, behavioral_trend_chart)
    make_behavioral_trend_chart(archive_rows, behavioral_trend_chart_en, lang="en")
    make_behavioral_components_chart(summary, behavioral_components_chart)
    make_behavioral_components_chart(summary, behavioral_components_chart_en, lang="en")
    make_layered_trade_check_chart(layered_trade_check, layered_trade_check_chart)
    make_layered_trade_check_chart(layered_trade_check_en, layered_trade_check_chart_en, lang="en")
    make_vanna_charm_proxy_chart(vanna_charm_section, vanna_charm_chart)
    make_vanna_charm_proxy_chart(vanna_charm_section, vanna_charm_chart_en, lang="en")
    make_strategy_action_chart(summary, strategy_action_chart)
    make_strategy_action_chart(summary, strategy_action_chart_en, lang="en")
    make_roll_transfer_chart(summary, roll_transfer_chart)
    make_roll_transfer_chart(summary, roll_transfer_chart_en, lang="en")
    make_normalized_skew_chart(archive_rows, normalized_skew_chart)
    make_normalized_skew_chart(archive_rows, normalized_skew_chart_en, lang="en")
    make_iv_regime_chart(archive_rows, iv_regime_chart)
    make_iv_regime_chart(archive_rows, iv_regime_chart_en, lang="en")
    make_pain_drift_chart(archive_rows, pain_chart)
    make_pain_drift_chart(archive_rows, pain_chart_en, lang="en")

    api_calls = expiry_calls + 11
    public_calls = 1
    html = render_html(
        args.currency,
        summary,
        change_section,
        dealer_pressure_section,
        proxy_hedge_impact_section,
        clean_position_map_section,
        position_map_drift_section,
        gamma_support_monitor_section,
        gex_term_structure_section,
        layered_trade_check,
        vanna_charm_section,
        iv_regime_section,
        flow_horizon_notes,
        expiry_rows,
        expiry_chart,
        gex_chart,
        strike_chart,
        archive_chart,
        flow_chart,
        heatmap_chart,
        clean_position_map_chart,
        position_map_drift_chart,
        gamma_support_monitor_chart,
        gex_term_structure_chart,
        gex_term_structure_delta_chart,
        iv_term_chart,
        iv_rv_chart,
        strategy_chart,
        inventory_chart,
        mm_position_map_chart,
        dealer_pressure_delta_chart,
        proxy_hedge_impact_chart,
        proxy_hedge_impact_trend_chart,
        inventory_trend_chart,
        behavioral_trend_chart,
        behavioral_components_chart,
        layered_trade_check_chart,
        vanna_charm_chart,
        strategy_action_chart,
        roll_transfer_chart,
        normalized_skew_chart,
        iv_regime_chart,
        pain_chart,
        pain_section,
        api_calls,
        public_calls,
    )
    html = html.replace("desteÄŸin", "destegin")
    html = html.replace("desteÄŸi", "destegi")
    html = html.replace("MonitÃ¶rÃ¼", "Monitoru")
    html = html.replace("gÃ¶rÃ¼nmÃ¼yor", "gorunmuyor")
    html = html.replace("taÅŸÄ±yan", "tasiyan")
    html = html.replace("anlamlÄ±", "anlamli")
    out_html = REPORT_DIR / f"{args.currency.lower()}_morning_report.html"
    out_html.write_text(html, encoding="utf-8")
    html_en = render_html_en(
        args.currency,
        summary,
        change_section,
        dealer_pressure_section,
        proxy_hedge_impact_section,
        clean_position_map_section,
        position_map_drift_section,
        gamma_support_monitor_section,
        gex_term_structure_section,
        layered_trade_check,
        vanna_charm_section,
        iv_regime_section,
        flow_horizon_notes,
        expiry_rows,
        expiry_chart_en,
        gex_chart_en,
        strike_chart_en,
        archive_chart_en,
        flow_chart_en,
        heatmap_chart_en,
        clean_position_map_chart_en,
        position_map_drift_chart_en,
        gamma_support_monitor_chart_en,
        gex_term_structure_chart_en,
        gex_term_structure_delta_chart_en,
        iv_term_chart_en,
        iv_rv_chart_en,
        strategy_chart_en,
        inventory_chart_en,
        mm_position_map_chart_en,
        dealer_pressure_delta_chart_en,
        proxy_hedge_impact_chart_en,
        proxy_hedge_impact_trend_chart_en,
        inventory_trend_chart_en,
        behavioral_trend_chart_en,
        behavioral_components_chart_en,
        layered_trade_check_chart_en,
        vanna_charm_chart_en,
        strategy_action_chart_en,
        roll_transfer_chart_en,
        normalized_skew_chart_en,
        iv_regime_chart_en,
        pain_chart_en,
        pain_section,
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
    print(f"clean_position_map_chart={clean_position_map_chart}")
    print(f"iv_term_chart={iv_term_chart}")
    print(f"iv_rv_chart={iv_rv_chart}")
    print(f"strategy_chart={strategy_chart}")
    print(f"inventory_chart={inventory_chart}")
    print(f"mm_position_map_chart={mm_position_map_chart}")
    print(f"inventory_trend_chart={inventory_trend_chart}")
    print(f"behavioral_trend_chart={behavioral_trend_chart}")
    print(f"behavioral_components_chart={behavioral_components_chart}")
    print(f"normalized_skew_chart={normalized_skew_chart}")
    print(f"iv_regime_chart={iv_regime_chart}")
    print(f"pain_chart={pain_chart}")
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
