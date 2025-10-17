# pool_exact_signal_dashboard.py
# Exact Signal Dashboard — 15m Pools (v2 + Dexscreener + Fast Scan + Auto-Relax)
# - Long Now / Short Now with Breakout, VWAP Bounce, EMA Pullback, Momentum
# - Modes: Conservative / Balanced / Aggressive / Panic (+ Auto-escalate by time left)
# - Dexscreener Pulse overlay to surface signals sooner (optional)
# - Fast Scan, Auto-relax (no-signal timer), Last-minute ultra-loose override
# - Safe symbols fallback + quick connectivity test
# - Manual execution only (no bots)

import time, requests, urllib.parse
import pandas as pd
import numpy as np
import streamlit as st

BINANCE_HTTP = "https://api.binance.com"
FAPI_HTTP    = "https://fapi.binance.com"

# Your requested symbols included
DEFAULT_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT",
    "AAVEUSDT","LDOUSDT","INJUSDT","ATOMUSDT","HBARUSDT","XLMUSDT",
    "FETUSDT","UNIUSDT","LINKUSDT","DOTUSDT","AVAXUSDT","TRXUSDT"
]

# ---------------- Data helpers ----------------
def _fetch_klines(symbol: str, interval="1m", limit=200):
    url = f"{BINANCE_HTTP}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=10); r.raise_for_status()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","open","high","low","close","volume"]]

@st.cache_data(ttl=5, show_spinner=False)
def fetch_klines_1m(symbol: str, limit=240) -> pd.DataFrame:
    return _fetch_klines(symbol, "1m", limit)

@st.cache_data(ttl=25, show_spinner=False)
def fetch_klines_5m(symbol: str, limit=300) -> pd.DataFrame:
    return _fetch_klines(symbol, "5m", limit)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_klines_15m(symbol: str, limit=200) -> pd.DataFrame:
    return _fetch_klines(symbol, "15m", limit)

def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def rma(s: pd.Series, period: int):
    return s.ewm(alpha=1/period, adjust=False).mean()

def last_price(df: pd.DataFrame) -> float:
    return float(df["close"].iloc[-1]) if not df.empty else float("nan")

def atr_pct(df: pd.DataFrame, period=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = rma(tr, period)
    return 100.0 * (atr / c.replace(0, np.nan))

def adx(df: pd.DataFrame, period=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff(); down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, period) / atr.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return rma(dx, period)

def vwap_today(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    if ts.empty: return float("nan")
    day = ts.dt.date.iloc[-1]
    mask = (ts.dt.date == day)
    if mask.sum() == 0: return float("nan")
    tp = (df.loc[mask, "high"] + df.loc[mask, "low"] + df.loc[mask, "close"]) / 3.0
    vol = df.loc[mask, "volume"]
    denom = vol.sum()
    if denom <= 0: return float("nan")
    return float((tp * vol).sum() / denom)

def ema_cross_bars_ago(close: pd.Series, short=9, long=21) -> int:
    s = ema(close, short); l = ema(close, long)
    diff = s - l
    sign = np.sign(diff.values)
    changes = np.where(sign[1:] != sign[:-1])[0]
    if changes.size > 0:
        last_idx = changes[-1] + 1
        return int(len(close) - 1 - last_idx)
    return 999

def micro_trigger(df1: pd.DataFrame, vol_thr=1.2) -> dict:
    if len(df1) < 21:
        return {"trigger":"None","rvol1":1.0}
    h, l, c, v = df1["high"], df1["low"], df1["close"], df1["volume"]
    prev_high = h.iloc[-2]; prev_low  = l.iloc[-2]
    rvol1 = float(v.iloc[-1] / max(1e-9, v.tail(20).mean()))
    if c.iloc[-1] > prev_high and rvol1 >= vol_thr: t = "BreakUp"
    elif c.iloc[-1] < prev_low and rvol1 >= vol_thr: t = "BreakDn"
    else: t = "None"
    return {"trigger": t, "rvol1": round(min(rvol1, 5.0), 2)}

def swing_levels(df1: pd.DataFrame, lookback=5):
    if len(df1) < lookback + 2:
        return (np.nan, np.nan)
    lows  = df1["low"].iloc[-(lookback+1):-1]
    highs = df1["high"].iloc[-(lookback+1):-1]
    return (float(lows.min()), float(highs.max()))

# ---------------- Dexscreener overlay ----------------
def base_asset(binance_symbol: str) -> str:
    s = binance_symbol.upper()
    for q in ["USDT","USDC","BUSD","USD"]:
        if s.endswith(q): return s[:-len(q)]
    return s

@st.cache_data(ttl=15, show_spinner=False)
def dex_search_pairs(query: str) -> dict:
    try:
        url = f"https://api.dexscreener.com/latest/dex/search/?q={urllib.parse.quote(query)}"
        r = requests.get(url, timeout=10); r.raise_for_status()
        return r.json() or {}
    except Exception:
        return {}

def pick_best_dex_pair(pairs: list, base_sym: str, prefer_chains=None, min_liq_usd=500_000):
    if not pairs: return None
    base_sym = base_sym.upper()
    valid_base = {base_sym, f"W{base_sym}"}
    prefer_chains = set((prefer_chains or ["ethereum","solana","bsc","arbitrum","base","optimism"]))
    best = None; best_score = -1.0
    for p in pairs:
        try:
            b = (p.get("baseToken") or {}).get("symbol","").upper()
            q = (p.get("quoteToken") or {}).get("symbol","").upper()
            if b not in valid_base: continue
            if q not in {"USDT","USDC"}: continue
            liq = float(((p.get("liquidity") or {}).get("usd") or 0))
            if liq < min_liq_usd: continue
            vol_h24 = float(((p.get("volume") or {}).get("h24") or 0))
            chain = str(p.get("chainId","")).lower()
            pref_bonus = 1.0 if chain in prefer_chains else 0.0
            score = liq + 0.05*vol_h24 + 1e6*pref_bonus
            if score > best_score:
                best_score = score; best = p
        except Exception:
            continue
    return best

def dex_pulse_from_pair(p: dict) -> dict:
    if not p:
        return {"pulse": None, "vol5m": None, "buys5": None, "sells5": None, "pc5": None, "chain": None, "pair": None}
    tx = (p.get("txns") or {}).get("m5") or {}
    buys = float(tx.get("buys") or 0); sells = float(tx.get("sells") or 0)
    vol5 = float(((p.get("volume") or {}).get("m5") or 0))
    pc5  = float(((p.get("priceChange") or {}).get("m5") or 0))
    denom = max(1.0, buys + sells)
    imb = (buys - sells) / denom
    pc_term = np.tanh(pc5 / 5.0)
    w = np.tanh(np.log10(max(1.0, vol5)) / 3.0)
    pulse = float(np.clip(0.6*imb + 0.4*pc_term, -1.0, 1.0) * w)
    return {"pulse": round(pulse, 3), "vol5m": round(vol5, 0), "buys5": int(buys), "sells5": int(sells),
            "pc5": round(pc5, 2), "chain": p.get("chainId"), "pair": p.get("pairAddress")}

@st.cache_data(ttl=15, show_spinner=False)
def dex_pulse_for_symbol(symbol: str, min_liq_usd=500_000):
    try:
        base = base_asset(symbol)
        data = dex_search_pairs(base)
        pairs = data.get("pairs") or []
        best = pick_best_dex_pair(pairs, base, min_liq_usd=min_liq_usd)
        return dex_pulse_from_pair(best)
    except Exception:
        return {"pulse": None, "vol5m": None, "buys5": None, "sells5": None, "pc5": None, "chain": None, "pair": None}

# ---------------- Futures / Spot helpers ----------------
@st.cache_data(ttl=60, show_spinner=False)
def spot_24h_quote_volume(symbol: str) -> float:
    try:
        url = f"{BINANCE_HTTP}/api/v3/ticker/24hr?symbol={symbol}"
        r = requests.get(url, timeout=10); r.raise_for_status()
        return float(r.json().get("quoteVolume", 0))
    except Exception:
        return 0.0

@st.cache_data(ttl=30, show_spinner=False)
def futures_oi_delta_5m(symbol: str):
    try:
        url = f"{FAPI_HTTP}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=2"
        r = requests.get(url, timeout=10); r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2: return None
        last = float(data[-1].get("sumOpenInterest") or data[-1].get("sumOpenInterestValue") or 0)
        prev = float(data[-2].get("sumOpenInterest") or data[-2].get("sumOpenInterestValue") or 0)
        if prev <= 0: return None
        return (last/prev - 1) * 100.0
    except Exception:
        return None

@st.cache_data(ttl=30, show_spinner=False)
def futures_taker_bs_ratio_5m(symbol: str):
    try:
        url = f"{FAPI_HTTP}/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=1"
        r = requests.get(url, timeout=10); r.raise_for_status()
        arr = r.json()
        if not isinstance(arr, list) or not arr: return None
        return float(arr[-1]["buySellRatio"])
    except Exception:
        return None

# ---------------- Signals & math ----------------
def tf_signal(df: pd.DataFrame) -> dict:
    if len(df) < 25:
        return {"dir":"Neutral","roc":0.0,"vratio":1.0,"ema200_bias":0,"ema200":np.nan}
    c, v = df["close"], df["volume"]
    ema9  = ema(c, 9).iloc[-1]; ema21 = ema(c, 21).iloc[-1]
    ema200_val = ema(c, min(200, max(50, len(c)-1))).iloc[-1]
    roc = float((c.iloc[-1]/c.iloc[-2] - 1) * 100) if len(c) > 1 else 0.0
    vratio = float(v.iloc[-1] / max(1e-9, v.tail(20).mean()))
    up = ema9 > ema21
    d = "Buy" if (up and roc > 0) else ("Sell" if ((not up) and roc < 0) else "Neutral")
    ema200_bias = 1 if c.iloc[-1] > ema200_val else -1
    return {"dir": d, "roc": round(roc,2), "vratio": round(min(vratio,4.0),2),
            "ema200_bias": ema200_bias, "ema200": float(ema200_val)}

def confidence_score(dir5, dir15, adx15, vratio5, price, vwap_val, oi_delta, taker_bs) -> int:
    conf = 50
    if dir5 == "Buy": conf += 15
    elif dir5 == "Sell": conf -= 15
    if dir15 == "Buy": conf += 10
    elif dir15 == "Sell": conf -= 10
    if isinstance(adx15,(int,float)): conf += 8 if adx15 >= 20 else -8
    if vratio5 >= 1.2: conf += 5
    if isinstance(vwap_val,(int,float)) and isinstance(price,(int,float)):
        if price > vwap_val and dir5=="Buy": conf += 5
        if price < vwap_val and dir5=="Sell": conf += 5
    if isinstance(oi_delta,(int,float)):
        if oi_delta > 0 and dir5=="Buy": conf += 3
        if oi_delta < 0 and dir5=="Sell": conf += 3
    if isinstance(taker_bs,(int,float)):
        if taker_bs > 1.05 and dir5=="Buy": conf += 2
        if taker_bs < 0.95 and dir5=="Sell": conf += 2
    return int(min(100, max(0, conf)))

def bps_to_percent(bps: float) -> float:
    return bps / 100.0

def gross_move_needed_pct(balance_usdt, size_frac, target_usdt, taker_fee_bps, slippage_bps):
    size = max(1e-9, balance_usdt * size_frac)
    per_side_fee_pct = bps_to_percent(taker_fee_bps)
    slip_pct = bps_to_percent(slippage_bps)
    return (target_usdt / size) * 100.0 + (2 * per_side_fee_pct) + slip_pct

# ---------------- Presets ----------------
PRESETS = {
    "Conservative": dict(adx_min=22, atr_min=0.20, atr_max=1.00, rvol_min=1.25, micro_rvol_min=1.35,
                         min_rr=1.20, allow_neutral_15m=False, use_vwap_gate=True,
                         taker_buy_thr=1.10, taker_sell_thr=0.90,
                         vwap_dist_bps_max=10, roc1_thr=0.12,
                         enable_bounce=True, enable_pullback=False, enable_momentum=False),
    "Balanced":     dict(adx_min=20, atr_min=0.20, atr_max=1.20, rvol_min=1.20, micro_rvol_min=1.30,
                         min_rr=1.10, allow_neutral_15m=True, use_vwap_gate=True,
                         taker_buy_thr=1.10, taker_sell_thr=0.90,
                         vwap_dist_bps_max=12, roc1_thr=0.10,
                         enable_bounce=True, enable_pullback=True, enable_momentum=False),
    "Aggressive":   dict(adx_min=15, atr_min=0.12, atr_max=1.50, rvol_min=1.00, micro_rvol_min=1.15,
                         min_rr=1.00, allow_neutral_15m=True, use_vwap_gate=True,
                         taker_buy_thr=1.05, taker_sell_thr=0.95,
                         vwap_dist_bps_max=15, roc1_thr=0.08,
                         enable_bounce=True, enable_pullback=True, enable_momentum=True),
    "Panic":        dict(adx_min=10, atr_min=0.08, atr_max=2.00, rvol_min=0.80, micro_rvol_min=1.00,
                         min_rr=0.80, allow_neutral_15m=True, use_vwap_gate=False,
                         taker_buy_thr=1.00, taker_sell_thr=1.00,
                         vwap_dist_bps_max=20, roc1_thr=0.06,
                         enable_bounce=True, enable_pullback=True, enable_momentum=True),
}

# ---------------- Triggers ----------------
def vwap_bounce_signal(side, price, vwap_val, atr1_px, sw_low, sw_high, thr):
    if not np.isfinite(vwap_val): return None
    dist_bps = abs(price - vwap_val) / price * 10000.0
    if dist_bps > thr["vwap_dist_bps_max"]: return None
    if side == "long" and price > vwap_val:
        stop = min(vwap_val - 0.6*atr1_px, price - 0.6*atr1_px)
        if np.isfinite(sw_low): stop = min(stop, sw_low)
        return dict(stop=stop, type="Bounce")
    if side == "short" and price < vwap_val:
        stop = max(vwap_val + 0.6*atr1_px, price + 0.6*atr1_px)
        if np.isfinite(sw_high): stop = max(stop, sw_high)
        return dict(stop=stop, type="Bounce")
    return None

def pullback_ema_signal(side, price, df1_close, atr1_px, sw_low, sw_high):
    if len(df1_close) < 25: return None
    e9  = ema(df1_close, 9).iloc[-1]
    e21 = ema(df1_close, 21).iloc[-1]
    if side == "long":
        touched = (df1_close.iloc[-2] <= e9 and df1_close.iloc[-1] > e9) or (df1_close.iloc[-2] <= e21 and df1_close.iloc[-1] > e21)
        if not touched: return None
        stop = min(price - 0.9*atr1_px, (sw_low if np.isfinite(sw_low) else price - 1.0*atr1_px))
        return dict(stop=stop, type="Pullback")
    else:
        touched = (df1_close.iloc[-2] >= e9 and df1_close.iloc[-1] < e9) or (df1_close.iloc[-2] >= e21 and df1_close.iloc[-1] < e21)
        if not touched: return None
        stop = max(price + 0.9*atr1_px, (sw_high if np.isfinite(sw_high) else price + 1.0*atr1_px))
        return dict(stop=stop, type="Pullback")

def momentum_signal(side, roc1_thr, roc1, atr1_px, price):
    if side == "long" and roc1 >= roc1_thr:
        return dict(stop=price - 0.9*atr1_px, type="Momentum")
    if side == "short" and (-roc1) >= roc1_thr:
        return dict(stop=price + 0.9*atr1_px, type="Momentum")
    return None

# ---------------- Auto-relax helper ----------------
def auto_relax_thresholds(thr: dict, relax: float) -> dict:
    # relax: 0..1 — softens gates over time
    t = thr.copy()
    t["min_adx_15m"]    = max(0, int(t["min_adx_15m"] - 10*relax))
    t["atr_min"]        = max(0.05, t["atr_min"] - 0.10*relax)
    t["atr_max"]        = min(2.00, t["atr_max"] + 0.50*relax)
    t["rvol_min"]       = max(0.60, t["rvol_min"] - 0.30*relax)
    t["micro_rvol_min"] = max(1.00, t["micro_rvol_min"] - 0.30*relax)
    t["min_rr"]         = max(0.70, t["min_rr"] - 0.20*relax)
    if relax >= 0.30: t["enable_momentum"] = True
    if relax >= 0.60: t["use_vwap_gate"]   = False
    return t

# ---------------- Build exact signal ----------------
def build_signal(price, dir5, dir15, micro, adx15, atr5pct, atr1pct, vwap_val, vratio5,
                 qvol, ema200_bias5, ema200_5, bars_since, oi_delta, taker_bs,
                 gross_needed_pct, time_left_min, thr, long_only=False):
    reasons = []
    if not np.isfinite(price):
        return {"signal":"Wait","reason":"No price","stop_price":None,"tp_price":None,"rr":None}

    if qvol < thr["min_quote_usdt"]:
        return {"signal":"Avoid","reason":f"Low liquidity (<{thr['min_quote_usdt']:.0f})","stop_price":None,"tp_price":None,"rr":None}

    if np.isfinite(adx15) and adx15 < thr["min_adx_15m"]:
        reasons.append(f"ADX15 {adx15:.1f}<{thr['min_adx_15m']}")
    if np.isfinite(atr5pct) and not (thr["atr_min"] <= atr5pct <= thr["atr_max"]):
        reasons.append(f"ATR5% {atr5pct:.2f}∉[{thr['atr_min']:.2f},{thr['atr_max']:.2f}]")
    if thr["use_volume"] and vratio5 < thr["rvol_min"]:
        reasons.append(f"RVOL5 {vratio5:.2f}<{thr['rvol_min']:.2f}")

    want_long  = (dir5 == "Buy"  and (dir15 in ["Buy","Neutral"] or thr["allow_neutral_15m"]))
    want_short = (dir5 == "Sell" and (dir15 in ["Sell","Neutral"] or thr["allow_neutral_15m"]))
    if thr["use_vwap_gate"] and np.isfinite(vwap_val):
        if want_long and not (price > vwap_val): reasons.append("VWAP gate (need above)")
        if want_short and not (price < vwap_val): reasons.append("VWAP gate (need below)")
    if long_only and want_short:
        reasons.append("Long-only mode"); want_short = False

    if reasons:
        return {"signal":"Wait","reason":"; ".join(reasons),"stop_price":None,"tp_price":None,"rr":None}

    flow_ok_long  = (oi_delta is None or oi_delta >= 0) and (taker_bs is None or taker_bs >= thr["taker_buy_thr"])
    flow_ok_short = (oi_delta is None or oi_delta <= 0) and (taker_bs is None or taker_bs <= thr["taker_sell_thr"])

    atr1pct = atr1pct if isinstance(atr1pct,(int,float)) and atr1pct>0 else 0.07
    atr1_px = price * atr1pct / 100.0
    sw_low, sw_high = thr["sw_low"], thr["sw_high"]
    long_tp  = price * (1.0 + gross_needed_pct/100.0)
    short_tp = price * (1.0 - gross_needed_pct/100.0)

    micro_ok_long  = (micro["trigger"] == "BreakUp" and micro["rvol1"] >= thr["micro_rvol_min"])
    micro_ok_short = (micro["trigger"] == "BreakDn" and micro["rvol1"] >= thr["micro_rvol_min"])

    if want_long and micro_ok_long and flow_ok_long:
        stop = (sw_low if np.isfinite(sw_low) else price - 0.8*atr1_px)
        rr = (long_tp - price) / max(1e-9, price - stop) if stop < price else 0.0
        if rr >= thr["min_rr"]:
            return {"signal":"Long Now","reason":"Breakout","stop_price":round(stop,6),"tp_price":round(long_tp,6),"rr":round(rr,2)}

    if want_short and micro_ok_short and flow_ok_short:
        stop = (sw_high if np.isfinite(sw_high) else price + 0.8*atr1_px)
        rr = (price - short_tp) / max(1e-9, stop - price) if stop > price else 0.0
        if rr >= thr["min_rr"]:
            return {"signal":"Short Now","reason":"Breakdown","stop_price":round(stop,6),"tp_price":round(short_tp,6),"rr":round(rr,2)}

    if thr["enable_bounce"]:
        if want_long and flow_ok_long:
            b = vwap_bounce_signal("long", price, vwap_val, atr1_px, sw_low, sw_high, thr)
            if b:
                stop = b["stop"]; rr = (long_tp - price) / max(1e-9, price - stop) if stop < price else 0.0
                if rr >= thr["min_rr"]:
                    return {"signal":"Long Now","reason":"VWAP Bounce","stop_price":round(stop,6),"tp_price":round(long_tp,6),"rr":round(rr,2)}
        if want_short and flow_ok_short:
            b = vwap_bounce_signal("short", price, vwap_val, atr1_px, sw_low, sw_high, thr)
            if b:
                stop = b["stop"]; rr = (price - short_tp) / max(1e-9, stop - price) if stop > price else 0.0
                if rr >= thr["min_rr"]:
                    return {"signal":"Short Now","reason":"VWAP Bounce","stop_price":round(stop,6),"tp_price":round(short_tp,6),"rr":round(rr,2)}

    if thr["enable_pullback"]:
        df1_close = thr["df1_close"]
        if want_long and flow_ok_long:
            p = pullback_ema_signal("long", price, df1_close, atr1_px, sw_low, sw_high)
            if p:
                stop = p["stop"]; rr = (long_tp - price) / max(1e-9, price - stop) if stop < price else 0.0
                if rr >= thr["min_rr"]:
                    return {"signal":"Long Now","reason":"EMA Pullback","stop_price":round(stop,6),"tp_price":round(long_tp,6),"rr":round(rr,2)}
        if want_short and flow_ok_short:
            p = pullback_ema_signal("short", price, df1_close, atr1_px, sw_low, sw_high)
            if p:
                stop = p["stop"]; rr = (price - short_tp) / max(1e-9, stop - price) if stop > price else 0.0
                if rr >= thr["min_rr"]:
                    return {"signal":"Short Now","reason":"EMA Pullback","stop_price":round(stop,6),"tp_price":round(short_tp,6),"rr":round(rr,2)}

    if thr["enable_momentum"]:
        roc1 = thr["roc1"]
        if want_long and flow_ok_long:
            m = momentum_signal("long", thr["roc1_thr"], roc1, atr1_px, price)
            if m:
                stop = m["stop"]; rr = (long_tp - price) / max(1e-9, price - stop) if stop < price else 0.0
                if rr >= thr["min_rr"]:
                    return {"signal":"Long Now","reason":"1m Momentum","stop_price":round(stop,6),"tp_price":round(long_tp,6),"rr":round(rr,2)}
        if want_short and flow_ok_short:
            m = momentum_signal("short", thr["roc1_thr"], roc1, atr1_px, price)
            if m:
                stop = m["stop"]; rr = (price - short_tp) / max(1e-9, stop - price) if stop > price else 0.0
                if rr >= thr["min_rr"]:
                    return {"signal":"Short Now","reason":"1m Momentum","stop_price":round(stop,6),"tp_price":round(short_tp,6),"rr":round(rr,2)}

    return {"signal":"Wait","reason":"No trigger match","stop_price":None,"tp_price":None,"rr":None}

# ---------------- UI ----------------
st.set_page_config(page_title="Exact Signal Dashboard — 15m Pool (+DEX)", layout="wide")
st.title("Exact Signal Dashboard — 15m Pools (v2 + Dexscreener + Fast Scan)")

# Symbols input with safe fallback and quick helpers
syms_default_text = ",".join(DEFAULT_SYMBOLS)
if "sym_text" not in st.session_state:
    st.session_state["sym_text"] = syms_default_text

syms_text = st.text_input("Symbols (comma-separated):", value=st.session_state["sym_text"], key="sym_text")
symbols = [s.strip().upper() for s in syms_text.split(",") if s.strip()] or DEFAULT_SYMBOLS.copy()

csa, csb, csc = st.columns([1,1,2])
with csa:
    if st.button("Load default symbols"):
        st.session_state["sym_text"] = syms_default_text
        st.rerun()
with csb:
    st.caption(f"{len(symbols)} symbols loaded")

# Quick connectivity test
def quick_ping():
    results = []
    try:
        r = requests.get(f"{BINANCE_HTTP}/api/v3/ping", timeout=5); results.append(("Binance Spot", r.status_code==200))
    except Exception: results.append(("Binance Spot", False))
    try:
        r = requests.get(f"{FAPI_HTTP}/fapi/v1/ping", timeout=5); results.append(("Binance Futures", r.status_code==200))
    except Exception: results.append(("Binance Futures", False))
    try:
        r = requests.get("https://api.dexscreener.com/pairs", timeout=5); results.append(("Dexscreener", r.status_code==200))
    except Exception: results.append(("Dexscreener", False))
    return results

with csc:
    if st.button("Run quick connectivity test"):
        for name, ok in quick_ping():
            st.write(f"{'✅' if ok else '❌'} {name}")

# Top controls
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
with c1:
    auto = st.checkbox("Auto-refresh", True)
with c2:
    refresh_ms = st.slider("Refresh (ms)", 2000, 120000, 15000, 1000)
with c3:
    align_5m = st.checkbox("Align to 5m close", True)
with c4:
    min_conf = st.slider("Min confidence (scanner filter)", 0, 100, 0, 5)
with c5:
    sort_col = st.selectbox("Sort scanner by", ["Confidence %","5m ROC %","15m ROC %","Symbol","Price"])
desc = st.checkbox("Descending", True)

# Pool settings
st.markdown("### Pool Settings")
cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
with cc1:
    balance_usdt = st.number_input("Start Balance (USDT)", min_value=1.0, value=500.0, step=10.0)
with cc2:
    target_usdt  = st.number_input("Target +USDT", min_value=0.1, value=2.0, step=0.1)
with cc3:
    time_limit_min = st.number_input("Time Limit (min)", min_value=1, value=15, step=1)
with cc4:
    size_frac = st.slider("Position Size %", 10, 100, 100, 5) / 100.0
with cc5:
    taker_fee_bps = st.number_input("Taker Fee (bps/side)", min_value=0.0, value=10.0, step=0.5)
with cc6:
    slippage_bps  = st.number_input("Slippage (bps RT)", min_value=0.0, value=5.0, step=0.5)

gross_needed = gross_move_needed_pct(balance_usdt, size_frac, target_usdt, taker_fee_bps, slippage_bps)
st.markdown(f"- Required gross move to hit ≈ +{target_usdt:.2f} USDT: **{gross_needed:.2f}%** using {size_frac*100:.0f}% size (fees+slip incl)")

# Timer
colt1, colt2 = st.columns(2)
with colt1:
    if st.button("Start Timer"):
        st.session_state["t0"] = time.time()
        st.session_state["limit_s"] = int(time_limit_min*60)
with colt2:
    if st.button("Reset Timer"):
        st.session_state.pop("t0", None); st.session_state.pop("limit_s", None)

time_left_min = None
progress_val = 0.0; time_left_text = "—"
if "t0" in st.session_state and "limit_s" in st.session_state:
    elapsed = time.time() - st.session_state["t0"]
    left = max(0, st.session_state["limit_s"] - elapsed)
    progress_val = min(1.0, elapsed / max(1, st.session_state["limit_s"]))
    m = int(left // 60); s = int(left % 60)
    time_left_text = f"{m:02d}:{s:02d} left"
    time_left_min = left / 60.0
st.progress(progress_val, text=time_left_text)

# Mode & Filters
st.markdown("### Mode & Filters")
mc1, mc2, mc3, mc4 = st.columns([1,1,1,2])
with mc1:
    mode_choice = st.selectbox("Signal Mode", ["Auto","Conservative","Balanced","Aggressive","Panic"], index=1)
with mc2:
    long_only = st.checkbox("Long-only", False)
with mc3:
    use_derivatives = st.checkbox("Use Futures flow (OI Δ / Taker)", True)
with mc4:
    min_quote_usdt = st.number_input("Min 24h Quote Volume (USDT)", min_value=0.0, value=80_000_000.0, step=5_000_000.0, format="%.0f")

dx1, dx2, dx3 = st.columns([1,1,2])
with dx1:
    use_dex = st.checkbox("Use Dexscreener Pulse", True)
with dx2:
    dex_min_liq = st.number_input("Min DEX Liquidity (USD)", min_value=0.0, value=500_000.0, step=50_000.0, format="%.0f")
with dx3:
    show_dex_in_notes = st.checkbox("Show DEX pulse in Notes", True)

# Speed/relax controls
fs1, fs2, fs3, fs4 = st.columns([1,1,1,2])
with fs1:
    fast_scan = st.checkbox("Fast Scan (looser gates, fewer calls)", True)
with fs2:
    escalate_sec = st.slider("Auto-relax if no signals (sec)", 30, 600, 180, 10)
with fs3:
    last_mins_override = st.slider("Last N min ultra-loose override", 0, 10, 3, 1)
with fs4:
    force_pick = st.checkbox("Always show a Go (Risky) pick when no cards", True)

def pick_preset(mode_choice, time_left_min):
    if mode_choice != "Auto":
        return mode_choice
    if time_left_min is None: return "Balanced"
    if time_left_min <= 5:  return "Panic"
    if time_left_min <= 8:  return "Aggressive"
    if time_left_min <= 12: return "Balanced"
    return "Conservative"

rows = []; signal_cards = []; fallback_candidates = []
if "last_signals" not in st.session_state: st.session_state["last_signals"] = {}
if "last_card_time" not in st.session_state: st.session_state["last_card_time"] = time.time()

def notify(msg):
    try: st.toast(msg)
    except Exception: st.info(msg)

# Scanner loop
for sym in symbols:
    try:
        df1  = fetch_klines_1m(sym, 240)
        df5  = fetch_klines_5m(sym, 300)
        df15 = fetch_klines_15m(sym, 200)

        price = last_price(df5)
        s5, s15 = tf_signal(df5), tf_signal(df15)

        qvol        = spot_24h_quote_volume(sym)
        adx15_val   = float(adx(df15, 14).iloc[-1]) if len(df15) > 30 else float("nan")
        atr5pct_val = float(atr_pct(df5, 14).iloc[-1]) if len(df5) > 30 else float("nan")
        atr1pct_val = float(atr_pct(df1, 14).iloc[-1]) if len(df1) > 30 else float("nan")
        vwap_val    = vwap_today(df5)
        bars_since  = ema_cross_bars_ago(df5["close"], 9, 21)
        micro       = micro_trigger(df1, vol_thr=1.2)
        sw_low, sw_high = swing_levels(df1, lookback=5)
        roc1 = float((df1["close"].iloc[-1]/df1["close"].iloc[-2] - 1) * 100) if len(df1) > 1 else 0.0

        preset_name = pick_preset(mode_choice, time_left_min)
        P = PRESETS[preset_name].copy()

        # Base thresholds
        thr = dict(
            min_quote_usdt=min_quote_usdt,
            min_adx_15m=P["adx_min"],
            atr_min=P["atr_min"], atr_max=P["atr_max"],
            use_volume=True, use_vwap_gate=P["use_vwap_gate"],
            rvol_min=P["rvol_min"], micro_rvol_min=P["micro_rvol_min"],
            min_rr=P["min_rr"],
            taker_buy_thr=P["taker_buy_thr"], taker_sell_thr=P["taker_sell_thr"],
            allow_neutral_15m=P["allow_neutral_15m"],
            vwap_dist_bps_max=P["vwap_dist_bps_max"],
            enable_bounce=P["enable_bounce"], enable_pullback=P["enable_pullback"], enable_momentum=P["enable_momentum"],
            df1_close=df1["close"], sw_low=sw_low, sw_high=sw_high, roc1=roc1,
        )

        # Local flags (Fast Scan may skip slow calls)
        local_use_derivatives = use_derivatives
        local_use_dex = use_dex

        if fast_scan:
            thr["min_adx_15m"]    = max(10, thr["min_adx_15m"] - 5)
            thr["rvol_min"]       = max(0.90, thr["rvol_min"] - 0.20)
            thr["micro_rvol_min"] = max(1.05, thr["micro_rvol_min"] - 0.20)
            thr["min_rr"]         = max(0.90, thr["min_rr"] - 0.10)
            thr["enable_momentum"]= True
            local_use_derivatives = False
            local_use_dex = False

        # External flows (respect local flags)
        oi_delta = futures_oi_delta_5m(sym) if local_use_derivatives else None
        taker_bs = futures_taker_bs_ratio_5m(sym) if local_use_derivatives else None
        dex = dex_pulse_for_symbol(sym, min_liq_usd=dex_min_liq) if local_use_dex else {"pulse": None}
        dex_p = dex.get("pulse")

        # DEX pulse nudges
        if local_use_dex and dex_p is not None:
            if s5["dir"] == "Buy":
                if dex_p >= 0.30:
                    thr["micro_rvol_min"] = max(1.0, thr["micro_rvol_min"] - 0.20)
                    thr["min_rr"] = max(0.7, thr["min_rr"] - 0.10)
                elif dex_p <= -0.30:
                    thr["micro_rvol_min"] += 0.20
                    thr["min_rr"] += 0.10
            elif s5["dir"] == "Sell":
                if dex_p <= -0.30:
                    thr["micro_rvol_min"] = max(1.0, thr["micro_rvol_min"] - 0.20)
                    thr["min_rr"] = max(0.7, thr["min_rr"] - 0.10)
                elif dex_p >= 0.30:
                    thr["micro_rvol_min"] += 0.20
                    thr["min_rr"] += 0.10

        # Auto-relax if no card recently
        time_since = time.time() - st.session_state["last_card_time"]
        relax = 0.0 if escalate_sec <= 0 else min(1.0, max(0.0, time_since / float(escalate_sec)))
        thr = auto_relax_thresholds(thr, relax)

        # Last minutes ultra-loose override
        if time_left_min is not None and time_left_min <= last_mins_override:
            thr["min_adx_15m"] = 0
            thr["use_vwap_gate"] = False
            thr["rvol_min"] = 0.80
            thr["micro_rvol_min"] = 1.00
            thr["min_rr"] = 0.70
            thr["allow_neutral_15m"] = True
            thr["enable_momentum"] = True
            thr["atr_min"], thr["atr_max"] = 0.05, 2.00

        # Build signal
        sig = build_signal(price, s5["dir"], s15["dir"], micro, adx15_val, atr5pct_val, atr1pct_val,
                           vwap_val, s5["vratio"], qvol, s5["ema200_bias"], s5["ema200"], bars_since,
                           oi_delta, taker_bs, gross_needed, time_left_min, thr, long_only=long_only)

        conf = confidence_score(s5["dir"], s15["dir"], adx15_val, s5["vratio"], price, vwap_val, oi_delta, taker_bs)

        if sig["signal"] in ("Long Now","Short Now"):
            size = balance_usdt * size_frac
            prof_gross = size * (gross_needed/100.0)
            fee_rt = size * bps_to_percent(taker_fee_bps) * 2
            slip_rt= size * bps_to_percent(slippage_bps)
            prof_net = prof_gross - fee_rt - slip_rt
            reason = sig["reason"]
            if local_use_dex and dex_p is not None:
                reason += f" | DEX {dex_p:+.2f}"
            signal_cards.append({
                "Symbol": sym, "Signal": sig["signal"], "Entry": price,
                "Stop": sig["stop_price"], "TP": sig["tp_price"], "RR": sig["rr"],
                "Est Net +USDT at TP": round(prof_net, 2),
                "Reason": reason, "Preset": preset_name
            })
            last_sig = st.session_state["last_signals"].get(sym)
            if last_sig != sig["signal"]:
                toast_msg = f"{sym}: {sig['signal']} ({sig['reason']})"
                if local_use_dex and dex_p is not None:
                    toast_msg += f" | DEX {dex_p:+.2f}"
                toast_msg += f" | TP≈{sig['tp_price']} Stop≈{sig['stop_price']}"
                notify(toast_msg)
                st.session_state["last_signals"][sym] = sig["signal"]
            st.session_state["last_card_time"] = time.time()

        notes = []
        if isinstance(adx15_val,float) and not np.isnan(adx15_val) and adx15_val >= P["adx_min"]: notes.append("Trend✅")
        if isinstance(vwap_val,float) and not np.isnan(vwap_val): notes.append("↑VWAP" if price > vwap_val else "↓VWAP")
        if isinstance(oi_delta,float): notes.append("OI↑" if oi_delta > 0 else "OI↓")
        if isinstance(taker_bs,float): notes.append("TB>SB" if taker_bs > 1 else "TB<SB")
        if micro["trigger"] != "None": notes.append("1m⚡" + ("↑" if micro["trigger"]=="BreakUp" else "↓"))
        if show_dex_in_notes and local_use_dex and dex_p is not None: notes.append(f"DEX{dex_p:+.2f}")
        note_str = " · ".join(notes)

        rows.append({
            "Symbol": sym, "Price": round(price,6),
            "5m": s5["dir"], "15m": s15["dir"],
            "Signal": sig["signal"], "Reason": sig["reason"],
            "Stop": sig["stop_price"], "TP": sig["tp_price"], "RR": sig["rr"],
            "Confidence %": conf, "5m ROC %": s5["roc"], "15m ROC %": s15["roc"],
            "5m Vratio": s5["vratio"], "Preset": preset_name, "Notes": note_str
        })

        # Fallback candidate (for Go Risky)
        fallback_candidates.append({
            "sym": sym, "price": price, "dir5": s5["dir"], "dir15": s15["dir"],
            "micro": micro["trigger"], "vwap": vwap_val, "vratio5": s5["vratio"],
            "roc1": roc1, "atr1pct": atr1pct_val, "sw_low": sw_low, "sw_high": sw_high,
            "conf": conf, "dex_p": dex_p if local_use_dex else None
        })

    except Exception as e:
        rows.append({
            "Symbol": sym, "Price": None, "5m":"—", "15m":"—",
            "Signal":"ERR", "Reason": str(e), "Stop":None, "TP":None, "RR":None,
            "Confidence %":0, "5m ROC %":None, "15m ROC %":None, "5m Vratio":None,
            "Preset": None, "Notes":"ERR"
        })

# Force a "Go (Risky)" pick if still no cards
if force_pick and not signal_cards and fallback_candidates:
    def fb_score(c):
        score = c["conf"]
        if c["dir5"] == "Buy" and c["micro"] == "BreakUp": score += 10
        if c["dir5"] == "Sell" and c["micro"] == "BreakDn": score += 10
        if isinstance(c["vwap"], float) and np.isfinite(c["vwap"]):
            if c["dir5"] == "Buy" and c["price"] > c["vwap"]: score += 6
            if c["dir5"] == "Sell" and c["price"] < c["vwap"]: score += 6
        if c.get("dex_p") is not None:
            if c["dir5"] == "Buy" and c["dex_p"] > 0: score += 3
            if c["dir5"] == "Sell" and c["dex_p"] < 0: score += 3
        if (c["dir5"], c["dir15"]) in [("Buy","Sell"), ("Sell","Buy")]: score -= 6
        return score

    c = max(fallback_candidates, key=fb_score)
    atr1pct = c["atr1pct"] if isinstance(c["atr1pct"], (int,float)) and c["atr1pct"]>0 else 0.07
    atr1_px = c["price"] * atr1pct / 100.0
    if c["dir5"] == "Buy":
        stop = (c["sw_low"] if np.isfinite(c["sw_low"]) else c["price"] - 0.8*atr1_px)
        tp   = c["price"] * (1.0 + gross_needed/100.0)
        rr   = (tp - c["price"]) / max(1e-9, c["price"] - stop)
        side = "Long Now"
    else:
        stop = (c["sw_high"] if np.isfinite(c["sw_high"]) else c["price"] + 0.8*atr1_px)
        tp   = c["price"] * (1.0 - gross_needed/100.0)
        rr   = (c["price"] - tp) / max(1e-9, stop - c["price"])
        side = "Short Now"
    est_net = balance_usdt*size_frac*(gross_needed/100.0) - balance_usdt*size_frac*bps_to_percent(taker_fee_bps)*2 - balance_usdt*size_frac*bps_to_percent(slippage_bps)
    signal_cards.append({
        "Symbol": c["sym"], "Signal": side, "Entry": c["price"],
        "Stop": round(stop,6), "TP": round(tp,6), "RR": round(rr,2),
        "Est Net +USDT at TP": round(est_net, 2),
        "Reason": "Go (Risky) — best available: micro/VWAP/conf blend", "Preset": "Fallback"
    })

# --------- Table ----------
df = pd.DataFrame(rows)

if df.empty:
    st.warning("No symbols processed. Click 'Load default symbols' or run the connectivity test. If it persists, try Fast Scan.")
else:
    df = df[df["Confidence %"] >= min_conf].copy()
    sort_map = {"Confidence %":"Confidence %","5m ROC %":"5m ROC %","15m ROC %":"15m ROC %","Symbol":"Symbol","Price":"Price"}
    df = df.sort_values(sort_map[sort_col], ascending=not desc, kind="mergesort")
    cols = ["Symbol","Price","5m","15m","Signal","Reason","Stop","TP","RR","Confidence %","Preset","Notes"]
    df_view = df[cols].copy()

    def color_side(val):
        if val in ["Buy","Long Now"]: return "background-color:#e8f7ee; color:#066e2c; font-weight:600"
        if val in ["Sell","Short Now"]: return "background-color:#fde9e7; color:#a02020; font-weight:600"
        if val in ["Neutral","Wait"]: return "color:#6b7280"
        return ""

    def color_conf(v):
        try: v = float(v)
        except: return ""
        if v >= 70: return "background-color:#16a34a; color:white; font-weight:600"
        if v >= 55: return "background-color:#bbf7d0; color:#064e3b"
        if v <= 30: return "background-color:#b91c1c; color:white; font-weight:600"
        if v <= 45: return "background-color:#ffe4e6; color:#7f1d1d"
        return ""

    st.markdown("#### Market Scanner")
    styler = (df_view.style
              .applymap(color_side, subset=[c for c in ["5m","15m","Signal"] if c in df_view.columns])
              .applymap(color_conf, subset=[c for c in ["Confidence %"] if c in df_view.columns])
              .format(precision=6, subset=["Price","Stop","TP"])
              .format(precision=2, subset=["RR","Confidence %"]))
    st.dataframe(styler, use_container_width=True, height=min(640, 120 + 28*max(3,len(df_view))))

# --------- Signal Cards ----------
st.markdown("#### Signal Cards (Top actionable picks)")
if signal_cards:
    cards = sorted(signal_cards, key=lambda x: (-x["RR"], x["Symbol"]))[:3]
    cols_c = st.columns(len(cards))
    for i, card in enumerate(cards):
        with cols_c[i]:
            st.metric(label=f"{card['Symbol']} — {card['Signal']}", value=f"RR {card['RR']}",
                      delta=f"Est +{card['Est Net +USDT at TP']:.2f} USDT | {card['Preset']}")
            st.write(f"Entry: {card['Entry']:.6f}")
            st.write(f"TP: {card['TP']:.6f}   |   Stop: {card['Stop']:.6f}")
            st.caption(card["Reason"])
else:
    st.info("No 'Long Now' / 'Short Now' yet. Fast Scan + Auto-relax will loosen gates until a card appears.")

st.caption("Tip: majors fill fastest; use 100% size in pools; cut fast if the 1m trigger fails. Manual signals only.")

# --------- Auto-refresh ----------
if auto:
    if align_5m:
        now_ms = int(time.time() * 1000)
        wait_ms = 300_000 - (now_ms % 300_000)
        if wait_ms < 500: wait_ms += 300_000
        wait_ms = min(wait_ms, refresh_ms)
    else:
        wait_ms = refresh_ms
    time.sleep(wait_ms / 1000.0)
    st.rerun()