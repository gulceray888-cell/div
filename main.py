import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# === TELEGRAM SECRETS (GitHub Actions ile gelecek) ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram secrets bulunamadı (BOT_TOKEN/CHAT_ID). Mesaj atılamıyor.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        if resp.status_code != 200:
            print(f"Telegram gönderildi ama HTTP {resp.status_code} geldi: {resp.text}")
        else:
            print("✅ Telegram gönderildi.")
    except Exception as e:
        print(f"Telegram gönderim hatası: {e}")

# === STATE FILE ===
STATE_FILE = "last_signal.json"

def load_last_signals():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"last_signal.json okunamadı: {e}")
            return {}
    return {}

def save_last_signals(data):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"last_signal.json yazılamadı: {e}")

# === DATA FETCHERS ===
def get_okx_ohlcv(symbol="ETH-USDT-SWAP", bar="6H", limit=300):
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol, "bar": bar, "limit": limit}
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
        if "data" not in data:
            print("⚠️ OKX: data alanı yok")
            return None
        df = pd.DataFrame(data["data"], columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"])
        df = df.astype({"o":float,"h":float,"l":float,"c":float,"v":float})
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        return df[["ts","o","h","l","c","v"]]
    except Exception as e:
        print(f"⚠️ OKX veri alınamadı: {e}")
        return None

def get_binance_ohlcv(symbol="ETHUSDT", interval="6h", limit=300):
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
        if not isinstance(data, list):
            print("⚠️ Binance: API cevap beklenenden farklı")
            return None
        df = pd.DataFrame(data, columns=["ts","o","h","l","c","v","ct","qav","trades","tbbav","tbqav","ignore"])
        df = df.astype({"o":float,"h":float,"l":float,"c":float,"v":float})
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        return df[["ts","o","h","l","c","v"]]
    except Exception as e:
        # Kullanıcı isteğine göre Binance hata durumunda sadece log yaz (telegram uyarısı gönderilmeyecek)
        print(f"⚠️ Binance veri alınamadı: {e}")
        return None

# === SIGNALS (a/c and b/d) ===
def calc_a_c_signal(df: pd.DataFrame):
    close = df["c"]
    high = df["h"]
    sma200 = close.rolling(window=200, min_periods=1).mean()
    length = 20
    basis = close.rolling(length).mean()
    dev = 2 * close.rolling(length).std()
    upper = basis + dev
    lower = basis - dev
    bbr = (close - lower) / (upper - lower)
    cond_a = (bbr < 0.144) & (sma200 > high.shift(1))
    cond_c = (bbr < 0.144) & (sma200 > high.shift(1)) & (bbr.shift(1) < bbr)
    return (cond_a | cond_c).fillna(False).astype(bool)

def calc_b_signal(df: pd.DataFrame):
    close = df["c"]
    low = df["l"]
    sma200 = close.rolling(window=200, min_periods=1).mean()
    length = 20
    basis = close.rolling(length).mean()
    dev = 2 * close.rolling(length).std()
    upper = basis + dev
    lower = basis - dev
    bbr = (close - lower) / (upper - lower)
    cond_b = (bbr > 0.856) & (low.shift(1) > sma200)
    cond_d = (bbr > 0.856) & (bbr.shift(1) > bbr) & (low.shift(1) > sma200)
    return (cond_b | cond_d).fillna(False).astype(bool)

# === INDICATORS for divergence ===
def _rsi(series: pd.Series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=length-1, adjust=False).mean()
    ma_down = down.ewm(com=length-1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def _cci(df: pd.DataFrame, n=10):
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md)

def _obv(df: pd.DataFrame):
    v = df["v"] if "v" in df.columns else pd.Series([0]*len(df))
    obv = [0]
    for i in range(1, len(df)):
        if df["c"].iat[i] > df["c"].iat[i-1]:
            obv.append(obv[-1] + float(v.iat[i]))
        elif df["c"].iat[i] < df["c"].iat[i-1]:
            obv.append(obv[-1] - float(v.iat[i]))
        else:
            obv.append(obv[-1])
    return pd.Series(obv)

def _stoch(df: pd.DataFrame, length=14, smooth_k=3):
    ll = df["l"].rolling(length).min()
    hh = df["h"].rolling(length).max()
    k = 100 * (df["c"] - ll) / (hh - ll)
    return k.rolling(smooth_k).mean()

# === PIVOTS & DIVERGENCE ===
PRD = 5      # pivot left/right (same as Pine script)
MAX_OFFSET = 7  # sinyal mumundan önce kaç bara kadar pivot aranır

def detect_pivots(values: np.ndarray, prd=PRD):
    ph_positions = []
    pl_positions = []
    n = len(values)
    for i in range(prd, n - prd):
        window = values[i-prd:i+prd+1]
        if values[i] == np.max(window):
            ph_positions.append(i)
        if values[i] == np.min(window):
            pl_positions.append(i)
    # newest-first to match pine approach used earlier
    return ph_positions[::-1], pl_positions[::-1]

def find_divergences_on_confirmation(df: pd.DataFrame, indicators_map: dict, mode: str):
    closes = df["c"].reset_index(drop=True)
    ph, pl = detect_pivots(closes.values, prd=PRD)
    results = []
    for name, series in indicators_map.items():
        s = series.reset_index(drop=True)
        # bullish (regular): pivots on lows
        if len(pl) >= 2:
            for i_new in range(len(pl)):
                idx_new = pl[i_new]
                for j in range(i_new+1, len(pl)):
                    idx_old = pl[j]
                    if idx_new - idx_old <= PRD:
                        continue
                    try:
                        cond = (s.iat[idx_new] > s.iat[idx_old]) and (closes.iat[idx_new] < closes.iat[idx_old])
                    except Exception:
                        cond = False
                    if cond:
                        results.append({"indicator": name, "type": "bull", "pivot_new": idx_new, "pivot_old": idx_old})
        # bearish (regular): pivots on highs
        if len(ph) >= 2:
            for i_new in range(len(ph)):
                idx_new = ph[i_new]
                for j in range(i_new+1, len(ph)):
                    idx_old = ph[j]
                    if idx_new - idx_old <= PRD:
                        continue
                    try:
                        cond = (s.iat[idx_new] < s.iat[idx_old]) and (closes.iat[idx_new] > closes.iat[idx_old])
                    except Exception:
                        cond = False
                    if cond:
                        results.append({"indicator": name, "type": "bear", "pivot_new": idx_new, "pivot_old": idx_old})
    return results

def collect_relevant_divergences(divs: list, signal_index: int, mode: str, max_offset=MAX_OFFSET):
    target = "bull" if mode == "long" else "bear"
    relevant = []
    for d in divs:
        if d["type"] != target:
            continue
        pnew = d["pivot_new"]
        if pnew > signal_index:
            continue
        offset = signal_index - pnew
        if 0 <= offset <= max_offset:
            relevant.append({"indicator": d["indicator"], "offset": offset})
    # deduplicate: keep smallest offset per indicator
    dedup = {}
    for r in relevant:
        ind = r["indicator"]
        if ind not in dedup or r["offset"] < dedup[ind]:
            dedup[ind] = r["offset"]
    result = [{"indicator": k, "offset": dedup[k]} for k in sorted(dedup.keys())]
    result.sort(key=lambda x: x["offset"])
    return result

def build_full_message(base_msg: str, divergences: list, mode: str):
    header = f"\n\n📊 Divergence Kontrolü ({'LONG' if mode=='long' else 'SHORT'}) – Son {MAX_OFFSET} mum\n\n"
    if not divergences:
        return base_msg + header + f"❌ Regular {'Bullish' if mode=='long' else 'Bearish'} divergence bulunamadı (0-{MAX_OFFSET} mum)."
    lines = [f"✅ Regular {'Bullish' if mode=='long' else 'Bearish'} divergence bulundu\n"]
    for d in divergences:
        lines.append(f"• {d['indicator']} → {d['offset']} mum önce")
    return base_msg + header + "\n".join(lines)

# === WRAPPERS: detect + send ===
def detect_and_send_latest(df: pd.DataFrame, symbol_name: str, last_signals: dict):
    cond = calc_a_c_signal(df)
    lows = df["l"].values
    closes = df["c"].values
    times = df["ts"]

    latest_signal = None
    for i in range(len(df)-1, 4, -1):
        try:
            if cond.iloc[i-1] and not cond.iloc[i]:
                latest_signal = {"index": i, "time": times.iloc[i],
                                 "entry": round(float(closes[i]), 2),
                                 "stop": round(float(min(lows[i-4:i])), 2)}
                break
        except Exception:
            continue

    if not latest_signal:
        print(f"{symbol_name}: Long sinyali bulunamadı.")
        return

    tr_time = latest_signal["time"].tz_convert("Europe/Istanbul").strftime("%Y-%m-%d %H:%M:%S")
    signal_key = f"{symbol_name}_{tr_time}_{latest_signal['entry']}_{latest_signal['stop']}"
    if signal_key in last_signals:
        print(f"{symbol_name}: {tr_time} long sinyali zaten gönderilmiş, atlandı.")
        return

    base_msg = (
        f"🟢 LONG SİNYALİ (a/c)\n"
        f"Pair: {symbol_name} (6H)\n"
        f"Zaman (TR): {tr_time}\n"
        f"Entry (Close): {latest_signal['entry']}\n"
        f"Stop (4 Mum Low): {latest_signal['stop']}"
    )

    df_local = df.copy().reset_index(drop=True)
    macd_val, macd_sig, macd_hist = _macd(df_local["c"])
    df_local["MACD"] = macd_val
    df_local["Hist"] = macd_hist
    df_local["RSI"] = _rsi(df_local["c"])
    df_local["CCI"] = _cci(df_local)
    df_local["OBV"] = _obv(df_local)
    df_local["STOCH"] = _stoch(df_local)

    indicators_map = {
        "MACD": df_local["MACD"],
        "Hist": df_local["Hist"],
        "RSI": df_local["RSI"],
        "CCI": df_local["CCI"],
        "OBV": df_local["OBV"],
        "STOCH": df_local["STOCH"]
    }

    raw_divs = find_divergences_on_confirmation(df_local, indicators_map, mode="long")
    rel = collect_relevant_divergences(raw_divs, latest_signal["index"], mode="long")
    full_msg = build_full_message(base_msg, rel, mode="long")

    send_telegram(full_msg)
    last_signals[signal_key] = str(datetime.now(timezone.utc))
    save_last_signals(last_signals)
    print(f"{symbol_name}: ✅ Long + divergence raporu gönderildi")

def detect_and_send_latest_short(df: pd.DataFrame, symbol_name: str, last_signals: dict):
    cond = calc_b_signal(df)
    highs = df["h"].values
    closes = df["c"].values
    times = df["ts"]

    latest_signal = None
    for i in range(len(df)-1, 4, -1):
        try:
            if cond.iloc[i-1] and not cond.iloc[i]:
                latest_signal = {"index": i, "time": times.iloc[i],
                                 "entry": round(float(closes[i]), 2),
                                 "stop": round(float(max(highs[i-4:i])), 2)}
                break
        except Exception:
            continue

    if not latest_signal:
        print(f"{symbol_name}: Short sinyali bulunamadı.")
        return

    tr_time = latest_signal["time"].tz_convert("Europe/Istanbul").strftime("%Y-%m-%d %H:%M:%S")
    signal_key = f"{symbol_name}_{tr_time}_{latest_signal['entry']}_{latest_signal['stop']}"
    if signal_key in last_signals:
        print(f"{symbol_name}: {tr_time} short sinyali zaten gönderilmiş, atlandı.")
        return

    base_msg = (
        f"🔴 SHORT SİNYALİ (b/d)\n"
        f"Pair: {symbol_name} (6H)\n"
        f"Zaman (TR): {tr_time}\n"
        f"Entry (Close): {latest_signal['entry']}\n"
        f"Stop (4 Mum High): {latest_signal['stop']}"
    )

    df_local = df.copy().reset_index(drop=True)
    macd_val, macd_sig, macd_hist = _macd(df_local["c"])
    df_local["MACD"] = macd_val
    df_local["Hist"] = macd_hist
    df_local["RSI"] = _rsi(df_local["c"])
    df_local["CCI"] = _cci(df_local)
    df_local["OBV"] = _obv(df_local)
    df_local["STOCH"] = _stoch(df_local)

    indicators_map = {
        "MACD": df_local["MACD"],
        "Hist": df_local["Hist"],
        "RSI": df_local["RSI"],
        "CCI": df_local["CCI"],
        "OBV": df_local["OBV"],
        "STOCH": df_local["STOCH"]
    }

    raw_divs = find_divergences_on_confirmation(df_local, indicators_map, mode="short")
    rel = collect_relevant_divergences(raw_divs, latest_signal["index"], mode="short")
    full_msg = build_full_message(base_msg, rel, mode="short")

    send_telegram(full_msg)
    last_signals[signal_key] = str(datetime.now(timezone.utc))
    save_last_signals(last_signals)
    print(f"{symbol_name}: ✅ Short + divergence raporu gönderildi")

# === MAIN RUN ===
def run_cycle():
    print(f"\n--- Çalışma başlatıldı: {datetime.utcnow()} UTC ---")
    last = load_last_signals()

    df_okx = get_okx_ohlcv()
    if df_okx is not None:
        detect_and_send_latest(df_okx, "OKX ETH-USDT-SWAP", last)
        detect_and_send_latest_short(df_okx, "OKX ETH-USDT-SWAP", last)
    else:
        print("⚠️ OKX veri alınamadı.")

    df_bin = get_binance_ohlcv()
    if df_bin is not None:
        detect_and_send_latest(df_bin, "BINANCE ETHUSDT_PERP", last)
        detect_and_send_latest_short(df_bin, "BINANCE ETHUSDT_PERP", last)
    else:
        # Kullanıcı kararı: Binance hata verirse sessiz geç (telegram uyarısı gönderilmez)
        print("⚠️ Binance çalışmadı, sadece OKX kontrol edildi (eğer OKX verisi varsa).")

    print("--- Kontrol tamamlandı ---")

if __name__ == "__main__":
    run_cycle()
