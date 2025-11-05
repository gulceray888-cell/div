import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timezone, timedelta

# === TELEGRAM SECRETS (GitHub Actions üzerinden geliyor) ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ Telegram secret bulunamadı!")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
        print("✅ Telegram gönderildi.")
    except Exception as e:
        print(f"Telegram gönderim hatası: {e}")

# === LOCAL DATA ===
STATE_FILE = "last_signal.json"

def load_last_signals():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_last_signals(data):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)

# === OKX API ===
def get_okx_ohlcv(symbol="ETH-USDT-SWAP", bar="6H", limit=300):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": symbol, "bar": bar, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()
    if "data" not in data:
        print("⚠️ OKX veri hatası")
        return None
    df = pd.DataFrame(data["data"], columns=["ts","o","h","l","c","v","volCcy","volCcyQuote","confirm"])
    df = df.astype({"o":float,"h":float,"l":float,"c":float,"v":float})
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    return df[["ts","o","h","l","c","v"]].sort_values("ts").reset_index(drop=True)

# === BINANCE API ===
def get_binance_ohlcv(symbol="ETHUSDT", interval="6h", limit=300):
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params)
        data = r.json()
        if not isinstance(data, list):
            print("⚠️ Binance veri hatası (API cevap bozuk)")
            return None
        df = pd.DataFrame(data, columns=["ts","o","h","l","c","v","ct","qav","trades","tbbav","tbqav","ignore"])
        df = df.astype({"o":float,"h":float,"l":float,"c":float,"v":float})
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        return df[["ts","o","h","l","c","v"]].sort_values("ts").reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Binance veri alınamadı: {e}")
        return None

# ===== BURAYA KADAR AYNI =====
# (Mesaj uzunluğu sınırlı, geri kalan kodun tamamını bir sonraki mesajda vereceğim)
# === SIGNAL & DIVERGENCE CORE ===
def detect_pivots(values, prd=5):
    ph, pl = [], []
    n = len(values)
    for i in range(prd, n - prd):
        window = values[i-prd:i+prd+1]
        if values[i] == np.max(window):
            ph.append(i)
        if values[i] == np.min(window):
            pl.append(i)
    return ph[::-1], pl[::-1]

def find_divergences_on_confirmation(df, indicators_map, mode):
    closes = df["c"].reset_index(drop=True)
    ph, pl = detect_pivots(closes.values, prd=5)
    results = []
    for name, series in indicators_map.items():
        s = series.reset_index(drop=True)

        # Bullish = lows
        if len(pl) >= 2:
            for i_new in range(len(pl)):
                idx_new = pl[i_new]
                for j in range(i_new+1, len(pl)):
                    idx_old = pl[j]
                    if idx_new - idx_old <= 5:
                        continue
                    try:
                        cond = (s.iat[idx_new] > s.iat[idx_old]) and (closes.iat[idx_new] < closes.iat[idx_old])
                    except: cond = False
                    if cond:
                        results.append({"indicator": name, "type": "bull", "pivot_new": idx_new})

        # Bearish = highs
        if len(ph) >= 2:
            for i_new in range(len(ph)):
                idx_new = ph[i_new]
                for j in range(i_new+1, len(ph)):
                    idx_old = ph[j]
                    if idx_new - idx_old <= 5:
                        continue
                    try:
                        cond = (s.iat[idx_new] < s.iat[idx_old]) and (closes.iat[idx_new] > closes.iat[idx_old])
                    except: cond = False
                    if cond:
                        results.append({"indicator": name, "type": "bear", "pivot_new": idx_new})
    return results

def collect_relevant_divergences(divs, signal_index, mode, max_offset=7):
    target = "bull" if mode == "long" else "bear"
    rel = []
    for d in divs:
        if d["type"] != target: 
            continue
        off = signal_index - d["pivot_new"]
        if 0 <= off <= max_offset:
            rel.append({"indicator": d["indicator"], "offset": off})
    # en yakın pivotu tut
    best = {}
    for r in rel:
        if r["indicator"] not in best or r["offset"] < best[r["indicator"]]:
            best[r["indicator"]] = r["offset"]
    return [{"indicator": k, "offset": v} for k, v in sorted(best.items(), key=lambda x: x[1])]

def build_full_message(base_msg, divs, mode):
    header = f"\n\n📊 Divergence Kontrolü ({'LONG' if mode=='long' else 'SHORT'}) – Son 7 mum\n\n"
    if not divs:
        return base_msg + header + "❌ Hiç regular divergence bulunamadı."
    lines = ["✅ Regular divergence bulundu:\n"]
    for d in divs:
        lines.append(f"• {d['indicator']} → {d['offset']} mum önce")
    return base_msg + header + "\n".join(lines)


# === LONG (a/c) ===
def detect_and_send_latest(df, symbol_name, last_signals):
    cond = calc_a_c_signal(df)
    lows = df["l"].values
    closes = df["c"].values
    times = df["ts"]

    latest = None
    for i in range(len(df)-1, 4, -1):
        if cond.iloc[i-1] and not cond.iloc[i]:
            latest = {"index": i, "time": times.iloc[i],
                      "entry": round(float(closes[i]), 2),
                      "stop": round(float(min(lows[i-4:i])), 2)}
            break
    if not latest:
        print(f"{symbol_name}: Long yok")
        return

    tr_time = latest["time"].tz_convert("Europe/Istanbul").strftime("%Y-%m-%d %H:%M:%S")
    key = f"{symbol_name}_{tr_time}_{latest['entry']}_{latest['stop']}"
    if key in last_signals:
        print(f"{symbol_name}: Önceden gönderilmiş long, atlandı.")
        return

    base_msg = (
        f"🟢 LONG SİNYALİ (a/c)\n"
        f"Pair: {symbol_name} (6H)\n"
        f"Zaman (TR): {tr_time}\n"
        f"Entry: {latest['entry']}\n"
        f"Stop: {latest['stop']}"
    )

    df2 = df.copy().reset_index(drop=True)
    macd, sig, hist = _macd(df2["c"])
    df2["MACD"], df2["Hist"] = macd, hist
    df2["RSI"] = _rsi(df2["c"])
    df2["CCI"] = _cci(df2)
    df2["OBV"] = _obv(df2)
    df2["STOCH"] = _stoch(df2)

    imap = {
        "MACD": df2["MACD"], "Hist": df2["Hist"], "RSI": df2["RSI"],
        "CCI": df2["CCI"], "OBV": df2["OBV"], "STOCH": df2["STOCH"]
    }

    raw = find_divergences_on_confirmation(df2, imap, "long")
    rel = collect_relevant_divergences(raw, latest["index"], "long")
    full_msg = build_full_message(base_msg, rel, "long")

    send_telegram(full_msg)
    last_signals[key] = str(datetime.now(timezone.utc))
    save_last_signals(last_signals)
    print(f"{symbol_name}: ✅ Long sinyal + divergence gönderildi")


# === SHORT (b/d) ===
def detect_and_send_latest_short(df, symbol_name, last_signals):
    cond = calc_b_signal(df)
    highs = df["h"].values
    closes = df["c"].values
    times = df["ts"]

    latest = None
    for i in range(len(df)-1, 4, -1):
        if cond.iloc[i-1] and not cond.iloc[i]:
            latest = {"index": i, "time": times.iloc[i],
                      "entry": round(float(closes[i]), 2),
                      "stop": round(float(max(highs[i-4:i])), 2)}
            break
    if not latest:
        print(f"{symbol_name}: Short yok")
        return

    tr_time = latest["time"].tz_convert("Europe/Istanbul").strftime("%Y-%m-%d %H:%M:%S")
    key = f"{symbol_name}_{tr_time}_{latest['entry']}_{latest['stop']}"
    if key in last_signals:
        print(f"{symbol_name}: Önceden gönderilmiş short, atlandı.")
        return

    base_msg = (
        f"🔴 SHORT SİNYALİ (b/d)\n"
        f"Pair: {symbol_name} (6H)\n"
        f"Zaman (TR): {tr_time}\n"
        f"Entry: {latest['entry']}\n"
        f"Stop: {latest['stop']}"
    )

    df2 = df.copy().reset_index(drop=True)
    macd, sig, hist = _macd(df2["c"])
    df2["MACD"], df2["Hist"] = macd, hist
    df2["RSI"] = _rsi(df2["c"])
    df2["CCI"] = _cci(df2)
    df2["OBV"] = _obv(df2)
    df2["STOCH"] = _stoch(df2)

    imap = {
        "MACD": df2["MACD"], "Hist": df2["Hist"], "RSI": df2["RSI"],
        "CCI": df2["CCI"], "OBV": df2["OBV"], "STOCH": df2["STOCH"]
    }

    raw = find_divergences_on_confirmation(df2, imap, "short")
    rel = collect_relevant_divergences(raw, latest["index"], "short")
    full_msg = build_full_message(base_msg, rel, "short")

    send_telegram(full_msg)
    last_signals[key] = str(datetime.now(timezone.utc))
    save_last_signals(last_signals)
    print(f"{symbol_name}: ✅ Short sinyal + divergence gönderildi")


# === MAIN EXECUTION ===
def run_cycle():
    print(f"\n--- Çalışma başlatıldı: {datetime.utcnow()} UTC ---")
    last = load_last_signals()

    df_okx = get_okx_ohlcv()
    if df_okx is not None:
        detect_and_send_latest(df_okx, "OKX ETH-USDT-SWAP", last)
        detect_and_send_latest_short(df_okx, "OKX ETH-USDT-SWAP", last)

    df_bin = get_binance_ohlcv()
    if df_bin is not None:
        detect_and_send_latest(df_bin, "BINANCE ETHUSDT_PERP", last)
        detect_and_send_latest_short(df_bin, "BINANCE ETHUSDT_PERP", last)
    else:
        print("⚠️ Binance çalışmadı, sadece OKX kontrol edildi.")

    print("--- Kontrol tamamlandı ---")


if __name__ == "__main__":
    run_cycle()
