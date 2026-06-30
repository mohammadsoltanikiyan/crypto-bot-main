import os
import json
import aiohttp
import asyncio
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
TOKEN      = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise SystemExit(
        "❌ BOT_TOKEN تنظیم نشده. فایل .env بساز و BOT_TOKEN=<توکن بات‌فادر> رو توش بذار."
    )

DATA_FILE  = os.environ.get("DATA_FILE", "users_data.json")
USERS_FILE = os.environ.get("USERS_FILE", "users_list.json")   # فایل ثبت کاربران
ADMIN_ID   = os.environ.get("ADMIN_ID", "")  # chat_id ادمین

scheduler = AsyncIOScheduler()
session = None
user_data = {}
user_jobs = {}
user_states = {}

AVAILABLE_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","AVAXUSDT","ADAUSDT","DOTUSDT","NEARUSDT",
    "ATOMUSDT","ALGOUSDT","FTMUSDT","INJUSDT","SUIUSDT","APTUSDT",
    "UNIUSDT","AAVEUSDT","MKRUSDT","CRVUSDT","LDOUSDT",
    "BNBUSDT","OKBUSDT","DOGEUSDT","SHIBUSDT","PEPEUSDT","FLOKIUSDT",
    "XRPUSDT","TRXUSDT","XLMUSDT","LTCUSDT","LINKUSDT","FILUSDT",
    "ARUSDT","RENDERUSDT","TONUSDT","NOTUSDT","FETUSDT","AGIXUSDT",
    "WLDUSDT","SANDUSDT","MANAUSDT","AXSUSDT","IMXUSDT",
]

TRADING_MODES = {
    "scalp": {
        "label": "⚡ اسکالپ (۵-۱۵ دقیقه)",
        "timeframes": ["1m","5m","15m"],
        "weights":    {"1m":1,"5m":2,"15m":3},
        "kline_limit": 150,
        "signal_threshold": 6,
        "min_agreement": 0.70,
        "hold_label": "۱۵-۳۰ دقیقه",
        "hold_hours": 0.5,
        "interval_minutes": 15,
        "risk_pct": 1.0,
    },
    "short": {
        "label": "🕐 کوتاه‌مدت (۱-۴ ساعت)",
        "timeframes": ["15m","1h","4h"],
        "weights":    {"15m":1,"1h":3,"4h":2},
        "kline_limit": 150,
        "signal_threshold": 8,
        "min_agreement": 0.70,
        "hold_label": "۴-۱۲ ساعت",
        "hold_hours": 6,
        "interval_minutes": 60,
        "risk_pct": 1.5,
    },
    "mid": {
        "label": "📅 میان‌مدت (روزانه)",
        "timeframes": ["1h","4h","1d"],
        "weights":    {"1h":1,"4h":2,"1d":3},
        "kline_limit": 200,
        "signal_threshold": 10,
        "min_agreement": 0.72,
        "hold_label": "۲-۷ روز",
        "hold_hours": 72,
        "interval_minutes": 240,
        "risk_pct": 2.0,
    },
    "long": {
        "label": "📈 بلندمدت (هفتگی)",
        "timeframes": ["4h","1d","1w"],
        "weights":    {"4h":1,"1d":3,"1w":4},
        "kline_limit": 200,
        "signal_threshold": 12,
        "min_agreement": 0.75,
        "hold_label": "۲-۸ هفته",
        "hold_hours": 336,
        "interval_minutes": 1440,
        "risk_pct": 3.0,
    },
}

# =========================
# LEVERAGE ENGINE
# =========================
def calc_leverage(signal_score, agreement_pct, mode_key, confidence_label, atr_pct):
    base_leverage = {
        "scalp": 5,
        "short": 3,
        "mid":   2,
        "long":  2,
    }.get(mode_key, 2)

    if agreement_pct >= 85:
        agree_factor = 1.4
    elif agreement_pct >= 78:
        agree_factor = 1.2
    elif agreement_pct >= 70:
        agree_factor = 1.0
    else:
        agree_factor = 0.6

    conf_factor = {
        "خیلی بالا 🔥🔥": 1.3,
        "بالا 🔥":         1.1,
        "متوسط ✅":        0.9,
    }.get(confidence_label, 1.0)

    if atr_pct >= 3.0:
        vol_factor = 0.5
    elif atr_pct >= 2.0:
        vol_factor = 0.7
    elif atr_pct >= 1.0:
        vol_factor = 0.9
    else:
        vol_factor = 1.1

    raw = base_leverage * agree_factor * conf_factor * vol_factor
    leverage = max(1, min(10, round(raw)))
    return leverage

def build_leverage_section(analysis, capital):
    if analysis["direction"] == "NEUTRAL ⚪" or not analysis["stop_loss"]:
        return ""

    entry    = analysis["entry"]
    sl       = analysis["stop_loss"]
    tp1      = analysis["tp1"]
    tp2      = analysis["tp2"]

    sl_pct   = abs(entry - sl) / entry * 100
    tp1_pct  = abs(tp1 - entry) / entry * 100 if tp1 else 0
    tp2_pct  = abs(tp2 - entry) / entry * 100 if tp2 else 0

    atr_pct = sl_pct / 1.5

    leverage = calc_leverage(
        signal_score    = analysis["score"],
        agreement_pct   = analysis["agreement"],
        mode_key        = analysis["mode"],
        confidence_label= analysis["confidence"],
        atr_pct         = atr_pct,
    )

    tp1_with_lev = round(tp1_pct * leverage, 2)
    tp2_with_lev = round(tp2_pct * leverage, 2)
    sl_with_lev  = round(sl_pct  * leverage, 2)

    capital_line = ""
    if capital:
        position_usd = capital * 0.20
        profit_tp1   = round(position_usd * (tp1_with_lev / 100), 2)
        profit_tp2   = round(position_usd * (tp2_with_lev / 100), 2)
        loss_sl      = round(position_usd * (sl_with_lev  / 100), 2)
        capital_line = (
            f"💵 با ۲۰٪ سرمایه (${position_usd:,.0f}):\n"
            f"  سود TP1: +${profit_tp1:,.1f}  |  سود TP2: +${profit_tp2:,.1f}\n"
            f"  زیان SL: -${loss_sl:,.1f}\n"
        )

    if leverage <= 2:
        lev_emoji = "🟢"
        lev_note  = "محافظه‌کارانه — مناسب برای مبتدی"
    elif leverage <= 5:
        lev_emoji = "🟡"
        lev_note  = "متعادل — مناسب برای معامله‌گر با تجربه"
    else:
        lev_emoji = "🔴"
        lev_note  = "ریسک بالا — فقط با استاپ‌لاس سخت"

    msg  = f"\n⚡ اهرم پیشنهادی:\n"
    msg += f"━━━━━━━━━━━━━━━\n"
    msg += f"{lev_emoji} اهرم: {leverage}x  ({lev_note})\n\n"
    msg += f"📊 سود/زیان با اهرم {leverage}x:\n"
    msg += f"  بدون اهرم → TP1: +{tp1_pct:.2f}%  |  SL: -{sl_pct:.2f}%\n"
    msg += f"  با اهرم   → TP1: +{tp1_with_lev}%  |  SL: -{sl_with_lev}%\n"
    if tp2:
        msg += f"  با اهرم   → TP2: +{tp2_with_lev}%\n"
    if capital_line:
        msg += f"\n{capital_line}"
    msg += f"\n⚠️ هشدار: اهرم سود و زیان رو هر دو چند برابر می‌کنه\n"
    msg += f"📌 همیشه استاپ‌لاس بذار — لیکویید نشی!\n"
    return msg

# =========================
# DATA
# =========================
def load_data():
    global user_data
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE,"r",encoding="utf-8") as f:
                user_data = json.load(f)
        except Exception as e:
            log.error(f"Load data error: {e}")
            user_data = {}

def save_data():
    try:
        tmp = DATA_FILE + ".tmp"
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, DATA_FILE)
    except Exception as e:
        log.error(f"Save error: {e}")

def backup_data():
    try:
        with open(DATA_FILE+".bak","w",encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Backup error: {e}")

# =========================
# USER REGISTRY
# =========================
def load_users_list():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_users_list(ul):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(ul, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Save users_list error: {e}")

def register_user(update: Update):
    user    = update.effective_user
    chat_id = str(update.effective_chat.id)
    ul      = load_users_list()
    is_new  = chat_id not in ul
    ul[chat_id] = {
        "chat_id":    chat_id,
        "first_name": user.first_name or "",
        "last_name":  user.last_name  or "",
        "username":   f"@{user.username}" if user.username else "ندارد",
        "first_seen": ul.get(chat_id, {}).get("first_seen", datetime.now().isoformat()),
        "last_seen":  datetime.now().isoformat(),
        "count":      ul.get(chat_id, {}).get("count", 0) + 1,
    }
    save_users_list(ul)
    return is_new, ul[chat_id]

def init_user(chat_id):
    if chat_id not in user_data:
        user_data[chat_id] = {
            "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT"],
            "interval": 60,
            "active": True,
            "trading_mode": "short",
            "capital": None,
            "active_positions": {},
            "signal_history": [],
            "signal_stats": {"total":0,"win":0,"loss":0,"neutral_exit":0},
            "price_alerts": {},
        }
        save_data()

# =========================
# BINANCE DATA
# =========================
async def get_klines(symbol, interval="1h", limit=150):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if len(data) < 30:
                return None
            return {
                "closes":  np.array([float(c[4]) for c in data]),
                "highs":   np.array([float(c[2]) for c in data]),
                "lows":    np.array([float(c[3]) for c in data]),
                "opens":   np.array([float(c[1]) for c in data]),
                "volumes": np.array([float(c[5]) for c in data]),
            }
    except Exception as e:
        log.error(f"get_klines {symbol}/{interval}: {e}")
        return None

async def get_ticker(symbol):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return None
            d = await resp.json()
            return {
                "price":        float(d["lastPrice"]),
                "change":       float(d["priceChangePercent"]),
                "high":         float(d["highPrice"]),
                "low":          float(d["lowPrice"]),
                "volume":       float(d["volume"]),
                "quote_volume": float(d["quoteVolume"]),
            }
    except Exception as e:
        log.error(f"get_ticker {symbol}: {e}")
        return None

async def validate_symbol(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    return symbol if await get_ticker(symbol) else None

# =========================
# INDICATORS
# =========================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def calc_rsi_series(closes, period=14):
    if len(closes) < period + 2:
        return [50.0]
    return [calc_rsi(closes[:i], period) for i in range(period + 1, len(closes) + 1)]

def calc_ema(closes, period):
    if len(closes) < period:
        return np.array([float(np.mean(closes))])
    ema = [float(np.mean(closes[:period]))]
    k = 2 / (period + 1)
    for p in closes[period:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return np.array(ema)

def calc_macd(closes):
    if len(closes) < 35:
        return None
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    mn = min(len(ema12), len(ema26))
    macd_line = ema12[-mn:] - ema26[-mn:]
    if len(macd_line) < 9:
        return None
    signal_line = calc_ema(macd_line, 9)
    hist = macd_line[-len(signal_line):] - signal_line
    cross = "none"
    if len(hist) >= 2:
        if hist[-2] < 0 and hist[-1] > 0: cross = "bullish_cross"
        elif hist[-2] > 0 and hist[-1] < 0: cross = "bearish_cross"
    return {
        "macd":      round(float(macd_line[-1]), 6),
        "signal":    round(float(signal_line[-1]), 6),
        "histogram": round(float(hist[-1]), 6),
        "prev_hist": round(float(hist[-2]), 6) if len(hist) >= 2 else 0,
        "cross":     cross,
    }

def calc_bollinger(closes, period=20):
    if len(closes) < period:
        return None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:], ddof=1)
    upper = sma + 2 * std
    lower = sma - 2 * std
    price = closes[-1]
    bb_pct = (price - lower) / (upper - lower) * 100 if upper != lower else 50
    return {
        "upper":     round(float(upper), 6),
        "mid":       round(float(sma), 6),
        "lower":     round(float(lower), 6),
        "bandwidth": round(float((upper - lower) / sma * 100), 2),
        "bb_pct":    round(float(bb_pct), 2),
    }

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return float(closes[-1] * 0.01)
    tr_list = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        for i in range(1, len(closes))
    ]
    atr = float(np.mean(tr_list[:period]))
    for v in tr_list[period:]:
        atr = (atr * (period - 1) + v) / period
    return round(atr, 6)

def calc_stochastic(highs, lows, closes, k_period=14, d_period=3, smooth_k=3):
    if len(closes) < k_period + smooth_k:
        return None
    raw_k = []
    for i in range(k_period - 1, len(closes)):
        lo = np.min(lows[i - k_period + 1:i + 1])
        hi = np.max(highs[i - k_period + 1:i + 1])
        raw_k.append(100 * (closes[i] - lo) / (hi - lo) if hi != lo else 50)
    sk = [float(np.mean(raw_k[i - smooth_k + 1:i + 1])) for i in range(smooth_k - 1, len(raw_k))]
    if len(sk) < d_period:
        return None
    d = float(np.mean(sk[-d_period:]))
    cross = "none"
    if len(sk) >= 2:
        if sk[-2] < d and sk[-1] > d: cross = "bullish"
        elif sk[-2] > d and sk[-1] < d: cross = "bearish"
    return {"k": round(sk[-1], 2), "d": round(d, 2), "cross": cross}

def calc_vwap(highs, lows, closes, volumes):
    typical = (highs + lows + closes) / 3
    tv = np.sum(volumes)
    return round(float(np.sum(typical * volumes) / tv), 6) if tv > 0 else float(closes[-1])

def calc_support_resistance(highs, lows, closes):
    pivot_highs, pivot_lows = [], []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivot_highs.append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivot_lows.append(float(lows[i]))
    price = float(closes[-1])
    support    = float(np.min(lows[-20:]))   if not pivot_lows   else min(pivot_lows[-3:])
    resistance = float(np.max(highs[-20:]))  if not pivot_highs  else max(pivot_highs[-3:])
    near_sup = max([l for l in pivot_lows  if l < price], default=support)
    near_res = min([h for h in pivot_highs if h > price], default=resistance)
    return {
        "support":         round(support, 6),
        "resistance":      round(resistance, 6),
        "near_support":    round(near_sup, 6),
        "near_resistance": round(near_res, 6),
    }

def calc_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2:
        return None
    pdm, mdm, tr_list = [], [], []
    for i in range(1, len(closes)):
        hd = highs[i] - highs[i-1]; ld = lows[i-1] - lows[i]
        pdm.append(hd if hd > ld and hd > 0 else 0)
        mdm.append(ld if ld > hd and ld > 0 else 0)
        tr_list.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
    def ws(arr, p):
        s = [sum(arr[:p])]
        for v in arr[p:]: s.append(s[-1] - s[-1] / p + v)
        return s
    trs = ws(tr_list, period); pds = ws(pdm, period); mds = ws(mdm, period)
    pdi = [100 * pds[i] / trs[i] if trs[i] != 0 else 0 for i in range(len(trs))]
    mdi = [100 * mds[i] / trs[i] if trs[i] != 0 else 0 for i in range(len(trs))]
    dx  = [100 * abs(pdi[i] - mdi[i]) / (pdi[i] + mdi[i]) if (pdi[i] + mdi[i]) != 0 else 0 for i in range(len(pdi))]
    adx_val = float(np.mean(dx[-period:]))
    return {
        "adx":   round(adx_val, 2),
        "+di":   round(pdi[-1], 2),
        "-di":   round(mdi[-1], 2),
        "trend": "strong" if adx_val > 25 else ("moderate" if adx_val > 20 else "weak"),
    }

def calc_ichimoku(highs, lows, closes):
    if len(closes) < 52: return None
    def mid(h, l): return (h + l) / 2
    tenkan   = mid(np.max(highs[-9:]),  np.min(lows[-9:]))
    kijun    = mid(np.max(highs[-26:]), np.min(lows[-26:]))
    senkou_a = (tenkan + kijun) / 2
    senkou_b = mid(np.max(highs[-52:]), np.min(lows[-52:]))
    price = closes[-1]
    sig = "neutral"
    if price > max(senkou_a, senkou_b) and tenkan > kijun:   sig = "bullish"
    elif price < min(senkou_a, senkou_b) and tenkan < kijun: sig = "bearish"
    return {
        "tenkan":   round(float(tenkan), 6),
        "kijun":    round(float(kijun), 6),
        "senkou_a": round(float(senkou_a), 6),
        "senkou_b": round(float(senkou_b), 6),
        "signal":   sig,
    }

def detect_divergence(closes, rsi_series, lookback=14):
    if not rsi_series or len(rsi_series) < lookback or len(closes) < lookback:
        return "none"
    rc = closes[-lookback:]; rr = rsi_series[-lookback:]
    ph = np.argmax(rc); pl = np.argmin(rc)
    half = len(rc) // 2
    if ph > half:
        if rr[ph] < np.max(rr[:half]) - 5: return "bearish_divergence"
    if pl > half:
        if rr[pl] > np.min(rr[:half]) + 5: return "bullish_divergence"
    return "none"

def calc_volume_profile(closes, volumes, lookback=20):
    if len(volumes) < lookback: return {"trend": "neutral", "ratio": 1.0}
    avg    = float(np.mean(volumes[-lookback*2:-lookback])) if len(volumes) >= lookback*2 else float(np.mean(volumes))
    recent = float(np.mean(volumes[-lookback:]))
    ratio  = recent / avg if avg > 0 else 1.0
    up = sum(volumes[i] for i in range(-lookback, 0) if closes[i] > closes[i-1])
    dn = sum(volumes[i] for i in range(-lookback, 0) if closes[i] < closes[i-1])
    trend = "bullish" if up > dn * 1.2 else ("bearish" if dn > up * 1.2 else "neutral")
    return {"trend": trend, "ratio": round(float(ratio), 2)}

def detect_candle_patterns(opens, closes, highs, lows):
    patterns = []
    if len(closes) < 3: return ["داده کافی نیست"]
    o, c, h, l = float(opens[-1]), float(closes[-1]), float(highs[-1]), float(lows[-1])
    body  = abs(c - o); total = h - l if h != l else 0.0001
    upper = h - max(o, c); lower = min(o, c) - l
    if body / total < 0.1: patterns.append("Doji ⚖️")
    if lower > body * 2 and upper < body * 0.3 and body / total > 0.05:
        patterns.append("Hammer 🔨" if c > o else "Hanging Man 🪢")
    if upper > body * 2 and lower < body * 0.3 and body / total > 0.05:
        patterns.append("Shooting Star ⭐" if c < o else "Inverted Hammer 🔁")
    if body / total > 0.8:
        patterns.append("Marubozu صعودی 💚" if c > o else "Marubozu نزولی 🔴")
    if len(closes) >= 2:
        po, pc = float(opens[-2]), float(closes[-2])
        if c > o and pc < po and c > po and o < pc: patterns.append("Bullish Engulfing 🟢")
        if c < o and pc > po and c < po and o > pc: patterns.append("Bearish Engulfing 🔴")
    if len(closes) >= 3:
        o3, c3 = float(opens[-3]), float(closes[-3])
        o2, c2 = float(opens[-2]), float(closes[-2])
        if c3 > o3 and abs(c2 - o2) / (highs[-2] - lows[-2] + 0.0001) < 0.3 and c < o and c < (c3 + o3) / 2:
            patterns.append("Evening Star 🌆")
        if c3 < o3 and abs(c2 - o2) / (highs[-2] - lows[-2] + 0.0001) < 0.3 and c > o and c > (c3 + o3) / 2:
            patterns.append("Morning Star 🌅")
    return patterns if patterns else ["کندل خنثی"]

# =========================
# MONEY MANAGEMENT ENGINE
# =========================
def calc_position_size(capital, entry, stop_loss, confidence, mode_key, signal_score, agreement_pct):
    mode = TRADING_MODES[mode_key]
    base_risk_pct = mode["risk_pct"]
    if agreement_pct >= 85:
        agreement_factor = 1.3
    elif agreement_pct >= 78:
        agreement_factor = 1.15
    elif agreement_pct >= 70:
        agreement_factor = 1.0
    else:
        agreement_factor = 0.7
    adjusted_risk_pct = min(base_risk_pct * agreement_factor, 5.0)
    risk_amount = capital * (adjusted_risk_pct / 100)
    sl_distance_pct = abs(entry - stop_loss) / entry * 100
    if sl_distance_pct <= 0:
        sl_distance_pct = 1.0
    position_size = risk_amount / (sl_distance_pct / 100)
    position_pct  = min((position_size / capital) * 100, 50.0)
    coin_amount   = position_size / entry if entry > 0 else 0
    return {
        "risk_pct":          round(adjusted_risk_pct, 2),
        "risk_amount":       round(risk_amount, 2),
        "position_pct":      round(position_pct, 2),
        "position_size_usd": round(position_size, 2),
        "coin_amount":       round(coin_amount, 6),
        "sl_distance_pct":   round(sl_distance_pct, 2),
        "advice":            _get_risk_advice(adjusted_risk_pct, agreement_pct, mode_key),
    }

def _get_risk_advice(risk_pct, agreement_pct, mode_key):
    advices = []
    if risk_pct <= 1.0:
        advices.append("✅ ریسک محافظه‌کارانه — مناسب برای بازار نامطمئن")
    elif risk_pct <= 2.5:
        advices.append("✅ ریسک متعادل — استاندارد برای این بازه")
    else:
        advices.append("⚠️ ریسک بالا — فقط با سرمایه‌ای که از دست دادنش مشکلی نداری")
    if agreement_pct < 75:
        advices.append("⚠️ هم‌راستایی تایم‌فریم‌ها پایینه — ورود با احتیاط")
    if mode_key == "scalp":
        advices.append("💡 اسکالپ: حتماً لیمیت اوردر بذار، نه مارکت")
    elif mode_key == "long":
        advices.append("💡 بلندمدت: DCA (خرید پله‌ای) رو در نظر بگیر")
    advices.append("📌 هرگز بیشتر از ۵٪ سرمایه کل رو در یک معامله ریسک نکن")
    return advices

def build_mm_section(capital, analysis):
    if not capital or analysis["direction"] == "NEUTRAL ⚪" or not analysis["stop_loss"]:
        return ""
    mm = calc_position_size(
        capital=capital, entry=analysis["entry"], stop_loss=analysis["stop_loss"],
        confidence=analysis["confidence"], mode_key=analysis["mode"],
        signal_score=analysis["score"], agreement_pct=analysis["agreement"],
    )
    tp1_pct = abs(analysis["tp1"] - analysis["entry"]) / analysis["entry"] * 100 if analysis["tp1"] else 0
    rr = round(tp1_pct / mm["sl_distance_pct"], 2) if mm["sl_distance_pct"] > 0 else 0
    msg  = f"\n💼 مدیریت سرمایه:\n"
    msg += f"━━━━━━━━━━━━━━━\n"
    msg += f"💰 سرمایه کل:        ${capital:,.0f}\n"
    msg += f"📊 حجم پیشنهادی:     {mm['position_pct']}٪  ≈ ${mm['position_size_usd']:,.0f}\n"
    msg += f"🎲 ریسک این معامله:  {mm['risk_pct']}٪  ≈ ${mm['risk_amount']:,.0f}\n"
    msg += f"📏 فاصله استاپ:      {mm['sl_distance_pct']}٪\n"
    msg += f"🪙 مقدار خرید:       {mm['coin_amount']} واحد\n"
    msg += f"⚖️ نسبت R/R:         1:{rr}\n"
    msg += f"\n💡 توصیه‌ها:\n"
    for a in mm["advice"]:
        msg += f"  {a}\n"
    return msg

# =========================
# TIMEFRAME ANALYSIS
# =========================
async def analyze_timeframe(symbol, tf, limit):
    kdata = await get_klines(symbol, tf, limit)
    if not kdata: return None

    closes  = kdata["closes"]; highs   = kdata["highs"]
    lows    = kdata["lows"];   opens   = kdata["opens"]
    volumes = kdata["volumes"]; price  = float(closes[-1])

    rsi        = calc_rsi(closes)
    rsi_series = calc_rsi_series(closes)
    macd       = calc_macd(closes)
    bb         = calc_bollinger(closes)
    atr        = calc_atr(highs, lows, closes)
    stoch      = calc_stochastic(highs, lows, closes)
    vwap       = calc_vwap(highs, lows, closes, volumes)
    sr         = calc_support_resistance(highs, lows, closes)
    adx        = calc_adx(highs, lows, closes)
    ichimoku   = calc_ichimoku(highs, lows, closes)
    divergence = detect_divergence(closes, rsi_series)
    vol        = calc_volume_profile(closes, volumes)
    patterns   = detect_candle_patterns(opens, closes, highs, lows)
    ema20      = float(calc_ema(closes, 20)[-1])
    ema50      = float(calc_ema(closes, 50)[-1]) if len(closes) >= 50  else None
    ema200     = float(calc_ema(closes, 200)[-1]) if len(closes) >= 200 else None

    votes = []
    def v(reason, score): votes.append((reason, score))

    if   rsi <= 30: v("RSI اشباع فروش شدید", +2)
    elif rsi <= 40: v("RSI نزدیک اشباع فروش", +1)
    elif rsi >= 70: v("RSI اشباع خرید شدید", -2)
    elif rsi >= 60: v("RSI نزدیک اشباع خرید", -1)
    else:           v("RSI خنثی", 0)

    if macd:
        if   macd["cross"] == "bullish_cross":  v("کراس صعودی MACD", +2)
        elif macd["cross"] == "bearish_cross":  v("کراس نزولی MACD", -2)
        elif macd["histogram"] > 0 and macd["prev_hist"] > 0: v("MACD هیستوگرام مثبت", +1)
        elif macd["histogram"] < 0 and macd["prev_hist"] < 0: v("MACD هیستوگرام منفی", -1)
        else: v("MACD خنثی", 0)

    if bb:
        if   bb["bb_pct"] < 10: v("زیر باند پایین بولینگر", +2)
        elif bb["bb_pct"] < 25: v("نزدیک باند پایین بولینگر", +1)
        elif bb["bb_pct"] > 90: v("بالای باند بالای بولینگر", -2)
        elif bb["bb_pct"] > 75: v("نزدیک باند بالای بولینگر", -1)
        else: v("بولینگر خنثی", 0)

    if stoch:
        if   stoch["k"] < 20 and stoch["cross"] == "bullish": v("کراس صعودی استوک در اشباع فروش", +2)
        elif stoch["k"] < 20: v("استوکستیک اشباع فروش", +1)
        elif stoch["k"] > 80 and stoch["cross"] == "bearish": v("کراس نزولی استوک در اشباع خرید", -2)
        elif stoch["k"] > 80: v("استوکستیک اشباع خرید", -1)
        else: v("استوکستیک خنثی", 0)

    if adx:
        if adx["trend"] in ("strong", "moderate"):
            w2 = 2 if adx["trend"] == "strong" else 1
            v("ADX روند صعودی قوی", +w2) if adx["+di"] > adx["-di"] else v("ADX روند نزولی قوی", -w2)
        else: v("ADX بازار رنج", 0)

    if   price > vwap * 1.002: v("قیمت بالای VWAP", +1)
    elif price < vwap * 0.998: v("قیمت زیر VWAP", -1)
    else: v("قیمت روی VWAP", 0)

    if ema50:
        if   ema20 > ema50 and price > ema20:  v("قیمت بالای EMA20>EMA50", +2)
        elif ema20 < ema50 and price < ema20:  v("قیمت زیر EMA20<EMA50", -2)
        elif price > ema20: v("قیمت بالای EMA20", +1)
        elif price < ema20: v("قیمت زیر EMA20", -1)
        else: v("EMA خنثی", 0)
    if ema200:
        v("بالای EMA200", +1) if price > ema200 else v("زیر EMA200", -1)

    if ichimoku:
        if   ichimoku["signal"] == "bullish": v("ایچیموکو صعودی", +2)
        elif ichimoku["signal"] == "bearish": v("ایچیموکو نزولی", -2)
        else: v("ایچیموکو خنثی", 0)

    if   divergence == "bullish_divergence": v("واگرایی مثبت RSI", +3)
    elif divergence == "bearish_divergence": v("واگرایی منفی RSI", -3)

    if vol["ratio"] > 1.5:
        if   vol["trend"] == "bullish": v("حجم صعودی بالا", +2)
        elif vol["trend"] == "bearish": v("حجم نزولی بالا", -2)
        else: v("حجم بالا بی‌جهت", 0)
    else: v("حجم معمولی", 0)

    dist_sup = abs(price - sr["near_support"])    / price * 100
    dist_res = abs(price - sr["near_resistance"]) / price * 100
    if   dist_sup < 1.0: v("نزدیک حمایت قوی", +2)
    elif dist_res < 1.0: v("نزدیک مقاومت قوی", -2)
    else: v("بین حمایت و مقاومت", 0)

    return {
        "tf":    tf,
        "votes": votes,
        "score": sum(sc for _, sc in votes),
        "indicators": {
            "rsi": rsi, "macd": macd, "bb": bb, "atr": atr, "stoch": stoch,
            "vwap": vwap, "sr": sr, "adx": adx, "ichimoku": ichimoku,
            "divergence": divergence, "vol": vol, "patterns": patterns,
            "ema20":  round(ema20, 6),
            "ema50":  round(ema50, 6)  if ema50  else None,
            "ema200": round(ema200, 6) if ema200 else None,
        }
    }

# =========================
# FULL ANALYSIS
# =========================
async def full_analysis(symbol, mode_key="short"):
    ticker = await get_ticker(symbol)
    if not ticker: return None

    mode  = TRADING_MODES[mode_key]
    price = ticker["price"]

    tasks       = [analyze_timeframe(symbol, tf, mode["kline_limit"]) for tf in mode["timeframes"]]
    raw_results = await asyncio.gather(*tasks)

    tf_results  = {}
    total_score = 0
    all_reasons = []

    for tf, result in zip(mode["timeframes"], raw_results):
        if not result: continue
        w = mode["weights"].get(tf, 1)
        total_score += result["score"] * w
        tf_results[tf] = result
        if w >= 2:
            for reason, vote in result["votes"]:
                if abs(vote) >= 2: all_reasons.append(f"{reason} ({tf})")

    tf_dirs = []
    for tf, result in tf_results.items():
        s = result["score"]
        tf_dirs.append(+1 if s > 2 else (-1 if s < -2 else 0))

    bullish_c = tf_dirs.count(+1)
    bearish_c = tf_dirs.count(-1)
    total_tfs = len(tf_dirs)
    agreement = max(bullish_c, bearish_c) / total_tfs if total_tfs > 0 else 0

    main_tf     = mode["timeframes"][1] if len(mode["timeframes"]) > 1 else mode["timeframes"][0]
    main_result = tf_results.get(main_tf, {})
    adx_info    = main_result.get("indicators", {}).get("adx", {}) if main_result else {}
    is_ranging  = bool(adx_info and adx_info.get("trend") == "weak")

    threshold  = mode["signal_threshold"]
    min_agree  = mode["min_agreement"]
    hold_hours = mode["hold_hours"]

    direction  = "NEUTRAL ⚪"
    confidence = "سیگنال ضعیف ⚠️"
    sl = tp1 = tp2 = tp3 = None; expiry = None

    if total_score >= threshold and agreement >= min_agree and not is_ranging:
        direction = "LONG 🟢"
    elif total_score <= -threshold and agreement >= min_agree and not is_ranging:
        direction = "SHORT 🔴"

    if direction != "NEUTRAL ⚪":
        main_ind = main_result.get("indicators", {}) if main_result else {}
        atr_v    = main_ind.get("atr", price * 0.01)
        sr_v     = main_ind.get("sr", {"near_support": price * 0.97, "near_resistance": price * 1.03})

        strength  = abs(total_score) / (threshold * 2)
        agree_pct = agreement * 100

        if agree_pct >= 85 and strength >= 1.0:
            confidence = "خیلی بالا 🔥🔥"; sl_m = 1.2; tp_m = [1.5, 3.0]
        elif agree_pct >= 75:
            confidence = "بالا 🔥";          sl_m = 1.5; tp_m = [2.0, 4.0]
        else:
            confidence = "متوسط ✅";         sl_m = 2.0; tp_m = [2.5, 5.0]

        if direction == "LONG 🟢":
            sl  = round(max(price - atr_v * sl_m, sr_v["near_support"] * 0.997), 6)
            tp1 = round(price + atr_v * tp_m[0], 6)
            tp2 = round(price + atr_v * tp_m[1], 6)
            tp3 = round(min(sr_v["near_resistance"] * 0.998, tp2 * 1.05), 6)
        else:
            sl  = round(min(price + atr_v * sl_m, sr_v["near_resistance"] * 1.003), 6)
            tp1 = round(price - atr_v * tp_m[0], 6)
            tp2 = round(price - atr_v * tp_m[1], 6)
            tp3 = round(max(sr_v["near_support"] * 1.002, tp2 * 0.95), 6)

        expiry = (datetime.now() + timedelta(hours=hold_hours)).isoformat()

    return {
        "symbol": symbol, "mode": mode_key, "mode_label": mode["label"],
        "price": price, "ticker": ticker,
        "direction": direction, "score": total_score,
        "confidence": confidence, "agreement": round(agreement * 100, 1),
        "entry": price, "stop_loss": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "hold_label": mode["hold_label"], "hold_hours": hold_hours, "expiry": expiry,
        "reasons": all_reasons[:8], "timeframes": tf_results,
        "is_ranging": is_ranging,
        "tf_agreement": {"bullish": bullish_c, "bearish": bearish_c, "neutral": tf_dirs.count(0)},
    }

# =========================
# MESSAGE BUILDERS
# =========================
def format_price(p):
    if p is None: return "N/A"
    if p >= 1000: return f"{p:,.2f}"
    if p >= 1:    return f"{p:,.4f}"
    return f"{p:,.6f}"

def build_analysis_message(a, capital=None):
    if not a: return "❌ خطا در دریافت داده"
    sym = a["symbol"]
    tv  = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}"
    now = datetime.now().strftime("%H:%M:%S")
    agg = a["tf_agreement"]

    msg  = f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 {sym} | {now}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"💵 قیمت: {format_price(a['price'])} USDT\n"
    msg += f"📈 تغییر ۲۴h: {a['ticker']['change']:+.2f}%\n"
    msg += f"🕐 بازه: {a['mode_label']}\n\n"
    msg += f"🎯 سیگنال: {a['direction']}\n"
    msg += f"💪 اطمینان: {a['confidence']}\n"
    msg += f"📐 امتیاز: {a['score']}  |  هم‌راستایی: {a['agreement']}%\n"
    msg += f"📊 تأیید TF: ✅{agg['bullish']} صعودی | ❌{agg['bearish']} نزولی | ⚪{agg['neutral']} خنثی\n"

    if a.get("is_ranging"):
        msg += f"\n⚠️ بازار رنج — از معامله خودداری کن\n"

    if a["direction"] != "NEUTRAL ⚪" and a["stop_loss"]:
        sl_pct  = abs(a["entry"] - a["stop_loss"]) / a["entry"] * 100
        tp1_pct = abs(a["tp1"] - a["entry"]) / a["entry"] * 100 if a["tp1"] else 0
        rr      = round(tp1_pct / sl_pct, 2) if sl_pct > 0 else 0

        msg += f"\n🔰 ورود:      {format_price(a['entry'])}\n"
        msg += f"🛑 Stop Loss: {format_price(a['stop_loss'])}  ({sl_pct:.2f}%)\n"
        msg += f"🎯 TP1:       {format_price(a['tp1'])}  ({tp1_pct:.2f}%)\n"
        msg += f"🎯 TP2:       {format_price(a['tp2'])}\n"
        msg += f"🎯 TP3:       {format_price(a['tp3'])}\n"
        msg += f"⚖️ R/R:       1:{rr}\n"
        if a["expiry"]:
            exp = datetime.fromisoformat(a["expiry"]).strftime("%H:%M  %Y-%m-%d")
            msg += f"⏰ نگهداری: {a['hold_label']}  |  انقضا: {exp}\n"

        msg += build_mm_section(capital, a)
        msg += build_leverage_section(a, capital)

    if a["reasons"]:
        msg += f"\n📋 دلایل اصلی:\n"
        for r in a["reasons"]:
            msg += f"  • {r}\n"

    mode    = TRADING_MODES.get(a["mode"], {})
    tfs     = mode.get("timeframes", [])
    main_tf = tfs[1] if len(tfs) > 1 else (tfs[0] if tfs else None)
    ind     = a["timeframes"].get(main_tf, {}).get("indicators", {}) if main_tf else {}

    if ind:
        stoch_v = ind.get("stoch") or {}
        macd_v  = ind.get("macd")  or {}
        adx_v   = ind.get("adx")   or {}
        bb_v    = ind.get("bb")    or {}
        ich     = ind.get("ichimoku") or {}
        div     = ind.get("divergence", "none")
        pats    = ind.get("patterns", [])
        msg += f"\n📉 اندیکاتور ({main_tf}):\n"
        msg += f"  RSI: {ind.get('rsi','N/A')}  |  Stoch K/D: {stoch_v.get('k','N/A')}/{stoch_v.get('d','N/A')}\n"
        msg += f"  MACD: {macd_v.get('macd','N/A')}  |  ADX: {adx_v.get('adx','N/A')} ({adx_v.get('trend','N/A')})\n"
        if bb_v:
            msg += f"  BB%: {bb_v.get('bb_pct','N/A')}  |  Bandwidth: {bb_v.get('bandwidth','N/A')}\n"
        if ich:
            msg += f"  Ichimoku: {ich['signal']}\n"
        if div != "none":
            msg += f"  ⚡ {div}\n"
        if pats:
            msg += f"  الگو: {', '.join(pats[:2])}\n"

    msg += f"\n🔗 {tv}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━"
    return msg

def build_price_message(symbol, ticker):
    if not ticker: return f"❌ ارز {symbol} یافت نشد"
    now   = datetime.now().strftime("%H:%M:%S")
    emoji = "📈" if ticker["change"] >= 0 else "📉"
    msg  = f"💰 قیمت لحظه‌ای\n"
    msg += f"━━━━━━━━━━━━━━\n"
    msg += f"🪙 {symbol}\n"
    msg += f"💵 قیمت:     {format_price(ticker['price'])} USDT\n"
    msg += f"{emoji} تغییر ۲۴h: {ticker['change']:+.2f}%\n"
    msg += f"🔼 بیشترین:  {format_price(ticker['high'])}\n"
    msg += f"🔽 کمترین:   {format_price(ticker['low'])}\n"
    msg += f"📦 حجم:      {ticker['volume']:,.0f}\n"
    msg += f"⏰ {now}"
    return msg

def build_history_message(history, stats):
    if not history: return "📭 هنوز هیچ سیگنالی ثبت نشده"
    total = stats["total"]; win = stats["win"]; loss = stats["loss"]
    ne    = stats.get("neutral_exit", 0)
    wr    = round(win / total * 100, 1) if total > 0 else 0
    msg  = f"📊 تاریخچه سیگنال‌ها\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"کل: {total} | ✅موفق: {win} | ❌ناموفق: {loss} | 🟡خنثی: {ne}\n"
    msg += f"نرخ موفقیت: {wr}%\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    for s in reversed(history[-10:]):
        t   = datetime.fromisoformat(s["time"]).strftime("%m/%d %H:%M")
        res = s.get("result", "در انتظار")
        msg += f"• {s['symbol']} | {s['direction']}\n"
        msg += f"  ورود: {format_price(s['entry'])} | نتیجه: {res} | {t}\n\n"
    return msg

# =========================
# MENUS
# =========================
def main_menu(chat_id):
    ud        = user_data.get(chat_id, {})
    is_active = ud.get("active", True)
    symbols   = ud.get("symbols", [])
    mode_key  = ud.get("trading_mode", "short")
    capital   = ud.get("capital")
    cap_text  = f"${capital:,.0f}" if capital else "تنظیم نشده ⚠️"
    toggle_btn = (
        InlineKeyboardButton("⏹ توقف ارسال",  callback_data="toggle_off")
        if is_active else
        InlineKeyboardButton("▶️ شروع ارسال", callback_data="toggle_on")
    )
    keyboard = [
        [InlineKeyboardButton("📊 تحلیل همین الان",    callback_data="do_analysis")],
        [InlineKeyboardButton("💰 قیمت لحظه‌ای",       callback_data="menu_price")],
        [toggle_btn],
        [
            InlineKeyboardButton("➕ افزودن ارز", callback_data="menu_add"),
            InlineKeyboardButton("➖ حذف ارز",    callback_data="menu_remove"),
        ],
        [InlineKeyboardButton("🎯 بازه معاملاتی",      callback_data="menu_mode")],
        [InlineKeyboardButton("💼 تنظیم سرمایه",       callback_data="menu_capital")],
        [InlineKeyboardButton("🔔 هشدار قیمت",         callback_data="menu_alerts")],
        [InlineKeyboardButton("📋 ارزهای فعال",        callback_data="menu_list")],
        [InlineKeyboardButton("📁 پوزیشن‌های فعال",    callback_data="menu_positions")],
        [InlineKeyboardButton("📈 تاریخچه سیگنال‌ها",  callback_data="menu_history")],
        [InlineKeyboardButton("⏱ بازه ارسال",          callback_data="menu_interval")],
    ]
    status = "🟢 فعال" if is_active else "🔴 متوقف"
    text   = (
        f"🤖 ربات تحلیل ارز دیجیتال\n"
        f"━━━━━━━━━━━━━━━\n"
        f"وضعیت: {status}\n"
        f"ارزها: {len(symbols)} عدد\n"
        f"بازه: {TRADING_MODES[mode_key]['label']}\n"
        f"سرمایه: {cap_text}\n\n"
        f"یک گزینه انتخاب کن:"
    )
    return text, InlineKeyboardMarkup(keyboard)

def price_menu(chat_id):
    symbols = user_data.get(chat_id, {}).get("symbols", [])
    keyboard = []
    row = []
    for sym in symbols:
        row.append(InlineKeyboardButton(sym.replace("USDT", ""), callback_data=f"price_{sym}"))
        if len(row) == 3: keyboard.append(row); row = []
    if row: keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔍 ارز دیگه",  callback_data="price_custom")])
    keyboard.append([InlineKeyboardButton("🔙 برگشت",      callback_data="back_main")])
    return "💰 قیمت لحظه‌ای کدوم ارز رو میخوای؟", InlineKeyboardMarkup(keyboard)

def mode_menu():
    keyboard = [[InlineKeyboardButton(v["label"], callback_data=f"setmode_{k}")] for k, v in TRADING_MODES.items()]
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    return "🎯 بازه معاملاتی خودت رو انتخاب کن:", InlineKeyboardMarkup(keyboard)

def add_symbol_menu(chat_id):
    current  = set(user_data.get(chat_id, {}).get("symbols", []))
    keyboard = []; row = []
    for sym in AVAILABLE_SYMBOLS:
        if sym not in current:
            row.append(InlineKeyboardButton(sym.replace("USDT", ""), callback_data=f"add_{sym}"))
            if len(row) == 3: keyboard.append(row); row = []
    if row: keyboard.append(row)
    keyboard.append([InlineKeyboardButton("✏️ ارز دلخواه", callback_data="add_custom")])
    keyboard.append([InlineKeyboardButton("🔙 برگشت",       callback_data="back_main")])
    return "➕ کدوم ارز رو اضافه کنی؟", InlineKeyboardMarkup(keyboard)

def remove_symbol_menu(chat_id):
    current  = user_data.get(chat_id, {}).get("symbols", [])
    keyboard = []; row = []
    for sym in current:
        row.append(InlineKeyboardButton(f"❌{sym.replace('USDT','')}", callback_data=f"rem_{sym}"))
        if len(row) == 3: keyboard.append(row); row = []
    if row: keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    return "➖ کدوم ارز رو حذف کنی؟", InlineKeyboardMarkup(keyboard)

def alerts_menu(chat_id):
    alerts  = user_data.get(chat_id, {}).get("price_alerts", {})
    symbols = user_data.get(chat_id, {}).get("symbols", [])
    keyboard = []
    for sym in symbols:
        a     = alerts.get(sym, {})
        label = sym.replace("USDT", "")
        info  = ""
        if a.get("above"): info += f"↑{format_price(a['above'])} "
        if a.get("below"): info += f"↓{format_price(a['below'])}"
        display = f"🔔{label}" + (f" [{info.strip()}]" if info else "")
        keyboard.append([InlineKeyboardButton(display, callback_data=f"alert_set_{sym}")])
    if alerts:
        keyboard.append([InlineKeyboardButton("🗑 حذف همه هشدارها", callback_data="alert_clear_all")])
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    return "🔔 هشدار قیمت — کدوم ارز رو تنظیم کنی؟", InlineKeyboardMarkup(keyboard)

# =========================
# PRICE ALERT CHECKER
# =========================
async def check_price_alerts(bot):
    for chat_id, udata in list(user_data.items()):
        alerts    = udata.get("price_alerts", {})
        triggered = []
        for sym, a in list(alerts.items()):
            ticker = await get_ticker(sym)
            if not ticker: continue
            price = ticker["price"]
            if a.get("above") and price >= a["above"]:
                try:
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=f"🔔 هشدار قیمت!\n{sym} به {format_price(price)} رسید\n✅ از {format_price(a['above'])} عبور کرد"
                    )
                except Exception as e:
                    log.error(f"Alert send error: {e}")
                triggered.append((sym, "above"))
            if a.get("below") and price <= a["below"]:
                try:
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=f"🔔 هشدار قیمت!\n{sym} به {format_price(price)} رسید\n❌ از {format_price(a['below'])} پایین‌تر رفت"
                    )
                except Exception as e:
                    log.error(f"Alert send error: {e}")
                triggered.append((sym, "below"))
        for sym, side in triggered:
            if sym in alerts:
                alerts[sym].pop(side, None)
                if not alerts[sym]:
                    alerts.pop(sym, None)
        if triggered:
            save_data()

# =========================
# POSITION TRACKER
# =========================
async def check_expired_positions(bot):
    now = datetime.now()
    for chat_id, udata in list(user_data.items()):
        positions = udata.get("active_positions", {})
        expired   = [s for s, p in positions.items() if p.get("expiry") and now >= datetime.fromisoformat(p["expiry"])]
        for sym in expired:
            try:
                pos       = positions.pop(sym)
                ticker    = await get_ticker(sym)
                if not ticker:
                    log.warning(f"No ticker for expired {sym}")
                    continue
                current   = ticker["price"]
                entry     = pos["entry"]
                direction = pos["direction"]
                tp1       = pos.get("tp1")
                sl        = pos.get("stop_loss")
                if "LONG" in direction:
                    pnl_pct = (current - entry) / entry * 100
                    hit_tp1 = tp1 and current >= tp1
                    hit_sl  = sl  and current <= sl
                else:
                    pnl_pct = (entry - current) / entry * 100
                    hit_tp1 = tp1 and current <= tp1
                    hit_sl  = sl  and current >= sl
                stats = udata.setdefault("signal_stats", {"total":0,"win":0,"loss":0,"neutral_exit":0})
                if hit_tp1:
                    result = "✅ موفق — به TP1 رسید!"; stats["win"] += 1
                elif hit_sl:
                    result = "❌ استاپ لاس خورد";       stats["loss"] += 1
                elif pnl_pct > 0:
                    result = f"🟡 سود جزئی ({pnl_pct:+.2f}%)"; stats["neutral_exit"] += 1
                else:
                    result = f"🟠 ضرر جزئی ({pnl_pct:+.2f}%)"; stats["neutral_exit"] += 1
                history = udata.setdefault("signal_history", [])
                for h in history:
                    if h.get("symbol") == sym and h.get("result") == "در انتظار":
                        h["result"] = result; h["exit_price"] = current; h["pnl_pct"] = round(pnl_pct, 2)
                        break
                await bot.send_message(
                    chat_id=int(chat_id),
                    text=(
                        f"⏰ پوزیشن {sym} منقضی شد!\n\n"
                        f"جهت: {direction}\n"
                        f"ورود: {format_price(entry)}\n"
                        f"قیمت الان: {format_price(current)}\n"
                        f"P&L: {pnl_pct:+.2f}%\n\n"
                        f"نتیجه: {result}"
                    )
                )
                log.info(f"Expired: {chat_id}/{sym} → {result}")
            except Exception as e:
                log.error(f"check_expired {chat_id}/{sym}: {e}")
        save_data()

# =========================
# CORE SENDER
# =========================
async def send_analysis(bot, chat_id, symbols):
    if not user_data.get(chat_id, {}).get("active", True): return
    mode_key = user_data.get(chat_id, {}).get("trading_mode", "short")
    capital  = user_data.get(chat_id, {}).get("capital")
    for symbol in symbols:
        try:
            a = await full_analysis(symbol, mode_key)
            if not a:
                await bot.send_message(chat_id=int(chat_id), text=f"❌ خطا در تحلیل {symbol}")
                continue
            if a["direction"] != "NEUTRAL ⚪" and a["expiry"]:
                user_data[chat_id].setdefault("active_positions", {})[symbol] = {
                    "direction": a["direction"], "entry": a["entry"],
                    "stop_loss": a["stop_loss"], "tp1":   a["tp1"],
                    "expiry":    a["expiry"],
                }
                stats = user_data[chat_id].setdefault("signal_stats", {"total":0,"win":0,"loss":0,"neutral_exit":0})
                stats["total"] += 1
                history = user_data[chat_id].setdefault("signal_history", [])
                history.append({
                    "symbol":    symbol,
                    "direction": a["direction"],
                    "entry":     a["entry"],
                    "tp1":       a["tp1"],
                    "stop_loss": a["stop_loss"],
                    "time":      datetime.now().isoformat(),
                    "result":    "در انتظار",
                    "mode":      mode_key,
                })
                if len(history) > 200: user_data[chat_id]["signal_history"] = history[-200:]
                save_data()
            await bot.send_message(
                chat_id=int(chat_id),
                text=build_analysis_message(a, capital),
                disable_web_page_preview=True
            )
        except Exception as e:
            log.error(f"send_analysis {chat_id}/{symbol}: {e}")

# =========================
# JOB SYSTEM
# =========================
def schedule_user_job(app, chat_id):
    chat_id  = str(chat_id)
    mode_key = user_data[chat_id].get("trading_mode", "short")
    interval = user_data[chat_id].get("interval") or TRADING_MODES[mode_key]["interval_minutes"]
    if chat_id in user_jobs:
        try: user_jobs[chat_id].remove()
        except: pass
    job = scheduler.add_job(
        send_analysis, "interval", minutes=interval,
        args=[app.bot, chat_id, user_data[chat_id]["symbols"]],
        id=f"user_{chat_id}", replace_existing=True
    )
    user_jobs[chat_id] = job
    log.info(f"Job scheduled: {chat_id} | mode={mode_key} | interval={interval}m")

# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    init_user(chat_id)

    is_new, uinfo = register_user(update)

    if is_new and ADMIN_ID and chat_id != ADMIN_ID:
        try:
            await context.bot.send_message(
                chat_id=int(ADMIN_ID),
                text=(
                    f"👤 کاربر جدید!\n"
                    f"━━━━━━━━━━━━━━\n"
                    f"🆔 ID:       {uinfo['chat_id']}\n"
                    f"📛 نام:      {uinfo['first_name']} {uinfo['last_name']}\n"
                    f"👤 یوزر:    {uinfo['username']}\n"
                    f"🕐 زمان:     {datetime.now().strftime('%H:%M  %Y-%m-%d')}"
                )
            )
        except Exception as e:
            log.error(f"Admin notify error: {e}")

    text, markup = main_menu(chat_id)
    await update.message.reply_text(text, reply_markup=markup)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    await query.answer()
    chat_id = str(query.message.chat.id)
    data    = query.data
    init_user(chat_id)

    if data == "toggle_off":
        user_data[chat_id]["active"] = False; save_data()
        t, m = main_menu(chat_id)
        await query.edit_message_text("⏹ ارسال خودکار متوقف شد.\n\n" + t, reply_markup=m); return

    if data == "toggle_on":
        user_data[chat_id]["active"] = True; save_data()
        schedule_user_job(context.application, chat_id)
        t, m = main_menu(chat_id)
        await query.edit_message_text("▶️ ارسال خودکار فعال شد.\n\n" + t, reply_markup=m); return

    if data == "do_analysis":
        await query.edit_message_text("⏳ در حال تحلیل چندلایه‌ای...")
        await send_analysis(context.bot, chat_id, user_data[chat_id]["symbols"])
        t, m = main_menu(chat_id)
        await context.bot.send_message(chat_id=int(chat_id), text=t, reply_markup=m); return

    if data == "menu_price":
        t, m = price_menu(chat_id); await query.edit_message_text(t, reply_markup=m); return

    if data.startswith("price_") and data != "price_custom":
        sym = data[6:]; ticker = await get_ticker(sym)
        await query.edit_message_text(build_price_message(sym, ticker))
        await asyncio.sleep(1)
        t, m = price_menu(chat_id)
        await context.bot.send_message(chat_id=int(chat_id), text=t, reply_markup=m); return

    if data == "price_custom":
        user_states[chat_id] = "waiting_price_symbol"
        await query.edit_message_text("🔍 نام ارز رو بنویس (مثلاً: BTC)\n\nبرای لغو /cancel بزن"); return

    if data == "menu_mode":
        t, m = mode_menu(); await query.edit_message_text(t, reply_markup=m); return

    if data.startswith("setmode_"):
        mk = data[8:]
        if mk in TRADING_MODES:
            user_data[chat_id]["trading_mode"] = mk
            user_data[chat_id]["interval"] = TRADING_MODES[mk]["interval_minutes"]
            save_data(); schedule_user_job(context.application, chat_id)
        t, m = main_menu(chat_id)
        await query.edit_message_text(f"✅ بازه روی {TRADING_MODES[mk]['label']} تنظیم شد.\n\n" + t, reply_markup=m); return

    if data == "menu_capital":
        cap = user_data[chat_id].get("capital")
        cur = f"سرمایه فعلی: ${cap:,.0f}" if cap else "سرمایه‌ای تنظیم نشده"
        user_states[chat_id] = "waiting_capital"
        await query.edit_message_text(
            f"💼 مدیریت سرمایه\n{cur}\n\n"
            "سرمایه کل خودت رو به دلار بنویس:\n"
            "(مثلاً: 1000 یا 5000)\n\n"
            "برای لغو /cancel بزن"
        ); return

    if data == "menu_alerts":
        t, m = alerts_menu(chat_id); await query.edit_message_text(t, reply_markup=m); return

    if data.startswith("alert_set_"):
        sym = data[10:]
        user_states[chat_id] = f"waiting_alert_{sym}"
        existing = user_data[chat_id].get("price_alerts", {}).get(sym, {})
        info = ""
        if existing.get("above"): info += f"\nهشدار بالا: {format_price(existing['above'])}"
        if existing.get("below"): info += f"\nهشدار پایین: {format_price(existing['below'])}"
        await query.edit_message_text(
            f"🔔 تنظیم هشدار برای {sym}{info}\n\n"
            "فرمت: above=50000 یا below=40000\n"
            "مثال: above=50000\n\n"
            "برای لغو /cancel بزن"
        ); return

    if data == "alert_clear_all":
        user_data[chat_id]["price_alerts"] = {}; save_data()
        t, m = alerts_menu(chat_id)
        await query.edit_message_text("✅ همه هشدارها حذف شدن.\n\n" + t, reply_markup=m); return

    if data == "menu_add":
        t, m = add_symbol_menu(chat_id); await query.edit_message_text(t, reply_markup=m); return

    if data.startswith("add_") and data != "add_custom":
        sym = data[4:]
        if sym not in user_data[chat_id]["symbols"]:
            user_data[chat_id]["symbols"].append(sym); save_data()
            schedule_user_job(context.application, chat_id)
        t, m = add_symbol_menu(chat_id)
        await query.edit_message_text(f"✅ {sym} اضافه شد!\n\n" + t, reply_markup=m); return

    if data == "add_custom":
        user_states[chat_id] = "waiting_custom_symbol"
        await query.edit_message_text("✏️ نام ارز رو بنویس (مثلاً: LINK)\n\nبرای لغو /cancel بزن"); return

    if data == "menu_remove":
        t, m = remove_symbol_menu(chat_id); await query.edit_message_text(t, reply_markup=m); return

    if data.startswith("rem_"):
        sym  = data[4:]; syms = user_data[chat_id]["symbols"]
        if len(syms) == 1:
            await query.answer("حداقل یک ارز باید فعال باشد!", show_alert=True); return
        if sym in syms:
            syms.remove(sym)
            user_data[chat_id].get("active_positions", {}).pop(sym, None)
            save_data(); schedule_user_job(context.application, chat_id)
        t, m = remove_symbol_menu(chat_id)
        await query.edit_message_text(f"🗑 {sym} حذف شد.\n\n" + t, reply_markup=m); return

    if data == "menu_list":
        syms = user_data[chat_id]["symbols"]
        text = "📋 ارزهای فعال:\n\n" + "\n".join(f"  • {s}" for s in syms)
        kb   = [[InlineKeyboardButton("🔙 برگشت", callback_data="back_main")]]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb)); return

    if data == "menu_positions":
        positions = user_data[chat_id].get("active_positions", {})
        stats     = user_data[chat_id].get("signal_stats", {"total":0,"win":0,"loss":0,"neutral_exit":0})
        text      = "📁 پوزیشن‌های فعال:\n\n" if positions else "هیچ پوزیشن فعالی نداری\n\n"
        for sym, pos in positions.items():
            exp   = datetime.fromisoformat(pos["expiry"]).strftime("%H:%M")
            text += f"• {sym} | {pos['direction']}\n"
            text += f"  ورود: {format_price(pos['entry'])} | SL: {format_price(pos['stop_loss'])}\n"
            text += f"  TP1: {format_price(pos['tp1'])} | انقضا: {exp}\n\n"
        total = stats["total"]; win = stats["win"]; loss = stats["loss"]
        wr    = round(win / total * 100, 1) if total > 0 else 0
        text += f"📊 آمار کلی: کل {total} | ✅{win} | ❌{loss} | نرخ موفقیت: {wr}%"
        kb    = [[InlineKeyboardButton("🔙 برگشت", callback_data="back_main")]]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb)); return

    if data == "menu_history":
        history = user_data[chat_id].get("signal_history", [])
        stats   = user_data[chat_id].get("signal_stats", {"total":0,"win":0,"loss":0,"neutral_exit":0})
        kb      = [[InlineKeyboardButton("🔙 برگشت", callback_data="back_main")]]
        await query.edit_message_text(build_history_message(history, stats), reply_markup=InlineKeyboardMarkup(kb)); return

    if data == "menu_interval":
        keyboard = [
            [InlineKeyboardButton("۱۵ دقیقه", callback_data="setint_15"),
             InlineKeyboardButton("۳۰ دقیقه", callback_data="setint_30"),
             InlineKeyboardButton("۱ ساعت",   callback_data="setint_60")],
            [InlineKeyboardButton("۲ ساعت",   callback_data="setint_120"),
             InlineKeyboardButton("۴ ساعت",   callback_data="setint_240"),
             InlineKeyboardButton("۸ ساعت",   callback_data="setint_480")],
            [InlineKeyboardButton("۱۲ ساعت",  callback_data="setint_720"),
             InlineKeyboardButton("۲۴ ساعت",  callback_data="setint_1440")],
            [InlineKeyboardButton("🔙 برگشت",  callback_data="back_main")],
        ]
        await query.edit_message_text("⏱ بازه ارسال خودکار رو انتخاب کن:", reply_markup=InlineKeyboardMarkup(keyboard)); return

    if data.startswith("setint_"):
        mins = int(data[7:]); user_data[chat_id]["interval"] = mins; save_data()
        schedule_user_job(context.application, chat_id)
        t, m = main_menu(chat_id)
        await query.edit_message_text(f"✅ بازه ارسال روی {mins} دقیقه تنظیم شد.\n\n" + t, reply_markup=m); return

    if data == "back_main":
        t, m = main_menu(chat_id); await query.edit_message_text(t, reply_markup=m); return

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    text    = update.message.text or ""
    state   = user_states.get(chat_id)

    if state == "waiting_price_symbol":
        user_states.pop(chat_id, None)
        sym = text.strip().upper()
        if not sym.endswith("USDT"): sym += "USDT"
        ticker = await get_ticker(sym)
        await update.message.reply_text(build_price_message(sym, ticker)); return

    if state == "waiting_capital":
        user_states.pop(chat_id, None)
        try:
            capital = float(text.strip().replace(",", "").replace("$", ""))
            if capital <= 0: raise ValueError
            user_data[chat_id]["capital"] = capital; save_data()
            t, m = main_menu(chat_id)
            await update.message.reply_text(
                f"✅ سرمایه ${capital:,.0f} ثبت شد!\n\n"
                f"از این به بعد موقع تحلیل، حجم، ریسک و اهرم مناسب پیشنهاد میدم.\n\n" + t,
                reply_markup=m
            )
        except:
            await update.message.reply_text("❌ مقدار وارد شده معتبر نیست. عدد دلاری بنویس (مثلاً: 1000)")
        return

    if state and state.startswith("waiting_alert_"):
        sym = state[14:]; user_states.pop(chat_id, None)
        try:
            parts  = text.strip().lower().split()
            alerts = user_data[chat_id].setdefault("price_alerts", {}).setdefault(sym, {})
            for part in parts:
                if "above=" in part:
                    alerts["above"] = float(part.split("=")[1])
                elif "below=" in part:
                    alerts["below"] = float(part.split("=")[1])
            save_data()
            t, m = alerts_menu(chat_id)
            await update.message.reply_text(f"✅ هشدار برای {sym} ثبت شد!\n\n" + t, reply_markup=m)
        except:
            await update.message.reply_text("❌ فرمت اشتباه. مثال: above=50000 یا below=40000")
        return

    if state == "waiting_custom_symbol":
        user_states.pop(chat_id, None)
        await update.message.reply_text("⏳ در حال بررسی ارز...")
        valid = await validate_symbol(text.strip())
        if valid:
            if valid not in user_data[chat_id]["symbols"]:
                user_data[chat_id]["symbols"].append(valid); save_data()
                schedule_user_job(context.application, chat_id)
            t, m = main_menu(chat_id)
            await update.message.reply_text(f"✅ {valid} اضافه شد!\n\n" + t, reply_markup=m)
        else:
            await update.message.reply_text(f"❌ ارز «{text}» پیدا نشد. دوباره امتحان کن.")
        return

    me = await context.bot.get_me()
    if me.username and f"@{me.username}" in text:
        init_user(chat_id)
        await update.message.reply_text("⏳ در حال تحلیل...")
        await send_analysis(context.bot, chat_id, user_data[chat_id]["symbols"])

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_states.pop(chat_id, None)
    t, m = main_menu(chat_id)
    await update.message.reply_text("لغو شد.\n\n" + t, reply_markup=m)

async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    backup_data()
    await update.message.reply_text("✅ بکاپ ساخته شد")

async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if ADMIN_ID and chat_id != ADMIN_ID:
        await update.message.reply_text("⛔ دسترسی ندارید")
        return

    ul = load_users_list()
    if not ul:
        await update.message.reply_text("📭 هنوز کاربری ثبت نشده")
        return

    total = len(ul)
    msg   = f"👥 لیست کاربران ({total} نفر)\n"
    msg  += f"━━━━━━━━━━━━━━━━━━━━\n\n"

    for i, (cid, u) in enumerate(ul.items(), 1):
        first  = datetime.fromisoformat(u["first_seen"]).strftime("%Y/%m/%d")
        last   = datetime.fromisoformat(u["last_seen"]).strftime("%m/%d %H:%M")
        name   = f"{u['first_name']} {u['last_name']}".strip() or "بدون نام"
        msg   += (
            f"{i}. {name}\n"
            f"   🆔 {cid}\n"
            f"   👤 {u['username']}\n"
            f"   📅 عضو از: {first}  |  آخرین: {last}\n"
            f"   🔢 تعداد start: {u['count']}\n\n"
        )
        if len(msg) > 3500 and i < total:
            msg += f"... و {total - i} نفر دیگه\n"
            break

    await update.message.reply_text(msg)

# =========================
# STARTUP
# =========================
async def post_init(app: Application):
    global session
    session = aiohttp.ClientSession()
    scheduler.start()
    load_data()
    for chat_id in user_data:
        if user_data[chat_id].get("active", True):
            try: schedule_user_job(app, chat_id)
            except Exception as e: log.error(f"Schedule error {chat_id}: {e}")
    scheduler.add_job(check_expired_positions, "interval", minutes=5,  args=[app.bot])
    scheduler.add_job(check_price_alerts,      "interval", minutes=2,  args=[app.bot])
    scheduler.add_job(backup_data,             "interval", minutes=10)
    log.info("✅ Bot started successfully")

def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start",  start))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("backup", backup_command))
    app.add_handler(CommandHandler("users",  users_command))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
