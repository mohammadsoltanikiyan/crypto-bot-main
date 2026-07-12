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
    raise SystemExit("❌ BOT_TOKEN تنظیم نشده.")

DATA_FILE  = os.environ.get("DATA_FILE", "users_data.json")
USERS_FILE = os.environ.get("USERS_FILE", "users_list.json")
ADMIN_ID   = os.environ.get("ADMIN_ID", "")
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
WEIGHTS_FILE = os.environ.get("WEIGHTS_FILE", "adaptive_weights.json")

CACHE_TTL_SECONDS = 120  # ۲ دقیقه — طبق نیاز کش

scheduler = AsyncIOScheduler()
session = None
user_data = {}
user_jobs = {}
user_states = {}
adaptive_weights = {}   # امتیازدهی پویا: {symbol: {category: {"win":n,"loss":n}}}

# =========================
# CACHE (۲ دقیقه‌ای)
# =========================
_cache_store = {}   # key -> (timestamp, value)
_cache_lock  = asyncio.Lock()

async def cached_call(key, coro_factory, ttl=CACHE_TTL_SECONDS):
    """
    نتیجه هر تابع async رو تا ttl ثانیه کش می‌کنه تا سرعت بره بالا و
    فشار روی صرافی‌ها (و ریسک بلاک IP) کم بشه.
    """
    now = datetime.now().timestamp()
    async with _cache_lock:
        hit = _cache_store.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = await coro_factory()
    if value is not None:   # نتیجه ناموفق (None) کش نمی‌شه تا شکست موقت رو برای کل TTL تکرار نکنه
        async with _cache_lock:
            _cache_store[key] = (now, value)
    return value

def clear_expired_cache():
    now = datetime.now().timestamp()
    expired = [k for k, (ts, _) in _cache_store.items() if now - ts > CACHE_TTL_SECONDS * 3]
    for k in expired: _cache_store.pop(k, None)

# =========================
# RATE LIMIT CONTROL — هر صرافی لیمیتر جدا داره
# =========================
class ExchangeRateLimiter:
    """
    جلوگیری از بلاک شدن IP: تعداد درخواست هم‌زمان و فاصله بین درخواست‌ها
    برای هر صرافی جدا کنترل می‌شه.
    """
    def __init__(self, max_concurrent=5, min_interval=0.12):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.min_interval = min_interval
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self.sem.acquire()
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()

    def release(self):
        self.sem.release()

RATE_LIMITERS = {
    "binance": ExchangeRateLimiter(max_concurrent=6, min_interval=0.10),
    "mexc":    ExchangeRateLimiter(max_concurrent=4, min_interval=0.15),
    "bybit":   ExchangeRateLimiter(max_concurrent=4, min_interval=0.15),
    "kucoin":  ExchangeRateLimiter(max_concurrent=4, min_interval=0.15),
    "toobit":  ExchangeRateLimiter(max_concurrent=3, min_interval=0.20),
}

async def throttled_get(exchange, url, timeout=15):
    """درخواست GET با کنترل Rate Limit اختصاصی هر صرافی."""
    limiter = RATE_LIMITERS.get(exchange)
    if limiter: await limiter.acquire()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            status = r.status
            if status == 429 or status == 418:
                log.warning(f"⛔ Rate limit از {exchange} — عقب‌نشینی")
                await asyncio.sleep(2.0)
                return None, status
            if status != 200:
                return None, status
            data = await r.json()
            return data, status
    except Exception as e:
        log.warning(f"throttled_get [{exchange}] {url}: {e}")
        return None, None
    finally:
        if limiter: limiter.release()

AVAILABLE_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","AVAXUSDT","ADAUSDT","DOTUSDT","NEARUSDT",
    "ATOMUSDT","ALGOUSDT","FTMUSDT","INJUSDT","SUIUSDT","APTUSDT",
    "UNIUSDT","AAVEUSDT","MKRUSDT","CRVUSDT","LDOUSDT",
    "BNBUSDT","OKBUSDT","DOGEUSDT","SHIBUSDT","PEPEUSDT","FLOKIUSDT",
    "XRPUSDT","TRXUSDT","XLMUSDT","LTCUSDT","LINKUSDT","FILUSDT",
    "ARUSDT","RENDERUSDT","TONUSDT","NOTUSDT","FETUSDT","AGIXUSDT",
    "WLDUSDT","SANDUSDT","MANAUSDT","AXSUSDT","IMXUSDT",
]

# =========================
# TRADING MODES — هر مود منطق جداگانه دارد
# =========================
TRADING_MODES = {
    "scalp": {
        "label": "⚡ اسکالپ (۵-۱۵ دقیقه)",
        "timeframes": ["1m", "5m", "15m"],
        "weights":    {"1m": 1, "5m": 2, "15m": 3},
        "kline_limit": 150,
        "signal_threshold": 5,
        "min_agreement": 0.65,
        "hold_label": "۱۵-۳۰ دقیقه",
        "hold_hours": 0.5,
        "interval_minutes": 15,
        "risk_pct": 0.5,
        # منطق اختصاصی اسکالپ
        "primary_indicators": ["stoch", "vwap", "bb", "volume", "candle_pattern"],
        "secondary_indicators": ["rsi", "macd"],
        "sl_atr_mult": 1.0,
        "tp_atr_mults": [1.5, 2.5],
        "require_volume_confirm": True,
        "use_market_structure": False,
    },
    "short": {
        "label": "🕐 کوتاه‌مدت (۱-۴ ساعت)",
        "timeframes": ["15m", "1h", "4h"],
        "weights":    {"15m": 1, "1h": 3, "4h": 2},
        "kline_limit": 150,
        "signal_threshold": 7,
        "min_agreement": 0.70,
        "hold_label": "۴-۱۲ ساعت",
        "hold_hours": 6,
        "interval_minutes": 60,
        "risk_pct": 1.0,
        "primary_indicators": ["rsi", "macd", "bb", "adx", "stoch"],
        "secondary_indicators": ["ichimoku", "vwap", "divergence"],
        "sl_atr_mult": 1.5,
        "tp_atr_mults": [2.0, 4.0],
        "require_volume_confirm": False,
        "use_market_structure": False,
    },
    "mid": {
        "label": "📅 میان‌مدت (روزانه)",
        "timeframes": ["1h", "4h", "1d"],
        "weights":    {"1h": 1, "4h": 2, "1d": 3},
        "kline_limit": 200,
        "signal_threshold": 9,
        "min_agreement": 0.72,
        "hold_label": "۲-۷ روز",
        "hold_hours": 72,
        "interval_minutes": 240,
        "risk_pct": 1.5,
        "primary_indicators": ["ichimoku", "ema_cross", "adx", "market_structure", "divergence"],
        "secondary_indicators": ["rsi", "macd", "volume"],
        "sl_atr_mult": 2.0,
        "tp_atr_mults": [3.0, 6.0],
        "require_volume_confirm": False,
        "use_market_structure": True,
    },
    "long": {
        "label": "📈 بلندمدت (هفتگی)",
        "timeframes": ["4h", "1d", "1w"],
        "weights":    {"4h": 1, "1d": 3, "1w": 4},
        "kline_limit": 200,
        "signal_threshold": 11,
        "min_agreement": 0.75,
        "hold_label": "۲-۸ هفته",
        "hold_hours": 336,
        "interval_minutes": 1440,
        "risk_pct": 2.0,
        "primary_indicators": ["ema200", "market_structure", "ichimoku", "volume_profile"],
        "secondary_indicators": ["rsi", "adx", "macd"],
        "sl_atr_mult": 2.5,
        "tp_atr_mults": [4.0, 8.0],
        "require_volume_confirm": False,
        "use_market_structure": True,
    },
}

# =========================
# MULTI-SOURCE API
# =========================
_TF_MEXC   = {"1m":"1m","5m":"5m","15m":"15m","1h":"60m","4h":"4h","1d":"1d","1w":"1W"}
_TF_BYBIT  = {"1m":"1","5m":"5","15m":"15","1h":"60","4h":"240","1d":"D","1w":"W"}
_TF_KUCOIN = {"1m":"1min","5m":"5min","15m":"15min","1h":"1hour","4h":"4hour","1d":"1day","1w":"1week"}

def _to_kucoin(symbol): return symbol[:-4] + "-" + symbol[-4:]

async def _klines_binance(symbol, interval, limit):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
        if r.status != 200: return None
        data = await r.json()
        if len(data) < 30: return None
        return {
            "closes":  np.array([float(c[4]) for c in data]),
            "highs":   np.array([float(c[2]) for c in data]),
            "lows":    np.array([float(c[3]) for c in data]),
            "opens":   np.array([float(c[1]) for c in data]),
            "volumes": np.array([float(c[5]) for c in data]),
        }

async def _klines_mexc(symbol, interval, limit):
    tf  = _TF_MEXC.get(interval, interval)
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={tf}&limit={limit}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
        if r.status != 200: return None
        data = await r.json()
        if len(data) < 30: return None
        return {
            "closes":  np.array([float(c[4]) for c in data]),
            "highs":   np.array([float(c[2]) for c in data]),
            "lows":    np.array([float(c[3]) for c in data]),
            "opens":   np.array([float(c[1]) for c in data]),
            "volumes": np.array([float(c[5]) for c in data]),
        }

async def _klines_bybit(symbol, interval, limit):
    tf  = _TF_BYBIT.get(interval, "60")
    url = (f"https://api.bybit.com/v5/market/kline"
           f"?category=spot&symbol={symbol}&interval={tf}&limit={limit}")
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
        if r.status != 200: return None
        d = await r.json()
        if d.get("retCode") != 0: return None
        rows = list(reversed(d["result"]["list"]))
        if len(rows) < 30: return None
        return {
            "opens":   np.array([float(c[1]) for c in rows]),
            "highs":   np.array([float(c[2]) for c in rows]),
            "lows":    np.array([float(c[3]) for c in rows]),
            "closes":  np.array([float(c[4]) for c in rows]),
            "volumes": np.array([float(c[5]) for c in rows]),
        }

async def _klines_kucoin(symbol, interval, limit):
    import time
    tf      = _TF_KUCOIN.get(interval, "1hour")
    end_ts  = int(time.time())
    secs_map = {"1min":60,"5min":300,"15min":900,"1hour":3600,"4hour":14400,"1day":86400,"1week":604800}
    step    = secs_map.get(tf, 3600)
    start_ts = end_ts - step * limit
    url = (f"https://api.kucoin.com/api/v1/market/candles"
           f"?type={tf}&symbol={_to_kucoin(symbol)}&startAt={start_ts}&endAt={end_ts}")
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
        if r.status != 200: return None
        d = await r.json()
        if d.get("code") != "200000": return None
        rows = list(reversed(d["data"]))
        if len(rows) < 30: return None
        return {
            "opens":   np.array([float(c[1]) for c in rows]),
            "closes":  np.array([float(c[2]) for c in rows]),
            "highs":   np.array([float(c[3]) for c in rows]),
            "lows":    np.array([float(c[4]) for c in rows]),
            "volumes": np.array([float(c[5]) for c in rows]),
        }

_TF_TOOBIT = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d","1w":"1w"}

async def _klines_toobit(symbol, interval, limit):
    """Toobit از فرمت اسپات مشابه Binance استفاده می‌کنه"""
    tf  = _TF_TOOBIT.get(interval, interval)
    url = f"https://api.toobit.com/api/v1/quote/klines?symbol={symbol}&interval={tf}&limit={limit}"
    limiter = RATE_LIMITERS["toobit"]; await limiter.acquire()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status != 200: return None
            data = await r.json()
            if not isinstance(data, list) or len(data) < 30: return None
            return {
                "closes":  np.array([float(c[4]) for c in data]),
                "highs":   np.array([float(c[2]) for c in data]),
                "lows":    np.array([float(c[3]) for c in data]),
                "opens":   np.array([float(c[1]) for c in data]),
                "volumes": np.array([float(c[5]) for c in data]),
            }
    except Exception as e:
        log.warning(f"_klines_toobit {symbol}: {e}")
        return None
    finally:
        limiter.release()

async def _get_klines_uncached(symbol, interval="1h", limit=150):
    sources = [
        ("Binance", _klines_binance),
        ("Toobit",  _klines_toobit),
        ("MEXC",    _klines_mexc),
        ("Bybit",   _klines_bybit),
        ("KuCoin",  _klines_kucoin),
    ]
    for name, fn in sources:
        try:
            limiter = RATE_LIMITERS.get(name.lower())
            if name.lower() != "toobit":  # toobit خودش داخل تابعش acquire می‌کنه
                if limiter: await limiter.acquire()
                try:
                    result = await fn(symbol, interval, limit)
                finally:
                    if limiter: limiter.release()
            else:
                result = await fn(symbol, interval, limit)
            if result is not None:
                if name != "Binance":
                    log.info(f"klines {symbol}/{interval} از {name}")
                return result
        except Exception as e:
            log.warning(f"get_klines [{name}] {symbol}/{interval}: {e}")
    return None

async def get_klines(symbol, interval="1h", limit=150):
    """کش ۲ دقیقه‌ای + fallback بین ۵ صرافی (Binance/Toobit/MEXC/Bybit/KuCoin)"""
    key = f"klines:{symbol}:{interval}:{limit}"
    return await cached_call(key, lambda: _get_klines_uncached(symbol, interval, limit))

async def _ticker_binance(symbol):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200: return None
        d = await r.json()
        return {"price": float(d["lastPrice"]), "change": float(d["priceChangePercent"]),
                "high": float(d["highPrice"]), "low": float(d["lowPrice"]),
                "volume": float(d["volume"]), "quote_volume": float(d["quoteVolume"])}

async def _ticker_mexc(symbol):
    url = f"https://api.mexc.com/api/v3/ticker/24hr?symbol={symbol}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200: return None
        d = await r.json()
        return {"price": float(d["lastPrice"]), "change": float(d["priceChangePercent"]),
                "high": float(d["highPrice"]), "low": float(d["lowPrice"]),
                "volume": float(d["volume"]), "quote_volume": float(d["quoteVolume"])}

async def _ticker_bybit(symbol):
    url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200: return None
        d = await r.json()
        if d.get("retCode") != 0 or not d["result"]["list"]: return None
        t = d["result"]["list"][0]
        price = float(t["lastPrice"]); prev = float(t.get("prevPrice24h") or price)
        return {"price": price, "change": round((price-prev)/prev*100 if prev else 0, 2),
                "high": float(t["highPrice24h"]), "low": float(t["lowPrice24h"]),
                "volume": float(t["volume24h"]), "quote_volume": float(t["turnover24h"])}

async def _ticker_kucoin(symbol):
    url = f"https://api.kucoin.com/api/v1/market/stats?symbol={_to_kucoin(symbol)}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200: return None
        d = await r.json()
        if d.get("code") != "200000": return None
        t = d["data"]; price = float(t["last"] or 0)
        if not price: return None
        return {"price": price, "change": round(float(t.get("changeRate") or 0)*100, 2),
                "high": float(t["high"] or price), "low": float(t["low"] or price),
                "volume": float(t["vol"] or 0), "quote_volume": float(t["volValue"] or 0)}

async def _ticker_toobit(symbol):
    url = f"https://api.toobit.com/api/v1/quote/ticker/24hr?symbol={symbol}"
    limiter = RATE_LIMITERS["toobit"]; await limiter.acquire()
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200: return None
            d = await r.json()
            if isinstance(d, list): d = d[0] if d else {}
            if not d.get("lastPrice"): return None
            return {"price": float(d["lastPrice"]), "change": float(d.get("priceChangePercent", 0)),
                    "high": float(d.get("highPrice", d["lastPrice"])), "low": float(d.get("lowPrice", d["lastPrice"])),
                    "volume": float(d.get("volume", 0)), "quote_volume": float(d.get("quoteVolume", 0))}
    except Exception as e:
        log.warning(f"_ticker_toobit {symbol}: {e}")
        return None
    finally:
        limiter.release()

async def _get_ticker_uncached(symbol):
    for name, fn in [("Binance",_ticker_binance),("Toobit",_ticker_toobit),
                      ("MEXC",_ticker_mexc),("Bybit",_ticker_bybit),("KuCoin",_ticker_kucoin)]:
        try:
            limiter = RATE_LIMITERS.get(name.lower())
            if name != "Toobit" and limiter: await limiter.acquire()
            try:
                r = await fn(symbol)
            finally:
                if name != "Toobit" and limiter: limiter.release()
            if r:
                if name != "Binance": log.info(f"ticker {symbol} از {name}")
                return r
        except Exception as e:
            log.warning(f"get_ticker [{name}] {symbol}: {e}")
    return None

async def get_ticker(symbol):
    """کش ۲ دقیقه‌ای برای قیمت لحظه‌ای"""
    return await cached_call(f"ticker:{symbol}", lambda: _get_ticker_uncached(symbol))

async def get_price_comparison(symbol):
    """
    مقایسه قیمت بین صرافی‌ها (به‌خصوص Toobit) — برای رفع اختلاف قیمت.
    میانگین صرافی‌های مرجع رو به عنوان قیمت واقعی برمی‌گردونه.
    """
    async def _fetch():
        names_fns = [("Binance",_ticker_binance),("Toobit",_ticker_toobit),
                      ("MEXC",_ticker_mexc),("Bybit",_ticker_bybit),("KuCoin",_ticker_kucoin)]
        prices = {}
        for name, fn in names_fns:
            try:
                r = await fn(symbol)
                if r and r.get("price"): prices[name] = r["price"]
            except Exception:
                pass
        if not prices: return None
        ref_prices = [p for n, p in prices.items() if n != "Toobit"]
        avg_ref = sum(ref_prices)/len(ref_prices) if ref_prices else None
        toobit_p = prices.get("Toobit")
        spread_pct = None
        if toobit_p and avg_ref:
            spread_pct = round((toobit_p - avg_ref) / avg_ref * 100, 3)
        return {"prices": prices, "avg_reference": round(avg_ref, 6) if avg_ref else None,
                "toobit_price": toobit_p, "toobit_spread_pct": spread_pct,
                "toobit_warning": bool(spread_pct is not None and abs(spread_pct) > 0.3)}
    return await cached_call(f"pricecmp:{symbol}", _fetch)

async def validate_symbol(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"): symbol += "USDT"
    return symbol if await get_ticker(symbol) else None

# =========================
# EXTERNAL DATA
# =========================
async def get_fear_greed():
    """Fear & Greed Index از alternative.me"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status != 200: return None
            d = await r.json()
            val  = int(d["data"][0]["value"])
            name = d["data"][0]["value_classification"]
            emoji = "😱" if val < 25 else ("😨" if val < 45 else ("😐" if val < 55 else ("😊" if val < 75 else "🤑")))
            return {"value": val, "label": name, "emoji": emoji}
    except Exception as e:
        log.warning(f"Fear&Greed: {e}")
        return None

async def _get_funding_rate_uncached(symbol):
    try:
        sym = symbol.replace("USDT", "") + "USDT"
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}"
        data, status = await throttled_get("binance", url, timeout=8)
        if not data: return None
        fr = float(data.get("lastFundingRate", 0)) * 100
        return {"rate": round(fr, 4), "bullish": fr < 0, "extreme": abs(fr) > 0.1}
    except Exception as e:
        log.warning(f"FundingRate {symbol}: {e}")
        return None

async def get_funding_rate(symbol):
    return await cached_call(f"funding:{symbol}", lambda: _get_funding_rate_uncached(symbol))

# =========================
# FUTURES-SPECIFIC DATA (فیوچرز اختصاصی)
# منبع: Binance USDS-M Futures API (fapi) — رایگان و بدون نیاز به کلید
# =========================

async def _get_open_interest_uncached(symbol):
    """Open Interest فعلی + تغییرات نسبت به ۶ ساعت قبل"""
    try:
        sym = symbol.replace("USDT", "") + "USDT"
        url_now  = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}"
        url_hist = (f"https://fapi.binance.com/futures/data/openInterestHist"
                    f"?symbol={sym}&period=1h&limit=6")
        now_d, _  = await throttled_get("binance", url_now, timeout=8)
        hist_d, _ = await throttled_get("binance", url_hist, timeout=8)
        if not now_d: return None
        oi_now = float(now_d.get("openInterest", 0))
        change_pct = None
        if hist_d and len(hist_d) >= 2:
            oi_old = float(hist_d[0].get("sumOpenInterest", oi_now))
            if oi_old > 0:
                change_pct = round((oi_now - oi_old) / oi_old * 100, 2)
        return {"oi": oi_now, "change_pct": change_pct,
                "rising": bool(change_pct is not None and change_pct > 2),
                "falling": bool(change_pct is not None and change_pct < -2)}
    except Exception as e:
        log.warning(f"OpenInterest {symbol}: {e}")
        return None

async def get_open_interest(symbol):
    return await cached_call(f"oi:{symbol}", lambda: _get_open_interest_uncached(symbol))

async def _get_long_short_ratio_uncached(symbol):
    """نسبت حساب‌های Long به Short (Global Account Ratio) — Binance"""
    try:
        sym = symbol.replace("USDT", "") + "USDT"
        url = (f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
               f"?symbol={sym}&period=1h&limit=1")
        data, _ = await throttled_get("binance", url, timeout=8)
        if not data: return None
        row = data[-1]
        long_pct  = float(row.get("longAccount", 0.5)) * 100
        short_pct = float(row.get("shortAccount", 0.5)) * 100
        ratio     = float(row.get("longShortRatio", 1.0))
        # اکستریم بودن یعنی احتمال اسکوییز مخالف جهت اکثریت
        extreme = ratio > 2.5 or ratio < 0.4
        return {"long_pct": round(long_pct, 1), "short_pct": round(short_pct, 1),
                "ratio": round(ratio, 2), "extreme": extreme,
                "crowd": "long" if ratio > 1 else "short"}
    except Exception as e:
        log.warning(f"LongShortRatio {symbol}: {e}")
        return None

async def get_long_short_ratio(symbol):
    return await cached_call(f"lsr:{symbol}", lambda: _get_long_short_ratio_uncached(symbol))

async def _get_orderbook_imbalance_uncached(symbol, depth=50):
    """عدم تعادل اردربوک — نسبت حجم بید به آسک در N سطح اول"""
    try:
        sym = symbol.replace("USDT", "") + "USDT"
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={sym}&limit={depth}"
        data, _ = await throttled_get("binance", url, timeout=8)
        if not data: return None
        bids = data.get("bids", []); asks = data.get("asks", [])
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        if total <= 0: return None
        imbalance_pct = round((bid_vol - ask_vol) / total * 100, 2)
        if imbalance_pct > 20:    bias = "strong_bid"
        elif imbalance_pct > 8:   bias = "bid"
        elif imbalance_pct < -20: bias = "strong_ask"
        elif imbalance_pct < -8:  bias = "ask"
        else:                     bias = "balanced"
        return {"bid_vol": round(bid_vol, 2), "ask_vol": round(ask_vol, 2),
                "imbalance_pct": imbalance_pct, "bias": bias}
    except Exception as e:
        log.warning(f"OrderBookImbalance {symbol}: {e}")
        return None

async def get_orderbook_imbalance(symbol):
    return await cached_call(f"obi:{symbol}", lambda: _get_orderbook_imbalance_uncached(symbol))

async def _get_cvd_uncached(symbol, limit=1000):
    """
    CVD واقعی (Cumulative Volume Delta) از معاملات فیوچرز اخیر Binance:
    هر ترید با isBuyerMaker=False یعنی خریدار تهاجمی بوده (تیکر بازار).
    """
    try:
        sym = symbol.replace("USDT", "") + "USDT"
        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={sym}&limit={limit}"
        data, _ = await throttled_get("binance", url, timeout=10)
        if not data: return None
        delta = 0.0; buy_vol = 0.0; sell_vol = 0.0
        for t in data:
            qty = float(t["q"])
            if t.get("m") is False:   # buyer aggressive (isBuyerMaker=False)
                buy_vol += qty; delta += qty
            else:
                sell_vol += qty; delta -= qty
        total = buy_vol + sell_vol
        strength = abs(delta) / total if total > 0 else 0
        bias = "bullish" if delta > 0 else ("bearish" if delta < 0 else "neutral")
        imbalance = "strong_bullish" if strength > 0.25 and bias == "bullish" else \
                    ("strong_bearish" if strength > 0.25 and bias == "bearish" else bias)
        return {"cvd": round(delta, 2), "buy_vol": round(buy_vol, 2), "sell_vol": round(sell_vol, 2),
                "bias": bias, "imbalance": imbalance, "strength": round(strength, 3)}
    except Exception as e:
        log.warning(f"CVD {symbol}: {e}")
        return None

async def get_cvd(symbol):
    return await cached_call(f"cvd:{symbol}", lambda: _get_cvd_uncached(symbol))

async def _get_liquidation_heatmap_uncached(symbol):
    """
    نقشه حرارتی لیکوئیدیشن — تخمینی.
    ⚠️ داده لیکوئیدیشن زنده و دقیق (مثل Coinglass) نیاز به API پولی داره.
    این تابع با ترکیب Open Interest بالا + Funding Rate افراطی + سطوح حجمی (POC/VA)
    نواحی احتمالی تجمع لیکوئیدیشن رو تخمین می‌زنه — نه داده واقعی صرافی.
    """
    try:
        ticker = await get_ticker(symbol)
        oi     = await get_open_interest(symbol)
        fr     = await get_funding_rate(symbol)
        if not ticker: return None
        price = ticker["price"]
        # هرچه Funding مثبت‌تر/منفی‌تر و OI بالاتر باشه، لیکوئیدیشن احتمالی نزدیک‌تره
        lev_zone_pct = 0.5 if (fr and fr.get("extreme")) else 1.2
        long_liq_zone  = round(price * (1 - lev_zone_pct/100 * 3), 6)   # لیکوئید لانگ‌ها زیر قیمت
        short_liq_zone = round(price * (1 + lev_zone_pct/100 * 3), 6)   # لیکوئید شورت‌ها بالای قیمت
        crowd = "long" if (fr and fr["rate"] > 0) else "short"
        return {
            "estimated": True,
            "long_liq_zone": long_liq_zone, "short_liq_zone": short_liq_zone,
            "crowded_side": crowd,
            "note": "تخمینی بر اساس OI و Funding — نه داده مستقیم صرافی"
        }
    except Exception as e:
        log.warning(f"LiquidationHeatmap {symbol}: {e}")
        return None

async def get_liquidation_heatmap(symbol):
    return await cached_call(f"liqmap:{symbol}", lambda: _get_liquidation_heatmap_uncached(symbol))

async def _get_news_sentiment_uncached(symbol):
    """
    تحلیل فاندامنتال اخبار فوری — از CryptoCompare News (رایگان)
    امتیازدهی ساده بر اساس کلیدواژه‌های مثبت/منفی در تیتر اخبار اخیر.
    """
    try:
        coin = symbol.replace("USDT", "")
        url  = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin}&lang=EN"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200: return None
            d = await r.json()
        articles = d.get("Data", [])[:15]
        if not articles: return None
        pos_kw = ["surge","rally","bullish","approval","partnership","upgrade","adoption",
                  "record high","breakout","integrat","launch","etf approv","gain"]
        neg_kw = ["hack","exploit","ban","lawsuit","crash","bearish","sec charges","delist",
                  "outflow","dump","fraud","investigation","sell-off","collapse"]
        score = 0; headlines = []
        for a in articles:
            title = (a.get("title") or "").lower()
            s = sum(1 for k in pos_kw if k in title) - sum(1 for k in neg_kw if k in title)
            score += s
            if len(headlines) < 3 and s != 0:
                headlines.append({"title": a.get("title"), "score": s, "url": a.get("url")})
        label = "مثبت 🟢" if score > 1 else ("منفی 🔴" if score < -1 else "خنثی ⚪")
        return {"score": score, "label": label, "sample": headlines, "count": len(articles)}
    except Exception as e:
        log.warning(f"NewsSentiment {symbol}: {e}")
        return None

async def get_news_sentiment(symbol):
    return await cached_call(f"news:{symbol}", lambda: _get_news_sentiment_uncached(symbol), ttl=600)

# =========================
# INDICATORS
# =========================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.mean(gains[:period]); avg_l = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_g = (avg_g*(period-1)+gains[i])/period
        avg_l = (avg_l*(period-1)+losses[i])/period
    return round(100 if avg_l == 0 else 100-(100/(1+avg_g/avg_l)), 2)

def calc_rsi_series(closes, period=14):
    if len(closes) < period+2: return [50.0]
    return [calc_rsi(closes[:i], period) for i in range(period+1, len(closes)+1)]

def calc_ema(closes, period):
    if len(closes) < period: return np.array([float(np.mean(closes))])
    ema = [float(np.mean(closes[:period]))]
    k = 2/(period+1)
    for p in closes[period:]: ema.append(p*k+ema[-1]*(1-k))
    return np.array(ema)

def calc_macd(closes):
    if len(closes) < 35: return None
    ema12 = calc_ema(closes, 12); ema26 = calc_ema(closes, 26)
    mn = min(len(ema12), len(ema26))
    ml = ema12[-mn:] - ema26[-mn:]
    if len(ml) < 9: return None
    sl = calc_ema(ml, 9); hist = ml[-len(sl):] - sl
    cross = "none"
    if len(hist) >= 2:
        if hist[-2] < 0 and hist[-1] > 0: cross = "bullish_cross"
        elif hist[-2] > 0 and hist[-1] < 0: cross = "bearish_cross"
    return {"macd": round(float(ml[-1]),6), "signal": round(float(sl[-1]),6),
            "histogram": round(float(hist[-1]),6), "prev_hist": round(float(hist[-2]),6) if len(hist)>=2 else 0,
            "cross": cross}

def calc_bollinger(closes, period=20):
    if len(closes) < period: return None
    sma = np.mean(closes[-period:]); std = np.std(closes[-period:], ddof=1)
    upper = sma+2*std; lower = sma-2*std; price = closes[-1]
    bb_pct = (price-lower)/(upper-lower)*100 if upper != lower else 50
    return {"upper": round(float(upper),6), "mid": round(float(sma),6), "lower": round(float(lower),6),
            "bandwidth": round(float((upper-lower)/sma*100),2), "bb_pct": round(float(bb_pct),2)}

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period+1: return float(closes[-1]*0.01)
    tr_list = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1,len(closes))]
    atr = float(np.mean(tr_list[:period]))
    for v in tr_list[period:]: atr = (atr*(period-1)+v)/period
    return round(atr, 6)

def calc_stochastic(highs, lows, closes, k_period=14, d_period=3, smooth_k=3):
    if len(closes) < k_period+smooth_k: return None
    raw_k = []
    for i in range(k_period-1, len(closes)):
        lo = np.min(lows[i-k_period+1:i+1]); hi = np.max(highs[i-k_period+1:i+1])
        raw_k.append(100*(closes[i]-lo)/(hi-lo) if hi != lo else 50)
    sk = [float(np.mean(raw_k[i-smooth_k+1:i+1])) for i in range(smooth_k-1, len(raw_k))]
    if len(sk) < d_period: return None
    d = float(np.mean(sk[-d_period:]))
    cross = "none"
    if len(sk) >= 2:
        if sk[-2] < d and sk[-1] > d: cross = "bullish"
        elif sk[-2] > d and sk[-1] < d: cross = "bearish"
    return {"k": round(sk[-1],2), "d": round(d,2), "cross": cross}

def calc_vwap(highs, lows, closes, volumes):
    typical = (highs+lows+closes)/3; tv = np.sum(volumes)
    return round(float(np.sum(typical*volumes)/tv),6) if tv > 0 else float(closes[-1])

def calc_support_resistance(highs, lows, closes):
    pivot_highs, pivot_lows = [], []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivot_highs.append(float(highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivot_lows.append(float(lows[i]))
    price = float(closes[-1])
    support    = float(np.min(lows[-20:]))  if not pivot_lows   else min(pivot_lows[-3:])
    resistance = float(np.max(highs[-20:])) if not pivot_highs  else max(pivot_highs[-3:])
    near_sup = max([l for l in pivot_lows  if l < price], default=support)
    near_res = min([h for h in pivot_highs if h > price], default=resistance)
    return {"support": round(support,6), "resistance": round(resistance,6),
            "near_support": round(near_sup,6), "near_resistance": round(near_res,6)}

def calc_adx(highs, lows, closes, period=14):
    if len(closes) < period*2: return None
    pdm, mdm, tr_list = [], [], []
    for i in range(1, len(closes)):
        hd = highs[i]-highs[i-1]; ld = lows[i-1]-lows[i]
        pdm.append(hd if hd>ld and hd>0 else 0)
        mdm.append(ld if ld>hd and ld>0 else 0)
        tr_list.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    def ws(arr, p):
        s = [sum(arr[:p])]
        for v in arr[p:]: s.append(s[-1]-s[-1]/p+v)
        return s
    trs=ws(tr_list,period); pds=ws(pdm,period); mds=ws(mdm,period)
    pdi=[100*pds[i]/trs[i] if trs[i]!=0 else 0 for i in range(len(trs))]
    mdi=[100*mds[i]/trs[i] if trs[i]!=0 else 0 for i in range(len(trs))]
    dx=[100*abs(pdi[i]-mdi[i])/(pdi[i]+mdi[i]) if (pdi[i]+mdi[i])!=0 else 0 for i in range(len(pdi))]
    adx_val = float(np.mean(dx[-period:]))
    return {"adx": round(adx_val,2), "+di": round(pdi[-1],2), "-di": round(mdi[-1],2),
            "trend": "strong" if adx_val>25 else ("moderate" if adx_val>20 else "weak")}

def calc_ichimoku(highs, lows, closes):
    if len(closes) < 52: return None
    def mid(h,l): return (h+l)/2
    tenkan  = mid(np.max(highs[-9:]),  np.min(lows[-9:]))
    kijun   = mid(np.max(highs[-26:]), np.min(lows[-26:]))
    sa = (tenkan+kijun)/2
    sb = mid(np.max(highs[-52:]), np.min(lows[-52:]))
    price = closes[-1]; sig = "neutral"
    if price > max(sa,sb) and tenkan > kijun:   sig = "bullish"
    elif price < min(sa,sb) and tenkan < kijun: sig = "bearish"
    tk_cross = "none"
    if len(closes) >= 2:
        prev_tk_above = highs[-2] > kijun
        curr_tk_above = tenkan > kijun
        if not prev_tk_above and curr_tk_above: tk_cross = "bullish"
        elif prev_tk_above and not curr_tk_above: tk_cross = "bearish"
    return {"tenkan": round(float(tenkan),6), "kijun": round(float(kijun),6),
            "senkou_a": round(float(sa),6), "senkou_b": round(float(sb),6),
            "signal": sig, "tk_cross": tk_cross}

def detect_divergence(closes, rsi_series, lookback=14):
    if not rsi_series or len(rsi_series) < lookback or len(closes) < lookback: return "none"
    rc = closes[-lookback:]; rr = rsi_series[-lookback:]
    ph = np.argmax(rc); pl = np.argmin(rc); half = len(rc)//2
    if ph > half and rr[ph] < np.max(rr[:half])-5: return "bearish_divergence"
    if pl > half and rr[pl] > np.min(rr[:half])+5: return "bullish_divergence"
    return "none"

def calc_volume_profile(closes, volumes, lookback=20):
    if len(volumes) < lookback: return {"trend":"neutral","ratio":1.0}
    avg    = float(np.mean(volumes[-lookback*2:-lookback])) if len(volumes)>=lookback*2 else float(np.mean(volumes))
    recent = float(np.mean(volumes[-lookback:]))
    ratio  = recent/avg if avg > 0 else 1.0
    up = sum(volumes[i] for i in range(-lookback,0) if closes[i]>closes[i-1])
    dn = sum(volumes[i] for i in range(-lookback,0) if closes[i]<closes[i-1])
    trend = "bullish" if up>dn*1.2 else ("bearish" if dn>up*1.2 else "neutral")
    return {"trend": trend, "ratio": round(float(ratio),2)}

# =========================
# FAIR VALUE GAP (FVG)
# =========================
def calc_fvg(highs, lows, closes, lookback=30):
    """
    تشخیص Fair Value Gap — ناحیه‌های پر نشده قیمتی
    FVG صعودی: low کندل سوم > high کندل اول
    FVG نزولی: high کندل سوم < low کندل اول
    """
    if len(closes) < lookback + 3: return {"bullish": [], "bearish": [], "nearest": None, "signal": "none"}
    price = float(closes[-1])
    bullish_fvgs = []
    bearish_fvgs = []
    for i in range(len(closes)-lookback, len(closes)-2):
        if i < 1: continue
        # FVG صعودی: gap بین high کندل i-1 و low کندل i+1
        if lows[i+1] > highs[i-1]:
            gap_low  = float(highs[i-1])
            gap_high = float(lows[i+1])
            mid = (gap_low + gap_high) / 2
            bullish_fvgs.append({"low": round(gap_low,6), "high": round(gap_high,6), "mid": round(mid,6), "idx": i})
        # FVG نزولی: gap بین low کندل i-1 و high کندل i+1
        if highs[i+1] < lows[i-1]:
            gap_high = float(lows[i-1])
            gap_low  = float(highs[i+1])
            mid = (gap_low + gap_high) / 2
            bearish_fvgs.append({"low": round(gap_low,6), "high": round(gap_high,6), "mid": round(mid,6), "idx": i})

    # نزدیک‌ترین FVG به قیمت فعلی
    nearest = None; nearest_dist = float("inf"); signal = "none"
    for fvg in bullish_fvgs[-5:]:
        if fvg["low"] < price:
            dist = abs(price - fvg["mid"]) / price * 100
            if dist < nearest_dist:
                nearest_dist = dist; nearest = fvg; nearest = {**fvg, "type": "bullish"}
    for fvg in bearish_fvgs[-5:]:
        if fvg["high"] > price:
            dist = abs(price - fvg["mid"]) / price * 100
            if dist < nearest_dist:
                nearest_dist = dist; nearest = {**fvg, "type": "bearish"}; nearest_dist = dist

    if nearest:
        if nearest["type"] == "bullish" and nearest_dist < 2.0: signal = "in_bullish_fvg"
        elif nearest["type"] == "bearish" and nearest_dist < 2.0: signal = "in_bearish_fvg"
        elif nearest["type"] == "bullish" and nearest_dist < 5.0: signal = "near_bullish_fvg"
        elif nearest["type"] == "bearish" and nearest_dist < 5.0: signal = "near_bearish_fvg"

    return {
        "bullish": bullish_fvgs[-3:],
        "bearish": bearish_fvgs[-3:],
        "nearest": nearest,
        "nearest_dist": round(nearest_dist, 2) if nearest_dist != float("inf") else None,
        "signal": signal,
    }

# =========================
# ORDER FLOW IMBALANCE
# =========================
def calc_order_flow(opens, closes, highs, lows, volumes, lookback=20):
    """
    تشخیص فشار خرید/فروش از روی کندل‌ها و حجم
    Delta = حجم خرید - حجم فروش (تخمین)
    """
    if len(closes) < lookback: return {"delta": 0, "bias": "neutral", "strength": 0, "cumulative": 0}
    
    deltas = []
    for i in range(-lookback, 0):
        o = float(opens[i]); c = float(closes[i])
        h = float(highs[i]); l = float(lows[i])
        v = float(volumes[i])
        body_pct = abs(c - o) / (h - l) if (h - l) > 0 else 0
        # تخمین: کندل صعودی = فشار خرید، نزولی = فشار فروش
        if c > o:
            buy_vol  = v * (0.5 + body_pct * 0.5)
            sell_vol = v * (0.5 - body_pct * 0.5)
        else:
            sell_vol = v * (0.5 + body_pct * 0.5)
            buy_vol  = v * (0.5 - body_pct * 0.5)
        deltas.append(buy_vol - sell_vol)

    cumulative = sum(deltas)
    recent_5   = sum(deltas[-5:])
    avg_abs    = float(np.mean(np.abs(deltas))) if deltas else 1

    strength = abs(recent_5) / avg_abs if avg_abs > 0 else 0
    bias = "bullish" if recent_5 > 0 else ("bearish" if recent_5 < 0 else "neutral")
    
    # imbalance شدید
    imbalance = "strong_bullish" if strength > 2.0 and bias=="bullish" else \
                ("strong_bearish" if strength > 2.0 and bias=="bearish" else bias)

    return {
        "delta":      round(float(recent_5), 2),
        "cumulative": round(float(cumulative), 2),
        "bias":       bias,
        "imbalance":  imbalance,
        "strength":   round(float(strength), 2),
    }

# =========================
# POINT OF CONTROL (POC) / VOLUME PROFILE
# =========================
def calc_poc(highs, lows, closes, volumes, lookback=50, bins=20):
    """
    Point of Control — قیمتی که بیشترین حجم معامله داشته
    این سطح حمایت/مقاومت واقعیه
    """
    if len(closes) < lookback: return None
    h = highs[-lookback:]; l = lows[-lookback:]
    v = volumes[-lookback:]; c = closes[-lookback:]
    
    price_min = float(np.min(l)); price_max = float(np.max(h))
    if price_max <= price_min: return None
    
    bin_size = (price_max - price_min) / bins
    vol_bins = np.zeros(bins)
    for i in range(len(c)):
        typical = (float(h[i]) + float(l[i]) + float(c[i])) / 3
        bin_idx = int((typical - price_min) / bin_size)
        bin_idx = min(bin_idx, bins-1)
        vol_bins[bin_idx] += float(v[i])

    poc_bin   = int(np.argmax(vol_bins))
    poc_price = price_min + (poc_bin + 0.5) * bin_size
    
    # VAH و VAL (Value Area High/Low) — ۷۰٪ حجم
    total_vol = np.sum(vol_bins)
    target    = total_vol * 0.70
    cumvol    = 0; vah_bin = poc_bin; val_bin = poc_bin
    for _ in range(bins):
        up_vol = vol_bins[vah_bin+1] if vah_bin+1 < bins else 0
        dn_vol = vol_bins[val_bin-1] if val_bin-1 >= 0 else 0
        if up_vol >= dn_vol and vah_bin+1 < bins:
            vah_bin += 1; cumvol += up_vol
        elif val_bin-1 >= 0:
            val_bin -= 1; cumvol += dn_vol
        else: break
        if cumvol >= target: break

    vah = price_min + (vah_bin + 1) * bin_size
    val = price_min + val_bin * bin_size
    price = float(closes[-1])

    return {
        "poc":       round(poc_price, 6),
        "vah":       round(vah, 6),
        "val":       round(val, 6),
        "dist_poc":  round(abs(price - poc_price) / price * 100, 2),
        "above_poc": price > poc_price,
        "in_value_area": val <= price <= vah,
    }

# =========================
# WHALE ALERT (on-chain بزرگ)
# =========================
WHALE_NOTIONAL_USD = 100_000  # آستانه دلاری برای شناسایی معامله «نهنگ»

async def _whale_trades_binance(symbol):
    sym = symbol.replace("USDT", "") + "USDT"
    url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={sym}&limit=1000"
    data, _ = await throttled_get("binance", url, timeout=10)
    if not data: return None
    big_buys=[]; big_sells=[]
    for t in data:
        notional = float(t["p"]) * float(t["q"])
        if notional >= WHALE_NOTIONAL_USD:
            (big_sells if t.get("m") else big_buys).append(notional)
    return big_buys, big_sells

async def _whale_trades_mexc(symbol):
    """فال‌بک: معاملات اسپات MEXC (فرمت مشابه Binance)"""
    sym = symbol.replace("USDT", "") + "USDT"
    url = f"https://api.mexc.com/api/v3/trades?symbol={sym}&limit=1000"
    data, _ = await throttled_get("mexc", url, timeout=10)
    if not data or not isinstance(data, list): return None
    big_buys=[]; big_sells=[]
    for t in data:
        try:
            notional = float(t["price"]) * float(t["qty"])
        except Exception:
            continue
        if notional >= WHALE_NOTIONAL_USD:
            (big_sells if t.get("isBuyerMaker") else big_buys).append(notional)
    return big_buys, big_sells

async def _whale_trades_bybit(symbol):
    """فال‌بک دوم: معاملات فیوچرز خطی Bybit"""
    sym = symbol.replace("USDT", "") + "USDT"
    url = f"https://api.bybit.com/v5/market/recent-trade?category=linear&symbol={sym}&limit=1000"
    data, _ = await throttled_get("bybit", url, timeout=10)
    if not data or not isinstance(data, dict): return None
    rows = (data.get("result") or {}).get("list") or []
    if not rows: return None
    big_buys=[]; big_sells=[]
    for t in rows:
        try:
            notional = float(t["price"]) * float(t["size"])
        except Exception:
            continue
        if notional >= WHALE_NOTIONAL_USD:
            (big_buys if t.get("side") == "Buy" else big_sells).append(notional)
    return big_buys, big_sells

async def _get_whale_alerts_uncached(symbol):
    """
    ردیابی معاملات بزرگ (نهنگ) با فال‌بک بین چند صرافی (Binance → MEXC → Bybit)
    تا در صورت غیرقابل‌دسترس بودن یکی (مثلاً محدودیت جغرافیایی)، بقیه امتحان بشن.
    ⚠️ این ردیابی حرکت‌های بزرگ داخل اردربوک صرافیه، نه رصد ولت‌های آنچین
    (رصد واقعی ولت‌ها به سرویس پولی مثل Whale Alert / Arkham نیاز داره).
    """
    for name, fn in [("Binance", _whale_trades_binance), ("MEXC", _whale_trades_mexc), ("Bybit", _whale_trades_bybit)]:
        try:
            result = await fn(symbol)
            if result is None: continue
            big_buys, big_sells = result
            buy_usd = sum(big_buys); sell_usd = sum(big_sells)
            large_tx = len(big_buys) + len(big_sells)
            net_bias = "bullish" if buy_usd > sell_usd * 1.3 else ("bearish" if sell_usd > buy_usd * 1.3 else "neutral")
            if name != "Binance": log.info(f"whale {symbol} از {name}")
            return {
                "large_tx": large_tx, "buy_usd": round(buy_usd, 0), "sell_usd": round(sell_usd, 0),
                "bullish": net_bias == "bullish", "bearish": net_bias == "bearish",
                "alert": large_tx >= 5 and net_bias != "neutral",
                "source": name,
            }
        except Exception as e:
            log.warning(f"WhaleAlert [{name}] {symbol}: {e}")
    return None

async def get_whale_alerts(symbol):
    return await cached_call(f"whale:{symbol}", lambda: _get_whale_alerts_uncached(symbol))

def detect_candle_patterns(opens, closes, highs, lows):
    patterns = []
    if len(closes) < 3: return ["داده کافی نیست"]
    o,c,h,l = float(opens[-1]),float(closes[-1]),float(highs[-1]),float(lows[-1])
    body=abs(c-o); total=h-l if h!=l else 0.0001
    upper=h-max(o,c); lower=min(o,c)-l
    if body/total < 0.1: patterns.append("Doji ⚖️")
    if lower>body*2 and upper<body*0.3 and body/total>0.05:
        patterns.append("Hammer 🔨" if c>o else "Hanging Man 🪢")
    if upper>body*2 and lower<body*0.3 and body/total>0.05:
        patterns.append("Shooting Star ⭐" if c<o else "Inverted Hammer 🔁")
    if body/total > 0.8:
        patterns.append("Marubozu صعودی 💚" if c>o else "Marubozu نزولی 🔴")
    if len(closes)>=2:
        po,pc=float(opens[-2]),float(closes[-2])
        if c>o and pc<po and c>po and o<pc: patterns.append("Bullish Engulfing 🟢")
        if c<o and pc>po and c<po and o>pc: patterns.append("Bearish Engulfing 🔴")
    if len(closes)>=3:
        o3,c3=float(opens[-3]),float(closes[-3]); o2,c2=float(opens[-2]),float(closes[-2])
        if c3>o3 and abs(c2-o2)/(highs[-2]-lows[-2]+0.0001)<0.3 and c<o and c<(c3+o3)/2:
            patterns.append("Evening Star 🌆")
        if c3<o3 and abs(c2-o2)/(highs[-2]-lows[-2]+0.0001)<0.3 and c>o and c>(c3+o3)/2:
            patterns.append("Morning Star 🌅")
    return patterns if patterns else ["کندل خنثی"]

# =========================
# MARKET STRUCTURE — جدید
# =========================
def calc_market_structure(highs, lows, closes, lookback=50):
    """تشخیص HH/HL/LL/LH برای مید و لانگ‌تم"""
    if len(closes) < lookback: return {"structure": "unknown", "bias": "neutral"}
    h = highs[-lookback:]; l = lows[-lookback:]
    # پیدا کردن swing high/low
    swing_highs = []; swing_lows = []
    for i in range(2, len(h)-2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append((i, float(h[i])))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append((i, float(l[i])))
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"structure": "unclear", "bias": "neutral"}
    hh = swing_highs[-1][1] > swing_highs[-2][1]  # Higher High
    hl = swing_lows[-1][1]  > swing_lows[-2][1]   # Higher Low
    ll = swing_lows[-1][1]  < swing_lows[-2][1]   # Lower Low
    lh = swing_highs[-1][1] < swing_highs[-2][1]  # Lower High
    if hh and hl:   structure = "HH+HL 📈"; bias = "bullish"
    elif ll and lh: structure = "LL+LH 📉"; bias = "bearish"
    elif hh and ll: structure = "HH+LL ⚡"; bias = "choppy"
    else:           structure = "رنج ↔️";   bias = "neutral"
    bos = "none"  # Break of Structure
    if len(swing_highs) >= 2 and closes[-1] > swing_highs[-2][1]: bos = "bullish_bos"
    elif len(swing_lows) >= 2 and closes[-1] < swing_lows[-2][1]: bos = "bearish_bos"
    return {"structure": structure, "bias": bias, "bos": bos,
            "last_high": round(swing_highs[-1][1],6) if swing_highs else None,
            "last_low":  round(swing_lows[-1][1],6)  if swing_lows  else None}

# =========================
# MODE-SPECIFIC SCORING ENGINES
# =========================

def score_scalp(indicators, price):
    """
    اسکالپ: Stochastic + VWAP + BB + Volume + کندل اولویت دارند
    سرعت ورود مهمه — باید فوری سیگنال بده
    """
    votes = []
    def v(r, s): votes.append((r, s))

    ind = indicators
    stoch  = ind.get("stoch")
    bb     = ind.get("bb")
    vwap   = ind.get("vwap")
    vol    = ind.get("vol")
    rsi    = ind.get("rsi", 50)
    macd   = ind.get("macd")
    pats   = ind.get("patterns", [])

    # Stochastic — مهم‌ترین برای اسکالپ
    if stoch:
        if stoch["k"] < 20 and stoch["cross"] == "bullish":   v("کراس صعودی استوک اشباع فروش", +3)
        elif stoch["k"] < 20:                                  v("استوک اشباع فروش", +2)
        elif stoch["k"] > 80 and stoch["cross"] == "bearish": v("کراس نزولی استوک اشباع خرید", -3)
        elif stoch["k"] > 80:                                  v("استوک اشباع خرید", -2)
        elif stoch["cross"] == "bullish":                      v("کراس صعودی استوک", +1)
        elif stoch["cross"] == "bearish":                      v("کراس نزولی استوک", -1)
        else:                                                   v("استوک خنثی", 0)

    # VWAP — فیلتر جهت
    if vwap:
        if price > vwap * 1.003:   v("قیمت بالای VWAP — momentum صعودی", +2)
        elif price < vwap * 0.997: v("قیمت زیر VWAP — momentum نزولی", -2)
        elif price > vwap:         v("کمی بالای VWAP", +1)
        else:                      v("کمی زیر VWAP", -1)

    # Bollinger — ورود از باند
    if bb:
        if bb["bb_pct"] < 5:   v("لمس باند پایین — bounce احتمالی", +3)
        elif bb["bb_pct"] < 15: v("نزدیک باند پایین", +2)
        elif bb["bb_pct"] > 95: v("لمس باند بالا — pullback احتمالی", -3)
        elif bb["bb_pct"] > 85: v("نزدیک باند بالا", -2)
        elif bb["bandwidth"] < 1.5: v("BB فشرده — breakout در راهه", 0)
        else:                   v("BB خنثی", 0)

    # Volume — تأیید حجم برای اسکالپ حیاتیه
    if vol:
        if vol["ratio"] > 2.0 and vol["trend"] == "bullish":   v("حجم بسیار بالا صعودی ✅", +3)
        elif vol["ratio"] > 1.5 and vol["trend"] == "bullish": v("حجم بالا صعودی", +2)
        elif vol["ratio"] > 2.0 and vol["trend"] == "bearish": v("حجم بسیار بالا نزولی ✅", -3)
        elif vol["ratio"] > 1.5 and vol["trend"] == "bearish": v("حجم بالا نزولی", -2)
        elif vol["ratio"] < 0.7: v("حجم پایین — سیگنال ضعیف ⚠️", 0)
        else:                    v("حجم معمولی", 0)

    # RSI — فقط اشباع‌ها مهمن
    if rsi <= 25:   v("RSI اشباع فروش شدید", +2)
    elif rsi >= 75: v("RSI اشباع خرید شدید", -2)
    else:           v("RSI خنثی (اسکالپ)", 0)

    # MACD — کراس
    if macd:
        if macd["cross"] == "bullish_cross":   v("کراس صعودی MACD", +1)
        elif macd["cross"] == "bearish_cross": v("کراس نزولی MACD", -1)

    # الگوهای کندل
    bullish_pats = ["Hammer 🔨","Bullish Engulfing 🟢","Morning Star 🌅","Inverted Hammer 🔁"]
    bearish_pats = ["Shooting Star ⭐","Bearish Engulfing 🔴","Evening Star 🌆","Hanging Man 🪢"]
    for p in pats:
        if p in bullish_pats: v(f"الگو {p}", +2)
        elif p in bearish_pats: v(f"الگو {p}", -2)

    # Order Flow — برای اسکالپ خیلی مهمه
    of = ind.get("order_flow")
    if of:
        if of["imbalance"] == "strong_bullish": v("Order Flow: فشار خرید قوی 🟢", +3)
        elif of["imbalance"] == "strong_bearish": v("Order Flow: فشار فروش قوی 🔴", -3)
        elif of["bias"] == "bullish": v("Order Flow: فشار خرید", +1)
        elif of["bias"] == "bearish": v("Order Flow: فشار فروش", -1)

    # FVG — نواحی جذب قیمت
    fvg = ind.get("fvg")
    if fvg:
        sig = fvg.get("signal","none")
        if sig == "in_bullish_fvg":    v("قیمت داخل FVG صعودی — bounce احتمالی", +3)
        elif sig == "near_bullish_fvg": v("نزدیک FVG صعودی", +2)
        elif sig == "in_bearish_fvg":   v("قیمت داخل FVG نزولی — rejection احتمالی", -3)
        elif sig == "near_bearish_fvg": v("نزدیک FVG نزولی", -2)

    # POC — قیمت نسبت به POC
    poc = ind.get("poc")
    if poc:
        if poc["dist_poc"] < 0.5:
            v("قیمت روی POC — ناحیه بحرانی ⚡", +1 if poc["above_poc"] else -1)
        elif poc["above_poc"] and poc["dist_poc"] < 2.0:
            v("کمی بالای POC — حمایت قوی زیر", +1)
        elif not poc["above_poc"] and poc["dist_poc"] < 2.0:
            v("کمی زیر POC — مقاومت قوی بالا", -1)

    return votes

def score_short(indicators, price):
    """
    کوتاه‌مدت: RSI + MACD + BB + ADX + Stoch
    تعادل بین سرعت و اطمینان
    """
    votes = []
    def v(r, s): votes.append((r, s))

    rsi    = indicators.get("rsi", 50)
    macd   = indicators.get("macd")
    bb     = indicators.get("bb")
    stoch  = indicators.get("stoch")
    adx    = indicators.get("adx")
    vwap   = indicators.get("vwap")
    div    = indicators.get("divergence", "none")
    ich    = indicators.get("ichimoku")
    ema20  = indicators.get("ema20")
    ema50  = indicators.get("ema50")
    vol    = indicators.get("vol")

    # RSI — مهم‌ترین
    if   rsi <= 30: v("RSI اشباع فروش شدید", +2)
    elif rsi <= 40: v("RSI نزدیک اشباع فروش", +1)
    elif rsi >= 70: v("RSI اشباع خرید شدید", -2)
    elif rsi >= 60: v("RSI نزدیک اشباع خرید", -1)
    else:           v("RSI خنثی", 0)

    # MACD
    if macd:
        if   macd["cross"] == "bullish_cross":  v("کراس صعودی MACD", +3)
        elif macd["cross"] == "bearish_cross":  v("کراس نزولی MACD", -3)
        elif macd["histogram"] > 0 and macd["prev_hist"] > 0: v("MACD هیستوگرام مثبت رو به رشد", +1)
        elif macd["histogram"] < 0 and macd["prev_hist"] < 0: v("MACD هیستوگرام منفی رو به افت", -1)
        else: v("MACD خنثی", 0)

    # Bollinger
    if bb:
        if   bb["bb_pct"] < 10: v("زیر باند پایین BB", +2)
        elif bb["bb_pct"] < 25: v("نزدیک باند پایین BB", +1)
        elif bb["bb_pct"] > 90: v("بالای باند بالای BB", -2)
        elif bb["bb_pct"] > 75: v("نزدیک باند بالای BB", -1)
        else: v("BB خنثی", 0)

    # Stochastic
    if stoch:
        if stoch["k"] < 20 and stoch["cross"] == "bullish": v("کراس صعودی استوک اشباع فروش", +2)
        elif stoch["k"] > 80 and stoch["cross"] == "bearish": v("کراس نزولی استوک اشباع خرید", -2)
        elif stoch["k"] < 20: v("استوک اشباع فروش", +1)
        elif stoch["k"] > 80: v("استوک اشباع خرید", -1)
        else: v("استوک خنثی", 0)

    # ADX — قدرت روند
    if adx:
        if adx["adx"] > 25:
            w = 2 if adx["adx"] > 35 else 1
            if adx["+di"] > adx["-di"]: v(f"ADX روند صعودی ({adx['adx']:.0f})", +w)
            else:                        v(f"ADX روند نزولی ({adx['adx']:.0f})", -w)
        else: v("ADX بازار ضعیف/رنج", 0)

    # VWAP
    if vwap:
        if price > vwap*1.002: v("بالای VWAP", +1)
        elif price < vwap*0.998: v("زیر VWAP", -1)

    # EMA Cross
    if ema20 and ema50:
        if ema20 > ema50 and price > ema20:  v("EMA20>EMA50 قیمت بالا", +2)
        elif ema20 < ema50 and price < ema20: v("EMA20<EMA50 قیمت پایین", -2)
        elif price > ema20: v("بالای EMA20", +1)
        else: v("زیر EMA20", -1)

    # Divergence
    if div == "bullish_divergence":  v("واگرایی مثبت RSI", +3)
    elif div == "bearish_divergence": v("واگرایی منفی RSI", -3)

    # Volume
    if vol and vol["ratio"] > 1.5:
        if vol["trend"] == "bullish": v("حجم صعودی بالا", +1)
        elif vol["trend"] == "bearish": v("حجم نزولی بالا", -1)

    # Order Flow
    of = indicators.get("order_flow")
    if of:
        if of["imbalance"] == "strong_bullish": v("Order Flow: فشار خرید قوی", +2)
        elif of["imbalance"] == "strong_bearish": v("Order Flow: فشار فروش قوی", -2)
        elif of["bias"] == "bullish" and of["strength"] > 1.2: v("Order Flow: خریداران غالب", +1)
        elif of["bias"] == "bearish" and of["strength"] > 1.2: v("Order Flow: فروشندگان غالب", -1)

    # FVG
    fvg = indicators.get("fvg")
    if fvg:
        sig = fvg.get("signal","none")
        if sig == "in_bullish_fvg":    v("قیمت داخل FVG صعودی", +2)
        elif sig == "near_bullish_fvg": v("نزدیک FVG صعودی", +1)
        elif sig == "in_bearish_fvg":   v("قیمت داخل FVG نزولی", -2)
        elif sig == "near_bearish_fvg": v("نزدیک FVG نزولی", -1)

    # POC
    poc = indicators.get("poc")
    if poc:
        if poc["dist_poc"] < 1.0:
            v("قیمت روی POC ⚡", +1 if poc["above_poc"] else -1)
        elif poc["above_poc"]: v("بالای POC — حمایت زیر", +1)
        else: v("زیر POC — مقاومت بالا", -1)

    return votes

def score_mid(indicators, price):
    """
    میان‌مدت: Ichimoku + EMA + Market Structure + ADX + Divergence
    نیاز به تأیید چند ابزار همزمان
    """
    votes = []
    def v(r, s): votes.append((r, s))

    ich  = indicators.get("ichimoku")
    adx  = indicators.get("adx")
    div  = indicators.get("divergence", "none")
    ms   = indicators.get("market_structure", {})
    ema20 = indicators.get("ema20")
    ema50 = indicators.get("ema50")
    ema200 = indicators.get("ema200")
    rsi  = indicators.get("rsi", 50)
    macd = indicators.get("macd")
    vol  = indicators.get("vol")
    sr   = indicators.get("sr")

    # Ichimoku — مهم‌ترین برای میان‌مدت
    if ich:
        if ich["signal"] == "bullish":
            w = 3
            if ich["tk_cross"] == "bullish": w += 1
            v(f"ایچیموکو صعودی (TK Cross: {ich['tk_cross']})", +w)
        elif ich["signal"] == "bearish":
            w = 3
            if ich["tk_cross"] == "bearish": w += 1
            v(f"ایچیموکو نزولی (TK Cross: {ich['tk_cross']})", -w)
        else:
            v("ایچیموکو داخل کومو (خنثی)", 0)

    # Market Structure
    if ms:
        if ms["bias"] == "bullish":     v(f"ساختار بازار: {ms['structure']}", +3)
        elif ms["bias"] == "bearish":   v(f"ساختار بازار: {ms['structure']}", -3)
        elif ms["bias"] == "choppy":    v("ساختار چاپی — احتیاط ⚠️", 0)
        if ms.get("bos") == "bullish_bos": v("شکست ساختار صعودی (BOS)", +2)
        elif ms.get("bos") == "bearish_bos": v("شکست ساختار نزولی (BOS)", -2)

    # EMA Stack
    if ema200:
        v("بالای EMA200 — روند کلی صعودی", +2) if price > ema200 else v("زیر EMA200 — روند کلی نزولی", -2)
    if ema50 and ema20:
        if ema20 > ema50 and price > ema50: v("EMA20>EMA50 قیمت بالا", +2)
        elif ema20 < ema50 and price < ema50: v("EMA20<EMA50 قیمت پایین", -2)

    # ADX
    if adx and adx["adx"] > 20:
        w = 2 if adx["adx"] > 30 else 1
        if adx["+di"] > adx["-di"]: v(f"ADX قوی صعودی ({adx['adx']:.0f})", +w)
        else:                        v(f"ADX قوی نزولی ({adx['adx']:.0f})", -w)

    # Divergence — خیلی مهم برای میان‌مدت
    if div == "bullish_divergence":   v("واگرایی مثبت RSI", +4)
    elif div == "bearish_divergence": v("واگرایی منفی RSI", -4)

    # RSI
    if   rsi <= 35: v("RSI اشباع فروش", +2)
    elif rsi >= 65: v("RSI اشباع خرید", -2)

    # MACD
    if macd:
        if macd["cross"] == "bullish_cross": v("کراس صعودی MACD", +2)
        elif macd["cross"] == "bearish_cross": v("کراس نزولی MACD", -2)

    # Volume Profile
    if vol and vol["ratio"] > 1.3:
        if vol["trend"] == "bullish": v("حجم صعودی بالا", +1)
        elif vol["trend"] == "bearish": v("حجم نزولی بالا", -1)

    # نزدیک حمایت/مقاومت
    if sr:
        dist_sup = abs(price-sr["near_support"])/price*100
        dist_res = abs(price-sr["near_resistance"])/price*100
        if dist_sup < 2.0: v("نزدیک حمایت میان‌مدت", +2)
        elif dist_res < 2.0: v("نزدیک مقاومت میان‌مدت", -2)

    # FVG — برای میان‌مدت ناحیه supply/demand مهمه
    fvg = indicators.get("fvg")
    if fvg:
        sig = fvg.get("signal","none")
        if sig in ("in_bullish_fvg","near_bullish_fvg"):   v("FVG صعودی — ناحیه تقاضا", +2)
        elif sig in ("in_bearish_fvg","near_bearish_fvg"): v("FVG نزولی — ناحیه عرضه", -2)

    # POC — برای میان‌مدت خیلی مهمه
    poc = indicators.get("poc")
    if poc:
        if poc["in_value_area"]:
            v("قیمت داخل Value Area — احتمال رنج", 0)
        elif poc["above_poc"]:
            v("بالای POC و Value Area — روند صعودی تأیید", +2)
        else:
            v("زیر POC و Value Area — روند نزولی تأیید", -2)

    # Order Flow
    of = indicators.get("order_flow")
    if of and of["strength"] > 1.5:
        if of["bias"] == "bullish": v("Order Flow میان‌مدت: خریداران غالب", +2)
        elif of["bias"] == "bearish": v("Order Flow میان‌مدت: فروشندگان غالب", -2)

    return votes

def score_long(indicators, price):
    """
    بلندمدت: EMA200 + Market Structure + Ichimoku + Volume Profile
    فقط تأییدهای قوی — سیگنال کمتر ولی با اطمینان بالاتر
    """
    votes = []
    def v(r, s): votes.append((r, s))

    ema200 = indicators.get("ema200")
    ema50  = indicators.get("ema50")
    ema20  = indicators.get("ema20")
    ms     = indicators.get("market_structure", {})
    ich    = indicators.get("ichimoku")
    vol    = indicators.get("vol")
    adx    = indicators.get("adx")
    rsi    = indicators.get("rsi", 50)
    div    = indicators.get("divergence", "none")
    macd   = indicators.get("macd")

    # EMA200 — مهم‌ترین فیلتر برای بلندمدت
    if ema200:
        diff_pct = (price - ema200) / ema200 * 100
        if diff_pct > 5:    v(f"بالای EMA200 ({diff_pct:.1f}%) — روند صعودی قوی", +4)
        elif diff_pct > 0:  v("کمی بالای EMA200", +2)
        elif diff_pct > -5: v("کمی زیر EMA200", -2)
        else:               v(f"زیر EMA200 ({diff_pct:.1f}%) — روند نزولی قوی", -4)

    # EMA Stack (Golden/Death Cross)
    if ema50 and ema200:
        if ema50 > ema200: v("Golden Cross — EMA50>EMA200 ✨", +3)
        else:              v("Death Cross — EMA50<EMA200 💀", -3)

    # Market Structure — برای بلندمدت خیلی مهمه
    if ms:
        if ms["bias"] == "bullish":   v(f"ساختار بلندمدت: {ms['structure']}", +4)
        elif ms["bias"] == "bearish": v(f"ساختار بلندمدت: {ms['structure']}", -4)
        if ms.get("bos") == "bullish_bos": v("BOS صعودی بلندمدت", +3)
        elif ms.get("bos") == "bearish_bos": v("BOS نزولی بلندمدت", -3)

    # Ichimoku
    if ich:
        if ich["signal"] == "bullish":   v("ایچیموکو بلندمدت صعودی", +3)
        elif ich["signal"] == "bearish": v("ایچیموکو بلندمدت نزولی", -3)

    # Volume Profile
    if vol:
        if vol["ratio"] > 1.5 and vol["trend"] == "bullish": v("حجم بلندمدت صعودی", +2)
        elif vol["ratio"] > 1.5 and vol["trend"] == "bearish": v("حجم بلندمدت نزولی", -2)
        elif vol["trend"] == "bullish": v("جریان حجم صعودی", +1)
        elif vol["trend"] == "bearish": v("جریان حجم نزولی", -1)

    # ADX
    if adx and adx["adx"] > 25:
        if adx["+di"] > adx["-di"]: v(f"روند قوی صعودی ADX={adx['adx']:.0f}", +2)
        else:                        v(f"روند قوی نزولی ADX={adx['adx']:.0f}", -2)

    # Divergence
    if div == "bullish_divergence":   v("واگرایی مثبت بلندمدت", +4)
    elif div == "bearish_divergence": v("واگرایی منفی بلندمدت", -4)

    # RSI
    if   rsi <= 40: v("RSI اشباع فروش بلندمدت", +2)
    elif rsi >= 60: v("RSI اشباع خرید بلندمدت", -2)

    # MACD
    if macd:
        if macd["cross"] == "bullish_cross": v("کراس صعودی MACD هفتگی", +2)
        elif macd["cross"] == "bearish_cross": v("کراس نزولی MACD هفتگی", -2)
        elif macd["histogram"] > 0 and macd["macd"] > 0: v("MACD مثبت در ناحیه مثبت", +1)
        elif macd["histogram"] < 0 and macd["macd"] < 0: v("MACD منفی در ناحیه منفی", -1)

    # POC بلندمدت — تأیید روند با ساختار حجمی
    poc = indicators.get("poc")
    if poc:
        if not poc["in_value_area"] and poc["above_poc"]:
            v("بالای Value Area — روند صعودی بلندمدت قوی", +3)
        elif not poc["in_value_area"] and not poc["above_poc"]:
            v("زیر Value Area — روند نزولی بلندمدت قوی", -3)

    # FVG بلندمدت
    fvg = indicators.get("fvg")
    if fvg:
        sig = fvg.get("signal","none")
        if sig in ("in_bullish_fvg","near_bullish_fvg"):   v("FVG صعودی بلندمدت — ناحیه انباشت", +2)
        elif sig in ("in_bearish_fvg","near_bearish_fvg"): v("FVG نزولی بلندمدت — ناحیه توزیع", -2)

    # Order Flow بلندمدت
    of = indicators.get("order_flow")
    if of and of["strength"] > 2.0:
        if of["bias"] == "bullish": v("Order Flow بلندمدت: انباشت نهنگ‌ها 🐋", +3)
        elif of["bias"] == "bearish": v("Order Flow بلندمدت: توزیع نهنگ‌ها 🐋", -3)

    return votes

SCORE_ENGINES = {
    "scalp": score_scalp,
    "short": score_short,
    "mid":   score_mid,
    "long":  score_long,
}

# =========================
# TIMEFRAME ANALYSIS
# =========================
async def analyze_timeframe(symbol, tf, limit, mode_key="short"):
    kdata = await get_klines(symbol, tf, limit)
    if not kdata: return None
    closes=kdata["closes"]; highs=kdata["highs"]
    lows=kdata["lows"];     opens=kdata["opens"]
    volumes=kdata["volumes"]; price=float(closes[-1])

    rsi        = calc_rsi(closes)
    rsi_series = calc_rsi_series(closes)
    macd       = calc_macd(closes)
    bb         = calc_bollinger(closes)
    atr        = calc_atr(highs, lows, closes)
    stoch      = calc_stochastic(highs, lows, closes)
    vwap_val   = calc_vwap(highs, lows, closes, volumes)
    sr         = calc_support_resistance(highs, lows, closes)
    adx        = calc_adx(highs, lows, closes)
    ichimoku   = calc_ichimoku(highs, lows, closes)
    divergence = detect_divergence(closes, rsi_series)
    vol        = calc_volume_profile(closes, volumes)
    patterns   = detect_candle_patterns(opens, closes, highs, lows)
    ema20      = float(calc_ema(closes, 20)[-1])
    ema50      = float(calc_ema(closes, 50)[-1])  if len(closes) >= 50  else None
    ema200     = float(calc_ema(closes, 200)[-1]) if len(closes) >= 200 else None
    ms         = calc_market_structure(highs, lows, closes) if TRADING_MODES[mode_key].get("use_market_structure") else {}
    fvg        = calc_fvg(highs, lows, closes)
    order_flow = calc_order_flow(opens, closes, highs, lows, volumes)
    poc        = calc_poc(highs, lows, closes, volumes)

    ind = {
        "rsi": rsi, "macd": macd, "bb": bb, "atr": atr, "stoch": stoch,
        "vwap": vwap_val, "sr": sr, "adx": adx, "ichimoku": ichimoku,
        "divergence": divergence, "vol": vol, "patterns": patterns,
        "ema20": round(ema20,6),
        "ema50": round(ema50,6)  if ema50  else None,
        "ema200": round(ema200,6) if ema200 else None,
        "market_structure": ms,
        "fvg":        fvg,
        "order_flow": order_flow,
        "poc":        poc,
    }

    # از موتور مناسب استفاده کن
    engine = SCORE_ENGINES.get(mode_key, score_short)
    votes  = engine(ind, price)
    weighted_votes = apply_dynamic_weights(symbol, votes)   # امتیازدهی پویا اعمال می‌شه

    return {
        "tf": tf,
        "votes": votes,
        "weighted_votes": weighted_votes,
        "categories": sorted(set(c for _, _, c in weighted_votes)),
        "score": round(sum(sc for _, sc, _ in weighted_votes), 3),
        "indicators": ind
    }

# =========================
# FULL ANALYSIS
# =========================
async def full_analysis(symbol, mode_key="short"):
    ticker = await get_ticker(symbol)
    if not ticker: return None

    mode  = TRADING_MODES[mode_key]
    price = ticker["price"]

    tasks = [analyze_timeframe(symbol, tf, mode["kline_limit"], mode_key) for tf in mode["timeframes"]]
    futures_tasks = [get_funding_rate(symbol), get_open_interest(symbol),
                      get_long_short_ratio(symbol), get_orderbook_imbalance(symbol), get_cvd(symbol)]
    n_tf = len(tasks)
    gathered = await asyncio.gather(*tasks, *futures_tasks)
    raw_results = gathered[:n_tf]
    fr_d, oi_d, lsr_d, obi_d, cvd_d = gathered[n_tf:]

    tf_results = {}; total_score = 0; all_reasons = []; all_categories = set()

    for tf, result in zip(mode["timeframes"], raw_results):
        if not result: continue
        w = mode["weights"].get(tf, 1)
        total_score += result["score"] * w
        tf_results[tf] = result
        all_categories.update(result.get("categories", []))
        for reason, vote in result["votes"]:
            if abs(vote) >= 2: all_reasons.append(f"{reason} ({tf})")

    # امتیاز اضافه از دیتای اختصاصی فیوچرز (Funding/OI/LSR/OBI/CVD)
    futures_score = 0
    if fr_d and fr_d.get("extreme"):
        futures_score += 2 if fr_d["bullish"] else -2
        all_reasons.append(f"Funding Rate افراطی ({fr_d['rate']}%)")
    if oi_d:
        if oi_d.get("rising") and total_score > 0: futures_score += 1
        elif oi_d.get("rising") and total_score < 0: futures_score -= 1
    if lsr_d and lsr_d.get("extreme"):
        # اکثریت شدید یک سمت → احتمال اسکوییز در جهت مخالف جمع
        futures_score += -1.5 if lsr_d["crowd"] == "long" else 1.5
        all_reasons.append(f"ازدحام {('لانگ' if lsr_d['crowd']=='long' else 'شورت')} — ریسک اسکوییز مخالف")
    if obi_d and obi_d["bias"] in ("strong_bid", "strong_ask"):
        futures_score += 1.5 if obi_d["bias"] == "strong_bid" else -1.5
        all_reasons.append(f"عدم تعادل اردربوک: {obi_d['bias']}")
    if cvd_d and cvd_d.get("imbalance") in ("strong_bullish", "strong_bearish"):
        futures_score += 2 if cvd_d["imbalance"] == "strong_bullish" else -2
        all_reasons.append(f"CVD: {cvd_d['imbalance']}")

    total_score += futures_score

    tf_dirs = []
    for tf, result in tf_results.items():
        s = result["score"]
        tf_dirs.append(+1 if s > 1 else (-1 if s < -1 else 0))

    bullish_c = tf_dirs.count(+1); bearish_c = tf_dirs.count(-1)
    total_tfs = len(tf_dirs)
    agreement = max(bullish_c, bearish_c)/total_tfs if total_tfs > 0 else 0

    main_tf     = mode["timeframes"][1] if len(mode["timeframes"]) > 1 else mode["timeframes"][0]
    main_result = tf_results.get(main_tf, {})
    adx_info    = main_result.get("indicators", {}).get("adx", {}) if main_result else {}
    is_ranging  = bool(adx_info and adx_info.get("trend") == "weak")

    threshold = mode["signal_threshold"]; min_agree = mode["min_agreement"]
    direction = "NEUTRAL ⚪"; confidence = "سیگنال ضعیف ⚠️"
    sl = tp1 = tp2 = tp3 = None; expiry = None

    # برای اسکالپ، تأیید حجم اجباریه
    volume_ok = True
    if mode.get("require_volume_confirm"):
        main_vol = main_result.get("indicators", {}).get("vol", {}) if main_result else {}
        if main_vol.get("ratio", 1.0) < 1.0: volume_ok = False

    if total_score >= threshold and agreement >= min_agree and not is_ranging and volume_ok:
        direction = "LONG 🟢"
    elif total_score <= -threshold and agreement >= min_agree and not is_ranging and volume_ok:
        direction = "SHORT 🔴"

    if direction != "NEUTRAL ⚪":
        main_ind = main_result.get("indicators", {}) if main_result else {}
        atr_v    = main_ind.get("atr", price*0.01)
        sr_v     = main_ind.get("sr", {"near_support": price*0.97, "near_resistance": price*1.03})

        agree_pct = agreement * 100
        strength  = abs(total_score) / (threshold * 2)

        if agree_pct >= 85 and strength >= 1.0:
            confidence = "خیلی بالا 🔥🔥"
            sl_m = mode["sl_atr_mult"] * 0.9
            tp_m = [mode["tp_atr_mults"][0]*0.9, mode["tp_atr_mults"][1]]
        elif agree_pct >= 75:
            confidence = "بالا 🔥"
            sl_m = mode["sl_atr_mult"]
            tp_m = mode["tp_atr_mults"]
        else:
            confidence = "متوسط ✅"
            sl_m = mode["sl_atr_mult"] * 1.2
            tp_m = [mode["tp_atr_mults"][0]*1.1, mode["tp_atr_mults"][1]*1.2]

        if direction == "LONG 🟢":
            sl  = round(max(price - atr_v*sl_m, sr_v["near_support"]*0.997), 6)
            tp1 = round(price + atr_v*tp_m[0], 6)
            tp2 = round(price + atr_v*tp_m[1], 6)
            tp3 = round(min(sr_v["near_resistance"]*0.998, tp2*1.05), 6)
        else:
            sl  = round(min(price + atr_v*sl_m, sr_v["near_resistance"]*1.003), 6)
            tp1 = round(price - atr_v*tp_m[0], 6)
            tp2 = round(price - atr_v*tp_m[1], 6)
            tp3 = round(max(sr_v["near_support"]*1.002, tp2*0.95), 6)

        expiry = (datetime.now() + timedelta(hours=mode["hold_hours"])).isoformat()

    return {
        "symbol": symbol, "mode": mode_key, "mode_label": mode["label"],
        "price": price, "ticker": ticker,
        "direction": direction, "score": total_score,
        "confidence": confidence, "agreement": round(agreement*100, 1),
        "entry": price, "stop_loss": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "hold_label": mode["hold_label"], "hold_hours": mode["hold_hours"], "expiry": expiry,
        "reasons": all_reasons[:8], "timeframes": tf_results,
        "is_ranging": is_ranging,
        "volume_ok": volume_ok,
        "tf_agreement": {"bullish": bullish_c, "bearish": bearish_c, "neutral": tf_dirs.count(0)},
        "futures_data": {"funding": fr_d, "oi": oi_d, "lsr": lsr_d, "obi": obi_d, "cvd": cvd_d},
        "categories_used": sorted(all_categories),
    }

# =========================
# BACKTEST ساده
# =========================
async def run_backtest(symbol, mode_key="short", periods=20):
    """
    بک‌تست ساده: سیگنال‌های گذشته رو شبیه‌سازی می‌کنیم
    با داده‌های تاریخی از API
    """
    mode    = TRADING_MODES[mode_key]
    main_tf = mode["timeframes"][1] if len(mode["timeframes"]) > 1 else mode["timeframes"][0]
    limit   = mode["kline_limit"] + periods

    kdata = await get_klines(symbol, main_tf, limit)
    if not kdata or len(kdata["closes"]) < 60: return None

    closes  = kdata["closes"]; highs   = kdata["highs"]
    lows    = kdata["lows"];   opens   = kdata["opens"]
    volumes = kdata["volumes"]

    results = []
    step    = max(1, len(closes) // periods)

    for i in range(50, len(closes) - 5, step):
        c = closes[:i]; h = highs[:i]; l = lows[:i]; o = opens[:i]; v = volumes[:i]
        price = float(c[-1])

        rsi      = calc_rsi(c)
        rsi_ser  = calc_rsi_series(c)
        macd     = calc_macd(c)
        bb       = calc_bollinger(c)
        atr      = calc_atr(h, l, c)
        stoch    = calc_stochastic(h, l, c)
        vwap_v   = calc_vwap(h, l, c, v)
        adx      = calc_adx(h, l, c)
        div      = detect_divergence(c, rsi_ser)
        vol      = calc_volume_profile(c, v)
        fvg      = calc_fvg(h, l, c)
        of_data  = calc_order_flow(o, c, h, l, v)
        poc_data = calc_poc(h, l, c, v)
        ema20    = float(calc_ema(c, 20)[-1])
        ema50    = float(calc_ema(c, 50)[-1]) if len(c)>=50 else None

        ind = {"rsi":rsi,"macd":macd,"bb":bb,"atr":atr,"stoch":stoch,"vwap":vwap_v,
               "adx":adx,"divergence":div,"vol":vol,"fvg":fvg,"order_flow":of_data,
               "poc":poc_data,"ema20":ema20,"ema50":ema50,"ema200":None,
               "ichimoku":None,"market_structure":{},"sr":None,"patterns":[]}

        engine = SCORE_ENGINES.get(mode_key, score_short)
        votes  = engine(ind, price)
        score  = sum(s for _, s in votes)
        threshold = mode["signal_threshold"]

        if abs(score) < threshold: continue
        direction = "LONG" if score >= threshold else "SHORT"
        categories = sorted(set(_categorize_reason(r) for r, sc in votes if abs(sc) >= 2))

        # بررسی ۵ کندل بعد
        future_closes = closes[i:i+5]
        if len(future_closes) < 3: continue
        future_max = float(np.max(future_closes))
        future_min = float(np.min(future_closes))
        atr_v      = float(atr)

        if direction == "LONG":
            sl_p  = price - atr_v * mode["sl_atr_mult"]
            tp1_p = price + atr_v * mode["tp_atr_mults"][0]
            hit_tp = future_max >= tp1_p
            hit_sl = future_min <= sl_p
        else:
            sl_p  = price + atr_v * mode["sl_atr_mult"]
            tp1_p = price - atr_v * mode["tp_atr_mults"][0]
            hit_tp = future_min <= tp1_p
            hit_sl = future_max >= sl_p

        if hit_tp and not hit_sl: outcome = "win"
        elif hit_sl:               outcome = "loss"
        else:
            last_price = float(future_closes[-1])
            pnl = (last_price-price)/price*100 if direction=="LONG" else (price-last_price)/price*100
            outcome = "partial_win" if pnl > 0 else "partial_loss"

        # امتیازدهی پویا: نتیجه این سیگنال تاریخی هم به یادگیری اضافه می‌شه
        if categories:
            update_indicator_performance(symbol, categories, outcome in ("win", "partial_win"))

        results.append({"direction": direction, "score": score, "outcome": outcome,
                         "price": round(price,4), "categories": categories})

    if not results: return None
    wins    = sum(1 for r in results if r["outcome"]=="win")
    losses  = sum(1 for r in results if r["outcome"]=="loss")
    partial = sum(1 for r in results if "partial" in r["outcome"])
    total   = len(results)
    wr      = round(wins/total*100, 1) if total > 0 else 0

    return {"symbol": symbol, "mode": mode_key, "total": total,
            "wins": wins, "losses": losses, "partial": partial,
            "win_rate": wr, "results": results[-10:]}

# =========================
# AUTO-BACKTEST — انتخاب خودکار بهترین روش تحلیل برای هر ارز
# =========================
BEST_MODE_FILE = os.environ.get("BEST_MODE_FILE", "best_mode.json")
best_mode_data = {}   # {symbol: {"best_mode":..., "win_rate":..., "total":..., "updated":...}}

def load_best_mode():
    global best_mode_data
    if os.path.exists(BEST_MODE_FILE):
        try:
            with open(BEST_MODE_FILE, "r", encoding="utf-8") as f: best_mode_data = json.load(f)
        except Exception as e:
            log.error(f"load_best_mode: {e}"); best_mode_data = {}

def save_best_mode():
    try:
        tmp = BEST_MODE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(best_mode_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, BEST_MODE_FILE)
    except Exception as e:
        log.error(f"save_best_mode: {e}")

async def run_backtest_matrix(symbol, periods=15):
    """بک‌تست همون ارز روی هر ۴ حالت معاملاتی (اسکالپ/کوتاه‌مدت/میان‌مدت/بلندمدت)"""
    out = {}
    for mk in TRADING_MODES:
        try:
            r = await run_backtest(symbol, mk, periods=periods)
            if r: out[mk] = r
        except Exception as e:
            log.warning(f"run_backtest_matrix {symbol}/{mk}: {e}")
        await asyncio.sleep(0.3)
    return out

def pick_best_mode(matrix, min_samples=6):
    """بهترین حالت رو بر اساس نرخ موفقیت انتخاب می‌کنه (ترجیحاً با حداقل نمونه کافی)"""
    candidates = [(mk, r) for mk, r in matrix.items() if r["total"] >= min_samples]
    if not candidates:
        candidates = list(matrix.items())
    if not candidates: return None
    best_mk, best_r = max(candidates, key=lambda x: x[1]["win_rate"])
    return best_mk, best_r["win_rate"], best_r["total"]

def get_best_mode(symbol, default="short"):
    """حالتی که بک‌تست خودکار برای این ارز بهترین تشخیص داده؛ اگه هنوز داده‌ای نبود، پیش‌فرض کاربر"""
    entry = best_mode_data.get(symbol)
    if not entry: return default
    return entry.get("best_mode", default)

async def auto_backtest_all_symbols(app=None):
    """
    بک‌تست خودکار دوره‌ای (هر ۱۲ ساعت): برای تمام ارزهای زیرنظر کاربرها،
    هر ۴ حالت معاملاتی رو تست می‌کنه، بهترین حالت هر ارز رو ذخیره می‌کنه،
    و از نتایج تاریخی هر سیگنال، امتیازدهی پویا (وزن هر اندیکاتور روی هر ارز) رو آپدیت می‌کنه.
    """
    symbols = set()
    for udata in user_data.values():
        symbols.update(udata.get("symbols", []))
    if not symbols:
        log.info("auto_backtest_all_symbols: هیچ ارزی زیر نظر نیست"); return

    log.info(f"🧪 شروع بک‌تست خودکار برای {len(symbols)} ارز...")
    for sym in symbols:
        try:
            matrix = await run_backtest_matrix(sym)
            if not matrix: continue
            picked = pick_best_mode(matrix)
            if not picked: continue
            best_mk, wr, total = picked
            best_mode_data[sym] = {"best_mode": best_mk, "win_rate": wr, "total": total,
                                     "updated": datetime.now().isoformat(),
                                     "all_modes": {mk: r["win_rate"] for mk, r in matrix.items()}}
            log.info(f"  {sym}: بهترین حالت={TRADING_MODES[best_mk]['label']} (نرخ موفقیت={wr}%, نمونه={total})")
        except Exception as e:
            log.error(f"auto_backtest_all_symbols {sym}: {e}")
        await asyncio.sleep(0.5)
    save_best_mode()
    log.info("✅ بک‌تست خودکار تمام شد.")

# =========================
# LEVERAGE ENGINE
# =========================
def calc_leverage(signal_score, agreement_pct, mode_key, confidence_label, atr_pct):
    base = {"scalp":8,"short":4,"mid":3,"long":2}.get(mode_key, 2)
    af   = 1.4 if agreement_pct>=85 else (1.2 if agreement_pct>=78 else (1.0 if agreement_pct>=70 else 0.6))
    cf   = {"خیلی بالا 🔥🔥":1.3,"بالا 🔥":1.1,"متوسط ✅":0.9}.get(confidence_label, 1.0)
    vf   = 0.5 if atr_pct>=3.0 else (0.7 if atr_pct>=2.0 else (0.9 if atr_pct>=1.0 else 1.1))
    return max(1, min(10, round(base*af*cf*vf)))

def build_leverage_section(analysis, capital):
    if analysis["direction"] == "NEUTRAL ⚪" or not analysis["stop_loss"]: return ""
    entry=analysis["entry"]; sl=analysis["stop_loss"]
    tp1=analysis["tp1"];     tp2=analysis["tp2"]
    sl_pct  = abs(entry-sl)/entry*100
    tp1_pct = abs(tp1-entry)/entry*100 if tp1 else 0
    tp2_pct = abs(tp2-entry)/entry*100 if tp2 else 0
    atr_pct = sl_pct/1.5
    leverage = calc_leverage(analysis["score"], analysis["agreement"], analysis["mode"], analysis["confidence"], atr_pct)
    tp1_l = round(tp1_pct*leverage,2); tp2_l = round(tp2_pct*leverage,2); sl_l = round(sl_pct*leverage,2)
    capital_line = ""
    if capital:
        pos = capital*0.20
        capital_line = (f"💵 با ۲۰٪ سرمایه (${pos:,.0f}):\n"
                        f"  سود TP1: +${round(pos*(tp1_l/100),2):,.1f}  |  TP2: +${round(pos*(tp2_l/100),2):,.1f}\n"
                        f"  زیان SL: -${round(pos*(sl_l/100),2):,.1f}\n")
    lev_emoji = "🟢" if leverage<=2 else ("🟡" if leverage<=5 else "🔴")
    lev_note  = "محافظه‌کارانه" if leverage<=2 else ("متعادل" if leverage<=5 else "ریسک بالا — فقط با SL")
    msg  = f"\n⚡ اهرم پیشنهادی:\n━━━━━━━━━━━━━━━\n"
    msg += f"{lev_emoji} اهرم: {leverage}x  ({lev_note})\n\n"
    msg += f"📊 سود/زیان با {leverage}x:\n"
    msg += f"  بدون اهرم → TP1: +{tp1_pct:.2f}%  |  SL: -{sl_pct:.2f}%\n"
    msg += f"  با اهرم   → TP1: +{tp1_l}%  |  SL: -{sl_l}%\n"
    if tp2: msg += f"  با اهرم   → TP2: +{tp2_l}%\n"
    if capital_line: msg += f"\n{capital_line}"
    msg += f"\n⚠️ اهرم سود و زیان رو هر دو چندبرابر می‌کنه\n📌 بدون SL وارد نشو!\n"
    return msg

# =========================
# MONEY MANAGEMENT
# =========================
def calc_position_size(capital, entry, stop_loss, mode_key, agreement_pct):
    mode = TRADING_MODES[mode_key]
    base_risk = mode["risk_pct"]
    af = 1.3 if agreement_pct>=85 else (1.15 if agreement_pct>=78 else (1.0 if agreement_pct>=70 else 0.7))
    adj_risk = min(base_risk*af, 5.0)
    risk_usd = capital*(adj_risk/100)
    sl_pct   = abs(entry-stop_loss)/entry*100 if stop_loss else 1.0
    if sl_pct <= 0: sl_pct = 1.0
    pos_size = risk_usd/(sl_pct/100)
    pos_pct  = min(pos_size/capital*100, 50.0)
    coin_amt = pos_size/entry if entry > 0 else 0
    return {"risk_pct": round(adj_risk,2), "risk_amount": round(risk_usd,2),
            "position_pct": round(pos_pct,2), "position_size_usd": round(pos_size,2),
            "coin_amount": round(coin_amt,6), "sl_distance_pct": round(sl_pct,2)}

def build_mm_section(capital, analysis):
    if not capital or analysis["direction"] == "NEUTRAL ⚪" or not analysis["stop_loss"]: return ""
    mm   = calc_position_size(capital, analysis["entry"], analysis["stop_loss"], analysis["mode"], analysis["agreement"])
    tp1_pct = abs(analysis["tp1"]-analysis["entry"])/analysis["entry"]*100 if analysis["tp1"] else 0
    rr   = round(tp1_pct/mm["sl_distance_pct"],2) if mm["sl_distance_pct"] > 0 else 0
    msg  = f"\n💼 مدیریت سرمایه:\n━━━━━━━━━━━━━━━\n"
    msg += f"💰 سرمایه کل:        ${capital:,.0f}\n"
    msg += f"📊 حجم پیشنهادی:     {mm['position_pct']}٪  ≈ ${mm['position_size_usd']:,.0f}\n"
    msg += f"🎲 ریسک این معامله:  {mm['risk_pct']}٪  ≈ ${mm['risk_amount']:,.0f}\n"
    msg += f"📏 فاصله استاپ:      {mm['sl_distance_pct']}٪\n"
    msg += f"🪙 مقدار خرید:       {mm['coin_amount']} واحد\n"
    msg += f"⚖️ نسبت R/R:         1:{rr}\n"
    if rr >= 2:     msg += f"  ✅ R/R مناسب\n"
    elif rr >= 1.5: msg += f"  🟡 R/R قابل قبول\n"
    else:           msg += f"  ⚠️ R/R پایین — احتیاط\n"
    return msg

# =========================
# DYNAMIC SCORING (امتیازدهی پویا)
# اگر یک اندیکاتور روی یک ارز خاص نتیجه خوبی داده، وزنش بالاتر می‌ره
# =========================
CATEGORY_KEYWORDS = {
    "RSI": ["RSI", "rsi"], "MACD": ["MACD"], "BB": ["BB", "باند"],
    "Stoch": ["استوک", "Stoch"], "ADX": ["ADX", "روند"], "Ichimoku": ["ایچیموکو"],
    "Volume": ["حجم"], "OrderFlow": ["Order Flow"], "FVG": ["FVG"],
    "POC": ["POC", "Value Area"], "Divergence": ["واگرایی"], "EMA": ["EMA", "Golden", "Death"],
    "MarketStructure": ["ساختار", "BOS"], "VWAP": ["VWAP"], "Pattern": ["الگو"],
}

def _categorize_reason(reason):
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in reason for kw in kws): return cat
    return "Other"

def load_adaptive_weights():
    global adaptive_weights
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f: adaptive_weights = json.load(f)
        except Exception as e:
            log.error(f"load_adaptive_weights: {e}"); adaptive_weights = {}

def save_adaptive_weights():
    try:
        tmp = WEIGHTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(adaptive_weights, f, ensure_ascii=False, indent=2)
        os.replace(tmp, WEIGHTS_FILE)
    except Exception as e:
        log.error(f"save_adaptive_weights: {e}")

def get_category_multiplier(symbol, category):
    """
    وزن اکتسابی: بین 0.6x (اندیکاتور ضعیف روی این ارز) تا 1.5x (اندیکاتور قوی روی این ارز)
    بر اساس نرخ موفقیت تاریخی محاسبه می‌شه. حداقل ۵ نمونه لازمه تا اثر بذاره.
    """
    stat = adaptive_weights.get(symbol, {}).get(category)
    if not stat: return 1.0
    total = stat.get("win", 0) + stat.get("loss", 0)
    if total < 5: return 1.0
    win_rate = stat["win"] / total
    return round(0.6 + win_rate * 0.9, 3)   # win_rate=0 -> 0.6x | win_rate=1 -> 1.5x

def apply_dynamic_weights(symbol, votes):
    """روی لیست votes یک اندیکاتور، ضریب یادگرفته‌شده رو اعمال می‌کنه"""
    adjusted = []
    for reason, score in votes:
        cat  = _categorize_reason(reason)
        mult = get_category_multiplier(symbol, cat)
        adjusted.append((reason, round(score * mult, 3), cat))
    return adjusted

def update_indicator_performance(symbol, categories_used, outcome_is_win):
    """بعد از بسته شدن هر پوزیشن، امتیاز اندیکاتورهایی که در اون سیگنال شرکت داشتن آپدیت می‌شه"""
    sym_w = adaptive_weights.setdefault(symbol, {})
    for cat in categories_used:
        stat = sym_w.setdefault(cat, {"win": 0, "loss": 0})
        if outcome_is_win: stat["win"] += 1
        else: stat["loss"] += 1
    save_adaptive_weights()

# =========================
# AI SIGNAL EXPLANATION (هوش مصنوعی صادرکننده سیگنال)
# =========================
async def get_ai_review(analysis, futures_data=None, news=None):
    """
    یک تماس با Groq که هم توضیح سیگنال رو می‌ده، هم به‌عنوان یک لایه دومِ advisory
    چک می‌کنه که آیا بین اجزای مختلف دیتا (تکنیکال/فاندینگ/اخبار/OI) تناقض معنی‌داری هست.

    ⚠️ طراحی عمدی: خروجی این تابع هرگز جهت/SL/TP/امتیاز رو تغییر نمی‌ده و نمی‌تونه سیگنال
    رو حذف کنه — فقط یک پرچم هشدار متنی («caution») اضافه به پیام می‌کنه. تصمیم نهایی
    همیشه دست موتور امتیازدهی قانون‌محور می‌مونه، نه AI.

    خروجی: {"explanation": str, "verdict": "confirm"|"caution", "concern": str|None}
    """
    if not analysis or analysis["direction"] == "NEUTRAL ⚪": return None
    default = {"explanation": _fallback_explanation(analysis, futures_data, news),
               "verdict": "confirm", "concern": None}
    if not GROQ_API_KEY:
        return default
    try:
        summary = {
            "symbol": analysis["symbol"], "direction": analysis["direction"],
            "score": analysis["score"], "agreement": analysis["agreement"],
            "confidence": analysis["confidence"], "reasons": analysis["reasons"],
            "mode": analysis["mode_label"],
            "funding_rate": futures_data.get("funding") if futures_data else None,
            "open_interest_change": futures_data.get("oi") if futures_data else None,
            "long_short_ratio": futures_data.get("lsr") if futures_data else None,
            "cvd": futures_data.get("cvd") if futures_data else None,
            "orderbook_imbalance": futures_data.get("obi") if futures_data else None,
            "news_sentiment": news.get("label") if news else None,
        }
        prompt = (
            "تو یک تحلیلگر ارشد فیوچرز کریپتو هستی و داری یه سیگنال از قبل صادرشده رو مرور می‌کنی "
            "(جهت و اعداد ورود/خروج قبلاً با موتور قانون‌محور مشخص شده و قابل تغییر نیست؛ فقط نظر مرورگر می‌خوایم).\n\n"
            f"دیتا:\n{json.dumps(summary, ensure_ascii=False)}\n\n"
            "فقط یک JSON خالص (بدون ```json و بدون هیچ متن اضافه) با این فرمت برگردون:\n"
            '{"explanation": "۳ تا ۵ جمله فارسی روان درباره چرایی سیگنال و ریسکش", '
            '"verdict": "confirm" یا "caution", '
            '"concern": "اگه caution، یک جمله کوتاه بگو چه تناقضی بین اجزای دیتا دیدی؛ وگرنه null"}\n'
            'verdict فقط وقتی caution باشه که واقعاً بین دیتای تکنیکال و فاندینگ/اخبار/OI/CVD تناقض معنی‌دار وجود داشته باشه '
            '(مثلاً سیگنال LONG ولی اکثریت شدید بازار هم لانگ‌ان و Funding به‌شدت مثبته، یا اخبار به‌وضوح منفیه).'
        )
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        body = {"model": GROQ_MODEL, "max_tokens": 400, "temperature": 0.3,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200:
                log.warning(f"Groq API status {r.status}: {await r.text()}")
                return default
            d = await r.json()
            raw = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            raw = raw.strip("`")
            if raw.lower().startswith("json"): raw = raw[4:].strip()
            parsed = json.loads(raw)
            verdict = parsed.get("verdict") if parsed.get("verdict") in ("confirm", "caution") else "confirm"
            return {
                "explanation": parsed.get("explanation") or default["explanation"],
                "verdict": verdict,
                "concern": parsed.get("concern") if verdict == "caution" else None,
            }
    except Exception as e:
        log.warning(f"AI review error: {e}")
        return default

async def get_ai_explanation(analysis, futures_data=None, news=None):
    """سازگاری با کد قدیمی — فقط رشته توضیح رو برمی‌گردونه (بدون verdict)"""
    review = await get_ai_review(analysis, futures_data, news)
    return review["explanation"] if review else None

def _fallback_explanation(analysis, futures_data=None, news=None):
    """توضیح قانون‌محور وقتی AI در دسترس نیست — هسته سیگنال هیچ‌وقت قطع نمی‌شه"""
    parts = []
    d = "صعودی" if "LONG" in analysis["direction"] else "نزولی"
    parts.append(f"سیگنال {d} با امتیاز {analysis['score']} و هم‌راستایی {analysis['agreement']}٪ بین تایم‌فریم‌ها صادر شده.")
    if analysis["reasons"]:
        parts.append("مهم‌ترین دلایل: " + "، ".join(analysis["reasons"][:3]) + ".")
    if futures_data:
        fr = futures_data.get("funding")
        if fr and fr.get("extreme"):
            parts.append(f"Funding Rate در ناحیه افراطی ({fr['rate']}%) قرار داره که احتمال اسکوییز رو بالا می‌بره.")
        lsr = futures_data.get("lsr")
        if lsr and lsr.get("extreme"):
            parts.append(f"اکثریت معامله‌گران سمت {lsr['crowd']} هستن که ریسک نقدشدن دسته‌جمعی مخالف جهت جمع رو ایجاد می‌کنه.")
    if news and news.get("label") and "خنثی" not in news["label"]:
        parts.append(f"اخبار اخیر بازار {news['label']} ارزیابی شده.")
    parts.append("⚠️ این یک تحلیل خودکاره و جایگزین تصمیم شخصی و مدیریت ریسک خودت نیست.")
    return " ".join(parts)

# =========================
# NEW/HOT COIN SCANNER (اسکن ارزهای جدید و پرنوسان)
# =========================
async def _scan_binance(top_n):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    data, _ = await throttled_get("binance", url, timeout=15)
    if not data: return None
    out = []
    for d in data:
        sym = d.get("symbol", "")
        if not sym.endswith("USDT"): continue
        try:
            change = abs(float(d.get("priceChangePercent", 0)))
            qvol   = float(d.get("quoteVolume", 0))
        except Exception:
            continue
        if qvol < 3_000_000 or change < 8: continue
        out.append({"symbol": sym, "change_pct": round(change, 2), "quote_volume": round(qvol, 0)})
    return out

async def _scan_mexc(top_n):
    """فال‌بک: تیکر ۲۴ ساعته اسپات MEXC"""
    url = "https://api.mexc.com/api/v3/ticker/24hr"
    data, _ = await throttled_get("mexc", url, timeout=15)
    if not data or not isinstance(data, list): return None
    out = []
    for d in data:
        sym = d.get("symbol", "")
        if not sym.endswith("USDT"): continue
        try:
            change = abs(float(d.get("priceChangePercent", 0)))
            qvol   = float(d.get("quoteVolume", 0))
        except Exception:
            continue
        if qvol < 3_000_000 or change < 8: continue
        out.append({"symbol": sym, "change_pct": round(change, 2), "quote_volume": round(qvol, 0)})
    return out

async def _scan_bybit(top_n):
    """فال‌بک دوم: تیکر فیوچرز خطی Bybit (price24hPcnt کسری‌ست، ×۱۰۰ می‌شه)"""
    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    data, _ = await throttled_get("bybit", url, timeout=15)
    if not data or not isinstance(data, dict): return None
    rows = (data.get("result") or {}).get("list") or []
    out = []
    for d in rows:
        sym = d.get("symbol", "")
        if not sym.endswith("USDT"): continue
        try:
            change = abs(float(d.get("price24hPcnt", 0))) * 100
            qvol    = float(d.get("turnover24h", 0))
        except Exception:
            continue
        if qvol < 3_000_000 or change < 8: continue
        out.append({"symbol": sym, "change_pct": round(change, 2), "quote_volume": round(qvol, 0)})
    return out

async def scan_hot_coins(top_n=8):
    """
    اسکن آلت‌کوین‌های پرنوسان با فال‌بک بین چند صرافی (Binance → MEXC → Bybit)
    تا اگر یکی در دسترس نبود (مثلاً محدودیت جغرافیایی)، بقیه امتحان بشن.
    ⚠️ این روش نشانگر «نوسان بالا» ست، نه تشخیص قطعی لیست جدید صرافی
    (تاریخ لیست‌شدن دقیق نیاز به endpoint اختصاصی هر صرافیه که رایگان در دسترس نیست).
    """
    for name, fn in [("Binance", _scan_binance), ("MEXC", _scan_mexc), ("Bybit", _scan_bybit)]:
        try:
            out = await fn(top_n)
            if out is None: continue
            out.sort(key=lambda x: x["change_pct"], reverse=True)
            if name != "Binance": log.info(f"scan_hot_coins از {name}")
            return out[:top_n]
        except Exception as e:
            log.warning(f"scan_hot_coins [{name}]: {e}")
    return []

# =========================
# DATA
# =========================
def load_data():
    global user_data
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE,"r",encoding="utf-8") as f: user_data = json.load(f)
        except Exception as e: log.error(f"Load error: {e}"); user_data = {}

def save_data():
    try:
        tmp = DATA_FILE+".tmp"
        with open(tmp,"w",encoding="utf-8") as f: json.dump(user_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, DATA_FILE)
    except Exception as e: log.error(f"Save error: {e}")

def backup_data():
    try:
        with open(DATA_FILE+".bak","w",encoding="utf-8") as f: json.dump(user_data, f, ensure_ascii=False, indent=2)
    except Exception as e: log.error(f"Backup error: {e}")

def load_users_list():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE,"r",encoding="utf-8") as f: return json.load(f)
        except: pass
    return {}

def save_users_list(ul):
    try:
        with open(USERS_FILE,"w",encoding="utf-8") as f: json.dump(ul, f, ensure_ascii=False, indent=2)
    except Exception as e: log.error(f"Save users: {e}")

def register_user(update):
    user=update.effective_user; chat_id=str(update.effective_chat.id)
    ul=load_users_list(); is_new=chat_id not in ul
    ul[chat_id] = {"chat_id":chat_id,"first_name":user.first_name or "",
                   "last_name":user.last_name or "","username":f"@{user.username}" if user.username else "ندارد",
                   "first_seen":ul.get(chat_id,{}).get("first_seen",datetime.now().isoformat()),
                   "last_seen":datetime.now().isoformat(),
                   "count":ul.get(chat_id,{}).get("count",0)+1}
    save_users_list(ul); return is_new, ul[chat_id]

def init_user(chat_id):
    if chat_id not in user_data:
        user_data[chat_id] = {
            "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT"],
            "interval": 60, "active": True, "trading_mode": "short",
            "capital": None, "active_positions": {}, "signal_history": [],
            "signal_stats": {"total":0,"win":0,"loss":0,"neutral_exit":0},
            "price_alerts": {},
            "auto_alert_threshold": 8,  # امتیاز حداقل برای آلرت خودکار
            "auto_alert_enabled": True,
        }
        save_data()

# =========================
# FORMAT HELPERS
# =========================
def format_price(p):
    if p is None: return "N/A"
    if p >= 1000: return f"{p:,.2f}"
    if p >= 1:    return f"{p:,.4f}"
    return f"{p:,.6f}"

# =========================
# MESSAGE BUILDERS
# =========================
def build_analysis_message(a, capital=None, fg=None, fr=None, whale=None,
                            news=None, price_cmp=None, liq_map=None, ai_review=None):
    if not a: return "❌ خطا در دریافت داده"
    sym = a["symbol"]
    tv  = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}"
    now = datetime.now().strftime("%H:%M:%S")
    agg = a["tf_agreement"]
    fd  = a.get("futures_data", {}) or {}
    if fr is None: fr = fd.get("funding")
    oi  = fd.get("oi"); lsr = fd.get("lsr"); obi = fd.get("obi"); cvd = fd.get("cvd")

    msg  = f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"📊 {sym} | {now}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"💵 قیمت: {format_price(a['price'])} USDT\n"
    msg += f"📈 تغییر ۲۴h: {a['ticker']['change']:+.2f}%\n"
    msg += f"🕐 بازه: {a['mode_label']}\n"

    if price_cmp and price_cmp.get("toobit_warning"):
        msg += f"⚠️ اختلاف قیمت Toobit: {price_cmp['toobit_spread_pct']:+.2f}٪ نسبت به میانگین صرافی‌ها\n"

    if fg:
        msg += f"😱 Fear&Greed: {fg['value']} — {fg['label']} {fg['emoji']}\n"
    if fr:
        fr_emoji = "🟢" if fr["bullish"] else "🔴"
        fr_warn  = " ⚠️ افراطی!" if fr["extreme"] else ""
        msg += f"{fr_emoji} Funding Rate: {fr['rate']}%{fr_warn}\n"
    if oi and oi.get("change_pct") is not None:
        oi_emoji = "📈" if oi["rising"] else ("📉" if oi["falling"] else "➖")
        msg += f"{oi_emoji} Open Interest: {oi['change_pct']:+.2f}٪ (۶ساعته)\n"
    if lsr:
        lsr_warn = " ⚠️ ازدحام!" if lsr["extreme"] else ""
        msg += f"⚖️ Long/Short: {lsr['long_pct']}٪ / {lsr['short_pct']}٪{lsr_warn}\n"
    if obi and obi["bias"] != "balanced":
        obi_emoji = "🟢" if "bid" in obi["bias"] else "🔴"
        msg += f"{obi_emoji} Order Book Imbalance: {obi['imbalance_pct']:+.1f}٪ ({obi['bias']})\n"
    if cvd and cvd.get("imbalance") in ("strong_bullish", "strong_bearish"):
        cvd_emoji = "🟢" if cvd["imbalance"] == "strong_bullish" else "🔴"
        msg += f"{cvd_emoji} CVD: {cvd['imbalance']} (قدرت: {cvd['strength']})\n"
    if liq_map:
        msg += f"🗺 لیکوئیدیشن تخمینی: پایین {format_price(liq_map['long_liq_zone'])} | بالا {format_price(liq_map['short_liq_zone'])}\n"
    if news and news.get("label") and "خنثی" not in news["label"]:
        msg += f"📰 اخبار: {news['label']}\n"
    if whale and whale.get("alert"):
        w_emoji = "🟢" if whale.get("bullish") else ("🔴" if whale.get("bearish") else "🐋")
        msg += f"{w_emoji} فشار نهنگ‌ها: {whale.get('large_tx',0)} معامله بزرگ (خرید ${whale.get('buy_usd',0):,.0f} / فروش ${whale.get('sell_usd',0):,.0f})\n"

    msg += f"\n🎯 سیگنال: {a['direction']}\n"
    msg += f"💪 اطمینان: {a['confidence']}\n"
    msg += f"📐 امتیاز: {a['score']}  |  هم‌راستایی: {a['agreement']}%\n"
    msg += f"📊 تأیید TF: ✅{agg['bullish']} صعودی | ❌{agg['bearish']} نزولی | ⚪{agg['neutral']} خنثی\n"

    if a.get("is_ranging"):
        msg += f"\n⚠️ بازار رنج — از ورود خودداری کن\n"
    if not a.get("volume_ok", True):
        msg += f"\n⚠️ حجم ضعیف — سیگنال تأیید نشد\n"

    # Market Structure (برای mid و long)
    main_tf = TRADING_MODES[a["mode"]]["timeframes"][1] if len(TRADING_MODES[a["mode"]]["timeframes"])>1 else TRADING_MODES[a["mode"]]["timeframes"][0]
    main_ms = a["timeframes"].get(main_tf,{}).get("indicators",{}).get("market_structure",{}) if main_tf in a["timeframes"] else {}
    if main_ms and main_ms.get("structure") and main_ms["structure"] != "unknown":
        msg += f"🏗 ساختار: {main_ms['structure']}"
        if main_ms.get("bos") != "none":
            msg += f" | BOS: {'↑' if main_ms['bos']=='bullish_bos' else '↓'}"
        msg += "\n"

    if a["direction"] != "NEUTRAL ⚪" and a["stop_loss"]:
        sl_pct  = abs(a["entry"]-a["stop_loss"])/a["entry"]*100
        tp1_pct = abs(a["tp1"]-a["entry"])/a["entry"]*100 if a["tp1"] else 0
        rr      = round(tp1_pct/sl_pct, 2) if sl_pct > 0 else 0

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
        for r in a["reasons"]: msg += f"  • {r}\n"

    ind = a["timeframes"].get(main_tf,{}).get("indicators",{}) if main_tf in a["timeframes"] else {}
    if ind:
        stoch_v=ind.get("stoch") or {}; macd_v=ind.get("macd") or {}
        adx_v=ind.get("adx") or {};     bb_v=ind.get("bb") or {}
        ich=ind.get("ichimoku") or {};  div=ind.get("divergence","none")
        pats=ind.get("patterns",[])
        msg += f"\n📉 اندیکاتور ({main_tf}):\n"
        msg += f"  RSI: {ind.get('rsi','N/A')}  |  Stoch K/D: {stoch_v.get('k','N/A')}/{stoch_v.get('d','N/A')}\n"
        msg += f"  MACD Hist: {macd_v.get('histogram','N/A')}  |  ADX: {adx_v.get('adx','N/A')} ({adx_v.get('trend','N/A')})\n"
        if bb_v: msg += f"  BB%: {bb_v.get('bb_pct','N/A')}  |  BW: {bb_v.get('bandwidth','N/A')}\n"
        if ich:  msg += f"  Ichimoku: {ich.get('signal','N/A')} | TK Cross: {ich.get('tk_cross','N/A')}\n"
        if div != "none": msg += f"  ⚡ {div}\n"
        if pats: msg += f"  الگو: {', '.join(pats[:2])}\n"
        if ind.get("ema200"): msg += f"  EMA200: {format_price(ind['ema200'])}\n"

        # FVG
        fvg_d = ind.get("fvg")
        if fvg_d and fvg_d.get("signal","none") != "none":
            fvg_emoji = "🟢" if "bullish" in fvg_d["signal"] else "🔴"
            msg += f"  {fvg_emoji} FVG: {fvg_d['signal']} (فاصله: {fvg_d.get('nearest_dist','?')}%)\n"

        # POC
        poc_d = ind.get("poc")
        if poc_d:
            poc_pos = "بالای POC ✅" if poc_d["above_poc"] else "زیر POC ⚠️"
            msg += f"  💹 POC: {format_price(poc_d['poc'])} | {poc_pos} | VA: {'داخل' if poc_d['in_value_area'] else 'خارج'}\n"

        # Order Flow
        of_d = ind.get("order_flow")
        if of_d and of_d.get("imbalance") in ("strong_bullish","strong_bearish"):
            of_emoji = "🟢" if of_d["bias"]=="bullish" else "🔴"
            msg += f"  {of_emoji} Order Flow: {of_d['imbalance']} (قدرت: {of_d['strength']}x)\n"

    if ai_review:
        if ai_review.get("verdict") == "caution" and ai_review.get("concern"):
            msg += f"\n⚠️ فیلتر AI — نکته احتیاطی:\n{ai_review['concern']}\n"
        msg += f"\n🤖 توضیح هوش مصنوعی:\n{ai_review.get('explanation','')}\n"

    msg += f"\n🔗 {tv}\n━━━━━━━━━━━━━━━━━━━━"
    return msg

def build_price_message(symbol, ticker):
    if not ticker: return f"❌ ارز {symbol} یافت نشد"
    now=datetime.now().strftime("%H:%M:%S"); emoji="📈" if ticker["change"]>=0 else "📉"
    msg  = f"💰 قیمت لحظه‌ای\n━━━━━━━━━━━━━━\n"
    msg += f"🪙 {symbol}\n"
    msg += f"💵 قیمت:     {format_price(ticker['price'])} USDT\n"
    msg += f"{emoji} تغییر ۲۴h: {ticker['change']:+.2f}%\n"
    msg += f"🔼 بیشترین:  {format_price(ticker['high'])}\n"
    msg += f"🔽 کمترین:   {format_price(ticker['low'])}\n"
    msg += f"📦 حجم:      {ticker['volume']:,.0f}\n⏰ {now}"
    return msg

def build_history_message(history, stats):
    if not history: return "📭 هنوز هیچ سیگنالی ثبت نشده"
    total=stats["total"]; win=stats["win"]; loss=stats["loss"]; ne=stats.get("neutral_exit",0)
    wr=round(win/total*100,1) if total > 0 else 0
    msg  = f"📊 تاریخچه سیگنال‌ها\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"کل: {total} | ✅موفق: {win} | ❌ناموفق: {loss} | 🟡خنثی: {ne}\n"
    msg += f"نرخ موفقیت: {wr}%\n━━━━━━━━━━━━━━━━━━━━\n\n"
    for s in reversed(history[-10:]):
        t=datetime.fromisoformat(s["time"]).strftime("%m/%d %H:%M"); res=s.get("result","در انتظار")
        msg += f"• {s['symbol']} | {s['direction']}\n"
        msg += f"  ورود: {format_price(s['entry'])} | نتیجه: {res} | {t}\n\n"
    return msg

def build_performance_report(chat_id):
    """گزارش عملکرد جامع"""
    stats   = user_data.get(chat_id, {}).get("signal_stats", {})
    history = user_data.get(chat_id, {}).get("signal_history", [])
    total   = stats.get("total", 0)
    if total == 0: return "📭 هنوز سیگنالی ثبت نشده"

    win  = stats.get("win", 0); loss = stats.get("loss", 0); ne = stats.get("neutral_exit", 0)
    wr   = round(win/total*100, 1) if total > 0 else 0

    # آمار بر اساس مود
    mode_stats = {}
    for h in history:
        m = h.get("mode", "short")
        if m not in mode_stats: mode_stats[m] = {"total":0,"win":0,"loss":0}
        mode_stats[m]["total"] += 1
        res = h.get("result", "")
        if "موفق" in res: mode_stats[m]["win"] += 1
        elif "استاپ" in res: mode_stats[m]["loss"] += 1

    msg  = f"📈 گزارش عملکرد ربات\n━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📊 کلی:\n"
    msg += f"  کل سیگنال:   {total}\n"
    msg += f"  ✅ موفق:     {win} ({wr}%)\n"
    msg += f"  ❌ ناموفق:   {loss} ({round(loss/total*100,1) if total else 0}%)\n"
    msg += f"  🟡 خنثی:     {ne}\n\n"
    msg += f"📊 بر اساس بازه:\n"
    for mk, ms in mode_stats.items():
        mwr = round(ms["win"]/ms["total"]*100, 0) if ms["total"] > 0 else 0
        label = TRADING_MODES.get(mk, {}).get("label", mk)
        msg += f"  {label}: {ms['total']} سیگنال | ✅{mwr}%\n"

    # آخرین ۵ سیگنال
    if history:
        msg += f"\n📋 آخرین ۵ سیگنال:\n"
        for h in reversed(history[-5:]):
            t = datetime.fromisoformat(h["time"]).strftime("%m/%d %H:%M")
            pnl = f" ({h['pnl_pct']:+.1f}%)" if h.get("pnl_pct") else ""
            msg += f"  • {h['symbol']} {h['direction']} | {h.get('result','در انتظار')}{pnl} | {t}\n"

    return msg

# =========================
# MENUS
# =========================
def main_menu(chat_id):
    ud = user_data.get(chat_id, {}); is_active = ud.get("active", True)
    symbols = ud.get("symbols", []); mode_key = ud.get("trading_mode", "short")
    capital = ud.get("capital"); cap_text = f"${capital:,.0f}" if capital else "تنظیم نشده ⚠️"
    auto_alert = ud.get("auto_alert_enabled", True)
    toggle_btn = (InlineKeyboardButton("⏹ توقف ارسال", callback_data="toggle_off")
                  if is_active else InlineKeyboardButton("▶️ شروع ارسال", callback_data="toggle_on"))
    alert_btn = (InlineKeyboardButton("🔕 آلرت خودکار: روشن", callback_data="toggle_autoalert")
                 if auto_alert else InlineKeyboardButton("🔔 آلرت خودکار: خاموش", callback_data="toggle_autoalert"))
    keyboard = [
        [InlineKeyboardButton("📊 تحلیل همین الان", callback_data="do_analysis")],
        [InlineKeyboardButton("💰 قیمت لحظه‌ای", callback_data="menu_price")],
        [toggle_btn],
        [alert_btn],
        [InlineKeyboardButton("➕ افزودن ارز", callback_data="menu_add"),
         InlineKeyboardButton("➖ حذف ارز", callback_data="menu_remove")],
        [InlineKeyboardButton("🎯 بازه معاملاتی", callback_data="menu_mode")],
        [InlineKeyboardButton("💼 تنظیم سرمایه", callback_data="menu_capital")],
        [InlineKeyboardButton("🔔 هشدار قیمت", callback_data="menu_alerts")],
        [InlineKeyboardButton("📋 ارزهای فعال", callback_data="menu_list")],
        [InlineKeyboardButton("📁 پوزیشن‌های فعال", callback_data="menu_positions")],
        [InlineKeyboardButton("📈 تاریخچه", callback_data="menu_history"),
         InlineKeyboardButton("🏆 عملکرد", callback_data="menu_performance")],
        [InlineKeyboardButton("🧪 بک‌تست", callback_data="menu_backtest"),
         InlineKeyboardButton("🐋 نهنگ‌ها", callback_data="menu_whale")],
        [InlineKeyboardButton("🆕 ارزهای داغ", callback_data="menu_hotcoins"),
         InlineKeyboardButton("⏱ بازه ارسال", callback_data="menu_interval")],
    ]
    status = "🟢 فعال" if is_active else "🔴 متوقف"
    text = (f"🤖 ربات تحلیل ارز دیجیتال\n━━━━━━━━━━━━━━━\n"
            f"وضعیت: {status}\n"
            f"ارزها: {len(symbols)} عدد\n"
            f"بازه: {TRADING_MODES[mode_key]['label']}\n"
            f"سرمایه: {cap_text}\n"
            f"آلرت خودکار: {'✅' if auto_alert else '❌'}\n\n"
            f"یک گزینه انتخاب کن:")
    return text, InlineKeyboardMarkup(keyboard)

def price_menu(chat_id):
    symbols = user_data.get(chat_id,{}).get("symbols",[])
    keyboard = []; row = []
    for sym in symbols:
        row.append(InlineKeyboardButton(sym.replace("USDT",""), callback_data=f"price_{sym}"))
        if len(row)==3: keyboard.append(row); row=[]
    if row: keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔍 ارز دیگه", callback_data="price_custom")])
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    return "💰 قیمت لحظه‌ای کدوم ارز؟", InlineKeyboardMarkup(keyboard)

def mode_menu(chat_id=None):
    keyboard = []
    for k, v in TRADING_MODES.items():
        desc = {"scalp":"سریع، ریسک کم، حجم مهم","short":"متعادل، MACD+RSI","mid":"ایچیموکو+ساختار","long":"EMA200+ساختار بلندمدت"}[k]
        keyboard.append([InlineKeyboardButton(f"{v['label']}\n{desc}", callback_data=f"setmode_{k}")])
    auto_on = user_data.get(chat_id, {}).get("auto_mode_enabled", True) if chat_id else True
    auto_label = "🤖 حالت خودکار (بهترین روش هر ارز): روشن ✅" if auto_on else "🤖 حالت خودکار: خاموش ❌"
    keyboard.append([InlineKeyboardButton(auto_label, callback_data="toggle_automode")])
    keyboard.append([InlineKeyboardButton("📊 بهترین حالت هر ارز (بک‌تست خودکار)", callback_data="view_bestmodes")])
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    note = ("\n\nℹ️ وقتی «حالت خودکار» روشنه، ربات بجای این انتخاب دستی، از نتیجه‌ی بک‌تست خودکار "
            "(هر ۱۲ ساعت) استفاده می‌کنه و برای هر ارز، بهترین روش خودش رو انتخاب می‌کنه؛ "
            "این انتخاب فقط وقتی خودکار خاموش باشه یا هنوز داده بک‌تست کافی نباشه استفاده می‌شه.")
    return "🎯 بازه معاملاتی رو انتخاب کن:" + note, InlineKeyboardMarkup(keyboard)

def add_symbol_menu(chat_id):
    current=set(user_data.get(chat_id,{}).get("symbols",[])); keyboard=[]; row=[]
    for sym in AVAILABLE_SYMBOLS:
        if sym not in current:
            row.append(InlineKeyboardButton(sym.replace("USDT",""), callback_data=f"add_{sym}"))
            if len(row)==3: keyboard.append(row); row=[]
    if row: keyboard.append(row)
    keyboard.append([InlineKeyboardButton("✏️ ارز دلخواه", callback_data="add_custom")])
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    return "➕ کدوم ارز رو اضافه کنی؟", InlineKeyboardMarkup(keyboard)

def remove_symbol_menu(chat_id):
    current=user_data.get(chat_id,{}).get("symbols",[]); keyboard=[]; row=[]
    for sym in current:
        row.append(InlineKeyboardButton(f"❌{sym.replace('USDT','')}", callback_data=f"rem_{sym}"))
        if len(row)==3: keyboard.append(row); row=[]
    if row: keyboard.append(row)
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    return "➖ کدوم ارز رو حذف کنی؟", InlineKeyboardMarkup(keyboard)

def alerts_menu(chat_id):
    alerts=user_data.get(chat_id,{}).get("price_alerts",{})
    symbols=user_data.get(chat_id,{}).get("symbols",[]); keyboard=[]
    for sym in symbols:
        a=alerts.get(sym,{}); label=sym.replace("USDT",""); info=""
        if a.get("above"): info += f"↑{format_price(a['above'])} "
        if a.get("below"): info += f"↓{format_price(a['below'])}"
        display=f"🔔{label}"+(f" [{info.strip()}]" if info else "")
        keyboard.append([InlineKeyboardButton(display, callback_data=f"alert_set_{sym}")])
    if alerts: keyboard.append([InlineKeyboardButton("🗑 حذف همه", callback_data="alert_clear_all")])
    keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
    return "🔔 هشدار قیمت:", InlineKeyboardMarkup(keyboard)

# =========================
# SMART AUTO ALERT
# =========================
async def check_auto_alerts(bot):
    """
    هسته آلرت خودکار — کاملاً مستقل از هسته پوزیشن (check_expired_positions).
    این تابع هرگز نباید به‌خاطر خطا در بخش دیگه‌ای از ربات متوقف بشه؛ هر
    خطا فقط برای همون symbol/user لاگ می‌شه و بقیه ادامه پیدا می‌کنن.
    """
    try:
        for chat_id, udata in list(user_data.items()):
            if not udata.get("auto_alert_enabled", True): continue
            if not udata.get("active", True): continue
            symbols   = udata.get("symbols", [])
            mode_key  = udata.get("trading_mode", "short")
            auto_mode = udata.get("auto_mode_enabled", True)
            threshold = udata.get("auto_alert_threshold", 8)
            capital   = udata.get("capital")

            for symbol in symbols:
                try:
                    sym_mode = get_best_mode(symbol, default=mode_key) if auto_mode else mode_key
                    a = await full_analysis(symbol, sym_mode)
                    if not a: continue
                    if a["direction"] == "NEUTRAL ⚪": continue
                    if abs(a["score"]) < threshold: continue
                    if a["agreement"] < 75: continue

                    last_alerts = udata.setdefault("last_auto_alerts", {})
                    last = last_alerts.get(symbol, {})
                    if last.get("direction") == a["direction"]:
                        last_time = datetime.fromisoformat(last["time"])
                        if datetime.now() - last_time < timedelta(hours=4): continue

                    fg   = await get_fear_greed()
                    news = await get_news_sentiment(symbol)
                    liq_map = await get_liquidation_heatmap(symbol)
                    ai_review = await get_ai_review(a, a.get("futures_data"), news)

                    msg  = f"🚨 آلرت خودکار — سیگنال قوی!\n"
                    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
                    msg += build_analysis_message(a, capital, fg, None, None, news, None, liq_map, ai_review)

                    await bot.send_message(chat_id=int(chat_id), text=msg, disable_web_page_preview=True)

                    last_alerts[symbol] = {"direction": a["direction"], "time": datetime.now().isoformat()}
                    if a["expiry"]:
                        udata.setdefault("active_positions", {})[symbol] = {
                            "direction": a["direction"], "entry": a["entry"],
                            "stop_loss": a["stop_loss"], "tp1": a["tp1"], "expiry": a["expiry"],
                            "mode": sym_mode, "categories_used": a.get("categories_used", []),
                        }
                        stats = udata.setdefault("signal_stats", {"total":0,"win":0,"loss":0,"neutral_exit":0})
                        stats["total"] += 1
                        history = udata.setdefault("signal_history", [])
                        history.append({"symbol": symbol, "direction": a["direction"], "entry": a["entry"],
                                        "tp1": a["tp1"], "stop_loss": a["stop_loss"],
                                        "time": datetime.now().isoformat(), "result": "در انتظار", "mode": sym_mode})
                        if len(history) > 200: udata["signal_history"] = history[-200:]
                    save_data()
                    await asyncio.sleep(1)
                except Exception as e:
                    log.error(f"check_auto_alerts {chat_id}/{symbol}: {e}")
    except Exception as e:
        # حتی اگر کل تابع خطای غیرمنتظره بده، جاب بعدی APScheduler طبق زمان‌بندی دوباره اجرا می‌شه
        log.error(f"check_auto_alerts fatal: {e}")

# =========================
# PRICE ALERT CHECKER
# =========================
async def check_price_alerts(bot):
    for chat_id, udata in list(user_data.items()):
        alerts=udata.get("price_alerts",{}); triggered=[]
        for sym, a in list(alerts.items()):
            ticker=await get_ticker(sym)
            if not ticker: continue
            price=ticker["price"]
            if a.get("above") and price >= a["above"]:
                try:
                    await bot.send_message(chat_id=int(chat_id),
                        text=f"🔔 هشدار قیمت!\n{sym} به {format_price(price)} رسید\n✅ از {format_price(a['above'])} عبور کرد")
                except Exception as e: log.error(f"Alert: {e}")
                triggered.append((sym,"above"))
            if a.get("below") and price <= a["below"]:
                try:
                    await bot.send_message(chat_id=int(chat_id),
                        text=f"🔔 هشدار قیمت!\n{sym} به {format_price(price)} رسید\n❌ از {format_price(a['below'])} پایین‌تر رفت")
                except Exception as e: log.error(f"Alert: {e}")
                triggered.append((sym,"below"))
        for sym, side in triggered:
            if sym in alerts: alerts[sym].pop(side,None); (alerts.pop(sym,None) if not alerts.get(sym) else None)
        if triggered: save_data()

# =========================
# POSITION TRACKER
# =========================
async def check_expired_positions(bot):
    """
    هسته پوزیشن — کاملاً جدا از هسته آلرت خودکار.
    باگ قبلی: پوزیشن قبل از موفقیت دریافت قیمت pop می‌شد و در صورت خطا
    برای همیشه گم می‌شد. الان: فقط بعد از دریافت موفق قیمت و ثبت نتیجه pop می‌شه؛
    در غیر این صورت با شمارنده retry دوباره تلاش می‌شه (حداکثر ۳ بار) تا داده گم نشه.
    """
    now = datetime.now()
    for chat_id, udata in list(user_data.items()):
        positions = udata.get("active_positions", {})
        expired = [s for s, p in positions.items() if p.get("expiry") and now >= datetime.fromisoformat(p["expiry"])]
        for sym in expired:
            try:
                pos = positions[sym]
                ticker = await get_ticker(sym)
                if not ticker:
                    pos["retry"] = pos.get("retry", 0) + 1
                    if pos["retry"] >= 3:
                        # بعد از ۳ بار تلاش ناموفق، بدون داده قیمت به‌عنوان نامشخص ثبت و بسته می‌شه
                        positions.pop(sym, None)
                        log.warning(f"position {sym}/{chat_id} بدون قیمت نهایی بسته شد (retry exceeded)")
                    continue

                positions.pop(sym, None)  # فقط بعد از موفقیت حذف می‌شه
                current = ticker["price"]; entry = pos["entry"]; direction = pos["direction"]
                tp1 = pos.get("tp1"); sl = pos.get("stop_loss")
                if "LONG" in direction:
                    pnl_pct = (current-entry)/entry*100; hit_tp1 = tp1 and current >= tp1; hit_sl = sl and current <= sl
                else:
                    pnl_pct = (entry-current)/entry*100; hit_tp1 = tp1 and current <= tp1; hit_sl = sl and current >= sl

                stats = udata.setdefault("signal_stats", {"total":0,"win":0,"loss":0,"neutral_exit":0})
                is_win = False
                if hit_tp1:     result = "✅ موفق — به TP1 رسید!"; stats["win"] += 1; is_win = True
                elif hit_sl:    result = "❌ استاپ لاس خورد";       stats["loss"] += 1
                elif pnl_pct>0: result = f"🟡 سود جزئی ({pnl_pct:+.2f}%)"; stats["neutral_exit"] += 1; is_win = True
                else:           result = f"🟠 ضرر جزئی ({pnl_pct:+.2f}%)"; stats["neutral_exit"] += 1

                history = udata.setdefault("signal_history", [])
                for h in history:
                    if h.get("symbol") == sym and h.get("result") == "در انتظار":
                        h["result"] = result; h["exit_price"] = current; h["pnl_pct"] = round(pnl_pct, 2); break

                # امتیازدهی پویا: اندیکاتورهایی که در این سیگنال شرکت داشتن آپدیت می‌شن
                cats = pos.get("categories_used", [])
                if cats: update_indicator_performance(sym, cats, is_win)

                await bot.send_message(chat_id=int(chat_id),
                    text=(f"⏰ پوزیشن {sym} منقضی شد!\n\n"
                          f"جهت: {direction}\nورود: {format_price(entry)}\n"
                          f"قیمت الان: {format_price(current)}\nP&L: {pnl_pct:+.2f}%\n\nنتیجه: {result}"))
            except Exception as e:
                log.error(f"expired {chat_id}/{sym}: {e}")
        save_data()

# =========================
# CORE SENDER
# =========================
async def send_analysis(bot, chat_id, symbols):
    """هسته ارسال تحلیل — کاملاً جدا از هسته آلرت خودکار (خطا در این تابع هرگز آلرت خودکار رو قطع نمی‌کنه)"""
    if not user_data.get(chat_id,{}).get("active",True): return
    mode_key=user_data.get(chat_id,{}).get("trading_mode","short")
    capital=user_data.get(chat_id,{}).get("capital")
    auto_mode=user_data.get(chat_id,{}).get("auto_mode_enabled", True)

    fg = await get_fear_greed()

    for symbol in symbols:
        try:
            sym_mode = get_best_mode(symbol, default=mode_key) if auto_mode else mode_key
            a=await full_analysis(symbol, sym_mode)
            if not a:
                await bot.send_message(chat_id=int(chat_id), text=f"❌ خطا در تحلیل {symbol}")
                continue
            whale     = await get_whale_alerts(symbol)
            news      = await get_news_sentiment(symbol)
            price_cmp = await get_price_comparison(symbol)
            liq_map   = await get_liquidation_heatmap(symbol) if a["direction"]!="NEUTRAL ⚪" else None
            ai_review = await get_ai_review(a, a.get("futures_data"), news) if a["direction"]!="NEUTRAL ⚪" else None

            if a["direction"]!="NEUTRAL ⚪" and a["expiry"]:
                user_data[chat_id].setdefault("active_positions",{})[symbol] = {
                    "direction":a["direction"],"entry":a["entry"],
                    "stop_loss":a["stop_loss"],"tp1":a["tp1"],"expiry":a["expiry"],
                    "mode":sym_mode,"categories_used":a.get("categories_used",[]),}
                stats=user_data[chat_id].setdefault("signal_stats",{"total":0,"win":0,"loss":0,"neutral_exit":0})
                stats["total"]+=1
                history=user_data[chat_id].setdefault("signal_history",[])
                history.append({"symbol":symbol,"direction":a["direction"],"entry":a["entry"],
                                 "tp1":a["tp1"],"stop_loss":a["stop_loss"],
                                 "time":datetime.now().isoformat(),"result":"در انتظار","mode":sym_mode})
                if len(history)>200: user_data[chat_id]["signal_history"]=history[-200:]
                save_data()
            await bot.send_message(chat_id=int(chat_id),
                text=build_analysis_message(a, capital, fg, None, whale, news, price_cmp, liq_map, ai_review),
                disable_web_page_preview=True)
        except Exception as e:
            log.error(f"send_analysis {chat_id}/{symbol}: {e}")

# =========================
# JOB SYSTEM
# =========================
def schedule_user_job(app, chat_id):
    chat_id=str(chat_id); mode_key=user_data[chat_id].get("trading_mode","short")
    interval=user_data[chat_id].get("interval") or TRADING_MODES[mode_key]["interval_minutes"]
    if chat_id in user_jobs:
        try: user_jobs[chat_id].remove()
        except: pass
    job=scheduler.add_job(send_analysis,"interval",minutes=interval,
                          args=[app.bot,chat_id,user_data[chat_id]["symbols"]],
                          id=f"user_{chat_id}",replace_existing=True)
    user_jobs[chat_id]=job
    log.info(f"Job scheduled: {chat_id} | mode={mode_key} | interval={interval}m")

# =========================
# HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=str(update.effective_chat.id); init_user(chat_id)
    is_new, uinfo = register_user(update)
    if is_new and ADMIN_ID and chat_id!=ADMIN_ID:
        try:
            await context.bot.send_message(chat_id=int(ADMIN_ID),
                text=(f"👤 کاربر جدید!\n━━━━━━━━━━━━━━\n"
                      f"🆔 ID: {uinfo['chat_id']}\n📛 نام: {uinfo['first_name']} {uinfo['last_name']}\n"
                      f"👤 یوزر: {uinfo['username']}\n🕐 زمان: {datetime.now().strftime('%H:%M  %Y-%m-%d')}"))
        except Exception as e: log.error(f"Admin notify: {e}")
    text, markup = main_menu(chat_id)
    await update.message.reply_text(text, reply_markup=markup)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query=update.callback_query; await query.answer()
    chat_id=str(query.message.chat.id); data=query.data; init_user(chat_id)

    if data=="toggle_off":
        user_data[chat_id]["active"]=False; save_data()
        t,m=main_menu(chat_id); await query.edit_message_text("⏹ ارسال خودکار متوقف شد.\n\n"+t,reply_markup=m); return

    if data=="toggle_on":
        user_data[chat_id]["active"]=True; save_data()
        schedule_user_job(context.application,chat_id)
        t,m=main_menu(chat_id); await query.edit_message_text("▶️ ارسال خودکار فعال شد.\n\n"+t,reply_markup=m); return

    if data=="toggle_autoalert":
        cur=user_data[chat_id].get("auto_alert_enabled",True)
        user_data[chat_id]["auto_alert_enabled"]=not cur; save_data()
        t,m=main_menu(chat_id)
        status="✅ روشن" if not cur else "❌ خاموش"
        await query.edit_message_text(f"🔔 آلرت خودکار {status} شد.\n\n"+t,reply_markup=m); return

    if data=="do_analysis":
        await query.edit_message_text("⏳ در حال تحلیل چندلایه‌ای...")
        await send_analysis(context.bot,chat_id,user_data[chat_id]["symbols"])
        t,m=main_menu(chat_id)
        await context.bot.send_message(chat_id=int(chat_id),text=t,reply_markup=m); return

    if data=="menu_price":
        t,m=price_menu(chat_id); await query.edit_message_text(t,reply_markup=m); return

    if data.startswith("price_") and data!="price_custom":
        sym=data[6:]; ticker=await get_ticker(sym)
        await query.edit_message_text(build_price_message(sym,ticker))
        await asyncio.sleep(1); t,m=price_menu(chat_id)
        await context.bot.send_message(chat_id=int(chat_id),text=t,reply_markup=m); return

    if data=="price_custom":
        user_states[chat_id]="waiting_price_symbol"
        await query.edit_message_text("🔍 نام ارز رو بنویس (مثلاً: BTC)\n\nبرای لغو /cancel بزن"); return

    if data=="menu_mode":
        t,m=mode_menu(chat_id); await query.edit_message_text(t,reply_markup=m); return

    if data.startswith("setmode_"):
        mk=data[8:]
        if mk in TRADING_MODES:
            user_data[chat_id]["trading_mode"]=mk
            user_data[chat_id]["interval"]=TRADING_MODES[mk]["interval_minutes"]
            save_data(); schedule_user_job(context.application,chat_id)
        t,m=main_menu(chat_id)
        await query.edit_message_text(f"✅ بازه روی {TRADING_MODES[mk]['label']} تنظیم شد.\n\n"+t,reply_markup=m); return

    if data=="toggle_automode":
        cur=user_data[chat_id].get("auto_mode_enabled", True)
        user_data[chat_id]["auto_mode_enabled"]=not cur; save_data()
        t,m=mode_menu(chat_id); await query.edit_message_text(t,reply_markup=m); return

    if data=="view_bestmodes":
        symbols=user_data[chat_id].get("symbols",[])
        if not best_mode_data:
            text="⏳ بک‌تست خودکار هنوز اجرا نشده — تا ۱۲ ساعت دیگه یا بعد از اولین اجرا (چند دقیقه بعد از بالا اومدن ربات) نتیجه آماده می‌شه."
        else:
            text="📊 بهترین حالت هر ارز (بر اساس بک‌تست خودکار):\n\n"
            shown=False
            for sym in symbols:
                info=best_mode_data.get(sym)
                if not info:
                    text+=f"• {sym}: هنوز داده کافی نیست\n"; continue
                shown=True
                mk=info["best_mode"]; wr=info["win_rate"]; total=info["total"]
                updated=datetime.fromisoformat(info["updated"]).strftime("%Y-%m-%d %H:%M")
                text+=f"• {sym}: {TRADING_MODES[mk]['label']} | نرخ موفقیت {wr}% ({total} نمونه) | آپدیت: {updated}\n"
            if not shown: text+="\nهنوز برای هیچ‌کدوم از ارزهات داده کافی نیست."
        kb=[[InlineKeyboardButton("🔙 برگشت",callback_data="menu_mode")]]
        await context.bot.send_message(chat_id=int(chat_id), text=text, reply_markup=InlineKeyboardMarkup(kb)); return

    if data=="menu_capital":
        cap=user_data[chat_id].get("capital")
        cur=f"سرمایه فعلی: ${cap:,.0f}" if cap else "سرمایه‌ای تنظیم نشده"
        user_states[chat_id]="waiting_capital"
        await query.edit_message_text(f"💼 مدیریت سرمایه\n{cur}\n\nسرمایه کل رو به دلار بنویس:\n(مثلاً: 1000)\n\nبرای لغو /cancel بزن"); return

    if data=="menu_alerts":
        t,m=alerts_menu(chat_id); await query.edit_message_text(t,reply_markup=m); return

    if data.startswith("alert_set_"):
        sym=data[10:]; user_states[chat_id]=f"waiting_alert_{sym}"
        existing=user_data[chat_id].get("price_alerts",{}).get(sym,{}); info=""
        if existing.get("above"): info+=f"\nهشدار بالا: {format_price(existing['above'])}"
        if existing.get("below"): info+=f"\nهشدار پایین: {format_price(existing['below'])}"
        await query.edit_message_text(f"🔔 تنظیم هشدار برای {sym}{info}\n\nفرمت: above=50000 یا below=40000\n\nبرای لغو /cancel بزن"); return

    if data=="alert_clear_all":
        user_data[chat_id]["price_alerts"]={}; save_data()
        t,m=alerts_menu(chat_id); await query.edit_message_text("✅ همه هشدارها حذف شدن.\n\n"+t,reply_markup=m); return

    if data=="menu_add":
        t,m=add_symbol_menu(chat_id); await query.edit_message_text(t,reply_markup=m); return

    if data.startswith("add_") and data!="add_custom":
        sym=data[4:]
        if sym not in user_data[chat_id]["symbols"]:
            user_data[chat_id]["symbols"].append(sym); save_data(); schedule_user_job(context.application,chat_id)
        t,m=add_symbol_menu(chat_id); await query.edit_message_text(f"✅ {sym} اضافه شد!\n\n"+t,reply_markup=m); return

    if data=="add_custom":
        user_states[chat_id]="waiting_custom_symbol"
        await query.edit_message_text("✏️ نام ارز رو بنویس (مثلاً: LINK)\n\nبرای لغو /cancel بزن"); return

    if data=="menu_remove":
        t,m=remove_symbol_menu(chat_id); await query.edit_message_text(t,reply_markup=m); return

    if data.startswith("rem_"):
        sym=data[4:]; syms=user_data[chat_id]["symbols"]
        if len(syms)==1:
            await query.answer("حداقل یک ارز باید فعال باشد!",show_alert=True); return
        if sym in syms:
            syms.remove(sym); user_data[chat_id].get("active_positions",{}).pop(sym,None)
            save_data(); schedule_user_job(context.application,chat_id)
        t,m=remove_symbol_menu(chat_id); await query.edit_message_text(f"🗑 {sym} حذف شد.\n\n"+t,reply_markup=m); return

    if data=="menu_list":
        syms=user_data[chat_id]["symbols"]
        text="📋 ارزهای فعال:\n\n"+"\n".join(f"  • {s}" for s in syms)
        kb=[[InlineKeyboardButton("🔙 برگشت",callback_data="back_main")]]
        await query.edit_message_text(text,reply_markup=InlineKeyboardMarkup(kb)); return

    if data=="menu_positions" or data=="refresh_positions":
        if data=="refresh_positions":
            await query.edit_message_text("⏳ در حال بروزرسانی پوزیشن‌های زنده...")
        positions=user_data[chat_id].get("active_positions",{})
        stats=user_data[chat_id].get("signal_stats",{"total":0,"win":0,"loss":0,"neutral_exit":0})
        now_str = datetime.now().strftime("%H:%M:%S")
        text=f"📁 پوزیشن‌های فعال (زنده — {now_str}):\n\n" if positions else "هیچ پوزیشن فعالی نداری\n\n"
        for sym, pos in positions.items():
            exp=datetime.fromisoformat(pos["expiry"]).strftime("%H:%M")
            live_ticker = await get_ticker(sym)
            direction = pos["direction"]; entry = pos["entry"]
            if live_ticker:
                current = live_ticker["price"]
                pnl_pct = (current-entry)/entry*100 if "LONG" in direction else (entry-current)/entry*100
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                live_line = f"  💵 قیمت لحظه‌ای: {format_price(current)}  |  {pnl_emoji} PnL: {pnl_pct:+.2f}%\n"
            else:
                live_line = "  ⚠️ قیمت لحظه‌ای در دسترس نیست\n"
            text+=(f"• {sym} | {direction}\n"
                   f"  ورود: {format_price(pos['entry'])} | SL: {format_price(pos['stop_loss'])}\n"
                   f"  TP1: {format_price(pos['tp1'])} | انقضا: {exp}\n"
                   f"{live_line}\n")
        total=stats["total"]; win=stats["win"]; loss=stats["loss"]
        wr=round(win/total*100,1) if total>0 else 0
        text+=f"📊 آمار کلی: کل {total} | ✅{win} | ❌{loss} | نرخ موفقیت: {wr}%"
        kb=[[InlineKeyboardButton("🔄 بروزرسانی زنده",callback_data="refresh_positions")],
            [InlineKeyboardButton("🔙 برگشت",callback_data="back_main")]]
        await query.edit_message_text(text,reply_markup=InlineKeyboardMarkup(kb)); return

    if data=="menu_history":
        history=user_data[chat_id].get("signal_history",[])
        stats=user_data[chat_id].get("signal_stats",{"total":0,"win":0,"loss":0,"neutral_exit":0})
        kb=[[InlineKeyboardButton("🔙 برگشت",callback_data="back_main")]]
        await query.edit_message_text(build_history_message(history,stats),reply_markup=InlineKeyboardMarkup(kb)); return

    if data=="menu_performance":
        report=build_performance_report(chat_id)
        kb=[[InlineKeyboardButton("🔙 برگشت",callback_data="back_main")]]
        await query.edit_message_text(report,reply_markup=InlineKeyboardMarkup(kb)); return

    if data=="menu_backtest":
        symbols = user_data[chat_id].get("symbols",[]); mode_key = user_data[chat_id].get("trading_mode","short")
        keyboard = []
        for sym in symbols[:6]:
            keyboard.append([InlineKeyboardButton(f"🧪 {sym.replace('USDT','')}", callback_data=f"bt_{sym}")])
        keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
        await query.edit_message_text(
            f"🧪 بک‌تست — مود: {TRADING_MODES[mode_key]['label']}\n\nکدوم ارز رو بک‌تست کنم؟\n⚠️ ممکنه ۱۵-۳۰ ثانیه طول بکشه",
            reply_markup=InlineKeyboardMarkup(keyboard)); return

    if data.startswith("bt_"):
        sym = data[3:]; mode_key = user_data[chat_id].get("trading_mode","short")
        await query.edit_message_text(f"⏳ در حال بک‌تست {sym} روی {TRADING_MODES[mode_key]['label']}...\nلطفاً صبر کن")
        result = await run_backtest(sym, mode_key, periods=15)
        if not result:
            await context.bot.send_message(chat_id=int(chat_id), text="❌ داده کافی برای بک‌تست وجود نداره")
        else:
            r = result
            emoji = "🟢" if r["win_rate"]>=55 else ("🟡" if r["win_rate"]>=45 else "🔴")
            msg  = f"🧪 نتیجه بک‌تست\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"📊 {r['symbol']} | {TRADING_MODES[r['mode']]['label']}\n\n"
            msg += f"🔢 کل سیگنال:    {r['total']}\n"
            msg += f"✅ موفق:         {r['wins']}\n"
            msg += f"❌ ناموفق:       {r['losses']}\n"
            msg += f"🟡 جزئی:         {r['partial']}\n"
            msg += f"{emoji} نرخ موفقیت: {r['win_rate']}%\n\n"
            msg += f"📋 آخرین سیگنال‌ها:\n"
            for res in r["results"][-5:]:
                oc_emoji = "✅" if res["outcome"]=="win" else ("❌" if res["outcome"]=="loss" else "🟡")
                msg += f"  {oc_emoji} {res['direction']} @ {res['price']} (امتیاز:{res['score']})\n"
            msg += f"\n⚠️ بک‌تست گذشته‌نگر است و ضمانت سود ندارد"
            await context.bot.send_message(chat_id=int(chat_id), text=msg)
        t,m=main_menu(chat_id)
        await context.bot.send_message(chat_id=int(chat_id),text=t,reply_markup=m); return

    if data=="menu_hotcoins":
        await query.edit_message_text("⏳ در حال اسکن ارزهای پرنوسان فیوچرز...")
        hot = await scan_hot_coins()
        if not hot:
            text = "❌ در حال حاضر ارز پرنوسان قابل‌توجهی پیدا نشد"
        else:
            text = "🆕 ارزهای داغ و پرنوسان (۲۴h)\n━━━━━━━━━━━━━━━━━━━━\n\n"
            for c in hot:
                text += f"• {c['symbol']} | {c['change_pct']:+.2f}%  |  حجم: ${c['quote_volume']:,.0f}\n"
            text += "\n⚠️ نوسان بالا = فرصت و ریسک بالا هر دو؛ حجم پایین‌تر معامله کن و حتماً SL بذار.\n"
            text += "ℹ️ این تشخیص بر اساس تغییر قیمت/حجم غیرعادیه، نه تاریخ دقیق لیست‌شدن در صرافی."
        kb=[[InlineKeyboardButton("🔙 برگشت",callback_data="back_main")]]
        await context.bot.send_message(chat_id=int(chat_id), text=text, reply_markup=InlineKeyboardMarkup(kb)); return

    if data=="menu_whale":
        symbols = user_data[chat_id].get("symbols",[]); keyboard=[]
        for sym in symbols[:6]:
            keyboard.append([InlineKeyboardButton(f"🐋 {sym.replace('USDT','')}", callback_data=f"whale_{sym}")])
        keyboard.append([InlineKeyboardButton("🔙 برگشت", callback_data="back_main")])
        await query.edit_message_text("🐋 بررسی معاملات بزرگ (نهنگ‌های صرافی):", reply_markup=InlineKeyboardMarkup(keyboard)); return

    if data.startswith("whale_"):
        sym = data[6:]
        await query.edit_message_text(f"⏳ در حال بررسی معاملات بزرگ {sym}...")
        wa = await get_whale_alerts(sym)
        if not wa:
            msg = f"❌ داده معاملات بزرگ برای {sym} در دسترس نیست"
        else:
            alert_emoji = "🚨" if wa.get("alert") else ("🟢" if wa.get("bullish") else ("🔴" if wa.get("bearish") else "🟡"))
            status = "هشدار نهنگ!" if wa["alert"] else ("فشار خرید" if wa["bullish"] else ("فشار فروش" if wa.get("bearish") else "عادی"))
            msg  = f"🐋 معاملات بزرگ (نهنگ) — {sym}\n━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"{alert_emoji} وضعیت: {status}\n\n"
            msg += f"📊 تعداد معاملات بزرگ:  {wa.get('large_tx',0)}\n"
            msg += f"🟢 حجم خرید نهنگ‌ها:   ${wa.get('buy_usd',0):,.0f}\n"
            msg += f"🔴 حجم فروش نهنگ‌ها:   ${wa.get('sell_usd',0):,.0f}\n\n"
            msg += "ℹ️ این آمار از معاملات بزرگ (≥$100k) داخل صرافیه، نه رصد ولت آنچین.\n"
            if wa["alert"]: msg += "⚠️ فشار قابل‌توجه نهنگ‌ها — احتمال حرکت قیمتی\n"
        await context.bot.send_message(chat_id=int(chat_id), text=msg)
        t,m=main_menu(chat_id)
        await context.bot.send_message(chat_id=int(chat_id),text=t,reply_markup=m); return

    if data=="menu_interval":
        keyboard=[[InlineKeyboardButton("۱۵ دقیقه",callback_data="setint_15"),
                   InlineKeyboardButton("۳۰ دقیقه",callback_data="setint_30"),
                   InlineKeyboardButton("۱ ساعت",callback_data="setint_60")],
                  [InlineKeyboardButton("۲ ساعت",callback_data="setint_120"),
                   InlineKeyboardButton("۴ ساعت",callback_data="setint_240"),
                   InlineKeyboardButton("۸ ساعت",callback_data="setint_480")],
                  [InlineKeyboardButton("۱۲ ساعت",callback_data="setint_720"),
                   InlineKeyboardButton("۲۴ ساعت",callback_data="setint_1440")],
                  [InlineKeyboardButton("🔙 برگشت",callback_data="back_main")]]
        await query.edit_message_text("⏱ بازه ارسال خودکار:",reply_markup=InlineKeyboardMarkup(keyboard)); return

    if data.startswith("setint_"):
        mins=int(data[7:]); user_data[chat_id]["interval"]=mins; save_data()
        schedule_user_job(context.application,chat_id)
        t,m=main_menu(chat_id); await query.edit_message_text(f"✅ بازه ارسال {mins} دقیقه.\n\n"+t,reply_markup=m); return

    if data=="back_main":
        t,m=main_menu(chat_id); await query.edit_message_text(t,reply_markup=m); return

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=str(update.effective_chat.id); text=update.message.text or ""; state=user_states.get(chat_id)

    if state=="waiting_price_symbol":
        user_states.pop(chat_id,None); sym=text.strip().upper()
        if not sym.endswith("USDT"): sym+="USDT"
        ticker=await get_ticker(sym)
        await update.message.reply_text(build_price_message(sym,ticker)); return

    if state=="waiting_capital":
        user_states.pop(chat_id,None)
        try:
            capital=float(text.strip().replace(",","").replace("$",""))
            if capital<=0: raise ValueError
            user_data[chat_id]["capital"]=capital; save_data()
            t,m=main_menu(chat_id)
            await update.message.reply_text(f"✅ سرمایه ${capital:,.0f} ثبت شد!\n\n"+t,reply_markup=m)
        except: await update.message.reply_text("❌ مقدار معتبر نیست. عدد دلاری بنویس (مثلاً: 1000)")
        return

    if state and state.startswith("waiting_alert_"):
        sym=state[14:]; user_states.pop(chat_id,None)
        try:
            parts=text.strip().lower().split()
            alerts=user_data[chat_id].setdefault("price_alerts",{}).setdefault(sym,{})
            for part in parts:
                if "above=" in part: alerts["above"]=float(part.split("=")[1])
                elif "below=" in part: alerts["below"]=float(part.split("=")[1])
            save_data(); t,m=alerts_menu(chat_id)
            await update.message.reply_text(f"✅ هشدار برای {sym} ثبت شد!\n\n"+t,reply_markup=m)
        except: await update.message.reply_text("❌ فرمت اشتباه. مثال: above=50000 یا below=40000")
        return

    if state=="waiting_custom_symbol":
        user_states.pop(chat_id,None)
        await update.message.reply_text("⏳ در حال بررسی ارز...")
        valid=await validate_symbol(text.strip())
        if valid:
            if valid not in user_data[chat_id]["symbols"]:
                user_data[chat_id]["symbols"].append(valid); save_data(); schedule_user_job(context.application,chat_id)
            t,m=main_menu(chat_id)
            await update.message.reply_text(f"✅ {valid} اضافه شد!\n\n"+t,reply_markup=m)
        else: await update.message.reply_text(f"❌ ارز «{text}» پیدا نشد.")
        return

    me=await context.bot.get_me()
    if me.username and f"@{me.username}" in text:
        init_user(chat_id)
        await update.message.reply_text("⏳ در حال تحلیل...")
        await send_analysis(context.bot,chat_id,user_data[chat_id]["symbols"])

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=str(update.effective_chat.id); user_states.pop(chat_id,None)
    t,m=main_menu(chat_id); await update.message.reply_text("لغو شد.\n\n"+t,reply_markup=m)

async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    backup_data(); await update.message.reply_text("✅ بکاپ ساخته شد")

async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id=str(update.effective_chat.id)
    if ADMIN_ID and chat_id!=ADMIN_ID:
        await update.message.reply_text("⛔ دسترسی ندارید"); return
    ul=load_users_list()
    if not ul: await update.message.reply_text("📭 هنوز کاربری ثبت نشده"); return
    total=len(ul); msg=f"👥 لیست کاربران ({total} نفر)\n━━━━━━━━━━━━━━━━━━━━\n\n"
    for i,(cid,u) in enumerate(ul.items(),1):
        first=datetime.fromisoformat(u["first_seen"]).strftime("%Y/%m/%d")
        last=datetime.fromisoformat(u["last_seen"]).strftime("%m/%d %H:%M")
        name=f"{u['first_name']} {u['last_name']}".strip() or "بدون نام"
        msg+=(f"{i}. {name}\n   🆔 {cid}\n   👤 {u['username']}\n"
              f"   📅 عضو از: {first}  |  آخرین: {last}\n   🔢 start: {u['count']}\n\n")
        if len(msg)>3500 and i<total: msg+=f"... و {total-i} نفر دیگه\n"; break
    await update.message.reply_text(msg)

# =========================
# STARTUP
# =========================
async def post_init(app: Application):
    global session
    session=aiohttp.ClientSession()
    scheduler.start(); load_data(); load_adaptive_weights(); load_best_mode()
    for chat_id in user_data:
        if user_data[chat_id].get("active",True):
            try: schedule_user_job(app,chat_id)
            except Exception as e: log.error(f"Schedule error {chat_id}: {e}")
    # job های سیستمی — هسته پوزیشن و هسته آلرت کاملاً جدا از هم هستن
    scheduler.add_job(check_expired_positions,"interval",minutes=5,  args=[app.bot])  # هسته پوزیشن
    scheduler.add_job(check_price_alerts,     "interval",minutes=2,  args=[app.bot])
    scheduler.add_job(check_auto_alerts,      "interval",minutes=30, args=[app.bot])  # هسته آلرت خودکار
    scheduler.add_job(backup_data,            "interval",minutes=10)
    scheduler.add_job(clear_expired_cache,    "interval",minutes=10)
    # بک‌تست خودکار هر ۱۲ ساعت + یک اجرای اولیه ۲ دقیقه بعد از بالا اومدن ربات
    scheduler.add_job(auto_backtest_all_symbols, "interval", hours=12, args=[app],
                       next_run_time=datetime.now()+timedelta(minutes=2))
    if not GROQ_API_KEY:
        log.warning("⚠️ GROQ_API_KEY تنظیم نشده — توضیح سیگنال‌ها قانون‌محور (fallback) خواهد بود")
    log.info("✅ Bot v3.0 (Futures Edition) started — موتورهای تحلیل فیوچرز فعال")

def main():
    app=Application.builder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start",  start))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("backup", backup_command))
    app.add_handler(CommandHandler("users",  users_command))
    app.add_handler(CallbackQueryHandler(button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.run_polling(drop_pending_updates=True)

if __name__=="__main__":
    main()
