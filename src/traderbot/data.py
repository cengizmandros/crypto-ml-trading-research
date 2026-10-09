"""Market data: Binance public archive (data.binance.vision) + public REST top-up.

No API key is used anywhere. Storage: one Parquet file per (interval, symbol).
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from .config import LEVERAGED_SUFFIXES, PARQUET, QUOTE, START, STABLE_BASES

log = logging.getLogger(__name__)

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE = "https://data.binance.vision"
REST = "https://api.binance.com/api/v3/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
INTERVAL_MS = {"1h": 3_600_000, "1d": 86_400_000}

_session = requests.Session()
_session.headers["User-Agent"] = "trader-bot-research/1.0"


def _get(url: str, params: dict | None = None, retries: int = 5) -> requests.Response:
    for i in range(retries):
        try:
            r = _session.get(url, params=params, timeout=30)
            if r.status_code in (429, 418) or r.status_code >= 500:
                time.sleep(2 ** i)
                continue
            return r
        except requests.RequestException:
            time.sleep(2 ** i)
    raise RuntimeError(f"failed: {url}")


def _s3_list(prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    """Return (common prefixes, keys) under a prefix, following pagination."""
    prefixes, keys, marker = [], [], ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        if delimiter:
            params["delimiter"] = delimiter
        txt = _get(S3, params).text
        p = re.findall(r"<Prefix>([^<]*)</Prefix>", txt)[1:]  # first is the query itself
        k = re.findall(r"<Key>([^<]*)</Key>", txt)
        prefixes += p
        keys += k
        if "<IsTruncated>true" not in txt:
            return prefixes, keys
        nm = re.search(r"<NextMarker>([^<]*)</NextMarker>", txt)
        marker = nm.group(1) if nm else (k[-1] if k else p[-1])


def is_tradable_symbol(sym: str) -> bool:
    if not sym.endswith(QUOTE):
        return False
    base = sym[: -len(QUOTE)]
    if not base or base in STABLE_BASES:
        return False
    if any(base.endswith(s) and len(base) > len(s) for s in LEVERAGED_SUFFIXES):
        return False
    return True


def list_symbols() -> list[str]:
    """All USDT spot symbols ever archived, including delisted ones."""
    prefixes, _ = _s3_list("data/spot/monthly/klines/")
    syms = [p.rstrip("/").split("/")[-1] for p in prefixes]
    return sorted(s for s in syms if is_tradable_symbol(s))


def _parse_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[:, :12].copy()
    df.columns = COLS
    for c in ("open_time", "close_time"):
        v = pd.to_numeric(df[c]).astype("int64")
        # Binance switched spot archives to microseconds in 2025.
        v = v.where(v < 10**14, v // 1000)
        df[c] = v
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64")
    df = df.drop(columns="ignore")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def _download_zip(key: str) -> pd.DataFrame | None:
    r = _get(f"{ARCHIVE}/{key}")
    if r.status_code != 200:
        return None
    blob = r.content
    c = _get(f"{ARCHIVE}/{key}.CHECKSUM")
    if c.status_code == 200:
        expected = c.text.split()[0]
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {key}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if isinstance(raw.iloc[0, 0], str):  # some files carry a header row
        raw = raw.iloc[1:]
    return _parse_kline_frame(raw)


def _rest_klines(symbol: str, interval: str, start_ms: int) -> pd.DataFrame:
    out, cur = [], start_ms
    now_ms = int(time.time() * 1000)
    while cur < now_ms:
        r = _get(REST, {"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000})
        if r.status_code != 200:
            break  # delisted / unknown symbol
        rows = r.json()
        if not rows:
            break
        out.append(pd.DataFrame(rows))
        cur = rows[-1][0] + INTERVAL_MS[interval]
        if len(rows) < 1000:
            break
        time.sleep(0.1)
    if not out:
        return pd.DataFrame()
    return _parse_kline_frame(pd.concat(out, ignore_index=True))


def path_for(symbol: str, interval: str) -> Path:
    p = PARQUET / interval
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{symbol}.parquet"


def load(symbol: str, interval: str) -> pd.DataFrame:
    p = path_for(symbol, interval)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def update_symbol(symbol: str, interval: str, start: str = START) -> int:
    """Incrementally bring one symbol up to date. Returns number of bars stored."""
    existing = load(symbol, interval)
    start_ts = pd.Timestamp(start, tz="UTC")
    last = existing["ts"].max() if len(existing) else None

    frames = [existing] if len(existing) else []
    _, keys = _s3_list(f"data/spot/monthly/klines/{symbol}/{interval}/", delimiter=None)
    months = sorted(k for k in keys if k.endswith(".zip"))
    for key in months:
        m = re.search(r"(\d{4}-\d{2})\.zip$", key).group(1)
        month_start = pd.Timestamp(m + "-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        if month_end <= start_ts:
            continue
        if last is not None and month_end - pd.Timedelta(milliseconds=INTERVAL_MS[interval]) <= last:
            continue
        df = _download_zip(key)
        if df is not None:
            frames.append(df)

    cur = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    resume = cur["ts"].max() + pd.Timedelta(milliseconds=INTERVAL_MS[interval]) if len(cur) else start_ts
    tail = _rest_klines(symbol, interval, int(resume.timestamp() * 1000))
    if len(tail):
        cur = pd.concat([cur, tail], ignore_index=True) if len(cur) else tail
    if not len(cur):
        return 0

    now = pd.Timestamp.now(tz="UTC")
    close_ts = pd.to_datetime(cur["close_time"], unit="ms", utc=True)
    cur = cur[(cur["ts"] >= start_ts) & (close_ts < now)]  # drop the still-forming bar
    cur = cur.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    cur = clean(cur)
    cur.to_parquet(path_for(symbol, interval), index=False)
    return len(cur)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic integrity: positive prices, high>=low, OHLC consistency."""
    ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["high"] >= df["low"])
    df = df[ok].copy()
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def update_many(symbols: list[str], interval: str, workers: int = 8) -> dict[str, int]:
    res: dict[str, int] = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(update_symbol, s, interval): s for s in symbols}
        for f in as_completed(futs):
            s = futs[f]
            try:
                res[s] = f.result()
            except Exception as e:  # keep going; report at end
                log.warning("%s %s failed: %s", s, interval, e)
                res[s] = -1
    return res


RELIST_GAP = pd.Timedelta(days=7)


def base_symbol(name: str) -> str:
    """'LUNAUSDT@2022-05-31' -> 'LUNAUSDT'."""
    return name.split("@")[0]


def load_segments(symbol: str, interval: str) -> dict[str, pd.DataFrame]:
    """Split a ticker's history at gaps > 7 days.

    Binance reuses tickers (e.g. LUNAUSDT: Terra Classic until 2022-05-13, Terra 2.0
    from 2022-05-31). A long gap means a different asset; the later segment gets the
    name 'SYMBOL@YYYY-MM-DD' so it is never stitched to the old price series.
    """
    df = load(symbol, interval)
    if not len(df):
        return {}
    seg_id = (df["ts"].diff() > RELIST_GAP).cumsum()
    out = {}
    for k, g in df.groupby(seg_id):
        name = symbol if k == 0 else f"{symbol}@{g['ts'].iloc[0].date()}"
        out[name] = g.reset_index(drop=True)
    return out


def _segments_for(names: list[str], interval: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    wanted = set(names)
    for b in sorted({base_symbol(n) for n in names}):
        for name, g in load_segments(b, interval).items():
            if name in wanted or b in wanted and name.startswith(b):
                out[name] = g
    return out


def panel(symbols: list[str], interval: str, field: str = "close") -> pd.DataFrame:
    """Wide frame ts x segment-name for one field (relisted tickers split)."""
    segs = _segments_for(symbols, interval)
    return pd.DataFrame({n: g.set_index("ts")[field] for n, g in segs.items()}).sort_index()


def load_long(symbols: list[str], interval: str) -> pd.DataFrame:
    segs = _segments_for(symbols, interval)
    out = [g.assign(symbol=n) for n, g in segs.items()]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
