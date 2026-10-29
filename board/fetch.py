"""行情抓取：郑商所单合约日线（新浪）、ICE 原糖连续（新浪外盘）、美元兑人民币（FRED）。

所有数据落盘到 data/raw/，后续计算只读本地 CSV，保证结果可复查。
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
}
SINA = "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_x=/"


def _jsonp(text: str):
    m = re.search(r"\((\[.*\])\)", text, re.S)
    if not m:
        return []
    return json.loads(m.group(1))


def _get(url: str, tries: int = 3) -> str:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def contract_list(cfg: dict) -> list[str]:
    out = []
    first, last = cfg["first_contract"], cfg["last_contract"]
    for yy in range(first // 100, last // 100 + 1):
        for mm in cfg["contract_months"]:
            n = yy * 100 + mm
            if first <= n <= last:
                out.append(f"{cfg['code']}{n:04d}")
    return out


def fetch_domestic(symbol: str) -> pd.DataFrame:
    rows = _jsonp(_get(SINA + f"InnerFuturesNewService.getDailyKLine?symbol={symbol}"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(
        columns={"d": "date", "o": "open", "h": "high", "l": "low", "c": "close",
                 "v": "volume", "p": "oi", "s": "settle"}
    )
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume", "oi", "settle"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["contract"] = symbol
    return df


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=max&interval=1d"
    j = json.loads(_get(url))["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({"date": pd.to_datetime(j["timestamp"], unit="s").normalize(),
                       "open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "volume": q.get("volume")})
    return df.dropna(subset=["close"]).drop_duplicates("date", keep="last")


def fetch_foreign_sina(symbol: str) -> pd.DataFrame:
    url = SINA + f"GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={symbol}&source=web"
    rows = _jsonp(_get(url))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_fx_fred(series: str) -> pd.DataFrame:
    text = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "fx"]
    df["date"] = pd.to_datetime(df["date"])
    df["fx"] = pd.to_numeric(df["fx"], errors="coerce")
    return df.dropna()


def main(cfg_path: str = "config/sugar.yaml") -> None:
    cfg = yaml.safe_load((ROOT / cfg_path).read_text(encoding="utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    report = {}

    frames = []
    for sym in contract_list(cfg):
        try:
            df = fetch_domestic(sym)
            report[sym] = len(df)
            if len(df):
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            report[sym] = f"ERR {e}"
        time.sleep(0.4)
    if frames:
        pd.concat(frames).to_csv(RAW / f"{cfg['code']}_contracts.csv", index=False)

    # 外盘：取历史最长的来源；最近 30 天与新浪对照，差异写进报告
    cands = {}
    for name, fn in [("yahoo", lambda: fetch_yahoo(cfg["foreign"]["yahoo"])),
                     ("sina", lambda: fetch_foreign_sina(cfg["foreign"]["sina_symbol"]))]:
        try:
            cands[name] = fn()
            report[f"foreign_{name}"] = f"{len(cands[name])} 行，{cands[name]['date'].min():%F} → {cands[name]['date'].max():%F}"
        except Exception as e:  # noqa: BLE001
            report[f"foreign_{name}"] = f"ERR {e}"
    if cands:
        best = max(cands, key=lambda k: len(cands[k]))
        cands[best].to_csv(RAW / "foreign.csv", index=False)
        report["foreign_used"] = best
        if len(cands) == 2:
            m = cands["yahoo"].merge(cands["sina"], on="date", suffixes=("_y", "_s")).tail(30)
            report["foreign_check_max_abs_diff_30d"] = float((m["close_y"] - m["close_s"]).abs().max())

    fx_sources = [("yahoo", lambda: fetch_yahoo(cfg["fx"]["yahoo"])[["date", "close"]].rename(columns={"close": "fx"})),
                  ("fred", lambda: fetch_fx_fred(cfg["fx"]["fred_series"]))]
    for name, fn in fx_sources:
        try:
            fx = fn().dropna()
            fx.to_csv(RAW / "fx.csv", index=False)
            report["fx"] = f"{name}: {len(fx)} 行，{fx['date'].min():%F} → {fx['date'].max():%F}"
            break
        except Exception as e:  # noqa: BLE001
            report[f"fx_{name}"] = f"ERR {e}"

    (RAW / "fetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not frames:
        sys.exit("没有抓到任何郑糖合约数据")


if __name__ == "__main__":
    main(*sys.argv[1:])
