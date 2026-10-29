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
            r = requests.get(url, headers=HEADERS, timeout=20)
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


def fetch_foreign(symbol: str) -> pd.DataFrame:
    url = SINA + f"GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={symbol}&source=web"
    rows = _jsonp(_get(url))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_fx(series: str) -> pd.DataFrame:
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

    try:
        f = fetch_foreign(cfg["foreign"]["sina_symbol"])
        f.to_csv(RAW / "foreign.csv", index=False)
        report["foreign"] = len(f)
    except Exception as e:  # noqa: BLE001
        report["foreign"] = f"ERR {e}"

    try:
        fx = fetch_fx(cfg["fx"]["fred_series"])
        fx.to_csv(RAW / "fx.csv", index=False)
        report["fx"] = len(fx)
    except Exception as e:  # noqa: BLE001
        report["fx"] = f"ERR {e}"

    (RAW / "fetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not frames:
        sys.exit("没有抓到任何郑糖合约数据")


if __name__ == "__main__":
    main(*sys.argv[1:])
