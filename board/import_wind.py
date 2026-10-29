"""把 Wind 主力连续（不复权）表转换成回测用的复权连续序列。

换月识别：当日"前结算价"≠ 上一行"结算价"，说明这一行已切换到新合约；
跳空比例 = 新合约前结算价 / 旧合约结算价。之前的价格整体乘以该比例（比例复权），
收益率不受影响；raw_close 保留原始价格，用于进口利润等需要真实价位的计算。

用法：python -m board.import_wind <xlsx> <sheet> <code>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = {"交易日期": "date", "开盘价": "open", "收盘价": "close", "最高价": "high", "最低价": "low",
        "成交量(手)": "volume", "持仓量(手)": "oi", "结算价": "settle", "前结算价": "pre_settle"}


def convert(xlsx: str, sheet: str, code: str) -> pd.DataFrame:
    d = pd.read_excel(xlsx, sheet_name=sheet).rename(columns=COLS)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    # 最新一行若无开高低（盘中或未收盘），丢弃；历史上的无成交日用收盘价补齐
    while len(d) and pd.isna(d.iloc[-1]["open"]):
        d = d.iloc[:-1]
    for c in ("open", "high", "low"):
        d[c] = d[c].fillna(d["close"])
    d["roll"] = (d["pre_settle"] - d["settle"].shift()).abs() > 0.5
    ratio = np.where(d["roll"], d["pre_settle"] / d["settle"].shift(), 1.0)
    # 第 i 行之前的所有价格乘以第 i 行之后（含）所有换月比例之积
    adj = pd.Series(ratio[::-1]).cumprod()[::-1].shift(-1).fillna(1.0).to_numpy()
    d["raw_close"] = d["close"]
    for c in ("open", "high", "low", "close", "settle"):
        d[c] = d[c] * adj
    d["contract"] = f"{code}_MAIN"
    return d[["date", "open", "high", "low", "close", "volume", "oi", "settle", "raw_close", "roll", "contract"]]


if __name__ == "__main__":
    xlsx, sheet, code = sys.argv[1:4]
    out = convert(xlsx, sheet, code)
    p = ROOT / "data" / "raw" / f"{code}_main.csv"
    out.to_csv(p, index=False)
    print(f"{p}: {len(out)} 行，{out['date'].min():%F} → {out['date'].max():%F}，换月 {int(out['roll'].sum())} 次")
