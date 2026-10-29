"""因子与打分。每个因子输出 −2..+2，规则全部写死在这里，参数来自配置。

关键约定（避免未来函数）：
- t 日的分数只用 t 日收盘及之前的数据；执行在 t+1 日开盘。
- 指标在各合约自己的序列上计算，再按 t 日主力取值，避免换月跳空污染均线。
- ICE 和汇率只用 t 日之前（不含 t）已收盘的值：ICE 的 t 日结算在北京时间 t+1 凌晨才出。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clip2(x):
    return np.clip(x, -2.0, 2.0)


def per_contract_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c, h, l = df["close"], df["high"], df["low"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(p["atr_n"]).mean()
    for n in (5, 10, 20, 60):
        df[f"ma{n}"] = c.rolling(n).mean()
    df["ma20_slope"] = df["ma20"] - df["ma20"].shift(5)
    df["mom"] = (c - c.shift(p["momentum_n"])) / df["atr"]
    df["oi_chg"] = df["oi"] / df["oi"].shift(p["oi_n"]) - 1
    df["px_chg_atr"] = (c - c.shift(p["oi_n"])) / df["atr"]
    n = p["channel_n"]
    df["ch_hi"] = h.shift(1).rolling(n).max()
    df["ch_lo"] = l.shift(1).rolling(n).min()
    # 前瞻收益：t+1 开盘进场，持有到 t+h 收盘（同一合约，不跨换月）
    for hz in (1, 5, 20):
        df[f"fwd{hz}"] = c.shift(-hz) / df["open"].shift(-1) - 1
    df["next_open"] = df["open"].shift(-1)
    df["next_close"] = c.shift(-1)
    return df


def pick_dominant(panel: pd.DataFrame, months_order: list[str]) -> pd.DataFrame:
    """按 t 日持仓最大认定主力；主力只向后换，不回到更早到期的合约。
    次主力 = 晚于主力的合约里持仓最大者（用于月差）。"""
    rank = {c: i for i, c in enumerate(months_order)}
    oi = panel.pivot_table(index="date", columns="contract", values="oi")
    dom, nxt, cur = [], [], None
    for d, row in oi.iterrows():
        row = row.dropna()
        row = row[row > 0]
        if row.empty:
            dom.append(None); nxt.append(None); continue
        cand = row.idxmax()
        if cur is None or cur not in row.index or rank[cand] > rank[cur]:
            cur = cand
        later = row[[c for c in row.index if rank[c] > rank[cur]]]
        dom.append(cur)
        nxt.append(later.idxmax() if len(later) else None)
    return pd.DataFrame({"date": oi.index, "dominant": dom, "next": nxt})


def score_frame(dom_rows: pd.DataFrame, panel: pd.DataFrame, foreign: pd.DataFrame,
                fx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = cfg["score"]["params"]
    d = dom_rows.copy()

    # ---- 1 趋势：四个布尔条件计数 − 2
    pts = ((d["close"] > d["ma20"]).astype(int) + (d["close"] > d["ma60"]).astype(int)
           + (d["ma20"] > d["ma60"]).astype(int) + (d["ma20_slope"] > 0).astype(int))
    d["f_trend"] = (pts - 2).where(d["ma60"].notna())

    # ---- 2 动量：20 日涨跌幅（ATR 倍数）/ 2
    d["f_momentum"] = clip2(d["mom"] / 2)

    # ---- 3 量仓：增仓时跟随价格方向，减仓时打五折
    pc = clip2(d["px_chg_atr"])
    d["f_oi_price"] = np.where(d["oi_chg"] > 0, pc, 0.5 * pc)
    d.loc[d["oi_chg"].isna(), "f_oi_price"] = np.nan

    # ---- 4 关键位：突破 20 日通道 ±2，通道内按位置 ±1
    mid = (d["ch_hi"] + d["ch_lo"]) / 2
    half = (d["ch_hi"] - d["ch_lo"]) / 2
    inside = clip2((d["close"] - mid) / half).clip(-1, 1)
    d["f_breakout"] = np.select([d["close"] > d["ch_hi"], d["close"] < d["ch_lo"]], [2.0, -2.0], inside)
    d.loc[d["ch_hi"].isna(), "f_breakout"] = np.nan

    # ---- 5 月差：主力 − 次主力，20 日变化 / (0.5 ATR)，近月走强偏多
    close_px = panel.pivot_table(index="date", columns="contract", values="close")
    dates = list(close_px.index)
    pos = {dt: i for i, dt in enumerate(dates)}
    sp_now, sp_chg = [], []
    n = p["spread_n"]
    for _, r in d.iterrows():
        a, b, dt = r["dominant"], r["next"], r["date"]
        if not b or b not in close_px or a not in close_px:
            sp_now.append(np.nan); sp_chg.append(np.nan); continue
        i = pos[dt]
        now = close_px.at[dt, a] - close_px.at[dt, b]
        sp_now.append(now)
        if i >= n:
            old_dt = dates[i - n]
            old = close_px.at[old_dt, a] - close_px.at[old_dt, b]
            sp_chg.append(now - old)
        else:
            sp_chg.append(np.nan)
    d["spread"] = sp_now
    d["spread_chg"] = sp_chg
    d["f_spread"] = clip2(d["spread_chg"] / (0.5 * d["atr"]))

    # ---- 6 外盘：ICE 自身趋势计数 − 2，只用 t 日之前的已收盘值
    f = foreign.sort_values("date").copy()
    fc = f["close"]
    f["ma20"], f["ma60"] = fc.rolling(20).mean(), fc.rolling(60).mean()
    f["r20"] = fc - fc.shift(20)
    f["f_foreign"] = ((fc > f["ma20"]).astype(int) + (fc > f["ma60"]).astype(int)
                      + (f["ma20"] > f["ma60"]).astype(int) + (f["r20"] > 0).astype(int)) - 2
    f.loc[f["ma60"].isna(), "f_foreign"] = np.nan
    f = f.rename(columns={"close": "ice", "date": "ice_date"})[["ice_date", "ice", "f_foreign"]]
    d = pd.merge_asof(d.sort_values("date"), f, left_on="date", right_on="ice_date",
                      allow_exact_matches=False)
    fx2 = fx.rename(columns={"date": "fx_date"}).sort_values("fx_date")
    d = pd.merge_asof(d, fx2, left_on="date", right_on="fx_date", allow_exact_matches=False)

    # ---- 7 进口窗口：配额外利润在过去 250 日的分位，越倒挂越偏多
    ic = cfg["import_cost"]
    cost = ((d["ice"] * ic["lb_to_ton"] + ic["premium_freight_usd"]) * d["fx"]
            * (1 + ic["tariff_out_of_quota"]) * (1 + ic["vat"]) + ic["port_processing_cny"])
    d["import_cost"] = cost
    d["import_profit"] = d["close"] - cost
    w = p["pct_window"]
    pct = d["import_profit"].rolling(w, min_periods=120).apply(
        lambda s: (s[:-1] < s[-1]).mean() if len(s) > 1 else np.nan, raw=True)
    d["profit_pct"] = pct
    d["f_import_window"] = clip2((0.5 - pct) * 4)

    # ---- 合成
    fac = {k: v for k, v in cfg["score"]["factors"].items() if v.get("enabled", True)}
    cols = [f"f_{k}" for k in fac]
    wts = np.array([fac[k]["weight"] for k in fac])
    X = d[cols].to_numpy(dtype=float)
    mask = ~np.isnan(X)
    num = np.nansum(X * wts, axis=1)
    den = (mask * wts).sum(axis=1)
    d["score"] = np.where(den >= 0.7 * wts.sum(), num / np.where(den == 0, 1, den), np.nan)
    t = cfg["score"]["tiers"]
    d["tier"] = pd.cut(d["score"], [-9, t["strong_short"], t["short"], t["long"], t["strong_long"], 9],
                       labels=["强空", "偏空", "中性", "偏多", "强多"], right=False)
    return d
