"""回测：把打分规则在历史上回放，看各档分数之后的收益分布。

输出 results/backtest_report.md、results/scores.csv、results/summary.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from board.factors import per_contract_indicators, pick_dominant, score_frame

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results"
TIERS = ["强空", "偏空", "中性", "偏多", "强多"]


def load(cfg):
    if cfg.get("data_source") == "main":  # 主力连续（如 Wind 导出，已比例复权）
        panel = pd.read_csv(RAW / f"{cfg['code']}_main.csv", parse_dates=["date"])
    else:
        panel = pd.read_csv(RAW / f"{cfg['code']}_contracts.csv", parse_dates=["date"])
    foreign = pd.read_csv(RAW / "foreign.csv", parse_dates=["date"]) if (RAW / "foreign.csv").exists() else None
    fx = pd.read_csv(RAW / "fx.csv", parse_dates=["date"]) if (RAW / "fx.csv").exists() else None
    return panel, foreign, fx


def build(cfg):
    panel, foreign, fx = load(cfg)
    p = cfg["score"]["params"]
    panel = pd.concat([per_contract_indicators(g, p) for _, g in panel.groupby("contract")])
    order = sorted(panel["contract"].unique(), key=lambda s: int(s[len(cfg["code"]):]) if s[len(cfg["code"]):].isdigit() else 0)
    dom = pick_dominant(panel, order)
    rows = dom.merge(panel, left_on=["date", "dominant"], right_on=["date", "contract"], how="left")
    return score_frame(rows, panel, foreign, fx, cfg), panel


def spearman(a, b):
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan
    return a[m].rank().corr(b[m].rank())


def tier_table(d, hz):
    col = f"fwd{hz}"
    out = []
    for t in TIERS:
        s = d.loc[d["tier"] == t, col].dropna()
        # 不重叠样本：每 hz 天取一次，用于 t 值
        sub = d.loc[d["tier"] == t].iloc[::hz][col].dropna()
        tstat = sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 2 else np.nan
        out.append({"档位": t, "天数": len(s), "平均%": s.mean() * 100, "中位%": s.median() * 100,
                    "上涨占比%": (s > 0).mean() * 100 if len(s) else np.nan, "t值(不重叠)": tstat})
    return pd.DataFrame(out)


def equity(d, rule):
    """t 日收盘定仓位，t+1 开盘执行。仓位不变时按收盘到收盘计，换仓当天按开盘到收盘计。
    换月日：旧合约平在 t+1 开盘，新合约开在 t+1 开盘（与持仓切换同样处理）。不含手续费和滑点。"""
    pos = rule(d["score"]).fillna(0).to_numpy()
    r_cc = (d["next_close"] / d["close"] - 1).to_numpy()
    r_oc = (d["next_close"] / d["next_open"] - 1).to_numpy()
    prev = np.r_[0, pos[:-1]]
    same_contract = np.r_[False, (d["dominant"].to_numpy()[1:] == d["dominant"].to_numpy()[:-1])]
    # 持仓 p_t 作用于 t→t+1；若 t 日相对 t−1 换仓或换月，从 t+1 开盘起算
    ret = np.where((pos == prev) & same_contract, pos * r_cc, pos * r_oc)
    ret = np.nan_to_num(ret)
    eq = (1 + pd.Series(ret, index=d["date"])).cumprod()
    yrs = len(ret) / 244
    ann = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = np.std(ret) * np.sqrt(244)
    dd = (eq / eq.cummax() - 1).min()
    return {"累计%": (eq.iloc[-1] - 1) * 100, "年化%": ann * 100,
            "夏普": ann / vol if vol else np.nan, "最大回撤%": dd * 100,
            "持仓天数占比%": (pos != 0).mean() * 100}, eq


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".2f")


def main(cfg_path="config/sugar.yaml", source=None, start=None):
    cfg = yaml.safe_load((ROOT / cfg_path).read_text(encoding="utf-8"))
    if source:
        cfg["data_source"] = source
    if start:
        cfg["backtest"]["start"] = start
    if cfg.get("data_source") == "main":  # 单一连续序列没有次主力，月差因子不可用
        cfg["score"]["factors"]["spread"]["enabled"] = False
    for fname in ("foreign", "fx"):
        if not (RAW / f"{fname}.csv").exists():  # 外盘或汇率缺失：外盘、进口窗口两个因子停用
            cfg["score"]["factors"]["foreign"]["enabled"] = False
            cfg["score"]["factors"]["import_window"]["enabled"] = False
    print("参与打分的因子：", [k for k, v in cfg["score"]["factors"].items() if v.get("enabled", True)])
    OUT.mkdir(exist_ok=True)
    tag = cfg.get("data_source", "contracts")
    d, panel = build(cfg)
    d = d[d["date"] >= pd.Timestamp(cfg["backtest"]["start"])].reset_index(drop=True)
    d.to_csv(OUT / f"scores_{tag}.csv", index=False)
    scored = d[d["score"].notna()]

    lines = [f"# {cfg['name']} 多空打分回测 v0", "",
             f"区间 {scored['date'].min():%Y-%m-%d} → {scored['date'].max():%Y-%m-%d}，"
             f"有效打分 {len(scored)} 天。执行假设：t 日收盘出分，t+1 开盘进场；收益在同一合约内计算，不跨换月；不含手续费。", ""]

    # 档位分布
    summary = {"days": len(scored), "tiers": {}, "ic": {}, "strategies": {}}
    for hz in cfg["backtest"]["horizons"]:
        tt = tier_table(scored, hz)
        summary["tiers"][hz] = tt.to_dict("records")
        lines += [f"## 各档之后 {hz} 日收益", "", fmt(tt), ""]

    # IC
    fac = [k for k, v in cfg["score"]["factors"].items() if v.get("enabled", True)]
    ic_rows = []
    for k in ["score"] + [f"f_{k}" for k in fac]:
        label = "总分" if k == "score" else cfg["score"]["factors"][k[2:]]["label"]
        ic_rows.append({"因子": label, "IC(5日)": spearman(scored[k], scored["fwd5"]),
                        "IC(20日)": spearman(scored[k], scored["fwd20"]),
                        "覆盖率%": scored[k].notna().mean() * 100,
                        "均值": scored[k].mean()})
    ic = pd.DataFrame(ic_rows)
    summary["ic"] = ic.to_dict("records")
    lines += ["## 因子信息系数（Spearman，越大说明分数越能排序未来收益）", "", fmt(ic), ""]

    # 策略
    rules = {
        "一直做多主力": lambda s: pd.Series(1.0, index=s.index),
        "闸门做多（≥偏多才持多）": lambda s: (s >= cfg["score"]["tiers"]["long"]).astype(float),
        "双向（≥偏多做多 / ≤偏空做空）": lambda s: np.sign(s.where(s.abs() >= cfg["score"]["tiers"]["long"], 0)),
        "只在强档（≥强多做多 / ≤强空做空）": lambda s: np.sign(s.where(s.abs() >= cfg["score"]["tiers"]["strong_long"], 0)),
    }
    strat, curves = [], {}
    for name, rule in rules.items():
        m, eq = equity(scored.reset_index(drop=True), rule)
        strat.append({"策略": name, **m}); curves[name] = eq
    st = pd.DataFrame(strat)
    summary["strategies"] = st.to_dict("records")
    pd.DataFrame(curves).to_csv(OUT / f"equity_{tag}.csv")
    lines += ["## 简单策略对比（日频，t+1 开盘执行）", "", fmt(st), ""]

    # 档位时间占比
    share = scored["tier"].value_counts(normalize=True).reindex(TIERS) * 100
    lines += ["## 各档出现频率", "", fmt(share.rename("占比%").reset_index().rename(columns={"index": "档位", "tier": "档位"})), ""]

    # 与 v11 的对账：最近一天的主力指标
    last = scored.iloc[-1]
    lines += ["## 数据对账（最近一个交易日）", "",
              f"日期 {last['date']:%Y-%m-%d}，主力 {last['dominant']}，次主力 {last['next']}；"
              f"收 {last['close']:.0f}，MA5 {last['ma5']:.0f} / MA10 {last['ma10']:.0f} / MA20 {last['ma20']:.0f} / MA60 {last['ma60']:.0f}，"
              f"ATR {last['atr']:.1f}；"
              f"总分 {last['score']:.2f}（{last['tier']}）", ""]
    chk = panel[(panel["date"] == "2026-10-21") & panel["contract"].isin(["SR2701", "SR2705", "SR_MAIN"])]
    if len(chk):
        lines += ["10/21 对照 v11 看板（v11：SR2701 MA5 5,388 / MA10 5,440 / MA20 5,397 / MA60 5,360 / ATR 58.8；"
                  "SR2705 MA5 5,470 / MA10 5,520 / MA20 5,475 / MA60 5,433 / ATR 59.7）", "",
                  fmt(chk[["contract", "close", "oi", "ma5", "ma10", "ma20", "ma60", "atr"]]), ""]
    if "roll" in d:
        rolls = d.loc[d["roll"].astype(bool), ["date", "dominant"]]
    else:
        rolls = d.loc[d["dominant"] != d["dominant"].shift(), ["date", "dominant"]]
    lines += ["## 主力换月记录", "", fmt(rolls.assign(date=rolls["date"].dt.strftime("%Y-%m-%d"))), ""]

    (OUT / f"backtest_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / f"summary_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
