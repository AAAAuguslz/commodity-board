# commodity-board

商品看板：数据层与计算层用代码完成（可复查），解读层由 Claude 撰写。第一个品种是白糖（`config/sugar.yaml`）。

- `board/fetch.py`：抓取郑商所单合约日线（新浪）、ICE 原糖（新浪外盘）、美元兑人民币（FRED），写入 `data/raw/`
- `board/factors.py`：8 个因子的打分规则（−2..+2）和加权总分（五档）
- `board/backtest.py`：历史回放，输出 `results/backtest_report.md`
- `.github/workflows/backtest.yml`：在 GitHub Actions 上手动触发回测

规则约定：t 日收盘出分，t+1 开盘执行；指标在各合约自身序列上计算后再按主力取值；ICE 和汇率只用 t 日之前已收盘的值。
