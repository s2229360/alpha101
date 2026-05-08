"""
step2_backtest.py
═══════════════════════════════════════════════════════════════════════
功能：读取因子文件 + 收益数据 → 分10组回测 → 计算 hedge 组合 PnL
兼容老师原版 backtest.ipynb 逻辑，同时修复了几个细节：
  1. 收益率 clip(-20%, +20%) 防极端值污染
  2. 用 zip 直接生成 return 数据，不需要预存 return_data 文件夹
  3. 自动识别交易日，不依赖外部 date.pkl
═══════════════════════════════════════════════════════════════════════
"""

import os, zipfile, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


# ───────────────────────────────────────────────
#  配置区
# ───────────────────────────────────────────────
DATA_PATH    = "/Users/jiayizhai/Desktop/daily_data.zip"
FACTOR_DIR   = "/Users/jiayizhai/Desktop/factors"
OUTPUT_DIR   = "/Users/jiayizhai/Desktop/backtest_results"
START_DATE   = "2022-01-01"
END_DATE     = "2024-06-01"
N_GROUPS     = 10           # 分组数
CLIP_RET     = 0.20         # 收益率截断（涨跌停保护）


# ───────────────────────────────────────────────
#  Step A：从 zip 构建 return 数据（ticker → 次日收益率）
# ───────────────────────────────────────────────

def build_return_dict(zip_path: str, start: str, end: str) -> dict:
    """
    返回 dict：{date_str: pd.Series(index=ticker, values=next_day_return)}
    每个 date_str 对应的是"当天持仓、次日收益"
    即：factor(T) 预测 return(T+1)
    """
    print("📊 构建收益率数据...")
    close_data = {}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = sorted([f for f in zf.namelist() if f.endswith('.csv.gz')])
        for fname in files:
            date_str = os.path.basename(fname).replace('.csv.gz', '')
            with zf.open(fname) as f:
                df = pd.read_csv(f, compression='gzip', usecols=['ticker','close'])
                close_data[date_str] = df.set_index('ticker')['close']

    all_dates = sorted(close_data.keys())
    # 只保留 [start, end+buffer] 范围（需要 end 后一天的收盘价算收益）
    relevant  = [d for d in all_dates if d >= start]

    ret_dict = {}
    for i, date in enumerate(relevant[:-1]):
        next_date = relevant[i + 1]
        if date > end:
            break
        s_t   = close_data[date]
        s_t1  = close_data[next_date]
        idx   = s_t.index.intersection(s_t1.index)
        ret   = ((s_t1[idx] - s_t[idx]) / s_t[idx]).clip(-CLIP_RET, CLIP_RET)
        ret_dict[date] = ret

    trade_dates = sorted(ret_dict.keys())
    print(f"✅ 收益率数据就绪：{len(trade_dates)} 个交易日")
    return ret_dict, trade_dates


# ───────────────────────────────────────────────
#  Step B：因子分组回测主循环
# ───────────────────────────────────────────────

def run_backtest(factor_dir: str, ret_dict: dict, trade_dates: list) -> tuple:
    """
    对 factor_dir 下的每个因子子文件夹跑分组回测。
    返回：
      pnl  dict: {factor_name: DataFrame(index=date, columns=1..10+'hedge')}
      ic   dict: {factor_name: list of daily IC values}
    """
    factor_names = sorted(os.listdir(factor_dir))
    # 过滤掉非文件夹
    factor_names = [f for f in factor_names if os.path.isdir(os.path.join(factor_dir, f))]
    print(f"📈 回测因子：{factor_names}")

    pnl = {f: pd.DataFrame() for f in factor_names}
    ic  = {f: []              for f in factor_names}

    date_set = set(trade_dates)

    for i, date in enumerate(trade_dates):
        ret_s = ret_dict[date]  # 次日收益 Series

        for factor_name in factor_names:
            factor_file = os.path.join(factor_dir, factor_name, f'{date}.csv')
            if not os.path.exists(factor_file):
                continue

            factor_s = pd.read_csv(factor_file, index_col=0, header=0).iloc[:, 0]
            factor_s.index.name = 'ticker'

            # 对齐
            common = factor_s.index.intersection(ret_s.index)
            if len(common) < N_GROUPS * 5:   # 样本太少跳过
                continue

            merged = pd.DataFrame({'factor': factor_s[common], 'ret': ret_s[common]})
            merged.dropna(inplace=True)
            if len(merged) < N_GROUPS * 5:
                continue

            # 分组（10组）
            merged['group'] = pd.qcut(merged['factor'], N_GROUPS,
                                       labels=False, duplicates='drop') + 1

            # 各组平均收益
            grp_ret = merged.groupby('group')['ret'].mean()
            row     = grp_ret.to_frame().T
            row.index = [date]

            pnl[factor_name] = pd.concat([pnl[factor_name], row])

            # IC（Spearman 秩相关）
            ic_val = merged['factor'].corr(merged['ret'], method='spearman')
            ic[factor_name].append(ic_val)

        if (i + 1) % 100 == 0:
            print(f"  回测进度 {i+1}/{len(trade_dates)}...")

    # 计算 hedge 列
    # 用 IC 均值的符号自动判断方向：
    #   IC > 0 → 因子值越大收益越高 → 做多G10、做空G1 → hedge = G10 - G1
    #   IC < 0 → 因子值越大收益越低 → 做多G1、做空G10 → hedge = G1 - G10
    for fname in factor_names:
        df = pnl[fname]
        if df.empty:
            continue
        if 1 not in df.columns or N_GROUPS not in df.columns:
            continue
        ic_arr  = [x for x in ic.get(fname, []) if not np.isnan(x)]
        ic_mean = np.mean(ic_arr) if ic_arr else 0
        if ic_mean >= 0:
            df['hedge'] = df[N_GROUPS] - df[1]
            print(f"  {fname}: IC={ic_mean:.4f} ≥ 0 → hedge = G{N_GROUPS} - G1")
        else:
            df['hedge'] = df[1] - df[N_GROUPS]
            print(f"  {fname}: IC={ic_mean:.4f} < 0 → hedge = G1 - G{N_GROUPS}")
        pnl[fname] = df

    print("✅ 回测完成")
    return pnl, ic


# ───────────────────────────────────────────────
#  主流程
# ───────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # A. 构建收益率
    ret_dict, trade_dates = build_return_dict(DATA_PATH, START_DATE, END_DATE)

    # B. 跑回测
    pnl, ic = run_backtest(FACTOR_DIR, ret_dict, trade_dates)

    # C. 保存 pnl 和 ic（供 step3 使用）
    import pickle
    with open(os.path.join(OUTPUT_DIR, 'pnl.pkl'), 'wb') as f:
        pickle.dump(pnl, f)
    with open(os.path.join(OUTPUT_DIR, 'ic.pkl'), 'wb') as f:
        pickle.dump(ic, f)

    # 打印 IC 摘要
    print("\n── IC 汇总 ──")
    for fname, ic_list in ic.items():
        if ic_list:
            ic_arr = [x for x in ic_list if not np.isnan(x)]
            print(f"  {fname:12s} | IC均值={np.mean(ic_arr):.4f}  "
                  f"ICIR={np.mean(ic_arr)/np.std(ic_arr):.3f}  "
                  f"胜率={np.mean(np.array(ic_arr)>0):.1%}")

    print(f"\n🎉 Step 2 完成！结果保存至 {OUTPUT_DIR}/")
    print("   下一步运行：python step3_performance.py")
