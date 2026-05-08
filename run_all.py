"""
run_all.py
═══════════════════════════════════════════════════════════════════════
一键跑完全流程：因子计算 → 分组回测 → 绩效分析 → 出图

使用方式：
    python run_all.py

如果想单独跑某一步：
    python step1_factor_compute.py
    python step2_backtest.py
    python step3_performance.py
═══════════════════════════════════════════════════════════════════════
"""

import time

print("=" * 60)
print("  Quant Alpha Pipeline  —  Full Run")
print("=" * 60)

t0 = time.time()

# ── Step 1：计算因子 ──
print("\n[Step 1/3] 因子计算...")
from step1_factor_compute import load_data_from_zip, compute_factors, save_factor_files

DATA_PATH  = "/Users/jiayizhai/Desktop/daily_data.zip"
FACTOR_DIR = "/Users/jiayizhai/Desktop/factors"
START      = "2020-01-01"
END        = "2025-06-06"

df = load_data_from_zip(DATA_PATH, START, END)
df_factors, factor_cols = compute_factors(df)
save_factor_files(df_factors, factor_cols, FACTOR_DIR)
del df, df_factors   # 释放内存

t1 = time.time()
print(f"  ⏱ Step 1 耗时 {(t1-t0)/60:.1f} 分钟")

# ── Step 2：回测 ──
print("\n[Step 2/3] 分组回测...")
from step2_backtest import build_return_dict, run_backtest
import os, pickle

BACKTEST_DIR = "/Users/jiayizhai/Desktop/backtest_results"
BT_START     = "2022-01-01"
BT_END       = "2024-06-01"
os.makedirs(BACKTEST_DIR, exist_ok=True)

ret_dict, trade_dates = build_return_dict(DATA_PATH, BT_START, BT_END)
pnl, ic = run_backtest(FACTOR_DIR, ret_dict, trade_dates)

with open(os.path.join(BACKTEST_DIR, 'pnl.pkl'), 'wb') as f:
    pickle.dump(pnl, f)
with open(os.path.join(BACKTEST_DIR, 'ic.pkl'), 'wb') as f:
    pickle.dump(ic, f)

import numpy as np
print("\n── IC 快速汇总 ──")
for fname, ic_list in ic.items():
    ic_arr = [x for x in ic_list if not np.isnan(x)]
    if ic_arr:
        print(f"  {fname:12s} | IC={np.mean(ic_arr):.4f}  "
              f"ICIR={np.mean(ic_arr)/np.std(ic_arr):.3f}  "
              f"IC>0={np.mean(np.array(ic_arr)>0):.1%}")

t2 = time.time()
print(f"  ⏱ Step 2 耗时 {(t2-t1)/60:.1f} 分钟")

# ── Step 3：绩效分析 & 出图 ──
print("\n[Step 3/3] 绩效分析...")
from step3_performance import main as run_performance
run_performance()

t3 = time.time()
print(f"  ⏱ Step 3 耗时 {(t3-t2)/60:.1f} 分钟")

print(f"\n🎉 全部完成！总耗时 {(t3-t0)/60:.1f} 分钟")
print(f"   因子文件：{FACTOR_DIR}/")
print(f"   回测结果：{BACKTEST_DIR}/")
print(f"   图表文件：{BACKTEST_DIR}/plots/")
