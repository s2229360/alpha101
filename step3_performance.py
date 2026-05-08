"""
step3_performance.py
═══════════════════════════════════════════════════════════════════════
功能：读取 backtest 结果 → 计算所有绩效指标 → 出图 → 汇总表
指标：年化收益、年化波动、Sharpe、最大回撤、Calmar、Sortino、
      胜率、盈亏比、IC均值、ICIR
═══════════════════════════════════════════════════════════════════════
"""

import os, pickle, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')
plt.rcParams['font.family']       = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False


# ───────────────────────────────────────────────
#  配置区
# ───────────────────────────────────────────────
BACKTEST_DIR = "/Users/jiayizhai/Desktop/backtest_results"
OUTPUT_DIR   = "/Users/jiayizhai/Desktop/backtest_results/plots"
FREQ         = 252     # 年化交易日数
RF           = 0.0     # 无风险利率（年化）


# ───────────────────────────────────────────────
#  核心指标计算
# ───────────────────────────────────────────────

def calc_performance(returns: pd.Series, ic_list: list = None,
                     freq: int = 252, rf: float = 0.0) -> dict:
    """
    输入日收益率序列，返回完整绩效指标字典。
    """
    s = returns.dropna()
    n = len(s)
    if n < 5:
        return {}

    # ── 收益类 ──
    total_ret  = (1 + s).prod() - 1
    ann_ret    = (1 + total_ret) ** (freq / n) - 1

    # ── 风险类 ──
    ann_vol    = s.std() * np.sqrt(freq)

    # ── Sharpe ──
    rf_daily   = (1 + rf) ** (1 / freq) - 1
    excess     = s - rf_daily
    sharpe     = (excess.mean() / excess.std() * np.sqrt(freq)
                  if excess.std() > 1e-10 else np.nan)

    # ── 最大回撤 ──
    cum        = (1 + s).cumprod()
    roll_max   = cum.cummax()
    drawdown   = (cum - roll_max) / roll_max
    max_dd     = drawdown.min()

    # ── Calmar ──
    calmar     = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    # ── Sortino（只惩罚下行） ──
    down       = s[s < rf_daily]
    down_vol   = down.std() * np.sqrt(freq) if len(down) > 1 else np.nan
    sortino    = (ann_ret - rf) / down_vol if down_vol and down_vol > 1e-10 else np.nan

    # ── 胜率 & 盈亏比 ──
    wins       = s[s > 0]
    losses     = s[s < 0]
    win_rate   = len(wins) / n
    profit_fac = (abs(wins.sum() / losses.sum())
                  if len(losses) > 0 and losses.sum() != 0 else np.nan)

    # ── IC 指标 ──
    ic_arr  = np.array([x for x in (ic_list or []) if not np.isnan(x)])
    ic_mean = float(np.mean(ic_arr))              if len(ic_arr) > 0 else np.nan
    icir    = float(np.mean(ic_arr)/np.std(ic_arr)) if len(ic_arr) > 1 else np.nan
    ic_pos  = float(np.mean(ic_arr > 0))          if len(ic_arr) > 0 else np.nan

    return {
        'Ann. Return':    ann_ret,
        'Ann. Vol':       ann_vol,
        'Sharpe':         sharpe,
        'Max Drawdown':   max_dd,
        'Calmar':         calmar,
        'Sortino':        sortino,
        'Win Rate':       win_rate,
        'Profit Factor':  profit_fac,
        'IC Mean':        ic_mean,
        'ICIR':           icir,
        'IC>0 Rate':      ic_pos,
        'Total Return':   total_ret,
        'Num Days':       n,
    }


def _fmt(k, v):
    """格式化打印"""
    pct_keys = {'Ann. Return','Ann. Vol','Max Drawdown','Win Rate','IC>0 Rate','Total Return'}
    if k in pct_keys:
        return f"{v:.2%}"
    elif k == 'Num Days':
        return str(int(v))
    else:
        return f"{v:.4f}"


# ───────────────────────────────────────────────
#  可视化：单因子 5 合 1 Dashboard
# ───────────────────────────────────────────────

def plot_factor_dashboard(pnl_df: pd.DataFrame, ic_list: list,
                          factor_name: str, save_dir: str):
    """
    5 合 1 看板：
      [0,0] 各组累计净值（层叠折线）
      [0,1] Hedge 组合累计净值
      [1,0] Hedge 回撤曲线
      [1,1] 滚动 60 日 Sharpe
      [2,0] 日收益分布直方图
      [2,1] 逐日 IC & 累计 IC
    """
    hedge = pnl_df['hedge'].dropna()
    if len(hedge) < 5:
        return

    hedge.index = pd.to_datetime(hedge.index)
    hedge = hedge.sort_index()

    cum_hedge = (1 + hedge).cumprod()
    dd        = (cum_hedge - cum_hedge.cummax()) / cum_hedge.cummax()

    roll_sr   = (hedge.rolling(60).mean() / hedge.rolling(60).std() * np.sqrt(252))

    ic_s = pd.Series(ic_list, name='IC')
    ic_s.index = range(len(ic_s))
    cum_ic = ic_s.cumsum()

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    DARK  = '#0f1117'
    PANEL = '#1a1d27'
    BLUE  = '#4f9cf9'
    GREEN = '#2ecc71'
    RED   = '#e74c3c'
    GOLD  = '#f1c40f'
    GREY  = '#8892a4'

    def style_ax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=8)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2d3a')

    fig.suptitle(f'Factor: {factor_name}  —  Hedge Portfolio Analysis',
                 fontsize=15, fontweight='bold', color='white', y=0.98)

    # ── [0,0] 各组累计净值 ──
    ax0 = fig.add_subplot(gs[0, 0])
    style_ax(ax0)
    group_cols = [c for c in pnl_df.columns if isinstance(c, (int, float)) and c != 'hedge']
    cmap = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(group_cols)))
    for j, col in enumerate(sorted(group_cols)):
        s = pnl_df[col].dropna()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        cum = (1 + s).cumprod()
        lw  = 2.5 if col in [1, 10] else 0.8
        ax0.plot(cum.index, cum.values, color=cmap[j], lw=lw,
                 label=f'G{int(col)}', alpha=0.9 if col in [1,10] else 0.5)
    ax0.set_title('All Groups — Cumulative Return')
    ax0.set_ylabel('NAV')
    ax0.legend(fontsize=6, ncol=5, loc='upper left',
               facecolor=PANEL, edgecolor='none', labelcolor=GREY)
    ax0.tick_params(axis='x', rotation=30)

    # ── [0,1] Hedge 净值 ──
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1)
    ax1.plot(cum_hedge.index, cum_hedge.values, color=BLUE, lw=2)
    ax1.axhline(1, color=GREY, lw=0.8, linestyle='--', alpha=0.5)
    ax1.fill_between(cum_hedge.index, 1, cum_hedge.values,
                     where=cum_hedge.values >= 1, color=GREEN, alpha=0.15)
    ax1.fill_between(cum_hedge.index, 1, cum_hedge.values,
                     where=cum_hedge.values < 1,  color=RED,   alpha=0.15)
    ax1.set_title('Hedge Portfolio — Cumulative Return (G1 − G10)')
    ax1.set_ylabel('NAV')
    ax1.tick_params(axis='x', rotation=30)

    # ── [1,0] 回撤曲线 ──
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.4)
    ax2.plot(dd.index, dd.values, color=RED, lw=0.8)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax2.tick_params(axis='x', rotation=30)

    # ── [1,1] 滚动 60 日 Sharpe ──
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    ax3.plot(roll_sr.index, roll_sr.values, color=GOLD, lw=1.2)
    ax3.axhline(0, color=GREY, lw=0.8, linestyle='--', alpha=0.5)
    ax3.axhline(1, color=GREEN, lw=0.6, linestyle=':', alpha=0.6)
    ax3.set_title('Rolling 60-Day Sharpe Ratio')
    ax3.set_ylabel('Sharpe')
    ax3.tick_params(axis='x', rotation=30)

    # ── [2,0] 日收益分布 ──
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4)
    ax4.hist(hedge.values, bins=60, color=BLUE, edgecolor=DARK, alpha=0.85)
    ax4.axvline(hedge.mean(), color=GOLD, lw=1.5, linestyle='--',
                label=f'Mean {hedge.mean():.4%}')
    ax4.axvline(0, color=GREY, lw=1)
    ax4.set_title('Daily Return Distribution')
    ax4.set_xlabel('Daily Return')
    ax4.legend(fontsize=8, facecolor=PANEL, edgecolor='none', labelcolor=GREY)

    # ── [2,1] IC 走势 ──
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5)
    ax5b = ax5.twinx()
    ax5b.set_facecolor(PANEL)
    ax5.bar(range(len(ic_s)), ic_s.values,
            color=[GREEN if v > 0 else RED for v in ic_s.values], alpha=0.5, width=1)
    ax5b.plot(cum_ic.values, color=GOLD, lw=1.5, label='Cum IC')
    ax5.axhline(0, color=GREY, lw=0.8)
    ax5.set_title(f'Daily IC  (Mean={np.nanmean(ic_s):.4f})')
    ax5.set_ylabel('IC', color=GREY)
    ax5b.set_ylabel('Cumulative IC', color=GOLD)
    ax5b.tick_params(colors=GOLD)
    ax5b.spines['right'].set_edgecolor(GOLD)

    plt.savefig(os.path.join(save_dir, f'{factor_name}_dashboard.png'),
                dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  📊 图表已保存：{factor_name}_dashboard.png")


# ───────────────────────────────────────────────
#  主流程
# ───────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取 backtest 结果
    with open(os.path.join(BACKTEST_DIR, 'pnl.pkl'), 'rb') as f:
        pnl = pickle.load(f)
    with open(os.path.join(BACKTEST_DIR, 'ic.pkl'), 'rb') as f:
        ic  = pickle.load(f)

    summary_rows = []

    for factor_name, pnl_df in pnl.items():
        if pnl_df.empty or 'hedge' not in pnl_df.columns:
            continue

        print(f"\n{'═'*55}")
        print(f"  Factor: {factor_name}")
        print(f"{'═'*55}")

        hedge_ret = pnl_df['hedge'].dropna()
        metrics   = calc_performance(hedge_ret, ic_list=ic.get(factor_name, []),
                                     freq=FREQ, rf=RF)

        for k, v in metrics.items():
            print(f"  {k:<20}: {_fmt(k, v)}")

        # 出图
        plot_factor_dashboard(pnl_df, ic.get(factor_name, []),
                              factor_name=factor_name, save_dir=OUTPUT_DIR)

        row = {'Factor': factor_name}
        row.update({k: v for k, v in metrics.items()})
        summary_rows.append(row)

    # ── 汇总表 ──
    if summary_rows:
        summary = pd.DataFrame(summary_rows).set_index('Factor')

        # 格式化为字符串表
        display = summary.copy()
        pct_cols = ['Ann. Return','Ann. Vol','Max Drawdown','Win Rate','IC>0 Rate','Total Return']
        for col in display.columns:
            if col in pct_cols:
                display[col] = display[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
            elif col == 'Num Days':
                display[col] = display[col].apply(lambda x: str(int(x)) if pd.notna(x) else 'N/A')
            else:
                display[col] = display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')

        print(f"\n{'═'*55}")
        print("  ALL FACTORS SUMMARY")
        print(f"{'═'*55}")
        print(display.to_string())

        # 保存 CSV
        save_path = os.path.join(BACKTEST_DIR, 'summary.csv')
        summary.to_csv(save_path)
        print(f"\n✅ 汇总表已保存：{save_path}")

    print(f"\n🎉 Step 3 完成！图表位于 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
