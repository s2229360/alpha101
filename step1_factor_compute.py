"""
step1_factor_compute.py
═══════════════════════════════════════════════════════════════════════
功能：从 daily_data.zip 读数据 → 计算 Alpha 因子 → 按日期存 CSV
包含：
  - 5 个 Alpha101 因子（#1, #3, #6, #12, #20）
  - 老师原有的 avg_price 因子（上下影线）
  - 每个因子的经济逻辑注释

数据格式：zip 内每个 csv.gz 含列 ticker/volume/open/close/high/low
═══════════════════════════════════════════════════════════════════════
"""

import os, zipfile, io, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


# ───────────────────────────────────────────────
#  配置区（只需改这里）
# ───────────────────────────────────────────────
DATA_PATH      = "/Users/jiayizhai/Desktop/daily_data.zip"   # 原始数据
OUTPUT_DIR     = "/Users/jiayizhai/Desktop/factors"          # 因子输出根目录
START_DATE     = "2020-01-01"
END_DATE       = "2025-06-06"


# ───────────────────────────────────────────────
#  Step 0：从 zip 读取全量数据到内存（宽表）
# ───────────────────────────────────────────────

def load_data_from_zip(zip_path: str, start: str, end: str) -> pd.DataFrame:
    """
    从 zip 直接读，不需要解压到磁盘。
    返回 columns: ticker, date, open, close, high, low, volume, transactions
    按 (ticker, date) 排序。
    """
    print(f"📂 读取 {zip_path} ...")
    frames = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = sorted([f for f in zf.namelist() if f.endswith('.csv.gz')])
        total = len(files)
        for i, fname in enumerate(files):
            # 从文件名提取日期，格式 daily_data/YYYY-MM-DD.csv.gz
            date_str = os.path.basename(fname).replace('.csv.gz', '')
            if date_str < start or date_str > end:
                continue
            with zf.open(fname) as f:
                df = pd.read_csv(f, compression='gzip')
                df['date'] = date_str
                frames.append(df[['ticker','date','open','close','high','low','volume','transactions']])
            if (i + 1) % 200 == 0:
                print(f"  已读 {i+1}/{total} 文件...")

    result = pd.concat(frames, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values(['ticker', 'date']).reset_index(drop=True)
    result.dropna(subset=['open','close','high','low','volume'], inplace=True)

    n_tickers = result['ticker'].nunique()
    n_dates   = result['date'].nunique()
    print(f"✅ 数据加载完成：{n_tickers} 只股票 × {n_dates} 个交易日 = {len(result):,} 行")
    return result


# ───────────────────────────────────────────────
#  时序工具函数（每个 ticker groupby 后用）
# ───────────────────────────────────────────────

def ts_delta(s, d):   return s - s.shift(d)
def ts_delay(s, d):   return s.shift(d)
def ts_corr(x, y, d): return x.rolling(d, min_periods=d).corr(y)
def ts_argmax(s, d):  return s.rolling(d, min_periods=d).apply(np.argmax, raw=True)

def signed_power(s, e):
    """SignedPower(x,e) = sign(x)*|x|^e，保留方向，放大幅度"""
    return np.sign(s) * (s.abs() ** e)


# ───────────────────────────────────────────────
#  因子计算（全部在 ticker 时序维度操作）
# ───────────────────────────────────────────────

def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：全量宽表（ticker, date, OHLCV）
    输出：增加各因子列的宽表，NaN 表示数据不足（窗口期未满）

    因子列表：
      avg_price  — 老师原版因子（典型价格）
      upshadow   — 上影线
      downshadow — 下影线
      alpha001   — 近期收益方向强度（动量变体）
      alpha003   — 开盘排名 vs 成交量排名负相关（反转）
      alpha006   — 开盘价 vs 成交量负相关（量价背离）
      alpha012   — 放量下跌看多/缩量上涨看空（短期反转）
      alpha020   — 跳空高开反转（gap fade）
    """
    print("⚙️  计算因子...")
    result_parts = []

    for ticker, grp in df.groupby('ticker', sort=False):
        g = grp.sort_values('date').copy()

        # ── 老师原版因子 ──────────────────────────────────────
        # 典型价格（Typical Price），比单纯收盘价更能代表当日成交重心
        g['avg_price']   = (g['open'] + g['close'] + g['low']) / 3

        # 上影线 = 最高价 - max(开盘,收盘)，衡量多头被压制的程度
        g['upshadow']    = g['high'] - g[['close','open']].max(axis=1)

        # 下影线 = min(开盘,收盘) - 最低价，衡量空头被吸收的程度
        g['downshadow']  = g[['close','open']].min(axis=1) - g['low']

        # ── Alpha#1 ───────────────────────────────────────────
        # rank(Ts_ArgMax(SignedPower(ret, 2), 5))
        # 逻辑：日收益率取符号幂次后，找过去5天里"最强那天"的位置
        #       位置越靠近今天 → 动量越新鲜 → 截面 rank 后做多
        ret              = g['close'].pct_change()
        sp               = signed_power(ret, 2)
        # min_periods=3 而非默认的 window=5，让早期数据也能算出值
        g['alpha001_raw']= sp.rolling(5, min_periods=3).apply(np.argmax, raw=True)

        # ── Alpha#3 ───────────────────────────────────────────
        # -correlation(rank(open), rank(volume), 10)
        # 逻辑：开盘价排名和成交量排名正相关 → 高价股放量，追涨行为
        #       这类股票短期内容易反转 → 取负做空
        r_open           = g['open'].rank(pct=True)   # 时序内近似截面rank
        r_vol            = g['volume'].rank(pct=True)
        g['alpha003']    = -ts_corr(r_open, r_vol, 10)

        # ── Alpha#6 ───────────────────────────────────────────
        # -correlation(open, volume, 10)
        # 逻辑：同 alpha003 但用原始值（不 rank），对极端行情更敏感
        #       开盘价涨时放量 → 主力拉高出货信号 → 取负
        g['alpha006']    = -ts_corr(g['open'], g['volume'], 10)

        # ── Alpha#12 ─────────────────────────────────────────
        # sign(delta(volume,1)) * (-delta(close,1))
        # 逻辑：放量下跌(vol↑, close↓) → 因子为正(看多，恐慌性抛盘)
        #       缩量上涨(vol↓, close↑) → 因子为负(看空，虚涨无量支撑)
        #       最直观的短期量价背离反转因子
        d_vol            = ts_delta(g['volume'], 1)
        d_close          = ts_delta(g['close'],  1)
        g['alpha012']    = np.sign(d_vol) * (-d_close)

        # ── Alpha#20 ─────────────────────────────────────────
        # -(rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1)))
        # 逻辑：今日开盘高于昨日high/close/low的程度
        #       三者同为正（完全跳空高开）→ 乘积大 → 取负做空
        #       经验：高开缺口在日内往往会被回补(gap fade)
        g['_dh']         = g['open'] - ts_delay(g['high'],  1)
        g['_dc']         = g['open'] - ts_delay(g['close'], 1)
        g['_dl']         = g['open'] - ts_delay(g['low'],   1)
        # alpha020 的截面 rank 需要跨 ticker，先暂存原始值，后面统一处理
        g['alpha020_dh'] = g['_dh']
        g['alpha020_dc'] = g['_dc']
        g['alpha020_dl'] = g['_dl']

        result_parts.append(g.drop(columns=['_dh','_dc','_dl']))

    out = pd.concat(result_parts).sort_values(['date','ticker']).reset_index(drop=True)

    # ── 截面 rank（跨 ticker，按日期）──────────────────────────
    print("  截面标准化...")

    # alpha001：Ts_ArgMax 结果做截面 rank
    out['alpha001'] = out.groupby('date')['alpha001_raw'].rank(pct=True)
    out.drop(columns=['alpha001_raw'], inplace=True)

    # alpha020：三个差值分别截面 rank 后相乘取负
    for col in ['alpha020_dh','alpha020_dc','alpha020_dl']:
        out[col + '_r'] = out.groupby('date')[col].rank(pct=True)
    out['alpha020'] = -(out['alpha020_dh_r'] * out['alpha020_dc_r'] * out['alpha020_dl_r'])
    out.drop(columns=['alpha020_dh','alpha020_dc','alpha020_dl',
                       'alpha020_dh_r','alpha020_dc_r','alpha020_dl_r'], inplace=True)

    # 上下影线标准化（老师代码逻辑，rolling 5日均值归一化）
    # 注意：很多股票下影线经常为0（开盘即最低），用 clip 而非直接除防止 inf
    for col in ['upshadow', 'downshadow']:
        norm_col = 'up' if col == 'upshadow' else 'down'
        # 加一个小基数（价格的0.01%）避免除以零
        base = out['close'] * 0.0001
        roll_mean = out.groupby('ticker')[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        denom = roll_mean.where(roll_mean > base, base)   # 分母最小值 = base
        out[norm_col] = (out[col] / denom).clip(0, 10)    # clip 防极端值

    factor_cols = ['avg_price','up','down','alpha001','alpha003','alpha006','alpha012','alpha020']
    print(f"✅ 因子计算完成，共 {len(factor_cols)} 个因子")
    return out, factor_cols


# ───────────────────────────────────────────────
#  按日期保存因子文件（兼容老师的 backtest 框架）
# ───────────────────────────────────────────────

def save_factor_files(df: pd.DataFrame, factor_cols: list, output_dir: str):
    """
    每个因子一个子文件夹，每天一个 CSV。
    文件格式：index=ticker，一列=因子值
    完全兼容老师 backtest.ipynb 里的读取逻辑。
    """
    for col in factor_cols:
        os.makedirs(os.path.join(output_dir, col), exist_ok=True)

    dates  = df['date'].unique()
    total  = len(dates)
    print(f"💾 写入因子文件（{total} 个交易日 × {len(factor_cols)} 个因子）...")

    for i, date in enumerate(sorted(dates)):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        day_df   = df[df['date'] == date].set_index('ticker')

        for col in factor_cols:
            if col not in day_df.columns:
                continue
            out = day_df[[col]].dropna()
            if len(out) == 0:
                continue
            path = os.path.join(output_dir, col, f'{date_str}.csv')
            out.to_csv(path)

        if (i + 1) % 200 == 0:
            print(f"  已写 {i+1}/{total} 天...")

    print(f"✅ 因子文件写入完成 → {output_dir}/")


# ───────────────────────────────────────────────
#  主流程
# ───────────────────────────────────────────────

if __name__ == '__main__':
    # 1. 读数据
    df = load_data_from_zip(DATA_PATH, START_DATE, END_DATE)

    # 2. 算因子
    df_factors, factor_cols = compute_factors(df)

    # 3. 存文件
    save_factor_files(df_factors, factor_cols, OUTPUT_DIR)

    print("\n🎉 Step 1 完成！")
    print(f"   因子文件位于：{OUTPUT_DIR}/")
    print(f"   共 {len(factor_cols)} 个因子：{factor_cols}")
    print(f"   下一步运行：python step2_backtest.py")
