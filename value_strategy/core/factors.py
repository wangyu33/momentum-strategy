"""因子构建。

核心纪律
--------
1. 财务数据是**年初至今累计值**（已验证：ROE 一季报中位数 2.02%，年报 8.51%），
   因此 TTM 必须用 `本期 + 上年Q4 − 上年同期`，不能简单把四个季度相加。
2. 每期财报只有在**生效日**（法定披露截止日之后）才对策略可见。
   调仓日 t 只能使用 `eff_date <= t` 的财报，用 merge_asof 保证这一点。
3. 估值用**不复权价**，收益用**后复权价**。混用会让 PB 失真。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# 中文列 -> 英文列
FIELD_MAP = {
    "日期": "report_date",
    "摊薄每股收益(元)": "eps",
    "扣除非经常性损益后的每股收益(元)": "eps_deduct",
    "每股净资产_调整后(元)": "bvps",
    "每股经营性现金流(元)": "ocfps",
    "每股未分配利润(元)": "retained_ps",
    "净资产收益率(%)": "roe_reported",
    "加权净资产收益率(%)": "roe_weighted",
    "总资产净利润率(%)": "roa_reported",
    "销售毛利率(%)": "gross_margin",
    "主营业务利润率(%)": "op_margin",
    "营业利润率(%)": "op_margin2",
    "销售净利率(%)": "net_margin",
    "总资产周转率(次)": "asset_turnover",
    "流动比率": "current_ratio",
    "速动比率": "quick_ratio",
    "资产负债率(%)": "debt_ratio",
    "长期负债比率(%)": "lt_debt_ratio",
    "长期债务与营运资金比率(%)": "lt_debt_to_wc",
    "股东权益比率(%)": "equity_ratio",
    "经营现金净流量与净利润的比率(%)": "ocf_to_ni",
    "净利润增长率(%)": "ni_growth",
    "主营业务收入增长率(%)": "rev_growth",
    "净资产增长率(%)": "equity_growth",
    "总资产增长率(%)": "asset_growth",
    "总资产(元)": "total_assets",
    "股息发放率(%)": "payout_ratio",
}

# 累计值字段：需要按「本期 + 上年Q4 - 上年同期」转 TTM
CUM_FIELDS = ["eps", "eps_deduct", "ocfps"]
# 时点值字段：直接取当期值，不转 TTM
POINT_FIELDS = ["bvps", "total_assets", "retained_ps"]
# 比率字段：TTM 由 eps_ttm 派生，不用加减法
RATIO_FIELDS = ["gross_margin", "op_margin", "op_margin2", "net_margin",
                "asset_turnover", "current_ratio", "quick_ratio",
                "debt_ratio", "lt_debt_ratio", "lt_debt_to_wc", "equity_ratio",
                "ocf_to_ni", "ni_growth", "rev_growth", "equity_growth",
                "asset_growth", "roe_reported", "roa_reported", "payout_ratio"]


def _quarter_of(month: int) -> int:
    return {3: 1, 6: 2, 9: 3, 12: 4}[month]


def effective_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """报告期 -> 生效日（法定披露截止日的次日）。"""
    m, d = cfg.REPORT_EFFECTIVE_LAG[int(report_date.month)]
    y = int(report_date.year) + cfg.REPORT_EFFECTIVE_YEAR_OFFSET[int(report_date.month)]
    return pd.Timestamp(year=y, month=m, day=d)


def prepare_financials(raw: pd.DataFrame) -> pd.DataFrame:
    """清洗财务长表：改名、转数值、加 effective_date、算 TTM 与同比基期。"""
    df = raw.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.rename(columns={k: v for k, v in FIELD_MAP.items() if k in df.columns})

    df["report_date"] = pd.to_datetime(df["report_date"])
    # 极个别脏记录（如 000768 的 2010-10-31）不是合法报告期，直接剔除
    df = df[df["report_date"].dt.month.isin((3, 6, 9, 12))].copy()
    df["year"] = df["report_date"].dt.year
    df["month"] = df["report_date"].dt.month
    df["quarter"] = df["month"].map(_quarter_of)
    df["eff_date"] = df["report_date"].apply(effective_date)

    num_cols = [c for c in FIELD_MAP.values()
                if c in df.columns and c != "report_date"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = list(dict.fromkeys(
        ["code", "report_date", "eff_date", "year", "quarter"] + num_cols))
    df = df[[c for c in keep if c in df.columns]]
    df = df.drop_duplicates(subset=["code", "report_date"], keep="last")
    df = df.sort_values(["code", "report_date"]).reset_index(drop=True)

    df = _add_ttm(df, CUM_FIELDS)
    df = _add_lags(df, RATIO_FIELDS + ["eps_ttm", "ocfps_ttm", "bvps", "total_assets"])
    df = _derive_ratios(df)
    # 派生字段（roa_ttm / margin / leverage）也要有同比基期，否则 F-Score 的
    # 「同比改善」判据会因一侧全为 NaN 而恒为 0 分
    df = _add_lags(df, ["roe_ttm", "roa_ttm", "margin", "leverage", "shares"])
    return df


def _add_ttm(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """累计值 -> TTM：TTM(y, q) = cum(y, q) + cum(y-1, Q4) - cum(y-1, q)。"""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    piv = df.pivot_table(index=["code", "year"], columns="quarter", values=cols, aggfunc="last")
    ttm = {}
    prev = piv.groupby(level="code").shift(1)          # 上一年同季
    if 4 not in prev.columns.get_level_values("quarter"):
        return df
    prev_q4 = prev.xs(4, level="quarter", axis=1)      # 上一年 Q4（= 上年全年累计）

    for q in (1, 2, 3, 4):
        if q not in piv.columns.get_level_values("quarter"):
            continue
        cur = piv.xs(q, level="quarter", axis=1)
        prev_q = prev.xs(q, level="quarter", axis=1)
        ttm[q] = cur + prev_q4 - prev_q

    records = []
    for q, v in ttm.items():
        v = v.copy()
        v.columns = [f"{c}_ttm" for c in v.columns]
        v = v.reset_index()
        v["quarter"] = q
        records.append(v)
    res = pd.concat(records, ignore_index=True)

    merged = df.merge(res, on=["code", "year", "quarter"], how="left")
    # 缺上年同期时退回当期累计值（仅影响最早若干期）
    for c in cols:
        t = f"{c}_ttm"
        if t in merged.columns:
            merged[t] = merged[t].fillna(merged[c])
    return merged


def _add_lags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """加 4 个季度前的值，供 F-Score 的同比判定使用。"""
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("code", sort=False)
    for c in cols:
        df[f"{c}_lag4"] = g[c].shift(4)
    return df


def _derive_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """派生 TTM 口径的回报率与总股本。"""
    # TTM ROE（期末净资产口径）：eps_ttm / bvps
    if "eps_ttm" in df.columns and "bvps" in df.columns:
        df["roe_ttm"] = np.where(df["bvps"] > 0, df["eps_ttm"] / df["bvps"] * 100, np.nan)

    # 总股本：净资产 / 每股净资产
    equity = df["total_assets"] * df.get("equity_ratio", np.nan) / 100.0
    df["shares"] = np.where((equity > 0) & (df["bvps"] > 0), equity / df["bvps"], np.nan)

    # TTM ROA：TTM 净利润 / 期末总资产
    if "eps_ttm" in df.columns:
        ni = df["eps_ttm"] * df["shares"]
        df["roa_ttm"] = np.where(df["total_assets"] > 0, ni / df["total_assets"] * 100, np.nan)

    # 毛利率缺失较多，用营业利润率兜底
    if "gross_margin" in df.columns and "op_margin" in df.columns:
        df["margin"] = df["gross_margin"].fillna(df["op_margin"])
    elif "gross_margin" in df.columns:
        df["margin"] = df["gross_margin"]
    else:
        df["margin"] = np.nan

    # 长期负债率缺失时，用资产负债率兜底（口径不同，仅用于「是否下降」的相对比较）
    if "lt_debt_ratio" in df.columns:
        df["leverage"] = df["lt_debt_ratio"].fillna(df["debt_ratio"])
    else:
        df["leverage"] = df.get("debt_ratio", np.nan)
    return df


# ---------------------------------------------------------------- F-Score

def _std_unexpected(df: pd.DataFrame, col: str, out: str,
                    min_hist: int = 6, win: int = 8,
                    cumulative: bool = True) -> pd.DataFrame:
    """标准化未预期值（SUE 的通用形式）。

        SUE = (Q_t − Q_{t−4}) / std( (Q_s − Q_{s−4}) , s ∈ [t-win, t-1] )

    `cumulative=True`（默认）：`col` 是**年初至今累计值**（EPS、扣非 EPS 等），
        先年内差分还原单季值（Q1 即累计值本身），再取同比增量。
    `cumulative=False`：`col` 已经是 TTM / 年化口径（如估算营收），
        直接取同比增量，不可再做年内差分——否则得到的是二阶变化量。

    分母 shift(1) 排除当期，避免自我参照；历史不足 `min_hist` 期则为 NaN。
    """
    if col not in df.columns:
        return df
    c = df.sort_values(["code", "year", "quarter"], kind="mergesort").copy()
    v = pd.to_numeric(c[col], errors="coerce")

    if cumulative:
        q = v.groupby([c["code"], c["year"]]).diff()
        q = q.where(c["quarter"] != 1, v)
    else:
        q = v
    c[f"{col}_q"] = q
    d = q.groupby(c["code"]).diff(4)
    sd = d.groupby(c["code"]).transform(
        lambda s: s.shift(1).rolling(win, min_periods=min_hist).std()
    )
    c[out] = d / sd.replace(0, np.nan)
    return c


def add_sue(df: pd.DataFrame, min_hist: int = 6) -> pd.DataFrame:
    """标准化未预期盈利 SUE —— PEAD（盈余公告后价格漂移）的代理变量。

    单季 EPS：年初至今累计值在**年内**做差分（Q1 的单季就是累计值本身）。
    SUE = (Q_t − Q_{t−4}) / std( (Q_s − Q_{s−4}) , s ∈ [t−8, t−1] )

    同时给出两个变体：
      - `sue_deduct`：扣非 EPS 口径，剔除资产处置/政府补助等一次性损益
      - `sue_mean4`：近 4 期 SUE 均值，抑制单季噪声（更稳、更钝）

    缺失报告期会让 diff(4) 错位，故要求序列连续——若不连续则结果为 NaN 而非错值。
    """
    df = _std_unexpected(df, "eps", "sue", min_hist=min_hist)
    df = _std_unexpected(df, "eps_deduct", "sue_deduct", min_hist=min_hist)
    if "sue" in df.columns:
        s = pd.to_numeric(df["sue"], errors="coerce")
        df["sue_mean4"] = s.groupby(df["code"]).transform(
            lambda x: x.rolling(4, min_periods=3).mean()
        )
    return df


def add_revenue_and_sur(df: pd.DataFrame, min_hist: int = 6) -> pd.DataFrame:
    """估算 TTM 营业收入，并给出标准化未预期营收 SUR（营收超预期）。

    新浪财务面板没有营业收入绝对值，但「总资产周转率(次)」×「总资产(元)」= 营收。
    注意周转率是**报告期累计值**（Q1 只有全年的约 1/4），须按报告期年化。
    这是估算值，误差来自周转率口径（部分公司用平均总资产而非期末），仅用于排序。
    """
    if "asset_turnover" not in df.columns or "total_assets" not in df.columns:
        return df
    c = df.copy()
    at = pd.to_numeric(c["asset_turnover"], errors="coerce")
    ta = pd.to_numeric(c["total_assets"], errors="coerce")
    qtr = pd.to_numeric(c["quarter"], errors="coerce")
    ann = np.where(qtr > 0, 4.0 / qtr, np.nan)
    c["revenue_est"] = at * ann * ta
    # revenue_est 是**年化（run-rate）口径**而非年初至今累计值，
    # 因此必须 cumulative=False：直接取同比增量，不能再做年内差分。
    c = _std_unexpected(c, "revenue_est", "sur", min_hist=min_hist,
                        cumulative=False)
    return c


def add_stability(df: pd.DataFrame, win: int = 12, min_periods: int = 8) -> pd.DataFrame:
    """盈利稳定性（QMJ 的「安全」腿）。

    - `roe_std12`：近 12 期 ROE(TTM) 的标准差，越小越稳定
    - `prof_ratio12`：近 12 期中 ROA(TTM)>0 的比例，越接近 1 越持续
    - `roe_min12`：近 12 期 ROE 的最低值，用来识别「从没亏过」的公司

    报告期缺失会让窗口内期数不足，用 min_periods 放宽；不足则为 NaN。
    """
    c = df.sort_values(["code", "report_date"], kind="mergesort").copy()
    g = c.groupby("code", sort=False)
    roe = pd.to_numeric(c.get("roe_ttm"), errors="coerce")
    roa = pd.to_numeric(c.get("roa_ttm"), errors="coerce")
    c["roe_std12"] = roe.groupby(c["code"]).transform(
        lambda s: s.rolling(win, min_periods=min_periods).std())
    c["roe_min12"] = roe.groupby(c["code"]).transform(
        lambda s: s.rolling(win, min_periods=min_periods).min())
    c["prof_ratio12"] = (roa > 0).groupby(c["code"]).transform(
        lambda s: s.rolling(win, min_periods=min_periods).mean())
    return c


def add_f_score(df: pd.DataFrame) -> pd.DataFrame:
    """Piotroski F-Score：9 条二元信号，作用于低估值股票池。

    关键细节：NaN 参与比较会得到 False，若不区分「判据失败」与「数据缺失」，
    缺数据的公司会被系统性判为低分。这里对每条判据同时给出
    `f{i}`（通过=1）与 `f_score_n`（有效判据数），缺失计 0 分但不计入分母。
    """
    c = lambda x: pd.to_numeric(df.get(x), errors="coerce")  # noqa: E731

    # (通过判定, 数据是否可用)
    def _cmp(cur, prev, op: str):
        a, b = c(cur), c(prev)
        ok = a.notna() & b.notna()
        res = (a > b) if op == "gt" else (a < b)
        return (res & ok).astype(float), ok.astype(float)

    def _pos(field):
        a = c(field)
        ok = a.notna()
        return (a > 0 & ok).astype(float), ok.astype(float)

    tests = [
        _pos("roa_ttm"),                 # f1 ROA > 0
        _pos("ocfps_ttm"),               # f2 经营现金流 > 0
        _cmp("roa_ttm", "roa_ttm_lag4", "gt"),          # f3 ROA 改善
        _cmp("ocfps_ttm", "eps_ttm", "gt"),             # f4 经营现金流 > 净利润
        _cmp("leverage", "leverage_lag4", "lt"),        # f5 杠杆下降
        _cmp("current_ratio", "current_ratio_lag4", "gt"),  # f6 流动比率改善
        _cmp("equity_growth", "roe_ttm", "le"),         # f7 未稀释（见下）
        _cmp("margin", "margin_lag4", "gt"),            # f8 毛利率改善
        _cmp("asset_turnover", "asset_turnover_lag4", "gt"),  # f9 资产周转率改善
    ]

    # f7 原判据是「当年未增发」。用净资产增速与 ROE 的关系近似：
    # 若未增发，净资产增速应不超过 ROE（利润留存带来的增长上限），给 1 个百分点容差。
    eq_g, roe = c("equity_growth"), c("roe_ttm")
    ok7 = eq_g.notna() & roe.notna()
    tests[6] = ((eq_g <= roe + 1.0) & ok7).astype(float), ok7.astype(float)

    df["f_score"] = sum(t[0] for t in tests)
    df["f_score_n"] = sum(t[1] for t in tests)
    for i, (p, _) in enumerate(tests, 1):
        df[f"f{i}"] = p
    return df


# ---------------------------------------------------------------- 调仓日因子截面

def build_snapshots(
    fin: pd.DataFrame,
    dates: list[pd.Timestamp],
    close_raw: pd.DataFrame,
    share: pd.DataFrame | None,
    dividends: pd.DataFrame | None = None,
    close_hfq: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """为每个调仓日构建因子截面。返回 MultiIndex(date, code) 的 DataFrame。

    只取 `eff_date <= 调仓日` 的最新一期财报（merge_asof 保证无未来函数）。
    `close_hfq` 仅用于计算过去一年波动率（红利低波类策略需要），不传则跳过。
    """
    f = fin.sort_values("eff_date", kind="mergesort").reset_index(drop=True)
    grid = pd.DataFrame(
        [(d, c) for d in dates for c in close_raw.columns],
        columns=["date", "code"],
    )
    grid["date"] = pd.to_datetime(grid["date"])
    grid = grid.sort_values("date", kind="mergesort").reset_index(drop=True)

    snap = pd.merge_asof(
        grid, f, left_on="date", right_on="eff_date", by="code", direction="backward"
    )
    snap = snap.drop(columns=["eff_date"], errors="ignore")

    # 价格（不复权）与股本
    px = close_raw.stack().rename("close").reset_index()
    px.columns = ["date", "code", "close"]
    px["date"] = pd.to_datetime(px["date"])
    snap = snap.merge(px, on=["date", "code"], how="left")

    if share is not None:
        sh = share.stack().rename("float_share").reset_index()
        sh.columns = ["date", "code", "float_share"]
        sh["date"] = pd.to_datetime(sh["date"])
        snap = snap.merge(sh, on=["date", "code"], how="left")

    # 估值
    eps = pd.to_numeric(snap.get("eps_ttm"), errors="coerce")
    bvps = pd.to_numeric(snap.get("bvps"), errors="coerce")
    snap["pe"] = np.where(eps > 0, snap["close"] / eps, np.nan)
    snap["pb"] = np.where(bvps > 0, snap["close"] / bvps, np.nan)
    snap["ep"] = np.where(snap["close"] > 0, eps / snap["close"], np.nan)     # 盈利收益率
    ocf = pd.to_numeric(snap.get("ocfps_ttm"), errors="coerce")
    snap["pcf"] = np.where(ocf > 0, snap["close"] / ocf, np.nan)

    # 市值（优先用财报反推的总股本，缺失时用流通股本）
    shares = pd.to_numeric(snap.get("shares"), errors="coerce")
    if "float_share" in snap.columns:
        shares = shares.fillna(pd.to_numeric(snap["float_share"], errors="coerce"))
    snap["total_share"] = shares
    snap["mktcap"] = snap["close"] * shares

    # 股息率：过去 12 个月每股派息 / 现价
    if dividends is not None and not dividends.empty:
        snap = _attach_dividend_yield(snap, dividends)
        # 过去 3 年平均股息率（中证红利低波 H30269 的选样口径）
        snap = _attach_dividend_yield(snap, dividends, days=1095,
                                      tag="3y", annualize=3.0)
        snap = _attach_div_years(snap, dividends)
    else:
        snap["div_years3"] = np.nan

    # ---------------------------------------------------------------- 新增因子
    # Novy-Marx 毛利润/总资产（2013《The Other Side of Value》）
    #   GP/A = (营收-营业成本)/总资产 = 销售毛利率 × 总资产周转率
    #   两个乘数都是财报原始字段，无需额外数据。
    #   注意：新浪的「总资产周转率」是报告期累计值（Q1 只有全年的约 1/4），
    #   实测中位 0.137（单季），必须按报告期年化后才与论文口径可比。
    gm = pd.to_numeric(snap.get("gross_margin"), errors="coerce") / 100.0
    at = pd.to_numeric(snap.get("asset_turnover"), errors="coerce")
    qtr = pd.to_numeric(snap.get("quarter"), errors="coerce")
    ann = pd.Series(np.where(qtr > 0, 4.0 / qtr, np.nan),
                    index=snap.index).fillna(4.0)
    snap["gp_to_assets"] = gm * at * ann

    # Sloan 应计（1996）：应计 = (TTM 净利润 - TTM 经营现金流) / 总资产
    #   应计越高 = 利润里「非现金」成分越多 = 盈余质量越差
    ta = pd.to_numeric(snap.get("total_assets"), errors="coerce")
    ni_amt = eps * shares
    ocf_amt = ocf * shares
    snap["accruals"] = np.where(
        (ta > 0) & ta.notna() & ni_amt.notna() & ocf_amt.notna(),
        (ni_amt - ocf_amt) / ta.where(ta > 0),
        np.nan,
    )

    # Cooper 资产增长率已在财务原始字段中（asset_growth），直接用

    # 市赚率 PR（雪球 @ericwarn丁宁）：PR = PE / ROE（ROE 取百分比数值）
    #   1PR 为合理估值，0.4~0.6PR 为理想买入区
    roe = pd.to_numeric(snap.get("roe_ttm"), errors="coerce")
    pr = pd.Series(
        np.where((roe > 0) & (snap["pe"] > 0), snap["pe"] / roe, np.nan),
        index=snap.index,
    )
    snap["pr"] = pr

    # 修正系数 N：以 50% 股利支付率为标杆，分红越少越可能「赚假钱」，PR 上调
    #   支付率≥50% → 1.0；≤25% → 2.0；其间 → 50%/支付率
    #   财报的「股息发放率(%)」字段口径不可靠（实测中位 0.025，不是百分比），
    #   因此优先用自算派息率 = 近 12 月每股派息 / TTM 每股收益
    eps_v = pd.to_numeric(snap.get("eps_ttm"), errors="coerce")
    dps_v = pd.to_numeric(snap.get("div_ps_12m"), errors="coerce")
    calc = pd.Series(
        np.where((eps_v > 0) & dps_v.notna(), dps_v / eps_v * 100.0, np.nan),
        index=snap.index,
    )
    payout = calc.fillna(pd.to_numeric(snap.get("payout_ratio"), errors="coerce"))
    safe = payout.where(payout > 0)
    n = (50.0 / safe).clip(lower=1.0, upper=2.0).fillna(2.0)
    snap["pr_n"] = n
    snap["pr_adj"] = n * pr

    # ---------- 价格类因子（动量/反转/波动/IVOL/Beta）----------
    # 全部基于后复权价的收益率；只用调仓日之前的信息，无未来函数。
    if close_hfq is not None and not close_hfq.empty:
        snap = _attach_price_factors(snap, close_hfq)

    # ---------- 换手率 / 流动性（第四轮新增）----------
    if volume is not None and not volume.empty:
        snap = _attach_turnover_factors(snap, volume, share, close_raw, close_hfq)

    snap = _attach_extra_factors(snap)
    # 换手 / 流动性与市值的秩相关高达 -0.30 / -0.77，必须给出中性化版本，
    # 否则「低换手」与「非流动性」因子的收益可能只是小市值的另一种说法。
    snap = _neutralize(snap, ["sue", "sue_deduct", "turnover_1m", "illiq_1y"],
                       "mktcap")
    snap = snap.set_index(["date", "code"]).sort_index()
    return snap


def _neutralize(snap: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    """把因子对 `by`（默认市值）做截面中性化，输出 `*_neu` 列。

    方法：每期截面用 OLS 回归 f ~ a + b·ln(by)，取残差。
    目的：判断因子的收益是来自自身，还是只是「恰好选到了小盘股」。
    """
    if by not in snap.columns:
        return snap
    x = np.log(pd.to_numeric(snap[by], errors="coerce").where(lambda v: v > 0))
    for c in cols:
        if c not in snap.columns:
            continue
        y = pd.to_numeric(snap[c], errors="coerce")
        out = pd.Series(np.nan, index=snap.index, dtype=float)
        for d, idx in snap.groupby("date", sort=False).groups.items():
            xv = x.loc[idx].values
            yv = y.loc[idx].values
            m = np.isfinite(xv) & np.isfinite(yv)
            if m.sum() < 30:
                continue
            X = np.column_stack([np.ones(m.sum()), xv[m]])
            beta, *_ = np.linalg.lstsq(X, yv[m], rcond=None)
            out.loc[idx[m]] = yv[m] - X @ beta
        snap[f"{c}_neu"] = out
    return snap


def _attach_extra_factors(snap: pd.DataFrame) -> pd.DataFrame:
    """第四轮新增的估值 / 质量 / 成长类因子（全部由现有字段派生）。"""
    close = pd.to_numeric(snap.get("close"), errors="coerce")
    eps = pd.to_numeric(snap.get("eps_ttm"), errors="coerce")
    ocf = pd.to_numeric(snap.get("ocfps_ttm"), errors="coerce")
    shares = pd.to_numeric(snap.get("total_share"), errors="coerce")
    ta = pd.to_numeric(snap.get("total_assets"), errors="coerce")

    # 现金流收益率 CFP（经营现金流 / 价格）。国投证券：剔除 FCFE<0 后
    # CFP 的 RankIC 由 0.81% 提升到 1.75%，故只保留现金流为正的样本。
    snap["cfp_yield"] = np.where((ocf > 0) & (close > 0), ocf / close, np.nan)

    # 剔除微利股的盈利收益率 EP（国投证券：剔归母净利<1000万后
    # EP 的 RankIC 6.61% -> 6.86%，多空年化 18.82% -> 23.19%）
    ni_amt = eps * shares
    snap["ep_ex"] = np.where((ni_amt >= 1e7) & (close > 0), eps / close, np.nan)

    # EBIT/EV（Greenblatt 的 Acquirer's Multiple）近似：
    #   EBIT ≈ 营业利润率 × 年化营收；年化营收 ≈ 总资产周转率(年化) × 总资产
    #   EV  ≈ 市值 + 总负债（**缺现金**，因此 EV 被系统性高估，EBIT/EV 偏低估）
    #   只能用于同截面排序，绝对值不可与论文直接比较。
    opm = pd.to_numeric(snap.get("op_margin2"), errors="coerce") / 100.0
    at = pd.to_numeric(snap.get("asset_turnover"), errors="coerce")
    qtr = pd.to_numeric(snap.get("quarter"), errors="coerce")
    ann = pd.Series(np.where(qtr > 0, 4.0 / qtr, np.nan),
                    index=snap.index).fillna(4.0)
    rev_est = at * ann * ta
    debt = ta * pd.to_numeric(snap.get("debt_ratio"), errors="coerce") / 100.0
    ev = pd.to_numeric(snap.get("mktcap"), errors="coerce") + debt.fillna(0.0)
    ebit = opm * rev_est
    snap["ebit_ev"] = np.where((ev > 0) & ebit.notna(), ebit / ev, np.nan)

    # 业绩加速度：增速的变化（二阶导）。只用同比增速字段的一阶差分，
    # lag4 由 prepare_financials 生成，天然对齐到去年同季。
    for src, out in (("ni_growth", "d_ni_growth"), ("rev_growth", "d_rev_growth")):
        cur = pd.to_numeric(snap.get(src), errors="coerce")
        pre = pd.to_numeric(snap.get(f"{src}_lag4"), errors="coerce")
        snap[out] = cur - pre

    # 股息增长：近 1 年派息 / 近 3 年年均派息 − 1（红利质量：能持续增长的分红）
    d1 = pd.to_numeric(snap.get("div_ps_12m"), errors="coerce")
    d3 = pd.to_numeric(snap.get("div_ps_3y"), errors="coerce") / 3.0
    snap["div_growth"] = np.where((d1 > 0) & (d3 > 0), d1 / d3 - 1.0, np.nan)

    # 盈利质量：经营现金流 / 净利润（财报原字段，越高说明利润含金量越高）
    src = snap["ocf_to_ni"] if "ocf_to_ni" in snap.columns else np.nan
    snap["ocf_ni_ratio"] = pd.to_numeric(src, errors="coerce") \
        if isinstance(src, pd.Series) else pd.Series(np.nan, index=snap.index)
    return snap


def _attach_div_years(snap: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    """过去 3 个完整年度中有现金分红的年数（0~3）。

    用「除权除息日」所在年份归集，按 code 累计有派息的年数后做差分，
    因此不需要逐年循环，且天然只用调仓日之前的信息。
    """
    d = div.copy()
    d["code"] = d["code"].astype(str).str.zfill(6)
    if "除权除息日" not in d.columns or "派息" not in d.columns:
        snap["div_years3"] = np.nan
        return snap
    d["ex_date"] = pd.to_datetime(d["除权除息日"], errors="coerce")
    d["cash"] = pd.to_numeric(d["派息"], errors="coerce") / 10.0
    d = d.dropna(subset=["ex_date", "cash"])
    d = d[d["cash"] > 0]
    if d.empty:
        snap["div_years3"] = np.nan
        return snap

    d["year"] = d["ex_date"].dt.year
    yr = d.groupby(["code", "year"])["cash"].sum().reset_index()
    yr["has"] = (yr["cash"] > 0).astype(float)
    yr = yr.sort_values(["code", "year"])
    yr["cum"] = yr.groupby("code")["has"].cumsum()
    cum = yr[["code", "year", "cum"]].rename(columns={"year": "y"})
    cum = cum.sort_values("y", kind="mergesort")

    key = snap[["date", "code"]].copy()
    key["_row"] = np.arange(len(key))
    key["y"] = key["date"].dt.year - 1          # 当年分红尚未派完，只统计完整年度
    key = key.sort_values("y", kind="mergesort")
    cur = pd.merge_asof(key, cum, left_on="y", right_on="y", by="code",
                        direction="backward")
    back = key.copy()
    back["y"] = back["y"] - 3
    back = back.sort_values("y", kind="mergesort")
    old = pd.merge_asof(back, cum, left_on="y", right_on="y", by="code",
                        direction="backward")

    cur = cur.sort_values("_row").reset_index(drop=True)
    old = old.sort_values("_row").reset_index(drop=True)
    n = cur["cum"].fillna(0.0).values - old["cum"].fillna(0.0).values
    snap["div_years3"] = np.clip(n, 0, 3)
    return snap


def _attach_turnover_factors(snap: pd.DataFrame, volume: pd.DataFrame,
                             share: pd.DataFrame | None,
                             close_raw: pd.DataFrame,
                             close_hfq: pd.DataFrame | None) -> pd.DataFrame:
    """换手率与非流动性。

    - 低换手率：A 股散户占比高，换手率是**典型的负向因子**（换手越高未来收益越低），
      量级通常强于估值类；其经济学解释是「高换手 = 投机关注度溢价被透支」
    - 异常换手 turn_abn = 近 1 月换手 / 近 1 年换手：剔除各股的固有换手水平，
      只看「近期是否被爆炒」
    - Amihud 非流动性 ILLIQ = mean(|日收益| / 日成交额)：值越大越难交易，
      理论上应有流动性溢价补偿（正向因子）
    """
    vol = volume.reindex(index=close_raw.index, columns=close_raw.columns)
    if share is not None:
        fl = share.reindex(index=close_raw.index, columns=close_raw.columns)
        fl = fl.ffill()
    else:
        fl = None
    to = (vol / fl) if fl is not None else None

    frames = []
    if to is not None:
        to = to.where(np.isfinite(to))
        for w, tag in ((21, "1m"), (63, "3m"), (250, "1y")):
            mp = max(int(w * 0.6), 10)
            t = to.rolling(w, min_periods=mp).mean() * 100.0
            frames.append(_stack(t, f"turnover_{tag}"))
        t1 = to.rolling(21, min_periods=13).mean()
        t12 = to.rolling(250, min_periods=150).mean()
        frames.append(_stack(t1 / t12.replace(0, np.nan), "turn_abn"))

    # Amihud ILLIQ：|r| / 成交额(亿元)，放大 1e8 便于阅读
    if close_hfq is not None:
        ret = close_hfq.reindex(index=close_raw.index,
                                columns=close_raw.columns).pct_change()
        amount = vol * close_raw
        illiq = (ret.abs() / amount.replace(0, np.nan)).rolling(
            250, min_periods=120).mean() * 1e8
        frames.append(_stack(illiq, "illiq_1y"))
        # 日成交额本身（流动性筛选用，单位：元）
        amt = amount.rolling(21, min_periods=13).mean()
        frames.append(_stack(amt, "amount_1m"))

    for f in frames:
        snap = snap.merge(f, on=["date", "code"], how="left")
    return snap


def _stack(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """宽表(日期×代码) -> 长表 [date, code, name]。"""
    out = df.stack().rename(name).reset_index()
    out.columns = ["date", "code", name]
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str)
    return out


def _attach_price_factors(snap: pd.DataFrame, close_hfq: pd.DataFrame) -> pd.DataFrame:
    """动量 / 反转 / 已实现波动 / 特质波动 / Beta。

    - 反转（A 股 1 个月内最有效的异象）：取过去 N 日涨幅的**相反数**
    - 动量：A 股长期动量不成立，仅作对照
    - IVOL：CAPM 残差波动（对池内等权收益回归），Ang(2006)/中信实证低 IVOL 有超额
    - Beta：Frazzini-Pedersen BAB 的低 Beta 腿
    """
    px = close_hfq.sort_index()
    ret = px.pct_change()
    mkt = ret.mean(axis=1)                      # 池内等权收益作为市场代理

    # 过去一年已实现波动率（年化 %），已有字段，保留
    vol = ret.rolling(250, min_periods=120).std() * np.sqrt(250) * 100.0
    frames = [_stack(vol, "vol_1y")]

    # 短窗口波动（60 日），低波策略常用口径
    vol60 = ret.rolling(60, min_periods=40).std() * np.sqrt(250) * 100.0
    frames.append(_stack(vol60, "vol_60d"))

    # 反转 / 动量：N 日累计涨幅
    for w, tag in ((21, "1m"), (63, "3m"), (126, "6m")):
        r = px / px.shift(w) - 1.0
        frames.append(_stack(r * 100.0, f"ret_{tag}"))
    # 12-1 动量：跳过最近 1 个月，避免短期反转污染
    mom121 = px.shift(21) / px.shift(252) - 1.0
    frames.append(_stack(mom121 * 100.0, "mom_12_1"))

    # Beta / IVOL：250 日滚动 CAPM 回归
    rm = ret.mul(mkt, axis=0)
    cov = rm.rolling(250, min_periods=120).mean() - (
        ret.rolling(250, min_periods=120).mean().mul(mkt.rolling(250).mean(), axis=0)
    )
    var = mkt.rolling(250, min_periods=120).var()
    beta = cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)
    resid = ret - beta.mul(mkt, axis=0)
    ivol = resid.rolling(250, min_periods=120).std() * np.sqrt(250) * 100.0
    frames.append(_stack(beta, "beta_1y"))
    frames.append(_stack(ivol, "ivol_1y"))

    for f in frames:
        snap = snap.merge(f, on=["date", "code"], how="left")
    return snap


def _attach_dividend_yield(snap: pd.DataFrame, div: pd.DataFrame,
                           days: int = 365, tag: str = "12m",
                           annualize: float = 1.0) -> pd.DataFrame:
    """按除权除息日汇总过去 `days` 天内的每股现金分红。

    annualize 用于把多年累计折算成年均（红利低波指数要求「过去 3 年平均」）。
    """
    d = div.copy()
    d["code"] = d["code"].astype(str).str.zfill(6)
    if "除权除息日" not in d.columns or "派息" not in d.columns:
        return snap
    d["ex_date"] = pd.to_datetime(d["除权除息日"], errors="coerce")
    d["cash"] = pd.to_numeric(d["派息"], errors="coerce")
    d = d.dropna(subset=["ex_date", "cash"])
    if d.empty:
        return snap
    # 新浪 stock_history_dividend_detail 的「派息」是「每 10 股派 X 元」。
    # 已用三个已知样本核对：茅台每10股派 308.76（每股约 30.9 元）、
    # 工行 3.035（每股 0.30 元）、招行 20.0（每股 2.0 元），均与公开分红方案一致。
    d["cash_ps"] = d["cash"] / 10.0
    d = d.sort_values("ex_date")

    d = d.sort_values(["code", "ex_date"]).reset_index(drop=True)
    d["cum_cash"] = d.groupby("code")["cash_ps"].cumsum()
    cum = d[["ex_date", "code", "cum_cash"]].rename(columns={"ex_date": "date"})
    cum = cum.sort_values("date", kind="mergesort")
    cum_idx = cum.set_index(["code", "date"])["cum_cash"]

    snap = snap.reset_index(drop=True)
    key = snap[["date", "code"]].copy()
    key["_row"] = np.arange(len(key))
    key = key.sort_values("date", kind="mergesort")

    # 截至调仓日的累计派息
    a = pd.merge_asof(key, cum, on="date", by="code", direction="backward")
    # 截至「调仓日 - 365 天」的累计派息，两者相减即过去 12 个月派息
    back = key.copy()
    back["date"] = back["date"] - pd.Timedelta(days=365)
    back = back.sort_values("date", kind="mergesort")
    b = pd.merge_asof(back, cum, on="date", by="code", direction="backward")

    a = a.sort_values("_row").reset_index(drop=True)
    b = b.sort_values("_row").reset_index(drop=True)
    div_n = a["cum_cash"].fillna(0.0).values - b["cum_cash"].fillna(0.0).values
    ps = np.where(div_n > 0, div_n, 0.0)
    snap[f"div_ps_{tag}"] = ps
    ycol = "dividend_yield" if tag == "12m" else f"dividend_yield_{tag}"
    snap[ycol] = np.where(snap["close"] > 0, ps / annualize / snap["close"] * 100, np.nan)
    return snap
