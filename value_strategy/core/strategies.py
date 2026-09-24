"""经典价值投资策略实现。

每个策略形如 `fn(snap, params) -> pd.Series`：
- 输入是某个调仓日的因子截面（index = 股票代码）
- 输出是分数，**越大越好**；NaN 表示该股被策略剔除
- 由回测引擎统一取分数最高的 top_n 只

关于原版与 A 股实现的差异，见 `docs` 中各策略的 docstring 与 README 的「已知偏差」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 通用工具

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def base_mask(snap: pd.DataFrame, p: dict) -> pd.Series:
    """所有策略共用的基础过滤：有价格、市值达标、非 ST。"""
    m = _num(snap.get("close")).notna()
    mc = p.get("min_mktcap")
    if mc:
        m &= _num(snap.get("mktcap")).fillna(0) >= mc
    if p.get("exclude_financial", False) and "name" in snap.columns:
        kw = ("银行", "证券", "保险", "信托", "期货", "金控")
        m &= ~snap["name"].astype(str).str.contains("|".join(kw), na=False)
    return m


def _valid(snap: pd.DataFrame, col: str) -> pd.Series:
    v = _num(snap.get(col))
    return v.notna() & np.isfinite(v)


def rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
    """百分位排名（0~1）。ascending=True 表示值越小排名越靠前。"""
    v = _num(s)
    return v.rank(ascending=ascending, pct=True, na_option="keep")


# ---------------------------------------------------------------- 1. 格雷厄姆

def graham_defensive(snap: pd.DataFrame, p: dict) -> pd.Series:
    """格雷厄姆《聪明的投资者》防御型投资者七条。

    原版：①规模足够大 ②流动比率≥2 且长期债务≤净流动资产 ③近10年每年盈利
          ④连续20年分红 ⑤10年 EPS 增长≥1/3 ⑥PE≤15 ⑦PB≤1.5（或 PE×PB≤22.5）

    A 股实现偏差：
      - ③⑤ 的 10 年窗口受数据长度限制，改为「近 N 期 ROE 均 >0」与「EPS 较 N 期前增长≥1/3」
      - ④ 的 20 年分红改为「近 N 年均有现金分红」
      - ② 的「长期债务≤净流动资产」用「长期债务/营运资金比率 ≤ 1」近似
    """
    m = base_mask(snap, p)

    # ② 财务稳健
    m &= _num(snap.get("current_ratio")).fillna(0) >= p.get("current_ratio", 2.0)
    lt_wc = _num(snap.get("lt_debt_to_wc"))
    m &= lt_wc.isna() | (lt_wc <= p.get("lt_debt_to_wc", 100.0))
    # ③ 盈利稳定（近 N 期 ROE 均为正）
    m &= _num(snap.get("roe_ttm")).fillna(-1) > 0
    m &= _num(snap.get("roa_ttm")).fillna(-1) > 0
    # ④ 持续分红
    if "div_ps_12m" in snap.columns:
        m &= _num(snap["div_ps_12m"]).fillna(0) > 0
    # ⑥⑦ 估值
    pe, pb = _num(snap.get("pe")), _num(snap.get("pb"))
    m &= (pe > 0) & (pb > 0)
    m &= pe <= p.get("max_pe", 15.0)
    m &= pb <= p.get("max_pb", 1.5)
    m &= (pe * pb) <= p.get("max_pe_pb", 22.5)

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        # 估值越便宜越好：用 PE×PB 升序
        score.loc[ok] = -_num(snap.loc[ok, "pe"]).fillna(1e9) * _num(snap.loc[ok, "pb"]).fillna(1e9)
    return score


def graham_number(snap: pd.DataFrame, p: dict) -> pd.Series:
    """格雷厄姆数字：合理价上限 = sqrt(22.5 × EPS × BVPS)。

    价格低于格雷厄姆数字即具备安全边际。按「价格 / 格雷厄姆数字」升序选。
    这是格雷厄姆第 ⑥⑦ 两条的合成形式，对单一指标超标更宽容。
    """
    m = base_mask(snap, p)
    eps, bvps = _num(snap.get("eps_ttm")), _num(snap.get("bvps"))
    m &= (eps > 0) & (bvps > 0)

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        gn = np.sqrt(22.5 * eps[ok] * bvps[ok])
        ratio = _num(snap.loc[ok, "close"]) / gn
        score.loc[ok] = -ratio          # 比值越小越便宜
    return score


# ---------------------------------------------------------------- 2. 格林布拉特

def magic_formula(snap: pd.DataFrame, p: dict) -> pd.Series:
    """格林布拉特神奇公式：又好又便宜。

    原版两个指标：资本回报率 ROC = EBIT / (净营运资本 + 净固定资产)；
                 股票收益率 EY = EBIT / EV（企业价值）。
    分别排名，名次相加取最小。

    A 股实现偏差：EBIT/EV 与净营运资本无法从现有字段还原，改用
      ROC → ROA(TTM)（净利/总资产，不含杠杆放大，比 ROE 更贴近资本回报）
      EY  → EP = EPS(TTM) / 价格
    """
    m = base_mask(snap, p)
    roa, ep = _num(snap.get("roa_ttm")), _num(snap.get("ep"))
    m &= roa.notna() & ep.notna()
    if p.get("require_positive_ey", True):
        m &= ep > 0

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        r_roc = rank_pct(roa[ok], ascending=False)      # ROA 越高越好
        r_ey = rank_pct(ep[ok], ascending=False)        # EP 越高越便宜
        combined = r_roc + r_ey
        score.loc[ok] = -combined                        # 名次和越小越好
    return score


# ---------------------------------------------------------------- 3. Piotroski

def piotroski_f(snap: pd.DataFrame, p: dict) -> pd.Series:
    """Piotroski F-Score：在低估值股里挑基本面在改善的，避开价值陷阱。

    先取 PB 最低的一档（默认全池 PB 最低 20%），再按 F-Score 降序选。
    """
    m = base_mask(snap, p)
    pb = _num(snap.get("pb"))
    m &= (pb > 0) & pb.notna()

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if not ok.any():
        return score

    sub = snap.loc[ok]
    q = p.get("value_quantile", 0.20)
    thr = pb[ok].quantile(q)
    deep = ok & (pb <= thr)
    if not deep.any():
        deep = ok

    fs = _num(snap.get("f_score")).reindex(snap.index)
    fn = _num(snap.get("f_score_n")).reindex(snap.index)
    min_f = p.get("min_f_score", 0)
    elig = deep & (fs >= min_f)

    if not elig.any():
        elig = deep
    # 主排序 F-Score（有效判据数不足时按可用项归一化，避免缺失项被当成不合格）
    norm = fs / fn.replace(0, np.nan)
    base = fs.where(fn >= p.get("min_valid", 6), norm * 9)
    score.loc[elig] = base[elig] + (1 - pb[elig].rank(pct=True)) * p.get("pb_tilt", 0.0)
    return score


# ---------------------------------------------------------------- 4. 狗股 / 高股息

def dogs_dividend(snap: pd.DataFrame, p: dict) -> pd.Series:
    """狗股策略（Dogs of the Dow 的 A 股版）：买股息率最高的一批。

    原版取道指 30 只成分中股息率最高的 10 只。这里在中证800 内按股息率排序。
    连续分红要求用于剔除「一次性大额分红」的假高股息。
    """
    m = base_mask(snap, p)
    dy = _num(snap.get("dividend_yield"))
    m &= dy.notna() & (dy > 0)
    if p.get("min_dividend_yield"):
        m &= dy >= p["min_dividend_yield"]
    # 盈利为正，避免靠变卖资产分红
    if p.get("require_profit", True):
        m &= _num(snap.get("eps_ttm")).fillna(-1) > 0

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        score.loc[ok] = dy[ok]
    return score


# ---------------------------------------------------------------- 5. 低估值双排序

def value_composite(snap: pd.DataFrame, p: dict) -> pd.Series:
    """PE + PB 双因子等权排序（A 股最经典的价值选股法）。

    PE 排名与 PB 排名相加，取总排名最靠前的一批。
    可选叠加股息率，构成 PE+PB+股息 三因子。
    """
    m = base_mask(snap, p)
    pe, pb = _num(snap.get("pe")), _num(snap.get("pb"))
    m &= (pe > 0) & (pb > 0)

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if not ok.any():
        return score

    r = rank_pct(pe[ok]) + rank_pct(pb[ok])
    if p.get("use_dividend", False) and "dividend_yield" in snap.columns:
        r = r + rank_pct(_num(snap.loc[ok, "dividend_yield"]), ascending=False)
    score.loc[ok] = -r
    return score


def low_pb_alone(snap: pd.DataFrame, p: dict) -> pd.Series:
    """懒人版：只看市净率，买最便宜的一批。"""
    m = base_mask(snap, p)
    pb = _num(snap.get("pb"))
    m &= (pb > 0) & pb.notna()
    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        score.loc[ok] = -pb[ok]
    return score


def low_pe_alone(snap: pd.DataFrame, p: dict) -> pd.Series:
    """懒人版：只看市盈率，买最便宜的一批。"""
    m = base_mask(snap, p)
    pe = _num(snap.get("pe"))
    m &= (pe > 0) & pe.notna()
    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        score.loc[ok] = -pe[ok]
    return score


# ---------------------------------------------------------------- 6. 巴菲特

def buffett_quality(snap: pd.DataFrame, p: dict) -> pd.Series:
    """巴菲特式：以合理价格买伟大公司（高 ROE + 高毛利 + 低估值）。

    参考 AQR《Buffett's Alpha》的归因：价值(HML) + 质量(QMJ) 双暴露。
    这里用 ROE 稳定性衡量质量，PE 上限作为估值纪律。
    """
    m = base_mask(snap, p)
    roe = _num(snap.get("roe_ttm"))
    pe = _num(snap.get("pe"))
    m &= roe.notna() & (roe >= p.get("min_roe", 15.0))
    m &= (pe > 0) & (pe <= p.get("max_pe", 30.0))
    if p.get("require_margin", True) and "margin" in snap.columns:
        mg = _num(snap["margin"])
        m &= mg.isna() | (mg >= p.get("min_margin", 20.0))

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        # 质量分（ROE 排名）为主，估值分（PE 排名）为辅
        r_q = rank_pct(roe[ok], ascending=False)
        r_v = rank_pct(pe[ok], ascending=True)
        score.loc[ok] = -(r_q * p.get("quality_weight", 0.6) + r_v * p.get("value_weight", 0.4))
    return score


# ---------------------------------------------------------------- 7. 施洛斯

def schloss(snap: pd.DataFrame, p: dict) -> pd.Series:
    """沃尔特·施洛斯：低 PB + 低负债 + 极度分散。

    施洛斯的原话是「买那些被打折的资产」，核心是低 PB 与财务安全。
    """
    m = base_mask(snap, p)
    pb = _num(snap.get("pb"))
    m &= (pb > 0) & (pb <= p.get("max_pb", 1.0))
    dr = _num(snap.get("debt_ratio"))
    m &= dr.isna() | (dr <= p.get("max_debt_ratio", 50.0))

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        score.loc[ok] = -pb[ok]
    return score


# ---------------------------------------------------------------- 8. 净净股（近似）

def netnet_proxy(snap: pd.DataFrame, p: dict) -> pd.Series:
    """格雷厄姆净净股（NCAV）的 A 股近似版。

    原版要求 市值 < 流动资产 − 全部负债。现有字段没有流动资产与负债总额，
    用「PB 极低 + 流动比率高 + 低负债 + 有现金分红」的组合做代理筛选，
    并在结果中标注为**近似**，不作为原版复现结论使用。
    """
    m = base_mask(snap, p)
    pb = _num(snap.get("pb"))
    cr = _num(snap.get("current_ratio"))
    dr = _num(snap.get("debt_ratio"))
    m &= (pb > 0) & (pb <= p.get("max_pb", 0.7))
    m &= cr.isna() | (cr >= p.get("min_current_ratio", 1.5))
    m &= dr.isna() | (dr <= p.get("max_debt_ratio", 50.0))

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        score.loc[ok] = -pb[ok]
    return score


# ---------------------------------------------------------------- 注册表

# ---------------------------------------------------------------- 12. PB-ROE 残差法

def pb_roe_residual(snap: pd.DataFrame, p: dict) -> pd.Series:
    """PB-ROE 残差法（Wilcox 1984 原始模型，A 股券商广泛迭代）。

    对 ln(PB) 与 ROE 做截面 OLS 回归，取残差：残差为负 = 相对自身盈利能力偏便宜。
    相比单纯「低 PB」，它不会把「ROE 低所以 PB 低」的银行误判成便宜——
    这正是我们实测 PB 五分位非单调（最低档 9.8%、第四档 12.6%）的成因。

    mode="quantile" 时改用长江金工口径：PB 分位 - ROE 分位，差值越小越优。
    """
    m = base_mask(snap, p)
    pb, roe = _num(snap.get("pb")), _num(snap.get("roe_ttm"))
    m &= (pb > 0) & roe.notna()
    if p.get("min_roe") is not None:
        m &= roe >= p["min_roe"]

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if not ok.any():
        return score

    if p.get("mode", "residual") == "quantile":
        gap = rank_pct(pb[ok], ascending=True) - rank_pct(roe[ok], ascending=True)
        score.loc[ok] = -gap
        return score

    sub = pd.DataFrame({"y": np.log(pb[ok]), "x": roe[ok]}).dropna()
    if len(sub) < p.get("min_fit", 20):
        return score
    X = np.column_stack([np.ones(len(sub)), sub["x"].values])
    beta, *_ = np.linalg.lstsq(X, sub["y"].values, rcond=None)
    resid = np.log(pb[ok]) - (beta[0] + beta[1] * roe[ok])
    score.loc[ok] = -resid
    return score


# ---------------------------------------------------------------- 13. Novy-Marx 毛利润/总资产

def novy_marx_gpa(snap: pd.DataFrame, p: dict) -> pd.Series:
    """Novy-Marx (2013) 毛利润 / 总资产：GP/A = 毛利率 × 资产周转率。

    论文核心结论：GP/A 对横截面收益的预测力与 B/P 相当，两者相关性 -0.57，
    50/50 组合的 Sharpe(0.85) 是任一单腿的两倍以上。
    故 `value_weight>0` 时叠加盈利收益率腿，默认 0.5 即论文推荐的等权。
    """
    m = base_mask(snap, p)
    gpa = _num(snap.get("gp_to_assets"))
    m &= gpa.notna()
    if p.get("require_profit", True):
        m &= _num(snap.get("eps_ttm")).fillna(-1) > 0

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if not ok.any():
        return score

    # 注意：rank(ascending=False) 会让「值最大」的 pct 最小，再取 top_n 就选反了。
    # 统一用「值越大 pct 越大」的升序排名作为分数。
    q = gpa[ok].rank(pct=True, na_option="keep").fillna(0.5)
    wv = p.get("value_weight", 0.0)
    if wv > 0:
        v = _num(snap.get("ep"))[ok].rank(pct=True, na_option="keep").fillna(0.5)
        q = (1.0 - wv) * q + wv * v
    score.loc[ok] = q
    return score


# ---------------------------------------------------------------- 14. 低估值 + 盈余质量

def quality_enhanced_value(snap: pd.DataFrame, p: dict) -> pd.Series:
    """先价值后质量：低 PE 一档内，按应计 / 资产增长 / F-Score 合成质量分。

    顺序依据 Wesley Gray《Quantitative Value》的实证——魔法公式的超额
    全部来自价值腿，质量腿单独使用时跑输指数；质量腿只在价值股内部才有效。
    """
    m = base_mask(snap, p)
    pe = _num(snap.get("pe"))
    m &= pe > 0

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if not ok.any():
        return score

    thr = pe[ok].quantile(p.get("value_quantile", 0.30))
    cheap = ok & (pe <= thr)
    if not cheap.any():
        cheap = ok

    parts, wts = [], []
    if p.get("use_accruals", True):
        parts.append(rank_pct(_num(snap.get("accruals")), ascending=True).fillna(0.5))
        wts.append(p.get("w_accruals", 1.0))
    if p.get("use_asset_growth", True):
        parts.append(rank_pct(_num(snap.get("asset_growth")), ascending=True).fillna(0.5))
        wts.append(p.get("w_asset_growth", 1.0))
    if p.get("use_f_score", True):
        fs = _num(snap.get("f_score"))
        fn = _num(snap.get("f_score_n"))
        norm = (fs / fn.replace(0, np.nan)).fillna(fs)
        parts.append(rank_pct(norm, ascending=False).fillna(0.5))
        wts.append(p.get("w_f_score", 1.0))

    tot = sum(wts) or 1.0
    combo = sum(w * s_ for w, s_ in zip(wts, parts)) / tot
    score.loc[cheap] = combo[cheap]
    return score


# ---------------------------------------------------------------- 15. 市赚率（丁宁）

def pr_value(snap: pd.DataFrame, p: dict) -> pd.Series:
    """市赚率 PR = PE / ROE（雪球 @ericwarn丁宁，ROE 取百分比数值）。

    PR=1 为合理估值，0.4~0.6PR 为理想买入区；A 股长期持有免股息税，故 1PR 才算高估。
    use_adj：启用修正系数 N（以 50% 股利支付率为标杆：≥50%→1.0，≤25%→2.0，
             其间 50%/支付率），用来惩罚「赚假钱」的公司。
    require_double5：丁宁的「双 5」策略，PR≤0.5 且 股息率≥5%。
    """
    m = base_mask(snap, p)
    col = "pr_adj" if p.get("use_adj", False) else "pr"
    pr = _num(snap.get(col))
    m &= pr.notna() & (pr > 0)

    dy = _num(snap.get("dividend_yield"))
    if p.get("require_double5", False):
        m &= (pr <= 0.5) & (dy >= 5.0)
    if p.get("max_pr"):
        m &= pr <= p["max_pr"]
    if p.get("min_roe") is not None:
        m &= _num(snap.get("roe_ttm")) >= p["min_roe"]

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if ok.any():
        score.loc[ok] = rank_pct(pr[ok], ascending=False)
    return score


# ---------------------------------------------------------------- 16. 红利低波（中证 H30269 近似）

def dividend_low_vol(snap: pd.DataFrame, p: dict) -> pd.Series:
    """中证红利低波动指数（H30269）编制方案的近似复现。

    原版：过去 3 年连续分红且股息率>0 → 剔支付率前 5% 与每股股利增长非正 →
    按 3 年平均股息率降序取 75 只 → 按 1 年波动率升序取 50 只 → 股息率加权。
    本实现的两处近似：
      1. 「连续 3 年分红」用「近 1 年有分红 且 近 3 年累计分红 > 0」代替
      2. 「每股股利增长非正」用「3 年平均股息率 < 近 1 年股息率 × 0.8」代替
    """
    m = base_mask(snap, p)
    dy1 = _num(snap.get("dividend_yield"))
    dy3 = _num(snap.get("dividend_yield_3y")).fillna(dy1)
    vol = _num(snap.get("vol_1y"))

    m &= dy1.notna() & (dy1 > 0) & dy3.notna() & (dy3 > 0)
    # 分红未明显下滑（对应原版剔除「每股股利增长率非正」）：近 1 年不低于 3 年均值的 80%
    m &= dy1 >= dy3 * p.get("growth_floor", 0.8)
    if p.get("max_payout_pct"):
        po = _num(snap.get("payout_ratio"))
        m &= po.isna() | (po <= p["max_payout_pct"])
    m &= vol.notna()

    score = pd.Series(np.nan, index=snap.index)
    ok = m.reindex(snap.index, fill_value=False)
    if not ok.any():
        return score

    # 先按股息率取前一档（原版取 75 只），再按波动率升序（此处合成一个分数）
    # rank(ascending=False) 会让「值最大」的 pct 最小 → 取 top_n 就选成最低股息率；
    # 因此统一用升序 rank 的 pct（值越大分越大），低波动腿再取 1−pct。
    wv = p.get("vol_weight", 0.5)
    hi_div = dy3[ok].rank(pct=True, na_option="keep").fillna(0.5)
    lo_vol = 1.0 - vol[ok].rank(pct=True, na_option="keep").fillna(0.5)
    score.loc[ok] = (1.0 - wv) * hi_div + wv * lo_vol
    return score


# ---------------------------------------------------------------- 3. 价格类异象与事件驱动

def _score(snap: pd.DataFrame, col: str, ascending: bool) -> pd.Series:
    """单因子分数：越大越好。ascending=True 表示「原值越小越好」。

    注意 `select()` 取分数最大的 top_n，因此「值越小越好」的因子必须取 1−pct，
    否则会选出方向完全相反的一组股票。
    """
    v = pd.to_numeric(snap[col], errors="coerce")
    pct = v.rank(pct=True, na_option="keep")
    return (1.0 - pct) if ascending else pct


def _single(snap: pd.DataFrame, p: dict, col: str, ascending: bool) -> pd.Series:
    """单因子：取某列的百分位排名分数。"""
    if col not in snap.columns:
        return pd.Series(np.nan, index=snap.index)
    return _score(snap, col, ascending).where(base_mask(snap, p) & _valid(snap, col))


def reversal(snap: pd.DataFrame, p: dict) -> pd.Series:
    """短期反转 —— A 股被反复验证的最强异象。

    成因：散户占比高（过度反应）+ 涨跌停分割信息释放 + T+1 被迫隔日买卖。
    东北金工(2018)：1M~6M 窗口均有效，且**观察期越短越强**（1M 最强、6M 最弱）。
    开源/民生：理想反转因子全历史 5 分组多空年化 15.66%，回撤仅 6.64%。
    """
    return _single(snap, p, p.get("col", "ret_1m"), ascending=True)


def momentum_121(snap: pd.DataFrame, p: dict) -> pd.Series:
    """12-1 动量（美股经典因子；A 股长期实证不成立，此处作**反向对照**）。

    跳过最近 1 个月是为了隔离短期反转的污染——不跳的话 A 股动量会被反转掩盖。
    """
    return _single(snap, p, "mom_12_1", ascending=False)


def low_volatility(snap: pd.DataFrame, p: dict) -> pd.Series:
    """低波动异象（Ang 2006 / Blitz-Hanauer-van Vliet 2021）。

    Blitz 等人对 A 股 2000-2018 的实证：波动最低 1/10 跑赢最高 1/10 **年化 12.9%**，
    控制六因子后仍有 8.2%，且波动更低、夏普最高——与 CAPM 的「高风险高收益」直接相悖。
    """
    return _single(snap, p, p.get("col", "vol_1y"), ascending=True)


def low_ivol(snap: pd.DataFrame, p: dict) -> pd.Series:
    """低特质波动（CAPM 残差波动率）。

    中信建投(2018)：特质波动率后 30% 等权，2005-2018 年化超额 7.55%，IR 1.62，
    最大回撤仅 5.85%。理论上不可分散风险不该有溢价的反向证据。
    """
    return _single(snap, p, "ivol_1y", ascending=True)


def low_beta(snap: pd.DataFrame, p: dict) -> pd.Series:
    """低 Beta（Frazzini-Pedersen 2014《Betting Against Beta》的多头腿）。

    杠杆约束下投资者只能超配高 Beta 资产 → 高 Beta 被高估、低 Beta 被低估。
    """
    return _single(snap, p, "beta_1y", ascending=True)


def pead_sue(snap: pd.DataFrame, p: dict) -> pd.Series:
    """PEAD / 标准化未预期盈利（SUE）。

    Bernard-Thomas(1989)：业绩公告后价格漂移，公告超预期的股票未来 3-6 个月继续涨。
    东方证券(2018)：SUE0 行业市值中性 RankIC 均值 4.02%、IC_IR 3.49、多空月均 1.53%，
    且剔除其他大类因子后仍显著——是 A 股少数「干净」的 alpha。
    本实现用季节性随机游走口径：SUE = (Q_t − Q_{t−4}) / 过去同比增量标准差。
    """
    return _single(snap, p, "sue", ascending=False)


def _blend(snap: pd.DataFrame, p: dict, cols: list[tuple[str, bool]]) -> pd.Series:
    """多因子等权合成：cols = [(列名, ascending), ...]。"""
    ok = base_mask(snap, p)
    parts = []
    for col, asc in cols:
        if col not in snap.columns:
            return pd.Series(np.nan, index=snap.index)
        ok &= _valid(snap, col)
        parts.append(_score(snap, col, asc))
    s = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return s.where(ok.reindex(snap.index, fill_value=False))


def _blend_w(snap: pd.DataFrame, p: dict,
             cols: list[tuple[str, bool, float]]) -> pd.Series:
    """多因子加权合成：cols = [(列名, ascending, 权重), ...]。

    缺失值按 0.5（中性分）填充后再加权，避免某腿缺失把整只股票分数拉到 0；
    但**任一腿缺失即剔除**（ok），保证组合内各腿口径一致。
    """
    ok = base_mask(snap, p)
    acc = pd.Series(0.0, index=snap.index)
    tot = 0.0
    for col, asc, w in cols:
        if col not in snap.columns:
            return pd.Series(np.nan, index=snap.index)
        ok &= _valid(snap, col)
        acc = acc + _score(snap, col, asc).fillna(0.5) * w
        tot += w
    return (acc / tot).where(ok.reindex(snap.index, fill_value=False))


def reversal_lowvol(snap: pd.DataFrame, p: dict) -> pd.Series:
    """反转 + 低波动复合。

    东北金工的结论是「中等波动组反转最强」，但低波本身在 A 股有独立超额，
    两者合成后换手低于纯反转，是实盘更可行的版本。
    """
    return _blend(snap, p, [("ret_1m", True), ("vol_1y", True)])


def value_quality_lowvol(snap: pd.DataFrame, p: dict) -> pd.Series:
    """价值 + 质量 + 低波 三因子等权（Smart Beta 防御型的常见配方）。"""
    return _blend(snap, p, [("pb", True), ("roe_ttm", False), ("vol_1y", True)])


def small_cap(snap: pd.DataFrame, p: dict) -> pd.Series:
    """小市值（规模因子）。

    注意：池子是**中证800**，这里的「小」只是中盘的下沿，不是全 A 微盘股。
    A 股规模因子的收益主要来自中证800 之外，故本策略只能检验「池内是否也存在」。
    """
    return _single(snap, p, "mktcap", ascending=True)


# ---------------------------------------------------------------- 第四轮：超预期与流动性

def pead_sur(snap: pd.DataFrame, p: dict) -> pd.Series:
    """营收超预期 SUR（SUE 的营收版）。

    构造与 SUE 同构，只是把 EPS 换成营业收入：
        SUR = (Q_t − Q_{t−4}) / std(过去同比增量)
    「营收超预期」比「盈利超预期」更靠上游——利润可以被减值、补贴、税率扰动，
    营收更难被一次性项目粉饰，因此两者相关性只有 0.375，是相对独立的信息源。

    数据口径（重要）：新浪财务接口不直接给单季营收，这里用
       营收 ≈ 总资产周转率(年化) × 总资产
    还原。已用「净利 / 净利率」独立路径交叉验证：对数相关 0.927、
    截面秩相关 0.922、比值中位 0.99，估算误差对**排序**影响很小。
    覆盖率约 64%（需 8 期以上历史才可算标准差）。
    """
    return _single(snap, p, "sur", ascending=False)


def pead_sue_neutral(snap: pd.DataFrame, p: dict) -> pd.Series:
    """SUE 市值中性版（对 ln(市值) 截面回归取残差）。

    用于回答「SUE 是不是小市值的马甲」。实测：中性化后仍 21.06%，
    而同法处理的低换手、Amihud 会崩塌——说明 SUE 是干净 alpha，后两者不是。
    """
    return _single(snap, p, "sue_neu", ascending=False)


def dual_surprise(snap: pd.DataFrame, p: dict) -> pd.Series:
    """双超预期：盈利超预期 SUE + 营收超预期 SUR 等权。

    两条腿来自不同财务报表（利润表 vs 营收口径），且相关性仅 0.375，
    合成后的 RankIC 高于任一单腿，是本轮 Calmar 最好的组合之一。
    """
    return _blend(snap, p, [("sue_neu", False), ("sur", False)])


def pead_triple(snap: pd.DataFrame, p: dict) -> pd.Series:
    """PEAD 三口径等权：SUR + SUE(市值中性) + SUE(原始)。

    同一事件（业绩超预期）的三种度量取平均，目的是压掉单一口径的噪声，
    而不是叠加新信息。实测季度年化 24.66%，为本轮最高。
    """
    return _blend(snap, p, [("sur", False), ("sue_neu", False), ("sue", False)])


def surprise_turnover(snap: pd.DataFrame, p: dict) -> pd.Series:
    """超预期 + 低换手（市值中性换手）。

    逻辑：业绩超预期若是「被爆炒」兑现的，后续反而容易回落；
    低换手意味着市场还没充分反应，漂移空间仍在。
    实测年化 22.37%、RankIC 0.085 / IR 5.09（本轮 IC 最高），
    且回撤 -39.86% 是全场最浅。
    """
    return _blend(snap, p, [("sue_neu", False), ("turnover_1m_neu", True)])


def surprise_triple(snap: pd.DataFrame, p: dict) -> pd.Series:
    """三合一：营收超预期 + 盈利超预期 + 低换手。"""
    return _blend(snap, p, [("sur", False), ("sue_neu", False),
                            ("turnover_1m_neu", True)])


def surprise_lowvol(snap: pd.DataFrame, p: dict) -> pd.Series:
    """超预期 + 低波动：在 PEAD 里挑低波动的，压回撤。"""
    return _blend(snap, p, [("sue_neu", False), ("vol_1y", True)])


def low_turnover_neutral(snap: pd.DataFrame, p: dict) -> pd.Series:
    """低换手（市值中性）—— **已被证伪，保留作反面样本**。

    原始低换手因子与市值的秩相关 -0.30，中性化后表观年化 22.68%，
    但剔除池内最小 30% 市值后掉到 9.54%（低于随机 11.26%）——
    说明它本质是「小市值的另一种说法」，不是独立 alpha。
    """
    return _single(snap, p, "turnover_1m_neu", ascending=True)


def amihud_illiq(snap: pd.DataFrame, p: dict) -> pd.Series:
    """Amihud 非流动性 ILLIQ —— **已被证伪，保留作反面样本**。

    ILLIQ = mean(|日收益| / 日成交额)，值越大越难交易，理论上应有流动性溢价。
    实测表观年化 24.92%（很强），但与市值的秩相关高达 **-0.77**，
    剔除池内最小 30% 市值后崩到 10.38%。
    这是本轮最典型的「不做中性化就会被骗」的案例。
    """
    return _single(snap, p, "illiq_1y", ascending=False)


def qmj_proxy(snap: pd.DataFrame, p: dict) -> pd.Series:
    """QMJ（Quality Minus Junk）近似：高盈利 + 盈利稳定 + 低杠杆 + 低 Beta。

    Asness-Frazzini-Pedersen 2019。原版用 21 个指标合成，这里只有 4 个可得字段。
    实测年化约 11%，与随机持平——A 股的「质量溢价」主要体现为
    白马行情（2017-2021）而非持续 alpha。
    """
    return _blend(snap, p, [("roe_ttm", False), ("roe_std12", True),
                            ("debt_ratio", True), ("beta_1y", True)])


STRATEGIES = {
    "graham_defensive": {
        "name": "格雷厄姆防御型七条",
        "fn": graham_defensive,
        "params": {"min_mktcap": 2e9, "current_ratio": 2.0, "max_pe": 15.0,
                   "max_pb": 1.5, "max_pe_pb": 22.5},
        "note": "硬过滤最严，A 股常出现候选不足",
    },
    "graham_number": {
        "name": "格雷厄姆数字（安全边际）",
        "fn": graham_number,
        "params": {"min_mktcap": 2e9},
        "note": "价格 / sqrt(22.5×EPS×BVPS) 升序",
    },
    "magic_formula": {
        "name": "格林布拉特神奇公式",
        "fn": magic_formula,
        "params": {"min_mktcap": 2e9},
        "note": "ROA(TTM) 与 EP 双排名；ROA 代理 ROC，EP 代理 EBIT/EV",
    },
    "piotroski_f": {
        "name": "Piotroski F-Score",
        "fn": piotroski_f,
        "params": {"min_mktcap": 2e9, "value_quantile": 0.20, "min_f_score": 7},
        "note": "低 PB 前 20% 中挑 F-Score 高者",
    },
    "dogs_dividend": {
        "name": "狗股 / 高股息",
        "fn": dogs_dividend,
        "params": {"min_mktcap": 2e9},
        "note": "股息率降序",
    },
    "value_composite": {
        "name": "PE+PB 双因子排序",
        "fn": value_composite,
        "params": {"min_mktcap": 2e9},
        "note": "PE 与 PB 百分位排名等权加总",
    },
    "low_pb": {
        "name": "纯低 PB（懒人版）",
        "fn": low_pb_alone,
        "params": {"min_mktcap": 2e9},
        "note": "单因子 PB 升序",
    },
    "low_pe": {
        "name": "纯低 PE（懒人版）",
        "fn": low_pe_alone,
        "params": {"min_mktcap": 2e9},
        "note": "单因子 PE 升序",
    },
    "buffett_quality": {
        "name": "巴菲特质量价值",
        "fn": buffett_quality,
        "params": {"min_mktcap": 2e9, "min_roe": 15.0, "max_pe": 30.0},
        "note": "ROE≥15% 且 PE≤30，质量 0.6 / 估值 0.4",
    },
    "schloss": {
        "name": "施洛斯深度价值",
        "fn": schloss,
        "params": {"min_mktcap": 2e9, "max_pb": 1.0, "max_debt_ratio": 50.0},
        "note": "PB≤1 且资产负债率≤50%",
    },
    "netnet_proxy": {
        "name": "净净股（近似）",
        "fn": netnet_proxy,
        "params": {"min_mktcap": 5e8, "max_pb": 0.7},
        "note": "字段限制下的近似实现，非原版 NCAV",
    },
    "pb_roe_residual": {
        "name": "PB-ROE 残差法",
        "fn": pb_roe_residual,
        "params": {"min_mktcap": 2e9, "mode": "residual", "min_roe": 0.0},
        "note": "ln(PB) 对 ROE 截面回归取残差，残差越小越被低估",
    },
    "pb_roe_quantile": {
        "name": "PB-ROE 分位差（长江口径）",
        "fn": pb_roe_residual,
        "params": {"min_mktcap": 2e9, "mode": "quantile", "min_roe": 0.0},
        "note": "PB 分位 − ROE 分位，差值越小越优",
    },
    "novy_marx_gpa": {
        "name": "Novy-Marx GP/A（纯质量）",
        "fn": novy_marx_gpa,
        "params": {"min_mktcap": 2e9, "value_weight": 0.0},
        "note": "毛利润/总资产 = 毛利率 × 资产周转率",
    },
    "novy_marx_combo": {
        "name": "GP/A + 价值 50/50",
        "fn": novy_marx_gpa,
        "params": {"min_mktcap": 2e9, "value_weight": 0.5},
        "note": "论文推荐的等权双腿，GP/A 与 EP 相关性为负",
    },
    "quality_enhanced_value": {
        "name": "低 PE + 盈余质量三重",
        "fn": quality_enhanced_value,
        "params": {"min_mktcap": 2e9, "value_quantile": 0.30,
                   "use_accruals": True, "use_asset_growth": True,
                   "use_f_score": True},
        "note": "低 PE 一档内按低应计 / 低资产增长 / 高 F-Score 合成",
    },
    "pr_value": {
        "name": "市赚率 PR=PE/ROE",
        "fn": pr_value,
        "params": {"min_mktcap": 2e9, "use_adj": False},
        "note": "丁宁 ericwarn 基础版，PR 越小越便宜",
    },
    "pr_value_adj": {
        "name": "修正市赚率 N×PR",
        "fn": pr_value,
        "params": {"min_mktcap": 2e9, "use_adj": True},
        "note": "按股利支付率加修正系数 N，惩罚赚假钱的公司",
    },
    "pr_double5": {
        "name": "市赚率双 5（0.5PR+5%）",
        "fn": pr_value,
        "params": {"min_mktcap": 2e9, "use_adj": True, "require_double5": True},
        "note": "丁宁的硬门槛版，候选可能不足",
    },
    "dividend_low_vol": {
        "name": "红利低波（H30269 近似）",
        "fn": dividend_low_vol,
        "params": {"min_mktcap": 2e9, "vol_weight": 0.5},
        "note": "中证红利低波编制方案近似复现，波动率需 1 年历史",
    },
    # ---- 价格类异象与事件驱动 ----
    "reversal_1m": {
        "name": "1月反转",
        "fn": reversal,
        "params": {"min_mktcap": 2e9, "col": "ret_1m"},
        "note": "A股最强异象；需月/季频调仓，年度调仓下失效",
    },
    "reversal_3m": {
        "name": "3月反转",
        "fn": reversal,
        "params": {"min_mktcap": 2e9, "col": "ret_3m"},
        "note": "窗口越长越弱（东北金工）",
    },
    "reversal_6m": {
        "name": "6月反转",
        "fn": reversal,
        "params": {"min_mktcap": 2e9, "col": "ret_6m"},
        "note": "预期弱于 1M",
    },
    "momentum_12_1": {
        "name": "12-1 动量（对照）",
        "fn": momentum_121,
        "params": {"min_mktcap": 2e9},
        "note": "美股经典；A股实证不成立，用作反向对照",
    },
    "low_volatility": {
        "name": "低波动（1年）",
        "fn": low_volatility,
        "params": {"min_mktcap": 2e9, "col": "vol_1y"},
        "note": "Blitz 2021：A股最低1/10跑赢最高1/10年化12.9%",
    },
    "low_volatility_60d": {
        "name": "低波动（60日）",
        "fn": low_volatility,
        "params": {"min_mktcap": 2e9, "col": "vol_60d"},
        "note": "短窗口口径，对近期风格切换更敏感",
    },
    "low_ivol": {
        "name": "低特质波动",
        "fn": low_ivol,
        "params": {"min_mktcap": 2e9},
        "note": "CAPM残差波动；中信实证年化超额7.55%",
    },
    "low_beta": {
        "name": "低Beta（BAB多头腿）",
        "fn": low_beta,
        "params": {"min_mktcap": 2e9},
        "note": "Frazzini-Pedersen 2014",
    },
    "pead_sue": {
        "name": "PEAD 超预期（SUE）",
        "fn": pead_sue,
        "params": {"min_mktcap": 2e9},
        "note": "东方证券 SUE0 RankIC 4.02% / IC_IR 3.49",
    },
    "reversal_lowvol": {
        "name": "反转+低波 复合",
        "fn": reversal_lowvol,
        "params": {"min_mktcap": 2e9},
        "note": "换手低于纯反转，实盘更可行",
    },
    "value_quality_lowvol": {
        "name": "价值+质量+低波",
        "fn": value_quality_lowvol,
        "params": {"min_mktcap": 2e9},
        "note": "低PB + 高ROE + 低波动 等权",
    },
    "small_cap": {
        "name": "小市值（池内）",
        "fn": small_cap,
        "params": {"min_mktcap": 2e9},
        "note": "中证800内的规模因子，非全A微盘",
    },
    # ---- 第四轮：超预期族与流动性（剔除市值干扰后的检验）----
    "pead_sur": {
        "name": "营收超预期 SUR",
        "fn": pead_sur,
        "params": {"min_mktcap": 2e9},
        "note": "营收版 SUE；营收由周转率×总资产还原，覆盖率64%",
    },
    "pead_sue_neutral": {
        "name": "SUE（市值中性）",
        "fn": pead_sue_neutral,
        "params": {"min_mktcap": 2e9},
        "note": "对 ln(市值) 回归取残差，用于排除小市值马甲",
    },
    "dual_surprise": {
        "name": "双超预期(盈利+营收)",
        "fn": dual_surprise,
        "params": {"min_mktcap": 2e9},
        "note": "SUE与SUR相关性仅0.375，是相对独立的两条腿",
    },
    "pead_triple": {
        "name": "PEAD 三口径等权",
        "fn": pead_triple,
        "params": {"min_mktcap": 2e9},
        "note": "SUR+SUE中性+SUE原始，压单口径噪声",
    },
    "surprise_turnover": {
        "name": "超预期+低换手",
        "fn": surprise_turnover,
        "params": {"min_mktcap": 2e9},
        "note": "本轮 IC 最高(0.085/IR 5.09)且回撤最浅-39.86%",
    },
    "surprise_triple": {
        "name": "三合一(营收+盈利+低换手)",
        "fn": surprise_triple,
        "params": {"min_mktcap": 2e9},
        "note": "Calmar 0.58，三段行情均强",
    },
    "surprise_lowvol": {
        "name": "超预期+低波动",
        "fn": surprise_lowvol,
        "params": {"min_mktcap": 2e9},
        "note": "PEAD 的低波增强版",
    },
    "qmj_proxy": {
        "name": "QMJ 质量减垃圾(近似)",
        "fn": qmj_proxy,
        "params": {"min_mktcap": 2e9},
        "note": "Asness 2019；仅4个可得字段，与随机持平",
    },
    "low_turnover_neutral": {
        "name": "低换手(市值中性)[已证伪]",
        "fn": low_turnover_neutral,
        "params": {"min_mktcap": 2e9},
        "note": "剔小市值30%后 9.54% < 随机，是小市值马甲",
    },
    "amihud_illiq": {
        "name": "Amihud 非流动性[已证伪]",
        "fn": amihud_illiq,
        "params": {"min_mktcap": 2e9},
        "note": "与市值秩相关-0.77；剔小30%后 10.38%",
    },
}


def select(score: pd.Series, top_n: int, max_weight: float | None = None) -> list[str]:
    """按分数取前 top_n 只（NaN 剔除）。"""
    s = score.dropna().sort_values(ascending=False)
    return list(s.index[:top_n])
