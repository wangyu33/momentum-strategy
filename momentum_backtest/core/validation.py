"""正式价格面板校验。"""

from __future__ import annotations

import pandas as pd

from .config import CORE_OUTPUT_DIR
from .io import write_dataframe_csv_atomic


SPLIT_LIKE_FACTORS = (2, 3, 4, 5, 10)


def detect_price_discontinuities(
    prices: pd.DataFrame,
    *,
    large_drop_cut: float = -0.35,
    large_jump_cut: float = 0.80,
    split_factor_tolerance: float = 0.10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    returns = prices.pct_change(fill_method=None)

    for code in prices.columns:
        series = pd.to_numeric(prices[code], errors="coerce")
        ret = pd.to_numeric(returns[code], errors="coerce")
        flagged = ret[(ret <= large_drop_cut) | (ret >= large_jump_cut)].dropna()
        if flagged.empty:
            continue

        for dt_idx, value in flagged.items():
            prev_loc = prices.index.get_loc(dt_idx)
            if isinstance(prev_loc, slice) or prev_loc <= 0:
                continue
            prev_price = float(series.iloc[prev_loc - 1])
            curr_price = float(series.iloc[prev_loc])
            if prev_price <= 0 or curr_price <= 0:
                continue

            raw_ratio = prev_price / curr_price
            split_like_factor = None
            for factor in SPLIT_LIKE_FACTORS:
                if abs(raw_ratio - factor) / factor <= split_factor_tolerance:
                    split_like_factor = factor
                    break

            rows.append(
                {
                    "date": pd.Timestamp(dt_idx),
                    "code": str(code),
                    "prev_price": prev_price,
                    "curr_price": curr_price,
                    "daily_return": float(value),
                    "prev_curr_ratio": raw_ratio,
                    "split_like_factor": split_like_factor,
                    "issue_type": "split_like_gap" if split_like_factor is not None else "abnormal_gap",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "code",
                "prev_price",
                "curr_price",
                "daily_return",
                "prev_curr_ratio",
                "split_like_factor",
                "issue_type",
            ]
        )
    return pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)


def persist_price_validation_report(
    prices: pd.DataFrame,
    *,
    output_path=None,
) -> pd.DataFrame:
    issues = detect_price_discontinuities(prices)
    target_path = CORE_OUTPUT_DIR / "price_validation_issues.csv" if output_path is None else output_path
    write_dataframe_csv_atomic(issues, target_path, index=False)
    return issues
