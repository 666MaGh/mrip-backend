"""Engle-Granger cointegration on log price levels (informational evidence)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True, slots=True)
class CointResult:
    n: int
    statistic: float
    p_value: float
    hedge_ratio: float  # slope of log(y) on log(x)


def engle_granger(y_prices: pd.Series, x_prices: pd.Series) -> CointResult:
    pair = pd.concat([y_prices, x_prices], axis=1, join="inner").dropna()
    if (pair <= 0).any().any():
        raise ValueError("prices must be positive")
    if len(pair) < 50:
        raise ValueError("need at least 50 observations for a cointegration test")
    ly, lx = np.log(pair.iloc[:, 0].to_numpy()), np.log(pair.iloc[:, 1].to_numpy())
    statistic, p_value, _ = coint(ly, lx, trend="c", autolag="aic")
    hedge = float(np.polyfit(lx, ly, 1)[0])
    return CointResult(n=len(pair), statistic=float(statistic), p_value=float(p_value), hedge_ratio=hedge)
