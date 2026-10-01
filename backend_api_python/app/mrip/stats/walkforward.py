"""Walk-forward validation: fit on a training window, score on the next unseen window.

Anything chosen from data (e.g. the best lag) must be chosen inside ``fit`` from
the training slice only, so the test slice is a genuine out-of-sample check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from app.mrip.stats.correlation import lagged_pair, lead_lag, pearson

P = TypeVar("P")


@dataclass(frozen=True, slots=True)
class Fold:
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_stat: float
    test_stat: float
    n_train: int
    n_test: int


def walk_forward(
    x: pd.Series,
    y: pd.Series,
    fit: Callable[[pd.Series, pd.Series], P],
    train_stat: Callable[[P], float],
    score: Callable[[P, pd.Series, pd.Series], float],
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[Fold]:
    """Rolling windows: train [i, i+train), test [i+train, i+train+test), advancing by ``step``."""
    if train_size < 10 or test_size < 5:
        raise ValueError("train_size must be >= 10 and test_size >= 5")
    stride = step or test_size
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    folds: list[Fold] = []
    start = 0
    while start + train_size + test_size <= len(pair):
        train = pair.iloc[start : start + train_size]
        test = pair.iloc[start + train_size : start + train_size + test_size]
        params = fit(train["x"], train["y"])
        folds.append(
            Fold(
                train_start=train.index[0],
                test_start=test.index[0],
                test_end=test.index[-1],
                train_stat=float(train_stat(params)),
                test_stat=float(score(params, test["x"], test["y"])),
                n_train=len(train),
                n_test=len(test),
            )
        )
        start += stride
    return folds


def lead_lag_walk_forward(
    x: pd.Series, y: pd.Series, max_lag: int, *, train_size: int, test_size: int, step: int | None = None
) -> list[Fold]:
    """Choose the best lag on each training window, then measure the correlation at that lag on the test window."""

    def fit(xt: pd.Series, yt: pd.Series) -> tuple[int, float]:
        res = lead_lag(xt, yt, max_lag)
        return res.best_lag, res.best.r

    def score(params: tuple[int, float], xs: pd.Series, ys: pd.Series) -> float:
        lag = params[0]
        # Use test data only; lagged pairs at the very start of the window lose `lag` rows.
        return pearson(*lagged_pair(xs, ys, lag)).r

    return walk_forward(
        x, y, fit, lambda p: p[1], score, train_size=train_size, test_size=test_size, step=step
    )


def sign_consistency(folds: list[Fold]) -> float | None:
    """Share of folds whose test statistic has the same non-zero sign as the training statistic."""
    usable = [f for f in folds if np.isfinite(f.train_stat) and np.isfinite(f.test_stat)]
    if not usable:
        return None
    hits = sum(1 for f in usable if f.test_stat != 0 and np.sign(f.test_stat) == np.sign(f.train_stat))
    return hits / len(usable)
