from __future__ import annotations

from typing import Collection, Dict, Iterable, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingWindowSum(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, column: str, *, target: str, window: int) -> None:
        self._column: str = column
        self._target: str = target
        self._window: int = window

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> RollingWindowSum:
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        levels: Collection[str] = X.index.names[0:-1]
        target: pd.Series = (
            X[[self._column]]
            .groupby(level=levels, as_index=False)
            .rolling(self._window, closed="left")
            .sum()[self._column]
        )
        return X.assign(**{self._target: target})


class SlidingWindowLag(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    # TODO: Using this one breaks everything, fix
    def __init__(
        self, column: str, *, window: int, forecasting: bool, dropna: bool = True
    ) -> None:
        self._column: str = column
        self._window: int = window
        self._forecasting: bool = forecasting
        self._dropna: bool = dropna

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> SlidingWindowLag:
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Builds a new DataFrame to facilitate regressing
        over all possible lagged features.
        """
        if self._forecasting:
            res: pd.DataFrame = self.__lagged_forecasting_features(X)
        else:
            res = self.__lagged_training_features(X)

        if self._dropna:
            return res.dropna()
        else:
            return res

    def __lagged_training_features(self, X: pd.DataFrame) -> pd.DataFrame:
        groupby_levels: Iterable[int] = range(0, X.index.nlevels - 1)
        data: Dict[str, pd.Series] = {}

        # create lagged Series
        for lag_ in range(0, self._window + 1):
            name: str = f"{self._column}_{lag_}" if lag_ else self._column
            data[name] = X.groupby(level=groupby_levels)[self._column].shift(lag_)

        return pd.DataFrame(data, index=X.index)

    def __lagged_forecasting_features(self, X: pd.DataFrame) -> pd.DataFrame:
        groupby_levels: Iterable[int] = range(0, X.index.nlevels - 1)
        data: Dict[str, pd.Series] = {}

        # create lagged Series
        for lag_ in range(1, self._window + 1):
            name: str = f"{self._column}_{lag_}"
            data[name] = X.groupby(level=groupby_levels)[self._column].shift(lag_ - 1)

        # Note: do not refactor index shifting, shifting directly on X.index will not work
        res: pd.DataFrame = pd.DataFrame(data, index=X.index)
        res.index = res.index.set_levels(res.index.levels[-1].shift(1, freq="D"), level=-1)
        return res
