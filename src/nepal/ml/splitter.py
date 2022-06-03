from __future__ import annotations

from typing import Iterator, Tuple, Union, cast

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import (
    CutoffSplitter,
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)

BaseSplitter = Union[
    CutoffSplitter,
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
]


class Splitter:
    def __init__(self, splitter: BaseSplitter) -> None:
        self._splitter: BaseSplitter = splitter

    @property
    def fh(self) -> ForecastingHorizon:
        return self._splitter.fh

    def train_test_splits(self, y: pd.DataFrame) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        return self.generate_window_splits(y=y)

    def generate_window_splits(
        self, y: pd.DataFrame
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        dates: pd.DataFrame = self._datetimeindex(y).to_frame()

        for train, test in self._splitter.split(dates):
            idx_train: pd.Series = dates.iloc[train, 0]
            idx_test: pd.Series = dates.iloc[test, 0]

            yield y.loc[pd.IndexSlice[:, idx_train], :], y.loc[pd.IndexSlice[:, idx_test], :]

    def get_n_splits(self, y: pd.DataFrame) -> int:
        dates: pd.DataFrame = self._datetimeindex(y).to_frame()
        return cast(int, self._splitter.get_n_splits(dates))

    @classmethod
    def _datetimeindex(cls, y: pd.DataFrame) -> pd.DatetimeIndex:
        return y.index.get_level_values(-1).unique()
