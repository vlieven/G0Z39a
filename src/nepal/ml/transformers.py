from __future__ import annotations

from typing import Collection, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RollingWindow(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, target: str, source: str, window: int) -> None:
        self._target: str = target
        self._source: str = source
        self._window: int = window

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> RollingWindow:
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        levels: Collection[str] = X.index.names[0:-1]
        target: pd.Series = (
            X[[self._source]]
            .groupby(level=levels, as_index=False)
            .rolling(10)
            .sum()[self._source]
        )
        return X.assign(**{self._target: target})


class AppendActiveInfections(RollingWindow):
    def __init__(self) -> None:
        super().__init__(target="infections", source="new_cases", window=10)
