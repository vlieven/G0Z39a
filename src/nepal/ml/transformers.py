from __future__ import annotations

from typing import Collection, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

__all__ = ["log_transformer", "RollingWindowSum"]

log_transformer: FunctionTransformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,
)


class RollingWindowSum(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(
        self,
        column: str,
        *,
        target: str,
        window: int,
        transformer: Optional[FunctionTransformer] = None,
    ) -> None:
        self._column: str = column
        self._target: str = target
        self._window: int = window
        self._transformer: FunctionTransformer = transformer or FunctionTransformer()

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> RollingWindowSum:
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        levels: Collection[str] = X.index.names[0:-1]
        target: pd.Series = (
            self._transformer.inverse_transform(X[self._column])
            .groupby(level=levels)
            .apply(
                lambda x: x.rolling(self._window, min_periods=1, closed="left").sum(
                    engine="numba"
                )
            )
        )
        return X.assign(**{self._target: self._transformer.transform(target)})
