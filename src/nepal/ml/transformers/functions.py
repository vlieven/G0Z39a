from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

log_transformer: FunctionTransformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,
    validate=False,
)


class LogScaler(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, name: str):
        self._name: str = name

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> LogScaler:
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        X.loc[:, self._name] = log_transformer.transform(X[self._name])
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X.loc[:, self._name] = log_transformer.inverse_transform(X[self._name])
        return X
