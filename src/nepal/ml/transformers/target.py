from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.transformations.series.detrend import Deseasonalizer

from nepal.datasets import Dataset

from .functions import LogScaler


class TargetTransform:
    __REPOSITORY: Final[Path] = Dataset.ROOT_DIR / "models"

    def __init__(self, refresh: bool = False):
        if refresh:
            self._pipeline: Pipeline = self.__new_pipeline()
        else:
            self._pipeline = self._load_pipeline()

    def fit(self, X: pd.DataFrame) -> TargetTransform:
        self._pipeline = cast(Pipeline, self._pipeline.fit(X))
        self._save_pipeline()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._pipeline.inverse_transform(X)

    @classmethod
    def __new_pipeline(cls) -> Pipeline:
        return Pipeline(
            steps=[("log_scale", LogScaler("new_cases")), ("deseasonalize", Deseasonalizer())]
        )

    @property
    def _filepath(self) -> Path:
        return self.__REPOSITORY / "target_transform.joblib"

    def _load_pipeline(self) -> Pipeline:
        try:
            return joblib.load(self._filepath)
        except FileNotFoundError:
            return self.__new_pipeline()

    def _save_pipeline(self) -> None:
        if not self._pipeline.__sklearn_is_fitted__():
            raise RuntimeError("An unfitted pipeline should not be stored.")
        else:
            self.__REPOSITORY.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pipeline, self._filepath)
