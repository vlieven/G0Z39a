from functools import lru_cache
from typing import Final

import pandas as pd
from sklearn.pipeline import Pipeline

from nepal.datasets import Dataset
from nepal.ml.transformers.functions import LogScaler


class ReducedData:
    measures_lag: Final[int] = 12

    @lru_cache(maxsize=None)
    def target(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_parquet(Dataset.ROOT_DIR / "reduced" / "endogenous.parquet")
        return self._scale_input(df[["new_cases"]])

    @classmethod
    def _scale_input(cls, df: pd.DataFrame) -> pd.DataFrame:
        preprocess = Pipeline(steps=[("log_scale", LogScaler("new_cases"))])
        return preprocess.transform(df)

    @lru_cache(maxsize=None)
    def exogenous(self) -> pd.DataFrame:
        y: pd.DataFrame = self.target()

        df: pd.DataFrame = pd.read_parquet(Dataset.ROOT_DIR / "reduced" / "exogenous.parquet")
        lagged_measures = df.groupby(level="state").shift(self.measures_lag)

        Xs = lagged_measures.groupby(level="state").apply(
            lambda x: x.fillna(method="ffill").fillna(method="bfill")
        )

        return Xs
