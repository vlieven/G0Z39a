from functools import lru_cache
from typing import cast

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.forecasting.base import ForecastingHorizon

from nepal.datasets import Dataset
from nepal.ml.forecaster import LGBMForecaster
from nepal.ml.transformers import LogScaler, RollingWindowSum


class Predictions:
    def __init__(self) -> None:
        self._model: LGBMForecaster = self._deserialize_model()
        self._fh: ForecastingHorizon = ForecastingHorizon(list(range(1, 15)))

    @classmethod
    def _deserialize_model(cls) -> LGBMForecaster:
        return cast(
            LGBMForecaster, joblib.load(Dataset.ROOT_DIR / "reduced" / "forecaster.joblib")
        )

    def load(self, endogenous: pd.DataFrame, exogenous: pd.DataFrame) -> pd.DataFrame:
        forecast: pd.DataFrame = self._model.forecast(fh=self._fh, y=endogenous, Xs=[exogenous])
        return (
            pd.concat([endogenous, forecast])
            .pipe(self._scale_output)
            .pipe(self._infection_count)
            .loc[forecast.index]
            .pipe(self._infections_per_10000)
            .pipe(pd.DataFrame.round)
            .pipe(self._string_index)
        )

    @classmethod
    def _scale_output(cls, df: pd.DataFrame) -> pd.DataFrame:
        preprocess = Pipeline(steps=[("log_scale", LogScaler("new_cases"))])
        return preprocess.inverse_transform(df).round()

    @classmethod
    def _infection_count(cls, df: pd.DataFrame) -> pd.DataFrame:
        return RollingWindowSum("new_cases", target="infections", window=10).transform(df)

    @classmethod
    def _string_index(cls, df: pd.DataFrame) -> pd.DataFrame:
        level: int = -1
        df.index = df.index.set_levels(df.index.levels[level].astype("string"), level=level)
        return df

    @classmethod
    def _infections_per_10000(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.join(cls._population_count())
        df["infections_per_10000"] = (df["infections"] / df["population"]) * 10000
        return df

    @classmethod
    @lru_cache(maxsize=None)
    def _population_count(cls) -> pd.DataFrame:
        return (
            pd.read_csv(Dataset.ROOT_DIR / "reduced" / "population.csv")
            .rename(columns={"State": "name", "Code": "state", "Pop": "population"})
            .set_index("state")
        )
