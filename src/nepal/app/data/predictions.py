from typing import cast

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from nepal.ml.forecaster import LGBMForecaster
from nepal.ml.transformers import TargetTransform


class Predictions:
    def __init__(self, target_transform: TargetTransform) -> None:
        self._model: LGBMForecaster = cast(LGBMForecaster, LGBMForecaster.load("forecast"))
        self._fh: ForecastingHorizon = ForecastingHorizon(list(range(1, 15)))
        self._transform: TargetTransform = target_transform

    def load(self, endogenous: pd.DataFrame, exogenous: pd.DataFrame) -> pd.DataFrame:
        return (
            self._model.forecast(fh=self._fh, y=endogenous, Xs=[exogenous])
            .pipe(self._transform.inverse_transform)
            .pipe(self._string_index)
        )

    @classmethod
    def _string_index(cls, df: pd.DataFrame) -> pd.DataFrame:
        level: int = -1
        df.index = df.index.set_levels(df.index.levels[level].astype("string"), level=level)
        return df
