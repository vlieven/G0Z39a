from typing import Sequence

import lightgbm as lgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import SlidingWindowSplitter

from nepal.datasets import NYTimes, PopulationDensity
from nepal.ml.forecaster import LGBMForecaster
from nepal.ml.loss import mape
from nepal.ml.preprocess import Cases, Population
from nepal.ml.splitter import Splitter
from nepal.ml.transformers import RollingWindowSum, log_transformer
from nepal.ml.validate import cross_validate

fh = ForecastingHorizon(list(range(1, 15)))
cv = SlidingWindowSplitter(fh=fh, window_length=120, step_length=60)
splitter: Splitter = Splitter(cv)

df: pd.DataFrame = Cases(NYTimes()).preprocessed()
extra_pop: pd.DataFrame = Population(PopulationDensity()).preprocessed()

df_y: pd.DataFrame = log_transformer.transform(df[["new_cases"]])

transformers: Pipeline = Pipeline(
    [
        (
            "with_active_infections",
            RollingWindowSum(
                "new_cases", target="infections", window=10, transformer=log_transformer
            ),
        ),
    ]
)

forecaster = LGBMForecaster(
    lgb.LGBMRegressor(objective=mape.name), lag=18, transformers=transformers
)

scores: Sequence[float] = cross_validate(
    forecaster, splitter=splitter, y=df_y, Xs=[extra_pop], loss=mape.function, threads=8
)
