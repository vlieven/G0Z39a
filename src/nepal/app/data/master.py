from functools import lru_cache
from typing import Final, Union

import pandas as pd

from nepal.datasets import GovernmentResponse, NYTimes, PopulationDensity, Vaccinations
from nepal.ml.features.embedding import PersistableEmbedding
from nepal.ml.features.preprocess import Cases, GovernmentMeasures, Population
from nepal.ml.transformers import TargetTransform


class MasterData:
    target_transform: Final[TargetTransform] = TargetTransform()
    measures_lag: Final[int] = 12

    def __init__(self, target_transform: TargetTransform, target_window: Union[pd.Timedelta, int] = 30):
        if not isinstance(target_window, pd.Timedelta):
            target_window = pd.Timedelta(days=target_window)
        self._target_window: pd.Timedelta = target_window
        self._target_transform: TargetTransform = target_transform

    @property
    def target_window(self) -> pd.Timedelta:
        return self._target_window

    @lru_cache(maxsize=None)
    def target(self) -> pd.DataFrame:
        df: pd.DataFrame = Cases(NYTimes()).preprocessed().pipe(self._filter_timeseries)
        y: pd.DataFrame = self.target_transform.transform(df[["new_cases"]])
        return y

    @lru_cache(maxsize=None)
    def exogenous(self) -> pd.DataFrame:
        extra_pop: pd.DataFrame = Population(PopulationDensity()).preprocessed()

        extra_measures: pd.DataFrame = GovernmentMeasures(
            response=GovernmentResponse(), vaccinations=Vaccinations()
        ).preprocessed()

        embedding: pd.DataFrame = PersistableEmbedding("counties").load()

        lagged_measures = extra_measures.groupby(level="fips").shift(self.measures_lag)

        y: pd.DataFrame = self.target()
        Xs = (
            y.join(lagged_measures)
            .join(extra_pop)
            .join(embedding)
            .drop(columns=y.columns)
            .groupby(level="fips")
            .apply(lambda x: x.fillna(method="ffill").fillna(method="bfill"))
        )

        return Xs

    def _filter_timeseries(self, ts: pd.DataFrame) -> pd.DataFrame:
        end: pd.Timestamp = ts.index.get_level_values(level=-1).max()
        start: pd.Timestamp = end - self.target_window
        return ts.loc[pd.IndexSlice[:, start:end], :]
