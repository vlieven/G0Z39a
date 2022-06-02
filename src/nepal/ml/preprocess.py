from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence

import pandas as pd

from nepal.datasets import Dataset, NYTimes, PopulationDensity


class Preprocessor(ABC):
    @abstractmethod
    def preprocessed(self) -> pd.DataFrame:
        raise NotImplementedError


class Cases(Preprocessor):
    def __init__(self, dataset: NYTimes) -> None:
        self._dataset: Dataset = dataset

        self._index: Sequence[str] = ["fips", "date"]
        self._targets: Sequence[str] = ["cases", "deaths"]

    def preprocessed(self) -> pd.DataFrame:
        df_covid: pd.DataFrame = self._dataset.load()

        return (
            df_covid.dropna(subset=self._index)
            .set_index(self._index)
            .pipe(self._fill_index, names=self._index)
            .sort_index(level=self._index)
            .pipe(self._cast_types_as_signed, cols=self._targets)
            .pipe(self._calculate_new, cols=self._targets)
        )

    @classmethod
    def _fill_index(cls, df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
        return df.pipe(cls._complete_index, names=names).pipe(cls._fill_na)

    @classmethod
    def _complete_index(cls, df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
        dates: pd.Index = df.index.get_level_values("date")

        labels: Mapping[str, pd.Index] = {
            "fips": df.index.get_level_values("fips").unique(),
            "date": pd.date_range(start=dates.min(), end=dates.max(), freq="D"),
        }

        complete: pd.MultiIndex = pd.MultiIndex.from_product(
            [labels[names[0]], labels[names[1]]], names=names
        )

        return df.reindex(complete)

    @classmethod
    def _fill_na(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            county=df.county.bfill(),
            state=df.state.bfill(),
            cases=df.cases.fillna(0),
            deaths=df.deaths.fillna(0),
        )

    @classmethod
    def _cast_types_as_signed(cls, df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        return df.astype({col: "int64" for col in cols})

    @classmethod
    def _calculate_new(cls, df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        for col in cols:
            diff: pd.Series = df.groupby(level="fips")[col].diff().fillna(0)
            avg: pd.Series = diff.groupby(level="fips").apply(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f"new_{col}"] = avg.clip(0)
        return df


class Population(Preprocessor):
    def __init__(self, dataset: PopulationDensity):
        self._dataset: Dataset = dataset

    def preprocessed(self) -> pd.DataFrame:
        df_population: pd.DataFrame = self._dataset.load()

        return df_population.set_index("fips")
