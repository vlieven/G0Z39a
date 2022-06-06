from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from nepal.datasets import Dataset, GovernmentResponse, NYTimes, PopulationDensity, Vaccinations


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


class GovernmentMeasures(Preprocessor):
    def __init__(self, *, response: GovernmentResponse, vaccinations: Vaccinations):
        self._response: Dataset = response
        self._vaccinations: Dataset = vaccinations

    def preprocessed(self) -> pd.DataFrame:
        df_gov: pd.DataFrame = self._government_response()
        df_vacc: pd.DataFrame = self._processed_vaccinations()

        df_joined: pd.DataFrame = df_gov.join(df_vacc, on=["RegionCode", "Date"])
        return self._postprocess_joined(df_joined)

    def _government_response(self) -> pd.DataFrame:
        df: pd.DataFrame = self._response.load()
        df_usa: pd.DataFrame = (
            df[(df["CountryCode"] == "USA") & (df["RegionCode"].notna())]
            .set_index(["RegionCode", "Date"])
            .sort_index()
        )
        return df_usa[
            [
                "StringencyIndex",
                "GovernmentResponseIndex",
                "ContainmentHealthIndex",
                "EconomicSupportIndex",
            ]
        ]

    def _processed_vaccinations(self) -> pd.DataFrame:
        return (
            self._vaccinations.load()
            .pipe(self._derived_vacc_columns)
            .pipe(self._subset_vacc_columns)
        )

    @classmethod
    def _derived_vacc_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["RegionCode"] = ("US_" + df["Recip_State"]).astype("string")
        df["Under5_Pop_Pct"] = (
            (df["Census2019"] - df["Census2019_5PlusPop"]) / df["Census2019"]
        ).astype("float64")
        df["Between5to17_Pop_Pct"] = (df["Census2019_5to17Pop"] / df["Census2019"]).astype(
            "float64"
        )
        df["Between18to65_Pop_Pct"] = (
            (df["Census2019_18PlusPop"] - df["Census2019_65PlusPop"]) / df["Census2019"]
        ).astype("float64")
        df["Plus65_Pop_Pct"] = (df["Census2019_65PlusPop"] / df["Census2019"]).astype("float64")
        df["Is_Metro"] = np.where(df["Metro_status"] == "Metro", 1, 0)
        df["SVI_A"] = np.where(df["SVI_CTGY"] == "A", 1, 0)
        df["SVI_B"] = np.where(df["SVI_CTGY"] == "B", 1, 0)
        df["SVI_C"] = np.where(df["SVI_CTGY"] == "C", 1, 0)
        df["SVI_D"] = np.where(df["SVI_CTGY"] == "D", 1, 0)
        return df

    @classmethod
    def _subset_vacc_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[
                [
                    "Date",
                    "RegionCode",
                    "FIPS",
                    "Completeness_pct",
                    "Administered_Dose1_Pop_Pct",
                    "Administered_Dose1_Recip_18PlusPop_Pct",
                    "Administered_Dose1_Recip_65PlusPop_Pct",
                    "Series_Complete_Pop_Pct",
                    "Series_Complete_18PlusPop_Pct",
                    "Series_Complete_65PlusPop_Pct",
                    "Booster_Doses_Vax_Pct",
                    "Booster_Doses_18Plus_Vax_Pct",
                    "Booster_Doses_50Plus_Vax_Pct",
                    "Booster_Doses_65Plus_Vax_Pct",
                    "Under5_Pop_Pct",
                    "Between5to17_Pop_Pct",
                    "Between18to65_Pop_Pct",
                    "Plus65_Pop_Pct",
                    "Is_Metro",
                    "SVI_A",
                    "SVI_B",
                    "SVI_C",
                    "SVI_D",
                ]
            ]
            .set_index(["RegionCode", "Date"])
            .sort_index()
        )

    @classmethod
    def _postprocess_joined(cls, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[df["FIPS"].notna()]
            .reset_index()
            .drop(columns="RegionCode")
            .rename(columns={"FIPS": "fips", "Date": "date"})
            .set_index(["fips", "date"])
            .sort_index()
        )
