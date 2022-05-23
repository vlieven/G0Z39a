import logging
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence, Union

import pandas as pd
import requests

from .base import Dataset
from .config import DATASETS_ROOT_DIR


class NYTimes(Dataset):
    """Class which represents the Covid-19 data published by the New York Times.
    Source: https://github.com/nytimes/covid-19-data
    """

    repository: Final[str] = "https://github.com/nytimes/covid-19-data"
    destination: Final[Path] = DATASETS_ROOT_DIR / "raw" / "nytimes"

    def __init__(self, *, years: Union[int, Iterable[int]] = (2020, 2021, 2022)) -> None:
        self._years: Sequence[int]

        if isinstance(years, int):
            self._years = [years]
        else:
            self._years = list(years)

    @classmethod
    def _filename(cls, year: int) -> str:
        return f"us-counties-{year}.csv"

    @classmethod
    def _filepath(cls, year: int) -> Path:
        return cls.destination / cls._filename(year)

    def collected(self) -> bool:
        return all(self._filepath(year).is_file() for year in self._years)

    def _collect_data(self) -> None:
        for year in self._years:
            file: str = self._filename(year)

            logging.info(f"Downloading '{file}'")
            with requests.get(f"{self.repository}/raw/master/{file}", stream=True) as response:
                self._store_response(
                    response,
                    folder=self.destination,
                    file=file,
                    description=f"US Covid {year} data",
                )

    def _load_dataframe(self) -> pd.DataFrame:
        return pd.concat(
            [
                pd.read_csv(self._filepath(year), dtype=self._schema(), parse_dates=["date"])
                for year in self._years
            ],
            ignore_index=True,
        )

    @classmethod
    def _schema(cls) -> Mapping[str, str]:
        return {
            "county": "string",
            "state": "string",
            "fips": "string",
            "cases": "UInt64",
            "deaths": "UInt64",
        }
