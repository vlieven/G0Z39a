import logging
from pathlib import Path
from typing import Dict, Final, Sequence

import pandas as pd
import requests

from .base import Dataset
from .config import DATASETS_ROOT_DIR


class NYTimes(Dataset):
    """Module which collects the Covid-19 data published by the New York Times.

    To use, you can simply run NYTimes.collect()
    """

    repository: Final[str] = "https://github.com/nytimes/covid-19-data"
    destination: Final[Path] = DATASETS_ROOT_DIR / "raw" / "nytimes"
    years: Sequence[int] = [2020, 2021, 2022]

    @classmethod
    def _filename(cls, year: int) -> str:
        return f"us-counties-{year}.csv"

    @classmethod
    def _filepath(cls, year: int) -> Path:
        return cls.destination / cls._filename(year)

    @classmethod
    def collected(cls) -> bool:
        return all(cls._filepath(year).is_file() for year in cls.years)

    @classmethod
    def _collect_data(cls) -> None:
        for year in cls.years:
            file: str = cls._filename(year)

            logging.info(f"Downloading '{file}'")
            with requests.get(f"{cls.repository}/raw/master/{file}", stream=True) as response:
                cls._store_response(response, folder=cls.destination, file=file, description=f"US Covid {year} data")

    @classmethod
    def load(cls) -> pd.DataFrame:
        return pd.concat(
            [pd.read_csv(cls._filepath(year), dtype=cls._schema(), parse_dates=["date"]) for year in cls.years],
            ignore_index=True,
        )

    @classmethod
    def _schema(cls) -> Dict[str, str]:
        return {
            "county": "string",
            "state": "string",
            "fips": "string",
            "cases": "UInt64",
            "deaths": "UInt64",
        }
