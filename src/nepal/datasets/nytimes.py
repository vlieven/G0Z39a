import logging
import os
from pathlib import Path
from typing import Final, Iterable, Union

import pandas as pd
import requests
from requests import Response

from .config import DATASETS_ROOT_DIR
from .base import Dataset


class NYTimes(Dataset):
    """Module which collects the Covid-19 data published by the New York Times.

    To use, you can simply run NYTimes.collect()
    """

    repository: Final[str] = "https://github.com/nytimes/covid-19-data"
    destination: Final[Path] = DATASETS_ROOT_DIR / "raw" / "nytimes"

    @classmethod
    def collect(cls, refresh: Union[int, Iterable[int]] = None) -> None:
        if not refresh:
            refresh = [2020, 2021, 2022]
        if isinstance(refresh, int):
            refresh = [refresh]

        for year in refresh:
            file: str = f"us-counties-{year}.csv"

            logging.info(f"Downloading '{file}'")
            with requests.get(f"{cls.repository}/raw/master/{file}", stream=True) as response:
                cls._store_response(response, file=file)

    @classmethod
    def _store_response(cls, response: Response, *, file: str) -> None:
        response.raise_for_status()

        os.makedirs(cls.destination, exist_ok=True)
        with open(cls.destination / file, mode="wb") as handle:
            for chunk in response.iter_content(chunk_size=None):
                handle.write(chunk)

    @classmethod
    def load(cls) -> pd.DataFrame:
        return pd.concat([
            pd.read_csv(cls.destination / f"us-counties-{year}.csv") for year in [2020, 2021, 2022]
        ])
