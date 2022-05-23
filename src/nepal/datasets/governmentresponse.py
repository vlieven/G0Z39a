import logging
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from .base import Dataset
from .config import DATASETS_ROOT_DIR


class GovernmentResponse(Dataset):
    """Class that represents the Government response dataset provided by the University of Oxford.
    More info: https://www.bsg.ox.ac.uk/research/research-projects/covid-19-government-response-tracker
    """

    url: Final[str] = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker"
    destination: Final[Path] = DATASETS_ROOT_DIR / "raw" / "oxcgrt"

    def collected(self) -> bool:
        return self._filepath.is_file()

    @property
    def _filename(self) -> str:
        return "OxCGRT_latest_combined.csv"

    @property
    def _filepath(self) -> Path:
        return self.destination / self._filename

    def _collect_data(self) -> None:
        file: str = self._filename

        logging.info(f"Downloading '{file}'")
        with requests.get(f"{self.url}/master/data/{file}", stream=True) as response:
            self._store_response(
                response,
                folder=self.destination,
                file=file,
                description=f"Covid-19 Government Response Tracker",
            )

    def _load_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath)
