import logging
from pathlib import Path
from typing import Final, Mapping

import pandas as pd
import requests

from .base import Dataset


class GovernmentResponse(Dataset):
    """Class that represents the Government response dataset provided by the University of Oxford.
    More info: https://www.bsg.ox.ac.uk/research/research-projects/covid-19-government-response-tracker
    """

    url: Final[str] = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker"
    destination: Final[Path] = Dataset.ROOT_DIR / "raw" / "oxcgrt"

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
        return pd.read_csv(self._filepath, dtype=self._schema(), parse_dates=["Date"])

    @classmethod
    def _schema(cls) -> Mapping[str, str]:
        return {
            "CountryName": "string",
            "CountryCode": "string",
            "RegionName": "string",
            "RegionCode": "string",
            "Jurisdiction": "string",
            "C1_combined": "string",
            "C2_combined": "string",
            "C3_combined": "string",
            "C4_combined": "string",
            "C5_combined": "string",
            "C6_combined": "string",
            "C7_combined": "string",
            "C8_combined": "string",
            "E1_combined": "string",
            "E2_combined": "string",
            "H1_combined": "string",
            "H2_combined": "string",
            "H3_combined": "string",
            "H6_combined": "string",
            "H7_combined": "string",
            "H8_combined": "string",
            "V1_combined": "string",
            "V2_combined": "string",
            "V3_combined": "string",
            "V4_combined": "string",
        }
