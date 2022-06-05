import logging
from pathlib import Path
from typing import Final

import pandas
import pandas as pd
import requests

from .base import Dataset


class PopulationDensity(Dataset):
    """Data feeding the dashboard at
    https://covid19.census.gov/datasets/21843f238cbb46b08615fc53e19e0daf_1
    """

    destination: Final[Path] = Dataset.ROOT_DIR / "raw" / "covid19census"
    url: Final[str] = (
        "https://opendata.arcgis.com/api/v3/datasets/21843f238cbb46b08615fc53e19e0daf_1"
        "/downloads/data?format=csv&spatialRefId=4326"
    )

    @property
    def _filename(self) -> str:
        return "Average_Household_Size_and_Population_Density_-_County.csv"

    def _filepath(self) -> Path:
        return self.destination / self._filename

    def collected(self) -> bool:
        return self._filepath().is_file()

    def _collect_data(self) -> None:
        file: str = self._filename

        logging.info(f"Downloading '{file}'")
        with requests.get(self.url, stream=True) as response:
            self._store_response(
                response,
                folder=self.destination,
                file=file,
                description=f"Average Household Size and Population Density - County",
            )

    def _load_dataframe(self) -> pd.DataFrame:
        return pandas.read_csv(
            self._filepath(),
            usecols=["GEOID", "B25010_001E", "B01001_calc_PopDensity"],
            dtype={"GEOID": "string"},
        ).rename(
            columns={
                "GEOID": "fips",
                "B25010_001E": "avg_household_size",
                "B01001_calc_PopDensity": "pop_density",
            }
        )
