from io import BytesIO
from pathlib import Path
from typing import Collection, Final, Mapping, Set, TypeVar
from zipfile import ZipFile

import pandas as pd
import requests
from requests import Response

from .base import Dataset
from .config import DATASETS_ROOT_DIR
from .util import progressbar

T = TypeVar("T")


class CountyDistance(Dataset):
    """Class which represents the country distance datasets provided by the NBER.
    Source: https://www.nber.org/research/data/county-distance-database
    """

    valid_census_year: Final[Set[int]] = {1990, 2000, 2010}
    valid_radius: Final[Set[int]] = {25, 50, 100, 500, -1}

    destination: Final[Path] = DATASETS_ROOT_DIR / "raw" / "countydistance"

    def __init__(self, *, census_year: int = 2010, radius: int = -1) -> None:
        self._census_year: int = self.__validate_input(
            census_year, name="census_year", accepted=self.valid_census_year
        )
        self._radius: int = self.__validate_input(
            radius, name="radius", accepted=self.valid_radius
        )

    @classmethod
    def __validate_input(cls, value: T, *, name: str, accepted: Collection[T]) -> T:
        if value not in accepted:
            raise ValueError(
                f"Illegal value for '{name}', got {value}, but expected a value in {accepted}."
            )
        else:
            return value

    @property
    def census_year(self) -> int:
        return self._census_year

    @property
    def radius(self) -> str:
        return str(self._radius) if self._radius > 0 else ""

    @property
    def url(self) -> str:
        root: str = "https://nber.org/distance"
        return f"{root}/{self.census_year}/{self._provider}/county/{self.filename}.zip"

    @property
    def filename(self) -> str:
        return f"{self._provider}{self.census_year}countydistance{self.radius}miles.csv"

    def _filepath(self) -> Path:
        return self.destination / self.filename

    @property
    def _provider(self) -> str:
        if self.census_year < 2000:
            return "gaz"
        else:
            return "sf1"

    def collected(self) -> bool:
        return self._filepath().is_file()

    def _collect_data(self) -> None:
        with requests.get(self.url, stream=True) as response:
            distance: str = self.radius or "\N{Infinity}"
            self._extract_and_store_response(
                response,
                folder=self.destination,
                description=f"County distance dataset {self.census_year} - {distance} miles",
            )

    @classmethod
    def _extract_and_store_response(
        cls, response: Response, *, folder: Path, description: str
    ) -> None:
        response.raise_for_status()
        folder.mkdir(parents=True, exist_ok=True)

        with BytesIO() as buffer, progressbar.download(response, description) as progress:
            chunk_size: int = 1024
            for chunk in response.iter_content(chunk_size=chunk_size):
                buffer.write(chunk)
                progress.update(chunk_size)

            file = ZipFile(buffer)
            file.extractall(path=folder)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath(), dtype=self._schema())

    @classmethod
    def _schema(cls) -> Mapping[str, str]:
        return {
            "county1": "string",
            "county2": "string",
            "mi_to_county": "Float32",
        }
