import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from requests import Response

from .util import progressbar


class Dataset(ABC):
    """Base class to represent datasets."""

    @abstractmethod
    def collected(self) -> bool:
        raise NotImplementedError

    def collect(self, refresh: bool = False) -> None:
        if refresh or not self.collected():
            self._collect_data()
        else:
            logging.info("Skipping data collection: already collected")

    @abstractmethod
    def _collect_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _store_response(
        cls, response: Response, *, folder: Path, file: str, description: str
    ) -> None:
        response.raise_for_status()
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / file, mode="wb") as handle, progressbar.download(
            response, description
        ) as progress:
            chunk_size: int = 1024
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
                progress.update(chunk_size)
