import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from requests import Response
from tqdm.auto import tqdm


class Dataset(ABC):
    @classmethod
    @abstractmethod
    def collected(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def collect(cls, refresh: bool = False) -> None:
        if refresh or not cls.collected():
            cls._collect_data()
        else:
            logging.info("Skipping data collection: already collected")

    @classmethod
    @abstractmethod
    def _collect_data(cls) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _store_response(cls, response: Response, *, folder: Path, file: str, description: Optional[str] = None) -> None:
        response.raise_for_status()

        os.makedirs(folder, exist_ok=True)

        file_size: Optional[int] = int(response.headers.get("Content-Length", 0)) or None
        with open(folder / file, mode="wb") as handle:
            for chunk in tqdm(response.iter_content(chunk_size=None), desc=description, total=file_size):
                handle.write(chunk)
