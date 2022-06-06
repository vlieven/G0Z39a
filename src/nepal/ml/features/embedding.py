from pathlib import Path
from typing import Final

import pandas as pd

from nepal.datasets import Dataset


class PersistableEmbedding:
    destination: Final[Path] = Dataset.ROOT_DIR / "embedding"

    def __init__(self, identifier: str):
        super().__init__()
        self.identifier: str = identifier

    @property
    def path(self) -> Path:
        return self.destination / f"{self.identifier}.parquet"

    def store(self, embedding: pd.DataFrame) -> None:
        self.destination.mkdir(parents=True, exist_ok=True)

        embedding.sort_index(inplace=True)
        embedding.to_parquet(self.path, engine="pyarrow", index=True)

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path, engine="pyarrow")
