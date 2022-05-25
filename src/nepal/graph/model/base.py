from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd

from ..connection import Neo4jConnection as Connection


class Mergeable(ABC):
    @abstractmethod
    def merge(self, connection: Connection) -> None:
        raise NotImplementedError

    def prepare_data(self) -> pd.DataFrame:
        raise NotImplementedError


class Steps:
    def __init__(self, *steps: Mergeable):
        self._steps: Sequence[Mergeable] = steps

    def merge_all(self, connection: Connection) -> None:
        for step in self._steps:
            step.merge(connection)


__all__ = ["Connection", "Mergeable", "Steps"]
