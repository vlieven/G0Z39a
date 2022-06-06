from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from ..connection import Neo4jConnection as Connection


class Mergeable(ABC):
    @abstractmethod
    def merge(self, connection: Connection) -> None:
        raise NotImplementedError

    def prepare_data(self) -> pd.DataFrame:
        raise NotImplementedError


class Steps:
    def __init__(self, *steps: Mergeable) -> None:
        self._steps: List[Mergeable] = list(steps)

    def add(self, *steps: Mergeable) -> Steps:
        self._steps.extend(steps)
        return self

    def merge_all(self, connection: Connection) -> None:
        for step in self._steps:
            step.merge(connection)


__all__ = ["Connection", "Mergeable", "Steps"]
