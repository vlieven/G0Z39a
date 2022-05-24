from abc import ABC, abstractmethod

import pandas as pd

from ..connection import Neo4jConnection as Connection


class NodeType(ABC):
    @abstractmethod
    def merge(self, connection: Connection) -> None:
        raise NotImplementedError

    def prepare_data(self) -> pd.DataFrame:
        raise NotImplementedError


class RelationshipType(ABC):
    pass


__all__ = ["Connection", "NodeType", "RelationshipType"]
