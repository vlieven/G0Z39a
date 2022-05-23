import pandas as pd
from neo4j import Query

from nepal.datasets import Dataset, NYTimes

from ..connection import Neo4jConnection as Connection


class NodeType:
    pass


class Relation:
    pass


class Date(NodeType):
    def __init__(self, dataset: NYTimes):
        self._dataset: Dataset = dataset

    @classmethod
    def create_constraint(cls, connection: Connection) -> None:
        connection.query(
            """CREATE CONSTRAINT dates IF NOT EXISTS ON (d:Date) ASSERT d.id IS UNIQUE"""
        )

    def prepare(self) -> pd.DataFrame:
        data: pd.DataFrame = self._dataset.load()

        return data[["date"]].drop_duplicates()

    def insert_nodes(self, connection: Connection) -> None:
        query: Query = Query(
            """
            UNWIND $rows AS row
            MERGE (c:Date {category: row.category})
            RETURN count(*) as total
            """
        )

        data: pd.DataFrame = self.prepare()
        return connection.insert_data(query, rows=data)


class County(NodeType):
    pass
