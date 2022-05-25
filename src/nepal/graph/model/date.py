import pandas as pd
from neo4j import Query

from nepal.datasets import Dataset, NYTimes

from .base import Connection, Mergeable


class Date(Mergeable):
    def __init__(self, dataset: NYTimes):
        self._dataset: Dataset = dataset

    def merge(self, connection: Connection) -> None:
        self.create_constraint(connection)
        self.insert_nodes(connection)
        self.connect_nodes(connection)

    @classmethod
    def create_constraint(cls, connection: Connection) -> None:
        connection.query(
            """CREATE CONSTRAINT dates IF NOT EXISTS ON (d:Date) ASSERT d.id IS UNIQUE"""
        )

    def prepare_data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._dataset.load()

        return data[["date"]].drop_duplicates()

    def insert_nodes(self, connection: Connection) -> None:
        query: Query = Query(
            """
            UNWIND $rows AS row
            MERGE (d:Date {id: date(row.date)})
            """
        )

        data: pd.DataFrame = self.prepare_data()
        return connection.insert_data(query, description="Date nodes", rows=data)

    @classmethod
    def connect_nodes(cls, connection: Connection) -> None:
        query: Query = Query(
            """
            MATCH (d:Date)
            MATCH (e:Date {id: d.id - duration({days: 1})})
            MERGE (d)-[:IS_AFTER]->(e)
            """
        )

        connection.query(query)
