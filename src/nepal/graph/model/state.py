import pandas as pd
from neo4j import Query

from nepal.datasets import Dataset, GovernmentResponse

from .base import Connection, Mergeable


class State(Mergeable):
    def __init__(self, dataset: GovernmentResponse):
        self._dataset: Dataset = dataset

    def merge(self, connection: Connection) -> None:
        self.create_constraint(connection)
        self.insert_nodes(connection)

    @classmethod
    def create_constraint(cls, connection: Connection) -> None:
        connection.query(
            """CREATE CONSTRAINT states IF NOT EXISTS ON (s:State) ASSERT s.code IS UNIQUE"""
        )

    def prepare_data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._dataset.load()
        data_us: pd.DataFrame = data[data["CountryName"] == "United States"]
        return data_us[["RegionName", "RegionCode"]].dropna()

    def insert_nodes(self, connection: Connection) -> None:
        query: Query = Query(
            """
            UNWIND $rows AS row
            MERGE (s:State {code: row.RegionCode})
            ON CREATE SET s.name = row.RegionName
            """
        )

        data: pd.DataFrame = self.prepare_data()
        return connection.insert_data(query, description="State nodes", rows=data)


class StateMeasures(Mergeable):
    def __init__(self, dataset: GovernmentResponse):
        self._dataset: Dataset = dataset

    def merge(self, connection: Connection) -> None:
        self.create_constraint(connection)

    @classmethod
    def create_constraint(cls, connection: Connection) -> None:
        connection.query(
            """CREATE CONSTRAINT measures IF NOT EXISTS ON (m:Measures) ASSERT m.id IS UNIQUE"""
        )

    def insert_nodes(self, connection: Connection) -> None:
        query: Query = Query(
            """
            UNWIND $rows AS row
            MERGE (m:Measures {id: coalesce(row.RegionCode, "") + ',' + coalesce(date(row.Date), "")})
            ON CREATE SET 
                m.stringency = row.StringencyIndex,
                m.government_response = row.GovernmentResponseIndex,
                m.containment_health = row.ContainmentHealthIndex,
                m.economic_support = row.economicSupportIndex
            
            WITH m, row
            MATCH (d:Date {id: date(row.Date)})
            MATCH (s:State {code: row.RegionCode})
            MERGE (s)<-[:IN_STATE]-(m)-[:ACTIVE_ON]->(d)
        """
        )

        rows: pd.DataFrame = self.prepare_data()
        connection.insert_data(
            query, description="Government measures", rows=rows, batch_size=5000
        )

    def prepare_data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._dataset.load()
        data_us: pd.DataFrame = data[data["CountryName"] == "United States"]
        return data_us[
            [
                "RegionCode",
                "Date",
                "StringencyIndex",
                "GovernmentResponseIndex",
                "ContainmentHealthIndex",
                "EconomicSupportIndex",
            ]
        ].dropna(subset=["RegionCode", "Date"])
