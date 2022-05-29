import numpy as np
import pandas as pd
from neo4j import Query

from nepal.datasets import CountyDistance, Dataset, Vaccinations

from .base import Connection, Mergeable


class County(Mergeable):
    def __init__(self, dataset: Vaccinations):
        self._dataset: Dataset = dataset

    def merge(self, connection: Connection) -> None:
        self.create_constraint(connection)
        self.insert_and_connect_nodes(connection)

    @classmethod
    def create_constraint(cls, connection: Connection) -> None:
        connection.query(
            """CREATE CONSTRAINT counties IF NOT EXISTS ON (c:County) ASSERT c.fips IS UNIQUE"""
        )

    def prepare_data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._dataset.load()
        result: pd.DataFrame = (
            data.pipe(self._keep_relevant_columns)
            .pipe(self._add_derived_columns)
            .pipe(self._select_output_columns)
        )

        return result

    @classmethod
    def _keep_relevant_columns(cls, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data[
                [
                    "FIPS",
                    "Recip_State",
                    "Metro_status",
                    "SVI_CTGY",
                    "Census2019",
                    "Census2019_5PlusPop",
                    "Census2019_5to17Pop",
                    "Census2019_18PlusPop",
                    "Census2019_65PlusPop",
                ]
            ]
            .drop_duplicates()
            .dropna(subset=["FIPS"])
        )

    @classmethod
    def _add_derived_columns(cls, data: pd.DataFrame) -> pd.DataFrame:
        data["RegionCode"] = "US_" + data["Recip_State"].astype(str)
        data["Under5_Pop_Pct"] = (data["Census2019"] - data["Census2019_5PlusPop"]) / data[
            "Census2019"
        ]
        data["Between5to17_Pop_Pct"] = data["Census2019_5to17Pop"] / data["Census2019"]
        data["Between18to65_Pop_Pct"] = (
            data["Census2019_18PlusPop"] - data["Census2019_65PlusPop"]
        ) / data["Census2019"]
        data["Plus65_Pop_Pct"] = data["Census2019_65PlusPop"] / data["Census2019"]
        data["Is_Metro"] = np.where(data["Metro_status"] == "Metro", 1, 0)
        data["SVI_A"] = np.where(data["SVI_CTGY"] == "A", 1, 0)
        data["SVI_B"] = np.where(data["SVI_CTGY"] == "B", 1, 0)
        data["SVI_C"] = np.where(data["SVI_CTGY"] == "C", 1, 0)
        data["SVI_D"] = np.where(data["SVI_CTGY"] == "D", 1, 0)
        return data

    @classmethod
    def _select_output_columns(cls, data: pd.DataFrame) -> pd.DataFrame:
        return data[
            [
                "FIPS",
                "RegionCode",
                "Census2019",
                "Under5_Pop_Pct",
                "Between5to17_Pop_Pct",
                "Between18to65_Pop_Pct",
                "Plus65_Pop_Pct",
                "Is_Metro",
                "SVI_A",
                "SVI_B",
                "SVI_C",
                "SVI_D",
            ]
        ]

    def insert_and_connect_nodes(self, connection: Connection) -> None:
        query: Query = Query(
            """
            UNWIND $rows AS row
            MERGE (c:County {fips: row.FIPS})
            ON CREATE SET 
                c.census = row.Census2019,
                c.pop_under_5 = row.Under5_Pop_Pct,
                c.pop_5_to_17 = row.Between5to17_Pop_Pct,
                c.pop_18_to_65 = row.Between18to65_Pop_Pct,
                c.pop_plus_65 = row.Plus65_Pop_Pct,
                c.is_metro = row.Is_Metro,
                c.svi_a = row.SVI_A,
                c.svi_b = row.SVI_B,
                c.svi_c = row.SVI_C,
                c.svi_d = row.SVI_D
            
            WITH c, row
            MATCH (s:State {code: row.RegionCode})
            MERGE (c)-[:IN_STATE]->(s)
            """
        )

        rows: pd.DataFrame = self.prepare_data()
        connection.insert_data(query, description="County nodes", rows=rows, batch_size=5000)


class CountyDistances(Mergeable):
    def __init__(self, dataset: CountyDistance):
        self._dataset: CountyDistance = dataset

    def merge(self, connection: Connection) -> None:
        query: Query = Query(
            """
            UNWIND $rows as row
            MATCH (a:County {fips: row.county1})
            MATCH (b:County {fips: row.county2})
            MERGE (a)-[:IS_NEAR {weight: row.weight}]-(b)
            """
        )

        rows: pd.DataFrame = self.prepare_data()
        connection.insert_data(query, description="County distances", rows=rows)

    def prepare_data(self) -> pd.DataFrame:
        radius: int = int(self._dataset.radius)

        data: pd.DataFrame = self._dataset.load()
        data["weight"] = (radius - data["mi_to_county"]) / radius
        return data[["county1", "county2", "weight"]]


class CountyVaccinations(Mergeable):
    def __init__(self, dataset: Vaccinations):
        self._dataset: Dataset = dataset

    def merge(self, connection: Connection) -> None:
        self.create_constraint(connection)
        self.insert_nodes(connection)

    @classmethod
    def create_constraint(cls, connection: Connection) -> None:
        connection.query(
            """CREATE CONSTRAINT vaccinations IF NOT EXISTS ON (v:Vaccinations) ASSERT v.id IS UNIQUE"""
        )

    def prepare_data(self) -> pd.DataFrame:
        data: pd.DataFrame = self._dataset.load()
        data = data[data["FIPS"] != "UNK"]
        return data[
            [
                "Date",
                "FIPS",
                "Completeness_pct",
                "Administered_Dose1_Pop_Pct",
                "Administered_Dose1_Recip_18PlusPop_Pct",
                "Administered_Dose1_Recip_65PlusPop_Pct",
                "Series_Complete_Pop_Pct",
                "Series_Complete_18PlusPop_Pct",
                "Series_Complete_65PlusPop_Pct",
                "Booster_Doses_Vax_Pct",
                "Booster_Doses_18Plus_Vax_Pct",
                "Booster_Doses_50Plus_Vax_Pct",
                "Booster_Doses_65Plus_Vax_Pct",
            ]
        ]

    def insert_nodes(self, connection: Connection) -> None:
        query: Query = Query(
            """
            UNWIND $rows AS row
            MERGE (v:Vaccinations {id: coalesce(row.FIPS, "") + ',' + coalesce(row.Date, "")})
            ON CREATE SET 
                v.completeness = row.Completeness_pct,
                v.dose1_pop = row.Administered_Dose1_Pop_Pct,
                v.dose1_18plus = row.Administered_Dose1_Recip_18PlusPop_Pct,
                v.dose1_65plus = row.Administered_Dose1_Recip_65PlusPop_Pct,
                v.series_complete = row.Series_Complete_Pop_Pct,
                v.series_complete_18plus = row.Series_Complete_18PlusPop_Pct,
                v.series_complete_65plus = row.Series_Complete_65PlusPop_Pct,
                v.booster_pop = row.Booster_Doses_Vax_Pct,
                v.booster_18plus = row.Booster_Doses_18Plus_Vax_Pct,
                v.booster_50plus = row.Booster_Doses_50Plus_Vax_Pct,
                v.booster_65plus = row.Booster_Doses_65Plus_Vax_Pct
            
            WITH v, row
            MATCH (d:Date {id: date(row.Date)})
            MATCH (c:County {fips: row.FIPS})
            MERGE (c)<-[:IN_COUNTY]-(v)-[:REPORTED_ON]->(d)
            """
        )

        rows: pd.DataFrame = self.prepare_data()
        connection.insert_data(
            query, description="Vaccination nodes", rows=rows, batch_size=5000
        )
