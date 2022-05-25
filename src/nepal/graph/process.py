from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence

from nepal.datasets import CountyDistance, GovernmentResponse, NYTimes, Vaccinations

from .connection import LocalConnection, Neo4jConnection
from .model import (
    County,
    CountyDistances,
    CountyVaccinations,
    Date,
    State,
    StateMeasures,
    Steps,
)


class GraphDB:
    def __init__(self, connection: Neo4jConnection):
        self._connection: Neo4jConnection = connection

    @classmethod
    def local(cls, db: str, pwd: str) -> GraphDB:
        return cls(LocalConnection(database=db, password=pwd))

    def close(self) -> None:
        self._connection.close()

    def populate_database(self) -> None:
        infections: NYTimes = NYTimes()
        measures: GovernmentResponse = GovernmentResponse()
        vaccinations: Vaccinations = Vaccinations()
        distances: CountyDistance = CountyDistance(radius=100)

        Steps(
            Date(infections),
            State(measures),
            StateMeasures(measures),
            County(vaccinations),
            CountyVaccinations(vaccinations),
            CountyDistances(distances),
        ).merge_all(self._connection)

    def wipe_database(self) -> Sequence[Mapping[Hashable, Any]]:
        """Remove all nodes, relationships, indexes and constraints."""
        return self._connection.query(
            """
            MATCH (n) DETACH DELETE n
            CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *
            """
        )
