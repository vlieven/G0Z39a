from __future__ import annotations

import itertools
from types import TracebackType
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

import pandas as pd
from neo4j import GraphDatabase, Neo4jDriver, Query, Record, Result, Session, basic_auth
from tqdm.auto import tqdm

T = TypeVar("T")


class Neo4jConnection:
    def __init__(self, *, uri: str, user: str, pwd: str) -> None:
        self.__driver: Neo4jDriver = GraphDatabase.driver(
            uri, auth=basic_auth(user=user, password=pwd)
        )

        self._db: Optional[str] = None

    @property
    def db(self) -> Optional[str]:
        return self._db

    @db.setter
    def db(self, db: Optional[str]) -> None:
        self._db = db

    def __enter__(self) -> Neo4jConnection:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self.close()
        return None

    def close(self) -> None:
        self.__driver.close()

    def session(self) -> Session:
        return (
            self.__driver.session(database=self.db)
            if self.db is not None
            else self.__driver.session()
        )

    def query(
        self, query: Union[str, Query], *, parameters: Optional[Mapping[Hashable, Any]] = None
    ) -> Sequence[Record]:
        with self.session() as session:
            response: Result = session.run(query, parameters)
        return list(response)

    def insert_data(
        self, query: Union[str, Query], *, rows: pd.DataFrame, batch_size: int = 10000
    ) -> None:
        """Function to handle the updating the Neo4j database in batch mode."""

        for start, stop in tqdm(self.pairwise(range(0, len(rows), batch_size))):
            self.query(
                query,
                parameters={"rows": rows[start:stop].to_dict("records")},
            )

    @classmethod
    def pairwise(cls, iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
        """pairwise('ABCDEFG') --> AB BC CD DE EF FG"""
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
