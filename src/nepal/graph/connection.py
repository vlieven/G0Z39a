from __future__ import annotations

import itertools
import warnings
from types import TracebackType
from typing import (
    Any,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from neo4j import GraphDatabase, Neo4jDriver, Query, Record, Result, Session, basic_auth
from tqdm.auto import tqdm

T = TypeVar("T")


class Neo4jConnection:
    def __init__(self, *, uri: str, user: str, pwd: str, db: Optional[str] = None) -> None:
        self.__driver: Neo4jDriver = GraphDatabase.driver(
            uri, auth=basic_auth(user=user, password=pwd)
        )

        self._db: Optional[str] = db

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

    def is_up(self) -> bool:
        connected: bool = True
        try:
            self.query("""Match () Return 1 Limit 1""")
        except Exception as e:
            connected = False
            warnings.warn(f"Error while connecting: {e}")
        return connected

    def query(
        self, query: Union[str, Query], *, parameters: Optional[Mapping[Hashable, Any]] = None
    ) -> Sequence[Record]:
        with self.session() as session:
            response: Result = session.run(query, parameters=parameters)
        return list(response)

    def insert_data(
        self,
        query: Union[str, Query],
        *,
        description: str,
        rows: pd.DataFrame,
        batch_size: int = 10000,
    ) -> None:
        """Function to handle the updating the Neo4j database in batch mode."""
        rows = as_serializable(rows)

        for start, stop in tqdm(self.chunks(rows, size=batch_size), desc=description):
            payload: Sequence[Mapping[str, Any]] = rows[start:stop].to_dict("records")

            self.query(
                query,
                parameters={"rows": drop_missing_values(payload)},
            )

    @classmethod
    def chunks(cls, df: pd.DataFrame, size: int) -> Sequence[Tuple[int, int]]:
        return list(pairwise(inclusive_range(0, len(df), size)))


class LocalConnection(Neo4jConnection):
    def __init__(self, *, database: str, password: str):
        super().__init__(uri="bolt://localhost:7687", user="neo4j", pwd=password, db=database)


def as_serializable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Timestamp values ('datetime64') cannot be processed by the Neo4J connector,
    hence we need to convert them to strings.
    """
    timestamps: Sequence[str] = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
    for ts in timestamps:
        df[ts] = df[ts].astype(str)

    return df


def drop_missing_values(rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    return [{k: v for k, v in row.items() if not pd.isna(v)} for row in rows]


def inclusive_range(start: int, stop: int, step: int) -> Iterator[int]:
    yield from range(start, stop, step)
    yield stop


def pairwise(iterable: Iterable[T]) -> Iterator[Tuple[T, T]]:
    """pairwise('ABCDEFG') --> AB BC CD DE EF FG"""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


__all__ = ["Neo4jConnection", "LocalConnection"]
