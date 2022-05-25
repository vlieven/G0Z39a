import pandas as pd

from nepal.graph.connection import Neo4jConnection


def test_pairwise() -> None:
    df = pd.DataFrame(range(10))
    result = Neo4jConnection.chunks(df, 4)

    assert list(result) == [(0, 4), (4, 8), (8, 10)]
