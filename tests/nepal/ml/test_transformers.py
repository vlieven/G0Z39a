import datetime as dt

import numpy as np
import pandas as pd

from nepal.ml.transformers import RollingWindowSum


def test_rolling_window_sum() -> None:
    df = pd.DataFrame(
        {
            "group": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "date": pd.date_range(dt.date(2021, 1, 1), dt.date(2021, 1, 10)),
            "values": [*range(1, 11)],
        }
    ).set_index(["group", "date"])

    result: pd.DataFrame = RollingWindowSum("values", target="roll", window=3).transform(df)
    expected: pd.DataFrame = pd.DataFrame(
        {
            "group": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "date": pd.date_range(dt.date(2021, 1, 1), dt.date(2021, 1, 10)),
            "values": [*range(1, 11)],
            "roll": [np.NaN, 1.0, 3.0, 6.0, 9.0, np.NaN, 6.0, 13.0, 21.0, 24.0],
        }
    ).set_index(["group", "date"])

    pd.testing.assert_frame_equal(result, expected)
