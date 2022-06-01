import warnings
from typing import Any, Iterable, List, Optional, Protocol, Sequence

import pandas as pd
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

from nepal.ml.forecaster import BaseForecaster
from nepal.ml.splitter import Splitter


class LossFunction(Protocol):
    def __call__(self, *, y_true: pd.DataFrame, y_pred: pd.DataFrame, **kwargs: Any) -> float:
        """Calculates loss metric."""


def cross_validate(
    forecaster: BaseForecaster,
    *,
    splitter: Splitter,
    y: pd.DataFrame,
    Xs: Optional[Iterable[pd.DataFrame]] = None,
    loss: LossFunction = MeanAbsolutePercentageError(),
) -> Sequence[float]:
    scores: List[float] = []
    for df_train, df_test in splitter.train_test_splits(y=y):
        model = forecaster.fit(y=df_train, Xs=Xs)
        df_pred = model.forecast(fh=splitter.fh, y=df_train, Xs=Xs)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            score: float = loss(y_true=df_test, y_pred=df_pred, y_train=df_train)
            scores.append(score)

    return scores
