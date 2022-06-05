from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Iterable, Optional, Protocol, Sequence, cast

import joblib
import pandas as pd
from joblib import Parallel, delayed
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from tqdm.auto import tqdm

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
    Xs: Optional[pd.DataFrame] = None,
    loss: LossFunction = MeanAbsolutePercentageError(),
    threads: Optional[int] = None,
) -> Sequence[float]:
    if not threads:
        threads = os.cpu_count()

    with tqdm_joblib(
        tqdm(desc="Cross Validation", total=splitter.get_n_splits(y))
    ) as progress_bar:
        return cast(
            Sequence[float],
            Parallel(n_jobs=threads)(
                delayed(_single_run)(
                    **{
                        "forecaster": forecaster,
                        "df_train": df_train,
                        "df_test": df_test,
                        "exogenous": Xs,
                        "loss": loss,
                        "fh": splitter.fh,
                    }
                )
                for df_train, df_test in splitter.train_test_splits(y=y)
            ),
        )


def _single_run(
    forecaster: BaseForecaster,
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    exogenous: Optional[pd.DataFrame],
    loss: LossFunction,
    fh: ForecastingHorizon,
) -> float:
    # Align indices and avoid information spill
    if exogenous:
        Xs: Iterable[pd.DataFrame] = [df_train.join(exogenous).drop(columns=df_train.columns)]
    else:
        Xs = []

    model = forecaster.fit(y=df_train, Xs=Xs)
    df_pred = model.forecast(fh=fh, y=df_train, Xs=Xs)

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        score: float = loss(y_true=df_test, y_pred=df_pred, y_train=df_train)
    return score


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> tqdm:
    """Context manager to patch joblib to report into tqdm progress bar given as argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):  # type: ignore[misc]
        def __call__(
            self, *args: Any, **kwargs: Any
        ) -> joblib.parallel.BatchCompletionCallBack:
            tqdm_object.update(n=self.batch_size)
            return cast(
                joblib.parallel.BatchCompletionCallBack, super().__call__(*args, **kwargs)
            )

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
