from typing import Any, NamedTuple

from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
)


class Loss(NamedTuple):
    name: str
    function: Any


mae = Loss(
    name="mean_absolute_error",
    function=MeanAbsoluteError(),
)

mape = Loss(
    name="mean_absolute_percentage_error",
    function=MeanAbsolutePercentageError(symmetric=True),
)
