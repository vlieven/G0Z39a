from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, Iterable, List, Optional

import lightgbm as lgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.forecasting.base import ForecastingHorizon


class BaseForecaster(ABC):
    def __init__(self) -> None:
        self._transformers: Pipeline = Pipeline(steps=[])

    def fit(
        self,
        y: pd.DataFrame,
        Xs: Optional[Iterable[pd.DataFrame]] = None,
        transformers: Optional[Pipeline] = None,
        **kwargs: Any,
    ) -> BaseForecaster:
        if transformers:
            self._transformers = transformers

        if not Xs:
            Xs = []

        return self._fit(y=y, Xs=Xs, **kwargs)

    @abstractmethod
    def _fit(
        self, y: pd.DataFrame, Xs: Iterable[pd.DataFrame], **kwargs: Any
    ) -> BaseForecaster:
        raise NotImplementedError

    def predict(
        self, fh: ForecastingHorizon, Xs: Optional[Iterable[pd.DataFrame]] = None, **kwargs: Any
    ) -> pd.DataFrame:
        if not Xs:
            Xs = []

        raise NotImplementedError(
            """Although sktime mirrors the regular fit-predict API provided by scikit-learn,
            I find this concept to be quite confusing in the context of forecasting problems.
            Please use the `forecast()` method instead.
            """
        )

    def forecast(
        self,
        fh: ForecastingHorizon,
        y: pd.DataFrame,
        *,
        Xs: Optional[Iterable[pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not Xs:
            Xs = []

        return self._forecast(fh=fh, y=y, Xs=Xs, **kwargs)

    @abstractmethod
    def _forecast(
        self,
        fh: ForecastingHorizon,
        y: pd.DataFrame,
        *,
        Xs: Iterable[pd.DataFrame],
        **kwargs: Any,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _add_lagged_features(
        cls, y: pd.DataFrame, *, lag: int, forecasting: bool, dropna: bool = True
    ) -> pd.DataFrame:
        """
        Builds a new DataFrame to facilitate regressing over all possible lagged features.
        """
        if not isinstance(y, pd.DataFrame):
            raise ValueError("Only works for DataFrame")
        else:
            if forecasting:
                res: pd.DataFrame = cls.__lagged_forecasting_features(y, lag=lag)
            else:
                res = cls.__lagged_training_features(y, lag=lag)

        if dropna:
            return res.dropna()
        else:
            return res

    @classmethod
    def __lagged_training_features(cls, y: pd.DataFrame, *, lag: int) -> pd.DataFrame:
        groupby_levels: Iterable[int] = range(0, y.index.nlevels - 1)
        data: Dict[str, pd.Series] = {}

        for col_name in y:
            # keep unlagged Series
            data[col_name] = y[col_name]

            # create lagged Series
            for lag_ in range(1, lag + 1):
                data[f"{col_name}_{lag_}"] = y.groupby(level=groupby_levels)[col_name].shift(
                    lag_
                )

        return pd.DataFrame(data, index=y.index)

    @classmethod
    def __lagged_forecasting_features(cls, y: pd.DataFrame, *, lag: int) -> pd.DataFrame:
        groupby_levels: Iterable[int] = range(0, y.index.nlevels - 1)
        data: Dict[str, pd.Series] = {}

        for col_name in y:
            # keep unlagged Series
            data[f"{col_name}_{1}"] = y[col_name]

            # create lagged Series
            for lag_ in range(2, lag + 1):
                data[f"{col_name}_{lag_}"] = y.groupby(level=groupby_levels)[col_name].shift(
                    lag_
                )

        # Note: do not refactor index shifting, doing it on y.index directly will not work
        res: pd.DataFrame = pd.DataFrame(data, index=y.index)
        res.index = res.index.set_levels(res.index.levels[-1].shift(1, freq="D"), level=-1)
        return res


class LGBMForecaster(BaseForecaster):
    def __init__(self, estimator: lgb.LGBMModel, lag: int = 0) -> None:
        super().__init__()

        self._model: lgb.LGBMModel = estimator
        self._lag: int = lag
        self._transformers: Optional[Pipeline] = None

    @property
    def lag(self) -> int:
        return self._lag

    def _fit(
        self, y: pd.DataFrame, Xs: Iterable[pd.DataFrame], **kwargs: Any
    ) -> LGBMForecaster:
        targets: Iterable[str] = y.columns
        y_lagged: pd.DataFrame = self._add_lagged_features(y, lag=self._lag, forecasting=False)

        X_t: pd.DataFrame = y_lagged.drop(columns=targets)
        for exogenous in Xs:
            X_t = X_t.merge(exogenous, how="left", left_index=True, right_index=True)

        y_t: pd.DataFrame = y_lagged[targets]

        self._model = self._model.fit(X=X_t, y=y_t, **kwargs)
        return self

    def _forecast(
        self,
        fh: ForecastingHorizon,
        y: pd.DataFrame,
        *,
        Xs: Iterable[pd.DataFrame],
        **kwargs: Any,
    ) -> pd.DataFrame:
        cutoff: pd.Timestamp = y.index.get_level_values(-1).max()
        start: pd.Timestamp = cutoff - pd.Timedelta(days=self.lag)
        absolute: ForecastingHorizon = fh.to_absolute(cutoff=cutoff.to_period(freq="D"))

        targets: Collection[str] = y.columns

        y_past: pd.DataFrame = y.loc[pd.IndexSlice[:, start:cutoff], :]

        results: List[pd.DataFrame] = []
        for _ in absolute.to_pandas():
            y_pred: pd.DataFrame = self._predict_single_iteration(
                y_past=y_past, Xs=Xs, targets=targets, **kwargs
            )
            results.append(y_pred)

            y_past = self._update_y_past(y_past, y_pred)
        return self._concat(*results)

    def _predict_single_iteration(
        self,
        y_past: pd.DataFrame,
        *,
        Xs: Iterable[pd.DataFrame],
        targets: Collection[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        y_lagged: pd.DataFrame = self._add_lagged_features(
            y_past, lag=self._lag, forecasting=True
        )

        X_t: pd.DataFrame = y_lagged
        for exogenous in Xs:
            X_t = X_t.merge(exogenous, how="left", left_index=True, right_index=True)

        y_pred = self._model.predict(X=X_t, **kwargs)
        return pd.DataFrame(y_pred, index=y_lagged.index, columns=targets)

    def _update_y_past(self, y_past: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        earliest: pd.Timestamp = y_past.index.get_level_values(-1).min()

        y_new: pd.DataFrame = self._concat(y_past, y_pred)
        return y_new.drop(earliest, level=-1)

    @classmethod
    def _concat(cls, *dfs: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(dfs, join="inner", copy=False).sort_index()
