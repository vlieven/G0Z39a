from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, TypeVar

import lightgbm as lgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.forecasting.base import ForecastingHorizon

Data = TypeVar("Data", pd.DataFrame, pd.Series)


class BaseForecaster(ABC):
    def __init__(self, *, lag: int, transformers: Optional[Pipeline] = None) -> None:
        self._lag: int = lag
        self._transformers: Pipeline = transformers or Pipeline(steps=[("passthrough", None)])

    @property
    def lag(self) -> int:
        return self._lag

    def fit(
        self,
        y: pd.DataFrame,
        Xs: Optional[Iterable[pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> BaseForecaster:
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

    def _add_lagged_features(
        self, y: pd.DataFrame, *, forecasting: bool, dropna: bool = True
    ) -> pd.DataFrame:
        """
        Builds a new DataFrame to facilitate regressing over all possible lagged features.
        """
        if not isinstance(y, pd.DataFrame):
            raise ValueError("Only works for DataFrame")
        else:
            if forecasting:
                res: pd.DataFrame = self.__lagged_forecasting_features(y, lag=self.lag)
            else:
                res = self.__lagged_training_features(y, lag=self.lag)

        if dropna:
            return res.dropna()
        else:
            return res

    def _calculate_transformed_features(self, y: pd.DataFrame) -> pd.DataFrame:
        return self._transformers.fit_transform(y).drop(columns=y.columns)

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

        res: pd.DataFrame = pd.DataFrame(data, index=y.index)
        return cls._shift_date_index(res)

    @classmethod
    def _shift_date_index(cls, y: Data, amount: int = 1) -> Data:
        y.index = y.index.set_levels(y.index.levels[-1].shift(amount, freq="D"), level=-1)
        return y


class LGBMForecaster(BaseForecaster):
    def __init__(
        self, estimator: lgb.LGBMModel, *, lag: int = 0, transformers: Optional[Pipeline] = None
    ) -> None:
        super().__init__(lag=lag, transformers=transformers)

        self._model: lgb.LGBMModel = estimator

    def _fit(
        self, y: pd.DataFrame, Xs: Iterable[pd.DataFrame], **kwargs: Any
    ) -> LGBMForecaster:
        target: str = self.__get_target(y)

        y_trans: pd.DataFrame = self._calculate_transformed_features(y)
        y_lagged: pd.DataFrame = self._add_lagged_features(y, forecasting=False)

        X_t: pd.DataFrame = y_lagged.drop(columns=[target])
        for exogenous in (y_trans, *Xs):
            X_t = X_t.join(exogenous, how="left")

        y_t: pd.DataFrame = y_lagged[[target]]

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
        past: pd.Timestamp = cutoff - pd.Timedelta(days=2 * self.lag)
        start: pd.Timestamp = cutoff + pd.Timedelta(days=1)

        target: str = self.__get_target(y)
        y_past: pd.DataFrame = y.loc[pd.IndexSlice[:, past:cutoff], :]

        date: pd.Timestamp = cutoff
        absolute: ForecastingHorizon = fh.to_absolute(cutoff=cutoff.to_period(freq="D"))
        for period in absolute.to_pandas():
            date = period.to_timestamp(freq="D")

            y_pred: pd.DataFrame = self._predict_single_iteration(
                y=y_past, Xs=Xs, target=target, to_predict=date, **kwargs
            )

            y_past = self._concat(y_past, y_pred)
        return y_past.loc[pd.IndexSlice[:, start:date], :]

    @classmethod
    def __get_target(cls, y: pd.DataFrame) -> str:
        if len(y.columns) != 1:
            raise ValueError("More than one dependent variable defined!")
        else:
            target: str = y.columns[0]
        return target

    def _predict_single_iteration(
        self,
        y: pd.DataFrame,
        *,
        Xs: Iterable[pd.DataFrame],
        target: str,
        to_predict: pd.Timestamp,
        **kwargs: Any,
    ) -> pd.DataFrame:
        y_trans: pd.DataFrame = self._shift_date_index(self._calculate_transformed_features(y))
        y_lagged: pd.DataFrame = self._add_lagged_features(y, forecasting=True)

        X_t: pd.DataFrame = y_lagged.loc[pd.IndexSlice[:, to_predict], :]
        for exogenous in (y_trans, *Xs):
            X_t = X_t.merge(exogenous, how="left", left_index=True, right_index=True)

        y_pred = self._model.predict(X=X_t, **kwargs)
        return pd.DataFrame(y_pred, index=X_t.index, columns=[target])

    @classmethod
    def _concat(cls, *dfs: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(dfs, join="inner", copy=False).sort_index()
