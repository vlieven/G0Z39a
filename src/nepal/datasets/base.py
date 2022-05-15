from abc import ABC, abstractmethod
import pandas as pd


class Dataset(ABC):

    @classmethod
    @abstractmethod
    def collect(cls) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls) -> pd.DataFrame:
        raise NotImplementedError
