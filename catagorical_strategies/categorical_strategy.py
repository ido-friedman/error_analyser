# categorical_strategy.py

from abc import ABC, abstractmethod
import pandas as pd


class CategoricalStrategy(ABC):
    @abstractmethod
    def calculate_probability(self, good_values: pd.Series, bad_values: pd.Series) -> float:
        pass
