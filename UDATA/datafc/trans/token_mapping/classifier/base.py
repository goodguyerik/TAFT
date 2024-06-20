from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar, Tuple, List, Dict

T = TypeVar("T")


class BaseClassifier(Generic[T], metaclass=ABCMeta):
    @abstractmethod
    def train(self, labeled_cols: List[Tuple[str, T]]):
        pass

    @abstractmethod
    def predict(self, labeled_col: T) -> str:
        pass

    @abstractmethod
    def predict_proba(self, labeled_col: T) -> Dict[str, float]:
        pass
