from pathlib import Path
from typing import List, Tuple, Dict, TypeVar, Callable, Union, Optional

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from datafc.trans.token_mapping.classifier.base import BaseClassifier

T = TypeVar("T")

DUMMY_STR = "|#$%^&"


class MultiBinary(BaseClassifier[T]):
    def __init__(
        self, sim_func: Optional[Callable[[T, T], List[float]]], method: str = "lr"
    ):
        self.labeled_data: List[Tuple[str, T]] = []
        if method == "random_forest":
            self.model = RandomForestClassifier(n_estimators=10)
        else:
            self.model = LogisticRegression(solver="lbfgs")
        self.sim_func = sim_func

    def save(self, path: Union[str, Path]):
        dump(self.model, str(path))

    def load(self, path: Union[str, Path]):
        self.model = load(str(path))

    def store(self, labeled_cols: List[Tuple[str, T]]):
        self.labeled_data.extend(labeled_cols)

    def clear(self):
        self.labeled_data.clear()

    def create_feature_vectors(
        self, labeled_cols1: List[Tuple[str, T]], labeled_cols2: List[Tuple[str, T]]
    ) -> Tuple[List[List[float]], List[bool]]:
        vectors = []
        labels = []
        for label1, col1 in labeled_cols1:
            for label2, col2 in labeled_cols2:
                if col1 == col2:
                    continue
                vectors.append(self.sim_func(col1, col2))
                labels.append(label1 == label2)
        return vectors, labels

    def create_feature_vectors_from_pairs(self, labeled_pairs: List[Tuple[T, T, bool]]):
        vectors = []
        labels = []
        for col1, col2, label in labeled_pairs:
            vectors.append(self.sim_func(col1, col2))
            labels.append(label)
        return vectors, labels

    def train(self, labeled_cols: List[Tuple[str, T]]):
        train_vectors, train_labels = self.create_feature_vectors(
            labeled_cols, labeled_cols
        )
        self.model.fit(train_vectors, train_labels)

    def train_from_pairs(self, labeled_pairs: List[Tuple[T, T, bool]]):
        train_vectors, train_labels = self.create_feature_vectors_from_pairs(
            labeled_pairs
        )
        self.model.fit(train_vectors, train_labels)

    def predict(self, labeled_cols: List[T]) -> List[str]:
        train_vectors, _ = zip(
            *self.create_feature_vectors(
                [(DUMMY_STR, labeled_col) for labeled_col in labeled_cols],
                self.labeled_data,
            )
        )
        return max(self.model.predict_proba(train_vectors), key=lambda x: x[1])

    def predict_proba(self, labeled_cols: List[T]) -> List[Dict[str, float]]:
        train_vectors, _ = zip(
            *self.create_feature_vectors(
                [(DUMMY_STR, labeled_col) for labeled_col in labeled_cols],
                self.labeled_data,
            )
        )

        labels = [label for label, _ in self.labeled_data]
        scores_list = self.model.predict_proba(train_vectors)

        return [dict(zip(labels, scores)) for scores in scores_list]

    def predict_similarity(
        self, original_col: T, target_col: T, is_print=False
    ) -> float:
        train_vectors, _ = self.create_feature_vectors(
            [(DUMMY_STR, original_col)], [(DUMMY_STR, target_col)]
        )
        return self.model.predict_proba(train_vectors)[0][1]


class MassMultiBinary(MultiBinary[T]):
    def __init__(self, mass_sim_func, method: str = "lr"):
        super().__init__(None, method)
        self.labeled_data: List[Tuple[str, T]] = []
        if method == "random_forest":
            self.model = RandomForestClassifier(n_estimators=10)
        else:
            self.model = LogisticRegression(solver="lbfgs")
        self.mass_sim_func = mass_sim_func

    def save(self, path: Union[str, Path]):
        dump(self.model, str(path))

    def load(self, path: Union[str, Path]):
        self.model = load(str(path))

    def store(self, labeled_cols: List[Tuple[str, T]]):
        self.labeled_data.extend(labeled_cols)

    def clear(self):
        self.labeled_data.clear()

    def create_feature_vectors_from_pairs(self, labeled_pairs: List[Tuple[T, T, bool]]):
        return self.mass_sim_func(labeled_pairs)

    def train_from_pairs(self, labeled_pairs: List[Tuple[T, T, bool]]):
        train_vectors, train_labels = self.create_feature_vectors_from_pairs(
            labeled_pairs
        )
        self.model.fit(train_vectors, train_labels)

    def predict(self, labeled_cols: List[T]) -> List[str]:
        train_vectors, _ = zip(
            *self.create_feature_vectors(
                [(DUMMY_STR, labeled_col) for labeled_col in labeled_cols],
                self.labeled_data,
            )
        )
        return max(self.model.predict_proba(train_vectors), key=lambda x: x[1])

    def predict_proba(self, labeled_cols: List[T]) -> List[Dict[str, float]]:
        train_vectors, _ = zip(
            *self.create_feature_vectors(
                [(DUMMY_STR, labeled_col) for labeled_col in labeled_cols],
                self.labeled_data,
            )
        )

        labels = [label for label, _ in self.labeled_data]
        scores_list = self.model.predict_proba(train_vectors)

        return [dict(zip(labels, scores)) for scores in scores_list]

    def predict_similarity(self, original_col: T, target_col: T) -> float:
        train_vectors, _ = self.create_feature_vectors(
            [(DUMMY_STR, original_col)], [(DUMMY_STR, target_col)]
        )
        return self.model.predict_proba(train_vectors)[0][1]
