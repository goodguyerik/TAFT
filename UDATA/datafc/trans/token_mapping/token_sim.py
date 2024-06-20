import logging
import time
from typing import List, Tuple

from sklearn.model_selection import KFold

from datafc.repr.column import Column
from datafc.syn.pattern import Pattern
from datafc.syn.token import Token
from datafc.trans.operators import Operation, Constant
from datafc.trans.token_mapping import TokenMappingBaseModel
from datafc.trans.token_mapping.classifier import MultiBinary, MassMultiBinary
from datafc.trans.token_mapping.feature.column_sim import (
    values_jaccard,
    syntactic_sim,
    text_cosine,
    # semantic_sim,
    ngram_jaccard,
    token_jaccard,
)

logger = logging.getLogger("myapp")


class TokenSimMappingModel(TokenMappingBaseModel):
    def __init__(self, features=None):
        if features is None:
            features = ["jaccard", "syn"]
        self.scoring_model: MultiBinary = MultiBinary[Column](self.compute_sim)
        self.features = features
        super().__init__()

    def compute_sim(self, col1: Column, col2: Column) -> List[float]:
        feature_values = []
        for feature in self.features:
            if feature == "jaccard":
                feature_values.extend([values_jaccard(col1, col2)])
            elif feature == "token_jaccard":
                feature_values.extend([token_jaccard(col1, col2)])
            elif feature == "ngram_jaccard":
                feature_values.extend([ngram_jaccard(col1, col2, 2)])
            elif feature == "cosine":
                feature_values.extend([text_cosine(col1, col2)])
            elif feature == "syn":
                feature_values.extend([syntactic_sim(col1, col2)])
            # elif feature == "sem":
            #     feature_values.extend([semantic_sim(col1, col2)])
        return feature_values

    def score_operation(self, operation) -> float:
        if isinstance(operation, Constant):
            return 1.0
        transformed_column = Column(values=operation.transform())
        target_column = Column(values=operation.target_token.values)
        result = self.scoring_model.predict_similarity(
            transformed_column, target_column
        )
        # print(
        #     operation,
        #     result,
        #     operation.transform()[:3],
        #     operation.target_token.values[:3],
        # )
        return result

    def generate_candidate_functions(self, source_token: Token, target_token: Token):
        operation_candidates = Operation.find_suitable_transformations(
            source_token, target_token
        )
        if operation_candidates:
            return max(
                {op: self.score_operation(op) for op in operation_candidates}.items(),
                key=lambda x: x[1],
            )
        return None, 0.0

    def train_scoring_model(self, example_patterns_by_groups: List[List[Pattern]]):
        labeled_pairs = []

        for idx, example_patterns in enumerate(example_patterns_by_groups):
            labeled_cols: List[Tuple[str, Column]] = []
            idx = 0
            for pattern in example_patterns:
                for token in pattern.tokens:
                    if len(token.values) <= 2:
                        continue
                    idx += 1

                    kf = KFold(n_splits=2)
                    for train_indices, test_indices in kf.split(token.values):
                        train_values = []
                        for index in test_indices:
                            train_values.append(token.values[index])
                        column = Column(str(idx), str(idx), train_values)
                        labeled_cols.append((str(idx), column))
            for label1, col1 in labeled_cols:
                for label2, col2 in labeled_cols:
                    # if label1 == label2 or abs(int(label1) - int(label2)) == 1:
                    labeled_pairs.append((col1, col2, label1 == label2))
        try:
            self.scoring_model.train_from_pairs(labeled_pairs)
        except Exception as e:
            logger.error(e)
