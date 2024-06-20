import logging
from typing import List, Tuple, Dict

import numpy as np

from datafc.syn.pattern import Pattern
from datafc.syn.token import Token
from datafc.trans.operators import Operation, Constant
from datafc.trans.token_mapping import TokenMappingBaseModel
from datafc.trans.token_mapping.classifier import MultiClass
from datafc.trans.token_mapping.feature.column_attr import (
    char_count,
    type_count,
    word_count,
    length,
)

logger = logging.getLogger("myapp")


class TokenAttrMappingModel(TokenMappingBaseModel):
    def __init__(self):
        self.scoring_model: MultiClass = MultiClass[str](
            self.compute_feature, "random_forest"
        )
        self.token_to_idx: Dict[Token, str] = {}

    @staticmethod
    def compute_feature(value):
        feature_vector = {}
        for feature_func in [char_count, type_count, word_count, length]:
            feature_vector.update(feature_func(value))
        return feature_vector

    def score_operation(self, operation: Operation) -> float:
        if isinstance(operation, Constant):
            return 1.0

        token_to_score_list = self.scoring_model.predict_proba(operation.transform())
        scores = [
            x[self.token_to_idx[operation.target_token]] for x in token_to_score_list
        ]
        return float(np.mean(scores))

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
        self.token_to_idx.clear()
        idx = 0
        label_cols: List[Tuple[str, dict]] = []

        for example_patterns in example_patterns_by_groups:
            for idx1, pattern in enumerate(example_patterns):
                for token in pattern.tokens:
                    self.token_to_idx[token] = str(idx)
                    for value in token.values:
                        label_cols.append((str(idx), value))
                idx += 1
        self.scoring_model.train(label_cols)
