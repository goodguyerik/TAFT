import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from datafc.eval.validator import Validator
from datafc.repr.column import Column
from datafc.syn.pattern import PatternTree, PatternNode
from datafc.syn.token import PATTERNS_BY_LEVEL
from datafc.trans.token_mapping import MultiBinary
from datafc.trans.token_mapping.token_attr import TokenAttrMappingModel
from datafc.trans.token_mapping.token_sim import TokenSimMappingModel

logger = logging.getLogger("myapp")


@dataclass
class TransformedResult:
    original_target_pairs: List[Tuple[str, str]]
    score: float


class PatternMappingModel:
    def __init__(self, method="sim", mapping_features=None):
        if mapping_features is None:
            mapping_features = ["jaccard", "syn"]
        self.mapping_method = method

        if method == "sim":
            self.token_mapper: TokenSimMappingModel = TokenSimMappingModel(
                mapping_features
            )
        elif method == "attr":
            self.token_mapper: TokenAttrMappingModel = TokenAttrMappingModel()
        self.validator: Validator = Validator(0.05)

    def learn(self, original_tree: PatternTree, target_tree: PatternTree):
        self.token_mapper.train_scoring_model(
            original_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True)
            + target_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True)
        )

        final_node_to_result: Dict[PatternNode, TransformedResult] = {}

        target_col = Column(values=target_tree.values)
        best_score, best_level = 0, 0

        for layer in [3, 2, 1]:

            transformed_col = Column()

            original_nodes = sorted(
                original_tree.node_in_layers[layer],
                key=lambda x: len(x.value.tokens),
                reverse=True,
            )
            transformed_nodes = sorted(
                target_tree.node_in_layers[layer],
                key=lambda x: len(x.value.tokens),
                reverse=True,
            )

            node_to_result: Dict[PatternNode, TransformedResult] = {}

            for original_node in original_nodes:
                transformed_results = []
                for target_node in transformed_nodes:
                    result = self.token_mapper.learn(
                        original_node.value, target_node.value
                    )

                    transformed_results.append(
                        TransformedResult(
                            original_target_pairs=result[0], score=result[1]
                        )
                    )

                node_to_result[original_node] = max(
                    transformed_results, key=lambda x: x.score
                )

                value_tuples = list(
                    zip(*node_to_result[original_node].original_target_pairs)
                )
                transformed_col.extend_values(list(value_tuples[1]))

            pattern_score = self.token_mapper.scoring_model.predict_similarity(
                target_col, transformed_col
            )

            if pattern_score > best_score:
                best_score = pattern_score
                best_level = layer
                final_node_to_result = node_to_result.copy()

        validated_original_to_transformed_tuples = []

        scores = []

        for node, result in final_node_to_result.items():

            """for original_target_pair in result.original_target_pairs:
                validation_result = self.validator.validate_results(
                    original_target_pair,
                    original_tree,
                    target_tree,
                    result.score,
                    best_level,
                )
                validated_original_to_transformed_tuples.append(
                    (original_target_pair[0], original_target_pair[1], validation_result)
                )

            scores.append(result.score)

        return validated_original_to_transformed_tuples, sum(scores) * 1.0 / len(scores)"""
            for original, transformed in result.original_target_pairs:
                validated_original_to_transformed_tuples.append((original, transformed))
        return validated_original_to_transformed_tuples, 0

    def learn_top_k(self, original_tree: PatternTree, target_tree: PatternTree, k: int):
        self.token_mapper.train_scoring_model(
            original_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True)
            + target_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True)
        )

        final_node_to_results: Dict[PatternNode, List[TransformedResult]] = {}
        target_col = Column(values=target_tree.values)
        best_score, best_level = -1, 0

        for layer in range(len(PATTERNS_BY_LEVEL) - 1, 0, -1):
            logger.debug("Mapping level: %s" % layer)

            transformed_col = Column()

            original_nodes = sorted(
                original_tree.node_in_layers[layer],
                key=lambda x: (len(x.value.values), -len(x.value.tokens)),
                reverse=True,
            )

            transformed_nodes = sorted(
                target_tree.node_in_layers[layer],
                key=lambda x: (len(x.value.values), -len(x.value.tokens)),
                reverse=True,
            )

            node_to_results = {}
            pattern_scores = []

            logger.debug(
                "Num combinations: %s * %s = %s"
                % (
                    len(original_nodes),
                    len(transformed_nodes),
                    len(original_nodes) * len(transformed_nodes),
                )
            )

            num_possible_nodes = 10

            for original_node in original_nodes[:num_possible_nodes]:
                token_mapping_results = []

                for target_node in transformed_nodes:
                    validated_values_by_pattern, scores_by_pattern = self.token_mapper.learn_top_k(
                        original_node.value, target_node.value, k
                    )

                    token_mapping_results.append(
                        [
                            TransformedResult(*result)
                            for result in zip(
                                validated_values_by_pattern, scores_by_pattern
                            )
                        ]
                    )

                node_to_results[original_node] = max(
                    token_mapping_results, key=lambda x: x[0].score
                )

                pattern_scores.append(
                    max([x.score for x in node_to_results[original_node]])
                )

                value_tuples = list(
                    zip(*node_to_results[original_node][0].original_target_pairs)
                )
                transformed_col.extend_values(list(value_tuples[1]))

            for original_node in original_nodes[num_possible_nodes:]:
                for i in range(k):
                    transform_result = TransformedResult(
                        [(str_value, "") for str_value in original_node.value.values],
                        0.0,
                    )
                    node_to_results[original_node] = [transform_result]
                    value_tuples = list(
                        zip(*node_to_results[original_node][0].original_target_pairs)
                    )
                    transformed_col.extend_values(list(value_tuples[1]))

            if self.mapping_method == "sim":
                assert isinstance(self.token_mapper.scoring_model, MultiBinary)
                pattern_score = (
                    self.token_mapper.scoring_model.predict_similarity(
                        target_col, transformed_col
                    )
                    * len([x for x in transformed_col.values if x])
                    * 1.0
                    / len(transformed_col.values)
                )
            else:
                pattern_score = np.mean(pattern_scores)

            # print(
            #     "Pattern Score",
            #     pattern_score,
            #     transformed_col.values,
            #     target_col.values,
            # )

            if pattern_score > best_score:
                best_score = pattern_score
                best_level = layer
                final_node_to_results = node_to_results.copy()

        validated_values_by_pattern = [[] for _ in range(len(final_node_to_results))]

        scores_by_pattern = [[] for _ in range(len(final_node_to_results))]
        idx = 0

        full_validation_result = False

        for node, results in final_node_to_results.items():
            full_validation_result = (
                full_validation_result
                or self.validator.validate_results(
                    results[0].original_target_pairs,
                    target_tree,
                    results[0].score,
                    0 if len(results) == 1 else results[1].score,
                    1,
                )[0]
            )

            validated_values_by_pattern[idx] = []
            for idx1, result in enumerate(results):
                for idx2, (original_value, transformed_value) in enumerate(
                    result.original_target_pairs
                ):
                    if len(results) == 1:
                        current_score = 1
                        next_score = 0
                    elif idx1 < len(results) - 1:
                        current_score = result.score
                        next_score = results[idx1 + 1].score
                    else:
                        current_score = 0
                        next_score = 0
                    validation_result = self.validator.validate_results(
                        [(original_value, transformed_value)],
                        target_tree,
                        current_score,
                        next_score,
                        1,
                    )
                    if idx1 == 0:
                        validated_values_by_pattern[idx].append(
                            (original_value, [transformed_value], [validation_result])
                        )
                    else:
                        current_values = validated_values_by_pattern[idx][idx2][0]
                        assert (
                            original_value == current_values
                        ), f"Original value should be the same f{original_value} vs {current_values}"
                        validated_values_by_pattern[idx][idx2][1].append(
                            transformed_value
                        )
                        validated_values_by_pattern[idx][idx2][2].append(
                            validation_result
                        )

                scores_by_pattern[idx].append(result.score)
            idx += 1

        return validated_values_by_pattern, scores_by_pattern, full_validation_result
