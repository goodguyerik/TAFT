import itertools
import logging
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

from datafc.syn.pattern import Pattern
from datafc.syn.token import Token
from datafc.trans.operators import Operation, Constant

logger = logging.getLogger("myapp")


class TokenMappingBaseModel(metaclass=ABCMeta):
    @abstractmethod
    def train_scoring_model(self, example_patterns: List[Pattern]):
        pass

    @abstractmethod
    def score_operation(self, operation: Operation) -> float:
        pass

    @abstractmethod
    def generate_candidate_functions(self, source_token: Token, target_token: Token):
        pass

    def generate_token_mappings(
        self, original_pattern: Pattern, target_pattern: Pattern
    ):
        assert original_pattern.tokens, "Original pattern cannot be empty"
        assert target_pattern.tokens, "Target pattern cannot be empty"

        token_pair_to_score = np.zeros(
            (len(target_pattern.tokens), len(original_pattern.tokens))
        )
        token_pair_to_operation = np.empty(
            (len(target_pattern.tokens), len(original_pattern.tokens)), dtype=object
        )

        for idx1, target_token in enumerate(target_pattern.tokens):
            for idx2, original_token in enumerate(original_pattern.tokens):
                operation, score = self.generate_candidate_functions(
                    original_token, target_token
                )

                token_pair_to_operation[idx1][idx2] = operation
                token_pair_to_score[idx1][idx2] = score

        return token_pair_to_score, token_pair_to_operation

    @staticmethod
    def find_best_mapping(
        token_pair_to_score, token_pair_to_operation
    ) -> Tuple[float, Dict[int, Operation]]:

        index_to_operation = {}
        constant_indices = []
        post_delete_mapping = {}

        for idx1 in range(token_pair_to_operation.shape[0]):
            for idx2 in range(token_pair_to_operation.shape[1]):
                if isinstance(token_pair_to_operation[idx1][idx2], Constant):
                    index_to_operation[idx1] = token_pair_to_operation[idx1][idx2]
                    constant_indices.append(idx1)
                    break
            else:
                post_delete_mapping[len(post_delete_mapping)] = idx1

        token_pair_to_score = np.delete(token_pair_to_score, constant_indices, axis=0)
        score_matrix = np.array(token_pair_to_score)

        row_indices, col_indices = linear_sum_assignment(-score_matrix)

        for target_idx, original_idx in zip(row_indices, col_indices):
            pre_delete_index = post_delete_mapping[target_idx]
            index_to_operation[pre_delete_index] = token_pair_to_operation[
                pre_delete_index
            ][original_idx]

        return (
            score_matrix[row_indices, col_indices].sum() / len(row_indices),
            index_to_operation,
        )

    @staticmethod
    def find_best_k_ambiguous_mappings(
        token_pair_to_score, token_pair_to_operation, k: int
    ) -> Tuple[List[float], List[Dict[int, Operation]]]:

        constant_indices = []
        pre_to_post_index = {}
        post_to_pre_index = {}

        ambiguous_choices = []

        # print("Token to token operation:")
        # print(
        #     "\n"
        #     + tabulate(
        #         token_pair_to_operation,
        #         range(len(token_pair_to_score)),
        #         tablefmt="fancy_grid",
        #     )
        # )
        #
        # print("Token to token score:")
        # print(
        #     "\n"
        #     + tabulate(
        #         token_pair_to_score,
        #         range(len(token_pair_to_score)),
        #         tablefmt="fancy_grid",
        #     )
        # )

        for idx1 in range(token_pair_to_operation.shape[0]):
            max_value = np.max(token_pair_to_score[idx1])
            for idx2 in range(token_pair_to_operation.shape[1]):
                if isinstance(token_pair_to_operation[idx1][idx2], Constant):
                    constant_indices.append(idx1)
                    break
                ambiguous_choices.append(
                    (idx1, idx2, max_value - token_pair_to_score[idx1][idx2])
                )
            else:
                pre_to_post_index[idx1] = len(pre_to_post_index)
                post_to_pre_index[len(post_to_pre_index)] = idx1

        possible_matches_by_rows = [[] for _ in range(len(post_to_pre_index))]

        for idx1, idx2, _ in sorted(ambiguous_choices, key=lambda x: x[2])[:10]:
            possible_matches_by_rows[pre_to_post_index[idx1]].append(idx2)

        candidate_mappings = list(itertools.product(*possible_matches_by_rows))
        mappings_with_scores = []
        for possible_mapping in candidate_mappings:
            if len(possible_mapping) != len(set(possible_mapping)):
                continue

            operations = {}
            score = 0

            for idx1 in constant_indices:
                operations[idx1] = token_pair_to_operation[idx1][0]

            for idx1, idx2 in enumerate(list(possible_mapping)):
                operations[post_to_pre_index[idx1]] = token_pair_to_operation[
                    post_to_pre_index[idx1]
                ][idx2]

                score += token_pair_to_score[post_to_pre_index[idx1]][idx2]
            mappings_with_scores.append((operations, score))

        mappings_with_scores.sort(key=lambda x: x[1], reverse=True)
        if mappings_with_scores:
            operations, scores = zip(*mappings_with_scores[: k * 2])
            return list(scores), list(operations)
        return [0], [{}]

    @staticmethod
    def generate_output_strings(original_pattern, target_pattern, token_to_operation):
        token_transformed_values = []
        for idx in range(len(target_pattern.tokens)):
            token_transformed_values.append(token_to_operation[idx].transform())

        original_to_transformed_pairs = []

        for idx, target_string in enumerate(original_pattern.values):
            target_string = "".join(
                [
                    transformed_strings[idx]
                    for transformed_strings in token_transformed_values
                ]
            )
            original_to_transformed_pairs.append(
                (original_pattern.values[idx], target_string)
            )

        return original_to_transformed_pairs

    def learn(
        self, original_pattern: Pattern, target_pattern: Pattern
    ) -> Tuple[List[Tuple[str, str]], float]:

        token_pair_to_score, token_pair_to_operation = self.generate_token_mappings(
            original_pattern, target_pattern
        )

        transformation_score, token_to_operation = TokenMappingBaseModel.find_best_mapping(
            token_pair_to_score, token_pair_to_operation
        )

        if None in token_to_operation.values() or len(target_pattern.tokens) != len(
            token_to_operation
        ):
            return (
                [(original_value, "") for original_value in original_pattern.values],
                0,
            )

        return (
            self.generate_output_strings(
                original_pattern, target_pattern, token_to_operation
            ),
            transformation_score,
        )

    def learn_top_k(
        self, original_pattern: Pattern, target_pattern: Pattern, k: int
    ) -> Tuple[List[List[Tuple[str, str]]], List[float]]:

        scores = []
        original_to_target_pairs_list = []

        token_pair_to_score, token_pair_to_operation = self.generate_token_mappings(
            original_pattern, target_pattern
        )

        # print(
        #     "original",
        #     original_pattern.values[:3],
        #     [token.values[:3] for token in original_pattern.tokens],
        #     [token for token in original_pattern.tokens],
        # )
        # print(
        #     "target",
        #     target_pattern.values[:3],
        #     [token.values[:3] for token in target_pattern.tokens],
        #     [token for token in target_pattern.tokens],
        # )

        logger.debug("Original pattern: %s" % original_pattern.tokens)
        logger.debug("Target pattern: %s" % target_pattern.tokens)

        transformation_scores, token_to_operation_list = TokenMappingBaseModel.find_best_k_ambiguous_mappings(
            token_pair_to_score, token_pair_to_operation, k
        )

        for transformation_score, token_to_operation in zip(
            transformation_scores, token_to_operation_list
        ):

            if None in token_to_operation.values() or len(target_pattern.tokens) != len(
                token_to_operation
            ):
                scores.append(0)
                original_to_target_pairs_list.append(
                    [(original_value, "") for original_value in original_pattern.values]
                )
            else:
                original_to_target_string_pairs = TokenMappingBaseModel.generate_output_strings(
                    original_pattern, target_pattern, token_to_operation
                )
                scores.append(transformation_score)
                original_to_target_pairs_list.append(original_to_target_string_pairs)
        return original_to_target_pairs_list, scores
