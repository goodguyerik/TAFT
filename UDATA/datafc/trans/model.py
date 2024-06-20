import logging
import time
from typing import List

from datafc.syn.pattern import PatternTree
from datafc.trans.pattern_mapping import PatternMappingModel

logger = logging.getLogger("myapp")


class TransformationModel:
    def __init__(self, mapping_method="sim", mapping_features=None):
        if mapping_features is None:
            mapping_features = ["jaccard", "syn"]
        self.pattern_mapper = PatternMappingModel(mapping_method, mapping_features)

    @staticmethod
    def generate_patterns(original_values: List[str], transformed_values: List[str]):
        transformed_tree = PatternTree.build_from_strings(transformed_values)
        logger.debug("Finish learning pattern tree from transformed values")

        original_tree = PatternTree.build_from_strings(original_values)
        logger.debug("Finish learning pattern tree from original values")

        return original_tree, transformed_tree

    def learn_top_k_active(
        self, original_values: List[str], transformed_values: List[str], k: int
    ):
        original_tree, transformed_tree = self.generate_patterns(
            original_values, transformed_values
        )
        original_to_transformed_pairs_by_pattern, scores_by_pattern = self.pattern_mapper.learn_top_k(
            original_tree, transformed_tree, k
        )

        return original_to_transformed_pairs_by_pattern, scores_by_pattern

    def learn_top_k(
        self, original_values: List[str], transformed_values: List[str], k: int
    ):
        original_tree, transformed_tree = self.generate_patterns(
            original_values, transformed_values
        )
        original_to_transformed_pairs_by_pattern, scores_by_pattern, full_validation_result = self.pattern_mapper.learn_top_k(
            original_tree, transformed_tree, k
        )

        original_to_transformed_results = []

        for original_to_transformed_pairs in original_to_transformed_pairs_by_pattern:
            for result in original_to_transformed_pairs:
                original_to_transformed_results.append(result)

        return (
            original_to_transformed_results,
            scores_by_pattern,
            full_validation_result,
        )

    def learn(self, original_values: List[str], transformed_values: List[str]):
        original_tree, transformed_tree = self.generate_patterns(
            original_values, transformed_values
        )

        original_to_transformed_tuples, score = self.pattern_mapper.learn(
            original_tree, transformed_tree
        )

        return original_to_transformed_tuples, score
