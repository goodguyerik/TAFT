from typing import List, Tuple

from datafc.syn.pattern import PatternTree


class Validator:
    def __init__(self, threshold: float):
        self.threshold: float = threshold

    def validate_results(
        self,
        original_transformed_pairs: Tuple[str, str],
        target_tree: PatternTree,
        score: float,
        next_score: float,
        level: int,
    ):
        pattern_tree = PatternTree.build_from_strings(
            [x[1] for x in original_transformed_pairs]
        )

        is_syntactic = set(pattern_tree.get_patterns_by_layers([level])).issubset(
            set(target_tree.get_patterns_by_layers([level]))
        )

        print(is_syntactic, score, next_score, original_transformed_pairs)

        if is_syntactic and (score - next_score) > self.threshold:
            return False, is_syntactic, round(score, 2), round(next_score, 2)
        return True, is_syntactic, round(score, 2), round(next_score, 2)
