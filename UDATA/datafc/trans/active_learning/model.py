from typing import Tuple, List


class ActiveLearner:
    def __init__(self, top_k_examples: List[List[Tuple[str, str]]]):
        self.top_k_examples = top_k_examples

    def choose_active_example(self):
        pass

    def learn(self):
        while len(self.top_k_examples) >= 1:
            best_example = self.choose_active_example()
