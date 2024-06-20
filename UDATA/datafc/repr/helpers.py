from typing import List


def create_histogram_data(values: List[str]) -> List[int]:
    distinct_values = set(values)

    return [list(distinct_values).index(val) for val in values]


def create_ngrams(string_value: str, n: int) -> List[str]:
    return [string_value[i : i + n] for i in range(len(string_value) - n)]
