from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union


@dataclass
class Example:
    original_value: str
    groundtruth_value: str
    transformed_values: List[str]
    validation_result: "ValidationResult"
    validation_components: List[Union[bool, float]]


class ValidationResult:
    TP = "TP"
    FP = "FP"
    TN = "TN"
    FN = "FN"


@dataclass
class TopKTransformationResult:
    name: str = ""
    method: str = ""
    total_count: float = 0

    correct_count: float = 0
    top_k_correct_count: float = 0

    mrr_count: float = 0
    running_time: float = 0

    top_k_failed_transformations: List[Example] = field(default_factory=list)
    top_k_failed_validations: List[Example] = field(default_factory=list)

    validation_tp_count: int = 0
    validation_tn_count: int = 0
    validation_fp_count: int = 0
    validation_fn_count: int = 0

    transformation_result: bool = True

    validation_result: ValidationResult = ValidationResult.TN
