import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Union

import pandas as pd
import ujson as json
import win32com.client

from datafc.eval.result import TopKTransformationResult, Example, ValidationResult
from datafc.trans.model import TransformationModel

logger = logging.getLogger("myapp")


class Evaluator:
    def __init__(
        self,
        mapping_method="sim",
        mapping_features=None,
        with_flashfill=False,
        num_examples=1000,
        k=10,
    ):
        if mapping_features is None:
            mapping_features = ["jaccard", "syn"]
        self.name_to_result: Dict[str, TopKTransformationResult] = {}

        self.num_examples = num_examples
        self.k = k

        self.mapping_method = mapping_method
        self.with_flashfill = with_flashfill
        self.mapping_features = mapping_features

    def check_validation_result(
        self,
        name,
        original_value,
        groundtruth_value,
        transformed_values,
        validation_result,
    ):
        if transformed_values[0] == groundtruth_value:
            if validation_result[0]:
                self.name_to_result[name].validation_fp_count += 1
                result = ValidationResult.FP
                example = Example(
                    original_value,
                    groundtruth_value,
                    transformed_values,
                    result,
                    validation_result[1:],
                )
                self.name_to_result[name].top_k_failed_validations.append(example)
            else:
                self.name_to_result[name].validation_tn_count += 1
                result = ValidationResult.TN
        else:
            if validation_result[0]:
                self.name_to_result[name].validation_tp_count += 1
                result = ValidationResult.TP
            else:
                self.name_to_result[name].validation_fn_count += 1
                result = ValidationResult.FN
                example = Example(
                    original_value,
                    groundtruth_value,
                    transformed_values,
                    result,
                    validation_result[1:],
                )
                self.name_to_result[name].top_k_failed_validations.append(example)

        return result, validation_result[1:]

    def check_top_k_transformation_result(
        self,
        name: Union[str, Tuple[str, int]],
        original_to_groundtruth_values: Dict[str, str],
        original_to_k_transformed_values: List[Tuple[str, List[str], List]],
    ):
        assert isinstance(
            name, str
        ), "Nonactive learning result requires name as string"
        tran_result = self.name_to_result[name]

        for (
            original_value,
            transformed_values,
            validated_results,
        ) in original_to_k_transformed_values:
            tran_result.total_count += 1
            transformed_values = transformed_values[: self.k]

            validation_result, reasons = self.check_validation_result(
                name,
                original_value,
                original_to_groundtruth_values[original_value],
                transformed_values,
                validated_results[0],
            )

            if transformed_values[0] == original_to_groundtruth_values[original_value]:
                tran_result.correct_count += 1
                tran_result.top_k_correct_count += 1
                tran_result.mrr_count += 1
            else:
                tran_result.transformation_result = False

                example = Example(
                    original_value,
                    original_to_groundtruth_values[original_value],
                    transformed_values,
                    validation_result,
                    reasons,
                )
                tran_result.top_k_failed_transformations.append(example)
                logger.debug(
                    "Failed case: %s vs '%s' (source: '%s')"
                    % (
                        transformed_values[:3],
                        original_to_groundtruth_values[original_value],
                        original_value,
                    )
                )
                try:
                    correct_rank = transformed_values.index(
                        original_to_groundtruth_values[original_value]
                    )
                except ValueError:
                    correct_rank = -1
                if correct_rank != -1:
                    tran_result.mrr_count += 1.0 / (correct_rank + 1)
                    tran_result.top_k_correct_count += 1

    def check_flashfill_result(
        self,
        name,
        original_values: List[str],
        transformed_values: List[str],
        groundtruth_values: List[str],
    ):
        self.name_to_result[name] = TopKTransformationResult(
            name=name, method="flashfill"
        )

        original_to_groundtruth_values: Dict[str, str] = {}
        original_to_transformed_values: List[Tuple[str, List[str], List]] = []

        for original_value, transformed_value, groundtruth_value in zip(
            original_values, transformed_values, groundtruth_values
        ):
            original_to_groundtruth_values[original_value] = groundtruth_value
            original_to_transformed_values.append(
                (original_value, [transformed_value], [(True, 1.0)])
            )

        self.check_top_k_transformation_result(
            name, original_to_groundtruth_values, original_to_transformed_values
        )

    def run_top_k_experiment(
        self,
        name: str,
        original_values: List[str],
        target_values: List[str],
        groundtruth_values: List[str],
        k: int,
    ):
        self.name_to_result[name] = TopKTransformationResult(name=name, method="udata")

        original_to_groundtruth_values: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            original_to_groundtruth_values[value] = groundtruth_values[index]

        starting_time = time.time()

        transformation_model = TransformationModel(
            self.mapping_method, self.mapping_features
        )

        validated_original_to_k_transformed_values, scores, full_validation = transformation_model.learn_top_k(
            original_values, target_values, k
        )

        self.name_to_result[name].running_time = time.time() - starting_time
        assert len(original_values) == len(
            validated_original_to_k_transformed_values
        ), (
            f"Dataset sizes before and after transformation "
            f"should be the same ({len(original_values)} vs {len(validated_original_to_k_transformed_values)}) "
        )

        self.check_top_k_transformation_result(
            name,
            original_to_groundtruth_values,
            validated_original_to_k_transformed_values,
        )

        if self.name_to_result[name].transformation_result:
            if full_validation:
                self.name_to_result[name].validation_result = ValidationResult.FP
            else:
                self.name_to_result[name].validation_result = ValidationResult.TN
        else:
            if full_validation:
                self.name_to_result[name].validation_result = ValidationResult.TP
            else:
                self.name_to_result[name].validation_result = ValidationResult.FN

        # print(self.name_to_result[name].validation_result)
        self.name_to_result[name].top_k_failed_validations = self.name_to_result[
            name
        ].top_k_failed_validations[:100]
        self.name_to_result[name].top_k_failed_transformations = self.name_to_result[
            name
        ].top_k_failed_transformations[:100]

    def run_scenario(self, scenario_folder):
        original_values = []
        target_values = []
        groundtruth_values = []

        for file in scenario_folder.iterdir():
            with file.open(encoding="utf-8") as reader:

                for row in reader.readlines():
                    row = row.encode("utf-8").decode("ascii", "ignore")
                    if "input" in file.name:
                        original_values.append(row.strip())
                    if "transformed" in file.name:
                        target_values.append(row.strip())
                    if "groundtruth" in file.name:
                        groundtruth_values.append(row.strip())

        self.run_top_k_experiment(
            scenario_folder.name,
            original_values[: self.num_examples],
            target_values[: self.num_examples],
            groundtruth_values[: self.num_examples],
            1,
        )

    def run_udata_dataset(self, dataset_folder):
        dataset_folder = Path(dataset_folder)

        for scenario_folder in list(dataset_folder.iterdir()):
            self.run_scenario(scenario_folder)

        return [x.__dict__ for x in self.name_to_result.values()]

    def run_ipbe_dataset(self, dataset_folder):
        name_to_stats = json.load(
            open(dataset_folder / ".." / "result" / f"{dataset_folder.name}.json", "r")
        )

        for name, stats in name_to_stats.items():
            self.name_to_result[name] = TopKTransformationResult(
                name=name, method="ipbe"
            )
            self.name_to_result[name].total_count = 1
            self.name_to_result[name].correct_count = stats[12]
            self.name_to_result[name].running_time = (
                stats[4] * 1.0 / len(name_to_stats) / 10
            )

        return [x.__dict__ for x in self.name_to_result.values()]

    def run_flashfill_dataset(self, dataset_folder):
        dataset_folder = Path(dataset_folder)

        for scenario_input_file in list((dataset_folder / "input").iterdir()):
            xl = win32com.client.DispatchEx("Excel.Application")
            xl.Workbooks.Open(str(Path("../lib/PERSONAL.XLSB").absolute()))
            xl.Workbooks.Open(str(scenario_input_file.absolute()))
            time1 = time.time()
            xl.Application.Run("PERSONAL.XLSB!FF")
            xl.Application.Quit()
            del xl

            running_time = time.time() - time1

            groundtruth_values = (
                pd.read_csv(dataset_folder / "groundtruth" / scenario_input_file.name)
                .iloc[:, 0]
                .tolist()
            )

            input_values = (
                pd.read_csv(dataset_folder / "input" / scenario_input_file.name)
                .iloc[:, 0]
                .tolist()
            )

            transformed_values = (
                pd.read_csv(dataset_folder / "result" / scenario_input_file.name)
                .iloc[:, 1]
                .tolist()
            )

            self.check_flashfill_result(
                scenario_input_file.name,
                input_values,
                transformed_values,
                groundtruth_values,
            )

            self.name_to_result[scenario_input_file.name].running_time = running_time
        return [x.__dict__ for x in self.name_to_result.values()]

    def run_dataset(self, dataset_folder, method="udata"):
        if method == "udata":
            return self.run_udata_dataset(dataset_folder)
        if method == "flashfill":
            return self.run_flashfill_dataset(dataset_folder)
        if method == "ipbe":
            return self.run_ipbe_dataset(dataset_folder)
        raise ValueError(f"Method {method} is currently not supported")
