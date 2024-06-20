from typing import Dict, Union, List
import numpy as np
import pandas as pd

from datafc.eval.result import TopKTransformationResult, ValidationResult


class ScenarioReport:
    def __init__(self, scenario_name, scenario_result: Dict):
        self.name = scenario_name
        self._result = TopKTransformationResult(**scenario_result)

    @property
    def failed_transformations(self):
        return self._result.top_k_failed_transformations

    @property
    def failed_validations(self):
        return self._result.top_k_failed_validations

    @property
    def length(self):
        return self._result.total_count

    @property
    def running_time(self):
        return self._result.running_time

    @property
    def validation_result(self):
        return self._result.validation_result

    @property
    def transformation_result(self):
        return self._result.transformation_result

    def micro_accuracy(self):
        return self._result.correct_count * 1.0 / self._result.total_count

    def micro_top_k_accuracy(self):
        return self._result.top_k_correct_count * 1.0 / self._result.total_count

    def mrr_score(self):
        return self._result.mrr_count / self._result.total_count

    def validation_precision(self):
        if (
            self._result.validation_tp_count == 0
            and self._result.validation_fp_count == 0
        ):
            return 1.0
        try:
            return (
                self._result.validation_tp_count
                * 1.0
                / (self._result.validation_tp_count + self._result.validation_fp_count)
            )
        except ZeroDivisionError:
            return 0.0

    def validation_recall(self):
        if (
            self._result.validation_tp_count == 0
            and self._result.validation_fn_count == 0
        ):
            return 1.0
        try:
            return (
                self._result.validation_tp_count
                * 1.0
                / (self._result.validation_tp_count + self._result.validation_fn_count)
            )
        except ZeroDivisionError:
            return 0.0

    def validation_f1(self):
        precision = self.validation_precision()
        recall = self.validation_recall()
        try:
            return (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def generate_report_frame(reports):
        reports = [report.generate_report() for report in reports]
        df = pd.DataFrame(reports, columns=reports[0].keys())
        return df

    def generate_report(self):
        report = {
            "name": self.name,
            "micro_acc": self.micro_accuracy(),
            "micro_top_k_acc": self.micro_top_k_accuracy(),
            "validation_p": self.validation_precision(),
            "validation_r": self.validation_recall(),
            "validation_f1": self.validation_f1(),
            "transformation_result": self.transformation_result,
            "validation_result": self.validation_result,
            "running_time": self._result.running_time,
        }
        return report


class DatasetReport:
    def __init__(self, dataset_name, dataset_result):
        self.name = dataset_name
        self.scenario_to_report: Dict[str, ScenarioReport] = {
            x["name"]: ScenarioReport(x["name"], x) for x in dataset_result
        }

    def exact_accuracy(self):
        return len(
            list(
                filter(
                    lambda x: x,
                    [
                        report.transformation_result
                        for report in self.scenario_to_report.values()
                    ],
                )
            )
        ) / len(self.scenario_to_report)

    def macro_mean_accuracy(self):
        return np.mean(
            [report.micro_accuracy() for report in self.scenario_to_report.values()]
        )

    def macro_mean_top_k_accuracy(self):
        return np.mean(
            [report.micro_accuracy() for report in self.scenario_to_report.values()]
        )

    def macro_mrr_score(self):
        return np.mean(
            [report.mrr_score() for report in self.scenario_to_report.values()]
        )

    def _validation_count(self, values):
        return len(
            [
                report.validation_result
                for report in self.scenario_to_report.values()
                if report.validation_result in values
            ]
        )

    def scenario_valid_p(self):
        try:
            return self._validation_count(
                [ValidationResult.TP]
            ) / self._validation_count([ValidationResult.TP, ValidationResult.FP])
        except ZeroDivisionError:
            if self._validation_count([ValidationResult.FP, ValidationResult.TP]) == 0:
                return 1
            return 0

    def scenario_valid_r(self):
        try:
            return self._validation_count(
                [ValidationResult.TP]
            ) / self._validation_count([ValidationResult.TP, ValidationResult.FN])
        except ZeroDivisionError:
            if self._validation_count([ValidationResult.FN, ValidationResult.TP]) == 0:
                return 1
            return 0

    def scenario_valid_f1(self):
        precision = self.scenario_valid_p()
        recall = self.scenario_valid_r()
        try:
            return (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0

    def mean_validation_precision(self):
        return np.mean(
            [
                report.validation_precision()
                for report in self.scenario_to_report.values()
            ]
        )

    def mean_validation_recall(self):
        return np.mean(
            [report.validation_recall() for report in self.scenario_to_report.values()]
        )

    def mean_validation_f1(self):
        return np.mean(
            [report.validation_f1() for report in self.scenario_to_report.values()]
        )

    def mean_running_time(self):
        return np.mean([x.running_time for x in self.scenario_to_report.values()])

    @staticmethod
    def generate_report_frame(reports: List["DatasetReport"]):
        reports = [report.generate_report() for report in reports]
        df = pd.DataFrame(reports, columns=reports[0].keys())
        return df

    def generate_report(self):
        report = {
            "name": self.name,
            "num_scenarios": len(self.scenario_to_report),
            "num_correct": len(
                [
                    x
                    for x in self.scenario_to_report.values()
                    if x.micro_accuracy() == 1.0
                ]
            ),
            "macro_mean_acc": self.macro_mean_accuracy(),
            "macro_top_k_acc": self.macro_mean_top_k_accuracy(),
            "example_valid_p": self.mean_validation_precision(),
            "example_valid_r": self.mean_validation_recall(),
            "example_valid_f1": self.mean_validation_f1(),
            "mean_running_time": self.mean_running_time(),
            "s_valid_p": self.scenario_valid_p(),
            "s_valid_r": self.scenario_valid_r(),
            "s_valid_f1": self.scenario_valid_f1(),
            "valid_accuracy": self._validation_count(
                [ValidationResult.TN, ValidationResult.TP]
            )
            * 1.0
            / len(self.scenario_to_report),
        }

        # for threshold in [0.99, 0.95, 0.9]:
        #     report.update(
        #         {
        #             "num_%f_correct"
        #             % threshold: len(
        #                 [
        #                     x
        #                     for x in self.scenario_to_report.values()
        #                     if threshold < x.micro_accuracy() < 1.0
        #                 ]
        #             )
        #         }
        #     )
        #
        # for num_failed_cases in [1, 5, 10]:
        #     report.update(
        #         {
        #             "num_lt_%d_wrong"
        #             % num_failed_cases: len(
        #                 [
        #                     x
        #                     for x in self.scenario_to_report.values()
        #                     if (1 - x.micro_accuracy()) * x.length < num_failed_cases
        #                 ]
        #             )
        #         }
        #     )

        return report
