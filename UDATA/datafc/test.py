from pathlib import Path

from datafc.eval import Evaluator
from datafc.eval.report import DatasetReport

if __name__ == "__main__":
    mapping_method = "sim"
    mapping_features = ["jaccard", "syn", "token_jaccard"]
    with_flashfill = False
    num_example = 1000
    k = 1
    evaluator = Evaluator(
        mapping_method, mapping_features, with_flashfill, num_example, k
    )

    result = evaluator.run_udata_dataset(Path("../data/udata/ijcai"))

    dataset_report = DatasetReport("udata", result)

    print(dataset_report.macro_mean_accuracy())
    print(dataset_report.scenario_valid_r())

    for name, scenario_report in dataset_report.scenario_to_report.items():
        print(name, scenario_report.micro_accuracy())
        print(name, scenario_report.validation_result)
