import csv
import logging
from pathlib import Path

from datafc.eval.evaluator import Evaluator
from datafc.utils.logging import setup_logging

setup_logging("conf/logging.yaml")

logger = logging.getLogger("myapp")

evaluator = Evaluator()

dataset = "nyc"

input_data_path = Path("flashfill") / dataset / "input"
output_data_path = Path("flashfill") / dataset / "output"

for input_file in input_data_path.iterdir():
    output_file = output_data_path / input_file.name

    original_values = []
    transformed_values = []
    groundtruth_values = []

    csv_reader = csv.reader(input_file.open("r", encoding="utf-8"))
    for line in csv_reader:
        original_values.append(line[0])
        transformed_values.append(line[1])

    csv_reader = csv.reader(output_file.open("r", encoding="utf-8"))
    for line in csv_reader:
        groundtruth_values.append(line[0])

    evaluator.check_flashfill_result(
        input_file.name, original_values, transformed_values, groundtruth_values
    )
    evaluator.generate_scenario_report(input_file.name, 10)

evaluator.generate_dataset_report(dataset, 10)
