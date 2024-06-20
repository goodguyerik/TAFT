import logging
import os
from pathlib import Path

from datafc.utils.logging import setup_logging

setup_logging("conf/logging.yaml")

logger = logging.getLogger("myapp")

for dataset in Path("data").iterdir():
    for sub_folder in dataset.iterdir():
        logger.info("Scenario: %s %s", dataset, sub_folder)
        input_file = sub_folder / "input.csv"
        groundtruth_file = sub_folder / "groundtruth.csv"

        os.makedirs(Path("ipbe") / dataset.name, exist_ok=True)

        input_data_path = Path("ipbe") / dataset.name / f"{sub_folder.name}.csv"

        input_values = input_file.open("r", encoding="utf-8").readlines()[:1000]
        groundtruth_values = groundtruth_file.open("r", encoding="utf-8").readlines()[
            :1000
        ]

        with input_data_path.open("w", encoding="utf-8") as input_writer:
            for index in range(len(input_values)):
                input_values[index] = input_values[index].replace('"', "")
                groundtruth_values[index] = groundtruth_values[index].replace('"', "")
                input_writer.write(
                    f'"{input_values[index].strip()}","{groundtruth_values[index].strip()}"\n'
                )
