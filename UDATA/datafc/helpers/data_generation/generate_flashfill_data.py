import logging
import os
from pathlib import Path

import numpy as np


from datafc.utils.logging import setup_logging

setup_logging("conf/logging.yaml")

logger = logging.getLogger("myapp")

for dataset in Path("../../../data/udata").iterdir():
    for sub_folder in dataset.iterdir():
        logger.info("Scenario %s %s" % (dataset, sub_folder))
        input_file = sub_folder / "input.csv"
        groundtruth_file = sub_folder / "groundtruth.csv"

        os.makedirs(
            Path("../../../data/flashfill") / dataset.name / "input", exist_ok=True
        )
        os.makedirs(
            Path("../../../data/flashfill") / dataset.name / "groundtruth",
            exist_ok=True,
        )

        input_data_path = (
            Path("../../../data/flashfill")
            / dataset.name
            / "input"
            / f"{sub_folder.name}.csv"
        )
        output_data_path = (
            Path("../../../data/flashfill")
            / dataset.name
            / "groundtruth"
            / f"{sub_folder.name}.csv"
        )

        input_values = input_file.open("r", encoding="utf-8").readlines()[:1000]
        groundtruth_values = groundtruth_file.open("r", encoding="utf-8").readlines()[
            :1000
        ]

        example_size = len(input_values) // 2

        sampled_indices = np.random.choice(range(len(input_values)), example_size)

        with input_data_path.open("w", encoding="utf-8") as input_writer:
            with output_data_path.open("w", encoding="utf-8") as output_writer:
                for index in range(len(input_values)):
                    input_values[index] = input_values[index].replace('"', "")
                    groundtruth_values[index] = groundtruth_values[index].replace(
                        '"', ""
                    )
                    if index in sampled_indices:
                        if input_values[index].strip():
                            input_writer.write(
                                f'"{input_values[index].strip()}","{groundtruth_values[index].strip()}"\n'
                            )
                        else:
                            input_writer.write(
                                f'"empty","{groundtruth_values[index].strip()}"\n'
                            )
                    else:
                        input_writer.write(f'"{input_values[index].strip()}",""\n')

                    output_writer.write(f'"{groundtruth_values[index].strip()}"\n')
