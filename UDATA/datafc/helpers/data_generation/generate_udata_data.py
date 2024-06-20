import logging
import shutil
from pathlib import Path

logger = logging.getLogger("myapp")


def convert_new_structure(folder_path):
    path = Path(folder_path)

    temp_path = path / "temp"

    if (
        not (path / "groundtruth").exists()
        or not (path / "input" / "raw").exists()
        or not (path / "input" / "transformed")
    ):
        logger.error("Wrong folder structure")
        return False

    for groundtruth_path in (path / "groundtruth").iterdir():
        input_raw_path = path / "input" / "raw" / groundtruth_path.name
        transformed_raw_path = path / "input" / "transformed" / groundtruth_path.name

        if input_raw_path.exists() and transformed_raw_path.exists():
            new_folder_path = temp_path / groundtruth_path.stem
            new_folder_path.mkdir(parents=True, exist_ok=True)

            shutil.copy(str(input_raw_path), str(new_folder_path / "input.csv"))
            shutil.copy(
                str(transformed_raw_path), str(new_folder_path / "transformed.csv")
            )
            shutil.copy(str(groundtruth_path), str(new_folder_path / "groundtruth.csv"))


if __name__ == "__main__":
    convert_new_structure("data/new_museum")
