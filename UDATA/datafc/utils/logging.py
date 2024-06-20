import logging.config
import os

import yaml


def setup_logging(path="logging.yaml", default_level=logging.INFO):
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
