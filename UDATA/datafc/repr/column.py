import logging
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd

logger = logging.getLogger("myapp")


class DataType(Enum):
    TEXTUAL = 1
    NUMERIC = 2


class Column:
    def __init__(self, name: str = None, semantic_type: str = None, values=None):
        if name is None:
            name = str(uuid.uuid4())
        if semantic_type is None:
            semantic_type = name
        if values is None:
            values = []
        self.name: str = name
        self.semantic_type: str = semantic_type
        self.values: List[str] = values
        self._numeric_values: List[float] = []
        self._textual_values: List[str] = []

    @property
    def label(self) -> str:
        return self.semantic_type

    @property
    def numeric_values(self) -> List[float]:
        if self._numeric_values:
            self.analyze()
        return self._numeric_values

    @property
    def textual_values(self) -> List[float]:
        if self._numeric_values:
            self.analyze()
        return self._numeric_values

    def is_numeric(self):
        return len(self._numeric_values) * 1.0 / len(self.values) == 1

    def text(self) -> str:
        return " ".join(self.values)

    def analyze(self):
        for value in self.values:
            try:
                self._numeric_values.append(float(value))
            except ValueError:
                self._textual_values.append(value)

    def add_value(self, value: str):
        self.values.append(value)

    def extend_values(self, values: List[str]):
        self.values.extend(values)

    def is_unique(self) -> bool:
        return len(set(self.values)) == len(self.values)


class ColumnBasedSource:
    def __init__(self, name: str, name_to_attribute=None):
        if name_to_attribute is None:
            name_to_attribute = {}
        self.name: str = name
        self.name_to_attribute: Dict[str, Column] = name_to_attribute

    def add_attribute(self, attribute):
        self.name_to_attribute[attribute.name] = attribute

    def get_attribute(self, name) -> Column:
        return self.name_to_attribute[name]

    def get_all_attributes(self) -> List[Column]:
        return list(self.name_to_attribute.values())

    @staticmethod
    def from_file(path: Union[Path, str]) -> "ColumnBasedSource":
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == ".csv":
            return ColumnBasedSource.__from_csv(path)

    @staticmethod
    def __from_csv(path: Path) -> "ColumnBasedSource":
        df = pd.DataFrame.from_csv(str(path))
        source = ColumnBasedSource(path.name.title())
        for column_name in df.columns.tolist():
            attr = Column(column_name, column_name, df[column_name].tolist())
            source.add_attribute(attr)
        return source
