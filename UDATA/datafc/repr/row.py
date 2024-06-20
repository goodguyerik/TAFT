from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from datafc.utils import to_number


class Row:
    def __init__(self, attribute_to_value=None):
        if attribute_to_value is None:
            attribute_to_value = {}
        self.attribute_to_value: Dict[str, str] = attribute_to_value

    def add_value(self, name: str, value: str):
        self.attribute_to_value[name] = value

    def __getitem__(self, name: str):
        return self.attribute_to_value[name]

    def full_text(self):
        text = ""
        for name in self.attribute_to_value.keys():
            if to_number(text) is not None:
                text += self[name]
        return text


class RowBasedSource:
    def __init__(self, name):
        self.name = name
        self.entries: List[Row] = []

    def add_entry(self, row: Row):
        self.entries.append(row)

    @staticmethod
    def from_file(path: Union[Path, str]) -> "RowBasedSource":
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == ".csv":
            return RowBasedSource.__from_csv(path)

    @staticmethod
    def __from_csv(path: Path) -> "RowBasedSource":
        df = pd.read_csv(str(path))
        source = RowBasedSource(path.name.title())
        for index, df_row in df.iterrows():
            row = Row()
            for column_name in df.columns.tolist():
                row.add_value(column_name, str(df_row[column_name]))
            source.add_entry(row)
        return source
