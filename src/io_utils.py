import json
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_directories(output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = output_directory / "figures"
    tables = output_directory / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    return figures, tables


def save_json(data, path: Path):
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value
    path.write_text(json.dumps(convert(data), indent=2))


def save_matrix(matrix, names, path: Path):
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path)
