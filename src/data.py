from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import ExperimentConfig

@dataclass(frozen=True)
class WindowDataset:
    x: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    inventory: pd.DataFrame
    raw_lengths: dict[str, int]

@dataclass(frozen=True)
class DataSplit:
    x_dev: np.ndarray
    x_test: np.ndarray
    y_dev: np.ndarray
    y_test: np.ndarray
    idx_dev: np.ndarray
    idx_test: np.ndarray


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    combined = [c for c in frame.columns if ";" in str(c)]
    if combined:
        parts = frame[combined[0]].astype(str).str.split(";", n=1, expand=True)
        frame["Time"] = parts[0]
        frame["Amplitude"] = parts[1]
    elif "Time - Voltage_1" in frame.columns and "Amplitude - Voltage_1" in frame.columns:
        frame = frame.rename(columns={"Time - Voltage_1": "Time", "Amplitude - Voltage_1": "Amplitude"})
    elif "Time - sec" in frame.columns and "Amplitude - g" in frame.columns:
        frame = frame.rename(columns={"Time - sec": "Time", "Amplitude - g": "Amplitude"})
    elif "Time" not in frame.columns or "Amplitude" not in frame.columns:
        raise ValueError(f"Unsupported signal columns: {list(frame.columns)}")
    frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
    frame["Amplitude"] = pd.to_numeric(frame["Amplitude"], errors="coerce")
    return frame.dropna(subset=["Amplitude"]).reset_index(drop=True)[["Time", "Amplitude"]]


def load_signals(config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    signals = {}
    for class_name in config.class_names:
        path = config.data_directory / config.data_files[class_name]
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        signals[class_name] = _normalize_columns(_read_frame(path))
    return signals


def build_window_dataset(config: ExperimentConfig) -> WindowDataset:
    signals = load_signals(config)
    windows = []
    labels = []
    classes = []
    starts_all = []
    raw_lengths = {}
    for label, class_name in enumerate(config.class_names):
        signal = signals[class_name]["Amplitude"].to_numpy(dtype=np.float32)
        raw_lengths[class_name] = len(signal)
        starts = np.arange(0, len(signal) - config.window_size + 1, config.stride, dtype=np.int64)
        class_windows = np.stack([signal[s:s + config.window_size] for s in starts]).astype(np.float32)
        windows.append(class_windows)
        labels.append(np.full(len(class_windows), label, dtype=np.int64))
        classes.extend([class_name] * len(class_windows))
        starts_all.extend(starts.tolist())
    x = np.vstack(windows).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    indices = np.arange(len(y), dtype=np.int64)
    starts_array = np.asarray(starts_all, dtype=np.int64)
    inventory = pd.DataFrame({
        "global_index": indices,
        "source_class": classes,
        "label": y,
        "class_name": [config.class_names[i] for i in y],
        "window_start": starts_array,
        "window_end": starts_array + config.window_size - 1,
    })
    return WindowDataset(x=x, y=y, indices=indices, inventory=inventory, raw_lengths=raw_lengths)


def split_dataset(dataset: WindowDataset, config: ExperimentConfig) -> DataSplit:
    x_dev, x_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
        dataset.x,
        dataset.y,
        dataset.indices,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=dataset.y,
    )
    return DataSplit(
        x_dev=x_dev,
        x_test=x_test,
        y_dev=y_dev,
        y_test=y_test,
        idx_dev=idx_dev,
        idx_test=idx_test,
    )
