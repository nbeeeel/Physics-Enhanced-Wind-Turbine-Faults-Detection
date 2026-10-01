from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    sampling_frequency_hz: int
    window_size: int
    stride: int
    test_size: float
    validation_size: float
    cv_folds: int
    epochs: int
    batch_size: int
    patience: int
    learning_rate: float
    class_names: tuple[str, ...]
    data_directory: Path
    data_files: dict[str, str]
    output_directory: Path


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    root = path.parent
    raw = yaml.safe_load(path.read_text())
    return ExperimentConfig(
        seed=int(raw["seed"]),
        sampling_frequency_hz=int(raw["sampling_frequency_hz"]),
        window_size=int(raw["window_size"]),
        stride=int(raw["stride"]),
        test_size=float(raw["test_size"]),
        validation_size=float(raw["validation_size"]),
        cv_folds=int(raw["cv_folds"]),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        patience=int(raw["patience"]),
        learning_rate=float(raw["learning_rate"]),
        class_names=tuple(raw["class_names"]),
        data_directory=(root / raw["data"]["directory"]).resolve(),
        data_files=dict(raw["data"]["files"]),
        output_directory=(root / raw["output_directory"]).resolve(),
    )
