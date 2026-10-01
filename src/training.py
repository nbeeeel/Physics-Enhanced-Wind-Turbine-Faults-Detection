from pathlib import Path
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from .config import ExperimentConfig
from .models import VARIANTS, build_model


def run_ablation(x_dev, y_dev, config: ExperimentConfig):
    splitter = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    splits = list(splitter.split(x_dev, y_dev))
    rows = []
    accuracy_by_variant = {variant: [] for variant in VARIANTS}
    for variant_id, variant in enumerate(VARIANTS):
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            tf.keras.backend.clear_session()
            gc.collect()
            tf.keras.utils.set_random_seed(config.seed + variant_id * 100 + fold)
            model = build_model(variant, config.window_size, len(config.class_names), config.learning_rate)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
                verbose=0,
            )
            history = model.fit(
                x_dev[train_idx],
                y_dev[train_idx],
                validation_data=(x_dev[val_idx], y_dev[val_idx]),
                epochs=config.epochs,
                batch_size=config.batch_size,
                verbose=0,
                callbacks=[early_stopping],
            )
            probabilities = model.predict(x_dev[val_idx], verbose=0)
            predictions = np.argmax(probabilities, axis=1)
            rows.append({
                "Variant": variant,
                "Fold": fold,
                "Accuracy": accuracy_score(y_dev[val_idx], predictions),
                "MacroPrecision": precision_score(y_dev[val_idx], predictions, average="macro", zero_division=0),
                "MacroRecall": recall_score(y_dev[val_idx], predictions, average="macro", zero_division=0),
                "MacroF1": f1_score(y_dev[val_idx], predictions, average="macro", zero_division=0),
                "BestEpoch": int(np.argmin(history.history["val_loss"]) + 1),
                "Parameters": int(model.count_params()),
            })
            accuracy_by_variant[variant].append(rows[-1]["Accuracy"])
            del model, history
            tf.keras.backend.clear_session()
            gc.collect()
    folds = pd.DataFrame(rows)
    summary = folds.groupby("Variant").agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std"),
        MacroPrecision_Mean=("MacroPrecision", "mean"),
        MacroPrecision_Std=("MacroPrecision", "std"),
        MacroRecall_Mean=("MacroRecall", "mean"),
        MacroRecall_Std=("MacroRecall", "std"),
        MacroF1_Mean=("MacroF1", "mean"),
        MacroF1_Std=("MacroF1", "std"),
        Parameters=("Parameters", "first"),
    ).reset_index()
    order = {variant: i for i, variant in enumerate(VARIANTS)}
    summary["order"] = summary["Variant"].map(order)
    summary = summary.sort_values("order").drop(columns="order")
    full_scores = np.asarray(accuracy_by_variant["WeightedRFFT_Hybrid"])
    pairwise = []
    for variant in VARIANTS[:-1]:
        other = np.asarray(accuracy_by_variant[variant])
        try:
            statistic, p_value = wilcoxon(full_scores, other, zero_method="zsplit")
        except Exception:
            statistic, p_value = np.nan, np.nan
        pairwise.append({
            "Compared_to_Full": variant,
            "Full_Accuracy_Mean": full_scores.mean(),
            "Other_Accuracy_Mean": other.mean(),
            "Mean_Accuracy_Gain": (full_scores - other).mean(),
            "Wilcoxon_Statistic": statistic,
            "Wilcoxon_p": p_value,
        })
    return folds, summary, pd.DataFrame(pairwise)


def train_full_model(x_dev, y_dev, idx_dev, config: ExperimentConfig, output_directory: Path):
    x_fit, x_val, y_fit, y_val, idx_fit, idx_val = train_test_split(
        x_dev,
        y_dev,
        idx_dev,
        test_size=config.validation_size,
        random_state=config.seed,
        stratify=y_dev,
    )
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(config.seed)
    model = build_model("WeightedRFFT_Hybrid", config.window_size, len(config.class_names), config.learning_rate)
    weight_path = output_directory / "WeightedRFFT_Hybrid_best.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weight_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
        callbacks=callbacks,
    )
    model.load_weights(weight_path)
    return model, pd.DataFrame(history.history), {
        "fit_indices": idx_fit,
        "validation_indices": idx_val,
        "fit_samples": len(x_fit),
        "validation_samples": len(x_val),
    }
