import argparse
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.reproducibility import set_global_seed
from src.data import build_window_dataset, split_dataset
from src.training import run_ablation, train_full_model
from src.evaluation import evaluate_model
from src.spectral import analyze_misclassifications, confidence_summary, analyze_frequency_weights, weighted_rfft_activations
from src.plots import configure_plots, plot_ablation, plot_confusion, plot_roc, plot_pr, plot_confidence, plot_training, plot_weight_alignment, plot_class_spectra, plot_activations, plot_misclassified_examples
from src.io_utils import ensure_directories, save_json, save_matrix


def main(config_path: str):
    config = load_config(config_path)
    set_global_seed(config.seed)
    configure_plots()
    figures_dir, tables_dir = ensure_directories(config.output_directory)
    dataset = build_window_dataset(config)
    split = split_dataset(dataset, config)
    dataset.inventory.to_csv(tables_dir / "window_inventory.csv", index=False)
    protocol = {
        "seed": config.seed,
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "window_size": config.window_size,
        "stride": config.stride,
        "test_fraction": config.test_size,
        "validation_fraction_within_development": config.validation_size,
        "cv_folds": config.cv_folds,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "early_stopping_patience": config.patience,
        "learning_rate": config.learning_rate,
        "class_mapping": {str(i): name for i, name in enumerate(config.class_names)},
        "raw_recording_lengths": dataset.raw_lengths,
        "total_windows": len(dataset.x),
        "development_windows": len(split.x_dev),
        "test_windows": len(split.x_test),
    }
    save_json(protocol, config.output_directory / "experimental_protocol.json")
    folds, summary, pairwise = run_ablation(split.x_dev, split.y_dev, config)
    folds.to_csv(tables_dir / "ablation_fold_results.csv", index=False)
    summary.to_csv(tables_dir / "ablation_summary.csv", index=False)
    pairwise.to_csv(tables_dir / "ablation_pairwise_vs_full.csv", index=False)
    plot_ablation(summary, figures_dir)
    model, history, fit_info = train_full_model(split.x_dev, split.y_dev, split.idx_dev, config, config.output_directory)
    history.to_csv(tables_dir / "full_model_training_history.csv", index=False)
    save_json(fit_info, config.output_directory / "fit_validation_split.json")
    evaluation = evaluate_model(model, split.x_test, split.y_test, split.idx_test, config.class_names)
    save_json(evaluation["metrics"], config.output_directory / "test_metrics.json")
    evaluation["predictions_frame"].to_csv(tables_dir / "test_predictions.csv", index=False)
    evaluation["classification_report"].to_csv(tables_dir / "classification_report.csv")
    evaluation["roc_points"].to_csv(tables_dir / "roc_curve_points.csv", index=False)
    evaluation["pr_points"].to_csv(tables_dir / "precision_recall_curve_points.csv", index=False)
    save_matrix(evaluation["confusion_counts"], config.class_names, tables_dir / "confusion_matrix_counts.csv")
    save_matrix(evaluation["confusion_normalized"], config.class_names, tables_dir / "confusion_matrix_normalized.csv")
    plot_confusion(evaluation["confusion_counts"], evaluation["confusion_normalized"], config.class_names, figures_dir)
    plot_roc(evaluation["roc_points"], evaluation["metrics"]["macro_auc"], figures_dir)
    plot_pr(evaluation["pr_points"], evaluation["metrics"]["macro_average_precision"], figures_dir)
    plot_confidence(evaluation["confidence"], evaluation["predictions"], split.y_test, figures_dir)
    plot_training(history, figures_dir)
    misclassified, prototypes, freq_hz = analyze_misclassifications(
        split.x_test,
        split.y_test,
        split.idx_test,
        evaluation["predictions"],
        evaluation["confidence"],
        evaluation["entropy"],
        split.x_dev,
        split.y_dev,
        config.class_names,
        config.sampling_frequency_hz,
    )
    misclassified.to_csv(tables_dir / "misclassified_samples_analysis.csv", index=False)
    confidence_summary(evaluation["confidence"], evaluation["entropy"], evaluation["predictions"], split.y_test).to_csv(
        tables_dir / "correct_vs_incorrect_confidence.csv", index=False
    )
    plot_misclassified_examples(
        split.x_test,
        split.y_test,
        evaluation["predictions"],
        evaluation["confidence"],
        prototypes,
        freq_hz,
        config.sampling_frequency_hz,
        config.class_names,
        figures_dir,
    )
    frequency = analyze_frequency_weights(model, split.x_dev, split.y_dev, config.class_names, config.sampling_frequency_hz)
    frequency["frequency_frame"].to_csv(tables_dir / "frequency_physics_analysis.csv", index=False)
    frequency["top_frequencies"].to_csv(tables_dir / "top_learned_frequencies.csv", index=False)
    frequency["dominant_frequencies"].to_csv(tables_dir / "class_dominant_frequencies.csv", index=False)
    save_json(frequency["statistics"], config.output_directory / "frequency_physics_statistics.json")
    activations, class_means = weighted_rfft_activations(model, split.x_test, split.y_test, config.class_names, frequency["freq_hz"])
    activations.to_csv(tables_dir / "class_specific_physics_activations.csv", index=False)
    plot_weight_alignment(frequency, figures_dir)
    plot_class_spectra(frequency, config.class_names, figures_dir)
    plot_activations(frequency["freq_hz"], class_means, figures_dir)
    summary_line = pd.DataFrame([evaluation["metrics"]])
    summary_line.to_csv(tables_dir / "test_metric_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(summary_line.to_string(index=False))
    print(pd.DataFrame([frequency["statistics"]]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
