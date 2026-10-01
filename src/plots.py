from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def configure_plots():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11.5,
        "axes.linewidth": 1.15,
        "lines.linewidth": 2.2,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _polish(ax, axis="y"):
    ax.tick_params(axis="both", which="major", width=1.0, length=4, pad=4)
    if axis:
        ax.grid(True, axis=axis, linestyle="--", linewidth=0.7, alpha=0.22)
    ax.set_axisbelow(True)


def _save(fig, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_ablation(summary, directory: Path):
    labels = [
        "Plain MLP",
        "Residual MLP",
        "Fixed RFFT + Res.",
        "Weighted RFFT + Res.",
        "Fixed RFFT + Hybrid",
        "Weighted RFFT + Hybrid\n(no residual)",
        "Full weighted RFFT + Hybrid",
    ]
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.errorbar(summary["Accuracy_Mean"], y - 0.10, xerr=summary["Accuracy_Std"], fmt="o", capsize=4, label="Accuracy")
    ax.errorbar(summary["MacroF1_Mean"], y + 0.10, xerr=summary["MacroF1_Std"], fmt="s", capsize=4, label="Macro-F1")
    for i, row in summary.reset_index(drop=True).iterrows():
        ax.text(min(row["Accuracy_Mean"] + row["Accuracy_Std"] + 0.012, 1.015), i - 0.10, f'{row["Accuracy_Mean"]:.3f}', va="center", fontsize=11.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, min(summary["Accuracy_Mean"].min(), summary["MacroF1_Mean"].min()) - 0.08), 1.03)
    ax.set_xlabel("5-fold cross-validation score")
    ax.set_title("Component-wise ablation study")
    ax.legend(loc="lower right", ncol=2)
    _polish(ax, "x")
    fig.tight_layout()
    _save(fig, directory, "ablation_performance_compact")


def plot_confusion(counts, normalized, class_names, directory: Path):
    fig, ax = plt.subplots(figsize=(5.7, 4.9))
    image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Recall by true class")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=28, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = normalized[i, j]
            ax.text(j, i, f"{100 * value:.0f}%\n(n={counts[i, j]})", ha="center", va="center", color="white" if value >= 0.55 else "black")
    fig.tight_layout()
    _save(fig, directory, "confusion_matrix_normalized")


def plot_roc(roc_points, macro_auc, directory: Path):
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    for class_name, group in roc_points.groupby("class", sort=False):
        ax.plot(group["fpr"], group["tpr"], label=f'{class_name}  AUC={group["auc"].iloc[0]:.3f}')
    ax.plot([0, 1], [0, 1], "--", linewidth=1.2, alpha=0.55, label="Chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC curves | macro AUC={macro_auc:.3f}")
    ax.legend(loc="lower right", fontsize=10.5)
    _polish(ax, "both")
    fig.tight_layout()
    _save(fig, directory, "roc_curves")


def plot_pr(pr_points, macro_ap, directory: Path):
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    for class_name, group in pr_points.groupby("class", sort=False):
        ax.plot(group["recall"], group["precision"], label=f'{class_name}  AP={group["ap"].iloc[0]:.3f}')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curves | macro AP={macro_ap:.3f}")
    ax.legend(loc="lower left", fontsize=10.5)
    _polish(ax, "both")
    fig.tight_layout()
    _save(fig, directory, "precision_recall_curves")


def plot_confidence(confidence, predictions, y_test, directory: Path):
    correct = predictions == y_test
    groups = [confidence[correct]]
    labels = ["Correct"]
    if np.any(~correct):
        groups.append(confidence[~correct])
        labels.append("Incorrect")
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.boxplot(groups, labels=labels, widths=0.45, showfliers=False)
    for i, values in enumerate(groups, start=1):
        x = np.full(len(values), i, dtype=float)
        ax.scatter(x, values, s=22, alpha=0.65)
    ax.set_ylabel("Maximum softmax probability")
    ax.set_ylim(0, 1.04)
    ax.set_title("Prediction confidence")
    _polish(ax, "y")
    fig.tight_layout()
    _save(fig, directory, "prediction_confidence")


def plot_training(history, directory: Path):
    epochs = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(epochs, history["accuracy"], label="Train")
    ax.plot(epochs, history["val_accuracy"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Full model accuracy")
    ax.legend()
    _polish(ax, "y")
    fig.tight_layout()
    _save(fig, directory, "full_model_accuracy_curve")
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(epochs, history["loss"], label="Train")
    ax.plot(epochs, history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Full model loss")
    ax.legend()
    _polish(ax, "y")
    fig.tight_layout()
    _save(fig, directory, "full_model_loss_curve")


def plot_weight_alignment(analysis, directory: Path):
    w = analysis["absolute_weights"] / (analysis["absolute_weights"].max() + 1e-12)
    f = analysis["fisher_score"] / (analysis["fisher_score"].max() + 1e-12)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(analysis["freq_hz"], w, label="|Learned frequency weight|")
    ax.plot(analysis["freq_hz"], f, label="Spectral class-separability score")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title("Learned frequency weights and spectral separability")
    ax.legend()
    _polish(ax, "both")
    fig.tight_layout()
    _save(fig, directory, "learned_weights_vs_spectral_separability")


def plot_class_spectra(analysis, class_names, directory: Path):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for c, name in enumerate(class_names):
        spectrum = analysis["class_mean_spectra"][c]
        ax.plot(analysis["freq_hz"], spectrum / (spectrum.max() + 1e-12), label=f"{name} mean FFT")
    weights = analysis["absolute_weights"] / (analysis["absolute_weights"].max() + 1e-12)
    ax.plot(analysis["freq_hz"], weights, "--", linewidth=2.8, label="|Learned weight|")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title("Class-average spectra and learned frequency weights")
    ax.legend(ncol=2, fontsize=10)
    _polish(ax, "both")
    fig.tight_layout()
    _save(fig, directory, "class_spectra_with_learned_weights")


def plot_activations(freq_hz, class_means, directory: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name, values in class_means.items():
        ax.plot(freq_hz, values, label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean weighted-RFFT activation")
    ax.set_title("Class-specific weighted-RFFT activations")
    ax.legend()
    _polish(ax, "both")
    fig.tight_layout()
    _save(fig, directory, "class_specific_physics_activations")


def plot_misclassified_examples(x_test, y_test, predictions, confidence, prototypes, freq_hz, sampling_frequency_hz, class_names, directory: Path):
    positions = np.where(predictions != y_test)[0]
    for rank, position in enumerate(positions[:10], start=1):
        true_class = int(y_test[position])
        predicted_class = int(predictions[position])
        spectrum = np.abs(np.fft.rfft(x_test[position]))
        spectrum = spectrum / (spectrum.max() + 1e-12)
        true_prototype = prototypes[true_class] / (prototypes[true_class].max() + 1e-12)
        predicted_prototype = prototypes[predicted_class] / (prototypes[predicted_class].max() + 1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
        axes[0].plot(np.arange(x_test.shape[1]) / sampling_frequency_hz, x_test[position])
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"True={class_names[true_class]}, Pred={class_names[predicted_class]}")
        _polish(axes[0], "both")
        axes[1].plot(freq_hz, spectrum, label="Sample")
        axes[1].plot(freq_hz, true_prototype, label=f"True prototype: {class_names[true_class]}")
        axes[1].plot(freq_hz, predicted_prototype, label=f"Pred. prototype: {class_names[predicted_class]}")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Normalized FFT magnitude")
        axes[1].set_title(f"Confidence={confidence[position]:.3f}")
        axes[1].legend(fontsize=9)
        _polish(axes[1], "both")
        fig.tight_layout()
        _save(fig, directory, f"misclassified_example_{rank}")
