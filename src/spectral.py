import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr


def spectral_prototypes(x_dev, y_dev, num_classes):
    spectra = np.abs(np.fft.rfft(x_dev, axis=1))
    spectra = spectra / (np.linalg.norm(spectra, axis=1, keepdims=True) + 1e-12)
    prototypes = np.zeros((num_classes, spectra.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        prototypes[c] = spectra[y_dev == c].mean(axis=0)
        prototypes[c] /= np.linalg.norm(prototypes[c]) + 1e-12
    return prototypes


def analyze_misclassifications(x_test, y_test, idx_test, predictions, confidence, entropy, x_dev, y_dev, class_names, sampling_frequency_hz):
    num_classes = len(class_names)
    freq_hz = np.fft.rfftfreq(x_test.shape[1], d=1.0 / sampling_frequency_hz)
    prototypes = spectral_prototypes(x_dev, y_dev, num_classes)
    rows = []
    for test_position in np.where(predictions != y_test)[0]:
        spectrum = np.abs(np.fft.rfft(x_test[test_position]))
        spectrum_unit = spectrum / (np.linalg.norm(spectrum) + 1e-12)
        distances = np.linalg.norm(prototypes - spectrum_unit[None, :], axis=1)
        nearest = int(np.argmin(distances))
        non_dc = spectrum.copy()
        non_dc[0] = 0
        top_bins = np.argsort(non_dc)[-3:][::-1]
        true_class = int(y_test[test_position])
        predicted_class = int(predictions[test_position])
        row = {
            "test_position": int(test_position),
            "global_index": int(idx_test[test_position]),
            "true_label": true_class,
            "true_class": class_names[true_class],
            "pred_label": predicted_class,
            "pred_class": class_names[predicted_class],
            "confidence": float(confidence[test_position]),
            "entropy": float(entropy[test_position]),
            "nearest_spectral_class": class_names[nearest],
            "distance_to_true_prototype": float(distances[true_class]),
            "distance_to_predicted_prototype": float(distances[predicted_class]),
            "top1_freq_hz": float(freq_hz[top_bins[0]]),
            "top2_freq_hz": float(freq_hz[top_bins[1]]),
            "top3_freq_hz": float(freq_hz[top_bins[2]]),
            "spectral_nearest_matches_prediction": bool(nearest == predicted_class),
        }
        for c, name in enumerate(class_names):
            row[f"distance_to_{name}"] = float(distances[c])
        rows.append(row)
    return pd.DataFrame(rows), prototypes, freq_hz


def confidence_summary(confidence, entropy, predictions, y_test):
    correct = predictions == y_test
    return pd.DataFrame({
        "group": ["Correct", "Incorrect"],
        "n": [int(correct.sum()), int((~correct).sum())],
        "mean_confidence": [
            float(np.mean(confidence[correct])) if np.any(correct) else np.nan,
            float(np.mean(confidence[~correct])) if np.any(~correct) else np.nan,
        ],
        "mean_entropy": [
            float(np.mean(entropy[correct])) if np.any(correct) else np.nan,
            float(np.mean(entropy[~correct])) if np.any(~correct) else np.nan,
        ],
    })


def analyze_frequency_weights(model, x_dev, y_dev, class_names, sampling_frequency_hz, top_k=10):
    weights = model.get_layer("weighted_rfft").get_weights()[0].astype(np.float64)
    absolute_weights = np.abs(weights)
    spectra = np.abs(np.fft.rfft(x_dev, axis=1)).astype(np.float64)
    num_classes = len(class_names)
    class_means = np.zeros((num_classes, spectra.shape[1]), dtype=np.float64)
    class_variances = np.zeros_like(class_means)
    for c in range(num_classes):
        class_means[c] = spectra[y_dev == c].mean(axis=0)
        class_variances[c] = spectra[y_dev == c].var(axis=0)
    between = np.var(class_means, axis=0)
    within = np.mean(class_variances, axis=0)
    fisher = between / (within + 1e-12)
    pearson_r, pearson_p = pearsonr(absolute_weights, fisher)
    spearman_r, spearman_p = spearmanr(absolute_weights, fisher)
    freq_hz = np.fft.rfftfreq(x_dev.shape[1], d=1.0 / sampling_frequency_hz)
    top_k = min(top_k, len(freq_hz))
    top_weight_bins = np.argsort(absolute_weights)[-top_k:][::-1]
    top_fisher_bins = np.argsort(fisher)[-top_k:][::-1]
    overlap = set(top_weight_bins).intersection(set(top_fisher_bins))
    union = set(top_weight_bins).union(set(top_fisher_bins))
    frame = pd.DataFrame({
        "frequency_bin": np.arange(len(freq_hz)),
        "frequency_hz": freq_hz,
        "learned_weight": weights,
        "abs_learned_weight": absolute_weights,
        "between_class_variance": between,
        "within_class_variance": within,
        "fisher_spectral_score": fisher,
    })
    for c, name in enumerate(class_names):
        frame[f"mean_spectrum_{name}"] = class_means[c]
    top_frame = pd.DataFrame([
        {
            "rank_by_abs_weight": rank,
            "frequency_bin": int(b),
            "frequency_hz": float(freq_hz[b]),
            "learned_weight": float(weights[b]),
            "abs_learned_weight": float(absolute_weights[b]),
            "fisher_spectral_score": float(fisher[b]),
            "also_in_top_discriminative_bins": bool(b in set(top_fisher_bins)),
        }
        for rank, b in enumerate(top_weight_bins, start=1)
    ])
    dominant_rows = []
    for c, name in enumerate(class_names):
        spectrum = class_means[c].copy()
        spectrum[0] = 0
        for rank, b in enumerate(np.argsort(spectrum)[-5:][::-1], start=1):
            dominant_rows.append({
                "class": name,
                "rank": rank,
                "frequency_bin": int(b),
                "frequency_hz": float(freq_hz[b]),
                "mean_fft_magnitude": float(spectrum[b]),
                "learned_weight_at_frequency": float(weights[b]),
            })
    statistics = {
        "pearson_r_abs_weight_vs_fisher": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r_abs_weight_vs_fisher": float(spearman_r),
        "spearman_p": float(spearman_p),
        "top_k": int(top_k),
        "top_weight_top_fisher_overlap_count": int(len(overlap)),
        "top_weight_top_fisher_jaccard": float(len(overlap) / len(union)),
        "frequency_resolution_hz": float(sampling_frequency_hz / x_dev.shape[1]),
        "nyquist_frequency_hz": float(sampling_frequency_hz / 2),
        "number_of_rfft_bins": int(len(freq_hz)),
    }
    return {
        "frequency_frame": frame,
        "top_frequencies": top_frame,
        "dominant_frequencies": pd.DataFrame(dominant_rows),
        "statistics": statistics,
        "freq_hz": freq_hz,
        "weights": weights,
        "absolute_weights": absolute_weights,
        "fisher_score": fisher,
        "class_mean_spectra": class_means,
    }


def weighted_rfft_activations(model, x_test, y_test, class_names, freq_hz):
    activation_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("weighted_rfft").output)
    activations = activation_model.predict(x_test, verbose=0)
    rows = []
    class_means = {}
    for c, name in enumerate(class_names):
        mean_activation = activations[y_test == c].mean(axis=0)
        class_means[name] = mean_activation
        rows.extend({
            "class": name,
            "frequency_bin": i,
            "frequency_hz": float(freq_hz[i]),
            "mean_weighted_rfft_activation": float(value),
        } for i, value in enumerate(mean_activation))
    return pd.DataFrame(rows), class_means
