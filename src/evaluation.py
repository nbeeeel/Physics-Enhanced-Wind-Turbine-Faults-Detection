import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score


def evaluate_model(model, x_test, y_test, idx_test, class_names):
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
    }
    prediction_frame = pd.DataFrame({
        "global_index": idx_test,
        "true_label": y_test,
        "true_class": [class_names[i] for i in y_test],
        "pred_label": predictions,
        "pred_class": [class_names[i] for i in predictions],
        "confidence": confidence,
        "entropy": entropy,
        "correct": y_test == predictions,
    })
    for i, name in enumerate(class_names):
        prediction_frame[f"prob_{name}"] = probabilities[:, i]
    report = pd.DataFrame(classification_report(
        y_test,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )).transpose()
    cm_count = confusion_matrix(y_test, predictions)
    cm_normalized = confusion_matrix(y_test, predictions, normalize="true")
    y_binary = tf.keras.utils.to_categorical(y_test, len(class_names))
    roc_rows = []
    pr_rows = []
    auc_values = {}
    ap_values = {}
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_binary[:, i], probabilities[:, i])
        class_auc = auc(fpr, tpr)
        auc_values[name] = float(class_auc)
        roc_rows.extend({"class": name, "fpr": a, "tpr": b, "auc": class_auc} for a, b in zip(fpr, tpr))
        precision, recall, _ = precision_recall_curve(y_binary[:, i], probabilities[:, i])
        class_ap = average_precision_score(y_binary[:, i], probabilities[:, i])
        ap_values[name] = float(class_ap)
        pr_rows.extend({"class": name, "recall": r, "precision": p, "ap": class_ap} for r, p in zip(recall, precision))
    metrics["macro_auc"] = float(np.mean(list(auc_values.values())))
    metrics["macro_average_precision"] = float(np.mean(list(ap_values.values())))
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "confidence": confidence,
        "entropy": entropy,
        "metrics": metrics,
        "predictions_frame": prediction_frame,
        "classification_report": report,
        "confusion_counts": cm_count,
        "confusion_normalized": cm_normalized,
        "roc_points": pd.DataFrame(roc_rows),
        "pr_points": pd.DataFrame(pr_rows),
        "auc_values": auc_values,
        "ap_values": ap_values,
    }
