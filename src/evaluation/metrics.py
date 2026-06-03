"""Compute Accuracy, Weighted F1, and MCC for 3-class sentiment predictions."""

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix


def compute_metrics(labels: list[str], preds: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "weighted_f1": round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
        "mcc": round(matthews_corrcoef(labels, preds), 4),
    }


def confusion_matrix_df(labels: list[str], preds: list[str]):
    """Return a labelled confusion matrix as a pandas DataFrame."""
    import pandas as pd

    classes = sorted(set(labels) | set(preds))
    cm = confusion_matrix(labels, preds, labels=classes)
    return pd.DataFrame(cm, index=classes, columns=classes)


def save_results(results: dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
