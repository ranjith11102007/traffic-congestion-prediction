"""Evaluation of the trained congestion model.

Every number this module emits is computed from the held-out test window of the
Stage 2 dataset. Nothing is hard-coded and nothing is estimated.

Reproducing the training split
------------------------------
The split boundary is read from ``model_metadata.json`` rather than recomputed,
so evaluation scores exactly the rows ``ml.train`` held out. Recomputing a
quantile would be fragile: a changed dataset would silently shift the boundary
and produce scores for a different sample.

Why a baseline is always reported
---------------------------------
Accuracy alone is misleading on imbalanced congestion data. The test window here
contains far more moderate than severe congestion, so a model that always
predicted "moderate" would score a deceptively high accuracy. This module always
computes that majority-class baseline alongside the model, plus macro-averaged
metrics that weight every class equally, per-class precision/recall/F1, and
per-class support so a reader can see how few rows a rare class had.

Outputs
-------
``ml/models/evaluation_results.json``
    Every metric, the baseline, the split record and the confusion matrix.
``ml/models/confusion_matrix.png``
    A rendered figure, when matplotlib is available.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from ml.preprocessing import (
    CONGESTION_NAMES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    DatasetError,
)
from ml.schema import FeatureSchema
from ml.train import (
    MODELS_DIR,
    _distribution,
    assert_no_leakage_columns,
    load_metadata,
    load_model,
    load_preprocessor,
    select_feature_columns,
)

RESULTS_FILENAME = "evaluation_results.json"
#: Written next to the JSON so the run is readable without re-running anything.
REPORT_FILENAME = "model_report.md"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

CONGESTION_COLOURS = {
    0: "#2e7d32",
    1: "#f9a825",
    2: "#ef6c00",
    3: "#c62828",
}


# ------------------------------------------------------------------- metrics


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None
) -> dict[str, Any]:
    """Return classification metrics computed from ``y_true`` and ``y_pred``.

    Args:
        y_true: actual labels.
        y_pred: predicted labels.
        labels: the full class list. Defaults to the labels present in
            ``y_true``, so a class absent from the test window is not silently
            reported as perfectly classified.

    Returns:
        A JSON-safe dictionary containing accuracy, macro/weighted averages,
        per-class scores, the confusion matrix and the text report.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [int(label) for label in labels]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        str(label): {
            "label_name": CONGESTION_NAMES.get(label, str(label)),
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6
        ),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 6
        ),
        "precision_weighted": round(
            float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 6
        ),
        "recall_weighted": round(
            float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 6
        ),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 6
        ),
        "macro_f1_is_primary": True,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[CONGESTION_NAMES.get(label, str(label)) for label in labels],
            zero_division=0,
            digits=4,
        ),
        "evaluated_rows": int(len(y_true)),
    }


def compute_majority_baseline(
    y_test: Sequence[int], y_train: Sequence[int]
) -> dict[str, Any]:
    """Score "always predict the most common class in the training window".

    Reported alongside the model so a high accuracy can be judged for what it is
    worth on imbalanced data.

    The majority class is taken from the *training* window, never from the test
    labels: choosing it from ``y_test`` would let test information influence the
    comparison, which is the same leak the chronological split exists to prevent.
    On this dataset both windows happen to be class 1, but relying on that
    coincidence would hide a real error the moment the data changed.
    """

    y_test = np.asarray(y_test, dtype=int)
    y_train = np.asarray(y_train, dtype=int)
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    majority = int(train_classes[int(np.argmax(train_counts))])
    constant = np.full_like(y_test, majority)

    return {
        "strategy": "always predict the most frequent training class",
        "majority_class": majority,
        "majority_class_name": CONGESTION_NAMES.get(majority, str(majority)),
        "training_share": round(
            float(train_counts[int(np.argmax(train_counts))] / train_counts.sum()), 6
        ),
        "accuracy": round(float(accuracy_score(y_test, constant)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, constant)), 6),
        "f1_macro": round(
            float(f1_score(y_test, constant, average="macro", zero_division=0)), 6
        ),
        "recall_macro": round(
            float(recall_score(y_test, constant, average="macro", zero_division=0)), 6
        ),
    }


def compare_to_baseline(
    model_metrics: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """State plainly whether the model beats the majority-class baseline.

    Accuracy alone can look like progress while macro F1 shows no improvement, so
    both deltas are reported and the verdict names the metric it used.
    """

    accuracy_delta = round(model_metrics["accuracy"] - baseline["accuracy"], 6)
    macro_delta = round(model_metrics["f1_macro"] - baseline["f1_macro"], 6)

    beats_accuracy = accuracy_delta > 0
    beats_macro_f1 = macro_delta > 0

    if beats_accuracy and beats_macro_f1:
        verdict = "beats the majority-class baseline on both accuracy and macro F1"
    elif beats_macro_f1:
        verdict = (
            "beats the baseline on macro F1 but not on accuracy: it does better on "
            "rare classes while not gaining overall accuracy"
        )
    elif beats_accuracy:
        verdict = (
            "beats the baseline on accuracy only; macro F1 does not improve, so the "
            "gain comes from the majority class"
        )
    else:
        verdict = "does not beat the majority-class baseline"

    return {
        "accuracy_delta": accuracy_delta,
        "macro_f1_delta": macro_delta,
        "beats_baseline_on_accuracy": beats_accuracy,
        "beats_baseline_on_macro_f1": beats_macro_f1,
        "verdict": verdict,
        "note": (
            "A chronological split cannot be stratified, so the test window's class "
            "proportions differ from the full dataset. Macro F1 weights all classes "
            "equally and is the fairer comparison."
        ),
    }


# --------------------------------------------------------------- confusion matrix


def save_confusion_matrix_image(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[int],
    target_path: str | Path,
    *,
    title: str = "Confusion matrix (held-out test set)",
    normalise: bool = False,
) -> Path | None:
    """Render the confusion matrix to a PNG.

    Returns:
        The written path, or ``None`` if matplotlib is unavailable. A missing
        optional plotting dependency must not stop evaluation.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(matrix, dtype=float)
    if normalise and array.sum():
        array = array / array.sum(axis=1, keepdims=True)

    names = [f"{label}\n{CONGESTION_NAMES.get(int(label), '')}".strip() for label in labels]

    figure, axes = plt.subplots(figsize=(1.9 + 0.85 * len(labels), 1.7 + 0.85 * len(labels)))
    image = axes.imshow(array, cmap="Blues")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    threshold = array.max() / 2.0 if array.size else 0.0
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            value = array[row, column]
            axes.text(
                column,
                row,
                f"{value:.2f}" if normalise else f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
            )

    axes.set_xticks(range(len(labels)), names, rotation=30, ha="right")
    axes.set_yticks(range(len(labels)), names)
    axes.set_xlabel("Predicted congestion level")
    axes.set_ylabel("Actual congestion level")
    axes.set_title(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target


# ------------------------------------------------------------------ evaluation


def evaluate(
    data_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    write_results: bool = True,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score the trained model on the held-out test window.

    Args:
        data_path: processed CSV; defaults to the Stage 2 output.
        models_dir: artifact directory; defaults to ``ml/models``.
        write_results: write ``evaluation_results.json``, ``model_report.md`` and
            the confusion matrix.
        artifact_path: Stage 1 compatibility argument. Only its parent directory
            is read, so existing callers keep working.

    Returns:
        A JSON-safe dictionary of measured results.

    Raises:
        ModelArtifactNotFoundError: if no model has been trained.
        DatasetError: if the dataset or split does not match the artifact.
    """

    if artifact_path is not None and models_dir is None:
        models_dir = Path(artifact_path).expanduser().resolve().parent

    from ml.train import load_processed_dataset

    metadata = load_metadata(models_dir)
    estimator = load_model(models_dir)
    schema = load_preprocessor(models_dir)

    frame, source = load_processed_dataset(data_path)
    if source.name != metadata["dataset_name"]:
        raise DatasetError(
            f"Evaluation dataset {source.name!r} does not match the dataset the "
            f"model was trained on ({metadata['dataset_name']!r}). Retrain, or pass "
            "the matching dataset."
        )

    feature_columns = tuple(metadata["features"])
    expected = select_feature_columns(frame)
    if feature_columns != expected:
        raise DatasetError(
            "The dataset's feature columns differ from those recorded in "
            "model_metadata.json. Retrain so the schema matches."
        )
    assert_no_leakage_columns(feature_columns)

    # Reproduce training's split from the stored boundary rather than a quantile.
    # The boundary is inclusive of training: split_chronologically() puts rows at
    # exactly the latest training timestamp on the training side, so "<=" here is
    # required to score exactly the rows ml.train held out.
    cutoff = pd.Timestamp(metadata["split_cutoff"])
    train_frame = frame.loc[frame[TIMESTAMP_COLUMN] <= cutoff]
    test_frame = frame.loc[frame[TIMESTAMP_COLUMN] > cutoff]
    if train_frame.empty or test_frame.empty:
        raise DatasetError(
            "The stored split boundary no longer splits this dataset. Retrain."
        )

    expected_train = int(metadata["training_records"])
    expected_test = int(metadata["test_records"])
    if (len(train_frame), len(test_frame)) != (expected_train, expected_test):
        raise DatasetError(
            f"Reproducing the stored split gave {len(train_frame)} train / "
            f"{len(test_frame)} test rows, but training recorded "
            f"{expected_train} / {expected_test}. The dataset changed since "
            "training. Retrain before evaluating."
        )

    X_test = test_frame.loc[:, list(feature_columns)]
    y_test = test_frame[TARGET_COLUMN].to_numpy(dtype=int)
    predictions = np.asarray(estimator.predict(X_test), dtype=int)

    labels = sorted(CONGESTION_NAMES)
    model_metrics = compute_classification_metrics(y_test, predictions, labels=labels)
    baseline = compute_majority_baseline(y_test, train_frame[TARGET_COLUMN].to_numpy(dtype=int))
    comparison = compare_to_baseline(model_metrics, baseline)

    results: dict[str, Any] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata["model_name"],
        "model_display_name": metadata["model_display_name"],
        "model_type": metadata["model_type"],
        "model_version": metadata["model_version"],
        "trained_at": metadata["trained_at"],
        "dataset": {
            "name": metadata["dataset_name"],
            "rows_total": int(len(frame)),
            "training_records": int(len(train_frame)),
            "test_records": int(len(test_frame)),
            "is_simulated": bool(metadata.get("dataset_is_simulated")),
        },
        "target_name": metadata["target_name"],
        "label_mapping": metadata["label_mapping"],
        "split": {
            "strategy": metadata["split_strategy"],
            "cutoff": metadata["split_cutoff"],
            "train_time_range": metadata["train_time_range"],
            "test_time_range": metadata["test_time_range"],
            "note": (
                "Identical to training: the boundary is read from "
                "model_metadata.json, not recomputed."
            ),
        },
        "train_class_distribution": _distribution(train_frame[TARGET_COLUMN]),
        "test_class_distribution": _distribution(test_frame[TARGET_COLUMN]),
        "metrics": model_metrics,
        "majority_class_baseline": baseline,
        "baseline_comparison": comparison,
        "class_imbalance_warning": _imbalance_warning(
            train_frame[TARGET_COLUMN], test_frame[TARGET_COLUMN]
        ),
        "feature_count": len(feature_columns),
        "features": list(feature_columns),
        "sklearn_version": metadata.get("sklearn_version"),
        "feature_importance": _load_feature_importance(
            (Path(models_dir).expanduser().resolve() if models_dir else MODELS_DIR)
        ),
    }

    target_dir = Path(models_dir).expanduser().resolve() if models_dir else MODELS_DIR

    if write_results:
        target_dir.mkdir(parents=True, exist_ok=True)
        results_path = target_dir / RESULTS_FILENAME
        report_path = target_dir / REPORT_FILENAME

        image_path = save_confusion_matrix_image(
            model_metrics["confusion_matrix"],
            model_metrics["confusion_matrix_labels"],
            target_dir / CONFUSION_MATRIX_FILENAME,
        )
        results["confusion_matrix_image"] = (
            str(image_path) if image_path else "not generated (matplotlib unavailable)"
        )
        results["results_path"] = str(results_path)
        results["report_path"] = str(report_path)

        # Paths are attached before serialising, so the JSON on disk describes
        # itself instead of being a subset of what the caller received.
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        report_path.write_text(render_markdown_report(results), encoding="utf-8")

    return results


def _load_feature_importance(target_dir: Path) -> list[dict[str, Any]]:
    """Read the importance file training wrote, so the report can rank features.

    Returns an empty list when the file is absent or unreadable; the report then
    points at the file rather than inventing an ordering.
    """

    from ml.train import FEATURE_IMPORTANCE_FILENAME

    path = target_dir / FEATURE_IMPORTANCE_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("features") if isinstance(payload, dict) else payload
    if entries is None and isinstance(payload, dict):
        entries = payload.get("importances")
    if not isinstance(entries, list):
        return []
    return [
        {"feature": str(item["feature"]), "importance": float(item["importance"])}
        for item in entries
        if isinstance(item, dict) and "feature" in item and "importance" in item
    ]


def _imbalance_warning(y_train: pd.Series, y_test: pd.Series) -> str | None:
    """Flag class distributions too thin for a score to mean much."""

    notes: list[str] = []
    train_ratio = y_train.value_counts(normalize=True)
    test_counts = y_test.value_counts()

    rarest = int(train_ratio.idxmin())
    if float(train_ratio.min()) < 0.1:
        notes.append(
            f"class {rarest} is {train_ratio.min() * 100:.1f}% of the training data"
        )
    for label in sorted(test_counts.index):
        if int(test_counts[label]) < 20:
            notes.append(
                f"class {int(label)} has only {int(test_counts[label])} test rows, so "
                "its per-class scores are unstable"
            )
    if not notes:
        return None
    return "; ".join(notes)


def format_report(results: dict[str, Any]) -> str:
    """Render evaluation results as a readable text block."""

    metrics = results["metrics"]
    baseline = results["majority_class_baseline"]
    comparison = results["baseline_comparison"]

    lines = [
        "=" * 72,
        f"model          : {results['model_display_name']} ({results['model_name']})",
        f"model version  : {results['model_version']}",
        f"trained at     : {results['trained_at']}",
        f"dataset        : {results['dataset']['name']}"
        f"{'  [SIMULATED]' if results['dataset']['is_simulated'] else ''}",
        f"target         : {results['target_name']}",
        f"features       : {results['feature_count']}",
        f"split          : {results['split']['strategy']} at {results['split']['cutoff']}",
        f"train / test   : {results['dataset']['training_records']} / "
        f"{results['dataset']['test_records']} rows",
        "=" * 72,
        "",
        "model metrics (held-out test set)",
        f"  accuracy           : {metrics['accuracy']:.4f}",
        f"  balanced accuracy  : {metrics['balanced_accuracy']:.4f}",
        f"  f1 (macro)         : {metrics['f1_macro']:.4f}   <- primary metric",
        f"  precision (macro)  : {metrics['precision_macro']:.4f}",
        f"  recall (macro)     : {metrics['recall_macro']:.4f}",
        f"  f1 (weighted)      : {metrics['f1_weighted']:.4f}",
        "",
        "majority-class baseline",
        f"  always predict     : {baseline['majority_class_name']} "
        f"(class {baseline['majority_class']})",
        f"  accuracy           : {baseline['accuracy']:.4f}",
        f"  f1 (macro)         : {baseline['f1_macro']:.4f}",
        f"  accuracy delta     : {comparison['accuracy_delta']:+.4f}",
        f"  macro f1 delta     : {comparison['macro_f1_delta']:+.4f}",
        f"  verdict            : {comparison['verdict']}",
        "",
        "per-class detail",
    ]

    for label, entry in metrics["per_class"].items():
        lines.append(
            f"  {label} {entry['label_name']:<9} "
            f"precision {entry['precision']:.4f}  "
            f"recall {entry['recall']:.4f}  "
            f"f1 {entry['f1']:.4f}  "
            f"support {entry['support']}"
        )

    lines += [
        "",
        f"confusion matrix (rows = actual, cols = predicted, order "
        f"{metrics['confusion_matrix_labels']})",
    ]
    lines.append(str(np.asarray(metrics["confusion_matrix"])))
    lines += ["", "classification report", metrics["classification_report"]]

    if results.get("class_imbalance_warning"):
        lines += ["", f"WARNING: {results['class_imbalance_warning']}"]

    return "\n".join(lines)


def render_markdown_report(results: dict[str, Any]) -> str:
    """Render the run as ``ml/models/model_report.md``.

    Every number here is read back from the evaluation results, so the report
    cannot drift from the JSON it was generated in the same pass.
    """

    from ml.train import FEATURE_IMPORTANCE_FILENAME, MODEL_FILENAME

    metrics = results["metrics"]
    baseline = results["majority_class_baseline"]
    comparison = results["baseline_comparison"]
    dataset = results["dataset"]

    lines: list[str] = [
        "# Traffic Congestion Model Report",
        "",
        f"Generated {results['evaluated_at']} by `python -m ml.evaluate`.",
        "",
    ]

    if dataset["is_simulated"]:
        lines += [
            "> **The training data is simulated.** Every figure below describes "
            "performance on a synthetic fixture generated by "
            "`ml/data/sample/generate_simulated_dataset.py`, not on real traffic. "
            "The numbers demonstrate that the pipeline works end to end. They are "
            "not evidence about real-world congestion and must not be quoted as "
            "such.",
            "",
        ]

    lines += [
        "## Run summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Model | {results['model_display_name']} (`{results['model_name']}`) |",
        f"| Model version | {results['model_version']} |",
        f"| Trained at | {results['trained_at']} |",
        f"| Dataset | `{dataset['name']}` ({dataset['rows_total']} rows) |",
        f"| Target | `{results['target_name']}` |",
        f"| Features | {results['feature_count']} |",
        f"| Split | {results['split']['strategy']} at `{results['split']['cutoff']}` |",
        f"| Train / test rows | {dataset['training_records']} / {dataset['test_records']} |",
        "",
        "## Held-out metrics",
        "",
        "Macro F1 is the primary metric: it weights every congestion level "
        "equally, so a model cannot look good by predicting only the most common "
        "level.",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |",
        f"| **F1 (macro)** | **{metrics['f1_macro']:.4f}** |",
        f"| Precision (macro) | {metrics['precision_macro']:.4f} |",
        f"| Recall (macro) | {metrics['recall_macro']:.4f} |",
        f"| F1 (weighted) | {metrics['f1_weighted']:.4f} |",
        "",
        "### Per-class detail",
        "",
        "| Class | Name | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, entry in metrics["per_class"].items():
        lines.append(
            f"| {label} | {entry['label_name']} | {entry['precision']:.4f} | "
            f"{entry['recall']:.4f} | {entry['f1']:.4f} | {entry['support']} |"
        )

    lines += [
        "",
        "### Confusion matrix",
        "",
        f"Rows are actual levels, columns are predicted levels, ordered "
        f"{metrics['confusion_matrix_labels']}.",
        "",
        "```",
        str(np.asarray(metrics["confusion_matrix"])),
        "```",
        "",
        f"![Confusion matrix]({Path(results.get('confusion_matrix_image', '')).name})",
        "",
        "## Baseline comparison",
        "",
        f"The baseline always predicts `{baseline['majority_class_name']}` "
        f"(class {baseline['majority_class']}), the most frequent class in the "
        "training window. Choosing it from the training data rather than the test "
        "labels keeps test information out of the comparison.",
        "",
        "| Metric | Model | Baseline | Delta |",
        "| --- | --- | --- | --- |",
        f"| Accuracy | {metrics['accuracy']:.4f} | {baseline['accuracy']:.4f} | "
        f"{comparison['accuracy_delta']:+.4f} |",
        f"| F1 (macro) | {metrics['f1_macro']:.4f} | {baseline['f1_macro']:.4f} | "
        f"{comparison['macro_f1_delta']:+.4f} |",
        "",
        f"**Verdict:** {comparison['verdict']}.",
        "",
        "## Class distribution",
        "",
        "| Class | Train rows | Test rows |",
        "| --- | --- | --- |",
    ]
    for label in sorted(results["test_class_distribution"], key=int):
        lines.append(
            f"| {label} | {results['train_class_distribution'].get(label, 0)} | "
            f"{results['test_class_distribution'][label]} |"
        )

    if results.get("class_imbalance_warning"):
        lines += [
            "",
            f"> **Caution:** {results['class_imbalance_warning']}. Rare-level "
            "scores are unstable and should not be read as precise estimates.",
        ]

    lines += [
        "",
        "## Leakage control",
        "",
        "- The target is derived from `speed_ratio`, so the contemporaneous "
        "`avg_speed_kph`, `flow_veh_per_hr`, `occupancy_pct` and `speed_ratio` "
        "measurements are excluded from the model's inputs. Training refuses to "
        "run if any of them appears in the feature set.",
        "- The split is chronological, never random: the model is always scored "
        "on timestamps it has not seen.",
        "- `speed_ratio` is recomputed from its components at inference time "
        "rather than accepted from the caller, so the target quantity cannot "
        "enter through the input.",
        "- Lag and rolling features read strictly backwards within each road "
        "segment.",
        "",
        "## Top features by importance",
        "",
    ]

    importances = results.get("feature_importance") or []
    if importances:
        lines += ["| Rank | Feature | Importance |", "| --- | --- | --- |"]
        for rank, entry in enumerate(importances[:15], start=1):
            lines.append(
                f"| {rank} | `{entry['feature']}` | {entry['importance']:.4f} |"
            )
    else:
        lines.append(
            f"Not embedded in this report; see `{FEATURE_IMPORTANCE_FILENAME}`."
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m ml.train",
        "python -m ml.evaluate",
        "python -m ml.predict <input.csv>",
        "```",
        "",
        f"Full metrics: `{RESULTS_FILENAME}`. Model: `{MODEL_FILENAME}`.",
        "",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained congestion model on its held-out test window."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Processed dataset CSV (default: ml/data/processed/traffic_processed.csv).",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="Artifact directory."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print results as JSON instead of text."
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Do not write results files."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from ml.train import ModelArtifactNotFoundError

    try:
        results = evaluate(
            data_path=args.data,
            models_dir=args.models_dir,
            write_results=not args.no_write,
        )
    except (DatasetError, ModelArtifactNotFoundError) as error:
        print(f"error: {error}")
        return 1

    if args.json:
        printable = {key: value for key, value in results.items()}
        print(json.dumps(printable, indent=2))
    else:
        print(format_report(results))
        if results.get("results_path"):
            print(f"\nresults written to: {results['results_path']}")
        if results.get("confusion_matrix_image"):
            print(f"confusion matrix  : {results['confusion_matrix_image']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())