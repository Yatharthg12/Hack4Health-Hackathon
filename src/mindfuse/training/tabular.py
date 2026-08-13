"""Leakage-safe classification and multi-output regression training."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mindfuse.config import FIGURES_DIR, METRICS_DIR, MODEL_DIR, RANDOM_SEED
from mindfuse.constants import (
    CLASSIFICATION_TARGET,
    FEATURE_METADATA,
    NUMERICAL_FEATURES,
    REGRESSION_TARGETS,
    SCORE_RANGES,
    STRESS_CLASSES,
)
from mindfuse.data.numerical import load_numerical_dataset
from mindfuse.evaluation.metrics import classification_metrics, regression_metrics
from mindfuse.evaluation.plots import (
    confusion_matrix_plot,
    feature_importance_plot,
    regression_scatter_plot,
)
from mindfuse.utils.json_io import write_json
from mindfuse.models.calibration import TemperatureScaledClassifier, fit_probability_temperature


def _snap_to_ui_step(feature: str, value: float) -> float:
    metadata = FEATURE_METADATA[feature]
    minimum = float(metadata["min"])
    maximum = float(metadata["max"])
    step = float(metadata["step"])
    snapped = minimum + round((float(value) - minimum) / step) * step
    decimals = max(0, len(str(metadata["step"]).split(".")[1]) if "." in str(metadata["step"]) else 0)
    return round(min(max(snapped, minimum), maximum), decimals)


def generate_demo_profiles(
    frame: pd.DataFrame,
    train_indices: np.ndarray | list[int],
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Create UI-ready representative profiles from the training partition only."""

    indices = np.asarray(train_indices, dtype=int)
    if indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= len(frame):
        raise ValueError("Demo-profile training indices are empty or outside the dataset")
    training = frame.iloc[indices]
    features = training[NUMERICAL_FEATURES]
    typical = features.median(numeric_only=True)
    scale = features.std(ddof=0).replace(0, 1)
    selections: dict[str, tuple[str, pd.Series, int, str]] = {
        "typical": ("Typical", typical, len(training), "all-training median"),
    }
    for key, label, class_name in (
        ("good", "Good", "Healthy"),
        ("high_strain", "High strain", "Severe_Stress"),
    ):
        population = training[training[CLASSIFICATION_TARGET] == class_name][NUMERICAL_FEATURES]
        if population.empty:
            raise ValueError(f"Training split has no rows for demo profile {key}")
        center = population.median(numeric_only=True)
        distance = (((population - center) / scale) ** 2).mean(axis=1)
        medoid = population.loc[distance.idxmin()]
        selections[key] = (
            label,
            medoid,
            len(population),
            f"real {class_name} training row nearest its class median",
        )
    profiles: dict[str, dict[str, Any]] = {}
    for key in ("good", "typical", "high_strain"):
        label, representative, population_size, selection = selections[key]
        values = {
            feature: _snap_to_ui_step(feature, float(representative[feature]))
            for feature in NUMERICAL_FEATURES
        }
        profiles[key] = {
            "label": label,
            "values": values,
            "derivation_rows": int(population_size),
            "selection": selection,
        }
    return {
        "schema_version": 1,
        "profiles": profiles,
        "derivation": (
            "Representative values derived only from the organizer numerical training split. Typical is the "
            "all-training median. Good and High strain are real Healthy and Severe_Stress training rows, "
            "respectively, selected as nearest to their class feature medians."
        ),
        "usage": "Demonstration inputs for model exploration only; they are not diagnoses or clinical reference profiles.",
        "split_seed": seed,
        "feature_count": len(NUMERICAL_FEATURES),
    }


def _preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])


def _classifier_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", LogisticRegression(max_iter=4_000, class_weight="balanced", C=1.0, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=350, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
            )),
        ]),
        "extra_trees": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=seed,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                l2_regularization=1.0, class_weight="balanced", random_state=seed,
            )),
        ]),
    }


def _regressor_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", Ridge(alpha=2.0)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=350, min_samples_leaf=2, max_features=0.8,
                n_jobs=-1, random_state=seed,
            )),
        ]),
        "extra_trees": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", ExtraTreesRegressor(
                n_estimators=400, min_samples_leaf=2, max_features=0.9,
                n_jobs=-1, random_state=seed,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", _preprocessor()),
            ("model", MultiOutputRegressor(HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                l2_regularization=1.0, random_state=seed,
            ))),
        ]),
    }


def _split_indices(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    labels = frame[CLASSIFICATION_TARGET].map({label: index for index, label in enumerate(STRESS_CLASSES)})
    train_validation, test = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=labels,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.1764705882,
        random_state=seed,
        stratify=labels.iloc[train_validation],
    )
    return np.sort(train), np.sort(validation), np.sort(test)


def _global_importance(estimator: Any, features: pd.DataFrame, labels: np.ndarray) -> list[dict[str, float | str]]:
    result = permutation_importance(
        estimator,
        features,
        labels,
        scoring="f1_macro",
        n_repeats=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rows = [
        {"feature": feature, "importance": float(mean), "std": float(std)}
        for feature, mean, std in zip(NUMERICAL_FEATURES, result.importances_mean, result.importances_std, strict=True)
    ]
    return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)


def train_tabular_models(csv_path: Path, seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Train, select, calibrate, evaluate, persist, and visualize both tabular tasks."""

    frame = load_numerical_dataset(csv_path)
    train_idx, validation_idx, test_idx = _split_indices(frame, seed)
    features = frame[NUMERICAL_FEATURES]
    label_to_index = {label: index for index, label in enumerate(STRESS_CLASSES)}
    labels = frame[CLASSIFICATION_TARGET].map(label_to_index).to_numpy(dtype=int)
    targets = frame[REGRESSION_TARGETS].to_numpy(dtype=float)

    classifier_comparison: dict[str, dict[str, float]] = {}
    best_classifier_name = ""
    best_classifier_score = (-np.inf, -np.inf)
    best_classifier: Pipeline | None = None
    for name, candidate in _classifier_candidates(seed).items():
        candidate.fit(features.iloc[train_idx], labels[train_idx])
        predictions = candidate.predict(features.iloc[validation_idx])
        scores = {
            "macro_f1": float(f1_score(labels[validation_idx], predictions, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[validation_idx], predictions)),
        }
        classifier_comparison[name] = scores
        rank = (scores["macro_f1"], scores["balanced_accuracy"])
        if rank > best_classifier_score:
            best_classifier_name, best_classifier_score, best_classifier = name, rank, candidate
    assert best_classifier is not None
    train_validation_idx = np.concatenate([train_idx, validation_idx])
    validation_probabilities = best_classifier.predict_proba(features.iloc[validation_idx])
    temperature = fit_probability_temperature(validation_probabilities, labels[validation_idx])
    final_base_classifier = clone(best_classifier).fit(
        features.iloc[train_validation_idx], labels[train_validation_idx]
    )
    calibrated = TemperatureScaledClassifier(final_base_classifier, temperature)
    classifier_probabilities = calibrated.predict_proba(features.iloc[test_idx])
    classifier_predictions = classifier_probabilities.argmax(axis=1)
    classifier_test_metrics = classification_metrics(
        labels[test_idx], classifier_predictions, classifier_probabilities, STRESS_CLASSES
    )
    importance = _global_importance(calibrated, features.iloc[test_idx], labels[test_idx])

    background = {
        "median": {feature: float(features.iloc[train_validation_idx][feature].median()) for feature in NUMERICAL_FEATURES},
        "std": {feature: float(features.iloc[train_validation_idx][feature].std(ddof=0)) for feature in NUMERICAL_FEATURES},
    }
    classifier_artifact = {
        "pipeline": calibrated,
        "feature_names": NUMERICAL_FEATURES,
        "class_names": STRESS_CLASSES,
        "background": background,
        "selected_model": best_classifier_name,
        "seed": seed,
    }
    joblib.dump(classifier_artifact, MODEL_DIR / "numerical_classifier.joblib", compress=3)

    classifier_payload: dict[str, Any] = {
        "model": best_classifier_name,
        "model_selection": classifier_comparison,
        "selection_partition": "validation",
        "test_metrics": classifier_test_metrics,
        "global_feature_importance": importance,
        "split": {"train": len(train_idx), "validation": len(validation_idx), "test": len(test_idx)},
        "class_distribution": frame[CLASSIFICATION_TARGET].value_counts().to_dict(),
        "probability_calibration": {
            "method": "scalar temperature scaling fitted on validation probabilities",
            "temperature": temperature,
            "decision_preserving": True,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }
    write_json(METRICS_DIR / "numerical_classification.json", classifier_payload)
    confusion_matrix_plot(
        classifier_test_metrics["confusion_matrix"], STRESS_CLASSES,
        FIGURES_DIR / "numerical_confusion_matrix.png", "Numerical classifier — held-out test",
    )
    feature_importance_plot(
        importance[:14], FIGURES_DIR / "numerical_feature_importance.png", "Global permutation importance",
    )

    target_scales = np.asarray([SCORE_RANGES[target][1] - SCORE_RANGES[target][0] for target in REGRESSION_TARGETS])
    regressor_comparison: dict[str, dict[str, float]] = {}
    best_regressor_name = ""
    best_regressor_score = np.inf
    best_regressor: Pipeline | None = None
    for name, candidate in _regressor_candidates(seed).items():
        candidate.fit(features.iloc[train_idx], targets[train_idx])
        predictions = candidate.predict(features.iloc[validation_idx])
        per_target_rmse = np.sqrt(np.mean((targets[validation_idx] - predictions) ** 2, axis=0))
        normalized_rmse = float(np.mean(per_target_rmse / target_scales))
        regressor_comparison[name] = {
            "mean_normalized_rmse": normalized_rmse,
            "mean_rmse": float(np.mean(per_target_rmse)),
        }
        if normalized_rmse < best_regressor_score:
            best_regressor_name, best_regressor_score, best_regressor = name, normalized_rmse, candidate
    assert best_regressor is not None
    final_regressor = clone(best_regressor).fit(features.iloc[train_validation_idx], targets[train_validation_idx])
    raw_regression_predictions = np.asarray(final_regressor.predict(features.iloc[test_idx]))
    regression_test_metrics = regression_metrics(targets[test_idx], raw_regression_predictions, REGRESSION_TARGETS)
    regressor_artifact = {
        "pipeline": final_regressor,
        "feature_names": NUMERICAL_FEATURES,
        "target_names": REGRESSION_TARGETS,
        "score_ranges": SCORE_RANGES,
        "selected_model": best_regressor_name,
        "seed": seed,
    }
    joblib.dump(regressor_artifact, MODEL_DIR / "das_regressor.joblib", compress=3)
    regression_payload = {
        "model": best_regressor_name,
        "model_selection": regressor_comparison,
        "selection_partition": "validation",
        "test_metrics": regression_test_metrics,
        "split": {"train": len(train_idx), "validation": len(validation_idx), "test": len(test_idx)},
        "clamping": "Predictions are evaluated raw; documented bounds are applied only at inference presentation.",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }
    write_json(METRICS_DIR / "regression.json", regression_payload)
    regression_scatter_plot(
        targets[test_idx], raw_regression_predictions, REGRESSION_TARGETS,
        FIGURES_DIR / "regression_predicted_vs_actual.png",
    )

    demo_profiles = generate_demo_profiles(frame, train_idx, seed)
    write_json(METRICS_DIR / "demo_profiles.json", demo_profiles)
    write_json(METRICS_DIR / "demo_profile.json", demo_profiles["profiles"]["typical"]["values"])
    write_json(
        METRICS_DIR / "tabular_split.json",
        {"seed": seed, "train_indices": train_idx.tolist(), "validation_indices": validation_idx.tolist(), "test_indices": test_idx.tolist()},
    )
    return {"classification": classifier_payload, "regression": regression_payload}
