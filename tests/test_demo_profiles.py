from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mindfuse.config import METRICS_DIR
from mindfuse.constants import FEATURE_METADATA, NUMERICAL_FEATURES, STRESS_CLASSES
from mindfuse.data.numerical import load_numerical_dataset, validate_numerical_payload
from mindfuse.training.tabular import generate_demo_profiles


def _profiles() -> dict:
    return json.loads((METRICS_DIR / "demo_profiles.json").read_text(encoding="utf-8"))


def test_demo_profile_artifact_has_three_input_only_profiles() -> None:
    payload = _profiles()
    assert list(payload["profiles"]) == ["good", "typical", "high_strain"]
    assert payload["feature_count"] == 18
    for profile in payload["profiles"].values():
        assert list(profile["values"]) == NUMERICAL_FEATURES
        assert not any("target" in key.lower() or "prediction" in key.lower() for key in profile["values"])
        assert validate_numerical_payload(profile["values"]) == profile["values"]


def test_demo_values_match_html_ranges_and_steps() -> None:
    for profile in _profiles()["profiles"].values():
        for feature, value in profile["values"].items():
            metadata = FEATURE_METADATA[feature]
            minimum, maximum, step = map(float, (metadata["min"], metadata["max"], metadata["step"]))
            assert minimum <= value <= maximum
            steps = (float(value) - minimum) / step
            assert steps == pytest.approx(round(steps), abs=1e-7)


def test_demo_artifact_reproduces_from_training_split_only() -> None:
    frame = load_numerical_dataset(Path("data/raw/mental_health_multimodal.csv"))
    split = json.loads((METRICS_DIR / "tabular_split.json").read_text(encoding="utf-8"))
    regenerated = generate_demo_profiles(frame, split["train_indices"], split["seed"])
    assert regenerated == _profiles()


def test_config_exposes_profile_metadata_and_backward_compatible_typical(real_client) -> None:
    response = real_client.get("/api/config")
    assert response.status_code == 200
    config = response.get_json()
    assert set(config["demo_profiles"]["profiles"]) == {"good", "typical", "high_strain"}
    assert config["demo_profile"] == config["demo_profiles"]["profiles"]["typical"]["values"]
    assert "not diagnoses" in config["demo_profiles"]["usage"]


def test_demo_profiles_receive_genuine_and_ordered_model_predictions(real_client) -> None:
    outputs = {}
    for key, profile in _profiles()["profiles"].items():
        response = real_client.post("/api/predict/numerical", json=profile["values"])
        assert response.status_code == 200
        result = response.get_json()
        assert result["stress_class"] in STRESS_CLASSES
        assert set(result["stress_probabilities"]) == set(STRESS_CLASSES)
        assert sum(result["stress_probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        assert set(result["scores"]) == {"Depression_Score", "Anxiety_Score", "Stress_Score"}
        outputs[key] = result
    severity = {
        key: sum(STRESS_CLASSES.index(label) * probability for label, probability in result["stress_probabilities"].items())
        for key, result in outputs.items()
    }
    assert severity["good"] < severity["typical"] < severity["high_strain"]
    assert outputs["good"]["stress_class"] == "Healthy"
    assert outputs["high_strain"]["stress_class"] == "Severe_Stress"


def test_assessment_ui_contains_profile_controls_and_audio_error_region(real_client) -> None:
    html = real_client.get("/assessment").get_data(as_text=True)
    for profile in ("good", "typical", "high_strain"):
        assert f'data-profile="{profile}"' in html
    assert 'type="button" class="profile-chip' in html
    assert 'id="audio-error"' in html
    assert "Load example profile" not in html


@pytest.mark.parametrize("path", ["/static/css/app.css", "/static/js/app.js", "/static/js/assessment.js", "/static/assets/favicon.svg", "/favicon.ico"])
def test_required_browser_assets_return_http_200(real_client, path: str) -> None:
    response = real_client.get(path)
    assert response.status_code == 200
    assert response.data
