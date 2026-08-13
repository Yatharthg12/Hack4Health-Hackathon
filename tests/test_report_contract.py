from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindfuse.config import METRICS_DIR


ROOT = Path(__file__).resolve().parents[1]


def test_detailed_report_surface_and_print_contract(real_client) -> None:
    page = real_client.get("/assessment")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'id="clinical-report"' in html
    assert "Generate detailed PDF report" in html
    script = (ROOT / "static/js/assessment.js").read_text(encoding="utf-8")
    for required_text in (
        "Executive interpretation and decision rationale",
        "Fused probability distribution",
        "How the modalities influenced the result",
        "Depression, Anxiety and Stress score estimates",
        "Professional review checklist",
        "Technical provenance and limitations",
        "Reviewer documentation",
    ):
        assert required_text in script
    assert "window.MindFuse.buildClinicalReport" in script
    stylesheet = (ROOT / "static/css/app.css").read_text(encoding="utf-8")
    assert "@page { size: A4" in stylesheet
    assert ".clinical-report { display: block !important" in stylesheet


def test_improved_face_artifacts_and_dependent_metrics_are_consistent() -> None:
    face = json.loads((METRICS_DIR / "face.json").read_text(encoding="utf-8"))
    summary = json.loads((METRICS_DIR / "summary.json").read_text(encoding="utf-8"))
    fusion = json.loads((METRICS_DIR / "fusion_weights.json").read_text(encoding="utf-8"))
    assert face["selected_model"] == "hog_rbf_svc_C=1"
    assert face["test_metrics"]["accuracy"] > 0.283
    assert face["test_metrics"]["macro_f1"] > 0.2784
    assert face["test_metrics"]["roc_auc_macro_ovr"] > 0.628
    assert summary["metrics"]["face_macro_f1"] == pytest.approx(face["test_metrics"]["macro_f1"])
    assert fusion["weights"]["face"] == pytest.approx(face["training"]["best_validation_macro_f1"])
    assert summary["fusion_weights"]["face"] == pytest.approx(fusion["weights"]["face"])
