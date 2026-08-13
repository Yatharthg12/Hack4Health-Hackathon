# Model Card — MindFuse XAI 1.0

## Summary

MindFuse XAI is a hackathon research system for explainable four-level stress decision support and numerical-feature-based Depression/Anxiety/Stress score estimation. It includes independent facial emotion, speech emotion, numerical classification, and multi-output regression components connected through transparent late fusion.

## Intended use

- Demonstrate scientifically defensible multimodal ML and explainable AI.
- Support research discussion and supervised decision-support demonstrations.
- Explore how calibrated distributions and explicit conflict reporting improve transparency.

## Non-intended use

- Medical or psychiatric diagnosis, screening, triage, treatment selection, emergency response, employment/insurance/education decisions, surveillance, or any autonomous high-stakes action.
- Inferring a person's mental condition from face or voice without informed consent.
- Using confidence as a guarantee or an explanation as clinical causality.

## Training data

The organizer supplies independent emotion image, emotion speech, and numerical multimodal datasets. There is no proven cross-dataset identity pairing. Actual local counts and quality findings are recorded in `artifacts/metrics/dataset_audit.json`; organizer data is excluded from Git.

## Models

| Component | Model | Output |
|---|---|---|
| Face | validation-selected HOG RBF-SVM primary plus residual CNN trained/evaluated from scratch | seven calibrated emotion probabilities |
| Speech | compact residual log-Mel CNN trained from scratch | seven observed calibrated emotion probabilities; documented missing class recorded |
| Numerical | validation-selected sklearn classifier with sigmoid calibration | four stress probabilities |
| Severity | validation-selected multi-output regressor | Depression, Anxiety, Stress scores |
| Fusion | reliability- and entropy-adjusted linear opinion pool | four stress probabilities and diagnostics |

No pretrained face/speech model or pretrained weights are used. The supplied folder contains 350 rather than 28,709 documented face images and omits the documented speech `Surprised` class; these discrepancies are explicit artifacts, not synthesized away.

## Metrics

The authoritative measured results are the generated JSON files under `artifacts/metrics/` and visualizations under `artifacts/figures/`. Classification artifacts include all aggregate, per-class, AUC, and confusion metrics required by `Metrics Used.docx`. Regression artifacts include MAE, MSE, RMSE, R², and explained variance per output and in aggregate. The dashboard reads those artifacts directly; it does not contain embedded claims.

The current face primary was selected from a fixed validation-only comparison of class-balanced HOG Logistic Regression and RBF-SVM candidates. Relative to the previous HOG Logistic artifact, held-out accuracy improved from 0.2830 to 0.3208, macro-F1 from 0.2785 to 0.2973, and macro ROC-AUC from 0.6281 to 0.7028 on the unchanged 53-image test partition. Expanded numerical candidates were rejected after they failed to preserve held-out performance.

## Explainability

Face occlusion sensitivity/Grad-CAM, audio input gradients, permutation importance, local median-replacement sensitivity, and complete fusion diagnostics are provided. The detailed printable report carries the complete probability and contribution audit trail plus explicit interpretation boundaries. These methods describe model behavior. They do not establish medical mechanisms or causality.

## Limitations and risks

- Emotion-to-stress mappings are organizer-defined simplifications and can be culturally/contextually wrong.
- Unpaired data prevents learned interaction modeling or joint multimodal performance evaluation.
- Small classes can have unstable estimates even with macro metrics and weighting.
- Face and speech systems are vulnerable to capture quality, occlusion/noise, language/accent, demographics, acted emotion, and distribution shift.
- Numerical associations may reflect how the dataset was constructed, not population clinical relationships.
- Calibration measured in-distribution may not hold elsewhere.

## Ethical considerations

Use requires informed consent, data minimization, strict access controls, purpose limitation, and human oversight. Before any clinical study, audit performance/calibration by relevant subgroups, analyze false-positive/negative harms, test robustness and shift, involve clinicians and people with lived experience, document dataset provenance/licensing, obtain institutional/ethical approval, and define an escalation policy independent of the model.

## Maintenance

Re-run `python scripts/train_all.py` when data, dependencies, preprocessing, model code, or seed changes. Do not compare artifacts from different versions without recording those changes. Treat missing or incompatible artifacts as a failed/degraded model—not permission to fabricate a fallback prediction.
