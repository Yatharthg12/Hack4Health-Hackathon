# Final Validation Record

Validated on 2026-08-13 with Python 3.13.2 on CPU.

## Executed checks

- Read and extracted all three supplied Word documents, including `Metrics Used.docx`.
- Downloaded all 2,751 objects in the organizer Google Drive manifest; resumed 2,052 after the initial transfer and recorded zero failed downloads.
- Opened/validated the 1,200 selected unique WAV files and 350 face images; no selected media corruption found.
- Validated the 4,000×22 numerical dataset: zero missing values, zero duplicate rows, valid status labels, and valid target ranges.
- Trained/evaluated the facial residual CNN benchmark and validation-selected the small-data HOG RBF-SVM primary.
- Trained/evaluated the actor-disjoint speech residual CNN from random initialization.
- Compared four numerical classifiers and four multi-output regression approaches, selected on validation results, refit, calibrated where applicable, and evaluated on untouched tests.
- Generated all JSON metrics, checkpoints/pipelines, confusion matrices, class distributions, histories, global importance, and regression plots.
- Ran `python -m compileall -q app.py src scripts` successfully.
- Ran `node --check` for every JavaScript file successfully.
- Final stabilization added real endpoint/loader regressions for organizer, mono, stereo, PCM 8/16/24-bit, floating-point, WAVE_FORMAT_EXTENSIBLE, RIFF metadata chunks, browser MIME variants, uppercase suffixes, invalid inputs, and explanation failure isolation.
- Added artifact/API/UI coverage for all three training-only numerical demo profiles and favicon/static assets.
- Ran `python -m pytest -q`: **61 passed** after final refinement.
- Exercised real artifact-backed face, audio, numerical, three-way fusion, and batch APIs. Explanations returned valid PNG data URLs; the fusion distribution summed to 1.0.
- Launched `python app.py`, observed `/api/health` = `ready` with all four artifacts, and received HTTP 200 from `/assessment`, `/performance`, and a generated figure before stopping the smoke-test process.
- Reproduced the browser audio 400 in a fresh Flask process: WAV decode, resampling, 64Ã—249 feature creation, model prediction, and saliency all completed; the optional plot then raised because `mako` was not a registered Matplotlib colormap. The preceding SciPy non-data-chunk warning was not the exception. The plot now uses a Matplotlib-native colormap and explanation rendering is failure-isolated.
- Verified SoundFile/libsndfile primary decoding, SciPy fallback behavior, structured audio errors, `/favicon.ico` HTTP 200, and Good/Typical/High strain profile inference from `artifacts/metrics/demo_profiles.json`.
- Built a dedicated A4 print/PDF report containing the full fused rationale, modality distributions and weights, D/A/S estimates, numerical inputs, local drivers, available explanation images, provenance, limitations, safety review, and sign-off fields.
- Executed fixed-partition model experiments. The face HOG RBF-SVM improved validation macro-F1 and held-out face accuracy/macro-F1/ROC-AUC; expanded numerical candidates were rejected because validation gains did not survive held-out evaluation.
- Ran the complete three-modality judge flow in headless Chrome, generated the populated report, and rendered it through Chromium's PDF engine: 5 A4 pages, 871,362 bytes, all required report sections present, and zero severe browser-console errors.

## Primary held-out results

| Task | Support | Accuracy | Balanced accuracy | Macro-F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Facial emotion HOG RBF-SVM | 53 | 0.3208 | 0.3265 | 0.2973 | 0.7028 |
| Speech emotion CNN | 300 | 0.4367 | 0.4454 | 0.4232 | 0.8218 |
| Numerical stress classifier | 600 | 0.3817 | 0.2819 | 0.2779 | 0.4955 |

The separate requested facial CNN benchmark reached macro-F1 0.1124 on the same 53-image test partition. See `artifacts/metrics/face_cnn.json`.

| Regression target | MAE | MSE | RMSE | R² | Explained variance |
|---|---:|---:|---:|---:|---:|
| Depression | 8.5534 | 98.3295 | 9.9161 | 0.0034 | 0.0034 |
| Anxiety | 6.2023 | 52.3433 | 7.2349 | -0.0129 | -0.0056 |
| Stress | 9.5678 | 124.2629 | 11.1473 | -0.0127 | -0.0126 |

These weak numerical relationships are preserved and visible; no result was fabricated or suppressed.

## Audited anomalies

- The Drive folder contains 2,400 WAV paths but only 1,200 unique seven-field filenames (two copies of each). Training deduplicates by the documented globally unique identifier.
- `Surprised` speech is absent; Disgust has 144 clips. The actual model covers the seven supplied classes. Complete eight-class parsing and stress mapping remain tested for a future complete dataset.
- The face folder has 350 images (50/class), not the documented 28,709. The predeclared small-data path therefore selects a HOG classifier; the current validation winner is an RBF-SVM, while the residual CNN remains implemented, trained, evaluated, and saved.
- Severe Stress has only 128/4,000 numerical rows; the held-out classifier has zero recall for its 19 Severe Stress test rows.
- Numerical classifier ROC-AUC is approximately chance and mean regression R² is slightly negative, indicating limited predictive signal in the supplied numerical predictors/targets.

The machine-readable sources of truth are under `artifacts/metrics/`.
