# Dataset Notes

## Source documents reviewed

- `Problem Statement.docx`
- `Dataset_Description.docx`
- `Metrics Used.docx`

The organizer documents describe three independent components and provide a public Google Drive folder. `scripts/download_data.py` resolves the actual folder manifest, downloads with resume/retry/content checks, and never embeds dataset content in application source.

## Expected schemas

### Speech filename

`modality-channel-emotion-intensity-statement-repetition-actor.wav`, such as `03-01-06-01-02-01-12.wav`. Valid emotion IDs are 01 Neutral, 02 Calm, 03 Happy, 04 Sad, 05 Angry, 06 Fearful, 07 Disgust, and 08 Surprised. Actors are 01–24; odd IDs are male and even IDs female according to the supplied description.

### Face labels

0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral. Discovery accepts these IDs or case-insensitive names in the nearest parent directory.

### Numerical columns

Eighteen predictors, `Mental_Health_Status`, `Depression_Score`, `Anxiety_Score`, and `Stress_Score`. Column matching safely adapts case, whitespace, underscores, and punctuation, but all required concepts must exist. Unknown labels and out-of-range targets fail validation.

## Audit policy

The audit reports rather than silently repairs corrupt media. Numerical missing predictors are reportable and compatible with train-only median imputation; missing labels/targets, unknown status labels, and target values outside documented ranges stop training. Duplicate counts are recorded. API inference is stricter and rejects missing/non-finite/out-of-plausibility-range inputs.

## Observed organizer-folder anomalies (2026-08-13 audit)

- Speech: 2,400 WAV paths resolve to two copies of 1,200 globally unique seven-field filenames. The loader keeps the shortest-path copy for every filename. No selected WAV is corrupt.
- Speech classes: Neutral 96; Calm, Happy, Sad, Angry, and Fearful 192 each; Disgust 144; Surprised 0. The model trains the seven observed classes and retains the full eight-class parser/mapping for a future complete folder.
- Face: 350 valid 48×48 images—exactly 50 per class—instead of the documented 28,709. This activates the predeclared small-data HOG primary while still training/evaluating the requested residual CNN benchmark.
- Numerical: exactly 4,000 rows and 22 required columns, zero missing/duplicate rows. Status counts are Healthy 1,629; Mild Stress 1,237; Moderate Stress 1,006; Severe Stress 128. All three score ranges are valid.

The machine-readable authority is `artifacts/metrics/dataset_audit.json`.

## Pairing and leakage

There is no cross-component subject key. Counts and row order are not evidence of correspondence. The application makes no patient-level join during training. Within speech, actor IDs are available and used as groups to prevent speaker leakage. The numerical classification and regression tasks share row partitions for consistent comparison but are trained separately.

## Licensing and repository policy

Dataset licensing/provenance should be confirmed with organizers before redistribution. Raw files and derived audio caches are ignored by Git. The repository contains download/audit/training code and small derived evaluation artifacts; it does not claim ownership of organizer data.
