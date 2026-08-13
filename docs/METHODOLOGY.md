# Methodology

## Dataset audit

`scripts/audit_data.py` recursively discovers files, checks the CSV by required schema, validates every speech filename, optionally decodes every WAV/image, and records class counts, actors, image dimensions, missing values, duplicates, target validity, and feature ranges. `scripts/train_all.py` refuses to proceed if a required component is absent.

## Partitions and leakage control

### Face

If paths contain both official `train` and `test` directories, the official test set is preserved and 15% of official training data becomes a stratified validation set. Otherwise a reproducible stratified 70/15/15 split is created. Augmentation is active only in the training dataset object.

### Speech

Actor ID is parsed from the seventh filename field. Two deterministic `GroupShuffleSplit` operations produce actor-disjoint train, validation, and test partitions. Every partition is checked for all eight emotions. This prevents the same speaker's vocal identity from leaking across evaluation partitions.

### Numerical

One stratified 70/15/15 index split is shared by classification and all regression targets. Candidate preprocessing is contained within each sklearn Pipeline, so imputation and scaling never see validation/test data during selection. Candidate selection uses validation only. The selected estimator is refit on train+validation, then evaluated once on test.

## Preprocessing

- **Face:** grayscale conversion and 48×48 resize. At 1,000+ audited images, the CNN path uses tensor normalization to `[-1,1]` and train-only flip/affine/photometric augmentation. Below 1,000 images, a predeclared small-data path computes 900-dimensional 2×2-block L2-Hys HOG; the supplied 350-image subset triggered that path.
- **Speech:** RIFF/WAVE signature validation, SoundFile/libsndfile decoding with SciPy fallback, deterministic channel mean, finite/DC/silence checks, peak normalization, polyphase 16 kHz resampling, four-second pad/truncate, Hann STFT (`n_fft=512`, hop 256), 64 triangular Mel filters, log power, per-clip standardization. Cached features are invalidated by path/size/mtime fingerprints.
- **Numerical:** median imputation with missing indicators and z-standardization inside the candidate Pipeline.

## Optimization and selection

Neural branches use AdamW, weighted cross-entropy, norm clipping, validation macro-F1 checkpointing, ReduceLROnPlateau, and patience-based early stopping. Class weights use square-root inverse frequency to support rare classes without the instability of full inverse weights. Both branches fit one positive temperature to validation logits using cross-entropy. The small-data face primary compares a fixed grid of class-balanced HOG Logistic Regression and RBF-SVM candidates on validation macro-F1 and fits decision-preserving temperature scaling. The selected primary is `hog_rbf_svc_C=1`.

Tabular candidates are intentionally conventional, strong, and auditable. Classification ranks macro-F1 first and balanced accuracy second. Regression ranks mean RMSE normalized by each documented target range.

## Evaluation

Classification reports accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, one-vs-rest macro ROC-AUC when valid, confusion matrix, and per-class precision/recall/F1/support. Regression reports MAE, MSE, RMSE, R², and explained variance independently for Depression, Anxiety, and Stress, plus arithmetic means.

## Explainability

- **Global numerical:** feature permutation on held-out data with macro-F1 scoring and repeated seeded shuffles.
- **Local numerical:** compare current expected stress severity with a counterfactual replacing one feature at a time by its training median. The signed delta is labelled risk-increasing or protective/lower-risk; it is sensitivity, not causality.
- **Face:** the selected HOG primary uses predicted-class probability decrease under local mean occlusion. The CNN path uses Grad-CAM at the final residual convolution. Explanations always correspond to the model that produced the prediction.
- **Audio:** absolute gradient of the predicted emotion logit with respect to standardized log-Mel input, displayed alongside waveform and spectrogram.
- **Fusion:** every effective weight, base reliability, sample confidence, modality class/distribution, agreement, uncertainty, and conflict flag is returned.

Optional rich explanation rendering is isolated from core prediction. If a visualization fails after inference, the genuine prediction remains available and the response marks the explanation unavailable.

## Printable decision-support report

The browser constructs a dedicated A4 report from the exact completed session state. It records the winning and runner-up fused probabilities, margin, per-modality complete distributions, held-out reliability, entropy confidence, normalized weights, D/A/S estimates, numerical inputs, local sensitivities, available explanation figures, system/session provenance, limitations, safety boundaries, and reviewer sign-off fields. The report is explicitly non-diagnostic and contains no invented patient identity or hard-coded interpretation.

## Fusion mathematics

Each modality's validation macro-F1 is persisted as `r_m`. With stress probability vector `p_m`, normalized entropy and sample confidence are:

```text
u_m = -Σ p_m,k log(p_m,k) / log(4)
c_m = 1 - u_m
```

Effective weights are `r_m × (0.5 + 0.5c_m)`, normalized over only available modalities. The output is a linear opinion pool. Pairwise agreement is `1 - TV(p_i,p_j)` averaged by pairwise contribution weight, where total-variation distance is half the L1 distance.

## Reproducibility

Seed 42 covers Python, NumPy, sklearn, and PyTorch. cuDNN benchmark mode is off and deterministic mode is enabled where applicable. Metrics store timestamps, seed, partitions/counts, selected candidate, histories, calibration temperatures, evaluation values, and model configuration. Results come from scripts, never HTML constants.
