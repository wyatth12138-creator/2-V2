# UIV-US V2

The training and evaluation entry points build UIV-US with ViT-B from `models/uiv_us/`. This directory contains the model and its components. Video diagnosis adds a mask-derived observation and a causal temporal observer before FLA. The observer learns residual corrections to the per-frame diagnosis features and starts at zero so a compatible V1 checkpoint makes the same predictions before adaptation. Images keep the original path.

## Modes

| `--temporal_mode` | Temporal branch |
|---|---|
| `baseline` | Original FLA without observer |
| `fixed` | Time-aware local trend with fixed noise rules, no wavelet terms in the residual |
| `wavelet` | Causal signed Haar detail only |
| `stft` | Causal Hann-windowed short-time FFT power at the first nonzero bin per scale |
| `adaptive` | Local trend, signed Haar detail and quality-conditioned process and observation noise |
| `projected` | Adaptive observer with local mask perturbation projection, image anchor disabled |
| `anchored` | Projection with contour-based image support |
| `registered` | Local two-frame registration and centroid-only correction, the default for new training |

The video decoder masks invalid frames separately from segmentation prompt availability. Invalid frames do not contribute to FLA or MRTA writes. The classification initialization can load a V1 state dictionary and initialize only the new observer tensors. Raw V1 checkpoints are strictly evaluated with `baseline`. The original source is pinned at [e23dd46](https://github.com/miccai26-1426/UIV-US/tree/e23dd46201a089e1fcd499f006bd16318d2d2c0a), with the upstream license retained.

## Ordered video diagnosis

Prepare `DATA_ROOT/ordered_video_cls/Breast/manifest.csv` with these fields:

```csv
patient_id,case_id,split,path,label
patient-01,clip-01,train,clips/clip-01.npz,0
patient-02,clip-02,val,clips/clip-02.npz,1
patient-03,clip-03,test,clips/clip-03.npz,0
```

Each NPZ contains `frames` (unsigned 8-bit `[T,H,W]` grayscale or `[T,H,W,3]` RGB), optional `timestamps` (`[T]`, finite and strictly increasing, seconds by default), optional `timestamp_unit` (`s` or `frame`), and optional boolean `frame_valid`. Omit timestamps to use original ordinal frame indices. Use stable pseudonymous patient IDs and check isolation across datasets and pretraining sources. The loader checks isolation within each manifest.

Clips over the frame budget are sampled across the archived clip. This evaluation therefore measures **offline classification**. For a prefix experiment, prepare each available prefix before sampling and keep every variant of a case in the same patient split. The Haar/STFT windows operate on sampled frame slots. Their axes do not represent physical frequency when acquisition intervals are irregular. The state transition separately receives the recorded time intervals. The CEUS manifest accepts one enhancement stream with no future-dependent baseline subtraction. Report relative display intensity unless original calibration supports stronger measurement claims.

## Train and independently evaluate

Run from this directory after installing the host PyTorch/torchvision build and `requirements.txt`. Set the encoder input size to match the ViT-B checkpoint. Example on PowerShell:

```powershell
$Python = 'python'
$Data = 'E:\path\to\authorized_prepared_data'
$V1 = 'E:\path\to\trained_v1.pth'
& $Python -B train.py --data_root $Data --init_checkpoint $V1 --classification_only `
  --only_selected --ordered_video_cls --bmode_vid_cls_datasets Breast `
  --video_frames 10 --cls_epochs 10 --temporal_mode registered
& $Python -B test.py --data_root $Data --checkpoint 'E:\path\to\stage2_classification.pth' `
  --only_selected --ordered_video_cls --bmode_vid_cls_datasets Breast --video_frames 10
```

Use `--ceus_vid_cls_datasets Breast` for CEUS with a separate prepared data root, or distinct dataset names for separate modalities. The final trained checkpoint records its mode in the adjacent `run_config.json`. If evaluating a checkpoint without this file, pass its mode explicitly where applicable. Use `--case_predictions` only for authorized pseudonymous score export. Compare two such files with `tools/compare_case_predictions.py --baseline ... --candidate ... --threshold ... --output ...`. Choose the shared threshold on development data. The original two-stage path remains available with selected segmentation and classification data and matching SAM/ResNet initial weights.

Set `UIV_US_PROCESS_ROOT` or pass `--output_root` to choose where runs and reports are saved. The independent test reports AUROC, AUPRC, Brier and confident errors for classification. Per-clip metrics and optional score export support later patient-grouped analysis.

The `anchored` mode corrects the five ROI descriptors entering the observer. The `registered` mode corrects only the two centroid descriptors. It uses past-mask probes and local registration of adjacent raw frames, with fallback on weak or unmatched evidence. Run `tools/benchmark_mask_intervention.py --output ...` to reproduce the synthetic matched-mask experiment.

For the predeclared image-perturbation evaluation use `tools/benchmark_echonet_intervention.py --data-root ... --split VAL --output ...`. A TEST run requires `--exclude-prior-report` to reconstruct and exclude the prior selection.

## Domain-aware training

Use the existing train command with one of these opt-in update rules.

| Argument | Update rule |
|---|---|
| `--domain_update round_robin` | The original task rotation with datasets pooled within each task. |
| `--domain_update pooled` | Compute the sample-count-weighted gradient for the full task in expectation when only some datasets are sampled. |
| `--domain_update balanced` | Sample distinct datasets uniformly within the current task, compute their mean-loss gradients at the same weights and average them for one optimizer step. |
| `--domain_update pcgrad` | Use the same sampling and update budget, with PCGrad applied only to parameter blocks shared by the sampled datasets. Prompt-bank gradients use ordinary averaging. |

`--domains_per_update` defaults to 2. `--domain_steps_per_epoch` fixes the optimizer update budget for matched comparisons. Domain-aware runs keep BatchNorm running statistics fixed and report each dataset separately in `metrics.json`. Domain optimization diagnostics are written to `domain_optimization.json`. The pooled update divides each selected domain coefficient by its inclusion probability before AdamW. Use consistent binary classification definitions across sources and prepare a zero-or-one label for every included case.

For the paired breast-lesion segmentation experiment, run the public-data audit with `tools/audit_segmentation_domains.py`. Use `tools/group_segmentation_split.py` to prepare the group-disjoint exclusion map and supply it to `tools/benchmark_domain_transfer.py`. The benchmark compares single-domain training, sample-count weighting, equal domain weighting and PCGrad with the same in-domain exposure. It uses positive lesion masks and one foreground click. Hold-out datasets appear only in evaluation; use source validation to freeze settings before testing.
