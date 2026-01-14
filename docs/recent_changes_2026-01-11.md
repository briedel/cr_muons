# Recent Changes — 2026-01-11

## Logging & Launcher
- Added tee-based logging in `train_with_restart.sh` to stream to screen and save per-range logs under `logs_training/`.
- Ensured periodic logs flush through tee by using `stdbuf -oL -eL python -u ...`.
- Per-attempt logs include both stdout (`*.out.log`) and stderr (`*.err.log`).

## Training Stability
- Implemented microbatching in multi-file mode to avoid skipping oversize batches; uses contiguous event ranges.
- Added `--max-muons-per-event` and per-event outlier skipping with optional outlier writing (sequential path).
- Fixed `non_blocking` transfers when `--pin-memory` on CUDA.
- Replaced unavailable GPU tracker calls with averaged stats; improved periodic global logs.

## Shuffling & Data Loading
- Activated multi-file shuffle (`--multi-file-shuffle`) with round-robin batching to avoid per-file overfitting.
- Removed global shuffle when multi-file shuffle is enabled to reduce memory pressure.
- Tuned DataLoader parameters: `--num-workers`, `--prefetch-factor`, `--persistent-workers`, `--pin-memory`.
- Background prefetch: `--prefetch-dir`, `--prefetch-ahead`, `--prefetch-max-files`.

## Metrics & Monitoring
- Added Wasserstein gap (`w_gap`) and 500-step moving average (`w_gap_ma_500`) to logs and TensorBoard.
- Periodic global logs every `--log-interval` batches with GPU memory/usage summary.

## Empty Events Handling
- New flag `--drop-empty-events` to skip zero-muon events during microbatching (both multi-file and sequential paths).
- Improves throughput and batch consistency when many events have `count=0`.

## Inference Improvements
- Added `ScalableGenerator.generate_with_counts(conditions, counts)` to produce exactly requested muon counts; returns empty tensors when total count is zero.
- New helper `utils/inference.py::generate_event()` that caps counts (`max_muons_per_event`) and returns empty results when `count=0`.
- New batch inference CLI `utils/batch_infer.py`:
  - Reads primaries from Parquet, normalizes, chooses counts from `predict | column | constant`.
  - Generates muons and writes Parquet (physical units with `--denormalize`).

## Range Automation
- Launcher limits to next N ranges (`RANGES_TO_PROCESS`) and maintains `.current_range` for resume.

## Notes
- Observed improved early/medium training stability (positive `w_ma`), with late-epoch drifts mitigated by microbatching and shuffling.
- Monitor beyond batches ~2600–2800; consider tuning `--critic-steps`, pooling mode (`mean`), or light GP if negative `w_ma` persists.
