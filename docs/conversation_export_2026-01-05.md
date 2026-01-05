# cr_muons — Conversation Export (Reconstructed)

Date: 2026-01-05  
Workspace: `/var/home/briedel/code/ml/cr_muons`

## Important note about fidelity

This document is **not a verbatim transcript** of the chat UI. The assistant does not have access to the full message-by-message conversation history from the client. This export is a **structured reconstruction** based on the conversation context and summary available in the workspace at the time of writing.

If you need a true verbatim transcript, use your chat UI’s “Export chat” / “Copy conversation” feature (if available) and paste it into a file.

---

## 1) High-level goals

- Make training work correctly on `.parquet` inputs (and not attempt to read them as HDF5).
- Support `pelican://` inputs, including wildcard expansion, token flow, and optional prefetch-to-disk.
- Improve throughput: fast Parquet iteration, parallel decode, pinned memory, background prefetch, and better timing logs.
- Add observability: per-batch timing split, throughput metrics, and TensorBoard logging + optional sync.
- Add single-node multi-GPU training via DDP (`torchrun`).
- Diagnose and mitigate gradual GPU memory growth leading to OOM; reduce peak spikes from ragged outliers.
- Implement outlier extraction: write oversize events to a separate Parquet dataset.

---

## 2) Key decisions / conclusions

### Data formats and IO

- `.parquet` and `pelican://` sources are handled via a streaming Parquet/HF-based loader.
- `.h5/.hdf5` remain supported for local paths via `h5py`.
- `pelican://.../*.parquet` wildcard expansion is supported.
- Optional Pelican **prefetch-to-disk** can stage remote files under a local cache directory.

### Performance

- Implemented a fast Parquet path that iterates Arrow record batches and yields already-batched tensors.
- Enabled parallel decode via DataLoader workers for the already-batched Parquet path.
- Added a lightweight background prefetch iterator to overlap input and compute.

### Multi-GPU

- Added a separate single-node DDP entrypoint: `training/train_distributed.py` (launched with `torchrun`).
- Implemented correct sharding across ranks and DataLoader workers to avoid duplicates.

### GPU memory growth (OOM) diagnosis

Observed behavior:

- CUDA “allocated/active” remained relatively flat.
- CUDA “reserved/max_reserved” ratcheted upward over time.

Interpretation:

- Consistent with **allocator caching/fragmentation** triggered by **rare peak allocations** (ragged outlier batches/events), rather than a classic “live tensor leak”.

Mitigation:

- Add microbatching to bound `sum(counts)` per step.
- Add per-event outlier detection and siphon oversize events to a separate Parquet dataset.

---

## 3) Major code changes (by file)

### `training/train.py`

**Correctness / robustness**

- Improved CUDA memory logging (`memory_allocated`, `memory_reserved`, `max_*`, allocator internals, and `mem_get_info`).
- Fixed device-selection structure regression so `--device auto/mps` behaves properly.
- Added CLI alias: `--critic-step` → `--critic-steps`.

**Outlier / OOM mitigations**

- Added microbatching guard:
  - `--max-muons-per-batch` (split ragged batches so each substep has bounded `sum(counts)`).
- Added per-event outlier guard:
  - `--max-muons-per-event` (skip oversize events).
  - If unset/0, outlier threshold falls back to `--max-muons-per-batch`.
- Added outlier capture:
  - `--outliers-dir` writes oversize events to Parquet (one small file per outlier event).
- Fixed a logic bug so outlier filtering triggers on **every batch** (not only when microbatching is active).

**TensorBoard / logging**

- Added optional TensorBoard logging via `--tb-logdir`, with scalars and optional histograms.
- Added optional TensorBoard event sync/copy via `--tb-sync-to` (local or `pelican://`).

**Bug fix**

- Fixed `UnboundLocalError` when `--report-first-batch` was enabled: moved signature logging to after `prims_feats` / `real_muons_feats` are defined.

### `training/train_distributed.py`

- Added/used as the single-node DDP entrypoint (invoked via `torchrun`).

### `training/README.md`

- Expanded docs for:
  - TensorBoard usage with `logs_tensorboard/` layout.
  - Ragged outlier memory stabilization flags.
  - `--critic-step` alias.
  - Run selection/curation using the merge helper.

### `README.md`

- Updated TensorBoard examples to match `logs_tensorboard/` layout.
- Added quick example for curating a subset of runs.

### `utils/merge_tensorboard_runs.py`

- New helper script to combine many TensorBoard runs under one logdir by copying/symlinking.
- Added filters:
  - `--since` (timestamp)
  - `--match` (regex)
  - `--glob` (glob pattern)

---

## 4) How to run (copy/paste)

### TensorBoard (your repo’s default layout)

If training logs to `logs_tensorboard/<run_name>/events.out.tfevents...`, you can view all runs directly:

```bash
tensorboard --logdir logs_tensorboard --port 6006
```

### Curate a subset into a new combined logdir

```bash
python utils/merge_tensorboard_runs.py \
  --dst tb_combined \
  --src-root logs_tensorboard \
  --mode symlink \
  --since 20260105-120000

tensorboard --logdir tb_combined --port 6006
```

### Training (outlier siphon + microbatch guard example)

Example shape (adjust paths/values to taste):

```bash
python training/train.py \
  -i /path/to/data/*.parquet \
  --parquet-batch-reader \
  --device cuda \
  --batch-size 8192 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --persistent-workers \
  --pin-memory \
  --max-muons-per-batch 200000 \
  --max-muons-per-event 50000 \
  --outliers-dir outliers_parquet \
  --tb-logdir logs_tensorboard \
  --log-interval 50
```

---

## 5) Open items / follow-ups

- Validate long-running stability: confirm CUDA `max_reserved` stops ratcheting toward OOM when outliers are removed.
- Optional: consolidate many per-event outlier Parquet “part-*” files into larger shards for easier downstream training.

---

## 6) File index

- Training entrypoint: `training/train.py`
- DDP entrypoint: `training/train_distributed.py`
- Training docs: `training/README.md`
- Repo README: `README.md`
- TensorBoard run merger: `utils/merge_tensorboard_runs.py`
