# Training: single GPU + distributed (DDP)

This repo contains two entrypoints:

- `training/train.py`: single-process training (single GPU or CPU).
- `training/train_distributed.py`: **single-node** multi-GPU training via PyTorch DistributedDataParallel (DDP), launched with `torchrun`.

The code supports both local files and `pelican://` URIs, and supports `.parquet` (recommended) and local `.h5/.hdf5`.

## Quick start

### Single GPU (recommended)

Fast Parquet reader + parallel decode:

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
  --log-interval 50
```

### Single node, multi-GPU (DDP)

**Requires** Parquet fast-path:

```bash
torchrun --standalone --nproc_per_node 8 training/train_distributed.py \
  -i /path/to/data/*.parquet \
  --parquet-batch-reader \
  --batch-size 4096 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --persistent-workers \
  --pin-memory \
  --log-interval 50
```

Notes:
- In DDP, `--batch-size` is **per GPU** (per rank). Effective global batch is `batch_size * world_size`.
- The Parquet batch reader is sharded across **DDP ranks** and **DataLoader workers** to avoid duplicates.

## What we added (high-level)

### Data / loader correctness

- Automatic routing by file format so `.parquet` is not opened by HDF5 (`h5py`).
- `pelican://.../*.parquet` wildcard expansion and normalization.
- Clearer errors when local paths are missing (e.g., not mounted in a container).

### Performance

- Fast Parquet batch reader (`--parquet-batch-reader`) that iterates Arrow record batches and yields already-batched tensors.
- Parallel decode with `--num-workers` (multiprocessing DataLoader workers) for the Parquet batch reader.
- Optional background batch prefetch queue (`--prefetch-batches`) to overlap decode with GPU compute.
- Load/step timing breakdown and throughput metrics printed to stdout.

### Robustness

- Primaries normalization tolerates 4-column vs 5+/6-column layouts.
- Zero-muon events are allowed; empty batches are skipped safely.

### Memory stability for ragged outliers

Ragged batches can have rare events with extremely large muon multiplicity. Those outliers can spike peak GPU memory and cause PyTorch to reserve more memory (allocator caching), which can lead to eventual OOM over long runs.

Mitigations in `training/train.py`:

- `--max-muons-per-batch`: microbatching guard; splits a ragged batch into smaller substeps so each substep has bounded `sum(counts)`.
- `--max-muons-per-event`: outlier guard; skips any single event with `count > threshold`.
- `--outliers-dir`: if set, writes skipped outlier events to a separate Parquet dataset (one small file per outlier event) so you can train on them later on a larger GPU.

### Training control

- `--batch-size` CLI.
- `--critic-steps` (WGAN-style multiple critic updates per generator update).

### TensorBoard

- Optional TensorBoard logging (`--tb-logdir`) with:
  - Scalars: losses + perf metrics
  - Optional histograms: primaries, counts, real/fake muon features
- Optional periodic sync/copy of TensorBoard event files (`--tb-sync-to`) to either a local directory or `pelican://...` destination.

### Torch compile

- Optional `torch.compile` support with per-model control.
- Important caveat: WGAN-GP gradient penalty requires higher-order gradients (“double backward”), so compiling the **critic** is not supported when `--lambda-gp > 0`.

## TensorBoard usage

Write logs locally:

```bash
python training/train.py ... \
  --tb-logdir logs_tensorboard \
  --tb-log-interval 50 \
  --tb-hist-interval 1000
```

Run the UI:

```bash
tensorboard --logdir logs_tensorboard --port 6006
```

Notes:
- `--tb-run-name` is optional; when omitted, the run directory defaults to a timestamp like `20260105-105330`.

View only a subset of runs by creating a curated combined logdir:

```bash
python utils/merge_tensorboard_runs.py \
  --dst tb_combined \
  --src-root logs_tensorboard \
  --mode symlink \
  --since 20260105-120000

tensorboard --logdir tb_combined --port 6006
```

Other selection options:

- Regex: `--match '^20260105-13'`
- Glob: `--glob '20260105-13*'`

Optional sync to Pelican (writes locally first, then uploads changed event files):

```bash
python training/train.py ... \
  --tb-logdir runs \
  --tb-run-name exp01 \
  --tb-sync-to pelican://osg-htc.org/icecube/.../tb \
  --tb-sync-interval 60 \
  --tb-io pelican
```

## Pelican prefetch cache

When reading `pelican://` inputs, you can stage them to local disk:

```bash
python training/train.py ... \
  --prefetch-dir /tmp/pelican_cache \
  --prefetch-ahead 4
```

Delete staged files after they are used:

```bash
python training/train.py ... \
  --prefetch-dir /tmp/pelican_cache \
  --prefetch-ahead 4 \
  --prefetch-delete-after-use
```

In DDP mode (`train_distributed.py`): rank 0 handles downloads; other ranks wait for the cached file to appear to avoid duplicating downloads.

## Torch compile usage

Base switch:

```bash
python training/train.py ... --torch-compile reduce-overhead
```

Per-model control:

```bash
python training/train.py ... \
  --torch-compile off \
  --torch-compile-gen reduce-overhead \
  --torch-compile-critic off
```

Notes:
- If `--lambda-gp > 0`, `--torch-compile-critic` must be `off` (or left `auto` to be forced off).

## CLI reference (single-process)

These options are defined in `training/train.py` (via `build_arg_parser()`).

### Inputs

- `-i, --infiles`: Input file paths/URIs. Can be local paths or `pelican://...` URIs. Wildcards are supported.

### Loader selection

- `--use-hf`: Force Hugging Face streaming dataset path (slower than the Parquet batch reader; mainly for compatibility).
- `--parquet-batch-reader`: Use the fast Arrow record-batch Parquet reader that yields already-batched tensors.

### DataLoader / decode parallelism

- `--batch-size`: Batch size (events per step). For `--parquet-batch-reader`, this is the Arrow record-batch size.
- `--num-workers`: DataLoader worker processes for parallel decode (supported with `--parquet-batch-reader`).
- `--prefetch-factor`: DataLoader prefetch factor per worker (only used when `--num-workers > 0`).
- `--persistent-workers`: Keep worker processes alive between iterations (only used when `--num-workers > 0`).
- `--pin-memory`: Use pinned host memory in DataLoader; enables non-blocking host→GPU transfers.
- `--prefetch-batches`: Background-thread prefetch of already-batched items from the DataLoader iterator.

### OOM guards (ragged batches)

- `--max-muons-per-batch`: If >0, split a batch into microbatches so each substep has `sum(counts) <= threshold`.
- `--max-muons-per-event`: If >0, skip any single event with `count > threshold`. If 0, the per-event threshold falls back to `--max-muons-per-batch`.
- `--outliers-dir`: If set, write skipped outlier events to Parquet under this directory (one file per outlier event).

### Logging / observability

- `--log-interval`: Print progress every N batches (0 disables).
- `--report-first-batch`: Print a one-time per-file summary of the first non-empty batch (sanity check that data changes across files).

### In-memory caching (advanced)

- `--memory-cache-mb`: Process-local LRU cache of Parquet *file bytes* in RAM. Mostly useful when repeatedly opening the same few files; not shared across DataLoader workers.

### Pelican prefetch-to-disk

- `--prefetch-dir`: Local directory to stage `pelican://` inputs.
- `--prefetch-max-files`: Limit number of `pelican://` inputs to prefetch.
- `--prefetch-ahead`: Prefetch N files ahead in a background thread.
- `--prefetch-delete-after-use`: Delete cached staged files after each file finishes training.

### Pelican auth / federation

- `--federation-url`: Federation URL (e.g. `pelican://osg-htc.org`). Usually inferred from input URIs.
- `--token`: Bearer token for Pelican.
- `--auto-token`: Use device-flow helper to fetch a token if `pelican://` inputs are present.
- `--pelican-scope-path`: Token scope path. If omitted, inferred from the first `pelican://` input.
- `--pelican-oidc-url`: OIDC issuer URL for device flow.
- `--pelican-auth-cache-file`: Cache file for device flow credentials.
- `--pelican-storage-prefix`: Prefix stripped when inferring scope path.

### Checkpointing

- `--checkpoint`: JSON file tracking processed files.
- `--model-checkpoint`: Model checkpoint file.
- `--checkpoint-io`: Where to read/write checkpoints: `auto` (based on path), `local`, or `pelican`.

### Device selection

- `--device`: `auto`, `cpu`, `cuda`, `rocm`, `mps`.

### Debug / inspection

- `--print-file-contents`: Print a preview of each input file and exit.
- `--print-max-events`: Max number of events to print per file in `--print-file-contents` mode.

### Model hyperparameters

- `--cond-dim`: Condition dimension (primaries features).
- `--feat-dim`: Muon feature dimension.
- `--latent-dim-global`: Global latent dimension.
- `--latent-dim-local`: Local latent dimension.
- `--hidden-dim`: Hidden dimension.

### Training hyperparameters

- `--lambda-gp`: Gradient penalty weight for WGAN-GP.
- `--critic-steps`: Number of critic updates per generator update.
- `--critic-step`: Alias for `--critic-steps`.

### torch.compile

- `--torch-compile`: Base compile mode (`off`, `default`, `reduce-overhead`, `max-autotune`).
- `--torch-compile-gen`: Generator compile mode (`auto` uses `--torch-compile`).
- `--torch-compile-critic`: Critic compile mode (`auto` uses `--torch-compile`, but will be forced off when `--lambda-gp > 0`).

### TensorBoard

- `--tb-logdir`: Enable TensorBoard logging and write logs under this directory.
- `--tb-run-name`: Subdirectory name under `--tb-logdir` (default: timestamp).
- `--tb-log-interval`: Write scalar metrics every N batches.
- `--tb-hist-interval`: Write histograms every N batches (0 disables).
- `--tb-max-muons`: Cap how many muons are included in histogram dumps.
- `--tb-sync-to`: Optional sync destination (local dir or `pelican://` URI prefix).
- `--tb-sync-interval`: Seconds between sync passes.
- `--tb-io`: Sync destination IO selection: `auto`, `local`, or `pelican`.

## CLI reference (distributed)

`training/train_distributed.py` reuses the exact same CLI as `training/train.py` and adds:

- `--dist-backend`: `nccl` (default) or `gloo`. Use `nccl` for NVIDIA GPUs.

Launch via `torchrun`:

```bash
torchrun --standalone --nproc_per_node <N> training/train_distributed.py ...
```

Distributed notes:
- Distributed mode currently requires `--parquet-batch-reader`.
- Rank 0 writes checkpoints and prints progress.
- Input sharding is handled inside the Parquet batch reader to avoid duplicates across ranks and DataLoader workers.
