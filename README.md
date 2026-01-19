# cr_muons

First attempt will be a regressor to get the number of muons in a bundle given a primary (E_primary, Mass_primary, zenith_primary, and slant depth) then a conditional wasserstein gan will take over to learn the kinematics from primary (E_primary, Mass_primary, zenith_primary, and slant depth) and output muons

## Training

Full training documentation (single GPU + single-node multi-GPU DDP), including a full explanation of all CLI flags, is in [training/README.md](training/README.md).

### High-throughput (SGD) example

Pelican streaming + Parquet batch reader + GP disabled + SGD for fastest throughput:

```bash
python training/train.py \
	-i 'pelican://.../testing/0000000-0000999/*.parquet' \
	--use-hf --parquet-batch-reader \
	--prefetch-batches 1000 --num-workers 12 --prefetch-factor 100 --persistent-workers --pin-memory \
	--max-muons-per-batch 200000 --outliers-dir ./outliers_parquet/ \
	--batch-size 48000 --critic-steps 5 --log-interval 100 \
	--tb-logdir ./logs_tensorboard/ --tb-hist-interval 10 \
	--prefetch-dir ./testdata/ --prefetch-delete-after-use --prefetch-max-files 100 --prefetch-ahead 10 \
	--auto-token --checkpoint ./training_checkpoint.json --model-checkpoint ./model_checkpoint.pt \
	--checkpoint-io local --device cuda \
	--optimizer sgd --lr 2e-5 --momentum 0.5 --grad-clip-norm 1e7 \
	--allow-tf32 --lambda-gp 0.0
```

**Performance:** 8.13 batches/s, 390k evt/s, 141k muons/s on single A100 (batch_size=48k, 48k muons/s).

**Why SGD?**

We use SGD instead of Adam for high-throughput training because:

1. **Stability with normalized critic:** After normalizing critic output by √(batch_size) and clamping to [-1000, 1000], SGD with low learning rate (2e-5) provides stable convergence without the adaptive step sizes that can amplify instability.

2. **Throughput:** SGD has minimal optimizer overhead compared to Adam's momentum buffers and per-parameter learning rates, achieving 8-9 batch/s on large batches.

3. **Large gradients handled:** The critic normalization produces gradients in the 7-9×10⁶ range. With `--grad-clip-norm 1e7`, SGD clips these safely while maintaining training progress. Adam's adaptive rates can struggle with this scale.

4. **Momentum tuning:** `--momentum 0.5` provides acceleration through noisy gradients without the oscillation risk of higher momentum values in adversarial training.

**Key tunings:**
- `--lambda-gp 0.0`: Disable GP for maximum speed (critic output clamped to [-1000, 1000] internally).
- `--allow-tf32`: Enable on Ampere+ GPUs for 2-4x speedup with acceptable accuracy tradeoff.
- `--grad-clip-norm 1e7`: Permit gradients up to 1e7 L2 norm; critic output normalization keeps loss stable.
- `--optimizer sgd --lr 2e-5 --momentum 0.5`: Fast SGD updates; use lower LR if losses destabilize.

**Stability notes:**
- Critic output is normalized by √(batch_size) and clamped to [-1000, 1000] to prevent loss explosion.
- If losses spike (c_loss > 1e6), reduce `--lr` (try 1e-5 or 5e-6).
- For better generation quality at cost of speed, re-enable GP: `--lambda-gp 0.01 --gp-every 2`.

### Local files (no Pelican)

Reproduce the speed-up when training from a local Parquet directory:

```bash
python training/train.py \
	-i '/path/to/local/*.parquet' \
	--parquet-batch-reader \
	--device cuda \
	--batch-size 48000 \
	--num-workers 8 --prefetch-factor 2 --persistent-workers --pin-memory \
	--critic-steps 5 \
	--optimizer sgd --lr 5e-5 --momentum 0.5 \
	--gp-max-pairs 4096 --gp-every 2 \
	--log-interval 100 \
	--tb-logdir ./logs_tensorboard/
```

Notes:
- Remove Pelican-specific flags (`--use-hf`, `--prefetch-dir`, `--auto-token`, etc.).
- Adjust `--batch-size` to fit your GPU memory if needed.

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

Launch with torchrun:

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
- In DDP, --batch-size is per GPU (per rank); effective batch is batch_size × world_size.
- Distributed mode currently requires --parquet-batch-reader.

### TensorBoard

Enable logging with:

```bash
python training/train.py ... --tb-logdir logs_tensorboard
tensorboard --logdir logs_tensorboard --port 6006
```

Notes:
- `--tb-run-name` is optional; when omitted, the run directory defaults to a timestamp like `20260105-105330`.
- If you want to view only a subset of runs in one UI, you can create a curated combined logdir via:
	`python utils/merge_tensorboard_runs.py --dst tb_combined --src-root logs_tensorboard --mode symlink --since 20260105-120000`

## Model

The current training code implements a **conditional Wasserstein GAN with gradient penalty (WGAN-GP)** for muon bundle generation.

- **Conditioning (primaries):** the generator and critic are conditioned on per-event primary features (e.g. energy, mass, zenith, slant depth).
- **Outputs:** the generator produces a *ragged* set of muons per event (variable multiplicity). Internally we work with a flattened muon tensor plus an index mapping each muon back to its event.
- **Training objective:** critic is trained with Wasserstein loss plus gradient penalty (WGAN-GP); generator is trained to maximize critic score on generated samples.

Implementation reference: [training/model.py](training/model.py)

## Normalization

* We want to be as close to the range of [-1, 1] for most of the values to make the NN happy. 
* Normalization for Primary
** Energy - log10(E_primary_PeV) - We picked PeV because we range from 10^0 to 10^9 GeV
** Zenith - cos(zenith_rads)
** Mass - log10(Mass_primary_GeV)
** Slant Depth - Slant depth in km
* Normalization for Muon
** Same as primary - log10(Muon_PeV)
** X, Y - Coordinates around the shower axis divided by 500 m

## Documentation

Detailed technical documentation and development history can be found in the [`docs/`](docs/) directory:
- [INSTABILITY_ANALYSIS.md](docs/INSTABILITY_ANALYSIS.md) - Analysis of training stability issues and fixes
- [adaptive_tuning_and_filtering_fixes.md](docs/adaptive_tuning_and_filtering_fixes.md) - Adaptive critic tuning implementation
- [gradient_accumulation_strategy.md](docs/gradient_accumulation_strategy.md) - Memory optimization via gradient accumulation
- Conversation summaries - Historical development notes

### Gradient Accumulation

For large batch training on memory-constrained GPUs, use `--grad-accum-steps`:

```bash
python training/train.py \
    ... \
    --grad-accum-steps 2  # Split each batch into 2 sub-batches
```

This reduces peak VRAM usage by 30-40% at the cost of 15-25% slower training. See [docs/gradient_accumulation_strategy.md](docs/gradient_accumulation_strategy.md) for details.

## Changelog

- 2026-01-19: Code cleanup and refactoring
	- Removed debug comments and cruft from model.py and train.py
	- Added `--grad-accum-steps` CLI argument for gradient accumulation
	- Added comprehensive docstrings to main training functions
	- Created reference documentation in docs/ directory
- 2026-01-09: Major training and performance updates
	- Added robust Pelican prefetch, GPU monitoring, profiler hooks, and consolidated utilities.
	- Integrated multiplicity prediction into the generator and stabilized WGAN-GP gradient penalty.
	- Introduced GP tuning flags (gp-max-pairs, gp-every, gp-sample-fraction) and optimizer selection (Adam/SGD).
	- Removed frequent GPU→CPU syncs, added vectorized GP pairing, and optional profiling.
	- Result: multi-x throughput improvements with low memory footprint.
	- Full summary: see [docs/conversation_summary_2026-01-09.md](docs/conversation_summary_2026-01-09.md)

