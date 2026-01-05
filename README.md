# cr_muons

First attempt will be a regressor to get the number of muons in a bundle given a primary (E_primary, Mass_primary, zenith_primary, and slant depth) then a conditional wasserstein gan will take over to learn the kinematics from primary (E_primary, Mass_primary, zenith_primary, and slant depth) and output muons

## Training

Full training documentation (single GPU + single-node multi-GPU DDP), including a full explanation of all CLI flags, is in [training/README.md](training/README.md).

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
