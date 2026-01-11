# Conversation Summary — 2026-01-09

## Overview
We improved training reliability, organization, and performance for the WGAN-GP muon generator, added robust Pelican handling and GPU monitoring, integrated multiplicity prediction into the generator, and achieved a multi‑x throughput speedup (ultimately ~5–6× per step with SGD), all while keeping memory stable.

## Key Changes
- Monitoring
  - Time-averaged GPU stats and memory metrics; optional TensorBoard logging.
  - Implemented in training/utils/gpu_monitor.py; integrated into training/train.py.
- Pelican IO & Prefetch
  - Wildcard expansion, token device-flow helper with importlib fallback, prefetch to local cache, background prefetcher with ahead window.
  - Implemented in training/utils/pelican_utils.py; wired via training/train.py flags.
- Utilities Reorg
  - Consolidated helpers under training/utils with a central __init__.py for clean imports.
- Model & Training Loop
  - ScalableGenerator now predicts multiplicity from conditions; removed standalone multiplicity module.
  - train_step_scalable returns (critic_loss, generator_loss, multiplicity_loss) and logs m_loss.
  - Gradient penalty compute_gp_flat stabilized for empty fake sets; optimized pairing:
    - First: per-event aligned subsets; then: vectorized indices using argsort + bincount offsets.
  - Generator forward hardened: sanitized multiplicity with nan_to_num; counts kept on-device; removed CPU syncs and non-negative exception.
- Runtime & Logging
  - Added m_loss to progress and TensorBoard; optional histograms.
  - Added profiler hook --profile-steps to print CPU/CUDA hotspots.
- Optimizer & Checkpoints
  - Added optimizer selection: --optimizer {adam, sgd}, --lr, --momentum.
  - Checkpoint loader now skips optimizer state on mismatch (e.g., switching Adam→SGD) while restoring model weights.
- GP Controls (performance):
  - --gp-max-pairs (default 4096), --gp-every (default 2), --gp-sample-fraction; large reductions in GP overhead.

## Performance Improvements
- Before: ~0.63–0.87 batch/s, 1.1–1.5 s/step, GPU usage ~10%.
- After eliminating .item() syncs and vectorizing GP pairing: ~1.1–1.7 batch/s, ~0.56–0.9 s/step.
- With SGD (lr tuned): ~5.8 batch/s, ~0.14 s/step, ~103k muons/s, low VRAM (~39–500 MiB depending on settings).

### Profiling Findings
- Initial bottleneck: thousands of .item() calls causing GPU→CPU syncs.
- Fixes: batched stats in training/train.py; moved GP loop to CPU tensors for counts/offsets in training/model.py.
- Remaining costs: optimizer step, scatter reduce backward, host↔device copies. SGD reduces optimizer overhead substantially.

## New/Updated CLI Options
- GP controls
  - --gp-max-pairs 4096 (default): cap interpolated pairs for GP per step.
  - --gp-every 2 (default): compute GP every N critic steps.
  - --gp-sample-fraction 0.0 (default): optional fractional sampling of GP pairs.
- Optimizer
  - --optimizer {adam, sgd} (default: adam)
  - --lr 1e-4 (adjustable; for SGD lower LR recommended initially)
  - --momentum 0.9 (for SGD)
- Profiling
  - --profile-steps N: print CPU/CUDA kernel table for first N steps.

## Example Commands
- High-throughput SGD (tuned):
  ```bash
  python training/train.py -i 'pelican://.../*.parquet' \
    --use-hf --parquet-batch-reader \
    --prefetch-batches 1000 --num-workers 12 --prefetch-factor 100 --persistent-workers --pin-memory \
    --max-muons-per-batch 200000 --outliers-dir ./outliers_parquet/ \
    --batch-size 48000 --critic-steps 5 \
    --tb-logdir ./logs_tensorboard/ --tb-hist-interval 10 \
    --prefetch-dir ./testdata/ --prefetch-delete-after-use --prefetch-max-files 100 --prefetch-ahead 10 \
    --auto-token --checkpoint ./training_checkpoint.json --model-checkpoint ./model_checkpoint.pt \
    --checkpoint-io local --device cuda \
    --optimizer sgd --lr 5e-5 --momentum 0.5 \
    --gp-max-pairs 4096 --gp-every 2
  ```
- Quick profiling of first 2 steps:
  ```bash
  ... --profile-steps 2
  ```

## Notable Files
- training/model.py: ScalableGenerator (with multiplicity), ScalableCritic, compute_gp_flat, train_step_scalable
- training/train.py: CLI, dataloaders, logging, TB, profiler, optimizer selection, GP controls
- training/utils/pelican_utils.py: pelican expansion/prefetch/token handling
- training/utils/gpu_monitor.py: GPU usage tracker & CUDA mem stats
- training/utils/checkpoint_io.py: load/save checkpoints; tolerant optimizer loading

## Recommendations & Notes
- SGD Stability: If NaNs occur, lower LR (e.g., 5e-5) and/or momentum (e.g., 0.5). Start conservative, then scale up.
- GP Cost: Defaults (max_pairs=4096, every=2) are good for throughput. Increase gradually if training quality needs it.
- torch.compile: Consider enabling for generator only to reduce Python overhead (critic compile off when GP>0).
- Utilization vs Quality: Wider models or higher GP work increase utilization but also cost. Focus on throughput for 2M files; revisit width once training stabilizes.

## Outcome
- Stable run with robust IO, improved monitoring and logging, and a large performance uplift via profiling-guided fixes and SGD, making large‑scale training feasible.
