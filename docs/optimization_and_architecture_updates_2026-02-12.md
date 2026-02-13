# Optimization and Architecture Updates (2026-02-12)

## 1. Issue Identification: The "1.67" Loss Plateau
During the training of the Conditional Normalizing Flow (`MuonFlow`), we observed a persistent plateau in the training loss around **1.67**.
- **Behavior**: The loss would drop rapidly in the first few epochs and then hit a hard wall.
- **Learning Rate**: Even when the scheduler aggressively reduced the learning rate (from `1e-4` down to `2.5e-5`), the loss did not improve.
- **Interpretation**: This pattern usually indicates that the model has successfully learned the **marginal distribution** (the average "shape" of a muon bundle) but is failing to learn the **conditional dependence** (how the shape changes exactly based on Energy, Zenith, etc.). It settles for predicting the "mean" shower for everyone.

## 2. Hypothesis & Diagnosis
We investigated two primary causes for this plateau:
1.  **Posterior Collapse**: The Flow network might be ignoring the conditional inputs (`conditions`), effectively treating them as noise.
2.  **Capacity Bottleneck**: The network architecture might be too small or simplistic to represent the complex, high-frequency details of the muon distribution, even if it "sees" the conditions.

### Diagnostic Monitoring
To verify Hypothesis #1, we added granular monitoring hooks to `src/models/flow_module.py`:
- `mon/ctx_std`: The standard deviation of the context embeddings. If this is ~0, the network is ignoring inputs.
- `mon/mult_pred_...`: Comparison of predicted vs actual multiplicity.

**Finding**:
- `mon/ctx_std` remained around **0.30**, verifying that the network **was** responding to different inputs.
- `mon/mult_pred_mean` closely tracked `mon/mult_target_mean` (0.81 vs 0.80), maximizing confidence in the conditioning signal.

**Conclusion**: The conditioning signal is valid. The plateau is caused by **Hypothesis #2 (Capacity Bottleneck)**. The model "knows" what to do but lacks the neurons/depth to execute the complex transformation required to map a Gaussian to the sharp Muon distribution.

## 3. Architecture Changes

### A. Context Embedding Network
We replaced the raw conditional input injection with a dedicated MLP (Multi-Layer Perceptron) embedding network.
- **Before**: Raw conditions (4 dims: LogE, CosZ, etc.) were passed directly into the Flow.
- **After**: A 3-layer MLP targets a high-dimensional features space.
    - `Linear(4 -> 128) -> ELU -> Linear(128 -> 128) -> ELU -> ...`
- **Benefit**: Allows the model to learn non-linear combinations of physical parameters (e.g., "High Energy AND Vertical") before the Flow tries to use them.

### B. Capacity Scaling (The "Wide & Deep" Fix)
We significantly increased the expressivity of the model configuration in `train_lightning_flow.sh`:

| Hyperparameter | Old Value | New Value | Reasoning |
| :--- | :--- | :--- | :--- |
| **`--hidden_dim`** | 256 | **512** | Wider layers allow the network to capture more disparate features/modes in the distribution. |
| **`--flow_transforms`** | 4 | **8** | More transformation steps allow for more complex warping of the probability space (crucial for sharp peaks). |
| **`--flow_bins`** | 32 | **64** | Increasing the spline bins gives the model finer resolution to model sharp edges or cutoffs in the data. |
| **`context_dim`** | 64 | **128** | Increased the embedding size to carry more information from the physical conditions to the flow. |

## 4. Summary of Code Updates

### `src/models/flow_module.py`
- Added `self.context_net` architecture.
- Updated `training_step` to log `mon/` metrics.
- Increased default `context_embedding_dim` to 128.

### `train_lightning_flow.sh`
- Updated arguments to reflect the higher capacity requirements.

## 5. Next Steps
- Monitor the loss with the new capacity. We expect it to break the **1.60** barrier.
- If VRAM becomes an issue due to the size increase (512 width), we may need to increase `accumulate_grad_batches` further or slightly reduce `batch_size`.

## 5. Alternative Base Distributions
The user asked if we could use a distribution other than a Gaussian.

### Why Gaussian?
The Standard Normal (Gaussian) distribution is the default choice for Normalizing Flows because:
1.  **Infinite Support**: It assigns non-zero probability to all real numbers (unlike Uniform), preventing `NaN` or `Infinity` losses if a data point falls outside expected bounds.
2.  **Smoothness**: The gradients are well-behaved everywhere, making optimization stable.
3.  **Maximum Entropy**: For an unbounded variable with fixed variance, the Gaussian makes the fewest assumptions (is the most "random"), letting the Flow layers learn the structure.

### Alternatives
1.  **Student-T Distribution**:
    - **Use Case**: Data with "Heavy Tails" (outliers that are far from the mean).
    - **Pros**: More robust to outliers than Gaussian.
    - **Cons**: Gradients can be smaller/harder to optimize near the center.
    
2.  **Box/Uniform Distribution**:
    - **Use Case**: Data that is strictly strictly bounded (e.g., pixel values [0, 255]).
    - **Pros**: Matches physics of bounded limits exactly.
    - **Cons**: **Dangerous for training.** If a single data point falls outside the trained box, the probability is 0 -> Log-Likelihood is -Infinity -> Gradient Explodes. Given our data normalization (LogE) extends effectively to himBHs\infty$ for low energy, a hard box is risky.

### Recommendation
Sticking to a **Gaussian Base** is recommended for training stability. To handle "sharp" or "box-like" data, the standard solution is to **increase the number of transforms** (capacity), which we have done. This allows the flow to learn a "Pseudo-Box" function that looks like a box but has soft Gaussian tails for stability.

## Notes
Trying student T with degrees of freedom between 3 and 5
tail transform of the gaussian
mixture of gaussians?
range of variables between -3,3 or -5,5

posterior collapse

for the muon features. primary features
quantile transform 
yeo-johnson transform
standard transforms
apply these to the normalized quanitites we already have, and/or just log10() for just E





python3 src/utils/visualize_flow.py \
    --ckpt_path logs_tensorboard_flow/YOUR_RUN_DATE/checkpoints/last.ckpt \
    --data_dir "testdata_flow/icecube/**/*.parquet" \
    --output_dir images_flow \
    --num_events 2000