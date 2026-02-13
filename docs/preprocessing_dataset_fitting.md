# Preprocessing Guide: Chunks vs. Batches

This document explains the data handling strategy used in `fit_and_test_preprocessor.py` when computing global statistics (Scalers, Power Transforms) on large remote datasets (Pelican/S3).

## The Core Concept

To fit a preprocessor on a multi-terabyte dataset without downloading everything or loading it all into RAM, we distinguish between **File Chunks** (download units) and **Data Batches** (statistical units).

### 1. File Chunk (`--prefetch_ahead`)
**Role:** Controls Disk I/O & Network Efficiency.

A **Chunk** is a group of files downloaded from the remote server to the local disk at one time.

*   **Flag:** `--prefetch_ahead N` (or `--multi_file_shuffle`)
    *   *Example:* `--prefetch_ahead 8` means the script downloads 8 `.parquet` files, processes them, and then deletes them before downloading the next 8.
*   **Why?**
    *   Optimizes bandwidth (concurrent downloads).
    *   Keeps local disk usage constant (we assume you can't fit 10TB of data locally, but you can fit 10GB).
    *   Ensures the preprocessor sees a mixture of data if the files are shuffled.

### 2. Data Batch (`--num_fit_batches`)
**Role:** Controls Statistical Precision.

A **Batch** is a tensor of events (e.g., 2,048 muons) fed into the fitting algorithm.

*   **Flag:** `--num_fit_batches M`
    *   *Example:* `--num_fit_batches 100` with a batch size of 2,048 means the scaler will be fitted on approximately $100 \times 2,048 = 204,800$ events.
*   **Why?**
    *   Statistical methods (Mean/Std, Quantiles) converge after seeing a sufficient number of random samples.
    *   You statistically do not need to read 1 billion events to know the mean energy of a muon. A random sample of 1 million (approx 500 batches) is usually sufficient.

---

## Workflow Scenarios

### Scenario A: Incremental Fitting (Standard, MinMax, MaxAbs)
These algorithms support `partial_fit`, meaning they can update their internal state (e.g., running mean/variance) incrementally.

1.  **Download Chunk 1** (e.g., 8 files).
2.  Iterate through all **Batches** in these files.
3.  Update the Scaler.
4.  **Delete Chunk 1** -> **Download Chunk 2**.
5.  Repeat until `total_batches_processed >= num_fit_batches`.

**Outcome:** The scaler learns from a massive range of files while using minimal disk space.

### Scenario B: One-Shot Fitting (Power, Quantile)
Algorithms like Yeo-Johnson (Power) or QuantileTransformer require sorting or global optimization and **cannot** be updated incrementally in the standard Scikit-Learn implementation.

1.  **Download Chunk 1** (e.g., 20 files).
    *   *Note:* You should set `--prefetch_ahead` large enough so this single chunk contains a representative sample of your data.
2.  Read events accumulating up to target **Batches**.
3.  Compute the transform **once** on this accumulated data.
4.  **Stop**. (Downloading Chunk 2 would be useless as we cannot merge the results easily).

**Outcome:** The scaler is fitted on a "Large Random Subsample" defined by the first chunk.

## Cheat Sheet

| Goal | Arg Configuration | Matches Logic |
| :--- | :--- | :--- |
| **Simple Test** | `--prefetch_ahead 1 --num_fit_batches 10` | Download 1 file, fit on a few events. |
| **Production Fit (Standard)** | `--prefetch_ahead 8 --num_fit_batches 5000` | Rotate through files 8 at a time until 5000 batches (~10M events) are seen. |
| **Production Fit (Power/Quantile)** | `--prefetch_ahead 50 --num_fit_batches 5000` | Download 50 files (large sample), fit on them, and stop. |

---

## Why Preprocessing Matters for Neural Spline Flows (NSF)

Neural Spline Flows (and Normalizing Flows in general) are generative models that learn a bijective mapping $f: X \to Z$ between your data distribution $X$ and a simple base distribution $Z$ (typically a Standard Gaussian $\mathcal{N}(0, I)$).

### 1. The "Starting Point" Problem
If your raw physical data (Energy, Zenith, etc.) looks nothing like a Gaussian, the Flow model must spend a significant portion of its expressive capacity just "warping" the marginal distributions to look bell-shaped.
*   **Energy** often follows a power law ($E^{-2.7}$), which is extremely non-Gaussian.
*   **Zenith** might have hard cutoffs or cosine dependencies.

By applying strong preprocessing, we perform a explicit "Gaussianization" step before the neural network starts learning. This lets the Flow focus on the hard part: learning the **complex correlations** between variables, rather than struggling with basic shapes.

### 2. Choosing the Right Method

*   **`StandardScaler` (Z-score):**
    *   *Operation:* Subtract mean, divide by variance.
    *   *Effect:* Centers data at 0 with unit width.
    *   *Use Case:* Good baseline. Essential for neural network stability, but doesn't fix skewness. If your data is a power law, it will still be a power law, just centered at 0.

*   **`PowerTransformer` (Yeo-Johnson):**
    *   *Operation:* Applies non-linear exponential transforms to minimize skew.
    *   *Effect:* Makes long-tailed distributions (like Energy) look much more symmetric/Gaussian.
    *   *Use Case:* **Highly Recommended for Physics Data**. It preserves magnitude relationships better than Quantile transforms while fixing the worst optimization difficulties.

*   **`QuantileTransformer` (The "Nuclear Option"):**
    *   *Operation:* Non-linear mapping based on cumulative density function (CDF).
    *   *Effect:* Forces every individual feature's marginal distribution to be **exactly** a Standard Gaussian.
    *   *Use Case:* Theoretically ideal for Flows. If marginals are perfect Gaussians, the Flow essentially learns the **Copula** (pure dependency structure).
    *   *Warning:* Can be sensitive to outliers and discretization artifacts. The inverse transform can sometimes be unstable at the edges of the range.

### 3. Recommendation for Comparison
When iterating on the `fit_and_test_preprocessor.py`, comparing `standard` vs `power` is usually the most critical test. If the standard scaler leaves the Energy distribution looking like a sharp spike, the Flow will likely fail to converge or require massive `flow_bins` to model it.

