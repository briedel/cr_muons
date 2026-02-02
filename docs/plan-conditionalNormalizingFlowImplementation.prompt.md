# Plan: Conditional Normalizing Flow Implementation

This plan outlines the implementation of a Conditional Normalizing Flow (CNF) for cosmic ray muon simulations in IceCube, leveraging the `zuko` library.

## 1. Dependencies

- **Action**: Add `zuko` to `requirements.txt`.
- **Reason**: `zuko` provides high-quality, reliable implementations of Neural Spline Flows (NSF) and Masked Autoregressive Flows (MAF) that integrate seamlessly with PyTorch.

## 2. Implementation: `src/models/flow_module.py`

Rewrite the `MuonFlow` class to implement the generative model.

### 2.1 Architecture
The model will consist of two distinct components sharing the same conditional inputs (Primary Energy, Zenith, etc.):

1.  **Multiplicity Network ($P(N|\mathbf{c})$)**:
    -   A Multi-Layer Perceptron (MLP).
    -   **Input**: Condition vector $\mathbf{c}$ (dim `cond_dim`).
    -   **Output**: Predicted number of muons $N$ (or parameters of a count distribution, e.g., Poisson rate $\lambda$).
    -   **Purpose**: To determine how many muons to sample for a given air shower.

2.  **Flow Network ($P(\mathbf{x}_\mu|\mathbf{c})$)**:
    -   **Type**: Neural Spline Flow (`zuko.flows.NSF`).
    -   **Input**: Muon features $\mathbf{x} = \{E, x, y, t\}$ (dim `feat_dim`).
    -   **Context**: Condition vector $\mathbf{c}$ (dim `cond_dim`).
    -   **Purpose**: To model the continuous properties of individual muons.
    -   **Assumption**: Muons are treated as independent samples conditioned on the shower properties (Factored distribution approximation for the bundle).

### 2.2 Training Logic (`training_step`)
The DataModule provides "ragged" batches where muons are flattened.

-   **Data Inputs**: `flat_muons` (Shape: $[M, D]$), `batch_idx` (Shape: $[M]$), `conditions` (Shape: $[B, C]$), `counts` (Shape: $[B]$).
-   **Step**:
    1.  **Count Loss**:
        -   Pass `conditions` through Multiplicity Network.
        -   Calculate loss against true `counts` (e.g., NLL of Poisson or MSE of log-counts).
    2.  **Flow Loss**:
        -   Expand conditions for every muon: `expanded_conds = conditions[batch_idx]`.
        -   Calculate log-likelihood: `ll = self.flow(flat_muons, context=expanded_conds).log_prob()`.
        -   Minimize NLL: `-ll.mean()`.
    3.  **Total Loss**: `loss = flow_loss + lambda * count_loss`.

### 2.3 Sampling Logic (`predict_step`)
To generate new events:

-   **Input**: `conditions`.
-   **Step**:
    1.  Predict $N$ for each event using Multiplicity Network.
    2.  Sample $N$ muons from the Flow, conditioning on the respective event parameters.
    3.  Format the output to match the expected structure (ragged list of tensors).

## 3. Data Alignment and Future Work

-   **Current Features**: The model will use existing features: Energy ($E$), Position ($x, y$), and Time ($t$).
-   **Missing Feature**: The research documentation emphasizes **Direction** (Zenith/Azimuth relative to shower axis). This is currently missing from the HDF5 dump.
    -   *Immediate Action*: Proceed without Direction.
    -   *Future Action*: Update `src/utils/data_converters/dump_muonitron_data.py` to extract and save local angular coordinates.
