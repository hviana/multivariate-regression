Model: # 🧠 ESNRegression

<div align="center">

**Echo State Network for Multivariate Autoregressive Regression**

_with RLS Online Learning & Welford Normalization_

[📦 JSR Package](https://jsr.io/@hviana/multivariate-regression) •
[📂 GitHub](https://github.com/hviana/multivariate-regression) •
[👤 Author: Henrique Emanoel Viana](https://github.com/hviana)

</div>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
  - [🎯 Reservoir Parameters](#-reservoir-parameters)
  - [📊 Training Parameters](#-training-parameters)
  - [🛡️ Robustness Parameters](#️-robustness-parameters)
  - [🔧 Utility Parameters](#-utility-parameters)
- [📖 API Reference](#-api-reference)
- [🎓 Use Case Examples](#-use-case-examples)
- [🔬 Parameter Optimization Guide](#-parameter-optimization-guide)
- [💾 Serialization](#-serialization)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔄 Online Learning

- **Recursive Least Squares (RLS)** algorithm
- Single-pass training without storing data
- Continuous adaptation to new patterns
- Memory-efficient for streaming data

</td>
<td width="50%">

### 🌀 Echo State Network

- **Reservoir Computing** paradigm
- Sparse, randomly initialized reservoir
- Spectral radius control for dynamics
- Leaky integrator neurons

</td>
</tr>
<tr>
<td width="50%">

### 📈 Multivariate Support

- Handle **multiple correlated time series**
- Joint prediction of all features
- Cross-feature dependencies captured
- Autoregressive roll-forward prediction

</td>
<td width="50%">

### 🛡️ Robustness Features

- **Welford online normalization**
- Outlier detection & downweighting
- Uncertainty quantification
- Confidence intervals

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Installation

```typescript
import { ESNRegression } from "https://esm.sh/jsr/@hviana/multivariate-regression";
```

### Basic Usage

```typescript
// 1️⃣ Create model with default configuration
const model = new ESNRegression();

// 2️⃣ Prepare your time series data
const coordinates = [
  [1.0, 2.0, 3.0], // t=0: [feature1, feature2, feature3]
  [1.5, 2.5, 3.5], // t=1
  [2.0, 3.0, 4.0], // t=2
  [2.5, 3.5, 4.5], // t=3
  [3.0, 4.0, 5.0], // t=4
  // ... more data points
];

// 3️⃣ Train the model (online learning)
const fitResult = model.fitOnline({ coordinates });
console.log(`📊 Average Loss: ${fitResult.averageLoss}`);

// 4️⃣ Predict future values
const prediction = model.predict(5); // Predict 5 steps ahead

console.log("🔮 Predictions:", prediction.predictions);
console.log("📉 Lower Bounds:", prediction.lowerBounds);
console.log("📈 Upper Bounds:", prediction.upperBounds);
console.log("🎯 Confidence:", prediction.confidence);
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ESNRegression Architecture                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Input x(t)    │
                              │  [n_features]   │
                              └────────┬────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │      📊 Welford Normalizer           │
                    │  • Online mean/std computation       │
                    │  • Warmup period handling            │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
     ┌────────────────────────────────────────────────────────────────────┐
     │                        🌀 ESN Reservoir                            │
     │  ┌─────────────────────────────────────────────────────────────┐   │
     │  │                                                             │   │
     │  │    ┌─────────┐        ┌─────────────────────────┐           │   │
     │  │    │  W_in   │──────▶│                         │           │   │ 
     │  │    │(input)  │        │    Reservoir State      │           │   │
     │  │    └─────────┘        │      h(t) ∈ ℝ^N         │           │   │
     │  │                       │                         │           │   │
     │  │    ┌─────────┐        │   h(t) = (1-α)h(t-1)    │           │   │
     │  │    │    W    │──────▶│   + α·f(W_in·x + W·h    │           │   │
     │  │    │(recur.) │        │        + bias)          │           │   │
     │  │    └─────────┘        │                         │           │   │
     │  │                       └───────────┬─────────────┘           │   │
     │  │    ┌─────────┐                    │                         │   │
     │  │    │  bias   │────────────────────┘                         │   │
     │  │    └─────────┘                                              │   │
     │  │                                                             │   │
     │  └─────────────────────────────────────────────────────────────┘   │
     │      Spectral Radius: ρ(W) < 1   │   Sparsity Control              │
     └─────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │      📐 Extended State z(t)          │
                    │  z = [h(t), x(t), 1]                 │
                    │  (reservoir + input + bias)          │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │                      📈 Linear Readout                              │
     │                                                                     │
     │              y(t) = W_out · z(t)                                    │
     │                                                                     │
     │   ┌─────────────────────────────────────────────────────────────┐   │
     │   │              🎯 RLS Optimizer (Online)                      │   │
     │   │  • Recursive weight updates                                 │   │
     │   │  • Forgetting factor (λ) for adaptation                     │   │
     │   │  • L2 regularization                                        │   │
     │   │  • Outlier downweighting                                    │   │
     │   └─────────────────────────────────────────────────────────────┘   │
     └─────────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   Output ŷ(t)   │
                              │  [n_features]   │
                              └─────────────────┘
```

### 📊 Data Flow During Training

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Training Data Flow                               │
└──────────────────────────────────────────────────────────────────────────┘

  coordinates[i]          coordinates[i+1]
       │                        │
       │    ┌───────────────┐   │
       └───▶│    INPUT      │   │
            │   x(t) = [i]  │   │
            └───────┬───────┘   │
                    │           │
                    ▼           │
            ┌───────────────┐   │
            │   RESERVOIR   │   │
            │   UPDATE      │   │
            └───────┬───────┘   │
                    │           │
                    ▼           │
            ┌───────────────┐   │
            │   PREDICT     │   │
            │   ŷ(t+1)      │◄──┘
            └───────┬───────┘    TARGET
                    │           y(t+1) = [i+1]
                    ▼
            ┌───────────────┐
            │   RLS UPDATE  │
            │   W_out       │
            │   minimize    │
            │   ||ŷ - y||²  │
            └───────────────┘
```

### 🔮 Prediction (Roll-Forward)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Autoregressive Prediction                             │
└──────────────────────────────────────────────────────────────────────────┘

  Latest State                                    
       │                                          
       ▼                                          
   ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
   │ x(T)  │──▶│ŷ(T+1) │──▶│ŷ(T+2) │──▶│ŷ(T+3) │ ─ ─ ─▶ ...
   └───────┘    └───────┘    └───────┘    └───────┘
                    │            │            │
                    ▼            ▼            ▼
              ┌─────────────────────────────────────┐
              │     Uncertainty grows with √step    │
              │     σ(step) = σ_residual × √step    │
              └─────────────────────────────────────┘
```

---

## ⚙️ Configuration Parameters

### Complete Configuration Interface

```typescript
interface ESNRegressionConfig {
  // 🌀 Reservoir Parameters
  reservoirSize: number; // Default: 256
  spectralRadius: number; // Default: 0.9
  leakRate: number; // Default: 0.3
  inputScale: number; // Default: 1.0
  biasScale: number; // Default: 0.1
  reservoirSparsity: number; // Default: 0.9
  inputSparsity: number; // Default: 0.0
  activation: "tanh" | "relu"; // Default: "tanh"

  // 📐 Readout Parameters
  useInputInReadout: boolean; // Default: true
  useBiasInReadout: boolean; // Default: true

  // 📈 Training Parameters
  readoutTraining: "rls"; // Default: "rls"
  rlsLambda: number; // Default: 0.999
  rlsDelta: number; // Default: 1.0
  l2Lambda: number; // Default: 0.0001

  // 🛡️ Robustness Parameters
  normalizationEpsilon: number; // Default: 1e-8
  normalizationWarmup: number; // Default: 10
  outlierThreshold: number; // Default: 3.0
  outlierMinWeight: number; // Default: 0.1
  uncertaintyMultiplier: number; // Default: 1.96

  // 🔧 Utility Parameters
  epsilon: number; // Default: 1e-8
  gradientClipNorm: number; // Default: 1.0
  weightInitScale: number; // Default: 0.1
  seed: number; // Default: 42
  verbose: boolean; // Default: false
}
```

---

### 🎯 Reservoir Parameters

<details>
<summary><b>📦 reservoirSize</b> — Size of the reservoir (number of neurons)</summary>

```typescript
reservoirSize: number; // Default: 256
```

**What it does:** Determines the dimensionality of the hidden state space.

**Impact:**

| Value            | Effect                               |
| ---------------- | ------------------------------------ |
| Small (32-128)   | Faster computation, limited capacity |
| Medium (256-512) | Good balance for most tasks          |
| Large (1024+)    | More expressive, risk of overfitting |

**Optimization Guide:**

```typescript
// 🔹 Simple patterns (seasonal, linear trends)
const simpleModel = new ESNRegression({ reservoirSize: 64 });

// 🔹 Moderate complexity (stock prices, weather)
const moderateModel = new ESNRegression({ reservoirSize: 256 });

// 🔹 Complex patterns (high-frequency data, many features)
const complexModel = new ESNRegression({ reservoirSize: 512 });

// 🔹 Rule of thumb: ~10-50x the number of input features
const nFeatures = 10;
const adaptiveModel = new ESNRegression({
  reservoirSize: Math.max(64, nFeatures * 25),
});
```

</details>

<details>
<summary><b>🌊 spectralRadius</b> — Controls memory and dynamics stability</summary>

```typescript
spectralRadius: number; // Default: 0.9, Range: (0, 1]
```

**What it does:** Scales the reservoir weight matrix to control how information
echoes through the network.

```
spectralRadius → 0: Shorter memory, faster forgetting
spectralRadius → 1: Longer memory, edge of chaos
spectralRadius > 1: Unstable (avoid!)
```

**Visual Guide:**

```
Memory Capacity vs Spectral Radius:

  Memory │
         │                    ●
         │                 ●
         │              ●
         │          ●
         │      ●
         │  ●
         └───────────────────────▶
            0.5  0.7  0.9  0.99   ρ
```

**Optimization Guide:**

```typescript
// 🔹 Short-term patterns (high-frequency trading, sensor data)
const shortMemory = new ESNRegression({ spectralRadius: 0.5 });

// 🔹 Medium-term patterns (daily patterns, typical time series)
const mediumMemory = new ESNRegression({ spectralRadius: 0.9 });

// 🔹 Long-term dependencies (monthly cycles, slow dynamics)
const longMemory = new ESNRegression({ spectralRadius: 0.99 });

// 🔹 Chaotic systems (need edge of chaos dynamics)
const chaoticSystem = new ESNRegression({
  spectralRadius: 0.95,
  leakRate: 0.1, // Combine with low leak rate
});
```

</details>

<details>
<summary><b>💧 leakRate</b> — Neuron integration speed</summary>

```typescript
leakRate: number; // Default: 0.3, Range: (0, 1]
```

**What it does:** Controls how fast neurons update their state (leaky
integrator).

**Formula:**

```
h(t) = (1 - leakRate) × h(t-1) + leakRate × f(input)
```

**Effect Diagram:**

```
leakRate = 0.1 (Slow):    leakRate = 0.9 (Fast):
     │                         │
   h │ ┌────────────           │     ┌─
     │/                        │    /│
     └─────────────▶ t        └───/──▶ t
     Smooth, slow response     Quick, responsive
```

**Optimization Guide:**

```typescript
// 🔹 Smooth, slowly changing data
const smoothData = new ESNRegression({ leakRate: 0.1 });

// 🔹 Balanced (default for most cases)
const balanced = new ESNRegression({ leakRate: 0.3 });

// 🔹 Rapidly changing data, needs quick response
const rapidData = new ESNRegression({ leakRate: 0.8 });

// 🔹 Match to your data's sampling rate
// If sampling 100Hz data that changes at ~10Hz:
const matchedRate = new ESNRegression({ leakRate: 0.1 }); // ~10% update
```

</details>

<details>
<summary><b>📏 inputScale</b> — Input weight magnitude</summary>

```typescript
inputScale: number; // Default: 1.0
```

**What it does:** Scales the input-to-reservoir weight matrix.

**Optimization Guide:**

```typescript
// 🔹 Normalized input (z-score normalized, range ~[-3, 3])
const normalizedInput = new ESNRegression({ inputScale: 1.0 });

// 🔹 Small input values (range ~[0, 0.1])
const smallInput = new ESNRegression({ inputScale: 5.0 });

// 🔹 Large input values (range ~[0, 1000])
// Note: Welford normalizer handles this automatically
const largeInput = new ESNRegression({ inputScale: 0.1 });

// 🔹 Nonlinear activation saturation control
// Higher inputScale → more nonlinear response
const nonlinear = new ESNRegression({
  inputScale: 2.0,
  activation: "tanh", // Will saturate more with larger inputs
});
```

</details>

<details>
<summary><b>⚡ biasScale</b> — Reservoir bias magnitude</summary>

```typescript
biasScale: number; // Default: 0.1
```

**What it does:** Scales the constant bias added to reservoir neurons.

```typescript
// 🔹 Default (subtle bias)
const defaultBias = new ESNRegression({ biasScale: 0.1 });

// 🔹 More diverse neuron responses
const diverseBias = new ESNRegression({ biasScale: 0.5 });

// 🔹 Minimal bias (rely on input/recurrent)
const minimalBias = new ESNRegression({ biasScale: 0.01 });
```

</details>

<details>
<summary><b>🕸️ reservoirSparsity</b> — Reservoir connection sparsity</summary>

```typescript
reservoirSparsity: number; // Default: 0.9, Range: [0, 1)
```

**What it does:** Fraction of zero connections in reservoir matrix.

```
sparsity = 0.9  →  90% zeros, 10% connections
sparsity = 0.0  →  0% zeros, fully connected
```

**Benefits of Sparsity:**

- ⚡ Faster computation (sparse matrix operations)
- 🎯 Better generalization
- 🧠 Encourages modularity

```typescript
// 🔹 Default sparse (efficient)
const sparse = new ESNRegression({ reservoirSparsity: 0.9 });

// 🔹 Dense reservoir (more expressive, slower)
const dense = new ESNRegression({ reservoirSparsity: 0.5 });

// 🔹 Very sparse (fast, may lose capacity)
const verySparse = new ESNRegression({ reservoirSparsity: 0.99 });
```

</details>

<details>
<summary><b>🔌 inputSparsity</b> — Input connection sparsity</summary>

```typescript
inputSparsity: number; // Default: 0.0 (fully connected)
```

**What it does:** Fraction of zero connections in input-to-reservoir matrix.

```typescript
// 🔹 Full input connectivity (default)
const fullInput = new ESNRegression({ inputSparsity: 0.0 });

// 🔹 Each neuron sees subset of inputs
const sparseInput = new ESNRegression({ inputSparsity: 0.5 });

// 🔹 Useful when features are independent
const independentFeatures = new ESNRegression({ inputSparsity: 0.7 });
```

</details>

<details>
<summary><b>⚡ activation</b> — Nonlinear activation function</summary>

```typescript
activation: "tanh" | "relu"; // Default: "tanh"
```

**Comparison:**

| Activation | Characteristics          | Best For                   |
| ---------- | ------------------------ | -------------------------- |
| `tanh`     | Bounded [-1, 1], smooth  | General purpose, stability |
| `relu`     | Unbounded [0, ∞), sparse | Positive-only patterns     |

```typescript
// 🔹 General purpose (recommended)
const tanhModel = new ESNRegression({ activation: "tanh" });

// 🔹 Sparse activations, positive patterns
const reluModel = new ESNRegression({ activation: "relu" });
```

</details>

---

### 📊 Training Parameters

<details>
<summary><b>📈 rlsLambda</b> — RLS forgetting factor</summary>

```typescript
rlsLambda: number; // Default: 0.999, Range: (0, 1]
```

**What it does:** Controls how quickly past observations are "forgotten".

```
Effective window ≈ 1 / (1 - rlsLambda)

λ = 0.999  →  ~1000 samples effective window
λ = 0.99   →  ~100 samples effective window  
λ = 0.9    →  ~10 samples effective window
```

**Adaptation Speed Diagram:**

```
λ = 0.999 (Slow adaptation):    λ = 0.9 (Fast adaptation):
Weight                           Weight
   │      ┌───────────             │     ┌─────────
   │     /                         │    / ↙ Quick
   │    / ↙ Gradual                │   /   response
   └───/───────────▶ t            └──/────────────▶ t
   Stable, slow learning          Tracks changes fast
```

**Optimization Guide:**

```typescript
// 🔹 Stationary data (stable patterns)
const stationary = new ESNRegression({ rlsLambda: 0.9999 });

// 🔹 Default (slight adaptivity)
const balanced = new ESNRegression({ rlsLambda: 0.999 });

// 🔹 Non-stationary data (drifting patterns)
const drifting = new ESNRegression({ rlsLambda: 0.99 });

// 🔹 Highly dynamic (concept drift, regime changes)
const dynamic = new ESNRegression({ rlsLambda: 0.95 });

// 🔹 Match to expected change rate
// If patterns change every ~500 samples:
const matched = new ESNRegression({ rlsLambda: 1 - 1 / 500 }); // 0.998
```

</details>

<details>
<summary><b>🎚️ rlsDelta</b> — RLS initialization parameter</summary>

```typescript
rlsDelta: number; // Default: 1.0
```

**What it does:** Initial value for the inverse covariance matrix diagonal (P =
I/δ).

```typescript
// 🔹 Default (balanced initial uncertainty)
const defaultDelta = new ESNRegression({ rlsDelta: 1.0 });

// 🔹 High initial uncertainty (conservative start)
const conservative = new ESNRegression({ rlsDelta: 0.1 });

// 🔹 Low initial uncertainty (confident start)
const confident = new ESNRegression({ rlsDelta: 10.0 });
```

</details>

<details>
<summary><b>🔒 l2Lambda</b> — L2 regularization strength</summary>

```typescript
l2Lambda: number; // Default: 0.0001
```

**What it does:** Weight decay to prevent overfitting.

```
Loss = MSE + l2Lambda × ||W_out||²
```

**Optimization Guide:**

```typescript
// 🔹 Minimal regularization (large data, simple patterns)
const minimal = new ESNRegression({ l2Lambda: 0.00001 });

// 🔹 Default (balanced)
const balanced = new ESNRegression({ l2Lambda: 0.0001 });

// 🔹 Strong regularization (small data, complex reservoir)
const strong = new ESNRegression({ l2Lambda: 0.01 });

// 🔹 Aggressive (prevent overfitting at all costs)
const aggressive = new ESNRegression({ l2Lambda: 0.1 });
```

</details>

<details>
<summary><b>🔗 useInputInReadout / useBiasInReadout</b> — Extended state configuration</summary>

```typescript
useInputInReadout: boolean; // Default: true
useBiasInReadout: boolean; // Default: true
```

**Extended State Structure:**

```
z = [ reservoir_state , input , 1 ]
         h(t)           x(t)   bias
    └──────┬──────┘  └───┬───┘  └┬┘
    reservoirSize    nFeatures   1

if useInputInReadout=false: z = [h(t), 1]
if useBiasInReadout=false:  z = [h(t), x(t)]
if both=false:              z = [h(t)]
```

```typescript
// 🔹 Full extended state (recommended)
const full = new ESNRegression({
  useInputInReadout: true,
  useBiasInReadout: true,
});

// 🔹 Reservoir-only (pure ESN)
const pureESN = new ESNRegression({
  useInputInReadout: false,
  useBiasInReadout: false,
});
```

</details>

---

### 🛡️ Robustness Parameters

<details>
<summary><b>📊 normalizationWarmup</b> — Samples before normalization activates</summary>

```typescript
normalizationWarmup: number; // Default: 10
```

**What it does:** Minimum samples needed to estimate reliable statistics.

```typescript
// 🔹 Quick start (small batches)
const quickStart = new ESNRegression({ normalizationWarmup: 5 });

// 🔹 Default
const balanced = new ESNRegression({ normalizationWarmup: 10 });

// 🔹 Conservative (noisy initial data)
const conservative = new ESNRegression({ normalizationWarmup: 50 });
```

</details>

<details>
<summary><b>🚨 outlierThreshold</b> — Z-score threshold for outlier detection</summary>

```typescript
outlierThreshold: number; // Default: 3.0
```

**What it does:** Samples with residual z-score > threshold get downweighted.

```
P(|z| > 3) ≈ 0.3%   (3-sigma rule)
P(|z| > 2) ≈ 4.5%   
P(|z| > 4) ≈ 0.006%
```

```typescript
// 🔹 Aggressive outlier rejection
const aggressive = new ESNRegression({ outlierThreshold: 2.0 });

// 🔹 Default (3-sigma rule)
const balanced = new ESNRegression({ outlierThreshold: 3.0 });

// 🔹 Permissive (only extreme outliers)
const permissive = new ESNRegression({ outlierThreshold: 5.0 });

// 🔹 Heavy-tailed data (expect more outliers)
const heavyTailed = new ESNRegression({
  outlierThreshold: 4.0,
  outlierMinWeight: 0.3, // Still consider them somewhat
});
```

</details>

<details>
<summary><b>⚖️ outlierMinWeight</b> — Minimum weight for outliers</summary>

```typescript
outlierMinWeight: number; // Default: 0.1, Range: [0, 1]
```

**What it does:** Floor for outlier sample weights (prevents complete
rejection).

```typescript
// 🔹 Aggressive rejection (near-zero weight for outliers)
const aggressive = new ESNRegression({ outlierMinWeight: 0.01 });

// 🔹 Default
const balanced = new ESNRegression({ outlierMinWeight: 0.1 });

// 🔹 Soft rejection (outliers still contribute)
const soft = new ESNRegression({ outlierMinWeight: 0.5 });
```

</details>

<details>
<summary><b>📏 uncertaintyMultiplier</b> — Confidence interval width</summary>

```typescript
uncertaintyMultiplier: number; // Default: 1.96
```

**What it does:** Multiplier for prediction interval bounds.

```
Bounds = prediction ± uncertaintyMultiplier × σ

1.96 → 95% confidence interval
1.645 → 90% confidence interval
2.576 → 99% confidence interval
```

```typescript
// 🔹 90% confidence interval
const ci90 = new ESNRegression({ uncertaintyMultiplier: 1.645 });

// 🔹 95% confidence interval (default)
const ci95 = new ESNRegression({ uncertaintyMultiplier: 1.96 });

// 🔹 99% confidence interval (conservative)
const ci99 = new ESNRegression({ uncertaintyMultiplier: 2.576 });
```

</details>

---

### 🔧 Utility Parameters

<details>
<summary><b>🌱 seed</b> — Random seed for reproducibility</summary>

```typescript
seed: number; // Default: 42
```

**What it does:** Initializes the random number generator for reservoir weights.

```typescript
// 🔹 Reproducible experiments
const reproducible = new ESNRegression({ seed: 12345 });

// 🔹 Different random initialization
const model1 = new ESNRegression({ seed: 1 });
const model2 = new ESNRegression({ seed: 2 });
const model3 = new ESNRegression({ seed: 3 });
// Can ensemble these for better predictions
```

</details>

<details>
<summary><b>🔢 epsilon / normalizationEpsilon</b> — Numerical stability constants</summary>

```typescript
epsilon: number; // Default: 1e-8
normalizationEpsilon: number; // Default: 1e-8
```

**What it does:** Prevents division by zero in numerical operations.

```typescript
// 🔹 Default (works for most cases)
const standard = new ESNRegression({ epsilon: 1e-8 });

// 🔹 Higher precision (if seeing numerical issues)
const highPrecision = new ESNRegression({ epsilon: 1e-10 });
```

</details>

<details>
<summary><b>📐 weightInitScale</b> — Output weight initialization scale</summary>

```typescript
weightInitScale: number; // Default: 0.1
```

**What it does:** Standard deviation for initializing readout weights.

```typescript
// 🔹 Conservative start (near-zero predictions initially)
const conservative = new ESNRegression({ weightInitScale: 0.01 });

// 🔹 Default
const balanced = new ESNRegression({ weightInitScale: 0.1 });

// 🔹 Aggressive initialization
const aggressive = new ESNRegression({ weightInitScale: 1.0 });
```

</details>

---

## 📖 API Reference

### Constructor

```typescript
constructor(config?: Partial<ESNRegressionConfig>)
```

Creates a new ESNRegression model with the specified configuration.

---

### Methods

#### `fitOnline(params: { coordinates: number[][] }): FitResult`

Train the model on a sequence of coordinate vectors.

```typescript
interface FitResult {
  samplesProcessed: number; // Number of training pairs processed
  averageLoss: number; // Mean squared error during training
  gradientNorm: number; // Magnitude of parameter updates
  driftDetected: boolean; // Concept drift detection flag
  sampleWeight: number; // Last sample's outlier weight
}
```

**Example:**

```typescript
const data = [
  [1, 2, 3],
  [2, 3, 4],
  [3, 4, 5],
  [4, 5, 6],
];

const result = model.fitOnline({ coordinates: data });
console.log(`Processed ${result.samplesProcessed} samples`);
console.log(`Average loss: ${result.averageLoss.toFixed(6)}`);
```

---

#### `predict(futureSteps: number): PredictionResult`

Generate multi-step ahead predictions with uncertainty bounds.

```typescript
interface PredictionResult {
  predictions: number[][]; // [step][feature] predicted values
  lowerBounds: number[][]; // Lower confidence bounds
  upperBounds: number[][]; // Upper confidence bounds
  confidence: number; // Overall model confidence [0, 1]
}
```

**Example:**

```typescript
const pred = model.predict(10);

for (let step = 0; step < 10; step++) {
  console.log(`Step ${step + 1}:`);
  console.log(`  Prediction: ${pred.predictions[step]}`);
  console.log(
    `  Range: [${pred.lowerBounds[step]}, ${pred.upperBounds[step]}]`,
  );
}
console.log(`Model confidence: ${(pred.confidence * 100).toFixed(1)}%`);
```

---

#### `getModelSummary(): ModelSummary`

Get summary statistics about the model.

```typescript
interface ModelSummary {
  totalParameters: number; // Total learnable parameters
  receptiveField: number; // Effective memory length
  spectralRadius: number; // Current spectral radius
  reservoirSize: number; // Reservoir dimension
  nFeatures: number; // Input/output dimension
  nTargets: number; // Target dimension (same as nFeatures)
  sampleCount: number; // Total samples processed
}
```

---

#### `getWeights(): WeightInfo`

Retrieve all weight matrices for inspection.

```typescript
interface WeightInfo {
  weights: Array<{
    name: string; // "Wout", "Win", "W", "bias"
    shape: number[]; // Matrix dimensions
    values: number[]; // Flattened values
  }>;
}
```

---

#### `getNormalizationStats(): NormalizationStats`

Get current normalization statistics.

```typescript
interface NormalizationStats {
  means: number[]; // Per-feature means
  stds: number[]; // Per-feature standard deviations
  count: number; // Samples used for estimation
  isActive: boolean; // Whether normalization is active
}
```

---

#### `reset(): void`

Reset the model to initial state.

---

#### `save(): string`

Serialize the model to a JSON string.

---

#### `load(str: string): void`

Load model state from a JSON string.

---

## 🎓 Use Case Examples

### 📈 Stock Price Prediction

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

// Configuration optimized for financial time series
const stockModel = new ESNRegression({
  reservoirSize: 256,
  spectralRadius: 0.95, // Good memory for trends
  leakRate: 0.2, // Smooth integration
  rlsLambda: 0.995, // Adapt to market changes
  outlierThreshold: 2.5, // Financial data has outliers
  outlierMinWeight: 0.2, // Don't completely ignore them
  l2Lambda: 0.001, // Regularization for stability
});

// Data: [open, high, low, close, volume]
const stockData = [
  [150.0, 152.0, 149.0, 151.5, 1000000],
  [151.5, 153.0, 150.0, 152.0, 1100000],
  [152.0, 154.0, 151.0, 153.5, 950000],
  // ... more data
];

// Train
const result = stockModel.fitOnline({ coordinates: stockData });

// Predict next 5 trading days
const forecast = stockModel.predict(5);

console.log("📈 5-Day Stock Forecast:");
forecast.predictions.forEach((pred, i) => {
  console.log(
    `Day ${i + 1}: Close = $${pred[3].toFixed(2)} ` +
      `[${forecast.lowerBounds[i][3].toFixed(2)} - ` +
      `${forecast.upperBounds[i][3].toFixed(2)}]`,
  );
});
```

---

### 🌡️ Weather Forecasting

```typescript
// Configuration for weather (has daily/seasonal patterns)
const weatherModel = new ESNRegression({
  reservoirSize: 512, // More capacity for complex patterns
  spectralRadius: 0.99, // Long memory for seasonal patterns
  leakRate: 0.1, // Slow dynamics
  rlsLambda: 0.9999, // Weather patterns are stable
  normalizationWarmup: 30, // Need good stats for weather
  seed: 42,
});

// Data: [temperature, humidity, pressure, wind_speed]
const weatherData: number[][] = [
  // ... hourly readings
];

weatherModel.fitOnline({ coordinates: weatherData });

// Predict next 24 hours
const forecast = weatherModel.predict(24);

console.log("🌡️ 24-Hour Weather Forecast:");
forecast.predictions.forEach((pred, hour) => {
  console.log(
    `Hour ${hour + 1}: ` +
      `Temp=${pred[0].toFixed(1)}°C, ` +
      `Humidity=${pred[1].toFixed(0)}%`,
  );
});
```

---

### 🤖 Sensor Data / IoT

```typescript
// Configuration for high-frequency sensor data
const sensorModel = new ESNRegression({
  reservoirSize: 128, // Smaller for speed
  spectralRadius: 0.7, // Shorter memory for sensors
  leakRate: 0.5, // Quick response
  rlsLambda: 0.99, // Adapt to sensor drift
  outlierThreshold: 3.5, // Sensors can be noisy
  activation: "relu", // Good for positive-only readings
});

// Real-time training loop
async function processSensorStream() {
  let buffer: number[][] = [];

  for await (const reading of sensorStream) {
    buffer.push(reading);

    if (buffer.length >= 100) {
      // Train on batch
      sensorModel.fitOnline({ coordinates: buffer });

      // Get next prediction for anomaly detection
      const pred = sensorModel.predict(1);

      // Check if current reading is within bounds
      const isAnomaly = reading.some((val, i) =>
        val < pred.lowerBounds[0][i] || val > pred.upperBounds[0][i]
      );

      if (isAnomaly) {
        console.log("⚠️ Anomaly detected!", reading);
      }

      buffer = buffer.slice(-50); // Keep recent context
    }
  }
}
```

---

### 🎮 Motion Prediction

```typescript
// Configuration for smooth trajectory prediction
const motionModel = new ESNRegression({
  reservoirSize: 192,
  spectralRadius: 0.85,
  leakRate: 0.4,
  rlsLambda: 0.99,
  useInputInReadout: true, // Direct path helps smooth predictions
  useBiasInReadout: true,
  uncertaintyMultiplier: 1.96,
});

// Data: [x, y, z, vx, vy, vz] (position + velocity)
const trajectoryData: number[][] = [
  // ... motion capture data
];

motionModel.fitOnline({ coordinates: trajectoryData });

// Predict next 30 frames (1 second at 30fps)
const trajectory = motionModel.predict(30);

// Smooth predictions for animation
trajectory.predictions.forEach((pred, frame) => {
  const [x, y, z, vx, vy, vz] = pred;
  renderFrame({ x, y, z, vx, vy, vz });
});
```

---

### 📊 Multi-variate Economic Indicators

```typescript
// Configuration for economic time series (monthly data)
const econModel = new ESNRegression({
  reservoirSize: 384,
  spectralRadius: 0.98, // Economic cycles are long
  leakRate: 0.15, // Slow-moving indicators
  rlsLambda: 0.9995, // Very stable patterns
  l2Lambda: 0.0005, // Moderate regularization
  normalizationWarmup: 24, // Need 2 years for good stats
});

// Data: [GDP_growth, unemployment, inflation, interest_rate]
const economicData: number[][] = [
  // ... monthly readings
];

econModel.fitOnline({ coordinates: economicData });

// Forecast next 12 months
const forecast = econModel.predict(12);
const confidence = forecast.confidence;

console.log(
  `📊 Economic Forecast (Confidence: ${(confidence * 100).toFixed(1)}%):`,
);
const months = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

forecast.predictions.forEach((pred, i) => {
  console.log(
    `${months[i]}: GDP=${pred[0].toFixed(2)}%, ` +
      `Unemployment=${pred[1].toFixed(1)}%, ` +
      `Inflation=${pred[2].toFixed(2)}%`,
  );
});
```

---

## 🔬 Parameter Optimization Guide

### Decision Flowchart

```
                 ┌─────────────────────────────────┐
                 │  What type of data do you have? │
                 └────────────────┬────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│High-frequency │       │   Standard    │       │ Low-frequency │
│  (>1Hz)       │       │ (hourly/daily)│       │(weekly/monthly)
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
 reservoirSize: 128      reservoirSize: 256      reservoirSize: 384
 spectralRadius: 0.7     spectralRadius: 0.9     spectralRadius: 0.98
 leakRate: 0.5-0.8       leakRate: 0.3           leakRate: 0.1-0.2
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────────────────────────────────────────────────────┐
│                   Is the data stationary?                     │
└───────────────────────────────┬───────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
        ┌───────────┐                       ┌───────────┐
        │    YES    │                       │    NO     │
        │ Stationary│                       │  Drifting │
        └─────┬─────┘                       └─────┬─────┘
              │                                   │
              ▼                                   ▼
       rlsLambda: 0.999-0.9999             rlsLambda: 0.95-0.99
       l2Lambda: 0.0001                    l2Lambda: 0.001
              │                                   │
              └───────────────┬───────────────────┘
                              │
                              ▼
                ┌───────────────────────────────────┐
                │        How noisy is the data?     │
                └───────────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
        ┌───────────┐         ┌───────────┐         ┌───────────┐
        │   Clean   │         │  Moderate │         │   Noisy   │
        └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
              │                     │                     │
              ▼                     ▼                     ▼
 outlierThreshold: 4.0    outlierThreshold: 3.0   outlierThreshold: 2.5
 outlierMinWeight: 0.01   outlierMinWeight: 0.1   outlierMinWeight: 0.3
```

### Quick Reference Table

| Scenario        | reservoirSize | spectralRadius | leakRate | rlsLambda    | outlierThreshold |
| --------------- | ------------- | -------------- | -------- | ------------ | ---------------- |
| **HFT/Sensors** | 64-128        | 0.5-0.7        | 0.5-0.8  | 0.95-0.99    | 3.0-4.0          |
| **Daily Stock** | 256-384       | 0.9-0.95       | 0.2-0.3  | 0.995-0.999  | 2.5-3.0          |
| **Weather**     | 384-512       | 0.95-0.99      | 0.1-0.2  | 0.999-0.9999 | 3.0-3.5          |
| **Economic**    | 256-384       | 0.95-0.98      | 0.1-0.15 | 0.9995+      | 3.0-4.0          |
| **Motion**      | 128-256       | 0.8-0.9        | 0.3-0.5  | 0.99-0.995   | 3.0-4.0          |

---

## 💾 Serialization

### Save and Load Models

```typescript
// Train and save
const model = new ESNRegression({ reservoirSize: 256 });
model.fitOnline({ coordinates: trainingData });

const modelJson = model.save();
// Store to file, database, etc.
Deno.writeTextFileSync("model.json", modelJson);

// Later: load and use
const loadedModel = new ESNRegression();
loadedModel.load(Deno.readTextFileSync("model.json"));

const prediction = loadedModel.predict(5);
```

### Incremental Training

```typescript
// Day 1: Initial training
const model = new ESNRegression();
model.fitOnline({ coordinates: day1Data });
const checkpoint1 = model.save();

// Day 2: Continue training
model.fitOnline({ coordinates: day2Data });
const checkpoint2 = model.save();

// Day 3: Continue training
model.fitOnline({ coordinates: day3Data });

// Rollback to Day 2 if needed
model.load(checkpoint2);
```

---

## 🔧 Troubleshooting

| Problem                    | Possible Cause        | Solution                                               |
| -------------------------- | --------------------- | ------------------------------------------------------ |
| High loss doesn't decrease | Learning rate issues  | Decrease `rlsDelta`, increase `reservoirSize`          |
| Predictions are constant   | Dead reservoir        | Increase `inputScale`, check `spectralRadius < 1`      |
| Predictions explode        | Numerical instability | Decrease `spectralRadius`, increase `l2Lambda`         |
| Slow training              | Large reservoir       | Decrease `reservoirSize`, increase `reservoirSparsity` |
| Poor long-term predictions | Short memory          | Increase `spectralRadius`, decrease `leakRate`         |
| Can't track fast changes   | Slow adaptation       | Decrease `rlsLambda`, increase `leakRate`              |

---

## 📄 License

MIT License © 2025 [Henrique Emanoel Viana](https://github.com/hviana)

---

<div align="center">

**[⬆ Back to Top](#-esnregression)**

Made with ❤️ for time series prediction

</div>
