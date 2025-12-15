# 🧠 ESNRegression - Echo State Network for Multivariate Regression

<div align="center">

**A high-performance Echo State Network library for real-time multivariate time
series prediction with incremental online learning**

[📦 JSR Package](https://jsr.io/@hviana/multivariate-regression) •
[💻 GitHub](https://github.com/hviana/multivariate-regression) •
[📖 Documentation](#-api-reference)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📚 Core Concepts](#-core-concepts)
  - [Echo State Networks](#-echo-state-networks-esn)
  - [Reservoir Computing](#-reservoir-computing)
  - [RLS Online Learning](#-rls-online-learning)
  - [Welford Normalization](#-welford-normalization)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
  - [Reservoir Parameters](#reservoir-parameters)
  - [Training Parameters](#training-parameters)
  - [Normalization & Robustness](#normalization--robustness)
  - [Prediction Parameters](#prediction-parameters)
- [🎯 Parameter Optimization Guide](#-parameter-optimization-guide)
- [📖 API Reference](#-api-reference)
- [💡 Examples](#-examples)
- [🏗️ Architecture](#️-architecture)
- [⚡ Performance Tips](#-performance-tips)
- [🔧 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Capabilities

- **🔄 Online Learning** - Incremental RLS training, no batching required
- **📈 Multi-step Prediction** - Direct or recursive multi-horizon forecasting
- **🎛️ Multivariate** - Handle multiple input features and output targets
- **⚡ High Performance** - Zero-allocation hot paths with arena allocators

</td>
<td width="50%">

### 🛡️ Robustness Features

- **📊 Auto-normalization** - Welford's algorithm for streaming statistics
- **🎚️ Outlier Handling** - Automatic sample downweighting
- **📉 Uncertainty Quantification** - Confidence bounds on predictions
- **💾 Serialization** - Save/load model state

</td>
</tr>
</table>

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ESNRegression Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input Data ──┬──► Welford Normalizer ──► ESN Reservoir ──┐               │
│                │                              (Fixed)       │               │
│                │                                            ▼               │
│                └──► Ring Buffer             Extended State: [r; x; 1]       │
│                     (History)                               │               │
│                                                             ▼               │
│   Target ──────► RLS Optimizer ◄───────── Linear Readout ──┘               │
│                       │                    (Trainable)                      │
│                       ▼                                                     │
│               Updated Weights ──────────► Predictions                       │
│                                           + Confidence                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";
```

### Basic Usage

```typescript
// 1️⃣ Create model
const model = new ESNRegression({
  maxFutureSteps: 5, // Predict up to 5 steps ahead
  reservoirSize: 256, // 256 reservoir neurons
  spectralRadius: 0.9, // Edge of chaos dynamics
});

// 2️⃣ Train online (streaming data)
for (const sample of dataStream) {
  const result = model.fitOnline({
    xCoordinates: [sample.features], // [[f1, f2, f3, ...]]
    yCoordinates: [sample.targets], // [[t1, t2, ...]]
  });

  console.log(`📊 Loss: ${result.averageLoss.toFixed(4)}`);
}

// 3️⃣ Predict future steps
const prediction = model.predict(3); // 3 steps ahead
console.log("🔮 Predictions:", prediction.predictions);
console.log("📈 Confidence:", prediction.confidence);
console.log("📉 Lower bounds:", prediction.lowerBounds);
console.log("📈 Upper bounds:", prediction.upperBounds);
```

---

## 📚 Core Concepts

### 🌊 Echo State Networks (ESN)

Echo State Networks are a type of recurrent neural network where:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          ESN Architecture                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│    ┌─────────┐      ┌─────────────────────────────────────┐               │
│    │  Input  │──Win─►│         RESERVOIR (Fixed)           │               │
│    │   x_t   │       │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     │               │
│    └─────────┘       │  │ ○ │─│ ○ │─│ ○ │─│ ○ │─│ ○ │     │               │
│                      │  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘     │               │
│                      │    │     │     │     │     │       │               │
│                      │  ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐ ┌─▼─┐     │               │
│                      │  │ ○ │─│ ○ │─│ ○ │─│ ○ │─│ ○ │     │               │
│                      │  └───┘ └───┘ └───┘ └───┘ └───┘     │               │
│                      │         Recurrent connections (W)   │               │
│                      └─────────────────┬───────────────────┘               │
│                                        │ r_t (reservoir state)             │
│                                        ▼                                   │
│                              ┌─────────────────┐                           │
│                              │  LINEAR READOUT │                           │
│                              │   (Trainable)   │─────────► Output y_t      │
│                              │    Wout × z     │                           │
│                              └─────────────────┘                           │
│                                                                            │
│    Key Insight: Only Wout is trained! Reservoir provides rich dynamics.   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**The "Echo State Property"**: Past inputs create "echoes" that gradually fade
in the reservoir state, allowing the network to process temporal sequences.

#### State Update Equation

$$r_t = (1 - \alpha) \cdot r_{t-1} + \alpha \cdot \text{tanh}(W_{in} \cdot x_t + W \cdot r_{t-1} + b)$$

Where:

- $r_t$ = reservoir state at time $t$
- $\alpha$ = leak rate (controls memory/responsiveness trade-off)
- $W_{in}$ = input weight matrix (fixed)
- $W$ = reservoir weight matrix (fixed, scaled to spectral radius)
- $b$ = bias vector (fixed)

---

### 💡 Reservoir Computing

The key insight of reservoir computing is **separation of concerns**:

| Component                       | Trained?   | Purpose                                             |
| ------------------------------- | ---------- | --------------------------------------------------- |
| **Input Weights ($W_{in}$)**    | ❌ Fixed   | Project input into high-dimensional reservoir space |
| **Reservoir Weights ($W$)**     | ❌ Fixed   | Create rich, nonlinear dynamics with memory         |
| **Readout Weights ($W_{out}$)** | ✅ Trained | Learn task-specific mapping from reservoir state    |

```
Input Space (low-dim) ──► Reservoir Space (high-dim) ──► Output Space
     n features              m neurons >> n              k targets
                                  │
                     Nonlinear transformation
                     with temporal memory
```

**Benefits:**

- 🚀 Fast training (only linear readout)
- 📈 Universal approximation capability
- ⏱️ Natural handling of temporal dependencies
- 💾 Online/incremental learning friendly

---

### 📐 RLS Online Learning

**Recursive Least Squares (RLS)** enables efficient online weight updates
without storing historical data.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RLS Update Algorithm                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  For each new sample (z_t, y_t):                                       │
│                                                                         │
│  1. Compute Kalman gain:                                                │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  k_t = P_{t-1} · z_t / (λ + z_t^T · P_{t-1} · z_t)          │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  2. Update weights:                                                     │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  W_t = W_{t-1} + k_t · (y_t - W_{t-1} · z_t)^T              │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  3. Update inverse correlation matrix:                                  │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  P_t = (P_{t-1} - k_t · z_t^T · P_{t-1}) / λ                │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Where:                                                                 │
│    λ = forgetting factor (0.99-0.9999)                                 │
│    P = inverse correlation matrix                                       │
│    k = Kalman gain vector                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Parameters:**

- **λ (rlsLambda)**: Forgetting factor - lower values adapt faster but may be
  noisier
- **δ (rlsDelta)**: Initial P scaling - affects early learning stability

---

### 📊 Welford Normalization

The library uses **Welford's online algorithm** for numerically stable streaming
statistics:

```typescript
// Welford's Algorithm (internal implementation)
for each new sample x:
    n += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

variance = M2 / (n - 1)
std = sqrt(variance)
```

**Advantages:**

- ✅ Single-pass computation
- ✅ Numerically stable (no catastrophic cancellation)
- ✅ Memory efficient (only stores running stats)
- ✅ Warmup period before activation

---

## ⚙️ Configuration Parameters

### 📊 Complete Parameter Reference

```typescript
interface ESNRegressionConfig {
  // Reservoir Architecture
  maxSequenceLength?: number; // Default: 64
  maxFutureSteps?: number; // Default: 1
  reservoirSize?: number; // Default: 256
  spectralRadius?: number; // Default: 0.9
  leakRate?: number; // Default: 0.3
  inputScale?: number; // Default: 1.0
  biasScale?: number; // Default: 0.1
  reservoirSparsity?: number; // Default: 0.9
  inputSparsity?: number; // Default: 0.0
  activation?: "tanh" | "relu"; // Default: "tanh"

  // Readout Configuration
  useInputInReadout?: boolean; // Default: true
  useBiasInReadout?: boolean; // Default: true
  useDirectMultiHorizon?: boolean; // Default: true

  // RLS Training
  readoutTraining?: "rls"; // Default: "rls"
  rlsLambda?: number; // Default: 0.999
  rlsDelta?: number; // Default: 1.0
  l2Lambda?: number; // Default: 0.0001
  gradientClipNorm?: number; // Default: 1.0

  // Normalization
  normalizationEpsilon?: number; // Default: 1e-8
  normalizationWarmup?: number; // Default: 10

  // Robustness
  outlierThreshold?: number; // Default: 3.0
  outlierMinWeight?: number; // Default: 0.1

  // Uncertainty
  residualWindowSize?: number; // Default: 100
  uncertaintyMultiplier?: number; // Default: 1.96

  // General
  epsilon?: number; // Default: 1e-8
  weightInitScale?: number; // Default: 0.1
  seed?: number; // Default: 42
  verbose?: boolean; // Default: false
}
```

---

### Reservoir Parameters

#### `reservoirSize`

**Default: `256`** | Range: `[32, 4096]`

The number of neurons in the reservoir. Larger reservoirs can capture more
complex patterns but require more computation.

```
┌────────────────────────────────────────────────────────────────┐
│              reservoirSize Impact                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Small (32-64)     │  Medium (128-512)   │  Large (512-2048)  │
│  ───────────────── │ ─────────────────── │ ────────────────── │
│  ✅ Fast           │  ✅ Balanced         │  ✅ High capacity   │
│  ✅ Low memory     │  ✅ Most use cases   │  ❌ Slow training   │
│  ❌ Limited        │                      │  ❌ High memory     │
│     capacity       │                      │  ❌ Overfitting     │
│                    │                      │     risk            │
└────────────────────────────────────────────────────────────────┘
```

**Optimization Guide:**

| Scenario                        | Recommended Size |
| ------------------------------- | ---------------- |
| Simple linear trends            | 32-64            |
| Moderate complexity             | 128-256          |
| Complex patterns, many features | 512-1024         |
| Very high-dimensional data      | 1024-2048        |

```typescript
// Simple time series
const simpleModel = new ESNRegression({ reservoirSize: 64 });

// Complex multivariate forecasting
const complexModel = new ESNRegression({ reservoirSize: 512 });
```

---

#### `spectralRadius`

**Default: `0.9`** | Range: `(0, 1.0]`

Controls the "memory" of the reservoir. The spectral radius is the largest
absolute eigenvalue of the reservoir weight matrix.

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Spectral Radius Effects                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ρ → 0.5          │    ρ → 0.9           │    ρ → 1.0               │
│   ──────────       │    ──────────        │    ──────────            │
│                    │                       │                          │
│   Short memory     │    Balanced          │    Long memory           │
│   Fast decay       │    "Edge of chaos"   │    Risk of instability   │
│   Quick response   │    Rich dynamics     │    Slow adaptation       │
│                    │                       │                          │
│   Input ─○─○─○─►   │    Input ─○─○─○─○─►  │    Input ─○─○─○─○─○─○─► │
│                    │                       │                          │
│   Use for:         │    Use for:           │    Use for:             │
│   • Fast signals   │    • Most cases       │    • Very slow dynamics │
│   • Little memory  │    • Time series      │    • Long dependencies  │
│     needed         │                       │                          │
│                    │                       │                          │
└────────────────────────────────────────────────────────────────────────┘
```

**Mathematical Insight:** $$\text{Memory} \propto \frac{1}{1 - \rho}$$

```typescript
// Fast-changing signals (e.g., high-frequency trading)
const fastModel = new ESNRegression({ spectralRadius: 0.7 });

// Standard time series forecasting
const standardModel = new ESNRegression({ spectralRadius: 0.9 });

// Very long-term dependencies (e.g., climate data)
const longMemoryModel = new ESNRegression({ spectralRadius: 0.99 });
```

---

#### `leakRate`

**Default: `0.3`** | Range: `(0, 1]`

Controls how quickly the reservoir state updates. Also known as the "leaky
integrator" parameter.

$$r_t = (1 - \alpha) \cdot r_{t-1} + \alpha \cdot f(...)$$

```
┌─────────────────────────────────────────────────────────────────┐
│                     Leak Rate Visualization                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   α = 0.1 (slow)        α = 0.5 (medium)      α = 1.0 (fast)   │
│                                                                 │
│   State response        State response        State response    │
│   to input step:        to input step:        to input step:    │
│                                                                 │
│   ─────────────────     ─────────────────     ─────────────     │
│   │     ╭──────────     │    ╭─────────       │   ╭─────────   │
│   │    ╱                │   ╱                 │   │             │
│   │   ╱                 │  ╱                  │   │             │
│   │  ╱                  │ ╱                   │   │             │
│   │ ╱                   │╱                    │  ─┘             │
│   └─────────────────    └─────────────────    └─────────────    │
│                                                                 │
│   Smooth, filtered      Balanced              Instant response  │
│   High inertia          Most use cases        No filtering      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| α Value | Effect                          | Use Case                     |
| ------- | ------------------------------- | ---------------------------- |
| 0.1-0.3 | Strong smoothing, slow response | Noisy data, stable forecasts |
| 0.3-0.7 | Balanced                        | General purpose              |
| 0.7-1.0 | Fast response, little smoothing | Fast-changing dynamics       |

```typescript
// Noisy sensor data
const smoothModel = new ESNRegression({ leakRate: 0.2 });

// Standard forecasting
const balancedModel = new ESNRegression({ leakRate: 0.3 });

// Fast signal tracking
const responsiveModel = new ESNRegression({ leakRate: 0.8 });
```

---

#### `reservoirSparsity`

**Default: `0.9`** | Range: `[0, 1)`

Fraction of reservoir weights that are zero. Sparse reservoirs are more
computationally efficient and can have better generalization.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Reservoir Sparsity Patterns                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   sparsity = 0.0          sparsity = 0.9          sparsity = 0.99      │
│   (fully connected)       (typical)               (very sparse)         │
│                                                                         │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐      │
│   │■■■■■■■■■■■■│         │■ □ □ □ ■ □ │         │□ □ □ □ ■ □ │      │
│   │■■■■■■■■■■■■│         │□ ■ □ □ □ □ │         │□ □ □ □ □ □ │      │
│   │■■■■■■■■■■■■│         │□ □ □ ■ □ □ │         │□ □ □ □ □ □ │      │
│   │■■■■■■■■■■■■│         │□ □ □ □ □ ■ │         │□ □ □ ■ □ □ │      │
│   │■■■■■■■■■■■■│         │■ □ ■ □ □ □ │         │□ □ □ □ □ □ │      │
│   │■■■■■■■■■■■■│         │□ □ □ □ ■ □ │         │□ ■ □ □ □ □ │      │
│   └─────────────┘         └─────────────┘         └─────────────┘      │
│                                                                         │
│   Dense, slow             Balanced, efficient     Very efficient        │
│   May overfit             Recommended             Risk of disconnection │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

```typescript
// Efficiency-focused (embedded systems)
const efficientModel = new ESNRegression({ reservoirSparsity: 0.95 });

// Research/accuracy-focused
const denseModel = new ESNRegression({ reservoirSparsity: 0.8 });
```

---

#### `inputScale`

**Default: `1.0`** | Range: `(0, ∞)`

Scaling factor for input weights. Controls how strongly inputs drive the
reservoir.

```typescript
// Weak input signal (when data is already normalized)
const weakInput = new ESNRegression({ inputScale: 0.5 });

// Strong input signal (when input variations are small)
const strongInput = new ESNRegression({ inputScale: 2.0 });
```

---

#### `activation`

**Default: `"tanh"`** | Options: `"tanh"` | `"relu"`

Activation function for reservoir neurons.

| Activation | Formula     | Properties                       |
| ---------- | ----------- | -------------------------------- |
| `tanh`     | $\tanh(x)$  | Bounded [-1,1], smooth, centered |
| `relu`     | $\max(0,x)$ | Unbounded, sparse activation     |

```typescript
// Standard (recommended for most cases)
const tanhModel = new ESNRegression({ activation: "tanh" });

// For positive outputs or sparse dynamics
const reluModel = new ESNRegression({ activation: "relu" });
```

---

### Training Parameters

#### `rlsLambda` (Forgetting Factor)

**Default: `0.999`** | Range: `(0, 1]`

Controls how quickly RLS "forgets" old samples. Critical for online learning.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    RLS Forgetting Factor (λ) Effects                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   λ = 0.99 (fast forgetting)                                              │
│   ─────────────────────────────                                           │
│   • Effective window: ~100 samples                                         │
│   • Quickly adapts to changes                                              │
│   • May be noisy/unstable                                                  │
│   • Use for: non-stationary data, concept drift                           │
│                                                                            │
│   Sample weights over time:                                                │
│   ████████░░░░░░░░░░░░░░░░░░░░░░░░  (recent samples dominate)            │
│                                                                            │
│   λ = 0.999 (slow forgetting) [DEFAULT]                                   │
│   ────────────────────────────────                                         │
│   • Effective window: ~1000 samples                                        │
│   • Balanced adaptation                                                    │
│   • Stable learning                                                        │
│   • Use for: most applications                                             │
│                                                                            │
│   Sample weights over time:                                                │
│   ██████████████████████░░░░░░░░░░  (smooth decay)                        │
│                                                                            │
│   λ = 0.9999 (very slow forgetting)                                       │
│   ───────────────────────────────                                          │
│   • Effective window: ~10000 samples                                       │
│   • Very stable                                                            │
│   • Slow to adapt                                                          │
│   • Use for: stationary data, long-term patterns                          │
│                                                                            │
│   Sample weights over time:                                                │
│   ████████████████████████████████  (nearly uniform)                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Effective memory formula:** $$N_{eff} \approx \frac{1}{1 - \lambda}$$

```typescript
// Fast adaptation (concept drift, non-stationary)
const adaptiveModel = new ESNRegression({ rlsLambda: 0.99 });

// Balanced (most use cases)
const balancedModel = new ESNRegression({ rlsLambda: 0.999 });

// High stability (stationary data)
const stableModel = new ESNRegression({ rlsLambda: 0.9999 });
```

---

#### `rlsDelta`

**Default: `1.0`** | Range: `(0, ∞)`

Initial scaling for the RLS inverse correlation matrix P.

```typescript
// Default initialization
const model = new ESNRegression({ rlsDelta: 1.0 });

// More conservative start (smaller initial updates)
const conservativeModel = new ESNRegression({ rlsDelta: 0.1 });

// More aggressive start (larger initial updates)
const aggressiveModel = new ESNRegression({ rlsDelta: 10.0 });
```

---

#### `l2Lambda`

**Default: `0.0001`** | Range: `[0, ∞)`

L2 regularization strength. Prevents weight explosion and improves
generalization.

$$W_{new} = W_{old} \cdot (1 - \lambda_{L2})$$

```typescript
// No regularization (rare)
const noRegModel = new ESNRegression({ l2Lambda: 0 });

// Light regularization (default)
const lightRegModel = new ESNRegression({ l2Lambda: 0.0001 });

// Strong regularization (small datasets, overfitting)
const strongRegModel = new ESNRegression({ l2Lambda: 0.01 });
```

---

### Normalization & Robustness

#### `normalizationWarmup`

**Default: `10`** | Range: `[1, ∞)`

Number of samples before normalization activates. During warmup, statistics are
collected but not applied.

```
┌────────────────────────────────────────────────────────────────┐
│                   Normalization Warmup                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Sample:  1  2  3  4  5  6  7  8  9  10  11  12  13  ...    │
│            ├─────────────────────────┤├──────────────────────  │
│                   WARMUP PHASE        ACTIVE NORMALIZATION     │
│            Collecting statistics      z = (x - μ) / σ          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

```typescript
// Quick warmup (stable, known data distribution)
const quickWarmup = new ESNRegression({ normalizationWarmup: 5 });

// Long warmup (uncertain distribution)
const longWarmup = new ESNRegression({ normalizationWarmup: 50 });
```

---

#### `outlierThreshold`

**Default: `3.0`** | Range: `(0, ∞)`

Z-score threshold for outlier detection. Samples with prediction errors
exceeding this threshold are downweighted.

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Outlier Downweighting                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Sample weight vs Error magnitude (z-score)                           │
│                                                                        │
│   Weight                                                               │
│   1.0 ┼────────────────┐                                              │
│       │                │                                               │
│       │                │                                               │
│   0.5 ┤                └──────────┐                                   │
│       │                           │                                    │
│       │                           └────────────────┐                  │
│   0.1 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─└─────────────────  │
│       │                           ↑                                    │
│   0.0 ┼────────────────┴──────────┴────────────────────────────────►  │
│       0        1        2        3        4        5       z-score    │
│                                   │                                    │
│                        outlierThreshold                                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

```typescript
// Strict outlier detection
const strictModel = new ESNRegression({ outlierThreshold: 2.0 });

// Lenient (include more samples)
const lenientModel = new ESNRegression({ outlierThreshold: 4.0 });
```

---

#### `outlierMinWeight`

**Default: `0.1`** | Range: `(0, 1]`

Minimum weight for any sample. Even extreme outliers contribute this much.

```typescript
// Completely ignore extreme outliers
const ignoreOutliers = new ESNRegression({ outlierMinWeight: 0.01 });

// Keep more outlier contribution
const keepOutliers = new ESNRegression({ outlierMinWeight: 0.3 });
```

---

### Prediction Parameters

#### `maxFutureSteps`

**Default: `1`** | Range: `[1, ∞)`

Maximum number of future time steps to predict.

```typescript
// Single-step prediction
const singleStep = new ESNRegression({ maxFutureSteps: 1 });

// Multi-step forecasting
const multiStep = new ESNRegression({ maxFutureSteps: 10 });
```

---

#### `useDirectMultiHorizon`

**Default: `true`**

Strategy for multi-step prediction:

```
┌────────────────────────────────────────────────────────────────────────────┐
│              Multi-Step Prediction Strategies                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   DIRECT (useDirectMultiHorizon: true) [Recommended]                      │
│   ────────────────────────────────────                                     │
│                                                                            │
│   One forward pass → All horizons                                          │
│                                                                            │
│   Input ──► Reservoir ──► Readout ──┬──► y_{t+1}                          │
│                                     ├──► y_{t+2}                          │
│                                     ├──► y_{t+3}                          │
│                                     └──► ...                               │
│                                                                            │
│   ✅ Single pass                                                           │
│   ✅ No error accumulation                                                 │
│   ❌ Larger output dimension                                               │
│                                                                            │
│   ──────────────────────────────────────────────────────────────────────  │
│                                                                            │
│   RECURSIVE (useDirectMultiHorizon: false)                                │
│   ─────────────────────────────────────────                                │
│                                                                            │
│   Multiple passes, feeding predictions back                                │
│                                                                            │
│   Input ──► Reservoir ──► y_{t+1} ─┐                                      │
│                                    ▼                                       │
│                    [y_{t+1}] ──► Reservoir ──► y_{t+2} ─┐                 │
│                                                         ▼                  │
│                                         [y_{t+2}] ──► Reservoir ──► ...   │
│                                                                            │
│   ✅ Smaller model                                                         │
│   ❌ Error accumulation                                                    │
│   ❌ Multiple forward passes                                               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

```typescript
// Direct multi-horizon (recommended)
const directModel = new ESNRegression({
  maxFutureSteps: 5,
  useDirectMultiHorizon: true,
});

// Recursive (smaller model, potential drift)
const recursiveModel = new ESNRegression({
  maxFutureSteps: 5,
  useDirectMultiHorizon: false,
});
```

---

#### `uncertaintyMultiplier`

**Default: `1.96`** | Range: `(0, ∞)`

Multiplier for confidence interval computation. Default 1.96 gives ~95%
confidence intervals (assuming normality).

| Value | Confidence Level |
| ----- | ---------------- |
| 1.0   | ~68%             |
| 1.65  | ~90%             |
| 1.96  | ~95%             |
| 2.58  | ~99%             |

```typescript
// 90% confidence intervals
const ci90Model = new ESNRegression({ uncertaintyMultiplier: 1.65 });

// 99% confidence intervals
const ci99Model = new ESNRegression({ uncertaintyMultiplier: 2.58 });
```

---

## 🎯 Parameter Optimization Guide

### Decision Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Parameter Selection Flowchart                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         START                                               │
│                           │                                                 │
│              ┌────────────▼────────────┐                                    │
│              │  Data characteristics?   │                                    │
│              └────────────┬────────────┘                                    │
│                           │                                                 │
│        ┌──────────────────┼──────────────────┐                              │
│        │                  │                  │                              │
│   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐                          │
│   │  Noisy  │       │ Moderate │       │  Clean  │                          │
│   └────┬────┘       └────┬────┘       └────┬────┘                          │
│        │                 │                 │                                │
│   leakRate: 0.2     leakRate: 0.3     leakRate: 0.5                        │
│   outlierThresh: 2.0 outlierThresh: 3.0 outlierThresh: 4.0                  │
│                           │                                                 │
│              ┌────────────▼────────────┐                                    │
│              │  Time scale of patterns? │                                    │
│              └────────────┬────────────┘                                    │
│                           │                                                 │
│        ┌──────────────────┼──────────────────┐                              │
│        │                  │                  │                              │
│   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐                          │
│   │  Fast   │       │ Medium  │       │  Slow   │                          │
│   └────┬────┘       └────┬────┘       └────┬────┘                          │
│        │                 │                 │                                │
│   spectralRadius: 0.7  spectralRadius: 0.9  spectralRadius: 0.99            │
│   reservoirSize: 128   reservoirSize: 256   reservoirSize: 512              │
│                           │                                                 │
│              ┌────────────▼────────────┐                                    │
│              │  Data stationarity?     │                                    │
│              └────────────┬────────────┘                                    │
│                           │                                                 │
│        ┌──────────────────┼──────────────────┐                              │
│        │                  │                  │                              │
│   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐                          │
│   │Changing │       │  Mildly │       │Stationary│                          │
│   │ (drift) │       │ varying │       │          │                          │
│   └────┬────┘       └────┬────┘       └────┬────┘                          │
│        │                 │                 │                                │
│   rlsLambda: 0.99   rlsLambda: 0.999   rlsLambda: 0.9999                   │
│                           │                                                 │
│                           ▼                                                 │
│                        DONE                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Preset Configurations

```typescript
// 📈 Stock/Financial Data
const financialModel = new ESNRegression({
  reservoirSize: 256,
  spectralRadius: 0.95,
  leakRate: 0.4,
  rlsLambda: 0.995, // Moderate adaptation
  outlierThreshold: 2.5, // Strict outlier handling
  maxFutureSteps: 5,
});

// 🌡️ Sensor/IoT Data
const sensorModel = new ESNRegression({
  reservoirSize: 128,
  spectralRadius: 0.85,
  leakRate: 0.2, // Smooth noisy readings
  rlsLambda: 0.999,
  normalizationWarmup: 20,
  outlierThreshold: 3.0,
});

// ⚡ Fast-Changing Signals
const fastSignalModel = new ESNRegression({
  reservoirSize: 128,
  spectralRadius: 0.7,
  leakRate: 0.8,
  rlsLambda: 0.99,
});

// 🌍 Long-Term Patterns (Climate, etc.)
const longTermModel = new ESNRegression({
  reservoirSize: 512,
  spectralRadius: 0.99,
  leakRate: 0.2,
  rlsLambda: 0.9999,
  maxFutureSteps: 30,
});

// 🚀 Low-Latency/Embedded
const embeddedModel = new ESNRegression({
  reservoirSize: 64,
  reservoirSparsity: 0.95,
  spectralRadius: 0.85,
  maxFutureSteps: 1,
});
```

---

## 📖 API Reference

### `ESNRegression`

#### Constructor

```typescript
new ESNRegression(config?: ESNRegressionConfig)
```

Creates a new ESN regression model with the specified configuration.

---

#### `fitOnline(params)`

```typescript
fitOnline(params: {
  xCoordinates: number[][];  // [nSamples, nFeatures]
  yCoordinates: number[][];  // [nSamples, nTargets] or [nSamples, nTargets * maxFutureSteps]
}): FitResult
```

Train the model incrementally with new data samples.

**Returns:**

```typescript
interface FitResult {
  samplesProcessed: number; // Number of samples processed
  averageLoss: number; // Average MSE loss
  gradientNorm: number; // Update magnitude
  driftDetected: boolean; // Always false (placeholder)
  sampleWeight: number; // Last sample's weight (outlier-adjusted)
}
```

**Example:**

```typescript
const result = model.fitOnline({
  xCoordinates: [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
  yCoordinates: [[0.5, 0.6], [0.55, 0.65]],
});
console.log(`Loss: ${result.averageLoss}`);
```

---

#### `predict(futureSteps)`

```typescript
predict(futureSteps: number): PredictionResult
```

Generate predictions for future time steps.

**Parameters:**

- `futureSteps`: Number of steps ahead to predict (1 to maxFutureSteps)

**Returns:**

```typescript
interface PredictionResult {
  predictions: number[][]; // [step][target]
  lowerBounds: number[][]; // [step][target]
  upperBounds: number[][]; // [step][target]
  confidence: number; // 0.0 to 1.0
}
```

**Example:**

```typescript
const result = model.predict(3);

for (let step = 0; step < 3; step++) {
  console.log(`Step ${step + 1}:`);
  console.log(`  Predictions: ${result.predictions[step]}`);
  console.log(
    `  95% CI: [${result.lowerBounds[step]}, ${result.upperBounds[step]}]`,
  );
}
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

---

#### `getModelSummary()`

```typescript
getModelSummary(): ModelSummary
```

Get model architecture and state information.

**Returns:**

```typescript
interface ModelSummary {
  totalParameters: number;
  receptiveField: number;
  spectralRadius: number;
  reservoirSize: number;
  nFeatures: number;
  nTargets: number;
  maxSequenceLength: number;
  maxFutureSteps: number;
  sampleCount: number;
  useDirectMultiHorizon: boolean;
}
```

---

#### `getWeights()`

```typescript
getWeights(): WeightInfo
```

Get model weights for inspection or debugging.

---

#### `getNormalizationStats()`

```typescript
getNormalizationStats(): NormalizationStats
```

Get current normalization parameters.

---

#### `save()` / `load(json)`

```typescript
save(): string              // Returns JSON string
load(json: string): void    // Restores from JSON string
```

Serialize/deserialize model state.

**Example:**

```typescript
// Save model
const modelState = model.save();
localStorage.setItem("esn_model", modelState);

// Load model later
const savedState = localStorage.getItem("esn_model");
model.load(savedState);
```

---

#### `reset()`

```typescript
reset(): void
```

Reset model to initial state, clearing all training history.

---

## 💡 Examples

### Example 1: Simple Time Series Prediction

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

// Generate synthetic sine wave data
function generateSineData(n: number): { x: number[][]; y: number[][] } {
  const x: number[][] = [];
  const y: number[][] = [];

  for (let i = 0; i < n; i++) {
    const t = i * 0.1;
    x.push([Math.sin(t), Math.cos(t)]);
    y.push([Math.sin(t + 0.1)]); // Predict next value
  }

  return { x, y };
}

// Create and train model
const model = new ESNRegression({
  reservoirSize: 100,
  maxFutureSteps: 1,
});

const { x, y } = generateSineData(1000);

// Train in batches
const batchSize = 100;
for (let i = 0; i < x.length; i += batchSize) {
  const result = model.fitOnline({
    xCoordinates: x.slice(i, i + batchSize),
    yCoordinates: y.slice(i, i + batchSize),
  });
  console.log(
    `Batch ${Math.floor(i / batchSize) + 1}: Loss = ${
      result.averageLoss.toFixed(6)
    }`,
  );
}

// Predict
const prediction = model.predict(1);
console.log("Next value prediction:", prediction.predictions[0]);
```

---

### Example 2: Multi-Feature, Multi-Target Forecasting

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

// Weather-like multivariate data
interface WeatherSample {
  features: number[]; // [temperature, humidity, pressure, wind_speed]
  targets: number[]; // [next_temp, next_humidity]
}

const model = new ESNRegression({
  reservoirSize: 256,
  maxFutureSteps: 6, // Predict 6 hours ahead
  useDirectMultiHorizon: true,
  spectralRadius: 0.95,
  leakRate: 0.3,
});

// Streaming training
async function* dataStream(): AsyncGenerator<WeatherSample> {
  // Your data source here
  yield { features: [20.5, 0.65, 1013.25, 5.2], targets: [20.8, 0.63] };
  // ...
}

for await (const sample of dataStream()) {
  const result = model.fitOnline({
    xCoordinates: [sample.features],
    yCoordinates: [sample.targets],
  });

  if (result.samplesProcessed % 100 === 0) {
    console.log(
      `Samples: ${model.getModelSummary().sampleCount}, Loss: ${
        result.averageLoss.toFixed(4)
      }`,
    );
  }
}

// 6-hour forecast
const forecast = model.predict(6);
for (let h = 0; h < 6; h++) {
  console.log(`Hour ${h + 1}:`);
  console.log(`  Temperature: ${forecast.predictions[h][0].toFixed(1)}°C`);
  console.log(`  Humidity: ${(forecast.predictions[h][1] * 100).toFixed(0)}%`);
  console.log(
    `  Confidence: ±${
      (forecast.upperBounds[h][0] - forecast.predictions[h][0]).toFixed(1)
    }°C`,
  );
}
```

---

### Example 3: Model Persistence

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

// Create and train model
const model = new ESNRegression({ reservoirSize: 128 });

// ... training ...

// Save model
const savedModel = model.save();
await Deno.writeTextFile("model.json", savedModel);

// Later: Load model
const loadedJson = await Deno.readTextFile("model.json");
const restoredModel = new ESNRegression({ reservoirSize: 128 });
restoredModel.load(loadedJson);

// Continue training or predict
const prediction = restoredModel.predict(1);
```

---

### Example 4: Handling Concept Drift

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

// For non-stationary data that changes over time
const adaptiveModel = new ESNRegression({
  reservoirSize: 200,
  rlsLambda: 0.99, // Fast forgetting
  outlierThreshold: 2.5, // Strict outlier detection
  maxFutureSteps: 3,
});

// Monitor loss for drift detection
let recentLosses: number[] = [];
const windowSize = 50;

for (const sample of streamingData) {
  const result = adaptiveModel.fitOnline({
    xCoordinates: [sample.x],
    yCoordinates: [sample.y],
  });

  // Track recent losses
  recentLosses.push(result.averageLoss);
  if (recentLosses.length > windowSize) {
    recentLosses.shift();
  }

  // Detect drift via loss increase
  if (recentLosses.length === windowSize) {
    const oldLoss = recentLosses.slice(0, windowSize / 2).reduce((a, b) =>
      a + b
    ) / (windowSize / 2);
    const newLoss = recentLosses.slice(windowSize / 2).reduce((a, b) => a + b) /
      (windowSize / 2);

    if (newLoss > oldLoss * 2) {
      console.log("⚠️ Possible concept drift detected!");
    }
  }
}
```

---

## 🏗️ Architecture

### Internal Components Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ESNRegression Internal Architecture                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                              PUBLIC API                                      │   │
│  │  ┌───────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │  fitOnline()  │  │  predict()  │  │ save/load()  │  │ getModelSummary│  │   │
│  │  └───────┬───────┘  └──────┬──────┘  └──────────────┘  └────────────────┘  │   │
│  └──────────┼─────────────────┼───────────────────────────────────────────────┘   │
│             │                 │                                                     │
│             ▼                 ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           CORE COMPONENTS                                    │   │
│  │                                                                              │   │
│  │  ┌──────────────────┐    ┌─────────────────────────────────────────────┐    │   │
│  │  │  WelfordNormalizer│    │              ESN Model                       │    │   │
│  │  │  ─────────────────│    │  ┌───────────────────────────────────────┐  │    │   │
│  │  │  • Online mean    │    │  │  ESN Reservoir                        │  │    │   │
│  │  │  • Online std     │───►│  │  • Win [rs × nF]                      │  │    │   │
│  │  │  • Z-score norm   │    │  │  • W   [rs × rs] (spectral scaled)   │  │    │   │
│  │  └──────────────────┘    │  │  • bias [rs]                          │  │    │   │
│  │                          │  │  • state [rs] (leaky integrator)      │  │    │   │
│  │  ┌──────────────────┐    │  └───────────────────┬───────────────────┘  │    │   │
│  │  │   Ring Buffer    │    │                      │                       │    │   │
│  │  │  ────────────────│    │                      ▼                       │    │   │
│  │  │  • History store │    │  ┌───────────────────────────────────────┐  │    │   │
│  │  │  • Window extract│    │  │  Linear Readout                       │  │    │   │
│  │  │  • Circular FIFO │    │  │  • Wout [output × input]              │  │    │   │
│  │  └──────────────────┘    │  │  • Extended state: z = [r; x; 1]      │  │    │   │
│  │                          │  └───────────────────┬───────────────────┘  │    │   │
│  │  ┌──────────────────┐    │                      │                       │    │   │
│  │  │  Residual Tracker│    │                      ▼                       │    │   │
│  │  │  ────────────────│    │  ┌───────────────────────────────────────┐  │    │   │
│  │  │  • Error stats   │◄───│  │  RLS Optimizer                        │  │    │   │
│  │  │  • Uncertainty   │    │  │  • P matrix [input × input]           │  │    │   │
│  │  │  • Confidence    │    │  │  • Kalman gain                        │  │    │   │
│  │  └──────────────────┘    │  │  • Sherman-Morrison updates           │  │    │   │
│  │                          │  └───────────────────────────────────────┘  │    │   │
│  │  ┌──────────────────┐    └─────────────────────────────────────────────┘    │   │
│  │  │ OutlierDownweight│                                                        │   │
│  │  │  ────────────────│                                                        │   │
│  │  │  • Z-score check │                                                        │   │
│  │  │  • Sample weights│                                                        │   │
│  │  └──────────────────┘                                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        MEMORY MANAGEMENT                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐  │   │
│  │  │ TensorArena  │  │ BufferPool   │  │     TensorOps (zero-copy)        │  │   │
│  │  │ (params)     │  │ (scratch)    │  │  matvec, outer, dot, scale, etc. │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance Tips

### 1️⃣ Memory Efficiency

```typescript
// ✅ Good: Process in batches
for (let i = 0; i < data.length; i += 100) {
  model.fitOnline({
    xCoordinates: data.slice(i, i + 100).map((d) => d.x),
    yCoordinates: data.slice(i, i + 100).map((d) => d.y),
  });
}

// ❌ Avoid: Single sample at a time (more overhead)
for (const sample of data) {
  model.fitOnline({ xCoordinates: [sample.x], yCoordinates: [sample.y] });
}
```

### 2️⃣ Reservoir Size vs. Speed

```typescript
// Trade-off analysis
const benchmarks = [
  { size: 64, opsPerSec: 50000 },
  { size: 128, opsPerSec: 20000 },
  { size: 256, opsPerSec: 8000 },
  { size: 512, opsPerSec: 2500 },
];
// Choose based on your latency requirements
```

### 3️⃣ Sparsity for Speed

```typescript
// High sparsity = fewer computations
const fastModel = new ESNRegression({
  reservoirSize: 512,
  reservoirSparsity: 0.95, // 95% sparse = 5% of weights active
});
```

### 4️⃣ Reuse Prediction Results

```typescript
// ✅ Good: Single predict call
const result = model.predict(5);
for (let i = 0; i < 5; i++) {
  process(result.predictions[i]);
}

// ❌ Avoid: Multiple predict calls for same horizon
for (let i = 1; i <= 5; i++) {
  const result = model.predict(i); // Redundant computation
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 🔴 "Model not initialized" Error

```typescript
// Problem: Calling predict() before fitOnline()
const model = new ESNRegression();
model.predict(1); // ❌ Error!

// Solution: Train first
model.fitOnline({ xCoordinates: [[1, 2]], yCoordinates: [[3]] });
model.predict(1); // ✅ Works
```

#### 🔴 High Loss / Poor Predictions

```typescript
// Check 1: Data normalization
console.log(model.getNormalizationStats());
// If stds are very large/small, data may have issues

// Check 2: Reservoir dynamics
const summary = model.getModelSummary();
console.log("Spectral radius:", summary.spectralRadius);
// Try adjusting spectralRadius if patterns are lost

// Check 3: Warmup period
if (summary.sampleCount < 100) {
  console.log("Need more training samples");
}
```

#### 🔴 Numerical Instability (NaN/Inf)

```typescript
// Solution: Add regularization and clipping
const stableModel = new ESNRegression({
  l2Lambda: 0.001, // Stronger regularization
  gradientClipNorm: 0.5, // More aggressive clipping
  inputScale: 0.5, // Reduce input magnitude
});
```

#### 🔴 Slow Training

```typescript
// Solution: Reduce reservoir size and increase sparsity
const fastModel = new ESNRegression({
  reservoirSize: 128, // Smaller reservoir
  reservoirSparsity: 0.95, // More sparse
  inputSparsity: 0.5, // Sparse input connections
});
```

---

## 📊 Comparison with Other Methods

| Method            | Online Learning | Multi-step | Memory Efficient | Setup Complexity |
| ----------------- | :-------------: | :--------: | :--------------: | :--------------: |
| **ESNRegression** |       ✅        |     ✅     |        ✅        |       Low        |
| ARIMA             |       ❌        |     ⚠️     |        ✅        |      Medium      |
| LSTM              |       ⚠️        |     ✅     |        ❌        |       High       |
| Transformer       |       ❌        |     ✅     |        ❌        |       High       |
| Prophet           |       ❌        |     ✅     |        ✅        |       Low        |

---

## 📄 License

**MIT License** © 2025 [Henrique Emanoel Viana](https://github.com/hviana)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

<div align="center">

**[⬆ Back to Top](#-esnregression---echo-state-network-for-multivariate-regression)**

Made with ❤️ by [Henrique Emanoel Viana](https://github.com/hviana)

</div>
