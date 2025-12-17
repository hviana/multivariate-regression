# 🧠 ESNRegression

<div align="center">
**Self-contained TypeScript Echo State Network (ESN) / Reservoir Computing library for online multivariate regression**

_Created by **Henrique Emanoel Viana**_

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🎓 Understanding Echo State Networks](#-understanding-echo-state-networks)
- [🔧 Configuration Parameters](#-configuration-parameters)
- [📖 API Reference](#-api-reference)
- [💡 Examples & Use Cases](#-examples--use-cases)
- [🎯 Parameter Optimization Guide](#-parameter-optimization-guide)
- [📊 Performance Tips](#-performance-tips)
- [📜 License](#-license)

---

## ✨ Features

<div align="center">

| Feature                           | Description                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| 🔄 **Online Learning**            | Real-time incremental learning with RLS (Recursive Least Squares) |
| 📈 **Multivariate Regression**    | Handle multiple input features and output targets simultaneously  |
| 🔮 **Multi-Horizon Prediction**   | Forecast multiple steps into the future with confidence intervals |
| 🎯 **Outlier Robust**             | Automatic outlier detection and downweighting                     |
| 📊 **Adaptive Normalization**     | Welford's online algorithm for streaming statistics               |
| 🔒 **Deterministic**              | Reproducible results with seeded random number generation         |
| ⚡ **Zero Dependencies**          | Self-contained implementation with no external libraries          |
| 🧮 **Memory Efficient**           | Pre-allocated tensor arena with minimal garbage collection        |
| 💾 **Serialization**              | Full save/load support for model persistence                      |
| 📐 **Uncertainty Quantification** | Prediction intervals with configurable confidence levels          |

</div>

### 🌟 Key Highlights

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ⚡ REAL-TIME          🎯 ACCURATE           📊 INTERPRETABLE           │
│     Processing            Predictions            Results                │
│                                                                         │
│  • Stream data          • Multi-horizon        • Confidence bounds      │
│  • No batching            forecasting          • Residual tracking      │
│  • Instant updates      • Autoregressive       • Weight inspection      │
│                           rollout                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Deno / JSR

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";
```

### NPM (via JSR)

```bash
npx jsr add @hviana/multivariate-regression
```

```typescript
import { ESNRegression } from "@hviana/multivariate-regression";
```

---

## ⚡ Quick Start

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

// 🔨 Create model with configuration
const model = new ESNRegression({
  reservoirSize: 256,
  maxSequenceLength: 64,
  spectralRadius: 0.9,
  leakRate: 0.3,
});

// 📥 Prepare training data
const xCoordinates = [
  [1.0, 2.0, 3.0], // Features at t=0
  [1.1, 2.1, 3.1], // Features at t=1
  [1.2, 2.2, 3.2], // Features at t=2
  // ... more samples
];

const yCoordinates = [
  [4.0, 5.0], // Targets at t=0
  [4.1, 5.1], // Targets at t=1
  [4.2, 5.2], // Targets at t=2
  // ... more samples
];

// 🎯 Train the model (online, incremental)
const fitResult = model.fitOnline({ xCoordinates, yCoordinates });

console.log(`📊 Samples processed: ${fitResult.samplesProcessed}`);
console.log(`📉 Average loss: ${fitResult.averageLoss.toFixed(6)}`);

// 🔮 Predict future values
const predictions = model.predict(10); // Predict 10 steps ahead

console.log("🔮 Predictions:", predictions.predictions);
console.log("📊 Confidence:", predictions.confidence);
console.log("📉 Lower bounds:", predictions.lowerBounds);
console.log("📈 Upper bounds:", predictions.upperBounds);
```

---

## 🎓 Understanding Echo State Networks

### 🧠 What is an Echo State Network?

An **Echo State Network (ESN)** is a type of recurrent neural network that
belongs to the **Reservoir Computing** paradigm. The key innovation is that only
the **output weights are trained**, while the internal reservoir weights remain
fixed after initialization.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    ECHO STATE NETWORK ARCHITECTURE                          │
│                                                                             │
│    ┌─────────┐      ┌────────────────────────────────────┐     ┌─────────┐  │
│    │         │      │           RESERVOIR                │     │         │  │
│    │  INPUT  │────▶│    ┌───┐  ┌───┐  ┌───┐  ┌───┐      │───▶│ OUTPUT  │  │
│    │   x(t)  │ Win  │    │ N₁├──┤ N₂├──┤ N₃├──┤ N₄│      │Wout │  y(t)   │  │
│    │         │      │    └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘      │     │         │  │
│    └─────────┘      │      │      │      │      │        │     └─────────┘  │
│                     │      └──────┴──────┴──────┘        │                  │
│                     │           Recurrent W              │                  │
│                     │         (Fixed weights)            │                  │
│                     └────────────────────────────────────┘                  │
│                                                                             │
│    Legend:                                                                  │
│    ═══════                                                                  │
│    Win  = Input weights (fixed after init)                                  │
│    W    = Reservoir weights (fixed after init)                              │
│    Wout = Output weights (TRAINED via RLS)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         ESN PROCESSING PIPELINE                             │
│                                                                             │
│   ┌──────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐      │
│   │ Raw Data │──▶│ Normalize  │──▶│  Reservoir │──▶│ Build State  │      │
│   │  x_raw   │    │   x_norm   │    │   Update   │    │   Vector z   │      │
│   └──────────┘    └────────────┘    └────────────┘    └──────┬───────┘      │
│                                                              │              │
│                                                              ▼              │
│   ┌──────────┐    ┌────────────┐    ┌────────────┐    ┌──────────────┐      │
│   │  Output  │◀──│   Linear   │◀──│  Weighted  │◀──│   Concat:    │      │
│   │   y_hat  │    │   Readout  │    │     RLS    │    │ [r, x, bias] │      │
│   └──────────┘    └────────────┘    └────────────┘    └──────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 📐 Mathematical Foundation

#### Reservoir State Update (Leaky Integration)

The reservoir state evolves according to:

```
r(t) = (1 - α) · r(t-1) + α · f(Win · (s · x(t)) + W · r(t-1) + b)
```

Where:

- **r(t)** = Current reservoir state
- **α** = Leak rate (temporal smoothing)
- **f** = Activation function (tanh or ReLU)
- **Win** = Input weight matrix
- **s** = Input scale factor
- **W** = Reservoir weight matrix
- **b** = Bias vector

#### Output Computation

```
z(t) = [r(t), x(t), 1]  (concatenation)
y(t) = Wout · z(t)
```

#### Recursive Least Squares (RLS) Update

```
k = P·z / (λ + z'·P·z)
Wout = Wout + k·(y_true - y_hat)'
P = (P - k·z'·P) / λ
```

### 🌀 Why Reservoir Computing Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    THE ECHO STATE PROPERTY                                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  When spectral radius < 1, the reservoir has "fading memory":    │       │
│  │                                                                  │       │
│  │  • Past inputs influence decays exponentially over time          │       │
│  │  • Network state is uniquely determined by input history         │       │
│  │  • No exploding/vanishing gradient problems                      │       │
│  │                                                                  │       │
│  │            Memory Decay                                          │       │
│  │        ▲                                                         │       │
│  │        │  ████                                                   │       │
│  │        │  ████ ▓▓▓▓                                              │       │
│  │        │  ████ ▓▓▓▓ ░░░░                                         │       │
│  │        │  ████ ▓▓▓▓ ░░░░ ····                                    │       │
│  │        └──────────────────────▶ Time                            │       │
│  │           t-3   t-2   t-1   t                                    │       │
│  │                                                                  │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration Parameters

### 📊 Complete Configuration Reference

```typescript
interface ESNRegressionConfig {
  // 🔄 Reservoir Architecture
  maxSequenceLength: number; // Default: 64
  reservoirSize: number; // Default: 256
  spectralRadius: number; // Default: 0.9
  leakRate: number; // Default: 0.3
  inputScale: number; // Default: 1.0
  biasScale: number; // Default: 0.1
  reservoirSparsity: number; // Default: 0.9
  inputSparsity: number; // Default: 0.0
  activation: "tanh" | "relu"; // Default: "tanh"

  // 📤 Readout Configuration
  useInputInReadout: boolean; // Default: true
  useBiasInReadout: boolean; // Default: true

  // 🎯 Training (RLS)
  readoutTraining: "rls"; // Default: "rls"
  rlsLambda: number; // Default: 0.999
  rlsDelta: number; // Default: 1.0
  epsilon: number; // Default: 1e-8
  l2Lambda: number; // Default: 0.0001
  gradientClipNorm: number; // Default: 1.0

  // 📊 Normalization
  normalizationEpsilon: number; // Default: 1e-8
  normalizationWarmup: number; // Default: 10

  // 🛡️ Outlier Handling
  outlierThreshold: number; // Default: 3.0
  outlierMinWeight: number; // Default: 0.1

  // 📈 Uncertainty
  residualWindowSize: number; // Default: 100
  uncertaintyMultiplier: number; // Default: 1.96

  // ⚙️ Initialization
  weightInitScale: number; // Default: 0.1
  seed: number; // Default: 42
  verbose: boolean; // Default: false
  rollforwardMode: "holdLastX" | "autoregressive"; // Default: "holdLastX"
}
```

---

### 🔄 Reservoir Architecture Parameters

#### `reservoirSize` 🎯

**What it does:** Determines the number of neurons in the reservoir (hidden
layer).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RESERVOIR SIZE IMPACT                                  │
│                                                                             │
│   Size: 64              Size: 256             Size: 1024                    │
│   ┌───────┐             ┌───────────┐         ┌─────────────────┐           │
│   │ • • • │             │ • • • • • │         │ • • • • • • • • │           │
│   │ • • • │             │ • • • • • │         │ • • • • • • • • │           │
│   │ • • • │             │ • • • • • │         │ • • • • • • • • │           │
│   └───────┘             │ • • • • • │         │ • • • • • • • • │           │
│   Fast, limited         │ • • • • • │         │ • • • • • • • • │           │
│   expressiveness        └───────────┘         └─────────────────┘           │
│                         Balanced              High capacity,                │
│                                               slower training               │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Use Case              | Recommended Size | Rationale            |
| --------------------- | ---------------- | -------------------- |
| Simple linear trends  | 32-64            | Low complexity, fast |
| Standard time series  | 128-256          | Good balance         |
| Complex patterns      | 512-1024         | High capacity needed |
| Multi-variate complex | 256-512          | Per-target capacity  |

**Example:**

```typescript
// 🚀 For simple univariate prediction
const simpleModel = new ESNRegression({
  reservoirSize: 64,
});

// 🎯 For complex multivariate forecasting
const complexModel = new ESNRegression({
  reservoirSize: 512,
});
```

---

#### `spectralRadius` 📊

**What it does:** Controls the "memory" of the network. It's the largest
eigenvalue of the reservoir weight matrix.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SPECTRAL RADIUS EFFECT                                  │
│                                                                             │
│  Memory Retention                                                           │
│       ▲                                                                     │
│       │                                                                     │
│  1.0 ─┤                               ┌───── ρ = 0.99 (Long memory)         │
│       │                          ┌────┘                                     │
│       │                     ┌────┘                                          │
│  0.5 ─┤                ┌────┘          ┌───── ρ = 0.9 (Medium memory)       │
│       │           ┌────┘               │                                    │
│       │      ┌────┘              ┌─────┘                                    │
│       │ ┌────┘              ┌────┘      ┌───── ρ = 0.5 (Short memory)       │
│  0.0 ─┴─┴───────────────────┴──────────┴────────────▶ Time Steps           │
│       0         5         10        15        20                            │
│                                                                             │
│  ⚠️  Warning: ρ ≥ 1.0 can cause instability (loss of echo state property)   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Data Characteristics          | Recommended ρ | Why                               |
| ----------------------------- | ------------- | --------------------------------- |
| Rapid changes, short patterns | 0.5 - 0.7     | Quick adaptation                  |
| Standard time series          | 0.8 - 0.95    | Balanced memory                   |
| Long-term dependencies        | 0.95 - 0.99   | Extended memory                   |
| Near edge of chaos            | 0.99          | Maximum expressiveness (careful!) |

**Example:**

```typescript
// 📈 Stock prices (long memory needed)
const stockModel = new ESNRegression({
  spectralRadius: 0.95,
});

// ⚡ Sensor data (rapid changes)
const sensorModel = new ESNRegression({
  spectralRadius: 0.7,
});
```

---

#### `leakRate` 💧

**What it does:** Controls temporal smoothing in the reservoir update. Values
closer to 1 mean faster updates; values closer to 0 provide more smoothing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEAK RATE DYNAMICS                                  │
│                                                                             │
│  r(t) = (1 - α) · r(t-1) + α · f(...)                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                                                                 │        │
│  │    α = 0.1 (Slow leak)     │    α = 0.9 (Fast leak)             │        │
│  │    ┌──────────────────┐    │    ┌──────────────────┐            │        │
│  │    │   ▓▓▓▓▓▓▓▓▓▓▓    │    │    │   ▓▓▓▓           │            │        │
│  │    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓   │    │    │  ▓▓▓▓▓▓▓▓        │            │        │
│  │    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │    │    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │            │        │
│  │    └──────────────────┘    │    └──────────────────┘            │        │
│  │    Smooth, averaged        │    Responsive, reactive            │        │
│  │                            │                                    │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Application           | Recommended α | Behavior           |
| --------------------- | ------------- | ------------------ |
| Noisy data            | 0.1 - 0.3     | Smoothing effect   |
| Standard forecasting  | 0.3 - 0.5     | Balanced           |
| Fast-changing signals | 0.6 - 0.9     | Quick response     |
| Real-time tracking    | 0.8 - 1.0     | Immediate reaction |

**Example:**

```typescript
// 🌊 Noisy sensor smoothing
const smoothModel = new ESNRegression({
  leakRate: 0.2,
});

// ⚡ High-frequency trading
const fastModel = new ESNRegression({
  leakRate: 0.8,
});
```

---

#### `inputScale` 📏

**What it does:** Scales the input before feeding to the reservoir.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT SCALING EFFECT                                 │
│                                                                             │
│   Input Scale = 0.1          Input Scale = 1.0          Input Scale = 3.0   │
│   ┌────────────────┐         ┌────────────────┐         ┌────────────────┐  │
│   │    ·····       │         │   ╱╲           │         │ ███████████████│  │
│   │   ·····        │         │  ╱  ╲          │         │██████████████ █│  │
│   │    ·····       │         │ ╱    ╲╱╲       │         │████ ████  ███  │  │
│   └────────────────┘         └────────────────┘         └────────────────┘  │
│   Weak influence             Balanced                   Strong, may saturate│
│   (underutilized)            (recommended)              activation function │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Data Type                 | Recommended Scale | Notes                       |
| ------------------------- | ----------------- | --------------------------- |
| Pre-normalized (-1 to 1)  | 0.5 - 1.0         | Standard range              |
| Large magnitude           | 0.1 - 0.5         | Prevent saturation          |
| Small signals             | 1.0 - 2.0         | Amplify for better dynamics |
| With online normalization | 1.0               | Let normalizer handle it    |

---

#### `biasScale` ⚖️

**What it does:** Scales the random bias values in the reservoir.

```typescript
// Typical configurations
const model = new ESNRegression({
  biasScale: 0.1, // Default - small bias contribution
});

// For breaking symmetry in sparse reservoirs
const sparseModel = new ESNRegression({
  reservoirSparsity: 0.95,
  biasScale: 0.2, // Slightly larger to add diversity
});
```

---

#### `reservoirSparsity` 🕸️

**What it does:** Controls the proportion of zero connections in the reservoir
matrix.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RESERVOIR SPARSITY                                    │
│                                                                             │
│   Sparsity = 0.0 (Dense)      Sparsity = 0.9 (90% zeros)                    │
│   ┌────────────────────┐      ┌──────────────────┐                          │
│   │ ████████████████   │      │ ·  ·  █  ·  ·  · │                          │
│   │ ████████████████   │      │ ·  █  ·  ·  █  · │                          │
│   │ ████████████████   │      │ █  ·  ·  ·  ·  █ │                          │
│   │ ████████████████   │      │ ·  ·  █  ·  ·  · │                          │
│   └────────────────────┘      └──────────────────┘                          │
│   Slow, potentially           Fast, biologically                            │
│   overfit                     plausible                                     │
│                                                                             │
│   🎯 Recommended: 0.8 - 0.95 for most applications                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Example:**

```typescript
// Standard sparse reservoir (recommended)
const model = new ESNRegression({
  reservoirSparsity: 0.9, // 90% zeros, 10% connections
});

// Dense reservoir (more capacity, slower)
const denseModel = new ESNRegression({
  reservoirSparsity: 0.5,
});
```

---

#### `inputSparsity` 📥

**What it does:** Controls sparsity of input-to-reservoir connections.

| Setting     | Use Case                       |
| ----------- | ------------------------------ |
| 0.0 (dense) | All features equally important |
| 0.3 - 0.5   | Feature selection effect       |
| 0.7 - 0.9   | Very high-dimensional inputs   |

---

#### `activation` ⚡

**What it does:** Non-linear activation function for reservoir neurons.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ACTIVATION FUNCTIONS                                  │
│                                                                             │
│      tanh                              relu                                 │
│       ▲                                 ▲                                   │
│   1.0─┤      ╭────────              1.0─┤           ╱╱╱╱                    │
│       │    ╭─╯                          │         ╱╱                        │
│   0.0─┼────╯──────────              0.0─┼────────╱────────                  │
│       │──╮                              │────────                           │
│  -1.0─┤  ╰────────                 -1.0─┤                                   │
│       └────────────▶                    └────────────▶                    │
│                                                                             │
│   • Bounded (-1, 1)              • Unbounded (0, ∞)                         │
│   • Smoother gradients           • Sparse activations                       │
│   • ✅ Default choice            • Good for positive data                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Example:**

```typescript
// Standard (recommended for most cases)
const tanhModel = new ESNRegression({
  activation: "tanh",
});

// For positive-only predictions
const reluModel = new ESNRegression({
  activation: "relu",
});
```

---

#### `maxSequenceLength` 📏

**What it does:** Maximum temporal context window and prediction horizon limit.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SEQUENCE LENGTH CONTEXT                                 │
│                                                                             │
│   maxSequenceLength = 64                                                    │
│                                                                             │
│   ◀──────────── History Buffer (Ring Buffer) ──────────────▶              │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐             │
│   │ t₀ │ t₁ │ t₂ │ t₃ │ .. │t₆₁│t₆₂│t₆₃│ 🔮 │ 🔮 │ 🔮 │ .. │                │
│   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘             │
│   ◀─────────── Stored Data ──────────▶│◀── predict(N) ────▶             │
│                                                                             │
│   ⚠️  predict(futureSteps) must be ≤ maxSequenceLength                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Scenario            | Recommended Length | Notes                   |
| ------------------- | ------------------ | ----------------------- |
| Real-time streaming | 32-64              | Low latency             |
| Daily forecasting   | 64-128             | ~2 months of daily data |
| Long-term patterns  | 128-512            | Seasonal effects        |

---

### 📤 Readout Configuration

#### `useInputInReadout` 📎

**What it does:** When `true`, appends current input to reservoir state for
output computation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      READOUT STATE COMPOSITION                              │
│                                                                             │
│   useInputInReadout: true    │    useInputInReadout: false                  │
│   useBiasInReadout: true     │    useBiasInReadout: false                   │
│                              │                                              │
│   z = [r₁,r₂,...,rₙ, x₁,x₂,xₘ, 1]    z = [r₁,r₂,...,rₙ]                     │
│       └────┬─────┘  └───┬───┘  └┬┘       └────┬─────┘                       │
│        reservoir      input   bias        reservoir                         │
│         state                  only                                         │
│                                                                             │
│   ✅ Better for:             │    ✅ Better for:                            │
│   • Direct input influence   │    • Pure temporal features                  │
│   • Skip connections         │    • Minimal state size                      │
│                              │                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Default recommendation:** Keep both `true` for most applications.

---

### 🎯 Training Parameters (RLS)

#### `rlsLambda` λ

**What it does:** Forgetting factor for Recursive Least Squares. Controls how
quickly old information is discarded.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RLS FORGETTING FACTOR                                 │
│                                                                             │
│   λ = 0.99 (Slow forget)         λ = 0.95 (Fast forget)                     │
│                                                                             │
│   Weight on past data:           Weight on past data:                       │
│   ████████████████████           ██████████████                             │
│    ██████████████████             █████████████                             │
│     █████████████████              ███████████                              │
│      ████████████████               █████████                               │
│       ███████████████                ███████                                │
│   ◀─── Past ─────────▶           ◀─── Past ────▶                        │
│                                                                             │
│   • Stable learning              • Adaptive to changes                      │
│   • Good for stationary data     • Good for non-stationary data             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Data Behavior   | λ Value        | Why                      |
| --------------- | -------------- | ------------------------ |
| Stationary      | 0.999 - 0.9999 | Stable, uses all history |
| Slowly drifting | 0.995 - 0.999  | Balanced                 |
| Concept drift   | 0.95 - 0.99    | Quick adaptation         |
| Rapid changes   | 0.9 - 0.95     | Very responsive          |

**Example:**

```typescript
// Stable environment
const stableModel = new ESNRegression({
  rlsLambda: 0.999,
});

// Non-stationary data with drift
const adaptiveModel = new ESNRegression({
  rlsLambda: 0.97,
});
```

---

#### `rlsDelta` δ

**What it does:** Initial value for the diagonal of the P matrix (inverse
covariance). Larger values = faster initial learning.

```typescript
// Quick initial convergence
const quickStart = new ESNRegression({
  rlsDelta: 10.0,
});

// Conservative start
const conservativeStart = new ESNRegression({
  rlsDelta: 0.1,
});
```

| Setting       | Effect                             |
| ------------- | ---------------------------------- |
| 0.01 - 0.1    | Slow, conservative initial updates |
| 1.0 (default) | Balanced                           |
| 10.0 - 100.0  | Aggressive initial learning        |

---

#### `l2Lambda` 🛡️

**What it does:** L2 regularization (weight decay) applied to readout weights.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        L2 REGULARIZATION EFFECT                             │
│                                                                             │
│   No Regularization (l2Lambda = 0)    With Regularization (l2Lambda > 0)    │
│                                                                             │
│   Weight magnitudes:                  Weight magnitudes:                    │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                     ▓▓▓▓▓▓                                 │
│       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                     ▓▓▓▓▓                              │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                 ▓▓▓▓▓▓▓                                │
│                                                                             │
│   • May overfit                      • Prevents overfitting                 │
│   • Potentially unstable             • More stable                          │
│   • Large weight swings              • Smoother predictions                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Data Size            | Recommended l2Lambda | Notes                 |
| -------------------- | -------------------- | --------------------- |
| Small (<100 samples) | 0.01 - 0.1           | Strong regularization |
| Medium (100-1000)    | 0.0001 - 0.001       | Moderate              |
| Large (>1000)        | 0.00001 - 0.0001     | Light regularization  |

---

#### `gradientClipNorm` ✂️

**What it does:** Clips the update norm to prevent explosive updates.

```typescript
// Standard (default)
const model = new ESNRegression({
  gradientClipNorm: 1.0,
});

// More aggressive clipping for unstable data
const safeModel = new ESNRegression({
  gradientClipNorm: 0.5,
});

// Disabled (not recommended)
const unclippedModel = new ESNRegression({
  gradientClipNorm: 0, // No clipping
});
```

---

### 📊 Normalization Parameters

#### `normalizationWarmup` 🔥

**What it does:** Number of samples before online normalization becomes active.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      NORMALIZATION WARMUP                                  │
│                                                                            │
│   Samples:  1   2   3   4   5   6   7   8   9  10  11  12 ...              │
│            ─────────────────┬────────────────────────────────              │
│            Warmup Phase     │    Normal Operation                          │
│            (collecting      │    (active normalization)                    │
│             statistics)     │                                              │
│                             │                                              │
│   normalizationWarmup = 10 ─┘                                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### 🛡️ Outlier Handling Parameters

#### `outlierThreshold` 🎯

**What it does:** Z-score threshold above which samples are considered outliers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OUTLIER DETECTION                                     │
│                                                                             │
│   Residual Distribution                                                     │
│                                                                             │
│                        ┌───────┐                                            │
│                       ╱│       │╲                                           │
│                      ╱ │       │ ╲                                          │
│                     ╱  │       │  ╲                                         │
│                    ╱   │       │   ╲                                        │
│   ─────────────────────┴───────┴─────────────────────                       │
│            ◀─3σ─▶│◀──Normal──▶│◀─3σ─▶                                 │
│                  │            │                                             │
│            Outlier Zone  │    Outlier Zone                                  │
│                                                                             │
│   outlierThreshold = 3.0 → Samples beyond 3σ are downweighted               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Setting       | Detection Rate   | Use Case              |
| ------------- | ---------------- | --------------------- |
| 2.0           | ~5% outliers     | Aggressive filtering  |
| 3.0 (default) | ~0.3% outliers   | Standard              |
| 4.0           | ~0.006% outliers | Only extreme outliers |

---

#### `outlierMinWeight` ⚖️

**What it does:** Minimum weight assigned to detected outliers (prevents
complete exclusion).

```typescript
// Standard - outliers still contribute minimally
const model = new ESNRegression({
  outlierThreshold: 3.0,
  outlierMinWeight: 0.1, // 10% weight for outliers
});

// Zero tolerance - completely ignore extreme outliers
const strictModel = new ESNRegression({
  outlierThreshold: 2.5,
  outlierMinWeight: 0.0, // Full exclusion
});
```

---

### 📈 Uncertainty Quantification

#### `residualWindowSize` 📊

**What it does:** Number of recent residuals used to estimate prediction
uncertainty.

```typescript
// Short window - reacts quickly to error changes
const reactiveModel = new ESNRegression({
  residualWindowSize: 50,
});

// Long window - stable uncertainty estimates
const stableModel = new ESNRegression({
  residualWindowSize: 200,
});
```

---

#### `uncertaintyMultiplier` 📐

**What it does:** Multiplier for confidence interval width (default 1.96 ≈ 95%
CI).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   UNCERTAINTY MULTIPLIER (Gaussian)                         │
│                                                                             │
│   Multiplier │ Confidence Level │ Interpretation                            │
│   ───────────┼──────────────────┼─────────────────────────                  │
│     1.00     │      68.3%       │ Within 1 std deviation                    │
│     1.64     │      90.0%       │ Common for forecasting                    │
│     1.96     │      95.0%       │ Standard (default)                        │
│     2.58     │      99.0%       │ High confidence                           │
│     3.00     │      99.7%       │ Very conservative                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Example:**

```typescript
// 90% confidence intervals
const model90 = new ESNRegression({
  uncertaintyMultiplier: 1.64,
});

// 99% confidence intervals (wider bands)
const model99 = new ESNRegression({
  uncertaintyMultiplier: 2.58,
});
```

---

### ⚙️ Initialization & Control

#### `seed` 🌱

**What it does:** Random seed for deterministic weight initialization.

```typescript
// Same seed = identical results
const model1 = new ESNRegression({ seed: 42 });
const model2 = new ESNRegression({ seed: 42 });
// model1 and model2 will produce identical results

// Different seeds for ensemble diversity
const ensemble = [
  new ESNRegression({ seed: 1 }),
  new ESNRegression({ seed: 2 }),
  new ESNRegression({ seed: 3 }),
];
```

---

#### `rollforwardMode` 🔄

**What it does:** Determines how multi-step predictions are generated.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PREDICTION ROLLFORWARD MODES                            │
│                                                                             │
│   "holdLastX" (Default)           │   "autoregressive"                      │
│   ────────────────────            │   ───────────────────                   │
│                                   │                                         │
│   x_known ─┬─▶ ŷ₁                │   x_known ─┬─▶ ŷ₁ ──┐                  │
│            │                      │            │        │                   │
│   x_known ─┼─▶ ŷ₂                │            └─▶ ŷ₂ ──┤ (ŷ₁ as x)        │
│            │                      │               │     │                   │
│   x_known ─┴─▶ ŷ₃                │               └─▶ ŷ₃ (ŷ₂ as x)         │
│                                   │                                         │
│   ✅ Safe, no error              │   ✅ True multi-step                     │
│      accumulation                │      (requires nFeatures == nTargets)    │
│                                   │                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Example:**

```typescript
// Standard forecasting (safer)
const holdModel = new ESNRegression({
  rollforwardMode: "holdLastX",
});

// Autoregressive (when features = targets)
const arModel = new ESNRegression({
  rollforwardMode: "autoregressive",
});
```

---

## 📖 API Reference

### 🔨 Constructor

```typescript
constructor(config?: Partial<ESNRegressionConfig>)
```

Creates a new ESNRegression instance with optional configuration overrides.

---

### 📥 `fitOnline()`

```typescript
fitOnline(args: { 
  xCoordinates: number[][]; 
  yCoordinates: number[][] 
}): FitResult
```

Incrementally trains the model with new samples.

```typescript
interface FitResult {
  samplesProcessed: number; // Number of samples in this batch
  averageLoss: number; // Running average MSE
  gradientNorm: number; // L2 norm of last weight update
  driftDetected: boolean; // Reserved for drift detection
  sampleWeight: number; // Weight of last sample (outlier handling)
}
```

**Example:**

```typescript
// Single sample update
const result = model.fitOnline({
  xCoordinates: [[1.0, 2.0]],
  yCoordinates: [[3.0]],
});

// Batch update
const batchResult = model.fitOnline({
  xCoordinates: [
    [1.0, 2.0],
    [1.1, 2.1],
    [1.2, 2.2],
  ],
  yCoordinates: [
    [3.0],
    [3.1],
    [3.2],
  ],
});
```

---

### 🔮 `predict()`

```typescript
predict(futureSteps: number): PredictionResult
```

Generates multi-horizon predictions with uncertainty bounds.

```typescript
interface PredictionResult {
  predictions: number[][]; // [futureSteps][nTargets]
  lowerBounds: number[][]; // Lower confidence bounds
  upperBounds: number[][]; // Upper confidence bounds
  confidence: number; // Overall confidence (0-1)
}
```

**Example:**

```typescript
const predictions = model.predict(5);

for (let step = 0; step < predictions.predictions.length; step++) {
  console.log(`Step ${step + 1}:`);
  console.log(`  Prediction: ${predictions.predictions[step]}`);
  console.log(
    `  95% CI: [${predictions.lowerBounds[step]}, ${
      predictions.upperBounds[step]
    }]`,
  );
}
console.log(
  `Overall confidence: ${(predictions.confidence * 100).toFixed(1)}%`,
);
```

---

### 📊 `getModelSummary()`

```typescript
getModelSummary(): ModelSummary
```

Returns model architecture and training statistics.

```typescript
interface ModelSummary {
  totalParameters: number;
  receptiveField: number;
  spectralRadius: number;
  reservoirSize: number;
  nFeatures: number;
  nTargets: number;
  maxSequenceLength: number;
  sampleCount: number;
}
```

---

### ⚖️ `getWeights()`

```typescript
getWeights(): WeightInfo
```

Returns all model weights for inspection or custom analysis.

```typescript
interface WeightInfo {
  weights: Array<{
    name: string; // "Win", "W", "b", "Wout", "P"
    shape: number[]; // Dimensions
    values: number[]; // Flattened values
  }>;
}
```

---

### 📈 `getNormalizationStats()`

```typescript
getNormalizationStats(): NormalizationStats
```

Returns current normalization statistics.

```typescript
interface NormalizationStats {
  means: number[]; // Running means per feature
  stds: number[]; // Running standard deviations
  count: number; // Samples seen
  isActive: boolean; // Whether warmup is complete
}
```

---

### 🔄 `reset()`

```typescript
reset(): void
```

Resets model to initial state while preserving configuration.

---

### 💾 `save()` / `load()`

```typescript
save(): string
load(serialized: string): void
```

Serializes/deserializes the complete model state.

**Example:**

```typescript
// Save model
const modelState = model.save();
localStorage.setItem("myModel", modelState);

// Load model
const loadedModel = new ESNRegression();
loadedModel.load(localStorage.getItem("myModel")!);
```

---

## 💡 Examples & Use Cases

### 📈 Time Series Forecasting

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

// Configuration for daily sales forecasting
const salesModel = new ESNRegression({
  reservoirSize: 256,
  maxSequenceLength: 90, // 3 months of history
  spectralRadius: 0.95, // Long-term patterns
  leakRate: 0.3, // Smooth transitions
  rlsLambda: 0.998, // Slow forgetting
  uncertaintyMultiplier: 1.96, // 95% CI
});

// Train with historical data
const historicalSales = [
  { features: [100, 5, 1], target: [120] }, // [base_sales, promo, weekday] -> [actual]
  { features: [110, 0, 2], target: [105] },
  // ... more data
];

for (const sample of historicalSales) {
  salesModel.fitOnline({
    xCoordinates: [sample.features],
    yCoordinates: [sample.target],
  });
}

// Forecast next 7 days
const forecast = salesModel.predict(7);

console.log("📊 7-Day Sales Forecast:");
forecast.predictions.forEach((pred, day) => {
  console.log(
    `  Day ${day + 1}: ${pred[0].toFixed(0)} ` +
      `[${forecast.lowerBounds[day][0].toFixed(0)} - ${
        forecast.upperBounds[day][0].toFixed(0)
      }]`,
  );
});
```

---

### 🤖 Online Sensor Fusion

```typescript
// Real-time sensor data processing
const sensorModel = new ESNRegression({
  reservoirSize: 128,
  maxSequenceLength: 32,
  leakRate: 0.7, // Fast response
  spectralRadius: 0.8, // Short memory
  rlsLambda: 0.95, // Quick adaptation
  outlierThreshold: 2.5, // Aggressive outlier rejection
  activation: "tanh",
});

// Streaming sensor loop
async function processSensorStream(sensorStream: AsyncIterable<SensorReading>) {
  for await (const reading of sensorStream) {
    // Input: [temperature, humidity, pressure, light]
    // Output: [predicted_occupancy, energy_demand]

    const result = sensorModel.fitOnline({
      xCoordinates: [[
        reading.temp,
        reading.humidity,
        reading.pressure,
        reading.light,
      ]],
      yCoordinates: [[reading.occupancy, reading.energy]],
    });

    if (result.sampleWeight < 0.5) {
      console.warn("⚠️ Potential sensor anomaly detected!");
    }

    // Get 1-step ahead prediction for real-time control
    const prediction = sensorModel.predict(1);

    await sendToController({
      predictedOccupancy: prediction.predictions[0][0],
      predictedEnergy: prediction.predictions[0][1],
      confidence: prediction.confidence,
    });
  }
}
```

---

### 📊 Multivariate Financial Prediction

```typescript
// Multi-asset price prediction
const financeModel = new ESNRegression({
  reservoirSize: 512, // High capacity
  maxSequenceLength: 128, // ~6 months daily data
  spectralRadius: 0.99, // Long memory (markets have trends)
  leakRate: 0.2, // Smooth (noisy data)
  inputSparsity: 0.3, // Feature selection
  rlsLambda: 0.995,
  l2Lambda: 0.001, // Regularization
  rollforwardMode: "autoregressive", // True multi-step
  uncertaintyMultiplier: 2.58, // 99% CI for risk management
});

// Input: [asset1_return, asset2_return, volatility_index, interest_rate]
// Output: [asset1_next, asset2_next] (same features for autoregressive)

const trainingData = prepareFinancialData();

// Batch training
financeModel.fitOnline({
  xCoordinates: trainingData.x,
  yCoordinates: trainingData.y,
});

// 5-day forecast
const forecast = financeModel.predict(5);

console.log("📈 5-Day Multi-Asset Forecast:");
console.log(`Confidence: ${(forecast.confidence * 100).toFixed(1)}%`);
forecast.predictions.forEach((pred, day) => {
  console.log(
    `  Day ${day + 1}: Asset1=${pred[0].toFixed(4)}, Asset2=${
      pred[1].toFixed(4)
    }`,
  );
});
```

---

### 🔄 Model Persistence & Deployment

```typescript
// Training phase
const model = new ESNRegression({ reservoirSize: 256 });

// ... train model ...

// Save for deployment
const modelState = model.save();
await Deno.writeTextFile("model.json", modelState);

// -----------------------------------

// Deployment / Loading
const deployedModel = new ESNRegression();
const savedState = await Deno.readTextFile("model.json");
deployedModel.load(savedState);

// Continue training (transfer learning)
deployedModel.fitOnline({
  xCoordinates: newData.x,
  yCoordinates: newData.y,
});
```

---

## 🎯 Parameter Optimization Guide

### 🗺️ Decision Flowchart

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│                    PARAMETER SELECTION GUIDE                               │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     START HERE                                      │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │    ┌─────────────────────────────────────────┐                      │   │
│  │    │     What is your data volume?           │                      │   │
│  │    └───────────────┬─────────────────────────┘                      │   │
│  │                    │                                                │   │
│  │         ┌──────────┼──────────┐                                     │   │
│  │         ▼          ▼          ▼                                     │   │
│  │      Small      Medium      Large                                   │   │
│  │    (<1000)    (1K-100K)    (>100K)                                  │   │
│  │         │          │          │                                     │   │
│  │         ▼          ▼          ▼                                     │   │
│  │    reservoirSize  reservoirSize  reservoirSize                      │   │
│  │      64-128       128-512      256-1024                             │   │
│  │    l2Lambda      l2Lambda     l2Lambda                              │   │
│  │     0.01          0.001       0.0001                                │   │
│  │                                                                     │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │    ┌─────────────────────────────────────────┐                      │   │
│  │    │     Is your data stationary?            │                      │   │
│  │    └───────────────┬─────────────────────────┘                      │   │
│  │                    │                                                │   │
│  │              ┌─────┴─────┐                                          │   │
│  │              ▼           ▼                                          │   │
│  │            Yes          No                                          │   │
│  │              │           │                                          │   │
│  │              ▼           ▼                                          │   │
│  │        rlsLambda    rlsLambda                                       │   │
│  │         0.999        0.95-0.99                                      │   │
│  │                                                                     │   │
│  │                          │                                          │   │
│  │                          ▼                                          │   │
│  │    ┌─────────────────────────────────────────┐                      │   │
│  │    │     Pattern length in your data?        │                      │   │
│  │    └───────────────┬─────────────────────────┘                      │   │
│  │                    │                                                │   │
│  │         ┌──────────┼──────────┐                                     │   │
│  │         ▼          ▼          ▼                                     │   │
│  │       Short     Medium      Long                                    │   │
│  │      (<10)     (10-50)     (>50)                                    │   │
│  │         │          │          │                                     │   │
│  │         ▼          ▼          ▼                                     │   │
│  │   spectralRadius  spectralRadius  spectralRadius                    │   │
│  │      0.5-0.7      0.8-0.9       0.95-0.99                           │   │
│  │   leakRate       leakRate      leakRate                             │   │
│  │      0.6-0.9      0.3-0.6      0.1-0.3                              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 📋 Quick Reference Presets

#### 🚀 Fast & Simple

```typescript
const quickModel = new ESNRegression({
  reservoirSize: 64,
  maxSequenceLength: 32,
  spectralRadius: 0.8,
  leakRate: 0.5,
  reservoirSparsity: 0.9,
});
```

#### ⚖️ Balanced (Default-like)

```typescript
const balancedModel = new ESNRegression({
  reservoirSize: 256,
  maxSequenceLength: 64,
  spectralRadius: 0.9,
  leakRate: 0.3,
  rlsLambda: 0.999,
});
```

#### 🎯 High Accuracy

```typescript
const accurateModel = new ESNRegression({
  reservoirSize: 512,
  maxSequenceLength: 128,
  spectralRadius: 0.95,
  leakRate: 0.2,
  rlsLambda: 0.9995,
  l2Lambda: 0.0001,
});
```

#### 🔄 Adaptive (Non-Stationary)

```typescript
const adaptiveModel = new ESNRegression({
  reservoirSize: 256,
  maxSequenceLength: 64,
  spectralRadius: 0.85,
  leakRate: 0.5,
  rlsLambda: 0.97,
  outlierThreshold: 2.5,
});
```

#### 🛡️ Robust (Noisy Data)

```typescript
const robustModel = new ESNRegression({
  reservoirSize: 256,
  maxSequenceLength: 64,
  spectralRadius: 0.9,
  leakRate: 0.2, // More smoothing
  outlierThreshold: 2.0, // Stricter
  outlierMinWeight: 0.05,
  l2Lambda: 0.01, // Strong regularization
});
```

---

## 📊 Performance Tips

### ⚡ Speed Optimization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    PERFORMANCE OPTIMIZATION                                 │
│                                                                             │
│  1. RESERVOIR SIZE - Primary cost factor                                    │
│     ─────────────────────────────────────                                   │
│     Memory:  O(N²)     Computation: O(N² + N×F)                             │
│                                                                             │
│     Tip: Start small (64-128), increase only if needed                      │
│                                                                             │
│  2. SPARSITY - Reduce effective computations                                │
│     ────────────────────────────────────────                                │
│     reservoirSparsity: 0.9  →  10% of weights active                        │
│     inputSparsity: 0.5      →  50% of inputs connected                      │
│                                                                             │
│  3. BATCH SIZE - Amortize overhead                                          │
│     ──────────────────────────────────                                      │
│     Single samples: Higher overhead                                         │
│     Batches of 10-100: Better throughput                                    │
│                                                                             │
│  4. PRE-ALLOCATION - Arena already handles this ✅                          │
│     ───────────────────────────────────────────                             │
│     No GC pressure from model internals                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🎯 Accuracy Tips

1. **Feature Engineering**: Normalize inputs before feeding to model
2. **Proper Warmup**: Allow `normalizationWarmup` samples before expecting good
   predictions
3. **Hyperparameter Tuning**: Use validation set to tune `spectralRadius`,
   `leakRate`
4. **Ensemble Methods**: Create multiple models with different seeds, average
   predictions

```typescript
// Simple ensemble
const ensemble = [
  new ESNRegression({ seed: 1, reservoirSize: 256 }),
  new ESNRegression({ seed: 2, reservoirSize: 256 }),
  new ESNRegression({ seed: 3, reservoirSize: 256 }),
];

function ensemblePredict(models: ESNRegression[], steps: number) {
  const predictions = models.map((m) => m.predict(steps));

  // Average predictions
  return predictions[0].predictions.map((_, stepIdx) =>
    predictions[0].predictions[stepIdx].map((_, targetIdx) => {
      const sum = predictions.reduce(
        (acc, p) => acc + p.predictions[stepIdx][targetIdx],
        0,
      );
      return sum / predictions.length;
    })
  );
}
```

---

## 🧪 Testing Your Configuration

```typescript
import { ESNRegression } from "jsr:@hviana/multivariate-regression";

function evaluateConfig(
  config: Partial<ESNRegressionConfig>,
  data: { x: number[][]; y: number[][] },
) {
  const model = new ESNRegression(config);

  // Split data
  const trainSize = Math.floor(data.x.length * 0.8);
  const trainX = data.x.slice(0, trainSize);
  const trainY = data.y.slice(0, trainSize);
  const testX = data.x.slice(trainSize);
  const testY = data.y.slice(trainSize);

  // Train
  model.fitOnline({ xCoordinates: trainX, yCoordinates: trainY });

  // Evaluate
  let mse = 0;
  for (let i = 0; i < testX.length; i++) {
    model.fitOnline({ xCoordinates: [testX[i]], yCoordinates: [testY[i]] });
    const pred = model.predict(1);

    for (let t = 0; t < testY[i].length; t++) {
      mse += Math.pow(pred.predictions[0][t] - testY[i][t], 2);
    }
  }

  mse /= testX.length * testY[0].length;

  return {
    mse,
    rmse: Math.sqrt(mse),
    summary: model.getModelSummary(),
  };
}

// Test different configurations
const configs = [
  { name: "Small", config: { reservoirSize: 64 } },
  { name: "Medium", config: { reservoirSize: 256 } },
  { name: "Large", config: { reservoirSize: 512 } },
];

for (const { name, config } of configs) {
  const result = evaluateConfig(config, myData);
  console.log(`${name}: RMSE = ${result.rmse.toFixed(4)}`);
}
```

---

## 📚 Additional Resources

### 📖 Learn More About ESNs

- [Scholarpedia: Echo State Network](http://www.scholarpedia.org/article/Echo_state_network)
- [A Practical Guide to ESNs](http://www.faculty.jacobs-university.de/hjaeger/pubs/ESNTutorialRev.pdf)

### 🔗 Related Projects

- [JSR Package](https://jsr.io/@hviana/multivariate-regression)
- [GitHub Repository](https://github.com/hviana/multivariate-regression)

---

## 📜 License

**MIT License** © 2025 Henrique Emanoel Viana

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**Made with ❤️ by [Henrique Emanoel Viana](https://github.com/hviana)**

⭐ Star this repo if you find it useful!

</div>
