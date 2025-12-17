/***********************
 * ESNRegression.ts
 * Self-contained TypeScript ESN / Reservoir Computing library for online multivariate regression.
 ***********************/

export interface FitResult {
  samplesProcessed: number;
  averageLoss: number;
  gradientNorm: number;
  driftDetected: boolean;
  sampleWeight: number;
}

export interface PredictionResult {
  predictions: number[][];
  lowerBounds: number[][];
  upperBounds: number[][];
  confidence: number;
}

export interface ModelSummary {
  totalParameters: number;
  receptiveField: number;
  spectralRadius: number;
  reservoirSize: number;
  nFeatures: number;
  nTargets: number;
  maxSequenceLength: number;
  sampleCount: number;
}

export interface WeightInfo {
  weights: Array<{ name: string; shape: number[]; values: number[] }>;
}

export interface NormalizationStats {
  means: number[];
  stds: number[];
  count: number;
  isActive: boolean;
}

export interface ESNRegressionConfig {
  maxSequenceLength: number; // Default: 64
  reservoirSize: number; // Default: 256
  spectralRadius: number; // Default: 0.9
  leakRate: number; // Default: 0.3
  inputScale: number; // Default: 1.0
  biasScale: number; // Default: 0.1
  reservoirSparsity: number; // Default: 0.9
  inputSparsity: number; // Default: 0.0
  activation: "tanh" | "relu"; // Default: "tanh"
  useInputInReadout: boolean; // Default: true
  useBiasInReadout: boolean; // Default: true
  readoutTraining: "rls"; // Default: "rls"
  rlsLambda: number; // Default: 0.999
  rlsDelta: number; // Default: 1.0
  epsilon: number; // Default: 1e-8
  l2Lambda: number; // Default: 0.0001
  gradientClipNorm: number; // Default: 1.0
  normalizationEpsilon: number; // Default: 1e-8
  normalizationWarmup: number; // Default: 10
  outlierThreshold: number; // Default: 3.0
  outlierMinWeight: number; // Default: 0.1
  residualWindowSize: number; // Default: 100
  uncertaintyMultiplier: number; // Default: 1.96
  weightInitScale: number; // Default: 0.1
  seed: number; // Default: 42
  verbose: boolean; // Default: false
  rollforwardMode: "holdLastX" | "autoregressive"; // Default: "holdLastX"
}

function defaultConfig(): ESNRegressionConfig {
  return {
    maxSequenceLength: 64,
    reservoirSize: 256,
    spectralRadius: 0.9,
    leakRate: 0.3,
    inputScale: 1.0,
    biasScale: 0.1,
    reservoirSparsity: 0.9,
    inputSparsity: 0.0,
    activation: "tanh",
    useInputInReadout: true,
    useBiasInReadout: true,
    readoutTraining: "rls",
    rlsLambda: 0.999,
    rlsDelta: 1.0,
    epsilon: 1e-8,
    l2Lambda: 0.0001,
    gradientClipNorm: 1.0,
    normalizationEpsilon: 1e-8,
    normalizationWarmup: 10,
    outlierThreshold: 3.0,
    outlierMinWeight: 0.1,
    residualWindowSize: 100,
    uncertaintyMultiplier: 1.96,
    weightInitScale: 0.1,
    seed: 42,
    verbose: false,
    rollforwardMode: "holdLastX",
  };
}

/** Fixed tensor shape metadata. */
class TensorShape {
  readonly dims: number[];
  readonly size: number;
  constructor(dims: number[]) {
    this.dims = dims.slice(0);
    let s = 1;
    for (let i = 0; i < dims.length; i++) s *= dims[i] | 0;
    this.size = s | 0;
  }
}

/** Lightweight view into a Float64Array slab. */
class TensorView {
  readonly data: Float64Array;
  readonly offset: number;
  readonly shape: TensorShape;
  readonly strides: number[];
  constructor(
    data: Float64Array,
    offset: number,
    shape: TensorShape,
    strides?: number[],
  ) {
    this.data = data;
    this.offset = offset | 0;
    this.shape = shape;
    if (strides) {
      this.strides = strides.slice(0);
    } else {
      const d = shape.dims;
      const s = new Array(d.length);
      let acc = 1;
      for (let i = d.length - 1; i >= 0; i--) {
        s[i] = acc;
        acc *= d[i] | 0;
      }
      this.strides = s;
    }
  }
}

/** Simple fixed-size buffer pool (not used in hot paths; provided for completeness). */
class BufferPool {
  private buffers: Float64Array[];
  private sizes: number[];
  private count: number;
  constructor(capacity: number) {
    this.buffers = new Array(capacity);
    this.sizes = new Array(capacity);
    this.count = 0;
  }
  acquire(size: number): Float64Array {
    size = size | 0;
    for (let i = 0; i < this.count; i++) {
      if (this.sizes[i] === size) {
        const b = this.buffers[i]!;
        // remove i
        this.count--;
        this.buffers[i] = this.buffers[this.count];
        this.sizes[i] = this.sizes[this.count];
        this.buffers[this.count] = undefined as any;
        this.sizes[this.count] = 0;
        return b;
      }
    }
    return new Float64Array(size);
  }
  release(buf: Float64Array): void {
    if (this.count >= this.buffers.length) return;
    this.buffers[this.count] = buf;
    this.sizes[this.count] = buf.length;
    this.count++;
  }
}

/** Deterministic bump allocator over a single slab. */
class TensorArena {
  readonly slab: Float64Array;
  private cursor: number;
  constructor(size: number) {
    this.slab = new Float64Array(size | 0);
    this.cursor = 0;
  }
  reset(): void {
    this.cursor = 0;
  }
  alloc(size: number): Float64Array {
    size = size | 0;
    const start = this.cursor;
    const end = start + size;
    if (end > this.slab.length) throw new Error("TensorArena: out of memory");
    this.cursor = end;
    return this.slab.subarray(start, end);
  }
}

/** Small numeric ops; allocation-free. */
class TensorOps {
  static fill(a: Float64Array, v: number): void {
    for (let i = 0; i < a.length; i++) a[i] = v;
  }
  static copy(dst: Float64Array, src: Float64Array): void {
    const n = dst.length < src.length ? dst.length : src.length;
    for (let i = 0; i < n; i++) dst[i] = src[i];
  }
  static dot(a: Float64Array, b: Float64Array): number {
    const n = a.length < b.length ? a.length : b.length;
    let s = 0.0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }
  static l2Norm(a: Float64Array): number {
    let s = 0.0;
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      s += v * v;
    }
    return Math.sqrt(s);
  }
  static clamp(x: number, lo: number, hi: number): number {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }
  static isFinite(x: number): boolean {
    return Number.isFinite(x);
  }
}

/** Activation functions. */
class ActivationOps {
  static applyInPlace(a: Float64Array, kind: "tanh" | "relu"): void {
    if (kind === "tanh") {
      for (let i = 0; i < a.length; i++) a[i] = Math.tanh(a[i]);
    } else {
      for (let i = 0; i < a.length; i++) a[i] = a[i] > 0 ? a[i] : 0;
    }
  }
  static applyScalar(x: number, kind: "tanh" | "relu"): number {
    if (kind === "tanh") return Math.tanh(x);
    return x > 0 ? x : 0;
  }
}

/** Deterministic PRNG (xorshift32). */
class RandomGenerator {
  private state: number;
  constructor(seed: number) {
    this.state = (seed | 0) >>> 0;
    if (this.state === 0) this.state = 0x9e3779b9;
  }
  nextU32(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }
  nextFloat(): number {
    // [0,1)
    return (this.nextU32() >>> 0) / 4294967296.0;
  }
  nextSigned(): number {
    // (-1,1)
    return this.nextFloat() * 2.0 - 1.0;
  }
}

/** Welford accumulator for mean/variance. */
class WelfordAccumulator {
  readonly mean: Float64Array;
  readonly m2: Float64Array;
  count: number;
  constructor(dim: number) {
    this.mean = new Float64Array(dim | 0);
    this.m2 = new Float64Array(dim | 0);
    this.count = 0;
  }
  reset(): void {
    this.count = 0;
    TensorOps.fill(this.mean, 0.0);
    TensorOps.fill(this.m2, 0.0);
  }
  /** Update with a vector x (no allocations). */
  update(x: Float64Array): void {
    const n = this.mean.length;
    this.count++;
    const c = this.count;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      const delta = xi - this.mean[i];
      this.mean[i] += delta / c;
      const delta2 = xi - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }
  /** Population variance; deterministic with epsilon floor applied by caller. */
  varianceAt(i: number): number {
    if (this.count < 2) return 0.0;
    return this.m2[i] / (this.count - 1);
  }
}

/** Online z-score normalizer (per feature). */
class WelfordNormalizer {
  readonly acc: WelfordAccumulator;
  readonly std: Float64Array;
  readonly eps: number;
  readonly warmup: number;
  constructor(dim: number, eps: number, warmup: number) {
    this.acc = new WelfordAccumulator(dim | 0);
    this.std = new Float64Array(dim | 0);
    this.eps = eps;
    this.warmup = warmup | 0;
  }
  reset(): void {
    this.acc.reset();
    TensorOps.fill(this.std, 0.0);
  }
  /** Update running stats from raw x, then compute std vector. */
  updateStats(xRaw: Float64Array): void {
    this.acc.update(xRaw);
    const n = this.std.length;
    for (let i = 0; i < n; i++) {
      const v = this.acc.varianceAt(i);
      const s = Math.sqrt(v > 0 ? v : 0.0);
      this.std[i] = s;
    }
  }
  /** Normalize raw x into dst (dst length == dim). */
  normalize(xRaw: Float64Array, dst: Float64Array): void {
    const n = dst.length;
    const mean = this.acc.mean;
    const std = this.std;
    const eps = this.eps;
    for (let i = 0; i < n; i++) {
      const denom = std[i] > eps ? std[i] : eps;
      dst[i] = (xRaw[i] - mean[i]) / denom;
    }
  }
  getStats(): NormalizationStats {
    const n = this.acc.mean.length;
    const means = new Array<number>(n);
    const stds = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      means[i] = this.acc.mean[i];
      stds[i] = this.std[i] > this.eps ? this.std[i] : this.eps;
    }
    return {
      means,
      stds,
      count: this.acc.count,
      isActive: this.acc.count >= this.warmup,
    };
  }
}

/** Fixed-capacity ring buffer for time series rows. */
class RingBuffer {
  private readonly capacity: number;
  private readonly dim: number;
  private data: Float64Array;
  private head: number;
  private count: number;
  constructor(capacity: number, dim: number) {
    this.capacity = capacity | 0;
    this.dim = dim | 0;
    this.data = new Float64Array((this.capacity * this.dim) | 0);
    this.head = 0;
    this.count = 0;
  }
  reset(): void {
    this.head = 0;
    this.count = 0;
    // data not cleared for performance; caller should treat count as authoritative.
  }
  size(): number {
    return this.count;
  }
  /** Push a raw row; copies values (no allocations). */
  pushRow(row: number[]): void {
    if (row.length !== this.dim) {
      throw new Error("RingBuffer.pushRow: row dimension mismatch");
    }
    const idx = this.head;
    const base = idx * this.dim;
    for (let j = 0; j < this.dim; j++) this.data[base + j] = row[j];
    this.head++;
    if (this.head >= this.capacity) this.head = 0;
    if (this.count < this.capacity) this.count++;
  }
  /** Copy latest row into dst (dst length == dim). */
  getLatestRow(dst: Float64Array): void {
    if (this.count <= 0) throw new Error("RingBuffer.getLatestRow: empty");
    let idx = this.head - 1;
    if (idx < 0) idx += this.capacity;
    const base = idx * this.dim;
    for (let j = 0; j < this.dim; j++) dst[j] = this.data[base + j];
  }
  /** Serialize full buffer state (allocations allowed). */
  toJSON(): {
    capacity: number;
    dim: number;
    head: number;
    count: number;
    data: number[];
  } {
    return {
      capacity: this.capacity,
      dim: this.dim,
      head: this.head,
      count: this.count,
      data: Array.from(this.data),
    };
  }
  /** Restore from JSON. */
  static fromJSON(obj: any): RingBuffer {
    const rb = new RingBuffer(obj.capacity | 0, obj.dim | 0);
    rb.head = obj.head | 0;
    rb.count = obj.count | 0;
    const arr = obj.data as number[];
    const n = rb.data.length;
    for (let i = 0; i < n; i++) rb.data[i] = arr[i];
    return rb;
  }
}

/** Tracks recent residual distribution per target using fixed window and O(1) updates. */
class ResidualStatsTracker {
  readonly windowSize: number;
  readonly nTargets: number;
  private buffers: Float64Array; // [nTargets * windowSize]
  private writeIndex: number;
  private count: number;
  private sum: Float64Array; // [nTargets]
  private sumsq: Float64Array; // [nTargets]
  private eps: number;

  constructor(nTargets: number, windowSize: number, eps: number) {
    this.nTargets = nTargets | 0;
    this.windowSize = windowSize | 0;
    this.eps = eps;
    this.buffers = new Float64Array((this.nTargets * this.windowSize) | 0);
    this.writeIndex = 0;
    this.count = 0;
    this.sum = new Float64Array(this.nTargets);
    this.sumsq = new Float64Array(this.nTargets);
  }

  reset(): void {
    this.writeIndex = 0;
    this.count = 0;
    TensorOps.fill(this.buffers, 0.0);
    TensorOps.fill(this.sum, 0.0);
    TensorOps.fill(this.sumsq, 0.0);
  }

  /** Update with residual vector r (y - yhat). */
  update(residuals: Float64Array): void {
    const nT = this.nTargets;
    const w = this.windowSize;
    const idx = this.writeIndex;

    if (this.count >= w) {
      // Remove old
      const baseOld = idx * nT;
      for (let t = 0; t < nT; t++) {
        const old = this.buffers[baseOld + t];
        this.sum[t] -= old;
        this.sumsq[t] -= old * old;
      }
    } else {
      this.count++;
    }

    // Add new
    const base = idx * nT;
    for (let t = 0; t < nT; t++) {
      const r = residuals[t];
      this.buffers[base + t] = r;
      this.sum[t] += r;
      this.sumsq[t] += r * r;
    }

    this.writeIndex++;
    if (this.writeIndex >= w) this.writeIndex = 0;
  }

  /** Standard deviation estimate per target; writes to dst. */
  getSigma(dst: Float64Array): void {
    const nT = this.nTargets;
    const c = this.count;
    const eps = this.eps;
    if (c <= 1) {
      for (let t = 0; t < nT; t++) dst[t] = eps;
      return;
    }
    for (let t = 0; t < nT; t++) {
      const mean = this.sum[t] / c;
      const var_ = this.sumsq[t] / c - mean * mean; // population variance over window
      const v = var_ > 0 ? var_ : 0.0;
      const s = Math.sqrt(v);
      dst[t] = s > eps ? s : eps;
    }
  }

  toJSON(): any {
    return {
      windowSize: this.windowSize,
      nTargets: this.nTargets,
      buffers: Array.from(this.buffers),
      writeIndex: this.writeIndex,
      count: this.count,
      sum: Array.from(this.sum),
      sumsq: Array.from(this.sumsq),
      eps: this.eps,
    };
  }
  static fromJSON(obj: any): ResidualStatsTracker {
    const tr = new ResidualStatsTracker(
      obj.nTargets | 0,
      obj.windowSize | 0,
      obj.eps,
    );
    const b = obj.buffers as number[];
    for (let i = 0; i < tr.buffers.length; i++) tr.buffers[i] = b[i];
    tr.writeIndex = obj.writeIndex | 0;
    tr.count = obj.count | 0;
    const s = obj.sum as number[];
    const ss = obj.sumsq as number[];
    for (let i = 0; i < tr.sum.length; i++) tr.sum[i] = s[i];
    for (let i = 0; i < tr.sumsq.length; i++) tr.sumsq[i] = ss[i];
    return tr;
  }
}

/** Downweights outliers based on residual z-score. */
class OutlierDownweighter {
  private threshold: number;
  private minWeight: number;
  constructor(threshold: number, minWeight: number) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }
  /** Returns sampleWeight in [minWeight, 1]. */
  computeWeight(residuals: Float64Array, sigma: Float64Array): number {
    let maxZ = 0.0;
    for (let i = 0; i < residuals.length; i++) {
      const s = sigma[i] > 0 ? sigma[i] : 1.0;
      const z = Math.abs(residuals[i]) / s;
      if (z > maxZ) maxZ = z;
    }
    if (!TensorOps.isFinite(maxZ)) return this.minWeight;
    if (maxZ <= this.threshold) return 1.0;
    const w = this.threshold / maxZ;
    return w > this.minWeight ? w : this.minWeight;
  }
}

/** Loss function utilities. */
class LossFunction {
  /** Mean squared error over targets. */
  static mse(y: number[], yhat: Float64Array): number {
    const n = y.length;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const d = y[i] - yhat[i];
      s += d * d;
    }
    return s / (n > 0 ? n : 1);
  }
}

/** Tracks running metrics (loss). */
class MetricsAccumulator {
  private sumLoss: number;
  private count: number;
  constructor() {
    this.sumLoss = 0.0;
    this.count = 0;
  }
  reset(): void {
    this.sumLoss = 0.0;
    this.count = 0;
  }
  add(loss: number): void {
    if (!TensorOps.isFinite(loss)) return;
    this.sumLoss += loss;
    this.count++;
  }
  mean(): number {
    if (this.count <= 0) return 0.0;
    return this.sumLoss / this.count;
  }
}

/** Masks for sparse initialization. */
class ReservoirInitMask {
  static initMask(
    rng: RandomGenerator,
    size: number,
    sparsity: number,
    dst: Uint8Array,
  ): void {
    // dst[i]=1 if keep, 0 if zero
    const pKeep = 1.0 - TensorOps.clamp(sparsity, 0.0, 1.0);
    for (let i = 0; i < size; i++) dst[i] = rng.nextFloat() < pKeep ? 1 : 0;
  }
}

/** Spectral radius estimation and scaling via power method. */
class SpectralRadiusScaler {
  static estimateSpectralRadius(
    W: Float64Array,
    n: number,
    iters: number,
    tmpV: Float64Array,
    tmpWv: Float64Array,
  ): number {
    // power method: v <- W v / ||W v|| ; return ||W v|| when v normalized
    for (let i = 0; i < n; i++) tmpV[i] = 1.0 / Math.sqrt(n); // deterministic init
    for (let it = 0; it < iters; it++) {
      // tmpWv = W * v
      for (let i = 0; i < n; i++) {
        let s = 0.0;
        const row = i * n;
        for (let j = 0; j < n; j++) s += W[row + j] * tmpV[j];
        tmpWv[i] = s;
      }
      let norm = 0.0;
      for (let i = 0; i < n; i++) {
        const v = tmpWv[i];
        norm += v * v;
      }
      norm = Math.sqrt(norm);
      if (!TensorOps.isFinite(norm) || norm <= 0) return 0.0;
      const inv = 1.0 / norm;
      for (let i = 0; i < n; i++) tmpV[i] = tmpWv[i] * inv;
    }
    // one final multiply to get magnitude
    for (let i = 0; i < n; i++) {
      let s = 0.0;
      const row = i * n;
      for (let j = 0; j < n; j++) s += W[row + j] * tmpV[j];
      tmpWv[i] = s;
    }
    let norm = 0.0;
    for (let i = 0; i < n; i++) {
      const v = tmpWv[i];
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (!TensorOps.isFinite(norm)) return 0.0;
    return norm;
  }

  static scaleToSpectralRadius(
    W: Float64Array,
    n: number,
    target: number,
    iters: number,
    tmpV: Float64Array,
    tmpWv: Float64Array,
    eps: number,
  ): number {
    const est = this.estimateSpectralRadius(W, n, iters, tmpV, tmpWv);
    const denom = est > eps ? est : eps;
    const scale = target / denom;
    for (let i = 0; i < W.length; i++) W[i] *= scale;
    return est * scale; // return scaled estimate ~ target
  }
}

class ESNReservoirParams {
  readonly reservoirSize: number;
  readonly nFeatures: number;
  readonly Win: Float64Array; // [N x F]
  readonly W: Float64Array; // [N x N]
  readonly b: Float64Array; // [N]
  constructor(
    reservoirSize: number,
    nFeatures: number,
    Win: Float64Array,
    W: Float64Array,
    b: Float64Array,
  ) {
    this.reservoirSize = reservoirSize | 0;
    this.nFeatures = nFeatures | 0;
    this.Win = Win;
    this.W = W;
    this.b = b;
  }
}

class ESNReservoir {
  readonly params: ESNReservoirParams;
  private activation: "tanh" | "relu";
  private leakRate: number;
  private inputScale: number;
  /** live state r_t */
  readonly state: Float64Array;
  /** internal prev buffer for update */
  private prev: Float64Array;

  constructor(
    params: ESNReservoirParams,
    activation: "tanh" | "relu",
    leakRate: number,
    inputScale: number,
    arena: TensorArena,
  ) {
    this.params = params;
    this.activation = activation;
    this.leakRate = leakRate;
    this.inputScale = inputScale;
    this.state = arena.alloc(params.reservoirSize);
    this.prev = arena.alloc(params.reservoirSize);
    TensorOps.fill(this.state, 0.0);
    TensorOps.fill(this.prev, 0.0);
  }

  reset(): void {
    TensorOps.fill(this.state, 0.0);
    TensorOps.fill(this.prev, 0.0);
  }

  /**
   * Update given normalized x (length F) into given state buffers.
   * Implements leaky ESN:
   * r_t = (1-α) r_{t-1} + α f( Win*(inputScale*x) + W*r_{t-1} + b )
   */
  updateStateInPlace(
    state: Float64Array,
    prev: Float64Array,
    xNorm: Float64Array,
  ): void {
    const N = this.params.reservoirSize;
    const F = this.params.nFeatures;
    const Win = this.params.Win;
    const W = this.params.W;
    const b = this.params.b;
    const leak = this.leakRate;
    const oneMinus = 1.0 - leak;
    const inScale = this.inputScale;

    // prev = state
    for (let i = 0; i < N; i++) prev[i] = state[i];

    for (let i = 0; i < N; i++) {
      let s = b[i];
      const winRow = i * F;
      for (let j = 0; j < F; j++) s += Win[winRow + j] * (xNorm[j] * inScale);
      const wRow = i * N;
      for (let j = 0; j < N; j++) s += W[wRow + j] * prev[j];
      const a = ActivationOps.applyScalar(s, this.activation);
      state[i] = oneMinus * prev[i] + leak * a;
    }
  }

  /** Update live reservoir state with normalized x (no allocations). */
  step(xNorm: Float64Array): void {
    this.updateStateInPlace(this.state, this.prev, xNorm);
  }
}

/** Readout config derived from ESNRegressionConfig. */
class ReadoutConfig {
  readonly useInputInReadout: boolean;
  readonly useBiasInReadout: boolean;
  readonly l2Lambda: number;
  readonly gradientClipNorm: number;
  constructor(cfg: ESNRegressionConfig) {
    this.useInputInReadout = cfg.useInputInReadout;
    this.useBiasInReadout = cfg.useBiasInReadout;
    this.l2Lambda = cfg.l2Lambda;
    this.gradientClipNorm = cfg.gradientClipNorm;
  }
}

class ReadoutParams {
  readonly nTargets: number;
  readonly zDim: number;
  readonly Wout: Float64Array; // [T x zDim]
  constructor(nTargets: number, zDim: number, Wout: Float64Array) {
    this.nTargets = nTargets | 0;
    this.zDim = zDim | 0;
    this.Wout = Wout;
  }
}

class LinearReadout {
  readonly params: ReadoutParams;
  constructor(params: ReadoutParams) {
    this.params = params;
  }
  /** y = Wout * z ; writes into yOut (len T). */
  forward(z: Float64Array, yOut: Float64Array): void {
    const T = this.params.nTargets;
    const D = this.params.zDim;
    const W = this.params.Wout;
    for (let t = 0; t < T; t++) {
      const base = t * D;
      let s = 0.0;
      for (let i = 0; i < D; i++) s += W[base + i] * z[i];
      yOut[t] = s;
    }
  }
  /** Apply weight decay (L2) deterministically: Wout *= (1 - l2Lambda). */
  applyL2WeightDecay(l2Lambda: number): void {
    if (l2Lambda <= 0) return;
    let factor = 1.0 - l2Lambda;
    if (factor < 0) factor = 0;
    const W = this.params.Wout;
    for (let i = 0; i < W.length; i++) W[i] *= factor;
  }
}

class RLSState {
  readonly zDim: number;
  readonly P: Float64Array; // [zDim x zDim]
  readonly Pz: Float64Array; // [zDim]
  readonly k: Float64Array; // [zDim]
  constructor(
    zDim: number,
    P: Float64Array,
    Pz: Float64Array,
    k: Float64Array,
  ) {
    this.zDim = zDim | 0;
    this.P = P;
    this.Pz = Pz;
    this.k = k;
  }
  reset(delta: number): void {
    const D = this.zDim;
    const P = this.P;
    TensorOps.fill(P, 0.0);
    const v = 1.0 / (delta > 0 ? delta : 1.0);
    for (let i = 0; i < D; i++) P[i * D + i] = v;
    TensorOps.fill(this.Pz, 0.0);
    TensorOps.fill(this.k, 0.0);
  }
}

class RLSOptimizer {
  private lambda: number;
  private eps: number;
  constructor(lambda: number, eps: number) {
    this.lambda = lambda;
    this.eps = eps;
  }

  /**
   * RLS update with shared gain k and shared P:
   * Pz = P z
   * denom = lambda + z^T P z
   * k = Pz / denom
   * Wout_row += k * e_row
   * P = (P - k * (Pz^T)) / lambda
   *
   * Supports optional sample weighting by providing zWeighted and errWeighted (already scaled by sqrt(w)).
   * Also supports deterministic gradient clipping by scaling errWeighted.
   *
   * @returns updateNorm (approx L2 norm of deltaW)
   */
  stepSharedP(
    state: RLSState,
    Wout: Float64Array,
    z: Float64Array,
    err: Float64Array, // [T] error in observation space corresponding to z (already weighted if needed)
    nTargets: number,
    gradientClipNorm: number,
  ): number {
    const D = state.zDim;
    const P = state.P;
    const Pz = state.Pz;
    const k = state.k;
    const lambda = this.lambda;
    const eps = this.eps;

    // Pz = P * z
    for (let i = 0; i < D; i++) {
      let s = 0.0;
      const row = i * D;
      for (let j = 0; j < D; j++) s += P[row + j] * z[j];
      Pz[i] = s;
    }

    // denom = lambda + z^T * Pz
    let zTPz = 0.0;
    for (let i = 0; i < D; i++) zTPz += z[i] * Pz[i];
    let denom = lambda + zTPz;
    if (!TensorOps.isFinite(denom) || Math.abs(denom) < eps) {
      denom = denom >= 0 ? eps : -eps;
    }

    const invDen = 1.0 / denom;

    // k = Pz / denom
    for (let i = 0; i < D; i++) k[i] = Pz[i] * invDen;

    // Compute approximate update norm for clipping:
    // deltaW_t = k * err[t] ; ||deltaW|| = ||k|| * sqrt(sum_t err[t]^2)
    let sumErr2 = 0.0;
    for (let t = 0; t < nTargets; t++) {
      const e = err[t];
      sumErr2 += e * e;
    }
    const kNorm = TensorOps.l2Norm(k);
    let updateNorm = kNorm * Math.sqrt(sumErr2);

    if (!TensorOps.isFinite(updateNorm)) {
      updateNorm = gradientClipNorm > 0 ? gradientClipNorm : 0.0;
    }

    if (
      gradientClipNorm > 0 && updateNorm > gradientClipNorm && updateNorm > eps
    ) {
      const scale = gradientClipNorm / updateNorm;
      for (let t = 0; t < nTargets; t++) err[t] *= scale;
      updateNorm = gradientClipNorm;
    }

    // Update Wout: Wout[t, i] += k[i] * err[t]
    for (let t = 0; t < nTargets; t++) {
      const base = t * D;
      const et = err[t];
      for (let i = 0; i < D; i++) Wout[base + i] += k[i] * et;
    }

    // Update P: P = (P - k * (Pz^T)) / lambda
    const invLambda = 1.0 / (lambda > eps ? lambda : eps);
    for (let i = 0; i < D; i++) {
      const ki = k[i];
      const row = i * D;
      for (let j = 0; j < D; j++) {
        P[row + j] = (P[row + j] - ki * Pz[j]) * invLambda;
      }
    }

    return updateNorm;
  }
}

/** Serialization helper. */
class SerializationHelper {
  static toNumberArray(a: Float64Array): number[] {
    return Array.from(a);
  }
  static fromNumberArray(dst: Float64Array, src: number[]): void {
    const n = dst.length;
    for (let i = 0; i < n; i++) dst[i] = src[i];
  }
  static toNumberArrayU8(a: Uint8Array): number[] {
    const out = new Array<number>(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i];
    return out;
  }
}

/** Main model. */
export class ESNRegression {
  readonly config: ESNRegressionConfig;

  private initialized: boolean;
  private nFeatures: number;
  private nTargets: number;
  private zDim: number;

  private rng: RandomGenerator;

  // Memory arena
  private arena: TensorArena | null;

  // Components
  private ring: RingBuffer | null;
  private normalizer: WelfordNormalizer | null;
  private residualTracker: ResidualStatsTracker | null;
  private outlier: OutlierDownweighter | null;

  private reservoirParams: ESNReservoirParams | null;
  private reservoir: ESNReservoir | null;

  private readoutCfg: ReadoutConfig | null;
  private readoutParams: ReadoutParams | null;
  private readout: LinearReadout | null;

  private rlsState: RLSState | null;
  private rlsOpt: RLSOptimizer | null;

  // Scratch buffers (training)
  private xRawScratch: Float64Array | null; // [F]
  private xNormScratch: Float64Array | null; // [F]
  private zScratch: Float64Array | null; // [D]
  private zWeightedScratch: Float64Array | null; // [D]
  private yHatScratch: Float64Array | null; // [T]
  private residualScratch: Float64Array | null; // [T]
  private sigmaScratch: Float64Array | null; // [T]
  private errWeightedScratch: Float64Array | null; // [T]

  // Scratch buffers (prediction roll-forward)
  private rScratch: Float64Array | null; // [N]
  private rPrevScratch: Float64Array | null; // [N]
  private xStepRawScratch: Float64Array | null; // [F]
  private xStepNormScratch: Float64Array | null; // [F]
  private zPredScratch: Float64Array | null; // [D]
  private yPredScratch: Float64Array | null; // [T]
  private sigmaPredScratch: Float64Array | null; // [T]

  // Results reused
  private fitRes: FitResult;
  private predRes: PredictionResult | null;

  // Metrics
  private metrics: MetricsAccumulator;
  private sampleCount: number;
  private scaledSpectralRadius: number;

  constructor(cfg?: Partial<ESNRegressionConfig>) {
    const base = defaultConfig();
    this.config = Object.assign(base, cfg || {});
    // Clamp some values for stability
    if (this.config.maxSequenceLength <= 0) this.config.maxSequenceLength = 1;
    if (this.config.reservoirSize <= 0) this.config.reservoirSize = 1;
    if (this.config.leakRate <= 0) this.config.leakRate = 0.01;
    if (this.config.leakRate > 1) this.config.leakRate = 1;
    this.config.rlsLambda = TensorOps.clamp(this.config.rlsLambda, 0.9, 1.0);

    this.initialized = false;
    this.nFeatures = 0;
    this.nTargets = 0;
    this.zDim = 0;

    this.rng = new RandomGenerator(this.config.seed);

    this.arena = null;
    this.ring = null;
    this.normalizer = null;
    this.residualTracker = null;
    this.outlier = null;

    this.reservoirParams = null;
    this.reservoir = null;

    this.readoutCfg = null;
    this.readoutParams = null;
    this.readout = null;

    this.rlsState = null;
    this.rlsOpt = null;

    this.xRawScratch = null;
    this.xNormScratch = null;
    this.zScratch = null;
    this.zWeightedScratch = null;
    this.yHatScratch = null;
    this.residualScratch = null;
    this.sigmaScratch = null;
    this.errWeightedScratch = null;

    this.rScratch = null;
    this.rPrevScratch = null;
    this.xStepRawScratch = null;
    this.xStepNormScratch = null;
    this.zPredScratch = null;
    this.yPredScratch = null;
    this.sigmaPredScratch = null;

    this.fitRes = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1.0,
    };
    this.predRes = null;

    this.metrics = new MetricsAccumulator();
    this.sampleCount = 0;
    this.scaledSpectralRadius = this.config.spectralRadius;
  }

  private ensureInitializedFromBatch(
    xCoordinates: number[][],
    yCoordinates: number[][],
  ): void {
    if (this.initialized) return;
    if (xCoordinates.length === 0) return;
    const F = xCoordinates[0].length | 0;
    const T = yCoordinates[0].length | 0;
    if (F <= 0) throw new Error("fitOnline: nFeatures must be > 0");
    if (T <= 0) throw new Error("fitOnline: nTargets must be > 0");
    // Validate all rows
    for (let i = 0; i < xCoordinates.length; i++) {
      if (xCoordinates[i].length !== F) {
        throw new Error(
          "fitOnline: inconsistent xCoordinates feature dimension",
        );
      }
      if (yCoordinates[i].length !== T) {
        throw new Error(
          "fitOnline: inconsistent yCoordinates target dimension",
        );
      }
    }

    this.nFeatures = F;
    this.nTargets = T;

    const N = this.config.reservoirSize | 0;
    const useX = this.config.useInputInReadout;
    const useB = this.config.useBiasInReadout;
    this.zDim = (N + (useX ? F : 0) + (useB ? 1 : 0)) | 0;

    // Preallocate arena: size estimate
    // Reservoir params: W (N*N), Win (N*F), b (N)
    // Reservoir state: live state (N), prev (N), scratch r (N), scratch prev (N)
    // RLS: P (D*D), Pz (D), k (D)
    // Readout: Wout (T*D)
    // Scratch: xRaw(F), xNorm(F), z(D), zW(D), yHat(T), residual(T), sigma(T), errW(T)
    // Predict scratch: xStepRaw(F), xStepNorm(F), zPred(D), yPred(T), sigmaPred(T)
    // Power method temp: v(N), wv(N)
    const D = this.zDim;
    let total = 0;
    total += N * N; // W
    total += N * F; // Win
    total += N; // b
    total += N; // live state
    total += N; // live prev
    total += N; // scratch r
    total += N; // scratch prev
    total += D * D; // P
    total += D; // Pz
    total += D; // k
    total += T * D; // Wout
    total += F; // xRawScratch
    total += F; // xNormScratch
    total += D; // zScratch
    total += D; // zWeightedScratch
    total += T; // yHatScratch
    total += T; // residualScratch
    total += T; // sigmaScratch
    total += T; // errWeightedScratch
    total += F; // xStepRawScratch
    total += F; // xStepNormScratch
    total += D; // zPredScratch
    total += T; // yPredScratch
    total += T; // sigmaPredScratch
    total += N; // power v
    total += N; // power wv
    total += 64; // small slack

    this.arena = new TensorArena(total | 0);

    // Allocate reservoir parameters
    const W = this.arena.alloc(N * N);
    const Win = this.arena.alloc(N * F);
    const b = this.arena.alloc(N);

    // Init fixed weights deterministically
    this.initReservoirWeights(W, Win, b);

    this.reservoirParams = new ESNReservoirParams(N, F, Win, W, b);
    this.reservoir = new ESNReservoir(
      this.reservoirParams,
      this.config.activation,
      this.config.leakRate,
      this.config.inputScale,
      this.arena,
    );

    // Allocate scratch for reservoir prediction roll-forward
    this.rScratch = this.arena.alloc(N);
    this.rPrevScratch = this.arena.alloc(N);
    TensorOps.fill(this.rScratch, 0.0);
    TensorOps.fill(this.rPrevScratch, 0.0);

    // Readout
    const Wout = this.arena.alloc(T * D);
    this.initReadoutWeights(Wout);
    this.readoutCfg = new ReadoutConfig(this.config);
    this.readoutParams = new ReadoutParams(T, D, Wout);
    this.readout = new LinearReadout(this.readoutParams);

    // RLS
    const P = this.arena.alloc(D * D);
    const Pz = this.arena.alloc(D);
    const k = this.arena.alloc(D);
    this.rlsState = new RLSState(D, P, Pz, k);
    this.rlsState.reset(this.config.rlsDelta);
    this.rlsOpt = new RLSOptimizer(this.config.rlsLambda, this.config.epsilon);

    // Normalizer and trackers
    this.ring = new RingBuffer(this.config.maxSequenceLength, F);
    this.normalizer = new WelfordNormalizer(
      F,
      this.config.normalizationEpsilon,
      this.config.normalizationWarmup,
    );
    this.residualTracker = new ResidualStatsTracker(
      T,
      this.config.residualWindowSize,
      this.config.epsilon,
    );
    this.outlier = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );

    // Scratch buffers
    this.xRawScratch = this.arena.alloc(F);
    this.xNormScratch = this.arena.alloc(F);
    this.zScratch = this.arena.alloc(D);
    this.zWeightedScratch = this.arena.alloc(D);
    this.yHatScratch = this.arena.alloc(T);
    this.residualScratch = this.arena.alloc(T);
    this.sigmaScratch = this.arena.alloc(T);
    this.errWeightedScratch = this.arena.alloc(T);

    this.xStepRawScratch = this.arena.alloc(F);
    this.xStepNormScratch = this.arena.alloc(F);
    this.zPredScratch = this.arena.alloc(D);
    this.yPredScratch = this.arena.alloc(T);
    this.sigmaPredScratch = this.arena.alloc(T);

    TensorOps.fill(this.xRawScratch, 0.0);
    TensorOps.fill(this.xNormScratch, 0.0);
    TensorOps.fill(this.zScratch, 0.0);
    TensorOps.fill(this.zWeightedScratch, 0.0);
    TensorOps.fill(this.yHatScratch, 0.0);
    TensorOps.fill(this.residualScratch, 0.0);
    TensorOps.fill(this.sigmaScratch, 0.0);
    TensorOps.fill(this.errWeightedScratch, 0.0);

    TensorOps.fill(this.xStepRawScratch, 0.0);
    TensorOps.fill(this.xStepNormScratch, 0.0);
    TensorOps.fill(this.zPredScratch, 0.0);
    TensorOps.fill(this.yPredScratch, 0.0);
    TensorOps.fill(this.sigmaPredScratch, 0.0);

    this.metrics.reset();
    this.sampleCount = 0;

    // Preallocate PredictionResult arrays (fixed maxSequenceLength x nTargets)
    this.predRes = this.allocatePredictionResult(
      this.config.maxSequenceLength,
      T,
    );

    this.initialized = true;
  }

  private allocatePredictionResult(
    maxSteps: number,
    nTargets: number,
  ): PredictionResult {
    const preds = new Array<number[]>(maxSteps);
    const lows = new Array<number[]>(maxSteps);
    const ups = new Array<number[]>(maxSteps);
    for (let i = 0; i < maxSteps; i++) {
      const p = new Array<number>(nTargets);
      const l = new Array<number>(nTargets);
      const u = new Array<number>(nTargets);
      for (let t = 0; t < nTargets; t++) {
        p[t] = 0.0;
        l[t] = 0.0;
        u[t] = 0.0;
      }
      preds[i] = p;
      lows[i] = l;
      ups[i] = u;
    }
    return {
      predictions: preds,
      lowerBounds: lows,
      upperBounds: ups,
      confidence: 0.0,
    };
  }

  private initReservoirWeights(
    W: Float64Array,
    Win: Float64Array,
    b: Float64Array,
  ): void {
    const cfg = this.config;
    const N = cfg.reservoirSize | 0;
    const F = this.nFeatures | 0;
    const rng = new RandomGenerator(cfg.seed);

    // Initialize Win with input sparsity
    const winMask = new Uint8Array(N * F);
    ReservoirInitMask.initMask(rng, winMask.length, cfg.inputSparsity, winMask);
    const wscale = cfg.weightInitScale;
    for (let i = 0; i < Win.length; i++) {
      Win[i] = winMask[i] ? rng.nextSigned() * wscale : 0.0;
    }

    // Initialize W with reservoir sparsity
    const wMask = new Uint8Array(N * N);
    ReservoirInitMask.initMask(rng, wMask.length, cfg.reservoirSparsity, wMask);
    for (let i = 0; i < W.length; i++) {
      W[i] = wMask[i] ? rng.nextSigned() * wscale : 0.0;
    }

    // Initialize bias b
    const bscale = cfg.biasScale;
    for (let i = 0; i < b.length; i++) b[i] = rng.nextSigned() * bscale;

    // Scale W to target spectral radius (power method)
    // Use temporary vectors from arena? Not available yet; allocate local typed arrays (not hot path, init only).
    const tmpV = new Float64Array(N);
    const tmpWv = new Float64Array(N);
    this.scaledSpectralRadius = SpectralRadiusScaler.scaleToSpectralRadius(
      W,
      N,
      cfg.spectralRadius,
      32,
      tmpV,
      tmpWv,
      cfg.epsilon,
    );
  }

  private initReadoutWeights(Wout: Float64Array): void {
    // Start at zeros for stability/determinism; user can warm up via RLS
    TensorOps.fill(Wout, 0.0);
  }

  private buildZ(
    rState: Float64Array,
    xNorm: Float64Array,
    zOut: Float64Array,
  ): void {
    const N = this.config.reservoirSize | 0;
    const F = this.nFeatures | 0;
    let idx = 0;
    // r_t
    for (let i = 0; i < N; i++) zOut[idx++] = rState[i];
    // optionally x_t
    if (this.config.useInputInReadout) {
      for (let j = 0; j < F; j++) zOut[idx++] = xNorm[j];
    }
    // optional bias
    if (this.config.useBiasInReadout) {
      zOut[idx++] = 1.0;
    }
    // idx should equal zDim
  }

  /**
   * Online training: one sample at a time, but accepts a batch of rows.
   *
   * Critical behavior enforced:
   * - xCoordinates.length must equal yCoordinates.length (else throw before ingesting anything).
   * - For each row i: push x into RingBuffer FIRST, then train using internal buffers only.
   */
  fitOnline(
    args: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const xCoordinates = args.xCoordinates;
    const yCoordinates = args.yCoordinates;

    if ((xCoordinates.length | 0) !== (yCoordinates.length | 0)) {
      throw new Error(
        "fitOnline: xCoordinates.length must equal yCoordinates.length",
      );
    }

    const N = xCoordinates.length | 0;
    // Validate before ingestion for already-initialized dimensions too
    if (N > 0) {
      if (!this.initialized) {
        // also validates all rows of first call
        this.ensureInitializedFromBatch(xCoordinates, yCoordinates);
      } else {
        const F = this.nFeatures | 0;
        const T = this.nTargets | 0;
        for (let i = 0; i < N; i++) {
          if (xCoordinates[i].length !== F) {
            throw new Error(
              "fitOnline: inconsistent xCoordinates feature dimension",
            );
          }
          if (yCoordinates[i].length !== T) {
            throw new Error(
              "fitOnline: inconsistent yCoordinates target dimension",
            );
          }
        }
      }
    } else {
      // empty batch: no-op, reuse fit result
      this.fitRes.samplesProcessed = 0;
      this.fitRes.averageLoss = this.metrics.mean();
      this.fitRes.gradientNorm = 0;
      this.fitRes.driftDetected = false;
      this.fitRes.sampleWeight = 1.0;
      return this.fitRes;
    }

    if (!this.initialized) {
      // if N==0 previously, but here N>0 we already initialized.
      throw new Error("fitOnline: failed to initialize");
    }

    const ring = this.ring!;
    const normalizer = this.normalizer!;
    const reservoir = this.reservoir!;
    const readout = this.readout!;
    const rlsState = this.rlsState!;
    const rlsOpt = this.rlsOpt!;
    const residualTracker = this.residualTracker!;
    const outlier = this.outlier!;

    const xRaw = this.xRawScratch!;
    const xNorm = this.xNormScratch!;
    const z = this.zScratch!;
    const zW = this.zWeightedScratch!;
    const yHat = this.yHatScratch!;
    const residual = this.residualScratch!;
    const sigma = this.sigmaScratch!;
    const errW = this.errWeightedScratch!;

    const T = this.nTargets | 0;
    const D = this.zDim | 0;

    let lastUpdateNorm = 0.0;
    let lastWeight = 1.0;

    for (let i = 0; i < N; i++) {
      // 1) Push X FIRST (authoritative latest-X)
      ring.pushRow(xCoordinates[i]);

      // Read latest raw x into xRaw scratch (from ring to enforce authoritative internal latest-X)
      ring.getLatestRow(xRaw);

      // Update normalization stats with raw x, then normalize
      normalizer.updateStats(xRaw);
      normalizer.normalize(xRaw, xNorm);

      // Update reservoir with normalized x
      reservoir.step(xNorm);

      // Build z and forward
      this.buildZ(reservoir.state, xNorm, z);
      readout.forward(z, yHat);

      // Residuals and sigma for outlier weighting computed from pre-update prediction
      residualTracker.getSigma(sigma);
      const yRow = yCoordinates[i];

      for (let t = 0; t < T; t++) residual[t] = yRow[t] - yHat[t];

      // Outlier weight based on z-score of residuals
      lastWeight = outlier.computeWeight(residual, sigma);
      this.fitRes.sampleWeight = lastWeight;

      // Loss (unweighted, deterministic)
      const loss = LossFunction.mse(yRow, yHat);
      this.metrics.add(loss);

      // Apply L2 weight decay to readout (deterministic)
      readout.applyL2WeightDecay(this.config.l2Lambda);

      // Weighted RLS update
      const sqrtW = Math.sqrt(lastWeight);
      // zW = sqrtW * z ; errW = sqrtW * (y - yHat)
      for (let d = 0; d < D; d++) zW[d] = z[d] * sqrtW;
      for (let t = 0; t < T; t++) errW[t] = residual[t] * sqrtW;

      lastUpdateNorm = rlsOpt.stepSharedP(
        rlsState,
        this.readoutParams!.Wout,
        zW,
        errW,
        T,
        this.config.gradientClipNorm,
      );

      // Update residual stats after observing residuals (1-step stats)
      residualTracker.update(residual);

      this.sampleCount++;
    }

    this.fitRes.samplesProcessed = N;
    this.fitRes.averageLoss = this.metrics.mean();
    this.fitRes.gradientNorm = lastUpdateNorm;
    this.fitRes.driftDetected = false;
    this.fitRes.sampleWeight = lastWeight;

    return this.fitRes;
  }

  /**
   * Multi-horizon prediction with deterministic roll-forward using scratch state.
   *
   * Critical behavior enforced:
   * - Uses ONLY internal RingBuffer (latest ingested X is authoritative).
   * - Context ends at most recently ingested X.
   * - Does NOT mutate RingBuffer or live reservoir state.
   * - futureSteps hard-capped by maxSequenceLength.
   */
  predict(futureSteps: number): PredictionResult {
    if (
      !this.initialized || !this.ring || !this.reservoir || !this.readout ||
      !this.normalizer || !this.residualTracker
    ) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }
    if ((futureSteps | 0) !== futureSteps || futureSteps < 1) {
      throw new Error("predict: futureSteps must be an integer >= 1");
    }
    if (futureSteps > this.config.maxSequenceLength) {
      throw new Error("predict: futureSteps must be <= maxSequenceLength");
    }
    if (this.ring.size() <= 0) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }

    const ring = this.ring;
    const normalizer = this.normalizer;
    const reservoir = this.reservoir;
    const readout = this.readout;
    const residualTracker = this.residualTracker;

    const N = this.config.reservoirSize | 0;
    const F = this.nFeatures | 0;
    const T = this.nTargets | 0;

    const rScratch = this.rScratch!;
    const rPrevScratch = this.rPrevScratch!;
    const xStepRaw = this.xStepRawScratch!;
    const xStepNorm = this.xStepNormScratch!;
    const zPred = this.zPredScratch!;
    const yPred = this.yPredScratch!;
    const sigma = this.sigmaPredScratch!;

    const res = this.predRes!;
    residualTracker.getSigma(sigma);

    // Copy live reservoir state to scratch (do not mutate live during predict)
    for (let i = 0; i < N; i++) rScratch[i] = reservoir.state[i];
    for (let i = 0; i < N; i++) rPrevScratch[i] = 0.0;

    // Authoritative latest x from ring buffer
    ring.getLatestRow(xStepRaw);

    // Confidence based on step-1 sigma (deterministic, clamped)
    let avgSigma = 0.0;
    for (let t = 0; t < T; t++) avgSigma += sigma[t];
    avgSigma /= T > 0 ? T : 1;
    let conf = 1.0 / (1.0 + avgSigma);
    if (!TensorOps.isFinite(conf)) conf = 0.0;
    res.confidence = TensorOps.clamp(conf, 0.0, 1.0);

    const uncMult = this.config.uncertaintyMultiplier;

    // Roll-forward K steps: for step k, we first advance reservoir by one step using x_{t+k} (default holdLastX)
    // and then compute y_hat_{t+k+1} from the advanced state.
    for (let step = 0; step < futureSteps; step++) {
      // Normalize xStepRaw for this step
      normalizer.normalize(xStepRaw, xStepNorm);

      // Advance scratch reservoir by one time step
      reservoir.updateStateInPlace(rScratch, rPrevScratch, xStepNorm);

      // Build z and forward
      this.buildZ(rScratch, xStepNorm, zPred);
      readout.forward(zPred, yPred);

      // Store predictions and bounds
      const pRow = res.predictions[step];
      const lRow = res.lowerBounds[step];
      const uRow = res.upperBounds[step];

      const horizonScale = Math.sqrt(step + 1); // sigma_k = sigma_1 * sqrt(k+1), where step=0 => 1-step
      for (let t = 0; t < T; t++) {
        const y = yPred[t];
        const s = sigma[t] * horizonScale;
        const delta = uncMult * s;
        pRow[t] = y;
        lRow[t] = y - delta;
        uRow[t] = y + delta;
      }

      // Determine next xStepRaw based on rollforwardMode (default holdLastX)
      if (this.config.rollforwardMode === "autoregressive" && F === T) {
        // x_{t+k+1} := y_hat_{t+k+1} (raw scale); deterministic, no allocations
        for (let j = 0; j < F; j++) xStepRaw[j] = yPred[j];
      } else {
        // holdLastX: keep xStepRaw unchanged
      }
    }

    return res;
  }

  getModelSummary(): ModelSummary {
    if (!this.initialized) {
      return {
        totalParameters: 0,
        receptiveField: this.config.maxSequenceLength | 0,
        spectralRadius: this.scaledSpectralRadius,
        reservoirSize: this.config.reservoirSize | 0,
        nFeatures: 0,
        nTargets: 0,
        maxSequenceLength: this.config.maxSequenceLength | 0,
        sampleCount: this.sampleCount | 0,
      };
    }
    const N = this.config.reservoirSize | 0;
    const F = this.nFeatures | 0;
    const T = this.nTargets | 0;
    const D = this.zDim | 0;
    const totalParams = (N * N + N * F + N + T * D + D * D) | 0;
    return {
      totalParameters: totalParams,
      receptiveField: this.config.maxSequenceLength | 0,
      spectralRadius: this.scaledSpectralRadius,
      reservoirSize: N,
      nFeatures: F,
      nTargets: T,
      maxSequenceLength: this.config.maxSequenceLength | 0,
      sampleCount: this.sampleCount | 0,
    };
  }

  getWeights(): WeightInfo {
    if (
      !this.initialized || !this.reservoirParams || !this.readoutParams ||
      !this.rlsState
    ) {
      return { weights: [] };
    }
    const N = this.config.reservoirSize | 0;
    const F = this.nFeatures | 0;
    const T = this.nTargets | 0;
    const D = this.zDim | 0;
    const rp = this.reservoirParams;
    const ro = this.readoutParams;
    const rls = this.rlsState;

    return {
      weights: [
        {
          name: "Win",
          shape: [N, F],
          values: SerializationHelper.toNumberArray(rp.Win),
        },
        {
          name: "W",
          shape: [N, N],
          values: SerializationHelper.toNumberArray(rp.W),
        },
        {
          name: "b",
          shape: [N],
          values: SerializationHelper.toNumberArray(rp.b),
        },
        {
          name: "Wout",
          shape: [T, D],
          values: SerializationHelper.toNumberArray(ro.Wout),
        },
        {
          name: "P",
          shape: [D, D],
          values: SerializationHelper.toNumberArray(rls.P),
        },
      ],
    };
  }

  getNormalizationStats(): NormalizationStats {
    if (!this.initialized || !this.normalizer) {
      return { means: [], stds: [], count: 0, isActive: false };
    }
    return this.normalizer.getStats();
  }

  reset(): void {
    if (!this.initialized) {
      this.metrics.reset();
      this.sampleCount = 0;
      this.fitRes.samplesProcessed = 0;
      this.fitRes.averageLoss = 0;
      this.fitRes.gradientNorm = 0;
      this.fitRes.driftDetected = false;
      this.fitRes.sampleWeight = 1;
      if (this.predRes) this.predRes.confidence = 0.0;
      return;
    }

    // Deterministically re-initialize weights and state in-place (no new allocations)
    const N = this.config.reservoirSize | 0;
    const F = this.nFeatures | 0;
    const T = this.nTargets | 0;
    const D = this.zDim | 0;

    // Reset ring buffer
    this.ring!.reset();

    // Reset normalizer and residual tracker
    this.normalizer!.reset();
    this.residualTracker!.reset();

    // Re-init reservoir weights and bias
    this.initReservoirWeights(
      this.reservoirParams!.W,
      this.reservoirParams!.Win,
      this.reservoirParams!.b,
    );

    // Reset reservoir state
    this.reservoir!.reset();
    TensorOps.fill(this.rScratch!, 0.0);
    TensorOps.fill(this.rPrevScratch!, 0.0);

    // Reset readout weights
    TensorOps.fill(this.readoutParams!.Wout, 0.0);

    // Reset RLS
    this.rlsState!.reset(this.config.rlsDelta);

    // Reset scratch buffers
    TensorOps.fill(this.xRawScratch!, 0.0);
    TensorOps.fill(this.xNormScratch!, 0.0);
    TensorOps.fill(this.zScratch!, 0.0);
    TensorOps.fill(this.zWeightedScratch!, 0.0);
    TensorOps.fill(this.yHatScratch!, 0.0);
    TensorOps.fill(this.residualScratch!, 0.0);
    TensorOps.fill(this.sigmaScratch!, 0.0);
    TensorOps.fill(this.errWeightedScratch!, 0.0);

    TensorOps.fill(this.xStepRawScratch!, 0.0);
    TensorOps.fill(this.xStepNormScratch!, 0.0);
    TensorOps.fill(this.zPredScratch!, 0.0);
    TensorOps.fill(this.yPredScratch!, 0.0);
    TensorOps.fill(this.sigmaPredScratch!, 0.0);

    // Reset metrics
    this.metrics.reset();
    this.sampleCount = 0;

    // Reset reusable results
    this.fitRes.samplesProcessed = 0;
    this.fitRes.averageLoss = 0;
    this.fitRes.gradientNorm = 0;
    this.fitRes.driftDetected = false;
    this.fitRes.sampleWeight = 1.0;

    if (this.predRes) {
      this.predRes.confidence = 0.0;
      // Arrays left as-is; caller should treat returned arrays as overwritten on next predict().
      // Optionally clear first row for determinism:
      for (let i = 0; i < this.config.maxSequenceLength; i++) {
        const p = this.predRes.predictions[i];
        const l = this.predRes.lowerBounds[i];
        const u = this.predRes.upperBounds[i];
        for (let t = 0; t < T; t++) {
          p[t] = 0.0;
          l[t] = 0.0;
          u[t] = 0.0;
        }
      }
    }

    // silence unused vars
    void N;
    void F;
    void T;
    void D;
  }

  save(): string {
    const obj: any = {
      version: 1,
      config: this.config,
      initialized: this.initialized,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      zDim: this.zDim,
      sampleCount: this.sampleCount,
      scaledSpectralRadius: this.scaledSpectralRadius,
      metrics: { meanLoss: this.metrics.mean() },
    };

    if (this.initialized) {
      obj.ring = this.ring!.toJSON();
      obj.normalizer = {
        mean: Array.from(this.normalizer!.acc.mean),
        m2: Array.from(this.normalizer!.acc.m2),
        std: Array.from(this.normalizer!.std),
        count: this.normalizer!.acc.count,
        eps: this.normalizer!.eps,
        warmup: this.normalizer!.warmup,
      };
      obj.residualTracker = this.residualTracker!.toJSON();
      obj.reservoirParams = {
        reservoirSize: this.reservoirParams!.reservoirSize,
        nFeatures: this.reservoirParams!.nFeatures,
        Win: Array.from(this.reservoirParams!.Win),
        W: Array.from(this.reservoirParams!.W),
        b: Array.from(this.reservoirParams!.b),
      };
      obj.reservoirState = Array.from(this.reservoir!.state);
      obj.readout = {
        nTargets: this.readoutParams!.nTargets,
        zDim: this.readoutParams!.zDim,
        Wout: Array.from(this.readoutParams!.Wout),
      };
      obj.rls = {
        zDim: this.rlsState!.zDim,
        P: Array.from(this.rlsState!.P),
      };
    }

    return JSON.stringify(obj);
  }

  load(w: string): void {
    const obj = JSON.parse(w);
    const cfg = obj.config as ESNRegressionConfig;

    // Replace config fields (keep defaults for any missing)
    const base = defaultConfig();
    const merged = Object.assign(base, cfg || {});
    (this as any).config = merged;

    this.initialized = false;
    this.nFeatures = obj.nFeatures | 0;
    this.nTargets = obj.nTargets | 0;
    this.zDim = obj.zDim | 0;
    this.sampleCount = obj.sampleCount | 0;
    this.scaledSpectralRadius = typeof obj.scaledSpectralRadius === "number"
      ? obj.scaledSpectralRadius
      : this.config.spectralRadius;

    this.rng = new RandomGenerator(this.config.seed);

    // Reinitialize full structures if saved initialized
    if (obj.initialized) {
      // Build minimal initializer expectations for allocations
      const dummyX: number[][] = [[...new Array(this.nFeatures).fill(0)]];
      const dummyY: number[][] = [[...new Array(this.nTargets).fill(0)]];
      this.ensureInitializedFromBatch(dummyX, dummyY);

      // Restore ring
      this.ring = RingBuffer.fromJSON(obj.ring);

      // Restore normalizer
      const norm = obj.normalizer;
      const normalizer = this.normalizer!;
      normalizer.acc.count = norm.count | 0;
      SerializationHelper.fromNumberArray(
        normalizer.acc.mean,
        norm.mean as number[],
      );
      SerializationHelper.fromNumberArray(
        normalizer.acc.m2,
        norm.m2 as number[],
      );
      SerializationHelper.fromNumberArray(normalizer.std, norm.std as number[]);

      // Restore residual tracker
      this.residualTracker = ResidualStatsTracker.fromJSON(obj.residualTracker);

      // Restore reservoir params
      const rp = obj.reservoirParams;
      SerializationHelper.fromNumberArray(
        this.reservoirParams!.Win,
        rp.Win as number[],
      );
      SerializationHelper.fromNumberArray(
        this.reservoirParams!.W,
        rp.W as number[],
      );
      SerializationHelper.fromNumberArray(
        this.reservoirParams!.b,
        rp.b as number[],
      );

      // Restore reservoir state
      const rs = obj.reservoirState as number[];
      const rState = this.reservoir!.state;
      for (let i = 0; i < rState.length; i++) rState[i] = rs[i];

      // Restore readout
      const ro = obj.readout;
      SerializationHelper.fromNumberArray(
        this.readoutParams!.Wout,
        ro.Wout as number[],
      );

      // Restore RLS P
      const rls = obj.rls;
      SerializationHelper.fromNumberArray(this.rlsState!.P, rls.P as number[]);

      // Reset internal metrics accumulator to reflect saved mean (best-effort, deterministic)
      this.metrics.reset();
      const meanLoss = obj.metrics && typeof obj.metrics.meanLoss === "number"
        ? obj.metrics.meanLoss
        : 0.0;
      // Store meanLoss approximately by adding one sample
      this.metrics.add(meanLoss);

      // Ensure prediction result allocated
      if (!this.predRes) {
        this.predRes = this.allocatePredictionResult(
          this.config.maxSequenceLength,
          this.nTargets,
        );
      }

      this.initialized = true;
    } else {
      this.reset();
    }
  }
}
