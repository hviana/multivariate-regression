/***********************
 * ESNRegression.ts
 * Self-contained TypeScript ESN / Reservoir Computing library for online multivariate regression.
 * Improved version with better numerical stability, prediction accuracy, and uncertainty estimation.
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
  maxSequenceLength: number;
  reservoirSize: number;
  spectralRadius: number;
  leakRate: number;
  inputScale: number;
  biasScale: number;
  reservoirSparsity: number;
  inputSparsity: number;
  activation: "tanh" | "relu";
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
  useSquaredFeatures: boolean;
  readoutTraining: "rls";
  rlsLambda: number;
  rlsDelta: number;
  epsilon: number;
  l2Lambda: number;
  gradientClipNorm: number;
  normalizationEpsilon: number;
  normalizationWarmup: number;
  washoutPeriod: number;
  outlierThreshold: number;
  outlierMinWeight: number;
  residualWindowSize: number;
  uncertaintyMultiplier: number;
  weightInitScale: number;
  seed: number;
  verbose: boolean;
  rollforwardMode: "holdLastX" | "autoregressive";
}

function defaultConfig(): ESNRegressionConfig {
  return {
    maxSequenceLength: 64,
    reservoirSize: 256,
    spectralRadius: 0.95,
    leakRate: 0.5,
    inputScale: 0.5,
    biasScale: 0.2,
    reservoirSparsity: 0.8,
    inputSparsity: 0.0,
    activation: "tanh",
    useInputInReadout: true,
    useBiasInReadout: true,
    useSquaredFeatures: false,
    readoutTraining: "rls",
    rlsLambda: 0.9995,
    rlsDelta: 0.1,
    epsilon: 1e-10,
    l2Lambda: 0.001,
    gradientClipNorm: 5.0,
    normalizationEpsilon: 1e-10,
    normalizationWarmup: 5,
    washoutPeriod: 0,
    outlierThreshold: 4.0,
    outlierMinWeight: 0.2,
    residualWindowSize: 100,
    uncertaintyMultiplier: 1.96,
    weightInitScale: 0.05,
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

/** Simple fixed-size buffer pool. */
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
  static safeDiv(num: number, denom: number, eps: number): number {
    const absDenom = Math.abs(denom);
    if (absDenom < eps) {
      return denom >= 0 ? num / eps : -num / eps;
    }
    return num / denom;
  }
  static ensureFinite(x: number, fallback: number): number {
    return Number.isFinite(x) ? x : fallback;
  }
  static softmax(x: number, temperature: number): number {
    return 1.0 / (1.0 + Math.exp(-x / temperature));
  }
}

/** Activation functions. */
class ActivationOps {
  static applyInPlace(a: Float64Array, kind: "tanh" | "relu"): void {
    if (kind === "tanh") {
      for (let i = 0; i < a.length; i++) a[i] = Math.tanh(a[i]);
    } else {
      for (let i = 0; i < a.length; i++) a[i] = a[i] > 0 ? a[i] : 0.01 * a[i]; // Leaky ReLU
    }
  }
  static applyScalar(x: number, kind: "tanh" | "relu"): number {
    if (kind === "tanh") return Math.tanh(x);
    return x > 0 ? x : 0.01 * x; // Leaky ReLU
  }
}

/** Deterministic PRNG (xorshift128+). */
class RandomGenerator {
  private s0: number;
  private s1: number;
  constructor(seed: number) {
    seed = (seed | 0) >>> 0;
    if (seed === 0) seed = 0x9e3779b9;
    this.s0 = seed;
    this.s1 = seed ^ 0xdeadbeef;
    for (let i = 0; i < 20; i++) this.nextU32();
  }
  nextU32(): number {
    let s1 = this.s0;
    const s0 = this.s1;
    this.s0 = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >>> 17;
    s1 ^= s0;
    s1 ^= s0 >>> 26;
    this.s1 = s1;
    return (this.s0 + this.s1) >>> 0;
  }
  nextFloat(): number {
    return (this.nextU32() >>> 0) / 4294967296.0;
  }
  nextSigned(): number {
    return this.nextFloat() * 2.0 - 1.0;
  }
  nextGaussian(): number {
    const u1 = Math.max(this.nextFloat(), 1e-10);
    const u2 = this.nextFloat();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    return r * Math.cos(2.0 * Math.PI * u2);
  }
}

/** Welford accumulator for mean/variance with numerical stability. */
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
  update(x: Float64Array): void {
    const n = this.mean.length;
    this.count++;
    const c = this.count;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      if (!TensorOps.isFinite(xi)) continue;
      const delta = xi - this.mean[i];
      this.mean[i] += delta / c;
      const delta2 = xi - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }
  varianceAt(i: number): number {
    if (this.count < 2) return 0.0;
    const v = this.m2[i] / (this.count - 1);
    return v > 0 ? v : 0.0;
  }
}

/** Online z-score normalizer with robust variance estimation. */
class WelfordNormalizer {
  readonly acc: WelfordAccumulator;
  readonly std: Float64Array;
  readonly minVals: Float64Array;
  readonly maxVals: Float64Array;
  readonly runningScale: Float64Array;
  readonly eps: number;
  readonly warmup: number;
  constructor(dim: number, eps: number, warmup: number) {
    this.acc = new WelfordAccumulator(dim | 0);
    this.std = new Float64Array(dim | 0);
    this.minVals = new Float64Array(dim | 0);
    this.maxVals = new Float64Array(dim | 0);
    this.runningScale = new Float64Array(dim | 0);
    this.eps = eps;
    this.warmup = warmup | 0;
    TensorOps.fill(this.minVals, Infinity);
    TensorOps.fill(this.maxVals, -Infinity);
    TensorOps.fill(this.runningScale, 1.0);
  }
  reset(): void {
    this.acc.reset();
    TensorOps.fill(this.std, 0.0);
    TensorOps.fill(this.minVals, Infinity);
    TensorOps.fill(this.maxVals, -Infinity);
    TensorOps.fill(this.runningScale, 1.0);
  }
  updateStats(xRaw: Float64Array): void {
    this.acc.update(xRaw);
    const n = this.std.length;
    for (let i = 0; i < n; i++) {
      const xi = xRaw[i];
      if (TensorOps.isFinite(xi)) {
        if (xi < this.minVals[i]) this.minVals[i] = xi;
        if (xi > this.maxVals[i]) this.maxVals[i] = xi;
      }
      const v = this.acc.varianceAt(i);
      this.std[i] = Math.sqrt(v);

      // Update running scale for smooth normalization during warmup
      const range = this.maxVals[i] - this.minVals[i];
      if (this.std[i] > this.eps) {
        this.runningScale[i] = this.std[i];
      } else if (range > this.eps) {
        this.runningScale[i] = range * 0.5;
      } else {
        this.runningScale[i] = 1.0;
      }
    }
  }
  normalize(xRaw: Float64Array, dst: Float64Array): void {
    const n = dst.length;
    const mean = this.acc.mean;
    const eps = this.eps;
    const count = this.acc.count;

    for (let i = 0; i < n; i++) {
      const xi = xRaw[i];
      if (!TensorOps.isFinite(xi)) {
        dst[i] = 0.0;
        continue;
      }

      // Progressive normalization - always center and scale, even during warmup
      if (count < 1) {
        // Very first sample - pass through
        dst[i] = xi;
        continue;
      }

      // Use running scale which updates smoothly
      let denom = this.runningScale[i];
      if (denom < eps) denom = 1.0;

      dst[i] = (xi - mean[i]) / denom;

      // Soft clipping for numerical stability
      const clipVal = 6.0;
      if (dst[i] > clipVal) dst[i] = clipVal + Math.log1p(dst[i] - clipVal);
      else if (dst[i] < -clipVal) {
        dst[i] = -clipVal - Math.log1p(-dst[i] - clipVal);
      }
    }
  }
  getStats(): NormalizationStats {
    const n = this.acc.mean.length;
    const means = new Array<number>(n);
    const stds = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      means[i] = this.acc.mean[i];
      stds[i] = this.runningScale[i] > this.eps
        ? this.runningScale[i]
        : this.eps;
    }
    return {
      means,
      stds,
      count: this.acc.count,
      isActive: this.acc.count >= this.warmup,
    };
  }
}

/** Target normalizer for output scaling (improves learning). */
class TargetNormalizer {
  readonly acc: WelfordAccumulator;
  readonly std: Float64Array;
  readonly runningScale: Float64Array;
  readonly eps: number;
  readonly warmup: number;
  constructor(dim: number, eps: number, warmup: number) {
    this.acc = new WelfordAccumulator(dim | 0);
    this.std = new Float64Array(dim | 0);
    this.runningScale = new Float64Array(dim | 0);
    this.eps = eps;
    this.warmup = warmup | 0;
    TensorOps.fill(this.std, 1.0);
    TensorOps.fill(this.runningScale, 1.0);
  }
  reset(): void {
    this.acc.reset();
    TensorOps.fill(this.std, 1.0);
    TensorOps.fill(this.runningScale, 1.0);
  }
  updateStats(yRaw: number[]): void {
    const n = this.acc.mean.length;
    this.acc.count++;
    const c = this.acc.count;
    for (let i = 0; i < n; i++) {
      const yi = yRaw[i];
      if (!TensorOps.isFinite(yi)) continue;
      const delta = yi - this.acc.mean[i];
      this.acc.mean[i] += delta / c;
      const delta2 = yi - this.acc.mean[i];
      this.acc.m2[i] += delta * delta2;
    }
    for (let i = 0; i < n; i++) {
      const v = this.acc.varianceAt(i);
      const s = Math.sqrt(v);
      this.std[i] = s > this.eps ? s : 1.0;
      // Exponential moving average for smooth scale changes
      const alpha = 0.1;
      this.runningScale[i] = (1 - alpha) * this.runningScale[i] +
        alpha * this.std[i];
    }
  }
  normalize(yRaw: number[], dst: Float64Array): void {
    const n = dst.length;
    const mean = this.acc.mean;
    const count = this.acc.count;
    for (let i = 0; i < n; i++) {
      if (count < 2) {
        dst[i] = yRaw[i];
      } else {
        const scale = this.runningScale[i] > this.eps
          ? this.runningScale[i]
          : 1.0;
        dst[i] = (yRaw[i] - mean[i]) / scale;
      }
    }
  }
  denormalize(yNorm: Float64Array, dst: Float64Array): void {
    const n = dst.length;
    const mean = this.acc.mean;
    const count = this.acc.count;
    for (let i = 0; i < n; i++) {
      if (count < 2) {
        dst[i] = yNorm[i];
      } else {
        const scale = this.runningScale[i] > this.eps
          ? this.runningScale[i]
          : 1.0;
        dst[i] = yNorm[i] * scale + mean[i];
      }
    }
  }
  denormalizeInPlace(y: Float64Array): void {
    const n = y.length;
    const mean = this.acc.mean;
    const count = this.acc.count;
    if (count < 2) return;
    for (let i = 0; i < n; i++) {
      const scale = this.runningScale[i] > this.eps
        ? this.runningScale[i]
        : 1.0;
      y[i] = y[i] * scale + mean[i];
    }
  }
  getScale(idx: number): number {
    return this.runningScale[idx] > this.eps ? this.runningScale[idx] : 1.0;
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
  }
  size(): number {
    return this.count;
  }
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
  getLatestRow(dst: Float64Array): void {
    if (this.count <= 0) throw new Error("RingBuffer.getLatestRow: empty");
    let idx = this.head - 1;
    if (idx < 0) idx += this.capacity;
    const base = idx * this.dim;
    for (let j = 0; j < this.dim; j++) dst[j] = this.data[base + j];
  }
  getRow(stepsBack: number, dst: Float64Array): boolean {
    if (stepsBack < 0 || stepsBack >= this.count) return false;
    let idx = this.head - 1 - stepsBack;
    while (idx < 0) idx += this.capacity;
    const base = idx * this.dim;
    for (let j = 0; j < this.dim; j++) dst[j] = this.data[base + j];
    return true;
  }
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

/** Tracks recent residual distribution per target with exponential smoothing. */
class ResidualStatsTracker {
  readonly windowSize: number;
  readonly nTargets: number;
  private buffers: Float64Array;
  private writeIndex: number;
  private count: number;
  private sum: Float64Array;
  private sumsq: Float64Array;
  private expMean: Float64Array;
  private expVar: Float64Array;
  private readonly alpha: number;
  private eps: number;

  constructor(nTargets: number, windowSize: number, eps: number) {
    this.nTargets = nTargets | 0;
    this.windowSize = windowSize | 0;
    this.eps = eps;
    this.alpha = 2.0 / (windowSize + 1);
    this.buffers = new Float64Array((this.nTargets * this.windowSize) | 0);
    this.writeIndex = 0;
    this.count = 0;
    this.sum = new Float64Array(this.nTargets);
    this.sumsq = new Float64Array(this.nTargets);
    this.expMean = new Float64Array(this.nTargets);
    this.expVar = new Float64Array(this.nTargets);
  }

  reset(): void {
    this.writeIndex = 0;
    this.count = 0;
    TensorOps.fill(this.buffers, 0.0);
    TensorOps.fill(this.sum, 0.0);
    TensorOps.fill(this.sumsq, 0.0);
    TensorOps.fill(this.expMean, 0.0);
    TensorOps.fill(this.expVar, 0.0);
  }

  update(residuals: Float64Array): void {
    const nT = this.nTargets;
    const w = this.windowSize;
    const idx = this.writeIndex;
    const alpha = this.alpha;

    if (this.count >= w) {
      const baseOld = idx * nT;
      for (let t = 0; t < nT; t++) {
        const old = this.buffers[baseOld + t];
        this.sum[t] -= old;
        this.sumsq[t] -= old * old;
      }
    } else {
      this.count++;
    }

    const base = idx * nT;
    for (let t = 0; t < nT; t++) {
      const r = TensorOps.ensureFinite(residuals[t], 0.0);
      this.buffers[base + t] = r;
      this.sum[t] += r;
      this.sumsq[t] += r * r;

      const delta = r - this.expMean[t];
      this.expMean[t] += alpha * delta;
      this.expVar[t] = (1 - alpha) * (this.expVar[t] + alpha * delta * delta);
    }

    this.writeIndex++;
    if (this.writeIndex >= w) this.writeIndex = 0;
  }

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
      const windowVar = Math.max(0, this.sumsq[t] / c - mean * mean);
      const expStd = Math.sqrt(Math.max(0, this.expVar[t]));
      const windowStd = Math.sqrt(windowVar);
      // Use more robust estimate combining both
      const s = 0.7 * Math.max(expStd, windowStd) +
        0.3 * (expStd + windowStd) * 0.5;
      dst[t] = s > eps ? s : eps;
    }
  }

  getMeanAbsResidual(): number {
    if (this.count === 0) return 0;
    let total = 0;
    for (let t = 0; t < this.nTargets; t++) {
      total += Math.abs(this.expMean[t]);
    }
    return total / this.nTargets;
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
      expMean: Array.from(this.expMean),
      expVar: Array.from(this.expVar),
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
    if (obj.expMean) {
      for (let i = 0; i < tr.expMean.length; i++) {
        tr.expMean[i] = obj.expMean[i];
      }
    }
    if (obj.expVar) {
      for (let i = 0; i < tr.expVar.length; i++) tr.expVar[i] = obj.expVar[i];
    }
    return tr;
  }
}

/** Downweights outliers based on residual z-score with smooth transition. */
class OutlierDownweighter {
  private threshold: number;
  private minWeight: number;
  constructor(threshold: number, minWeight: number) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }
  computeWeight(residuals: Float64Array, sigma: Float64Array): number {
    let maxZ = 0.0;
    let sumZ = 0.0;
    let n = 0;
    for (let i = 0; i < residuals.length; i++) {
      const s = sigma[i] > 1e-10 ? sigma[i] : 1.0;
      const z = Math.abs(residuals[i]) / s;
      if (TensorOps.isFinite(z)) {
        if (z > maxZ) maxZ = z;
        sumZ += z;
        n++;
      }
    }
    if (!TensorOps.isFinite(maxZ) || n === 0) return this.minWeight;

    // Use combination of max and mean z-score
    const avgZ = sumZ / n;
    const combinedZ = 0.7 * maxZ + 0.3 * avgZ;

    if (combinedZ <= this.threshold) return 1.0;

    // Smoother decay using tanh
    const excess = combinedZ - this.threshold;
    const w = this.minWeight +
      (1.0 - this.minWeight) * (1.0 - Math.tanh(excess * 0.5));
    return TensorOps.clamp(w, this.minWeight, 1.0);
  }
}

/** Loss function utilities. */
class LossFunction {
  static mse(y: number[], yhat: Float64Array): number {
    const n = y.length;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const d = y[i] - yhat[i];
      s += d * d;
    }
    return s / (n > 0 ? n : 1);
  }
  static mae(y: number[], yhat: Float64Array): number {
    const n = y.length;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      s += Math.abs(y[i] - yhat[i]);
    }
    return s / (n > 0 ? n : 1);
  }
  static huber(y: number[], yhat: Float64Array, delta: number = 1.0): number {
    const n = y.length;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(y[i] - yhat[i]);
      if (d <= delta) {
        s += 0.5 * d * d;
      } else {
        s += delta * (d - 0.5 * delta);
      }
    }
    return s / (n > 0 ? n : 1);
  }
}

/** Tracks running metrics with exponential smoothing. */
class MetricsAccumulator {
  private sumLoss: number;
  private count: number;
  private expLoss: number;
  private minLoss: number;
  private maxLoss: number;
  private readonly alpha: number;
  constructor() {
    this.sumLoss = 0.0;
    this.count = 0;
    this.expLoss = 0.0;
    this.minLoss = Infinity;
    this.maxLoss = -Infinity;
    this.alpha = 0.05;
  }
  reset(): void {
    this.sumLoss = 0.0;
    this.count = 0;
    this.expLoss = 0.0;
    this.minLoss = Infinity;
    this.maxLoss = -Infinity;
  }
  add(loss: number): void {
    if (!TensorOps.isFinite(loss)) return;
    this.sumLoss += loss;
    this.count++;
    if (loss < this.minLoss) this.minLoss = loss;
    if (loss > this.maxLoss) this.maxLoss = loss;
    if (this.count === 1) {
      this.expLoss = loss;
    } else {
      this.expLoss = (1 - this.alpha) * this.expLoss + this.alpha * loss;
    }
  }
  mean(): number {
    if (this.count <= 0) return 0.0;
    return this.sumLoss / this.count;
  }
  exponentialMean(): number {
    return this.expLoss;
  }
  getMin(): number {
    return this.minLoss === Infinity ? 0 : this.minLoss;
  }
  getMax(): number {
    return this.maxLoss === -Infinity ? 0 : this.maxLoss;
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
    // Initialize with random vector for better convergence
    let norm = 0.0;
    for (let i = 0; i < n; i++) {
      tmpV[i] = Math.sin(i * 0.1) + 0.5;
      norm += tmpV[i] * tmpV[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      const inv = 1.0 / norm;
      for (let i = 0; i < n; i++) tmpV[i] *= inv;
    }

    let eigenvalue = 0.0;
    for (let it = 0; it < iters; it++) {
      // W * v
      for (let i = 0; i < n; i++) {
        let s = 0.0;
        const row = i * n;
        for (let j = 0; j < n; j++) s += W[row + j] * tmpV[j];
        tmpWv[i] = s;
      }

      // Compute norm
      norm = 0.0;
      for (let i = 0; i < n; i++) {
        const v = tmpWv[i];
        norm += v * v;
      }
      norm = Math.sqrt(norm);

      if (!TensorOps.isFinite(norm) || norm <= 1e-12) return 0.0;

      eigenvalue = norm;
      const inv = 1.0 / norm;
      for (let i = 0; i < n; i++) tmpV[i] = tmpWv[i] * inv;
    }

    return eigenvalue;
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
    if (est < eps) {
      return target;
    }
    const scale = target / est;
    for (let i = 0; i < W.length; i++) W[i] *= scale;
    return target;
  }
}

class ESNReservoirParams {
  readonly reservoirSize: number;
  readonly nFeatures: number;
  readonly Win: Float64Array;
  readonly W: Float64Array;
  readonly b: Float64Array;
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
  readonly state: Float64Array;
  private prev: Float64Array;
  private preAct: Float64Array;

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
    this.preAct = arena.alloc(params.reservoirSize);
    TensorOps.fill(this.state, 0.0);
    TensorOps.fill(this.prev, 0.0);
    TensorOps.fill(this.preAct, 0.0);
  }

  reset(): void {
    TensorOps.fill(this.state, 0.0);
    TensorOps.fill(this.prev, 0.0);
    TensorOps.fill(this.preAct, 0.0);
  }

  /**
   * ESN leaky integrator update:
   * r_t = (1-α) r_{t-1} + α tanh( Win*(inputScale*x) + W*r_{t-1} + b )
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

    for (let i = 0; i < N; i++) prev[i] = state[i];

    for (let i = 0; i < N; i++) {
      let s = b[i];
      const winRow = i * F;
      for (let j = 0; j < F; j++) {
        const xj = xNorm[j];
        if (TensorOps.isFinite(xj)) {
          s += Win[winRow + j] * xj * inScale;
        }
      }
      const wRow = i * N;
      for (let j = 0; j < N; j++) {
        s += W[wRow + j] * prev[j];
      }
      const a = ActivationOps.applyScalar(s, this.activation);
      state[i] = oneMinus * prev[i] + leak * a;
      if (!TensorOps.isFinite(state[i])) {
        state[i] = prev[i] * 0.9; // Graceful degradation
      }
    }
  }

  step(xNorm: Float64Array): void {
    this.updateStateInPlace(this.state, this.prev, xNorm);
  }
}

class ReadoutConfig {
  readonly useInputInReadout: boolean;
  readonly useBiasInReadout: boolean;
  readonly useSquaredFeatures: boolean;
  readonly l2Lambda: number;
  readonly gradientClipNorm: number;
  constructor(cfg: ESNRegressionConfig) {
    this.useInputInReadout = cfg.useInputInReadout;
    this.useBiasInReadout = cfg.useBiasInReadout;
    this.useSquaredFeatures = cfg.useSquaredFeatures;
    this.l2Lambda = cfg.l2Lambda;
    this.gradientClipNorm = cfg.gradientClipNorm;
  }
}

class ReadoutParams {
  readonly nTargets: number;
  readonly zDim: number;
  readonly Wout: Float64Array;
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
  forward(z: Float64Array, yOut: Float64Array): void {
    const T = this.params.nTargets;
    const D = this.params.zDim;
    const W = this.params.Wout;
    for (let t = 0; t < T; t++) {
      const base = t * D;
      let s = 0.0;
      for (let i = 0; i < D; i++) s += W[base + i] * z[i];
      yOut[t] = TensorOps.ensureFinite(s, 0.0);
    }
  }
}

class RLSState {
  readonly zDim: number;
  readonly P: Float64Array;
  readonly Pz: Float64Array;
  readonly k: Float64Array;
  private readonly l2Lambda: number;
  private readonly eps: number;
  private updateCount: number;
  constructor(
    zDim: number,
    P: Float64Array,
    Pz: Float64Array,
    k: Float64Array,
    l2Lambda: number,
    eps: number,
  ) {
    this.zDim = zDim | 0;
    this.P = P;
    this.Pz = Pz;
    this.k = k;
    this.l2Lambda = l2Lambda;
    this.eps = eps;
    this.updateCount = 0;
  }
  reset(delta: number): void {
    const D = this.zDim;
    const P = this.P;
    TensorOps.fill(P, 0.0);
    const v = 1.0 / (delta > 0 ? delta : 1.0);
    for (let i = 0; i < D; i++) P[i * D + i] = v;
    TensorOps.fill(this.Pz, 0.0);
    TensorOps.fill(this.k, 0.0);
    this.updateCount = 0;
  }
  stabilize(): void {
    const D = this.zDim;
    const P = this.P;
    const eps = this.eps;
    const l2 = this.l2Lambda;

    // Enforce symmetry
    for (let i = 0; i < D; i++) {
      for (let j = i + 1; j < D; j++) {
        const avg = 0.5 * (P[i * D + j] + P[j * D + i]);
        P[i * D + j] = avg;
        P[j * D + i] = avg;
      }
    }

    // Compute trace for adaptive regularization
    let trace = 0.0;
    for (let i = 0; i < D; i++) {
      trace += P[i * D + i];
    }
    const avgDiag = trace / D;
    const minDiag = Math.max(eps, l2, avgDiag * 0.001);
    const maxDiag = Math.max(avgDiag * 100, 1e6);

    for (let i = 0; i < D; i++) {
      let pii = P[i * D + i];
      if (!TensorOps.isFinite(pii)) pii = avgDiag > 0 ? avgDiag : 1.0;
      if (pii < minDiag) pii = minDiag;
      if (pii > maxDiag) pii = maxDiag;
      P[i * D + i] = pii;
    }

    this.updateCount++;
  }
  getUpdateCount(): number {
    return this.updateCount;
  }
}

class RLSOptimizer {
  private lambda: number;
  private eps: number;
  private l2Lambda: number;
  constructor(lambda: number, eps: number, l2Lambda: number) {
    this.lambda = lambda;
    this.eps = eps;
    this.l2Lambda = l2Lambda;
  }

  /**
   * RLS update with regularization using Joseph form for better numerical stability
   */
  stepSharedP(
    state: RLSState,
    Wout: Float64Array,
    z: Float64Array,
    err: Float64Array,
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
      Pz[i] = TensorOps.ensureFinite(s, 0.0);
    }

    // zTPz = z^T * P * z
    let zTPz = 0.0;
    for (let i = 0; i < D; i++) zTPz += z[i] * Pz[i];
    zTPz = TensorOps.ensureFinite(zTPz, 1.0);

    // denom = lambda + zTPz
    let denom = lambda + zTPz;
    if (!TensorOps.isFinite(denom) || denom < eps) {
      denom = eps;
    }

    // k = Pz / denom (Kalman gain)
    const invDen = 1.0 / denom;
    for (let i = 0; i < D; i++) {
      k[i] = Pz[i] * invDen;
      if (!TensorOps.isFinite(k[i])) k[i] = 0.0;
    }

    // Compute update magnitude for gradient clipping
    let sumErr2 = 0.0;
    for (let t = 0; t < nTargets; t++) {
      const e = err[t];
      if (TensorOps.isFinite(e)) {
        sumErr2 += e * e;
      }
    }
    const kNorm = TensorOps.l2Norm(k);
    let updateNorm = kNorm * Math.sqrt(sumErr2);
    if (!TensorOps.isFinite(updateNorm)) {
      updateNorm = gradientClipNorm > 0 ? gradientClipNorm : 1.0;
    }

    let scale = 1.0;
    if (
      gradientClipNorm > 0 && updateNorm > gradientClipNorm && updateNorm > eps
    ) {
      scale = gradientClipNorm / updateNorm;
      updateNorm = gradientClipNorm;
    }

    // Update weights: W += k * e^T (scaled)
    for (let t = 0; t < nTargets; t++) {
      const base = t * D;
      const et = TensorOps.ensureFinite(err[t], 0.0) * scale;
      for (let i = 0; i < D; i++) {
        const delta = k[i] * et;
        if (TensorOps.isFinite(delta)) {
          Wout[base + i] += delta;
        }
      }
    }

    // Joseph form P update: P = (I - k*z^T) * P * (I - k*z^T)^T / lambda + k*k^T * sigma_e
    // Simplified: P = (P - k * Pz^T) / lambda
    const invLambda = 1.0 / (lambda > eps ? lambda : eps);
    for (let i = 0; i < D; i++) {
      const ki = k[i];
      const row = i * D;
      for (let j = 0; j < D; j++) {
        P[row + j] = (P[row + j] - ki * Pz[j]) * invLambda;
      }
    }

    state.stabilize();

    return updateNorm;
  }
}

class SerializationHelper {
  static toNumberArray(a: Float64Array): number[] {
    return Array.from(a);
  }
  static fromNumberArray(dst: Float64Array, src: number[]): void {
    const n = dst.length;
    for (let i = 0; i < n; i++) dst[i] = src[i];
  }
}

/** Main model. */
export class ESNRegression {
  readonly config: ESNRegressionConfig;

  private initialized: boolean;
  private nFeatures: number;
  private nTargets: number;
  private zDim: number;
  private baseZDim: number;

  private rng: RandomGenerator;
  private arena: TensorArena | null;
  private ring: RingBuffer | null;
  private normalizer: WelfordNormalizer | null;
  private targetNormalizer: TargetNormalizer | null;
  private residualTracker: ResidualStatsTracker | null;
  private outlier: OutlierDownweighter | null;
  private reservoirParams: ESNReservoirParams | null;
  private reservoir: ESNReservoir | null;
  private readoutCfg: ReadoutConfig | null;
  private readoutParams: ReadoutParams | null;
  private readout: LinearReadout | null;
  private rlsState: RLSState | null;
  private rlsOpt: RLSOptimizer | null;

  private xRawScratch: Float64Array | null;
  private xNormScratch: Float64Array | null;
  private zScratch: Float64Array | null;
  private yHatScratch: Float64Array | null;
  private yNormScratch: Float64Array | null;
  private residualScratch: Float64Array | null;
  private sigmaScratch: Float64Array | null;
  private errScratch: Float64Array | null;

  private rScratch: Float64Array | null;
  private rPrevScratch: Float64Array | null;
  private xStepRawScratch: Float64Array | null;
  private xStepNormScratch: Float64Array | null;
  private zPredScratch: Float64Array | null;
  private yPredScratch: Float64Array | null;
  private yPredDenormScratch: Float64Array | null;
  private sigmaPredScratch: Float64Array | null;

  private fitRes: FitResult;
  private predRes: PredictionResult | null;

  private metrics: MetricsAccumulator;
  private sampleCount: number;
  private scaledSpectralRadius: number;
  private trainedSamples: number;
  private lastPredictions: Float64Array | null;

  constructor(cfg?: Partial<ESNRegressionConfig>) {
    const base = defaultConfig();
    this.config = Object.assign(base, cfg || {});
    if (this.config.maxSequenceLength <= 0) this.config.maxSequenceLength = 1;
    if (this.config.reservoirSize <= 0) this.config.reservoirSize = 1;
    if (this.config.leakRate <= 0) this.config.leakRate = 0.01;
    if (this.config.leakRate > 1) this.config.leakRate = 1;
    this.config.rlsLambda = TensorOps.clamp(this.config.rlsLambda, 0.9, 1.0);
    if (this.config.washoutPeriod < 0) this.config.washoutPeriod = 0;

    this.initialized = false;
    this.nFeatures = 0;
    this.nTargets = 0;
    this.zDim = 0;
    this.baseZDim = 0;
    this.rng = new RandomGenerator(this.config.seed);
    this.arena = null;
    this.ring = null;
    this.normalizer = null;
    this.targetNormalizer = null;
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
    this.yHatScratch = null;
    this.yNormScratch = null;
    this.residualScratch = null;
    this.sigmaScratch = null;
    this.errScratch = null;

    this.rScratch = null;
    this.rPrevScratch = null;
    this.xStepRawScratch = null;
    this.xStepNormScratch = null;
    this.zPredScratch = null;
    this.yPredScratch = null;
    this.yPredDenormScratch = null;
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
    this.trainedSamples = 0;
    this.scaledSpectralRadius = this.config.spectralRadius;
    this.lastPredictions = null;
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
    const useSq = this.config.useSquaredFeatures;

    // Base z dimension: reservoir + optional input + optional bias
    this.baseZDim = N + (useX ? F : 0) + (useB ? 1 : 0);
    // With squared features: add squared reservoir states (subsampled)
    const sqDim = useSq ? Math.min(N, 64) : 0;
    this.zDim = (this.baseZDim + sqDim) | 0;

    const D = this.zDim;
    let total = 0;
    total += N * N;
    total += N * F;
    total += N;
    total += N;
    total += N;
    total += N;
    total += N;
    total += N;
    total += D * D;
    total += D;
    total += D;
    total += T * D;
    total += F;
    total += F;
    total += D;
    total += T;
    total += T;
    total += T;
    total += T;
    total += T;
    total += F;
    total += F;
    total += D;
    total += T;
    total += T;
    total += T;
    total += N;
    total += N;
    total += T;
    total += 256;

    this.arena = new TensorArena(total | 0);

    const W = this.arena.alloc(N * N);
    const Win = this.arena.alloc(N * F);
    const b = this.arena.alloc(N);

    this.initReservoirWeights(W, Win, b);

    this.reservoirParams = new ESNReservoirParams(N, F, Win, W, b);
    this.reservoir = new ESNReservoir(
      this.reservoirParams,
      this.config.activation,
      this.config.leakRate,
      this.config.inputScale,
      this.arena,
    );

    this.rScratch = this.arena.alloc(N);
    this.rPrevScratch = this.arena.alloc(N);
    TensorOps.fill(this.rScratch, 0.0);
    TensorOps.fill(this.rPrevScratch, 0.0);

    const Wout = this.arena.alloc(T * D);
    this.initReadoutWeights(Wout);
    this.readoutCfg = new ReadoutConfig(this.config);
    this.readoutParams = new ReadoutParams(T, D, Wout);
    this.readout = new LinearReadout(this.readoutParams);

    const P = this.arena.alloc(D * D);
    const Pz = this.arena.alloc(D);
    const k = this.arena.alloc(D);
    this.rlsState = new RLSState(
      D,
      P,
      Pz,
      k,
      this.config.l2Lambda,
      this.config.epsilon,
    );
    this.rlsState.reset(this.config.rlsDelta);
    this.rlsOpt = new RLSOptimizer(
      this.config.rlsLambda,
      this.config.epsilon,
      this.config.l2Lambda,
    );

    this.ring = new RingBuffer(this.config.maxSequenceLength, F);
    this.normalizer = new WelfordNormalizer(
      F,
      this.config.normalizationEpsilon,
      this.config.normalizationWarmup,
    );
    this.targetNormalizer = new TargetNormalizer(
      T,
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

    this.xRawScratch = this.arena.alloc(F);
    this.xNormScratch = this.arena.alloc(F);
    this.zScratch = this.arena.alloc(D);
    this.yHatScratch = this.arena.alloc(T);
    this.yNormScratch = this.arena.alloc(T);
    this.residualScratch = this.arena.alloc(T);
    this.sigmaScratch = this.arena.alloc(T);
    this.errScratch = this.arena.alloc(T);

    this.xStepRawScratch = this.arena.alloc(F);
    this.xStepNormScratch = this.arena.alloc(F);
    this.zPredScratch = this.arena.alloc(D);
    this.yPredScratch = this.arena.alloc(T);
    this.yPredDenormScratch = this.arena.alloc(T);
    this.sigmaPredScratch = this.arena.alloc(T);
    this.lastPredictions = this.arena.alloc(T);

    TensorOps.fill(this.xRawScratch, 0.0);
    TensorOps.fill(this.xNormScratch, 0.0);
    TensorOps.fill(this.zScratch, 0.0);
    TensorOps.fill(this.yHatScratch, 0.0);
    TensorOps.fill(this.yNormScratch, 0.0);
    TensorOps.fill(this.residualScratch, 0.0);
    TensorOps.fill(this.sigmaScratch, 0.0);
    TensorOps.fill(this.errScratch, 0.0);
    TensorOps.fill(this.xStepRawScratch, 0.0);
    TensorOps.fill(this.xStepNormScratch, 0.0);
    TensorOps.fill(this.zPredScratch, 0.0);
    TensorOps.fill(this.yPredScratch, 0.0);
    TensorOps.fill(this.yPredDenormScratch, 0.0);
    TensorOps.fill(this.sigmaPredScratch, 0.0);
    TensorOps.fill(this.lastPredictions, 0.0);

    this.metrics.reset();
    this.sampleCount = 0;
    this.trainedSamples = 0;

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

    const winMask = new Uint8Array(N * F);
    ReservoirInitMask.initMask(rng, winMask.length, cfg.inputSparsity, winMask);
    const wscale = cfg.weightInitScale;

    // Initialize input weights with slight structure
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < F; j++) {
        const idx = i * F + j;
        if (winMask[idx]) {
          Win[idx] = rng.nextGaussian() * wscale;
        } else {
          Win[idx] = 0.0;
        }
      }
    }

    const wMask = new Uint8Array(N * N);
    ReservoirInitMask.initMask(rng, wMask.length, cfg.reservoirSparsity, wMask);

    // Initialize reservoir with slight scale variation for richness
    for (let i = 0; i < N * N; i++) {
      if (wMask[i]) {
        W[i] = rng.nextGaussian() * wscale;
      } else {
        W[i] = 0.0;
      }
    }

    // Initialize biases
    const bscale = cfg.biasScale;
    for (let i = 0; i < N; i++) {
      b[i] = rng.nextGaussian() * bscale;
    }

    // Scale to target spectral radius with more iterations for accuracy
    const tmpV = new Float64Array(N);
    const tmpWv = new Float64Array(N);
    this.scaledSpectralRadius = SpectralRadiusScaler.scaleToSpectralRadius(
      W,
      N,
      cfg.spectralRadius,
      100, // More iterations for better accuracy
      tmpV,
      tmpWv,
      cfg.epsilon,
    );
  }

  private initReadoutWeights(Wout: Float64Array): void {
    // Initialize with small random values for faster initial learning
    const rng = new RandomGenerator(this.config.seed + 12345);
    const scale = 0.001;
    for (let i = 0; i < Wout.length; i++) {
      Wout[i] = rng.nextGaussian() * scale;
    }
  }

  private buildZ(
    rState: Float64Array,
    xNorm: Float64Array,
    zOut: Float64Array,
  ): void {
    const N = this.config.reservoirSize | 0;
    const F = this.nFeatures | 0;
    const useSq = this.config.useSquaredFeatures;
    let idx = 0;

    // Reservoir states
    for (let i = 0; i < N; i++) {
      zOut[idx++] = rState[i];
    }

    // Optional input features
    if (this.config.useInputInReadout) {
      for (let j = 0; j < F; j++) {
        zOut[idx++] = xNorm[j];
      }
    }

    // Optional bias
    if (this.config.useBiasInReadout) {
      zOut[idx++] = 1.0;
    }

    // Optional squared features (subsampled for efficiency)
    if (useSq) {
      const sqDim = Math.min(N, 64);
      const step = Math.max(1, Math.floor(N / sqDim));
      for (let i = 0; i < sqDim; i++) {
        const ri = rState[i * step];
        zOut[idx++] = ri * ri;
      }
    }
  }

  /**
   * Online training with one sample at a time (batch accepted).
   * @param args.xCoordinates Feature rows [nSamples][nFeatures]
   * @param args.yCoordinates Target rows [nSamples][nTargets]
   * @returns FitResult with metrics
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
    if (N > 0) {
      if (!this.initialized) {
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
      this.fitRes.samplesProcessed = 0;
      this.fitRes.averageLoss = this.metrics.mean();
      this.fitRes.gradientNorm = 0;
      this.fitRes.driftDetected = false;
      this.fitRes.sampleWeight = 1.0;
      return this.fitRes;
    }

    if (!this.initialized) {
      throw new Error("fitOnline: failed to initialize");
    }

    const ring = this.ring!;
    const normalizer = this.normalizer!;
    const targetNormalizer = this.targetNormalizer!;
    const reservoir = this.reservoir!;
    const readout = this.readout!;
    const rlsState = this.rlsState!;
    const rlsOpt = this.rlsOpt!;
    const residualTracker = this.residualTracker!;
    const outlier = this.outlier!;

    const xRaw = this.xRawScratch!;
    const xNorm = this.xNormScratch!;
    const z = this.zScratch!;
    const yHat = this.yHatScratch!;
    const yNorm = this.yNormScratch!;
    const residual = this.residualScratch!;
    const sigma = this.sigmaScratch!;
    const err = this.errScratch!;

    const T = this.nTargets | 0;
    const warmup = this.config.normalizationWarmup;
    const washout = this.config.washoutPeriod;

    let lastUpdateNorm = 0.0;
    let lastWeight = 1.0;
    let driftDetected = false;

    for (let i = 0; i < N; i++) {
      // Store input
      ring.pushRow(xCoordinates[i]);
      ring.getLatestRow(xRaw);

      // Update input statistics BEFORE normalization
      normalizer.updateStats(xRaw);
      normalizer.normalize(xRaw, xNorm);

      // Step reservoir
      reservoir.step(xNorm);

      // Build feature vector and compute prediction
      this.buildZ(reservoir.state, xNorm, z);
      readout.forward(z, yHat);

      // Denormalize prediction BEFORE updating target stats (consistent scale)
      targetNormalizer.denormalizeInPlace(yHat);

      // Now update target statistics with new data
      const yRow = yCoordinates[i];
      targetNormalizer.updateStats(yRow);

      // Compute residual in original space
      for (let t = 0; t < T; t++) {
        residual[t] = yRow[t] - yHat[t];
        this.lastPredictions![t] = yHat[t];
      }

      // Get current uncertainty estimate
      residualTracker.getSigma(sigma);

      // Compute outlier weight
      lastWeight = outlier.computeWeight(residual, sigma);
      this.fitRes.sampleWeight = lastWeight;

      // Compute loss
      const loss = LossFunction.mse(yRow, yHat);
      this.metrics.add(loss);

      // Detect potential drift
      const expLoss = this.metrics.exponentialMean();
      if (this.trainedSamples > warmup * 2 && loss > expLoss * 5) {
        driftDetected = true;
      }

      // RLS update (skip during washout and warmup)
      if (this.sampleCount >= warmup + washout) {
        // Normalize target for learning
        targetNormalizer.normalize(yRow, yNorm);

        // Recompute prediction in normalized space for RLS
        this.buildZ(reservoir.state, xNorm, z);
        readout.forward(z, yHat);

        // Compute weighted error in normalized space
        for (let t = 0; t < T; t++) {
          err[t] = (yNorm[t] - yHat[t]) * Math.sqrt(lastWeight);
        }

        // RLS update
        lastUpdateNorm = rlsOpt.stepSharedP(
          rlsState,
          this.readoutParams!.Wout,
          z,
          err,
          T,
          this.config.gradientClipNorm,
        );
        this.trainedSamples++;
      }

      // Update residual statistics
      residualTracker.update(residual);
      this.sampleCount++;
    }

    this.fitRes.samplesProcessed = N;
    this.fitRes.averageLoss = this.metrics.mean();
    this.fitRes.gradientNorm = lastUpdateNorm;
    this.fitRes.driftDetected = driftDetected;
    this.fitRes.sampleWeight = lastWeight;

    return this.fitRes;
  }

  /**
   * Predict multiple steps ahead using deterministic roll-forward.
   * @param futureSteps Number of steps to predict (1 to maxSequenceLength)
   * @returns PredictionResult with predictions and uncertainty bounds
   */
  predict(futureSteps: number): PredictionResult {
    if (
      !this.initialized ||
      !this.ring ||
      !this.reservoir ||
      !this.readout ||
      !this.normalizer ||
      !this.residualTracker ||
      !this.targetNormalizer
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
    const targetNormalizer = this.targetNormalizer;
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
    const yPredDenorm = this.yPredDenormScratch!;
    const sigma = this.sigmaPredScratch!;

    const res = this.predRes!;
    residualTracker.getSigma(sigma);

    // Copy current reservoir state for rollforward
    for (let i = 0; i < N; i++) rScratch[i] = reservoir.state[i];

    // Get latest input
    ring.getLatestRow(xStepRaw);

    // Compute confidence based on multiple factors
    let avgSigma = 0.0;
    for (let t = 0; t < T; t++) avgSigma += sigma[t];
    avgSigma /= T > 0 ? T : 1;

    const meanAbsRes = residualTracker.getMeanAbsResidual();
    const expLoss = this.metrics.exponentialMean();
    const minLoss = this.metrics.getMin();

    // Relative error normalized by recent performance
    const relativeError = avgSigma > 1e-10
      ? Math.sqrt(expLoss) / avgSigma
      : 1.0;

    // Compute confidence score
    let conf = 1.0 / (1.0 + relativeError * 0.5 + meanAbsRes * 0.05);

    // Penalize if we haven't trained enough
    const minSamples = this.config.normalizationWarmup +
      this.config.washoutPeriod;
    if (this.trainedSamples < minSamples * 2) {
      conf *= this.trainedSamples / (minSamples * 2);
    }

    // Boost confidence if recent loss is near minimum
    if (minLoss > 0 && expLoss < minLoss * 2) {
      conf = Math.min(conf * 1.2, 1.0);
    }

    if (!TensorOps.isFinite(conf)) conf = 0.0;
    res.confidence = TensorOps.clamp(conf, 0.0, 1.0);

    const uncMult = this.config.uncertaintyMultiplier;
    const rollMode = this.config.rollforwardMode;
    const canAutoregress = rollMode === "autoregressive" && F === T;

    for (let step = 0; step < futureSteps; step++) {
      // Normalize current input
      normalizer.normalize(xStepRaw, xStepNorm);

      // Update reservoir state
      reservoir.updateStateInPlace(rScratch, rPrevScratch, xStepNorm);

      // Build features and predict
      this.buildZ(rScratch, xStepNorm, zPred);
      readout.forward(zPred, yPred);

      // Denormalize to original space
      targetNormalizer.denormalize(yPred, yPredDenorm);

      const pRow = res.predictions[step];
      const lRow = res.lowerBounds[step];
      const uRow = res.upperBounds[step];

      // Uncertainty grows with horizon (sub-linear for stability)
      const horizonFactor = 1.0 + 0.1 * Math.log1p(step);
      const horizonScale = Math.sqrt(step + 1) * horizonFactor;

      for (let t = 0; t < T; t++) {
        const y = yPredDenorm[t];
        const targetScale = targetNormalizer.getScale(t);
        const s = sigma[t] * horizonScale * targetScale;
        const delta = uncMult * s;
        pRow[t] = TensorOps.ensureFinite(y, this.lastPredictions![t]);
        lRow[t] = TensorOps.ensureFinite(y - delta, pRow[t]);
        uRow[t] = TensorOps.ensureFinite(y + delta, pRow[t]);
      }

      // For autoregressive mode, feed prediction back as input
      if (canAutoregress) {
        for (let j = 0; j < F; j++) {
          const newVal = yPredDenorm[j];
          xStepRaw[j] = TensorOps.ensureFinite(newVal, xStepRaw[j]);
        }
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
      !this.initialized ||
      !this.reservoirParams ||
      !this.readoutParams ||
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
      this.trainedSamples = 0;
      this.fitRes.samplesProcessed = 0;
      this.fitRes.averageLoss = 0;
      this.fitRes.gradientNorm = 0;
      this.fitRes.driftDetected = false;
      this.fitRes.sampleWeight = 1;
      if (this.predRes) this.predRes.confidence = 0.0;
      return;
    }

    const T = this.nTargets | 0;

    this.ring!.reset();
    this.normalizer!.reset();
    this.targetNormalizer!.reset();
    this.residualTracker!.reset();

    this.initReservoirWeights(
      this.reservoirParams!.W,
      this.reservoirParams!.Win,
      this.reservoirParams!.b,
    );

    this.reservoir!.reset();
    TensorOps.fill(this.rScratch!, 0.0);
    TensorOps.fill(this.rPrevScratch!, 0.0);

    this.initReadoutWeights(this.readoutParams!.Wout);

    this.rlsState!.reset(this.config.rlsDelta);

    TensorOps.fill(this.xRawScratch!, 0.0);
    TensorOps.fill(this.xNormScratch!, 0.0);
    TensorOps.fill(this.zScratch!, 0.0);
    TensorOps.fill(this.yHatScratch!, 0.0);
    TensorOps.fill(this.yNormScratch!, 0.0);
    TensorOps.fill(this.residualScratch!, 0.0);
    TensorOps.fill(this.sigmaScratch!, 0.0);
    TensorOps.fill(this.errScratch!, 0.0);

    TensorOps.fill(this.xStepRawScratch!, 0.0);
    TensorOps.fill(this.xStepNormScratch!, 0.0);
    TensorOps.fill(this.zPredScratch!, 0.0);
    TensorOps.fill(this.yPredScratch!, 0.0);
    TensorOps.fill(this.yPredDenormScratch!, 0.0);
    TensorOps.fill(this.sigmaPredScratch!, 0.0);
    TensorOps.fill(this.lastPredictions!, 0.0);

    this.metrics.reset();
    this.sampleCount = 0;
    this.trainedSamples = 0;

    this.fitRes.samplesProcessed = 0;
    this.fitRes.averageLoss = 0;
    this.fitRes.gradientNorm = 0;
    this.fitRes.driftDetected = false;
    this.fitRes.sampleWeight = 1.0;

    if (this.predRes) {
      this.predRes.confidence = 0.0;
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
  }

  save(): string {
    const obj: any = {
      version: 3,
      config: this.config,
      initialized: this.initialized,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      zDim: this.zDim,
      baseZDim: this.baseZDim,
      sampleCount: this.sampleCount,
      trainedSamples: this.trainedSamples,
      scaledSpectralRadius: this.scaledSpectralRadius,
      metrics: {
        meanLoss: this.metrics.mean(),
        expLoss: this.metrics.exponentialMean(),
        minLoss: this.metrics.getMin(),
        maxLoss: this.metrics.getMax(),
      },
    };

    if (this.initialized) {
      obj.ring = this.ring!.toJSON();
      obj.normalizer = {
        mean: Array.from(this.normalizer!.acc.mean),
        m2: Array.from(this.normalizer!.acc.m2),
        std: Array.from(this.normalizer!.std),
        minVals: Array.from(this.normalizer!.minVals),
        maxVals: Array.from(this.normalizer!.maxVals),
        runningScale: Array.from(this.normalizer!.runningScale),
        count: this.normalizer!.acc.count,
        eps: this.normalizer!.eps,
        warmup: this.normalizer!.warmup,
      };
      obj.targetNormalizer = {
        mean: Array.from(this.targetNormalizer!.acc.mean),
        m2: Array.from(this.targetNormalizer!.acc.m2),
        std: Array.from(this.targetNormalizer!.std),
        runningScale: Array.from(this.targetNormalizer!.runningScale),
        count: this.targetNormalizer!.acc.count,
        eps: this.targetNormalizer!.eps,
        warmup: this.targetNormalizer!.warmup,
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
      obj.lastPredictions = Array.from(this.lastPredictions!);
    }

    return JSON.stringify(obj);
  }

  load(w: string): void {
    const obj = JSON.parse(w);
    const cfg = obj.config as ESNRegressionConfig;

    const base = defaultConfig();
    const merged = Object.assign(base, cfg || {});
    (this as any).config = merged;

    this.initialized = false;
    this.nFeatures = obj.nFeatures | 0;
    this.nTargets = obj.nTargets | 0;
    this.zDim = obj.zDim | 0;
    this.baseZDim = obj.baseZDim | 0;
    this.sampleCount = obj.sampleCount | 0;
    this.trainedSamples = obj.trainedSamples | 0;
    this.scaledSpectralRadius = typeof obj.scaledSpectralRadius === "number"
      ? obj.scaledSpectralRadius
      : this.config.spectralRadius;

    this.rng = new RandomGenerator(this.config.seed);

    if (obj.initialized) {
      const dummyX: number[][] = [[...new Array(this.nFeatures).fill(0)]];
      const dummyY: number[][] = [[...new Array(this.nTargets).fill(0)]];
      this.ensureInitializedFromBatch(dummyX, dummyY);

      this.ring = RingBuffer.fromJSON(obj.ring);

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
      if (norm.minVals) {
        SerializationHelper.fromNumberArray(
          normalizer.minVals,
          norm.minVals as number[],
        );
      }
      if (norm.maxVals) {
        SerializationHelper.fromNumberArray(
          normalizer.maxVals,
          norm.maxVals as number[],
        );
      }
      if (norm.runningScale) {
        SerializationHelper.fromNumberArray(
          normalizer.runningScale,
          norm.runningScale as number[],
        );
      }

      if (obj.targetNormalizer) {
        const tnorm = obj.targetNormalizer;
        const targetNormalizer = this.targetNormalizer!;
        targetNormalizer.acc.count = tnorm.count | 0;
        SerializationHelper.fromNumberArray(
          targetNormalizer.acc.mean,
          tnorm.mean as number[],
        );
        SerializationHelper.fromNumberArray(
          targetNormalizer.acc.m2,
          tnorm.m2 as number[],
        );
        SerializationHelper.fromNumberArray(
          targetNormalizer.std,
          tnorm.std as number[],
        );
        if (tnorm.runningScale) {
          SerializationHelper.fromNumberArray(
            targetNormalizer.runningScale,
            tnorm.runningScale as number[],
          );
        }
      }

      this.residualTracker = ResidualStatsTracker.fromJSON(obj.residualTracker);

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

      const rs = obj.reservoirState as number[];
      const rState = this.reservoir!.state;
      for (let i = 0; i < rState.length; i++) rState[i] = rs[i];

      const ro = obj.readout;
      SerializationHelper.fromNumberArray(
        this.readoutParams!.Wout,
        ro.Wout as number[],
      );

      const rls = obj.rls;
      SerializationHelper.fromNumberArray(this.rlsState!.P, rls.P as number[]);

      if (obj.lastPredictions) {
        SerializationHelper.fromNumberArray(
          this.lastPredictions!,
          obj.lastPredictions as number[],
        );
      }

      this.metrics.reset();
      const meanLoss = obj.metrics && typeof obj.metrics.meanLoss === "number"
        ? obj.metrics.meanLoss
        : 0.0;
      this.metrics.add(meanLoss);

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
