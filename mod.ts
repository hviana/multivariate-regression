/*******************************************************
 * ESNRegression: Echo State Network for Multivariate Regression
 * Incremental online learning via RLS readout training
 * Welford z-score normalization + residual-based uncertainty
 *
 * Single self-contained TypeScript module.
 *******************************************************/

// ============================================================================
// PUBLIC INTERFACES
// ============================================================================

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

// ============================================================================
// DEFAULT CONFIGURATION (MUST NOT CHANGE DEFAULT VALUES)
// ============================================================================

const DEFAULT_CONFIG: ESNRegressionConfig = {
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

// ============================================================================
// SMALL UTILS (no allocations in hot paths)
// ============================================================================

function clamp01(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function isFiniteNumber(x: number): boolean {
  return Number.isFinite(x);
}

// ============================================================================
// 1. MEMORY INFRA: TensorShape, TensorView, BufferPool, TensorArena, TensorOps
// ============================================================================

class TensorShape {
  readonly dims: readonly number[];
  readonly size: number;
  readonly strides: readonly number[];

  constructor(dims: number[]) {
    const dd = new Array<number>(dims.length);
    let size = 1;
    for (let i = 0; i < dims.length; i++) {
      const v = dims[i] | 0;
      if (v <= 0) {
        throw new Error("TensorShape: dims must be positive integers");
      }
      dd[i] = v;
      size *= v;
    }
    this.dims = Object.freeze(dd);
    this.size = size;

    const strides = new Array<number>(dims.length);
    let stride = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= dd[i];
    }
    this.strides = Object.freeze(strides);
  }

  index(indices: number[]): number {
    let idx = 0;
    for (let i = 0; i < indices.length; i++) {
      idx += (indices[i] | 0) * (this.strides[i] | 0);
    }
    return idx;
  }
}

class TensorView {
  data: Float64Array;
  offset: number;
  shape: TensorShape;

  constructor(data: Float64Array, offset: number, shape: TensorShape) {
    this.data = data;
    this.offset = offset | 0;
    this.shape = shape;
  }

  getFlat(i: number): number {
    return this.data[this.offset + (i | 0)];
  }
  setFlat(i: number, v: number): void {
    this.data[this.offset + (i | 0)] = v;
  }

  fill(v: number): void {
    const start = this.offset;
    const end = start + this.shape.size;
    for (let i = start; i < end; i++) this.data[i] = v;
  }

  copyFrom(src: TensorView): void {
    const n = this.shape.size;
    const a = this.data;
    const b = src.data;
    let ia = this.offset;
    let ib = src.offset;
    for (let i = 0; i < n; i++) a[ia++] = b[ib++];
  }
}

class BufferPool {
  private pool: TensorView[] = [];

  acquire(data: Float64Array, offset: number, shape: TensorShape): TensorView {
    const v = this.pool.pop();
    if (v) {
      v.data = data;
      v.offset = offset | 0;
      v.shape = shape;
      return v;
    }
    return new TensorView(data, offset, shape);
  }

  release(v: TensorView): void {
    this.pool.push(v);
  }
}

class TensorArena {
  private buf: Float64Array;
  private used: number = 0;

  constructor(totalSize: number) {
    if ((totalSize | 0) <= 0) {
      throw new Error("TensorArena: totalSize must be > 0");
    }
    this.buf = new Float64Array(totalSize | 0);
  }

  allocate(shape: TensorShape): TensorView {
    const offset = this.used;
    this.used += shape.size;
    if (this.used > this.buf.length) {
      throw new Error("TensorArena: out of memory");
    }
    return new TensorView(this.buf, offset, shape);
  }

  reset(): void {
    this.used = 0;
    this.buf.fill(0);
  }
}

class TensorOps {
  static vecCopy(
    src: Float64Array,
    srcOff: number,
    dst: Float64Array,
    dstOff: number,
    n: number,
  ): void {
    let i0 = srcOff | 0;
    let o0 = dstOff | 0;
    for (let i = 0; i < (n | 0); i++) dst[o0++] = src[i0++];
  }

  static vecFill(
    dst: Float64Array,
    dstOff: number,
    v: number,
    n: number,
  ): void {
    let o0 = dstOff | 0;
    for (let i = 0; i < (n | 0); i++) dst[o0++] = v;
  }

  static dot(
    a: Float64Array,
    aOff: number,
    b: Float64Array,
    bOff: number,
    n: number,
  ): number {
    let sum = 0;
    let ia = aOff | 0;
    let ib = bOff | 0;
    for (let i = 0; i < (n | 0); i++) sum += a[ia++] * b[ib++];
    return sum;
  }

  static norm2(a: Float64Array, aOff: number, n: number): number {
    let sum = 0;
    let ia = aOff | 0;
    for (let i = 0; i < (n | 0); i++) {
      const v = a[ia++];
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  /**
   * Dense mat-vec: y = A*x, A row-major [rows x cols]
   */
  static matVec(
    A: Float64Array,
    aOff: number,
    rows: number,
    cols: number,
    x: Float64Array,
    xOff: number,
    y: Float64Array,
    yOff: number,
  ): void {
    const r = rows | 0;
    const c = cols | 0;
    for (let i = 0; i < r; i++) {
      let sum = 0;
      let aIdx = (aOff + i * c) | 0;
      let xIdx = xOff | 0;
      for (let j = 0; j < c; j++) sum += A[aIdx++] * x[xIdx++];
      y[(yOff + i) | 0] = sum;
    }
  }

  /**
   * Sparse mat-vec using 0/1 mask: y = A*x, A row-major [rows x cols]
   */
  static sparseMatVec(
    A: Float64Array,
    aOff: number,
    rows: number,
    cols: number,
    mask: Uint8Array,
    mOff: number,
    x: Float64Array,
    xOff: number,
    y: Float64Array,
    yOff: number,
  ): void {
    const r = rows | 0;
    const c = cols | 0;
    for (let i = 0; i < r; i++) {
      let sum = 0;
      let aIdx = (aOff + i * c) | 0;
      let mIdx = (mOff + i * c) | 0;
      let xIdxBase = xOff | 0;
      for (let j = 0; j < c; j++) {
        if (mask[mIdx++]) sum += A[aIdx] * x[xIdxBase + j];
        aIdx++;
      }
      y[(yOff + i) | 0] = sum;
    }
  }
}

// ============================================================================
// 2. NUMERICS: ActivationOps, RandomGenerator, WelfordAccumulator, Normalizer,
//    LossFunction, MetricsAccumulator
// ============================================================================

class ActivationOps {
  static apply(act: "tanh" | "relu", x: number): number {
    if (act === "tanh") return Math.tanh(x);
    return x > 0 ? x : 0;
  }
}

class RandomGenerator {
  private state: number;
  constructor(seed: number) {
    this.state = (seed >>> 0) || 1;
  }
  nextU32(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }
  uniform(): number {
    return this.nextU32() / 4294967296;
  }
  normal(mean: number = 0, std: number = 1): number {
    const u1 = this.uniform();
    const u2 = this.uniform();
    const z = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }
  bernoulli(p: number): boolean {
    return this.uniform() < p;
  }
  getState(): number {
    return this.state;
  }
  setState(s: number): void {
    this.state = (s >>> 0) || 1;
  }
}

class WelfordAccumulator {
  count: number = 0;
  mean: number = 0;
  m2: number = 0;

  update(x: number): void {
    this.count++;
    const d = x - this.mean;
    this.mean += d / this.count;
    const d2 = x - this.mean;
    this.m2 += d * d2;
  }

  variance(): number {
    if (this.count < 2) return 0;
    return this.m2 / (this.count - 1);
  }

  std(): number {
    return Math.sqrt(this.variance());
  }

  reset(): void {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }
}

class WelfordNormalizer {
  private nFeatures: number;
  private counts: Float64Array;
  private means: Float64Array;
  private m2s: Float64Array;
  private eps: number;
  private warmup: number;

  constructor(nFeatures: number, eps: number, warmup: number) {
    this.nFeatures = nFeatures | 0;
    this.counts = new Float64Array(this.nFeatures);
    this.means = new Float64Array(this.nFeatures);
    this.m2s = new Float64Array(this.nFeatures);
    this.eps = eps;
    this.warmup = warmup | 0;
  }

  update(x: Float64Array, xOff: number): void {
    let idx = xOff | 0;
    for (let j = 0; j < this.nFeatures; j++) {
      const v = x[idx++];
      const c = (this.counts[j] += 1);
      const d = v - this.means[j];
      this.means[j] += d / c;
      const d2 = v - this.means[j];
      this.m2s[j] += d * d2;
    }
  }

  /**
   * x_norm[j] = (x[j] - mean[j]) / max(std[j], eps)
   */
  normalize(
    x: Float64Array,
    xOff: number,
    out: Float64Array,
    outOff: number,
  ): void {
    let xi = xOff | 0;
    let oi = outOff | 0;
    const eps = this.eps;
    for (let j = 0; j < this.nFeatures; j++) {
      const c = this.counts[j];
      const mean = this.means[j];
      const varr = c > 1 ? this.m2s[j] / (c - 1) : 0;
      const std = Math.sqrt(varr);
      const denom = std > eps ? std : eps;
      const v = (x[xi++] - mean) / denom;
      out[oi++] = isFiniteNumber(v) ? v : 0;
    }
  }

  isActive(): boolean {
    if (this.nFeatures === 0) return false;
    return this.counts[0] >= this.warmup;
  }

  getCount(): number {
    if (this.nFeatures === 0) return 0;
    return this.counts[0];
  }

  getMeans(): number[] {
    return Array.from(this.means);
  }

  getStds(): number[] {
    const out = new Array<number>(this.nFeatures);
    for (let j = 0; j < this.nFeatures; j++) {
      const c = this.counts[j];
      const varr = c > 1 ? this.m2s[j] / (c - 1) : 0;
      out[j] = Math.sqrt(varr);
    }
    return out;
  }

  reset(): void {
    this.counts.fill(0);
    this.means.fill(0);
    this.m2s.fill(0);
  }

  serialize(): { counts: number[]; means: number[]; m2s: number[] } {
    return {
      counts: Array.from(this.counts),
      means: Array.from(this.means),
      m2s: Array.from(this.m2s),
    };
  }

  deserialize(d: { counts: number[]; means: number[]; m2s: number[] }): void {
    for (let i = 0; i < this.nFeatures; i++) {
      this.counts[i] = d.counts[i] || 0;
      this.means[i] = d.means[i] || 0;
      this.m2s[i] = d.m2s[i] || 0;
    }
  }
}

class LossFunction {
  static mse(
    pred: Float64Array,
    pOff: number,
    y: Float64Array,
    yOff: number,
    n: number,
  ): number {
    let sum = 0;
    let ip = pOff | 0;
    let iy = yOff | 0;
    for (let i = 0; i < (n | 0); i++) {
      const d = pred[ip++] - y[iy++];
      sum += d * d;
    }
    return sum / (n | 0);
  }

  static residuals(
    pred: Float64Array,
    pOff: number,
    y: Float64Array,
    yOff: number,
    out: Float64Array,
    oOff: number,
    n: number,
  ): void {
    let ip = pOff | 0;
    let iy = yOff | 0;
    let io = oOff | 0;
    for (let i = 0; i < (n | 0); i++) out[io++] = pred[ip++] - y[iy++];
  }
}

class MetricsAccumulator {
  private lossSum = 0;
  private gradSum = 0;
  private count = 0;
  private lastWeight = 1;

  reset(): void {
    this.lossSum = 0;
    this.gradSum = 0;
    this.count = 0;
    this.lastWeight = 1;
  }

  update(loss: number, grad: number, w: number): void {
    this.lossSum += loss;
    this.gradSum += grad;
    this.count++;
    this.lastWeight = w;
  }

  avgLoss(): number {
    return this.count > 0 ? this.lossSum / this.count : 0;
  }

  avgGrad(): number {
    return this.count > 0 ? this.gradSum / this.count : 0;
  }

  getLastWeight(): number {
    return this.lastWeight;
  }
}

// ============================================================================
// 6. TRAINING UTILITIES: RingBuffer, ResidualStatsTracker, OutlierDownweighter
// ============================================================================

class RingBuffer {
  private buf: Float64Array;
  private head: number = 0;
  private count: number = 0;
  private cap: number;
  private nFeat: number;

  constructor(capacity: number, nFeatures: number) {
    this.cap = capacity | 0;
    this.nFeat = nFeatures | 0;
    this.buf = new Float64Array(this.cap * this.nFeat);
  }

  push(row: number[] | Float64Array, rowOff: number = 0): void {
    const base = (this.head * this.nFeat) | 0;
    if (row instanceof Float64Array) {
      let ri = rowOff | 0;
      for (let j = 0; j < this.nFeat; j++) this.buf[base + j] = row[ri++];
    } else {
      for (let j = 0; j < this.nFeat; j++) this.buf[base + j] = row[j];
    }

    this.head = (this.head + 1) % this.cap;
    if (this.count < this.cap) this.count++;
  }

  getLatest(out: Float64Array, outOff: number): boolean {
    if (this.count === 0) return false;
    const idx = (((this.head - 1 + this.cap) % this.cap) * this.nFeat) | 0;
    let oi = outOff | 0;
    for (let j = 0; j < this.nFeat; j++) out[oi++] = this.buf[idx + j];
    return true;
  }

  isEmpty(): boolean {
    return this.count === 0;
  }

  getCount(): number {
    return this.count;
  }

  reset(): void {
    this.head = 0;
    this.count = 0;
    this.buf.fill(0);
  }

  serialize(): { buffer: number[]; head: number; count: number } {
    return { buffer: Array.from(this.buf), head: this.head, count: this.count };
  }

  deserialize(d: { buffer: number[]; head: number; count: number }): void {
    const b = d.buffer;
    for (let i = 0; i < this.buf.length; i++) this.buf[i] = b[i] || 0;
    this.head = d.head | 0;
    this.count = d.count | 0;
  }
}

class ResidualStatsTracker {
  private win: number;
  private nT: number;
  private buf: Float64Array; // [nTargets * win]
  private head: Int32Array;
  private count: Int32Array;
  private sum: Float64Array;
  private sumSq: Float64Array;

  constructor(windowSize: number, nTargets: number) {
    this.win = windowSize | 0;
    this.nT = nTargets | 0;
    this.buf = new Float64Array(this.win * this.nT);
    this.head = new Int32Array(this.nT);
    this.count = new Int32Array(this.nT);
    this.sum = new Float64Array(this.nT);
    this.sumSq = new Float64Array(this.nT);
  }

  update(residuals: Float64Array, off: number): void {
    let ri = off | 0;
    for (let t = 0; t < this.nT; t++) {
      const r = residuals[ri++];
      const h = this.head[t] | 0;
      const base = (t * this.win) | 0;
      const idx = base + h;

      if (this.count[t] === this.win) {
        const old = this.buf[idx];
        this.sum[t] -= old;
        this.sumSq[t] -= old * old;
      } else {
        this.count[t] = (this.count[t] + 1) | 0;
      }

      this.buf[idx] = r;
      this.sum[t] += r;
      this.sumSq[t] += r * r;
      this.head[t] = ((h + 1) % this.win) | 0;
    }
  }

  getStd(t: number): number {
    const n = this.count[t] | 0;
    if (n < 2) return 0;
    const mean = this.sum[t] / n;
    const varr = this.sumSq[t] / n - mean * mean;
    return Math.sqrt(varr > 0 ? varr : 0);
  }

  getStds(out: Float64Array, off: number): void {
    let oi = off | 0;
    for (let t = 0; t < this.nT; t++) out[oi++] = this.getStd(t);
  }

  reset(): void {
    this.buf.fill(0);
    this.head.fill(0);
    this.count.fill(0);
    this.sum.fill(0);
    this.sumSq.fill(0);
  }

  serialize(): {
    buffers: number[];
    heads: number[];
    counts: number[];
    sums: number[];
    sumSqs: number[];
  } {
    return {
      buffers: Array.from(this.buf),
      heads: Array.from(this.head),
      counts: Array.from(this.count),
      sums: Array.from(this.sum),
      sumSqs: Array.from(this.sumSq),
    };
  }

  deserialize(
    d: {
      buffers: number[];
      heads: number[];
      counts: number[];
      sums: number[];
      sumSqs: number[];
    },
  ): void {
    const bb = d.buffers;
    for (let i = 0; i < this.buf.length; i++) this.buf[i] = bb[i] || 0;
    for (let i = 0; i < this.nT; i++) {
      this.head[i] = d.heads[i] | 0;
      this.count[i] = d.counts[i] | 0;
      this.sum[i] = d.sums[i] || 0;
      this.sumSq[i] = d.sumSqs[i] || 0;
    }
  }
}

class OutlierDownweighter {
  private thr: number;
  private minW: number;

  constructor(threshold: number, minWeight: number) {
    this.thr = threshold;
    this.minW = minWeight;
  }

  computeWeight(
    residuals: Float64Array,
    rOff: number,
    stds: Float64Array,
    sOff: number,
    nTargets: number,
  ): number {
    let maxZ = 0;
    let ir = rOff | 0;
    let is = sOff | 0;
    for (let t = 0; t < (nTargets | 0); t++) {
      const std = stds[is++];
      if (std > 1e-12) {
        const z = Math.abs(residuals[ir]) / std;
        if (z > maxZ) maxZ = z;
      }
      ir++;
    }
    if (maxZ <= this.thr) return 1.0;
    const excess = maxZ - this.thr;
    const w = 1.0 / (1.0 + excess);
    return w < this.minW ? this.minW : w;
  }
}

// ============================================================================
// 4. RESERVOIR: ReservoirInitMask, SpectralRadiusScaler, ESNReservoirParams, ESNReservoir
// ============================================================================

class ReservoirInitMask {
  static generateReservoirMask(
    size: number,
    sparsity: number,
    rng: RandomGenerator,
  ): Uint8Array {
    const n = (size | 0) * (size | 0);
    const mask = new Uint8Array(n);
    const density = 1.0 - sparsity;
    for (let i = 0; i < n; i++) mask[i] = rng.bernoulli(density) ? 1 : 0;
    return mask;
  }

  static generateInputMask(
    rows: number,
    cols: number,
    sparsity: number,
    rng: RandomGenerator,
  ): Uint8Array {
    const n = (rows | 0) * (cols | 0);
    const mask = new Uint8Array(n);
    const density = 1.0 - sparsity;
    for (let i = 0; i < n; i++) mask[i] = rng.bernoulli(density) ? 1 : 0;
    return mask;
  }
}

class SpectralRadiusScaler {
  // fixed iteration count for determinism and speed
  private static readonly ITERS = 60;

  /**
   * Power iteration estimate for |lambda_max|
   * Uses deterministic initial vector derived from indices (not RNG-consuming).
   */
  static estimate(
    W: Float64Array,
    wOff: number,
    size: number,
    mask: Uint8Array,
    mOff: number,
    scratch: Float64Array,
    scratchOff: number,
  ): number {
    const n = size | 0;
    const vOff = scratchOff | 0;
    const uOff = (scratchOff + n) | 0;

    // deterministic non-zero init
    for (let i = 0; i < n; i++) {
      const x = ((i * 2654435761) >>> 0) / 4294967296; // [0,1)
      scratch[vOff + i] = x - 0.5;
      scratch[uOff + i] = 0;
    }

    let vNorm = TensorOps.norm2(scratch, vOff, n);
    if (vNorm < 1e-12) vNorm = 1;
    const inv0 = 1 / vNorm;
    for (let i = 0; i < n; i++) scratch[vOff + i] *= inv0;

    let est = 0;
    for (let it = 0; it < SpectralRadiusScaler.ITERS; it++) {
      TensorOps.sparseMatVec(
        W,
        wOff,
        n,
        n,
        mask,
        mOff,
        scratch,
        vOff,
        scratch,
        uOff,
      );

      const uNorm = TensorOps.norm2(scratch, uOff, n);
      if (uNorm < 1e-12) return 0;

      // Rayleigh quotient approx: v^T u (v normalized)
      est = TensorOps.dot(scratch, vOff, scratch, uOff, n);

      const inv = 1 / uNorm;
      for (let i = 0; i < n; i++) scratch[vOff + i] = scratch[uOff + i] * inv;
    }
    return Math.abs(est);
  }

  static scale(
    W: Float64Array,
    wOff: number,
    size: number,
    currentRadius: number,
    targetRadius: number,
  ): void {
    if (currentRadius <= 1e-12) return;
    const s = targetRadius / currentRadius;
    const n = (size | 0) * (size | 0);
    let idx = wOff | 0;
    for (let i = 0; i < n; i++) W[idx++] *= s;
  }
}

export interface ESNReservoirParams {
  Win: Float64Array;
  W: Float64Array;
  b: Float64Array;
  WinMask: Uint8Array;
  WMask: Uint8Array;
}

class ESNReservoir {
  private rs: number;
  private nf: number;
  private leak: number;
  private act: "tanh" | "relu";
  private inputScale: number;

  private Win: Float64Array;
  private W: Float64Array;
  private b: Float64Array;
  private WinMask: Uint8Array;
  private WMask: Uint8Array;

  private state: Float64Array;

  // scratch (shared across live + predict scratch updates; safe in single-thread usage)
  private tmpPre: Float64Array;
  private tmpIn: Float64Array;
  private tmpRec: Float64Array;

  private estimatedRadius: number = 0;

  constructor(
    reservoirSize: number,
    nFeatures: number,
    config: ESNRegressionConfig,
    rng: RandomGenerator,
  ) {
    this.rs = reservoirSize | 0;
    this.nf = nFeatures | 0;
    this.leak = config.leakRate;
    this.act = config.activation;
    this.inputScale = config.inputScale;

    this.Win = new Float64Array(this.rs * this.nf);
    this.W = new Float64Array(this.rs * this.rs);
    this.b = new Float64Array(this.rs);

    this.WinMask = ReservoirInitMask.generateInputMask(
      this.rs,
      this.nf,
      config.inputSparsity,
      rng,
    );
    this.WMask = ReservoirInitMask.generateReservoirMask(
      this.rs,
      config.reservoirSparsity,
      rng,
    );

    // init Win
    const wScale = config.weightInitScale;
    for (let i = 0; i < this.Win.length; i++) {
      this.Win[i] = this.WinMask[i] ? rng.normal(0, wScale) : 0;
    }

    // init W, and force diagonal = 0 for stability
    for (let i = 0; i < this.W.length; i++) {
      this.W[i] = this.WMask[i] ? rng.normal(0, wScale) : 0;
    }
    for (let i = 0; i < this.rs; i++) {
      const d = (i * this.rs + i) | 0;
      this.W[d] = 0;
      this.WMask[d] = 0;
    }

    // bias
    const bScale = config.biasScale;
    for (let i = 0; i < this.rs; i++) this.b[i] = rng.normal(0, bScale);

    // spectral radius scale (deterministic estimate; does not consume RNG)
    const scratch = new Float64Array(this.rs * 2);
    this.estimatedRadius = SpectralRadiusScaler.estimate(
      this.W,
      0,
      this.rs,
      this.WMask,
      0,
      scratch,
      0,
    );
    SpectralRadiusScaler.scale(
      this.W,
      0,
      this.rs,
      this.estimatedRadius,
      config.spectralRadius,
    );

    // state and scratch
    this.state = new Float64Array(this.rs);
    this.tmpPre = new Float64Array(this.rs);
    this.tmpIn = new Float64Array(this.rs);
    this.tmpRec = new Float64Array(this.rs);
  }

  /**
   * r_t = (1-leak)*r_{t-1} + leak*act( Win*(inputScale*x) + W*r_{t-1} + b )
   */
  update(xNorm: Float64Array, xOff: number): void {
    this.stepStateInPlace(this.state, xNorm, xOff);
  }

  /**
   * Allocation-free state step for an arbitrary state vector (used by predict scratch roll-forward).
   * NOTE: uses internal scratch buffers, but does not touch the live ring buffer.
   */
  stepStateInPlace(
    state: Float64Array,
    xNorm: Float64Array,
    xOff: number,
  ): void {
    const rs = this.rs;
    const nf = this.nf;
    const inScale = this.inputScale;
    const leak = this.leak;
    const oneMinus = 1 - leak;

    // tmpIn = Win * (inScale * x)
    let winIdx = 0;
    let mIdx = 0;
    for (let i = 0; i < rs; i++) {
      let sum = 0;
      const xBase = xOff | 0;
      for (let j = 0; j < nf; j++) {
        if (this.WinMask[mIdx++]) {
          sum += this.Win[winIdx] * (inScale * xNorm[xBase + j]);
        }
        winIdx++;
      }
      this.tmpIn[i] = sum;
    }

    // tmpRec = W * state
    TensorOps.sparseMatVec(
      this.W,
      0,
      rs,
      rs,
      this.WMask,
      0,
      state,
      0,
      this.tmpRec,
      0,
    );

    // tmpPre = act(tmpIn + tmpRec + b)
    for (let i = 0; i < rs; i++) {
      const pre = this.tmpIn[i] + this.tmpRec[i] + this.b[i];
      const a = ActivationOps.apply(this.act, pre);
      this.tmpPre[i] = isFiniteNumber(a) ? a : 0;
    }

    // leaky integration
    for (let i = 0; i < rs; i++) {
      const v = oneMinus * state[i] + leak * this.tmpPre[i];
      state[i] = isFiniteNumber(v) ? v : 0;
    }
  }

  getState(): Float64Array {
    return this.state;
  }

  copyStateTo(out: Float64Array, outOff: number): void {
    TensorOps.vecCopy(this.state, 0, out, outOff, this.rs);
  }

  setStateFrom(src: Float64Array, srcOff: number): void {
    TensorOps.vecCopy(src, srcOff, this.state, 0, this.rs);
  }

  resetState(): void {
    this.state.fill(0);
  }

  getParams(): ESNReservoirParams {
    return {
      Win: this.Win,
      W: this.W,
      b: this.b,
      WinMask: this.WinMask,
      WMask: this.WMask,
    };
  }

  getEstimatedSpectralRadiusBeforeScaling(): number {
    return this.estimatedRadius;
  }

  serialize(): {
    state: number[];
    Win: number[];
    W: number[];
    b: number[];
    WinMask: number[];
    WMask: number[];
    estimatedRadius: number;
  } {
    return {
      state: Array.from(this.state),
      Win: Array.from(this.Win),
      W: Array.from(this.W),
      b: Array.from(this.b),
      WinMask: Array.from(this.WinMask),
      WMask: Array.from(this.WMask),
      estimatedRadius: this.estimatedRadius,
    };
  }

  deserialize(
    d: {
      state: number[];
      Win: number[];
      W: number[];
      b: number[];
      WinMask: number[];
      WMask: number[];
      estimatedRadius?: number;
    },
  ): void {
    for (let i = 0; i < this.state.length; i++) this.state[i] = d.state[i] || 0;
    for (let i = 0; i < this.Win.length; i++) this.Win[i] = d.Win[i] || 0;
    for (let i = 0; i < this.W.length; i++) this.W[i] = d.W[i] || 0;
    for (let i = 0; i < this.b.length; i++) this.b[i] = d.b[i] || 0;
    for (let i = 0; i < this.WinMask.length; i++) {
      this.WinMask[i] = (d.WinMask[i] || 0) as any;
    }
    for (let i = 0; i < this.WMask.length; i++) {
      this.WMask[i] = (d.WMask[i] || 0) as any;
    }
    this.estimatedRadius = typeof d.estimatedRadius === "number"
      ? d.estimatedRadius
      : 0;
  }
}

// ============================================================================
// 3. READOUT TRAINING: RLSState, RLSOptimizer (shared P), + LinearReadout
// ============================================================================

export interface ReadoutConfig {
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
}

export interface ReadoutParams {
  Wout: Float64Array; // [nTargets x zDim]
}

export interface RLSState {
  P: Float64Array; // [zDim x zDim]
  gain: Float64Array; // [zDim]
  Pz: Float64Array; // [zDim]
  zTP: Float64Array; // [zDim]
}

class RLSOptimizer {
  private zDim: number;
  private nTargets: number;
  private lambda: number;
  private delta: number;
  private l2: number;
  private eps: number;
  private clip: number;

  private P: Float64Array; // [zDim x zDim]
  private gain: Float64Array;
  private Pz: Float64Array;
  private zTP: Float64Array;

  private Wout: Float64Array; // [nTargets x zDim]

  constructor(zDim: number, nTargets: number, config: ESNRegressionConfig) {
    this.zDim = zDim | 0;
    this.nTargets = nTargets | 0;
    this.lambda = config.rlsLambda;
    this.delta = config.rlsDelta;
    this.l2 = config.l2Lambda;
    this.eps = config.epsilon;
    this.clip = config.gradientClipNorm;

    this.P = new Float64Array(this.zDim * this.zDim);
    this.gain = new Float64Array(this.zDim);
    this.Pz = new Float64Array(this.zDim);
    this.zTP = new Float64Array(this.zDim);
    this.Wout = new Float64Array(this.nTargets * this.zDim);

    this.reset();
  }

  reset(): void {
    this.P.fill(0);
    const d = this.delta;
    // Use ridge-style initialization: P0 = (1/delta) * I (delta is scale of prior covariance).
    const diag = d > 0 ? 1.0 / d : 1.0;
    for (let i = 0; i < this.zDim; i++) this.P[i * this.zDim + i] = diag;
    this.Wout.fill(0);
  }

  getParams(): ReadoutParams {
    return { Wout: this.Wout };
  }

  /**
   * Predict y_hat = Wout * z
   */
  predict(
    z: Float64Array,
    zOff: number,
    out: Float64Array,
    outOff: number,
  ): void {
    TensorOps.matVec(
      this.Wout,
      0,
      this.nTargets,
      this.zDim,
      z,
      zOff,
      out,
      outOff,
    );
  }

  /**
   * Weighted RLS update with shared P and gain across output dims.
   * Implements sample weight w via z' = sqrt(w)*z and y' = sqrt(w)*y.
   *
   * Pz = P * z'
   * denom = lambda + z'^T * P * z'
   * k = Pz / denom
   * e = y' - Wout*z'
   * Wout += e * k^T  (for each target row)
   * P = (P - k*(z'^T P)) / lambda
   *
   * Includes deterministic gradient clipping on error RMS before applying updates.
   * L2 regularization applied as stable shrink: Wout *= 1/(1+l2) per update.
   */
  update(
    z: Float64Array,
    zOff: number,
    y: Float64Array,
    yOff: number,
    weight: number,
  ): number {
    const zDim = this.zDim;
    const nT = this.nTargets;

    // sqrt-weighted z'
    const w = weight > 0 ? weight : 0;
    const wS = w > 0 ? Math.sqrt(w) : 0;

    // Compute Pz = P * z'  (z' = wS * z)
    // Avoid allocating z'; scale during multiply: Pz = P*(wS*z) = wS*(P*z)
    TensorOps.matVec(this.P, 0, zDim, zDim, z, zOff, this.Pz, 0);
    if (wS !== 1.0) {
      for (let i = 0; i < zDim; i++) this.Pz[i] *= wS;
    }

    // denom = lambda + z'^T * P * z' = lambda + (wS*z)^T * Pz
    let denom = this.lambda;
    if (wS !== 0) {
      let zi = zOff | 0;
      for (let i = 0; i < zDim; i++) {
        denom += (wS * z[zi++]) * this.Pz[i];
      }
    }
    if (denom < this.eps) denom = this.eps;

    // gain = k = Pz / denom
    const invDen = 1.0 / denom;
    for (let i = 0; i < zDim; i++) this.gain[i] = this.Pz[i] * invDen;

    // Compute predictions and errors (using z')
    // yHat = Wout * z' = Wout * (wS*z)
    // e = y' - yHat = wS*y - yHat
    // Clip based on RMS of unscaled error (in target units): (y - Wout*z)
    // For stability and determinism: compute both in one pass.

    // First compute yHatUnscaled = Wout * z (unscaled)
    // Then errorUnscaled = y - yHatUnscaled
    // Then apply scaling wS to error for weighted update: e = wS * errorUnscaled
    let errRmsSq = 0;
    for (let t = 0; t < nT; t++) {
      const row = t * zDim;
      let yHat = 0;
      let zi = zOff | 0;
      for (let i = 0; i < zDim; i++) yHat += this.Wout[row + i] * z[zi++];
      const eUn = y[(yOff + t) | 0] - yHat;
      errRmsSq += eUn * eUn;
    }
    const errRms = Math.sqrt(errRmsSq / nT);

    let clipScale = 1.0;
    if (this.clip > 0 && errRms > this.clip) clipScale = this.clip / errRms;

    // Apply updates: Wout += (wS * clipScale * errorUnscaled) * gain^T
    const eScale = wS * clipScale;

    for (let t = 0; t < nT; t++) {
      const row = t * zDim;
      // recompute yHat unscaled for determinism and no extra buffers
      let yHat = 0;
      let zi = zOff | 0;
      for (let i = 0; i < zDim; i++) yHat += this.Wout[row + i] * z[zi++];
      const eUn = y[(yOff + t) | 0] - yHat;
      const e = eScale * eUn;

      // Wout_row += gain * e
      for (let i = 0; i < zDim; i++) this.Wout[row + i] += this.gain[i] * e;
    }

    // Compute zTP = z'^T * P  (vector length zDim)
    // zTP[j] = sum_i z'_i * P[i,j]
    // z'_i = wS * z_i
    if (wS !== 0) {
      // Compute using columns, no allocations
      for (let j = 0; j < zDim; j++) {
        let sum = 0;
        let zi = zOff | 0;
        let pIdx = j | 0; // P[i*zDim + j]
        for (let i = 0; i < zDim; i++) {
          sum += (wS * z[zi++]) * this.P[pIdx];
          pIdx += zDim;
        }
        this.zTP[j] = sum;
      }
    } else {
      this.zTP.fill(0);
    }

    // P = (P - k * zTP) / lambda
    const invLam = 1.0 / this.lambda;
    for (let i = 0; i < zDim; i++) {
      const ki = this.gain[i];
      let pRow = (i * zDim) | 0;
      for (let j = 0; j < zDim; j++) {
        this.P[pRow + j] = invLam * (this.P[pRow + j] - ki * this.zTP[j]);
      }
    }

    // Ensure symmetry (numerically stabilizing). Cost: O(zDim^2), but same order as P update.
    // P = 0.5*(P + P^T)
    for (let i = 0; i < zDim; i++) {
      const ii = i * zDim;
      for (let j = i + 1; j < zDim; j++) {
        const a = this.P[ii + j];
        const b = this.P[j * zDim + i];
        const m = 0.5 * (a + b);
        this.P[ii + j] = m;
        this.P[j * zDim + i] = m;
      }
    }

    // L2 shrink (stable, deterministic)
    if (this.l2 > 0) {
      const shrink = 1.0 / (1.0 + this.l2);
      for (let i = 0; i < this.Wout.length; i++) this.Wout[i] *= shrink;
    }

    return errRms;
  }

  serialize(): { P: number[]; Wout: number[] } {
    return { P: Array.from(this.P), Wout: Array.from(this.Wout) };
  }

  deserialize(d: { P: number[]; Wout: number[] }): void {
    const p = d.P;
    const w = d.Wout;
    for (let i = 0; i < this.P.length; i++) this.P[i] = p[i] || 0;
    for (let i = 0; i < this.Wout.length; i++) this.Wout[i] = w[i] || 0;
  }

  getZDim(): number {
    return this.zDim;
  }

  getNTargets(): number {
    return this.nTargets;
  }

  getWout(): Float64Array {
    return this.Wout;
  }
}

class LinearReadout {
  private rs: number;
  private nf: number;
  private nT: number;
  private useX: boolean;
  private useB: boolean;
  private zDim: number;

  private rls: RLSOptimizer;

  private z: Float64Array; // scratch
  constructor(
    reservoirSize: number,
    nFeatures: number,
    nTargets: number,
    config: ESNRegressionConfig,
  ) {
    this.rs = reservoirSize | 0;
    this.nf = nFeatures | 0;
    this.nT = nTargets | 0;
    this.useX = !!config.useInputInReadout;
    this.useB = !!config.useBiasInReadout;

    let zDim = this.rs;
    if (this.useX) zDim += this.nf;
    if (this.useB) zDim += 1;
    this.zDim = zDim;

    this.rls = new RLSOptimizer(this.zDim, this.nT, config);
    this.z = new Float64Array(this.zDim);
  }

  buildZ(
    r: Float64Array,
    rOff: number,
    x: Float64Array,
    xOff: number,
    out: Float64Array,
    outOff: number,
  ): void {
    let k = outOff | 0;

    // r
    let ir = rOff | 0;
    for (let i = 0; i < this.rs; i++) out[k++] = r[ir++];

    // x
    if (this.useX) {
      let ix = xOff | 0;
      for (let i = 0; i < this.nf; i++) out[k++] = x[ix++];
    }

    // bias
    if (this.useB) out[k] = 1.0;
  }

  forward(
    r: Float64Array,
    rOff: number,
    x: Float64Array,
    xOff: number,
    out: Float64Array,
    outOff: number,
  ): void {
    this.buildZ(r, rOff, x, xOff, this.z, 0);
    this.rls.predict(this.z, 0, out, outOff);
  }

  train(
    r: Float64Array,
    rOff: number,
    x: Float64Array,
    xOff: number,
    y: Float64Array,
    yOff: number,
    weight: number,
  ): number {
    this.buildZ(r, rOff, x, xOff, this.z, 0);
    return this.rls.update(this.z, 0, y, yOff, weight);
  }

  getZDim(): number {
    return this.zDim;
  }

  getWout(): Float64Array {
    return this.rls.getWout();
  }

  getRLS(): RLSOptimizer {
    return this.rls;
  }

  reset(): void {
    this.rls.reset();
  }

  serialize(): { rls: { P: number[]; Wout: number[] } } {
    return { rls: this.rls.serialize() };
  }

  deserialize(d: { rls: { P: number[]; Wout: number[] } }): void {
    this.rls.deserialize(d.rls);
  }
}

// ============================================================================
// 7. Serialization helper
// ============================================================================

class SerializationHelper {
  static serialize(model: ESNRegression): string {
    return JSON.stringify(model.getSerializableState());
  }

  static deserialize(model: ESNRegression, s: string): void {
    const obj = JSON.parse(s);
    model.setSerializableState(obj);
  }
}

// ============================================================================
// 8. ESNRegression Public API
// ============================================================================

export class ESNRegression {
  private config: ESNRegressionConfig;
  private initialized = false;

  private nFeatures = 0;
  private nTargets = 0;
  private sampleCount = 0;

  private rng: RandomGenerator;

  private ring: RingBuffer | null = null;
  private normalizer: WelfordNormalizer | null = null;
  private reservoir: ESNReservoir | null = null;
  private readout: LinearReadout | null = null;
  private residualStats: ResidualStatsTracker | null = null;
  private outlier: OutlierDownweighter | null = null;
  private metrics: MetricsAccumulator | null = null;

  // scratch
  private xRaw: Float64Array | null = null;
  private xNorm: Float64Array | null = null;
  private y: Float64Array | null = null;
  private yHat: Float64Array | null = null;
  private resid: Float64Array | null = null;
  private residStds: Float64Array | null = null;
  private scratchR: Float64Array | null = null;

  // reusable results
  private fitResultObj: FitResult;
  private predResultObj: PredictionResult;

  constructor(config: Partial<ESNRegressionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.rng = new RandomGenerator(this.config.seed);

    this.fitResultObj = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1,
    };

    const maxSteps = this.config.maxSequenceLength | 0;
    this.predResultObj = {
      predictions: [],
      lowerBounds: [],
      upperBounds: [],
      confidence: 0,
    };
    for (let i = 0; i < maxSteps; i++) {
      this.predResultObj.predictions.push([]);
      this.predResultObj.lowerBounds.push([]);
      this.predResultObj.upperBounds.push([]);
    }
  }

  private initialize(nFeatures: number, nTargets: number): void {
    this.nFeatures = nFeatures | 0;
    this.nTargets = nTargets | 0;

    this.ring = new RingBuffer(this.config.maxSequenceLength, this.nFeatures);
    this.normalizer = new WelfordNormalizer(
      this.nFeatures,
      this.config.normalizationEpsilon,
      this.config.normalizationWarmup,
    );
    this.reservoir = new ESNReservoir(
      this.config.reservoirSize,
      this.nFeatures,
      this.config,
      this.rng,
    );
    this.readout = new LinearReadout(
      this.config.reservoirSize,
      this.nFeatures,
      this.nTargets,
      this.config,
    );
    this.residualStats = new ResidualStatsTracker(
      this.config.residualWindowSize,
      this.nTargets,
    );
    this.outlier = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );
    this.metrics = new MetricsAccumulator();

    this.xRaw = new Float64Array(this.nFeatures);
    this.xNorm = new Float64Array(this.nFeatures);
    this.y = new Float64Array(this.nTargets);
    this.yHat = new Float64Array(this.nTargets);
    this.resid = new Float64Array(this.nTargets);
    this.residStds = new Float64Array(this.nTargets);
    this.scratchR = new Float64Array(this.config.reservoirSize);

    // preallocate output arrays (no per-call allocations)
    const maxSteps = this.config.maxSequenceLength | 0;
    for (let i = 0; i < maxSteps; i++) {
      this.predResultObj.predictions[i] = new Array<number>(this.nTargets);
      this.predResultObj.lowerBounds[i] = new Array<number>(this.nTargets);
      this.predResultObj.upperBounds[i] = new Array<number>(this.nTargets);
      for (let t = 0; t < this.nTargets; t++) {
        this.predResultObj.predictions[i][t] = 0;
        this.predResultObj.lowerBounds[i][t] = 0;
        this.predResultObj.upperBounds[i][t] = 0;
      }
    }

    this.initialized = true;
  }

  /**
   * Online training.
   *
   * Critical enforced behavior:
   * - xCoordinates.length MUST equal yCoordinates.length else throw BEFORE ingestion.
   * - For each sample: push X into ring buffer FIRST, then all training steps use internal buffers.
   *
   * @param param0 xCoordinates: [N x nFeatures], yCoordinates: [N x nTargets]
   * @returns FitResult (reused object; caller must copy if needed)
   */
  fitOnline(
    { xCoordinates, yCoordinates }: {
      xCoordinates: number[][];
      yCoordinates: number[][];
    },
  ): FitResult {
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `fitOnline: xCoordinates.length (${xCoordinates.length}) must equal yCoordinates.length (${yCoordinates.length})`,
      );
    }

    const N = xCoordinates.length | 0;
    if (N === 0) {
      this.fitResultObj.samplesProcessed = 0;
      this.fitResultObj.averageLoss = 0;
      this.fitResultObj.gradientNorm = 0;
      this.fitResultObj.driftDetected = false;
      this.fitResultObj.sampleWeight = 1;
      return this.fitResultObj;
    }

    if (!this.initialized) {
      const nf = xCoordinates[0].length | 0;
      const nt = yCoordinates[0].length | 0;
      if (nf <= 0) throw new Error("fitOnline: nFeatures must be > 0");
      if (nt <= 0) throw new Error("fitOnline: nTargets must be > 0");
      this.initialize(nf, nt);
    }

    // Strict shape validation before processing
    for (let i = 0; i < N; i++) {
      if (xCoordinates[i].length !== this.nFeatures) {
        throw new Error(
          `fitOnline: xCoordinates[${i}].length (${
            xCoordinates[i].length
          }) must equal nFeatures (${this.nFeatures})`,
        );
      }
      if (yCoordinates[i].length !== this.nTargets) {
        throw new Error(
          `fitOnline: yCoordinates[${i}].length (${
            yCoordinates[i].length
          }) must equal nTargets (${this.nTargets})`,
        );
      }
    }

    const ring = this.ring!;
    const norm = this.normalizer!;
    const res = this.reservoir!;
    const head = this.readout!;
    const rStats = this.residualStats!;
    const outlier = this.outlier!;
    const metrics = this.metrics!;

    const xRaw = this.xRaw!;
    const xNorm = this.xNorm!;
    const y = this.y!;
    const yHat = this.yHat!;
    const resid = this.resid!;
    const stds = this.residStds!;

    metrics.reset();

    for (let i = 0; i < N; i++) {
      // 1) Push X FIRST (authoritative latest-X)
      const xr = xCoordinates[i];
      for (let j = 0; j < this.nFeatures; j++) xRaw[j] = xr[j];
      ring.push(xRaw, 0);

      // 2) Update normalization stats
      norm.update(xRaw, 0);

      // 3) Normalize x
      norm.normalize(xRaw, 0, xNorm, 0);

      // 4) Reservoir update (live)
      res.update(xNorm, 0);

      // 5) Copy target
      const yr = yCoordinates[i];
      for (let t = 0; t < this.nTargets; t++) y[t] = yr[t];

      // 6) Forward (before update)
      head.forward(res.getState(), 0, xNorm, 0, yHat, 0);

      // 7) residuals (pred - y), for stats/loss/outliers
      LossFunction.residuals(yHat, 0, y, 0, resid, 0, this.nTargets);

      // 8) outlier stds
      rStats.getStds(stds, 0);

      // 9) sample weight
      const w = outlier.computeWeight(resid, 0, stds, 0, this.nTargets);

      // 10) RLS update
      const gradNorm = head.train(res.getState(), 0, xNorm, 0, y, 0, w);

      // 11) loss
      const loss = LossFunction.mse(yHat, 0, y, 0, this.nTargets);

      // 12) update residual stats
      rStats.update(resid, 0);

      // 13) metrics
      metrics.update(loss, gradNorm, w);

      this.sampleCount++;
    }

    this.fitResultObj.samplesProcessed = N;
    this.fitResultObj.averageLoss = metrics.avgLoss();
    this.fitResultObj.gradientNorm = metrics.avgGrad();
    this.fitResultObj.driftDetected = false;
    this.fitResultObj.sampleWeight = metrics.getLastWeight();
    return this.fitResultObj;
  }

  /**
   * Multi-step roll-forward prediction using scratch reservoir state.
   * Uses internal RingBuffer for the latest X (caller does NOT pass latest-X).
   *
   * @param futureSteps integer >= 1 and <= maxSequenceLength
   * @returns PredictionResult (reused object; caller must copy if needed)
   */
  predict(futureSteps: number): PredictionResult {
    if (
      !this.initialized || !this.ring || !this.normalizer || !this.reservoir ||
      !this.readout || !this.residualStats
    ) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }
    if (!Number.isInteger(futureSteps) || futureSteps < 1) {
      throw new Error("predict: futureSteps must be a positive integer >= 1");
    }
    if (futureSteps > this.config.maxSequenceLength) {
      throw new Error(
        `predict: futureSteps (${futureSteps}) exceeds maxSequenceLength (${this.config.maxSequenceLength})`,
      );
    }
    if (this.ring.isEmpty()) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }

    const ring = this.ring;
    const norm = this.normalizer;
    const res = this.reservoir;
    const head = this.readout;
    const rStats = this.residualStats;

    const xRaw = this.xRaw!;
    const xNorm = this.xNorm!;
    const yHat = this.yHat!;
    const stds = this.residStds!;
    const scratchR = this.scratchR!;
    const out = this.predResultObj;

    // context ends at most recent ingested X
    ring.getLatest(xRaw, 0);

    // scratch copy of reservoir state (do not mutate live state in predict)
    res.copyStateTo(scratchR, 0);

    // base residual stds for 1-step
    rStats.getStds(stds, 0);

    // roll-forward deterministically
    const mode = this.config.rollforwardMode;
    const canAR = mode === "autoregressive" && this.nFeatures === this.nTargets;

    for (let step = 0; step < futureSteps; step++) {
      // Normalize x for this step using running stats
      norm.normalize(xRaw, 0, xNorm, 0);

      // Advance scratch reservoir one step
      res.stepStateInPlace(scratchR, xNorm, 0);

      // Predict
      head.forward(scratchR, 0, xNorm, 0, yHat, 0);

      // Store prediction and bounds
      const horizonScale = Math.sqrt(step + 1); // sigma_k = sigma_1 * sqrt(k+1)
      for (let t = 0; t < this.nTargets; t++) {
        const pred = yHat[t];
        out.predictions[step][t] = pred;

        const sigma1 = stds[t];
        const sigma = sigma1 * horizonScale;
        const half = this.config.uncertaintyMultiplier * sigma;

        out.lowerBounds[step][t] = pred - half;
        out.upperBounds[step][t] = pred + half;
      }

      // Next xRaw (default holdLastX: keep constant)
      if (canAR) {
        // Deterministic autoregressive roll-forward:
        // Set x_{t+k} := y_hat_{t+k-1} in raw space, then normalize next loop.
        for (let j = 0; j < this.nFeatures; j++) {
          const v = yHat[j];
          xRaw[j] = isFiniteNumber(v) ? v : 0;
        }
      }
    }

    // confidence: deterministic finite clamp [0,1]
    let avgStd = 0;
    for (let t = 0; t < this.nTargets; t++) avgStd += stds[t];
    avgStd = this.nTargets > 0 ? avgStd / this.nTargets : 0;

    const base = avgStd > 0 ? 1.0 / (1.0 + avgStd) : 1.0;
    const horizonPenalty = 1.0 / Math.sqrt(futureSteps);
    out.confidence = clamp01(base * horizonPenalty);

    return out;
  }

  /**
   * @returns ModelSummary
   */
  getModelSummary(): ModelSummary {
    const zDim = this.readout ? this.readout.getZDim() : 0;
    const totalParams = (this.nTargets | 0) * (zDim | 0);
    return {
      totalParameters: totalParams,
      receptiveField: this.config.maxSequenceLength,
      spectralRadius: this.config.spectralRadius,
      reservoirSize: this.config.reservoirSize,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      maxSequenceLength: this.config.maxSequenceLength,
      sampleCount: this.sampleCount,
    };
  }

  /**
   * @returns WeightInfo (allocates for serialization/export; not hot path)
   */
  getWeights(): WeightInfo {
    const weights: Array<{ name: string; shape: number[]; values: number[] }> =
      [];

    if (this.readout) {
      const Wout = this.readout.getWout();
      weights.push({
        name: "Wout",
        shape: [this.nTargets, this.readout.getZDim()],
        values: Array.from(Wout),
      });
    }

    if (this.reservoir) {
      const p = this.reservoir.getParams();
      weights.push({
        name: "Win",
        shape: [this.config.reservoirSize, this.nFeatures],
        values: Array.from(p.Win),
      });
      weights.push({
        name: "W",
        shape: [this.config.reservoirSize, this.config.reservoirSize],
        values: Array.from(p.W),
      });
      weights.push({
        name: "b",
        shape: [this.config.reservoirSize],
        values: Array.from(p.b),
      });
    }

    return { weights };
  }

  /**
   * @returns NormalizationStats
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.normalizer) {
      return { means: [], stds: [], count: 0, isActive: false };
    }
    return {
      means: this.normalizer.getMeans(),
      stds: this.normalizer.getStds(),
      count: this.normalizer.getCount(),
      isActive: this.normalizer.isActive(),
    };
  }

  /**
   * Reset model to a clean state with the same config.
   */
  reset(): void {
    if (this.reservoir) this.reservoir.resetState();
    if (this.readout) this.readout.reset();
    if (this.normalizer) this.normalizer.reset();
    if (this.ring) this.ring.reset();
    if (this.residualStats) this.residualStats.reset();
    if (this.metrics) this.metrics.reset();
    this.sampleCount = 0;
    this.rng = new RandomGenerator(this.config.seed);
  }

  /**
   * Save full model state as JSON string.
   */
  save(): string {
    return SerializationHelper.serialize(this);
  }

  /**
   * Load full model state from JSON string.
   */
  load(s: string): void {
    SerializationHelper.deserialize(this, s);
  }

  // Internal serialization hooks (no public contract beyond save/load)
  getSerializableState(): object {
    return {
      config: this.config,
      initialized: this.initialized,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      sampleCount: this.sampleCount,
      rngState: this.rng.getState(),
      ring: this.ring ? this.ring.serialize() : null,
      normalizer: this.normalizer ? this.normalizer.serialize() : null,
      reservoir: this.reservoir ? this.reservoir.serialize() : null,
      readout: this.readout ? this.readout.serialize() : null,
      residualStats: this.residualStats ? this.residualStats.serialize() : null,
    };
  }

  setSerializableState(state: any): void {
    this.config = { ...DEFAULT_CONFIG, ...(state.config || {}) };
    this.initialized = !!state.initialized;
    this.nFeatures = state.nFeatures | 0;
    this.nTargets = state.nTargets | 0;
    this.sampleCount = state.sampleCount | 0;
    this.rng = new RandomGenerator(this.config.seed);
    if (typeof state.rngState === "number") this.rng.setState(state.rngState);

    if (this.initialized && this.nFeatures > 0 && this.nTargets > 0) {
      // Recreate all components deterministically, then overwrite arrays with saved values
      this.initialize(this.nFeatures, this.nTargets);

      if (state.ring) this.ring!.deserialize(state.ring);
      if (state.normalizer) this.normalizer!.deserialize(state.normalizer);
      if (state.reservoir) this.reservoir!.deserialize(state.reservoir);
      if (state.readout) this.readout!.deserialize(state.readout);
      if (state.residualStats) {
        this.residualStats!.deserialize(state.residualStats);
      }
    }
  }
}

export default ESNRegression;
