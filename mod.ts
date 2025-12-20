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
  readoutTraining: "rls";
  rlsLambda: number;
  rlsDelta: number;
  epsilon: number;
  l2Lambda: number;
  gradientClipNorm: number;
  normalizationEpsilon: number;
  normalizationWarmup: number;
  outlierThreshold: number;
  outlierMinWeight: number;
  residualWindowSize: number;
  uncertaintyMultiplier: number;
  weightInitScale: number;
  seed: number;
  verbose: boolean;
  rollforwardMode: "holdLastX" | "autoregressive";
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
// DEFAULT CONFIGURATION
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
// UTILITY FUNCTIONS
// ============================================================================

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function isFiniteNum(x: number): boolean {
  return Number.isFinite(x);
}

function safeVal(x: number, fallback: number = 0): number {
  return isFiniteNum(x) ? x : fallback;
}

// ============================================================================
// 1. MEMORY INFRA
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
      if (v <= 0) throw new Error("TensorShape: dims must be positive");
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
    const end = this.offset + this.shape.size;
    for (let i = this.offset; i < end; i++) this.data[i] = v;
  }

  copyFrom(src: TensorView): void {
    const n = this.shape.size;
    for (let i = 0; i < n; i++) {
      this.data[this.offset + i] = src.data[src.offset + i];
    }
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
    for (let i = 0; i < (n | 0); i++) {
      dst[(dstOff + i) | 0] = src[(srcOff + i) | 0];
    }
  }

  static vecFill(
    dst: Float64Array,
    dstOff: number,
    v: number,
    n: number,
  ): void {
    for (let i = 0; i < (n | 0); i++) dst[(dstOff + i) | 0] = v;
  }

  static dot(
    a: Float64Array,
    aOff: number,
    b: Float64Array,
    bOff: number,
    n: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < (n | 0); i++) {
      sum += a[(aOff + i) | 0] * b[(bOff + i) | 0];
    }
    return sum;
  }

  static norm2(a: Float64Array, aOff: number, n: number): number {
    let sum = 0;
    for (let i = 0; i < (n | 0); i++) {
      const v = a[(aOff + i) | 0];
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

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
    const r = rows | 0, c = cols | 0;
    for (let i = 0; i < r; i++) {
      let sum = 0;
      const rowStart = (aOff + i * c) | 0;
      for (let j = 0; j < c; j++) sum += A[rowStart + j] * x[(xOff + j) | 0];
      y[(yOff + i) | 0] = sum;
    }
  }

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
    const r = rows | 0, c = cols | 0;
    for (let i = 0; i < r; i++) {
      let sum = 0;
      const rowStart = (aOff + i * c) | 0;
      const maskStart = (mOff + i * c) | 0;
      for (let j = 0; j < c; j++) {
        if (mask[maskStart + j]) sum += A[rowStart + j] * x[(xOff + j) | 0];
      }
      y[(yOff + i) | 0] = sum;
    }
  }
}

// ============================================================================
// 2. NUMERICS
// ============================================================================

class ActivationOps {
  static apply(act: "tanh" | "relu", x: number): number {
    if (act === "tanh") return Math.tanh(x);
    return x > 0 ? x : 0;
  }

  static derivative(act: "tanh" | "relu", y: number): number {
    if (act === "tanh") return 1 - y * y;
    return y > 0 ? 1 : 0;
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
    const u1 = this.uniform() + 1e-12;
    const u2 = this.uniform();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
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
    if (!isFiniteNum(x)) return;
    this.count++;
    const d = x - this.mean;
    this.mean += d / this.count;
    const d2 = x - this.mean;
    this.m2 += d * d2;
  }

  variance(): number {
    return this.count < 2 ? 0 : Math.max(0, this.m2 / (this.count - 1));
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
  private cachedStds: Float64Array;
  private stdsDirty: boolean = true;

  constructor(nFeatures: number, eps: number, warmup: number) {
    this.nFeatures = nFeatures | 0;
    this.counts = new Float64Array(this.nFeatures);
    this.means = new Float64Array(this.nFeatures);
    this.m2s = new Float64Array(this.nFeatures);
    this.cachedStds = new Float64Array(this.nFeatures);
    this.eps = eps;
    this.warmup = warmup | 0;
  }

  update(x: Float64Array, xOff: number): void {
    for (let j = 0; j < this.nFeatures; j++) {
      const v = x[(xOff + j) | 0];
      if (!isFiniteNum(v)) continue;
      const c = ++this.counts[j];
      const d = v - this.means[j];
      this.means[j] += d / c;
      const d2 = v - this.means[j];
      this.m2s[j] += d * d2;
    }
    this.stdsDirty = true;
  }

  private updateCachedStds(): void {
    if (!this.stdsDirty) return;
    for (let j = 0; j < this.nFeatures; j++) {
      const c = this.counts[j];
      const varr = c > 1 ? Math.max(0, this.m2s[j] / (c - 1)) : 0;
      this.cachedStds[j] = Math.sqrt(varr);
    }
    this.stdsDirty = false;
  }

  normalize(
    x: Float64Array,
    xOff: number,
    out: Float64Array,
    outOff: number,
  ): void {
    this.updateCachedStds();
    for (let j = 0; j < this.nFeatures; j++) {
      const mean = this.means[j];
      const std = this.cachedStds[j];
      const denom = std > this.eps ? std : this.eps;
      const v = (x[(xOff + j) | 0] - mean) / denom;
      out[(outOff + j) | 0] = safeVal(v);
    }
  }

  denormalize(
    xNorm: Float64Array,
    xNormOff: number,
    out: Float64Array,
    outOff: number,
  ): void {
    this.updateCachedStds();
    for (let j = 0; j < this.nFeatures; j++) {
      const mean = this.means[j];
      const std = this.cachedStds[j];
      const denom = std > this.eps ? std : this.eps;
      const v = xNorm[(xNormOff + j) | 0] * denom + mean;
      out[(outOff + j) | 0] = safeVal(v);
    }
  }

  isActive(): boolean {
    return this.nFeatures > 0 && this.counts[0] >= this.warmup;
  }

  getCount(): number {
    return this.nFeatures > 0 ? this.counts[0] : 0;
  }

  getMeans(): number[] {
    return Array.from(this.means);
  }

  getStds(): number[] {
    this.updateCachedStds();
    return Array.from(this.cachedStds);
  }

  reset(): void {
    this.counts.fill(0);
    this.means.fill(0);
    this.m2s.fill(0);
    this.cachedStds.fill(0);
    this.stdsDirty = true;
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
    this.stdsDirty = true;
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
    for (let i = 0; i < (n | 0); i++) {
      const d = pred[(pOff + i) | 0] - y[(yOff + i) | 0];
      sum += d * d;
    }
    return sum / Math.max(1, n | 0);
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
    for (let i = 0; i < (n | 0); i++) {
      out[(oOff + i) | 0] = pred[(pOff + i) | 0] - y[(yOff + i) | 0];
    }
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
    if (isFiniteNum(loss)) this.lossSum += loss;
    if (isFiniteNum(grad)) this.gradSum += grad;
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
// 6. TRAINING UTILITIES
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
      for (let j = 0; j < this.nFeat; j++) {
        this.buf[base + j] = row[(rowOff + j) | 0];
      }
    } else {
      for (let j = 0; j < this.nFeat; j++) this.buf[base + j] = row[j];
    }
    this.head = (this.head + 1) % this.cap;
    if (this.count < this.cap) this.count++;
  }

  getLatest(out: Float64Array, outOff: number): boolean {
    if (this.count === 0) return false;
    const idx = (((this.head - 1 + this.cap) % this.cap) * this.nFeat) | 0;
    for (let j = 0; j < this.nFeat; j++) {
      out[(outOff + j) | 0] = this.buf[idx + j];
    }
    return true;
  }

  getAtOffset(offset: number, out: Float64Array, outOff: number): boolean {
    if (offset < 0 || offset >= this.count) return false;
    const idx =
      (((this.head - 1 - offset + this.cap * 2) % this.cap) * this.nFeat) | 0;
    for (let j = 0; j < this.nFeat; j++) {
      out[(outOff + j) | 0] = this.buf[idx + j];
    }
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
    for (let i = 0; i < this.buf.length; i++) this.buf[i] = d.buffer[i] || 0;
    this.head = d.head | 0;
    this.count = d.count | 0;
  }
}

class ResidualStatsTracker {
  private win: number;
  private nT: number;
  private buf: Float64Array;
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
    for (let t = 0; t < this.nT; t++) {
      const r = residuals[(off + t) | 0];
      if (!isFiniteNum(r)) continue;
      const h = this.head[t] | 0;
      const base = (t * this.win) | 0;
      const idx = base + h;

      if (this.count[t] >= this.win) {
        const old = this.buf[idx];
        this.sum[t] -= old;
        this.sumSq[t] -= old * old;
      } else {
        this.count[t]++;
      }

      this.buf[idx] = r;
      this.sum[t] += r;
      this.sumSq[t] += r * r;
      this.head[t] = ((h + 1) % this.win) | 0;
    }
  }

  getStd(t: number): number {
    const n = this.count[t] | 0;
    if (n < 2) return 1.0;
    const mean = this.sum[t] / n;
    const varr = Math.max(0, this.sumSq[t] / n - mean * mean);
    const std = Math.sqrt(varr);
    return std > 1e-12 ? std : 1e-12;
  }

  getMean(t: number): number {
    const n = this.count[t] | 0;
    return n > 0 ? this.sum[t] / n : 0;
  }

  getStds(out: Float64Array, off: number): void {
    for (let t = 0; t < this.nT; t++) out[(off + t) | 0] = this.getStd(t);
  }

  getTotalCount(): number {
    return this.nT > 0 ? this.count[0] : 0;
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

  deserialize(d: {
    buffers: number[];
    heads: number[];
    counts: number[];
    sums: number[];
    sumSqs: number[];
  }): void {
    for (let i = 0; i < this.buf.length; i++) this.buf[i] = d.buffers[i] || 0;
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
    for (let t = 0; t < (nTargets | 0); t++) {
      const std = stds[(sOff + t) | 0];
      if (std > 1e-12) {
        const z = Math.abs(residuals[(rOff + t) | 0]) / std;
        if (z > maxZ) maxZ = z;
      }
    }
    if (maxZ <= this.thr) return 1.0;
    const w = 1.0 / (1.0 + (maxZ - this.thr) * (maxZ - this.thr));
    return Math.max(this.minW, w);
  }
}

// ============================================================================
// 4. RESERVOIR
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
  private static readonly ITERS = 100;

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

    // Deterministic initial vector
    let initNorm = 0;
    for (let i = 0; i < n; i++) {
      const x = (((i * 2654435761 + 1) >>> 0) % 1000) / 1000.0 - 0.5;
      scratch[vOff + i] = x;
      initNorm += x * x;
    }
    initNorm = Math.sqrt(initNorm);
    if (initNorm < 1e-12) initNorm = 1;
    for (let i = 0; i < n; i++) scratch[vOff + i] /= initNorm;

    let prevEst = 0;
    for (let it = 0; it < SpectralRadiusScaler.ITERS; it++) {
      // u = W * v
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
      if (uNorm < 1e-14) return 0;

      // Rayleigh quotient
      const est = Math.abs(TensorOps.dot(scratch, vOff, scratch, uOff, n));

      // Early convergence check
      if (it > 10 && Math.abs(est - prevEst) < 1e-10 * Math.max(1, est)) {
        return est;
      }
      prevEst = est;

      // Normalize
      const inv = 1 / uNorm;
      for (let i = 0; i < n; i++) scratch[vOff + i] = scratch[uOff + i] * inv;
    }

    return prevEst;
  }

  static scale(
    W: Float64Array,
    wOff: number,
    size: number,
    currentRadius: number,
    targetRadius: number,
  ): void {
    if (currentRadius < 1e-12) return;
    const s = targetRadius / currentRadius;
    const len = (size | 0) * (size | 0);
    for (let i = 0; i < len; i++) W[(wOff + i) | 0] *= s;
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
  private tmpPre: Float64Array;
  private tmpIn: Float64Array;
  private tmpRec: Float64Array;

  private estimatedRadius: number = 0;
  private actualSpectralRadius: number;

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
    this.actualSpectralRadius = config.spectralRadius;

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

    // Initialize Win with uniform distribution scaled by input weight scale
    const wScale = config.weightInitScale;
    for (let i = 0; i < this.Win.length; i++) {
      if (this.WinMask[i]) {
        this.Win[i] = rng.normal(0, wScale);
      }
    }

    // Initialize W with uniform distribution
    for (let i = 0; i < this.W.length; i++) {
      if (this.WMask[i]) {
        this.W[i] = rng.normal(0, wScale);
      }
    }

    // Force diagonal to zero for stability
    for (let i = 0; i < this.rs; i++) {
      const d = i * this.rs + i;
      this.W[d] = 0;
      this.WMask[d] = 0;
    }

    // Initialize bias
    const bScale = config.biasScale;
    for (let i = 0; i < this.rs; i++) {
      this.b[i] = rng.normal(0, bScale);
    }

    // Scale to target spectral radius
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

    if (this.estimatedRadius > 1e-12) {
      SpectralRadiusScaler.scale(
        this.W,
        0,
        this.rs,
        this.estimatedRadius,
        config.spectralRadius,
      );
    }

    this.state = new Float64Array(this.rs);
    this.tmpPre = new Float64Array(this.rs);
    this.tmpIn = new Float64Array(this.rs);
    this.tmpRec = new Float64Array(this.rs);
  }

  update(xNorm: Float64Array, xOff: number): void {
    this.stepStateInPlace(this.state, xNorm, xOff);
  }

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
    for (let i = 0; i < rs; i++) {
      let sum = 0;
      const rowStart = i * nf;
      for (let j = 0; j < nf; j++) {
        if (this.WinMask[rowStart + j]) {
          sum += this.Win[rowStart + j] * (inScale * xNorm[(xOff + j) | 0]);
        }
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

    // pre-activation and activation
    for (let i = 0; i < rs; i++) {
      const pre = this.tmpIn[i] + this.tmpRec[i] + this.b[i];
      const clampedPre = Math.max(-20, Math.min(20, pre)); // Prevent overflow in tanh
      const a = ActivationOps.apply(this.act, clampedPre);
      this.tmpPre[i] = safeVal(a);
    }

    // Leaky integration
    for (let i = 0; i < rs; i++) {
      const v = oneMinus * state[i] + leak * this.tmpPre[i];
      state[i] = safeVal(v);
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

  deserialize(d: {
    state: number[];
    Win: number[];
    W: number[];
    b: number[];
    WinMask: number[];
    WMask: number[];
    estimatedRadius?: number;
  }): void {
    for (let i = 0; i < this.state.length; i++) this.state[i] = d.state[i] || 0;
    for (let i = 0; i < this.Win.length; i++) this.Win[i] = d.Win[i] || 0;
    for (let i = 0; i < this.W.length; i++) this.W[i] = d.W[i] || 0;
    for (let i = 0; i < this.b.length; i++) this.b[i] = d.b[i] || 0;
    for (let i = 0; i < this.WinMask.length; i++) {
      this.WinMask[i] = (d.WinMask[i] || 0) as number;
    }
    for (let i = 0; i < this.WMask.length; i++) {
      this.WMask[i] = (d.WMask[i] || 0) as number;
    }
    this.estimatedRadius = d.estimatedRadius ?? 0;
  }
}

// ============================================================================
// 3. READOUT TRAINING
// ============================================================================

export interface ReadoutConfig {
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
}

export interface ReadoutParams {
  Wout: Float64Array;
}

export interface RLSState {
  P: Float64Array;
  gain: Float64Array;
  Pz: Float64Array;
  zTP: Float64Array;
}

class RLSOptimizer {
  private zDim: number;
  private nTargets: number;
  private lambda: number;
  private delta: number;
  private l2: number;
  private eps: number;
  private clip: number;

  private P: Float64Array;
  private gain: Float64Array;
  private Pz: Float64Array;
  private zTP: Float64Array;
  private tmpZ: Float64Array;

  private Wout: Float64Array;
  private updateCount: number = 0;

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
    this.tmpZ = new Float64Array(this.zDim);
    this.Wout = new Float64Array(this.nTargets * this.zDim);

    this.reset();
  }

  reset(): void {
    this.P.fill(0);
    // Initialize P = I / delta (standard RLS initialization)
    const initVal = this.delta > 0 ? 1.0 / this.delta : 1.0;
    for (let i = 0; i < this.zDim; i++) {
      this.P[i * this.zDim + i] = initVal;
    }
    this.Wout.fill(0);
    this.updateCount = 0;
  }

  getParams(): ReadoutParams {
    return { Wout: this.Wout };
  }

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

  update(
    z: Float64Array,
    zOff: number,
    y: Float64Array,
    yOff: number,
    weight: number,
  ): number {
    const zDim = this.zDim;
    const nT = this.nTargets;

    const w = Math.max(0, weight);
    if (w < 1e-12) return 0;

    const sqrtW = Math.sqrt(w);

    // Copy and scale z
    for (let i = 0; i < zDim; i++) {
      this.tmpZ[i] = sqrtW * z[(zOff + i) | 0];
    }

    // Pz = P * z'
    TensorOps.matVec(this.P, 0, zDim, zDim, this.tmpZ, 0, this.Pz, 0);

    // denom = lambda + z'^T * Pz + l2
    let zPz = TensorOps.dot(this.tmpZ, 0, this.Pz, 0, zDim);
    let denom = this.lambda + zPz + this.l2;
    if (denom < this.eps) denom = this.eps;

    // gain = Pz / denom
    const invDen = 1.0 / denom;
    for (let i = 0; i < zDim; i++) {
      this.gain[i] = this.Pz[i] * invDen;
    }

    // Compute errors and apply gradient clipping
    let errSumSq = 0;
    for (let t = 0; t < nT; t++) {
      const row = t * zDim;
      let pred = 0;
      for (let i = 0; i < zDim; i++) {
        pred += this.Wout[row + i] * this.tmpZ[i];
      }
      const yScaled = sqrtW * y[(yOff + t) | 0];
      const err = yScaled - pred;
      errSumSq += err * err;
    }

    const errRms = Math.sqrt(errSumSq / Math.max(1, nT));
    let clipScale = 1.0;
    if (this.clip > 0 && errRms > this.clip) {
      clipScale = this.clip / errRms;
    }

    // Update Wout
    for (let t = 0; t < nT; t++) {
      const row = t * zDim;
      let pred = 0;
      for (let i = 0; i < zDim; i++) {
        pred += this.Wout[row + i] * this.tmpZ[i];
      }
      const yScaled = sqrtW * y[(yOff + t) | 0];
      const err = (yScaled - pred) * clipScale;

      for (let i = 0; i < zDim; i++) {
        this.Wout[row + i] += this.gain[i] * err;
      }
    }

    // zTP = z'^T * P
    for (let j = 0; j < zDim; j++) {
      let sum = 0;
      for (let i = 0; i < zDim; i++) {
        sum += this.tmpZ[i] * this.P[i * zDim + j];
      }
      this.zTP[j] = sum;
    }

    // P = (P - gain * zTP) / lambda
    const invLam = 1.0 / this.lambda;
    for (let i = 0; i < zDim; i++) {
      const ki = this.gain[i];
      const rowStart = i * zDim;
      for (let j = 0; j < zDim; j++) {
        this.P[rowStart + j] = invLam *
          (this.P[rowStart + j] - ki * this.zTP[j]);
      }
    }

    // Symmetrize P for numerical stability
    for (let i = 0; i < zDim; i++) {
      for (let j = i + 1; j < zDim; j++) {
        const a = this.P[i * zDim + j];
        const b = this.P[j * zDim + i];
        const m = 0.5 * (a + b);
        this.P[i * zDim + j] = m;
        this.P[j * zDim + i] = m;
      }
    }

    // Add small regularization to diagonal periodically to prevent P from becoming singular
    this.updateCount++;
    if (this.updateCount % 100 === 0) {
      const reg = this.eps * 10;
      for (let i = 0; i < zDim; i++) {
        this.P[i * zDim + i] += reg;
      }
    }

    return errRms / sqrtW;
  }

  serialize(): { P: number[]; Wout: number[]; updateCount: number } {
    return {
      P: Array.from(this.P),
      Wout: Array.from(this.Wout),
      updateCount: this.updateCount,
    };
  }

  deserialize(d: { P: number[]; Wout: number[]; updateCount?: number }): void {
    for (let i = 0; i < this.P.length; i++) this.P[i] = d.P[i] || 0;
    for (let i = 0; i < this.Wout.length; i++) this.Wout[i] = d.Wout[i] || 0;
    this.updateCount = d.updateCount || 0;
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
  private z: Float64Array;

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

    this.zDim = this.rs + (this.useX ? this.nf : 0) + (this.useB ? 1 : 0);
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

    for (let i = 0; i < this.rs; i++) out[k++] = r[(rOff + i) | 0];
    if (this.useX) {
      for (let i = 0; i < this.nf; i++) out[k++] = x[(xOff + i) | 0];
    }
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

  serialize(): { rls: { P: number[]; Wout: number[]; updateCount: number } } {
    return { rls: this.rls.serialize() };
  }

  deserialize(
    d: { rls: { P: number[]; Wout: number[]; updateCount?: number } },
  ): void {
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
    model.setSerializableState(JSON.parse(s));
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
  private yNormalizer: WelfordNormalizer | null = null;
  private reservoir: ESNReservoir | null = null;
  private readout: LinearReadout | null = null;
  private residualStats: ResidualStatsTracker | null = null;
  private outlier: OutlierDownweighter | null = null;
  private metrics: MetricsAccumulator | null = null;

  // Scratch buffers
  private xRaw: Float64Array | null = null;
  private xNorm: Float64Array | null = null;
  private y: Float64Array | null = null;
  private yHat: Float64Array | null = null;
  private resid: Float64Array | null = null;
  private residStds: Float64Array | null = null;
  private scratchR: Float64Array | null = null;
  private scratchX: Float64Array | null = null;

  // Reusable results
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
    this.yNormalizer = new WelfordNormalizer(
      this.nTargets,
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
    this.scratchX = new Float64Array(this.nFeatures);

    const maxSteps = this.config.maxSequenceLength | 0;
    for (let i = 0; i < maxSteps; i++) {
      this.predResultObj.predictions[i] = new Array<number>(this.nTargets).fill(
        0,
      );
      this.predResultObj.lowerBounds[i] = new Array<number>(this.nTargets).fill(
        0,
      );
      this.predResultObj.upperBounds[i] = new Array<number>(this.nTargets).fill(
        0,
      );
    }

    this.initialized = true;
  }

  /**
   * Online training with incremental RLS readout updates.
   *
   * @param data Object containing xCoordinates [N x nFeatures] and yCoordinates [N x nTargets]
   * @returns FitResult (reused object; caller must copy if persistence needed)
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

    // Validate all rows before processing
    for (let i = 0; i < N; i++) {
      if (xCoordinates[i].length !== this.nFeatures) {
        throw new Error(
          `fitOnline: xCoordinates[${i}].length must equal nFeatures (${this.nFeatures})`,
        );
      }
      if (yCoordinates[i].length !== this.nTargets) {
        throw new Error(
          `fitOnline: yCoordinates[${i}].length must equal nTargets (${this.nTargets})`,
        );
      }
    }

    const ring = this.ring!;
    const norm = this.normalizer!;
    const yNorm = this.yNormalizer!;
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
      // 1) Push X into ring buffer FIRST (authoritative latest-X)
      const xr = xCoordinates[i];
      for (let j = 0; j < this.nFeatures; j++) xRaw[j] = xr[j];
      ring.push(xRaw, 0);

      // 2) Update X normalization stats
      norm.update(xRaw, 0);

      // 3) Normalize x
      norm.normalize(xRaw, 0, xNorm, 0);

      // 4) Update reservoir state
      res.update(xNorm, 0);

      // 5) Copy target and update Y normalizer
      const yr = yCoordinates[i];
      for (let t = 0; t < this.nTargets; t++) y[t] = yr[t];
      yNorm.update(y, 0);

      // 6) Forward pass (before training update) to compute prediction
      head.forward(res.getState(), 0, xNorm, 0, yHat, 0);

      // 7) Compute residuals for outlier detection and stats
      LossFunction.residuals(yHat, 0, y, 0, resid, 0, this.nTargets);

      // 8) Get current residual standard deviations
      rStats.getStds(stds, 0);

      // 9) Compute sample weight (downweight outliers)
      const w = outlier.computeWeight(resid, 0, stds, 0, this.nTargets);

      // 10) RLS update
      const gradNorm = head.train(res.getState(), 0, xNorm, 0, y, 0, w);

      // 11) Compute loss for metrics
      const loss = LossFunction.mse(yHat, 0, y, 0, this.nTargets);

      // 12) Update residual statistics with new residuals
      rStats.update(resid, 0);

      // 13) Update metrics accumulator
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
   * Multi-step ahead prediction using roll-forward with scratch reservoir state.
   *
   * @param futureSteps Number of steps to predict (1 to maxSequenceLength)
   * @returns PredictionResult (reused object; caller must copy if persistence needed)
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
    const scratchX = this.scratchX!;
    const out = this.predResultObj;

    // Get latest X from ring buffer (prediction context ends here)
    ring.getLatest(xRaw, 0);

    // Copy live reservoir state to scratch (do NOT mutate live state)
    res.copyStateTo(scratchR, 0);

    // Get base residual standard deviations for uncertainty
    rStats.getStds(stds, 0);

    // Determine rollforward mode
    const mode = this.config.rollforwardMode;
    const canAutoregress = mode === "autoregressive" &&
      this.nFeatures === this.nTargets;

    // Copy initial x for rollforward
    for (let j = 0; j < this.nFeatures; j++) scratchX[j] = xRaw[j];

    for (let step = 0; step < futureSteps; step++) {
      // Normalize current x using running stats
      norm.normalize(scratchX, 0, xNorm, 0);

      // Advance scratch reservoir state one step
      res.stepStateInPlace(scratchR, xNorm, 0);

      // Compute prediction using readout
      head.forward(scratchR, 0, xNorm, 0, yHat, 0);

      // Store prediction and compute uncertainty bounds
      // Uncertainty grows with horizon: sigma_k = sigma_1 * sqrt(k + 1)
      const horizonScale = Math.sqrt(step + 1);

      for (let t = 0; t < this.nTargets; t++) {
        const pred = safeVal(yHat[t]);
        out.predictions[step][t] = pred;

        const sigma1 = stds[t];
        const sigma = sigma1 * horizonScale;
        const halfWidth = this.config.uncertaintyMultiplier * sigma;

        out.lowerBounds[step][t] = pred - halfWidth;
        out.upperBounds[step][t] = pred + halfWidth;
      }

      // Update scratchX for next step
      if (canAutoregress) {
        // Autoregressive: use prediction as next input
        for (let j = 0; j < this.nFeatures; j++) {
          scratchX[j] = safeVal(yHat[j]);
        }
      }
      // else: holdLastX mode - scratchX remains unchanged
    }

    // Compute confidence score
    let avgStd = 0;
    for (let t = 0; t < this.nTargets; t++) avgStd += stds[t];
    avgStd = this.nTargets > 0 ? avgStd / this.nTargets : 0;

    // Base confidence from residual std (lower std = higher confidence)
    const baseConf = avgStd > 0 ? 1.0 / (1.0 + avgStd) : 1.0;

    // Penalize confidence for longer horizons
    const horizonPenalty = 1.0 / Math.sqrt(futureSteps);

    // Penalize confidence for insufficient training
    const samplePenalty = Math.min(
      1.0,
      this.sampleCount / Math.max(1, this.config.normalizationWarmup * 2),
    );

    out.confidence = clamp01(baseConf * horizonPenalty * samplePenalty);

    return out;
  }

  /**
   * @returns Model summary statistics
   */
  getModelSummary(): ModelSummary {
    const zDim = this.readout ? this.readout.getZDim() : 0;
    return {
      totalParameters: (this.nTargets | 0) * (zDim | 0),
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
   * @returns Weight information for inspection/debugging
   */
  getWeights(): WeightInfo {
    const weights: Array<{ name: string; shape: number[]; values: number[] }> =
      [];

    if (this.readout) {
      weights.push({
        name: "Wout",
        shape: [this.nTargets, this.readout.getZDim()],
        values: Array.from(this.readout.getWout()),
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
   * @returns Normalization statistics for inputs
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
   * Reset model to initial state while preserving configuration.
   */
  reset(): void {
    if (this.reservoir) this.reservoir.resetState();
    if (this.readout) this.readout.reset();
    if (this.normalizer) this.normalizer.reset();
    if (this.yNormalizer) this.yNormalizer.reset();
    if (this.ring) this.ring.reset();
    if (this.residualStats) this.residualStats.reset();
    if (this.metrics) this.metrics.reset();
    this.sampleCount = 0;
    this.rng = new RandomGenerator(this.config.seed);
  }

  /**
   * Serialize complete model state to JSON string.
   */
  save(): string {
    return SerializationHelper.serialize(this);
  }

  /**
   * Restore model state from JSON string.
   */
  load(s: string): void {
    SerializationHelper.deserialize(this, s);
  }

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
      yNormalizer: this.yNormalizer ? this.yNormalizer.serialize() : null,
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
      this.initialize(this.nFeatures, this.nTargets);

      if (state.ring) this.ring!.deserialize(state.ring);
      if (state.normalizer) this.normalizer!.deserialize(state.normalizer);
      if (state.yNormalizer) this.yNormalizer!.deserialize(state.yNormalizer);
      if (state.reservoir) this.reservoir!.deserialize(state.reservoir);
      if (state.readout) this.readout!.deserialize(state.readout);
      if (state.residualStats) {
        this.residualStats!.deserialize(state.residualStats);
      }
    }
  }
}

export default ESNRegression;
