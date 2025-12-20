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
      if (this.sizes[i] >= size) {
        const b = this.buffers[i]!;
        this.count--;
        this.buffers[i] = this.buffers[this.count];
        this.sizes[i] = this.sizes[this.count];
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
    return x < lo ? lo : x > hi ? hi : x;
  }
  static isFiniteNum(x: number): boolean {
    return x === x && x !== Infinity && x !== -Infinity;
  }
  static ensureFinite(x: number, fallback: number): number {
    return TensorOps.isFiniteNum(x) ? x : fallback;
  }
}

class ActivationOps {
  static applyScalar(x: number, kind: "tanh" | "relu"): number {
    if (kind === "tanh") return Math.tanh(x);
    return x > 0 ? x : 0;
  }
}

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
  nextGaussian(): number {
    const u1 = Math.max(this.nextFloat(), 1e-10);
    const u2 = this.nextFloat();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }
}

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
      if (!TensorOps.isFiniteNum(xi)) continue;
      const delta = xi - this.mean[i];
      this.mean[i] += delta / c;
      const delta2 = xi - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }
  stdAt(i: number): number {
    if (this.count < 2) return 0.0;
    const v = this.m2[i] / (this.count - 1);
    return v > 0 ? Math.sqrt(v) : 0.0;
  }
}

class WelfordNormalizer {
  readonly acc: WelfordAccumulator;
  readonly eps: number;
  readonly warmup: number;
  constructor(dim: number, eps: number, warmup: number) {
    this.acc = new WelfordAccumulator(dim | 0);
    this.eps = eps;
    this.warmup = warmup | 0;
  }
  reset(): void {
    this.acc.reset();
  }
  updateStats(xRaw: Float64Array): void {
    this.acc.update(xRaw);
  }
  normalize(xRaw: Float64Array, dst: Float64Array): void {
    const n = dst.length;
    const mean = this.acc.mean;
    const eps = this.eps;
    for (let i = 0; i < n; i++) {
      const xi = xRaw[i];
      if (!TensorOps.isFiniteNum(xi)) {
        dst[i] = 0.0;
        continue;
      }
      const std = this.acc.stdAt(i);
      const denom = std > eps ? std : eps;
      dst[i] = (xi - mean[i]) / denom;
      if (!TensorOps.isFiniteNum(dst[i])) dst[i] = 0.0;
    }
  }
  getStats(): NormalizationStats {
    const n = this.acc.mean.length;
    const means = new Array<number>(n);
    const stds = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      means[i] = this.acc.mean[i];
      stds[i] = Math.max(this.acc.stdAt(i), this.eps);
    }
    return {
      means,
      stds,
      count: this.acc.count,
      isActive: this.acc.count >= this.warmup,
    };
  }
}

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
    TensorOps.fill(this.data, 0.0);
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
    for (let i = 0; i < rb.data.length && i < arr.length; i++) {
      rb.data[i] = arr[i];
    }
    return rb;
  }
}

class ResidualStatsTracker {
  readonly windowSize: number;
  readonly nTargets: number;
  private buffers: Float64Array;
  private writeIndex: number;
  private count: number;
  private sum: Float64Array;
  private sumsq: Float64Array;
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

  update(residuals: Float64Array): void {
    const nT = this.nTargets;
    const w = this.windowSize;
    const idx = this.writeIndex;

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
    }

    this.writeIndex++;
    if (this.writeIndex >= w) this.writeIndex = 0;
  }

  getSigma(dst: Float64Array): void {
    const nT = this.nTargets;
    const c = this.count;
    const eps = this.eps;
    if (c < 2) {
      for (let t = 0; t < nT; t++) dst[t] = eps;
      return;
    }
    for (let t = 0; t < nT; t++) {
      const mean = this.sum[t] / c;
      const variance = Math.max(0, this.sumsq[t] / c - mean * mean);
      const s = Math.sqrt(variance * c / (c - 1));
      dst[t] = s > eps ? s : eps;
    }
  }

  getMeanAbsResidual(): number {
    if (this.count === 0) return 0;
    let total = 0;
    const c = this.count;
    for (let t = 0; t < this.nTargets; t++) {
      total += Math.abs(this.sum[t] / c);
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
    for (let i = 0; i < tr.buffers.length && i < b.length; i++) {
      tr.buffers[i] = b[i];
    }
    tr.writeIndex = obj.writeIndex | 0;
    tr.count = obj.count | 0;
    const s = obj.sum as number[];
    const ss = obj.sumsq as number[];
    for (let i = 0; i < tr.sum.length && i < s.length; i++) tr.sum[i] = s[i];
    for (let i = 0; i < tr.sumsq.length && i < ss.length; i++) {
      tr.sumsq[i] = ss[i];
    }
    return tr;
  }
}

class OutlierDownweighter {
  private threshold: number;
  private minWeight: number;
  constructor(threshold: number, minWeight: number) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }
  computeWeight(residuals: Float64Array, sigma: Float64Array): number {
    let maxZ = 0.0;
    for (let i = 0; i < residuals.length; i++) {
      const s = sigma[i] > 1e-12 ? sigma[i] : 1.0;
      const z = Math.abs(residuals[i]) / s;
      if (TensorOps.isFiniteNum(z) && z > maxZ) maxZ = z;
    }
    if (!TensorOps.isFiniteNum(maxZ)) return this.minWeight;
    if (maxZ <= this.threshold) return 1.0;
    const ratio = this.threshold / maxZ;
    const w = ratio * ratio;
    return TensorOps.clamp(w, this.minWeight, 1.0);
  }
}

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
}

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
    if (!TensorOps.isFiniteNum(loss)) return;
    this.sumLoss += loss;
    this.count++;
  }
  mean(): number {
    if (this.count <= 0) return 0.0;
    return this.sumLoss / this.count;
  }
  getCount(): number {
    return this.count;
  }
}

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

class SpectralRadiusScaler {
  static estimateSpectralRadius(
    W: Float64Array,
    n: number,
    iters: number,
    tmpV: Float64Array,
    tmpWv: Float64Array,
  ): number {
    for (let i = 0; i < n; i++) tmpV[i] = 1.0 / Math.sqrt(n);
    let eigenvalue = 0.0;
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < n; i++) {
        let s = 0.0;
        const row = i * n;
        for (let j = 0; j < n; j++) s += W[row + j] * tmpV[j];
        tmpWv[i] = s;
      }
      let norm = 0.0;
      for (let i = 0; i < n; i++) norm += tmpWv[i] * tmpWv[i];
      norm = Math.sqrt(norm);
      if (!TensorOps.isFiniteNum(norm) || norm < 1e-15) return 0.0;
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
    if (est < eps) return target;
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
        if (TensorOps.isFiniteNum(xj)) {
          s += Win[winRow + j] * xj * inScale;
        }
      }
      const wRow = i * N;
      for (let j = 0; j < N; j++) {
        s += W[wRow + j] * prev[j];
      }
      const a = ActivationOps.applyScalar(s, this.activation);
      state[i] = oneMinus * prev[i] + leak * a;
      if (!TensorOps.isFiniteNum(state[i])) state[i] = 0.0;
    }
  }

  step(xNorm: Float64Array): void {
    this.updateStateInPlace(this.state, this.prev, xNorm);
  }
}

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
  stabilize(): void {
    const D = this.zDim;
    const P = this.P;
    const eps = this.eps;
    const l2 = this.l2Lambda;
    for (let i = 0; i < D; i++) {
      for (let j = i + 1; j < D; j++) {
        const avg = 0.5 * (P[i * D + j] + P[j * D + i]);
        P[i * D + j] = avg;
        P[j * D + i] = avg;
      }
    }
    const minDiag = Math.max(eps, l2);
    const maxDiag = 1e8;
    for (let i = 0; i < D; i++) {
      let pii = P[i * D + i];
      if (!TensorOps.isFiniteNum(pii)) pii = 1.0;
      if (pii < minDiag) pii = minDiag;
      if (pii > maxDiag) pii = maxDiag;
      P[i * D + i] = pii;
    }
  }
}

class RLSOptimizer {
  private lambda: number;
  private eps: number;
  constructor(lambda: number, eps: number) {
    this.lambda = lambda;
    this.eps = eps;
  }

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

    for (let i = 0; i < D; i++) {
      let s = 0.0;
      const row = i * D;
      for (let j = 0; j < D; j++) s += P[row + j] * z[j];
      Pz[i] = TensorOps.ensureFinite(s, 0.0);
    }

    let zTPz = 0.0;
    for (let i = 0; i < D; i++) zTPz += z[i] * Pz[i];
    zTPz = TensorOps.ensureFinite(zTPz, 1.0);

    let denom = lambda + zTPz;
    if (!TensorOps.isFiniteNum(denom) || denom < eps) denom = eps;

    const invDen = 1.0 / denom;
    for (let i = 0; i < D; i++) {
      k[i] = Pz[i] * invDen;
      if (!TensorOps.isFiniteNum(k[i])) k[i] = 0.0;
    }

    let sumErr2 = 0.0;
    for (let t = 0; t < nTargets; t++) {
      const e = err[t];
      if (TensorOps.isFiniteNum(e)) sumErr2 += e * e;
    }
    const kNorm = TensorOps.l2Norm(k);
    let updateNorm = kNorm * Math.sqrt(sumErr2);
    if (!TensorOps.isFiniteNum(updateNorm)) {
      updateNorm = gradientClipNorm > 0 ? gradientClipNorm : 1.0;
    }

    let scale = 1.0;
    if (
      gradientClipNorm > 0 && updateNorm > gradientClipNorm && updateNorm > eps
    ) {
      scale = gradientClipNorm / updateNorm;
      updateNorm = gradientClipNorm;
    }

    for (let t = 0; t < nTargets; t++) {
      const base = t * D;
      const et = TensorOps.ensureFinite(err[t], 0.0) * scale;
      for (let i = 0; i < D; i++) {
        const delta = k[i] * et;
        if (TensorOps.isFiniteNum(delta)) Wout[base + i] += delta;
      }
    }

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
    const n = Math.min(dst.length, src.length);
    for (let i = 0; i < n; i++) dst[i] = src[i];
  }
}

export class ESNRegression {
  readonly config: ESNRegressionConfig;

  private initialized: boolean;
  private nFeatures: number;
  private nTargets: number;
  private zDim: number;

  private rng: RandomGenerator;
  private arena: TensorArena | null;
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

  private xRawScratch: Float64Array | null;
  private xNormScratch: Float64Array | null;
  private zScratch: Float64Array | null;
  private yHatScratch: Float64Array | null;
  private residualScratch: Float64Array | null;
  private sigmaScratch: Float64Array | null;
  private errScratch: Float64Array | null;

  private rScratch: Float64Array | null;
  private rPrevScratch: Float64Array | null;
  private xStepRawScratch: Float64Array | null;
  private xStepNormScratch: Float64Array | null;
  private zPredScratch: Float64Array | null;
  private yPredScratch: Float64Array | null;
  private sigmaPredScratch: Float64Array | null;

  private fitRes: FitResult;
  private predRes: PredictionResult | null;

  private metrics: MetricsAccumulator;
  private sampleCount: number;
  private scaledSpectralRadius: number;

  constructor(cfg?: Partial<ESNRegressionConfig>) {
    const base = defaultConfig();
    this.config = Object.assign(base, cfg || {});
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
    this.yHatScratch = null;
    this.residualScratch = null;
    this.sigmaScratch = null;
    this.errScratch = null;

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

    const D = this.zDim;
    let total = 0;
    total += N * N;
    total += N * F;
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
    total += F;
    total += F;
    total += D;
    total += T;
    total += T;
    total += N;
    total += N;
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
    TensorOps.fill(Wout, 0.0);
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
    this.rlsOpt = new RLSOptimizer(this.config.rlsLambda, this.config.epsilon);

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

    this.xRawScratch = this.arena.alloc(F);
    this.xNormScratch = this.arena.alloc(F);
    this.zScratch = this.arena.alloc(D);
    this.yHatScratch = this.arena.alloc(T);
    this.residualScratch = this.arena.alloc(T);
    this.sigmaScratch = this.arena.alloc(T);
    this.errScratch = this.arena.alloc(T);

    this.xStepRawScratch = this.arena.alloc(F);
    this.xStepNormScratch = this.arena.alloc(F);
    this.zPredScratch = this.arena.alloc(D);
    this.yPredScratch = this.arena.alloc(T);
    this.sigmaPredScratch = this.arena.alloc(T);

    TensorOps.fill(this.xRawScratch, 0.0);
    TensorOps.fill(this.xNormScratch, 0.0);
    TensorOps.fill(this.zScratch, 0.0);
    TensorOps.fill(this.yHatScratch, 0.0);
    TensorOps.fill(this.residualScratch, 0.0);
    TensorOps.fill(this.sigmaScratch, 0.0);
    TensorOps.fill(this.errScratch, 0.0);
    TensorOps.fill(this.xStepRawScratch, 0.0);
    TensorOps.fill(this.xStepNormScratch, 0.0);
    TensorOps.fill(this.zPredScratch, 0.0);
    TensorOps.fill(this.yPredScratch, 0.0);
    TensorOps.fill(this.sigmaPredScratch, 0.0);

    this.metrics.reset();
    this.sampleCount = 0;

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

    for (let i = 0; i < N * F; i++) {
      if (winMask[i]) {
        Win[i] = rng.nextGaussian() * wscale;
      } else {
        Win[i] = 0.0;
      }
    }

    const wMask = new Uint8Array(N * N);
    ReservoirInitMask.initMask(rng, wMask.length, cfg.reservoirSparsity, wMask);

    for (let i = 0; i < N * N; i++) {
      if (wMask[i]) {
        W[i] = rng.nextGaussian() * wscale;
      } else {
        W[i] = 0.0;
      }
    }

    const bscale = cfg.biasScale;
    for (let i = 0; i < N; i++) {
      b[i] = rng.nextGaussian() * bscale;
    }

    const tmpV = new Float64Array(N);
    const tmpWv = new Float64Array(N);
    this.scaledSpectralRadius = SpectralRadiusScaler.scaleToSpectralRadius(
      W,
      N,
      cfg.spectralRadius,
      50,
      tmpV,
      tmpWv,
      cfg.epsilon,
    );
  }

  private buildZ(
    rState: Float64Array,
    xNorm: Float64Array,
    zOut: Float64Array,
  ): void {
    const N = this.config.reservoirSize | 0;
    const F = this.nFeatures | 0;
    let idx = 0;
    for (let i = 0; i < N; i++) zOut[idx++] = rState[i];
    if (this.config.useInputInReadout) {
      for (let j = 0; j < F; j++) zOut[idx++] = xNorm[j];
    }
    if (this.config.useBiasInReadout) {
      zOut[idx++] = 1.0;
    }
  }

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

    if (!this.initialized) throw new Error("fitOnline: failed to initialize");

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
    const yHat = this.yHatScratch!;
    const residual = this.residualScratch!;
    const sigma = this.sigmaScratch!;
    const err = this.errScratch!;

    const T = this.nTargets | 0;
    const warmup = this.config.normalizationWarmup;

    let lastUpdateNorm = 0.0;
    let lastWeight = 1.0;

    for (let i = 0; i < N; i++) {
      ring.pushRow(xCoordinates[i]);
      ring.getLatestRow(xRaw);

      normalizer.updateStats(xRaw);
      normalizer.normalize(xRaw, xNorm);

      reservoir.step(xNorm);

      this.buildZ(reservoir.state, xNorm, z);
      readout.forward(z, yHat);

      const yRow = yCoordinates[i];
      for (let t = 0; t < T; t++) {
        residual[t] = yRow[t] - yHat[t];
      }

      residualTracker.getSigma(sigma);
      lastWeight = outlier.computeWeight(residual, sigma);

      const loss = LossFunction.mse(yRow, yHat);
      this.metrics.add(loss);

      if (this.sampleCount >= warmup) {
        for (let t = 0; t < T; t++) {
          err[t] = residual[t] * Math.sqrt(lastWeight);
        }
        lastUpdateNorm = rlsOpt.stepSharedP(
          rlsState,
          this.readoutParams!.Wout,
          z,
          err,
          T,
          this.config.gradientClipNorm,
        );
      }

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

    for (let i = 0; i < N; i++) rScratch[i] = reservoir.state[i];

    ring.getLatestRow(xStepRaw);

    let avgSigma = 0.0;
    for (let t = 0; t < T; t++) avgSigma += sigma[t];
    avgSigma /= T > 0 ? T : 1;

    const trainedSamples = this.sampleCount - this.config.normalizationWarmup;
    let conf = 1.0;
    if (trainedSamples > 0) {
      const avgLoss = this.metrics.mean();
      const relError = avgSigma > 1e-12 ? Math.sqrt(avgLoss) / avgSigma : 1.0;
      conf = 1.0 / (1.0 + relError);
      conf *= Math.min(1.0, trainedSamples / 50.0);
    } else {
      conf = 0.0;
    }
    if (!TensorOps.isFiniteNum(conf)) conf = 0.0;
    res.confidence = TensorOps.clamp(conf, 0.0, 1.0);

    const uncMult = this.config.uncertaintyMultiplier;
    const rollMode = this.config.rollforwardMode;
    const canAutoregress = rollMode === "autoregressive" && F === T;

    for (let step = 0; step < futureSteps; step++) {
      normalizer.normalize(xStepRaw, xStepNorm);

      reservoir.updateStateInPlace(rScratch, rPrevScratch, xStepNorm);

      this.buildZ(rScratch, xStepNorm, zPred);
      readout.forward(zPred, yPred);

      const pRow = res.predictions[step];
      const lRow = res.lowerBounds[step];
      const uRow = res.upperBounds[step];

      const horizonScale = Math.sqrt(step + 1);

      for (let t = 0; t < T; t++) {
        const y = yPred[t];
        const s = sigma[t] * horizonScale;
        const delta = uncMult * s;
        pRow[t] = TensorOps.ensureFinite(y, 0.0);
        lRow[t] = TensorOps.ensureFinite(y - delta, pRow[t]);
        uRow[t] = TensorOps.ensureFinite(y + delta, pRow[t]);
      }

      if (canAutoregress) {
        for (let j = 0; j < F; j++) {
          xStepRaw[j] = TensorOps.ensureFinite(yPred[j], xStepRaw[j]);
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

    const T = this.nTargets | 0;

    this.ring!.reset();
    this.normalizer!.reset();
    this.residualTracker!.reset();

    this.initReservoirWeights(
      this.reservoirParams!.W,
      this.reservoirParams!.Win,
      this.reservoirParams!.b,
    );

    this.reservoir!.reset();
    TensorOps.fill(this.rScratch!, 0.0);
    TensorOps.fill(this.rPrevScratch!, 0.0);

    TensorOps.fill(this.readoutParams!.Wout, 0.0);
    this.rlsState!.reset(this.config.rlsDelta);

    TensorOps.fill(this.xRawScratch!, 0.0);
    TensorOps.fill(this.xNormScratch!, 0.0);
    TensorOps.fill(this.zScratch!, 0.0);
    TensorOps.fill(this.yHatScratch!, 0.0);
    TensorOps.fill(this.residualScratch!, 0.0);
    TensorOps.fill(this.sigmaScratch!, 0.0);
    TensorOps.fill(this.errScratch!, 0.0);

    TensorOps.fill(this.xStepRawScratch!, 0.0);
    TensorOps.fill(this.xStepNormScratch!, 0.0);
    TensorOps.fill(this.zPredScratch!, 0.0);
    TensorOps.fill(this.yPredScratch!, 0.0);
    TensorOps.fill(this.sigmaPredScratch!, 0.0);

    this.metrics.reset();
    this.sampleCount = 0;

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
      version: 4,
      config: this.config,
      initialized: this.initialized,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      zDim: this.zDim,
      sampleCount: this.sampleCount,
      scaledSpectralRadius: this.scaledSpectralRadius,
      metricsMean: this.metrics.mean(),
      metricsCount: this.metrics.getCount(),
    };

    if (this.initialized) {
      obj.ring = this.ring!.toJSON();
      obj.normalizer = {
        mean: Array.from(this.normalizer!.acc.mean),
        m2: Array.from(this.normalizer!.acc.m2),
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
      for (let i = 0; i < rState.length && i < rs.length; i++) {
        rState[i] = rs[i];
      }

      const ro = obj.readout;
      SerializationHelper.fromNumberArray(
        this.readoutParams!.Wout,
        ro.Wout as number[],
      );

      const rls = obj.rls;
      SerializationHelper.fromNumberArray(this.rlsState!.P, rls.P as number[]);

      this.metrics.reset();
      if (
        typeof obj.metricsMean === "number" &&
        typeof obj.metricsCount === "number"
      ) {
        for (let i = 0; i < obj.metricsCount; i++) {
          this.metrics.add(obj.metricsMean);
        }
      }

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
