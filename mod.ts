/******************************************************
 * ESNRegression - single-file TypeScript implementation
 * Deterministic, allocation-free hot paths (fit/predict)
 ******************************************************/

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
  maxFutureSteps: number;
  sampleCount: number;
  useDirectMultiHorizon: boolean;
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
  maxFutureSteps: number; // Default: 1
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
  useDirectMultiHorizon: boolean; // Default: true
  residualWindowSize: number; // Default: 100
  uncertaintyMultiplier: number; // Default: 1.96
  weightInitScale: number; // Default: 0.1
  seed: number; // Default: 42
  verbose: boolean; // Default: false
}

/** -------------- 1. Memory infra -------------- */

export class TensorShape {
  readonly dims: number[];
  readonly size: number;

  constructor(dims: number[]) {
    this.dims = dims.slice(0);
    let s = 1;
    for (let i = 0; i < dims.length; i++) s *= dims[i] | 0;
    this.size = s | 0;
  }
}

export class TensorView {
  data: Float64Array;
  offset: number;
  shape: TensorShape;
  strides: number[];

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
      this.strides = TensorOps.computeRowMajorStrides(shape.dims);
    }
  }
}

export class BufferPool {
  // Simple size-class pool; used only during initialization or optional calls.
  private free: Map<number, Float64Array[]> = new Map();

  rent(size: number): Float64Array {
    const s = size | 0;
    const arr = this.free.get(s);
    if (arr && arr.length > 0) return arr.pop() as Float64Array;
    return new Float64Array(s);
  }

  release(buf: Float64Array): void {
    const s = buf.length | 0;
    let arr = this.free.get(s);
    if (!arr) {
      arr = [];
      this.free.set(s, arr);
    }
    arr.push(buf);
  }
}

export class TensorArena {
  private buffer: Float64Array;
  private offset: number;

  constructor(size: number) {
    this.buffer = new Float64Array(size | 0);
    this.offset = 0;
  }

  reset(): void {
    this.offset = 0;
  }

  alloc(size: number): Float64Array {
    const s = size | 0;
    const o = this.offset | 0;
    const n = o + s;
    if (n > this.buffer.length) throw new Error("TensorArena out of memory");
    this.offset = n;
    return this.buffer.subarray(o, n);
  }
}

export class TensorOps {
  static computeRowMajorStrides(dims: number[]): number[] {
    const nd = dims.length | 0;
    const strides = new Array<number>(nd);
    let acc = 1;
    for (let i = nd - 1; i >= 0; i--) {
      strides[i] = acc;
      acc *= dims[i] | 0;
    }
    return strides;
  }

  static fill(x: Float64Array, v: number): void {
    const n = x.length | 0;
    for (let i = 0; i < n; i++) x[i] = v;
  }

  static copy(dst: Float64Array, src: Float64Array): void {
    const n = dst.length | 0;
    for (let i = 0; i < n; i++) dst[i] = src[i];
  }

  static dot(a: Float64Array, b: Float64Array): number {
    const n = a.length | 0;
    let s = 0.0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }

  static norm2(a: Float64Array): number {
    const n = a.length | 0;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const v = a[i];
      s += v * v;
    }
    return Math.sqrt(s);
  }

  static axpy(y: Float64Array, a: number, x: Float64Array): void {
    const n = y.length | 0;
    for (let i = 0; i < n; i++) y[i] += a * x[i];
  }

  static scale(x: Float64Array, a: number): void {
    const n = x.length | 0;
    for (let i = 0; i < n; i++) x[i] *= a;
  }

  /** y = A*x, A row-major [m,n] */
  static matVec(
    A: Float64Array,
    m: number,
    n: number,
    x: Float64Array,
    y: Float64Array,
  ): void {
    const M = m | 0;
    const N = n | 0;
    let idx = 0;
    for (let i = 0; i < M; i++) {
      let s = 0.0;
      for (let j = 0; j < N; j++) {
        s += A[idx++] * x[j];
      }
      y[i] = s;
    }
  }

  /** y = W*x + b, W row-major [m,n], b [m] */
  static matVecAddBias(
    W: Float64Array,
    m: number,
    n: number,
    x: Float64Array,
    b: Float64Array | null,
    y: Float64Array,
  ): void {
    const M = m | 0;
    const N = n | 0;
    let idx = 0;
    if (b) {
      for (let i = 0; i < M; i++) {
        let s = b[i];
        for (let j = 0; j < N; j++) s += W[idx++] * x[j];
        y[i] = s;
      }
    } else {
      for (let i = 0; i < M; i++) {
        let s = 0.0;
        for (let j = 0; j < N; j++) s += W[idx++] * x[j];
        y[i] = s;
      }
    }
  }
}

/** -------------- 2. Numerics -------------- */

export class ActivationOps {
  static applyInPlace(x: Float64Array, kind: "tanh" | "relu"): void {
    const n = x.length | 0;
    if (kind === "tanh") {
      for (let i = 0; i < n; i++) x[i] = Math.tanh(x[i]);
    } else {
      for (let i = 0; i < n; i++) {
        const v = x[i];
        x[i] = v > 0 ? v : 0;
      }
    }
  }
}

export class RandomGenerator {
  // xorshift32 deterministic RNG
  private state: number;

  constructor(seed: number) {
    const s = (seed | 0) >>> 0;
    this.state = s === 0 ? 0x9e3779b9 : s;
  }

  nextUint32(): number {
    let x = this.state >>> 0;
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x >>>= 0;
    x ^= x << 5;
    x >>>= 0;
    this.state = x >>> 0;
    return x >>> 0;
  }

  nextFloat(): number {
    // [0,1)
    return (this.nextUint32() >>> 0) / 4294967296.0;
  }

  nextSignedFloat(): number {
    // (-1,1)
    return this.nextFloat() * 2.0 - 1.0;
  }

  nextGaussian(): number {
    // Box-Muller (deterministic); avoid allocations
    let u = 0.0;
    let v = 0.0;
    // ensure non-zero
    u = this.nextFloat();
    if (u < 1e-12) u = 1e-12;
    v = this.nextFloat();
    const r = Math.sqrt(-2.0 * Math.log(u));
    const t = 2.0 * Math.PI * v;
    return r * Math.cos(t);
  }
}

export class WelfordAccumulator {
  count: number;
  mean: number;
  m2: number;

  constructor() {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }

  reset(): void {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }

  add(x: number): void {
    const c1 = this.count + 1;
    const delta = x - this.mean;
    const mean = this.mean + delta / c1;
    const delta2 = x - mean;
    this.m2 += delta * delta2;
    this.mean = mean;
    this.count = c1;
  }

  variance(epsilon: number): number {
    if (this.count < 2) return 0;
    const v = this.m2 / (this.count - 1);
    return v > epsilon ? v : epsilon;
  }

  std(epsilon: number): number {
    return Math.sqrt(this.variance(epsilon));
  }
}

export class WelfordNormalizer {
  readonly n: number;
  readonly epsilon: number;
  readonly warmup: number;

  private count: number;
  private means: Float64Array;
  private m2: Float64Array;
  private stds: Float64Array;
  private active: boolean;

  constructor(n: number, epsilon: number, warmup: number) {
    this.n = n | 0;
    this.epsilon = epsilon;
    this.warmup = warmup | 0;
    this.count = 0;
    this.means = new Float64Array(this.n);
    this.m2 = new Float64Array(this.n);
    this.stds = new Float64Array(this.n);
    this.active = false;
    for (let i = 0; i < this.n; i++) this.stds[i] = 1.0;
  }

  reset(): void {
    this.count = 0;
    this.active = false;
    TensorOps.fill(this.means, 0);
    TensorOps.fill(this.m2, 0);
    for (let i = 0; i < this.n; i++) this.stds[i] = 1.0;
  }

  observe(x: Float64Array): void {
    // Welford per feature
    const n = this.n;
    const c1 = (this.count + 1) | 0;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      const mi = this.means[i];
      const delta = xi - mi;
      const mean = mi + delta / c1;
      const delta2 = xi - mean;
      this.m2[i] += delta * delta2;
      this.means[i] = mean;
    }
    this.count = c1;

    if (!this.active && this.count >= this.warmup) {
      this.active = true;
      for (let i = 0; i < n; i++) {
        const denom = this.count > 1 ? (this.count - 1) : 1;
        let v = this.m2[i] / denom;
        if (!(v > 0) || !Number.isFinite(v)) v = this.epsilon;
        if (v < this.epsilon) v = this.epsilon;
        const s = Math.sqrt(v);
        this.stds[i] = s > this.epsilon ? s : this.epsilon;
      }
    } else if (this.active) {
      // update stds each step (stable enough) with current m2
      for (let i = 0; i < n; i++) {
        const denom = this.count > 1 ? (this.count - 1) : 1;
        let v = this.m2[i] / denom;
        if (!(v > 0) || !Number.isFinite(v)) v = this.epsilon;
        if (v < this.epsilon) v = this.epsilon;
        const s = Math.sqrt(v);
        this.stds[i] = s > this.epsilon ? s : this.epsilon;
      }
    }
  }

  normalize(x: Float64Array, out: Float64Array): void {
    const n = this.n;
    if (!this.active) {
      for (let i = 0; i < n; i++) out[i] = x[i];
      return;
    }
    for (let i = 0; i < n; i++) {
      out[i] = (x[i] - this.means[i]) / this.stds[i];
    }
  }

  denormalize(y: Float64Array, out: Float64Array): void {
    // This normalizer is for X only; keep passthrough for API symmetry.
    const n = y.length | 0;
    for (let i = 0; i < n; i++) out[i] = y[i];
  }

  getCount(): number {
    return this.count | 0;
  }

  isActive(): boolean {
    return this.active;
  }

  getMeansArray(): Float64Array {
    return this.means;
  }

  getStdsArray(): Float64Array {
    return this.stds;
  }
}

export class LayerNormParams {
  gamma: Float64Array;
  beta: Float64Array;
  epsilon: number;

  constructor(size: number, epsilon: number) {
    this.gamma = new Float64Array(size | 0);
    this.beta = new Float64Array(size | 0);
    this.epsilon = epsilon;
    for (let i = 0; i < this.gamma.length; i++) this.gamma[i] = 1.0;
  }
}

export class LayerNormOps {
  static forward(
    x: Float64Array,
    params: LayerNormParams,
    out: Float64Array,
  ): void {
    const n = x.length | 0;
    let mean = 0.0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let v = 0.0;
    for (let i = 0; i < n; i++) {
      const d = x[i] - mean;
      v += d * d;
    }
    v /= n;
    const inv = 1.0 / Math.sqrt(v + params.epsilon);
    for (let i = 0; i < n; i++) {
      const xn = (x[i] - mean) * inv;
      out[i] = xn * params.gamma[i] + params.beta[i];
    }
  }
}

export class GradientAccumulator {
  // Placeholder for API completeness
  grad: Float64Array;
  constructor(size: number) {
    this.grad = new Float64Array(size | 0);
  }
  reset(): void {
    TensorOps.fill(this.grad, 0);
  }
}

/** -------------- 3. Readout training (RLS) -------------- */

export class RLSState {
  readonly dim: number;
  readonly lambda: number;
  readonly delta: number;
  P: Float64Array; // [dim, dim] row-major
  k: Float64Array; // [dim]
  Pz: Float64Array; // [dim]

  constructor(dim: number, lambda: number, delta: number) {
    this.dim = dim | 0;
    this.lambda = lambda;
    this.delta = delta;
    this.P = new Float64Array(this.dim * this.dim);
    this.k = new Float64Array(this.dim);
    this.Pz = new Float64Array(this.dim);
    this.reset();
  }

  reset(): void {
    // P = (1/delta) * I
    TensorOps.fill(this.P, 0);
    const inv = 1.0 / (this.delta > 0 ? this.delta : 1.0);
    const d = this.dim;
    for (let i = 0; i < d; i++) this.P[i * d + i] = inv;
    TensorOps.fill(this.k, 0);
    TensorOps.fill(this.Pz, 0);
  }
}

export class RLSOptimizer {
  readonly dim: number;
  readonly outDim: number;
  readonly epsilon: number;
  readonly l2Lambda: number;
  readonly gradientClipNorm: number;

  private state: RLSState;
  private z: Float64Array; // reference/scratch externally set
  private denom: number;
  private kNorm: number;
  private zDim: number;
  private lastUpdateNorm: number;

  constructor(
    state: RLSState,
    outDim: number,
    epsilon: number,
    l2Lambda: number,
    gradientClipNorm: number,
  ) {
    this.state = state;
    this.dim = state.dim | 0;
    this.zDim = this.dim;
    this.outDim = outDim | 0;
    this.epsilon = epsilon;
    this.l2Lambda = l2Lambda;
    this.gradientClipNorm = gradientClipNorm;
    this.z = new Float64Array(this.dim); // will be overwritten by setZRef; allocated once at init
    this.denom = 1.0;
    this.kNorm = 0.0;
    this.lastUpdateNorm = 0.0;
  }

  setZRef(z: Float64Array): void {
    if (z.length !== this.dim) {
      throw new Error("RLSOptimizer.setZRef: dim mismatch");
    }
    this.z = z;
  }

  getLastUpdateNorm(): number {
    return this.lastUpdateNorm;
  }

  /** Computes gain k and updates P; then caller updates weights with k and errors. */
  computeGainAndUpdateP(): void {
    const d = this.dim;
    const P = this.state.P;
    const Pz = this.state.Pz;
    const z = this.z;

    // Pz = P * z
    for (let i = 0; i < d; i++) {
      const row = i * d;
      let s = 0.0;
      for (let j = 0; j < d; j++) s += P[row + j] * z[j];
      Pz[i] = s;
    }

    // denom = lambda + z^T P z
    let zTPz = 0.0;
    for (let i = 0; i < d; i++) zTPz += z[i] * Pz[i];
    let denom = this.state.lambda + zTPz;
    if (!(denom > 0) || !Number.isFinite(denom)) {
      denom = this.state.lambda + this.epsilon;
    }
    if (denom < this.epsilon) denom = this.epsilon;
    this.denom = denom;

    // k = Pz / denom
    const k = this.state.k;
    let kss = 0.0;
    const invDen = 1.0 / denom;
    for (let i = 0; i < d; i++) {
      const kv = Pz[i] * invDen;
      k[i] = kv;
      kss += kv * kv;
    }
    this.kNorm = Math.sqrt(kss);

    // P = (P - k * (Pz^T)) / lambda
    const invLam = 1.0 / this.state.lambda;
    for (let i = 0; i < d; i++) {
      const ki = k[i];
      const row = i * d;
      for (let j = 0; j < d; j++) {
        const v = (P[row + j] - ki * Pz[j]) * invLam;
        P[row + j] = Number.isFinite(v) ? v : 0.0;
      }
    }
  }

  /**
   * Applies weight updates:
   * Wout[o,:] += k * e_o (with optional clipping and weight decay)
   */
  applyWeightUpdate(
    Wout: Float64Array,
    errors: Float64Array,
    sampleWeight: number,
  ): void {
    const d = this.dim;
    const outD = this.outDim;
    const k = this.state.k;
    const clip = this.gradientClipNorm;
    const eps = this.epsilon;
    const sw = sampleWeight;

    // weight decay (simple ridge-like shrink; stable and cheap)
    if (this.l2Lambda > 0) {
      const decay = 1.0 - this.l2Lambda;
      if (decay > 0 && decay < 1) {
        TensorOps.scale(Wout, decay);
      }
    }

    // Compute max update norm for reporting
    // updateNorm(o) ~ |e_o| * ||k||
    let maxNorm = 0.0;

    // Apply updates per output row
    for (let o = 0; o < outD; o++) {
      const eRaw = errors[o] * sw;
      const absE = Math.abs(eRaw);
      const updNorm = absE * this.kNorm;
      if (updNorm > maxNorm) maxNorm = updNorm;

      let scale = 1.0;
      if (clip > 0 && updNorm > clip) {
        scale = clip / (updNorm + eps);
      }
      const e = eRaw * scale;

      const base = o * d;
      for (let j = 0; j < d; j++) {
        Wout[base + j] += k[j] * e;
      }
    }

    this.lastUpdateNorm = maxNorm;
  }
}

/** -------------- 4. Reservoir -------------- */

export class ReservoirInitMask {
  // Placeholder: deterministic sparsity masks (not strictly needed if we directly sample zeros)
  readonly size: number;
  constructor(size: number) {
    this.size = size | 0;
  }
}

export class SpectralRadiusScaler {
  static estimateSpectralRadius(
    W: Float64Array,
    n: number,
    iters: number,
    rng: RandomGenerator,
    epsilon: number,
    scratchV: Float64Array,
    scratchWv: Float64Array,
  ): number {
    const N = n | 0;
    const v = scratchV;
    const wv = scratchWv;
    if (v.length !== N || wv.length !== N) {
      throw new Error("SpectralRadiusScaler: scratch dim mismatch");
    }

    // init v random
    for (let i = 0; i < N; i++) v[i] = rng.nextSignedFloat();

    // normalize
    let vn = TensorOps.norm2(v);
    if (!(vn > 0) || !Number.isFinite(vn)) vn = 1.0;
    const inv = 1.0 / vn;
    for (let i = 0; i < N; i++) v[i] *= inv;

    let est = 0.0;
    for (let t = 0; t < iters; t++) {
      // wv = W * v
      let idx = 0;
      for (let i = 0; i < N; i++) {
        let s = 0.0;
        for (let j = 0; j < N; j++) s += W[idx++] * v[j];
        wv[i] = s;
      }
      const wvn = TensorOps.norm2(wv);
      if (!(wvn > 0) || !Number.isFinite(wvn)) return 0.0;
      est = wvn; // since v is normalized, ||Wv|| approximates spectral radius
      const invn = 1.0 / (wvn + epsilon);
      for (let i = 0; i < N; i++) v[i] = wv[i] * invn;
    }
    return est;
  }

  static scaleToRadius(
    W: Float64Array,
    n: number,
    targetRadius: number,
    rng: RandomGenerator,
    epsilon: number,
    scratchV: Float64Array,
    scratchWv: Float64Array,
  ): number {
    const iters = Math.max(20, Math.min(100, (n | 0) + 10));
    const est = this.estimateSpectralRadius(
      W,
      n,
      iters,
      rng,
      epsilon,
      scratchV,
      scratchWv,
    );
    if (!(est > 0) || !Number.isFinite(est)) return 0.0;
    const scale = targetRadius / (est + epsilon);
    TensorOps.scale(W, scale);
    return est * scale;
  }
}

export class ESNReservoirParams {
  reservoirSize: number;
  nFeatures: number;
  leakRate: number;
  spectralRadius: number;
  inputScale: number;
  biasScale: number;
  reservoirSparsity: number;
  inputSparsity: number;
  activation: "tanh" | "relu";
  weightInitScale: number;
  seed: number;
  epsilon: number;

  constructor(cfg: ESNRegressionConfig, nFeatures: number) {
    this.reservoirSize = cfg.reservoirSize | 0;
    this.nFeatures = nFeatures | 0;
    this.leakRate = cfg.leakRate;
    this.spectralRadius = cfg.spectralRadius;
    this.inputScale = cfg.inputScale;
    this.biasScale = cfg.biasScale;
    this.reservoirSparsity = cfg.reservoirSparsity;
    this.inputSparsity = cfg.inputSparsity;
    this.activation = cfg.activation;
    this.weightInitScale = cfg.weightInitScale;
    this.seed = cfg.seed | 0;
    this.epsilon = cfg.epsilon;
  }
}

export class ESNReservoir {
  readonly params: ESNReservoirParams;

  // Fixed weights
  readonly Win: Float64Array; // [N, F]
  readonly W: Float64Array; // [N, N]
  readonly bias: Float64Array; // [N]

  // State
  readonly r: Float64Array; // [N]
  private preAct: Float64Array; // [N] scratch

  constructor(params: ESNReservoirParams) {
    this.params = params;
    const N = params.reservoirSize | 0;
    const F = params.nFeatures | 0;
    this.Win = new Float64Array(N * F);
    this.W = new Float64Array(N * N);
    this.bias = new Float64Array(N);
    this.r = new Float64Array(N);
    this.preAct = new Float64Array(N);
    this.initWeightsDeterministic();
  }

  resetState(): void {
    TensorOps.fill(this.r, 0);
  }

  private initWeightsDeterministic(): void {
    const N = this.params.reservoirSize | 0;
    const F = this.params.nFeatures | 0;
    const rng = new RandomGenerator(this.params.seed);
    const scale = this.params.weightInitScale;

    // Win init (possibly sparse)
    const inSparsity = this.params.inputSparsity;
    const win = this.Win;
    let idx = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < F; j++) {
        const keep = inSparsity <= 0 ? true : (rng.nextFloat() >= inSparsity);
        win[idx++] = keep ? (rng.nextSignedFloat() * scale) : 0.0;
      }
    }

    // W init (sparse)
    const sparsity = this.params.reservoirSparsity;
    const W = this.W;
    idx = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const keep = sparsity >= 1 ? false : (rng.nextFloat() >= sparsity);
        // avoid very dense diagonal dominance; allow diagonal too but sparse rule applies
        W[idx++] = keep ? (rng.nextSignedFloat() * scale) : 0.0;
      }
    }

    // bias init
    const b = this.bias;
    const bScale = this.params.biasScale;
    for (let i = 0; i < N; i++) b[i] = rng.nextSignedFloat() * bScale;

    // scale spectral radius
    const scratchV = new Float64Array(N);
    const scratchWv = new Float64Array(N);
    const achieved = SpectralRadiusScaler.scaleToRadius(
      W,
      N,
      this.params.spectralRadius,
      rng,
      this.params.epsilon,
      scratchV,
      scratchWv,
    );
    if (!(achieved > 0) || !Number.isFinite(achieved)) {
      // fallback: identity * spectralRadius
      TensorOps.fill(W, 0);
      for (let i = 0; i < N; i++) W[i * N + i] = this.params.spectralRadius;
    }
  }

  /**
   * Leaky integrator ESN update:
   * r_t = (1-a) r_{t-1} + a * act( Win*(s*x) + W*r_{t-1} + bias )
   */
  update(xNorm: Float64Array): void {
    const N = this.params.reservoirSize | 0;
    const F = this.params.nFeatures | 0;
    const a = this.params.leakRate;
    const oneMinusA = 1.0 - a;
    const inputScale = this.params.inputScale;
    const Win = this.Win;
    const W = this.W;
    const b = this.bias;
    const r = this.r;
    const pre = this.preAct;

    // pre = bias
    for (let i = 0; i < N; i++) pre[i] = b[i];

    // pre += Win * (inputScale*x)
    let idx = 0;
    for (let i = 0; i < N; i++) {
      let s = pre[i];
      for (let j = 0; j < F; j++) {
        s += Win[idx++] * (xNorm[j] * inputScale);
      }
      pre[i] = s;
    }

    // pre += W * r
    idx = 0;
    for (let i = 0; i < N; i++) {
      let s = pre[i];
      for (let j = 0; j < N; j++) {
        s += W[idx++] * r[j];
      }
      pre[i] = s;
    }

    // act(pre)
    ActivationOps.applyInPlace(pre, this.params.activation);

    // leaky update r = (1-a)*r + a*pre
    for (let i = 0; i < N; i++) {
      r[i] = oneMinusA * r[i] + a * pre[i];
    }
  }
}

/** -------------- 5. Readout/head -------------- */

export class ReadoutConfig {
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
  useDirectMultiHorizon: boolean;
  maxFutureSteps: number;

  constructor(cfg: ESNRegressionConfig) {
    this.useInputInReadout = cfg.useInputInReadout;
    this.useBiasInReadout = cfg.useBiasInReadout;
    this.useDirectMultiHorizon = cfg.useDirectMultiHorizon;
    this.maxFutureSteps = cfg.maxFutureSteps | 0;
  }
}

export class ReadoutParams {
  readonly outDim: number;
  readonly zDim: number;
  Wout: Float64Array; // [outDim, zDim]
  constructor(outDim: number, zDim: number) {
    this.outDim = outDim | 0;
    this.zDim = zDim | 0;
    this.Wout = new Float64Array(this.outDim * this.zDim);
  }
}

export class LinearReadout {
  readonly params: ReadoutParams;
  constructor(params: ReadoutParams) {
    this.params = params;
  }

  forward(z: Float64Array, yOut: Float64Array): void {
    const outDim = this.params.outDim | 0;
    const zDim = this.params.zDim | 0;
    TensorOps.matVec(this.params.Wout, outDim, zDim, z, yOut);
  }
}

export class LinearLayerParams {
  W: Float64Array;
  b: Float64Array | null;
  inDim: number;
  outDim: number;

  constructor(inDim: number, outDim: number, useBias: boolean) {
    this.inDim = inDim | 0;
    this.outDim = outDim | 0;
    this.W = new Float64Array(this.outDim * this.inDim);
    this.b = useBias ? new Float64Array(this.outDim) : null;
  }
}

export class LinearLayer {
  params: LinearLayerParams;
  constructor(params: LinearLayerParams) {
    this.params = params;
  }
  forward(x: Float64Array, y: Float64Array): void {
    TensorOps.matVecAddBias(
      this.params.W,
      this.params.outDim,
      this.params.inDim,
      x,
      this.params.b,
      y,
    );
  }
}

export class DropoutMask {
  // Placeholder (not used)
  mask: Uint8Array;
  constructor(size: number) {
    this.mask = new Uint8Array(size | 0);
  }
}

/** -------------- 6. Training utilities -------------- */

export class ForwardContext {
  // Placeholder for API completeness
  constructor() {}
}

export class BackwardContext {
  // Placeholder
  constructor() {}
}

export class GradientTape {
  // Placeholder
  constructor() {}
}

export class RingBuffer {
  readonly capacity: number;
  readonly nFeatures: number;
  private data: Float64Array;
  private head: number; // next write
  private size: number;
  private totalPushed: number;

  constructor(capacity: number, nFeatures: number) {
    this.capacity = capacity | 0;
    this.nFeatures = nFeatures | 0;
    this.data = new Float64Array(this.capacity * this.nFeatures);
    this.head = 0;
    this.size = 0;
    this.totalPushed = 0;
  }

  reset(): void {
    TensorOps.fill(this.data, 0);
    this.head = 0;
    this.size = 0;
    this.totalPushed = 0;
  }

  pushRow(x: number[] | Float64Array): void {
    const F = this.nFeatures;
    const base = (this.head * F) | 0;

    for (let j = 0; j < F; j++) this.data[base + j] = (x as any)[j];

    this.head = (this.head + 1) % this.capacity;
    if (this.size < this.capacity) this.size++;
    this.totalPushed++;
  }

  /** Returns number of rows currently stored */
  length(): number {
    return this.size | 0;
  }

  /** Total rows pushed since reset (monotonic) */
  totalCount(): number {
    return this.totalPushed | 0;
  }

  /** Copies latest row (most recently pushed) into out */
  copyLatestRow(out: Float64Array): void {
    const F = this.nFeatures;
    if (this.size <= 0) throw new Error("RingBuffer is empty");
    const latestIndex = (this.head - 1 + this.capacity) % this.capacity;
    const base = latestIndex * F;
    for (let j = 0; j < F; j++) out[j] = this.data[base + j];
  }

  /** Copies the row ending at latest with offsetFromLatest (0=latest, 1=prev, ...) */
  copyRowFromLatest(offsetFromLatest: number, out: Float64Array): void {
    const off = offsetFromLatest | 0;
    const F = this.nFeatures;
    if (off < 0 || off >= this.size) {
      throw new Error("RingBuffer offset out of range");
    }
    const idx = (this.head - 1 - off + this.capacity * 4) % this.capacity;
    const base = idx * F;
    for (let j = 0; j < F; j++) out[j] = this.data[base + j];
  }

  /** Copies last k rows into out sequentially oldest->newest. out length = k*F */
  copyLastKRows(k: number, out: Float64Array): void {
    const K = k | 0;
    const F = this.nFeatures;
    if (K < 0 || K > this.size) throw new Error("copyLastKRows: invalid k");
    // Oldest of last K is offsetFromLatest = K-1
    let outBase = 0;
    for (let i = K - 1; i >= 0; i--) {
      const idx = (this.head - 1 - i + this.capacity * 4) % this.capacity;
      const base = idx * F;
      for (let j = 0; j < F; j++) out[outBase + j] = this.data[base + j];
      outBase += F;
    }
  }

  /** For serialization */
  getRawData(): Float64Array {
    return this.data;
  }
  getHead(): number {
    return this.head | 0;
  }
  getSize(): number {
    return this.size | 0;
  }
  setInternal(
    data: Float64Array,
    head: number,
    size: number,
    totalPushed: number,
  ): void {
    if (data.length !== this.data.length) {
      throw new Error("RingBuffer.setInternal: data length mismatch");
    }
    TensorOps.copy(this.data, data);
    this.head = head | 0;
    this.size = size | 0;
    this.totalPushed = totalPushed | 0;
  }
}

export class ResidualStatsTracker {
  readonly windowSize: number;
  readonly n: number;

  private buf: Float64Array; // [windowSize, n] residuals (not squared)
  private head: number;
  private count: number;
  private sum: Float64Array;
  private sumsq: Float64Array;

  constructor(windowSize: number, n: number) {
    this.windowSize = windowSize | 0;
    this.n = n | 0;
    this.buf = new Float64Array(this.windowSize * this.n);
    this.head = 0;
    this.count = 0;
    this.sum = new Float64Array(this.n);
    this.sumsq = new Float64Array(this.n);
  }

  reset(): void {
    TensorOps.fill(this.buf, 0);
    TensorOps.fill(this.sum, 0);
    TensorOps.fill(this.sumsq, 0);
    this.head = 0;
    this.count = 0;
  }

  addResiduals(residuals: Float64Array): void {
    const n = this.n;
    const ws = this.windowSize;
    const h = this.head;
    const base = h * n;

    if (this.count === ws) {
      // remove oldest at head position (overwritten)
      for (let i = 0; i < n; i++) {
        const old = this.buf[base + i];
        this.sum[i] -= old;
        this.sumsq[i] -= old * old;
      }
    } else {
      this.count++;
    }

    for (let i = 0; i < n; i++) {
      const r = residuals[i];
      this.buf[base + i] = r;
      this.sum[i] += r;
      this.sumsq[i] += r * r;
    }

    this.head = (h + 1) % ws;
  }

  getCount(): number {
    return this.count | 0;
  }

  mean(i: number): number {
    const c = this.count | 0;
    if (c <= 0) return 0.0;
    return this.sum[i | 0] / c;
  }

  variance(i: number, epsilon: number): number {
    const c = this.count | 0;
    if (c <= 1) return epsilon;
    const m = this.sum[i | 0] / c;
    let v = this.sumsq[i | 0] / c - m * m;
    if (!Number.isFinite(v) || v < epsilon) v = epsilon;
    return v;
  }

  std(i: number, epsilon: number): number {
    return Math.sqrt(this.variance(i, epsilon));
  }

  meanMSE(epsilon: number): number {
    const c = this.count | 0;
    if (c <= 0) return epsilon;
    let s = 0.0;
    for (let i = 0; i < this.n; i++) {
      // mse per target ~ E[r^2] = sumsq/c
      s += this.sumsq[i] / c;
    }
    const v = s / this.n;
    return Number.isFinite(v) && v > epsilon ? v : epsilon;
  }
}

export class OutlierDownweighter {
  readonly threshold: number;
  readonly minWeight: number;
  readonly epsilon: number;

  constructor(threshold: number, minWeight: number, epsilon: number) {
    this.threshold = threshold;
    this.minWeight = minWeight;
    this.epsilon = epsilon;
  }

  computeWeight(residuals: Float64Array, stats: ResidualStatsTracker): number {
    const c = stats.getCount();
    if (c < 10) return 1.0;

    const n = stats.n;
    let maxZ = 0.0;
    for (let i = 0; i < n; i++) {
      const mu = stats.mean(i);
      const sd = stats.std(i, this.epsilon);
      const z = Math.abs((residuals[i] - mu) / (sd + this.epsilon));
      if (z > maxZ) maxZ = z;
    }

    if (maxZ <= this.threshold) return 1.0;
    // smooth downweight: w = threshold / z
    let w = this.threshold / (maxZ + this.epsilon);
    if (w < this.minWeight) w = this.minWeight;
    if (w > 1.0) w = 1.0;
    return w;
  }
}

export class LossFunction {
  static mse(yTrue: Float64Array, yPred: Float64Array): number {
    const n = yTrue.length | 0;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const d = yTrue[i] - yPred[i];
      s += d * d;
    }
    return s / n;
  }
}

export class MetricsAccumulator {
  private sumLoss: number;
  private count: number;

  constructor() {
    this.sumLoss = 0;
    this.count = 0;
  }

  reset(): void {
    this.sumLoss = 0;
    this.count = 0;
  }

  add(loss: number): void {
    this.sumLoss += loss;
    this.count++;
  }

  mean(): number {
    return this.count > 0 ? this.sumLoss / this.count : 0.0;
  }

  getCount(): number {
    return this.count | 0;
  }
}

/** -------------- 7. ESNModel assembly -------------- */

export class ESNModelConfig {
  cfg: ESNRegressionConfig;
  constructor(cfg: ESNRegressionConfig) {
    this.cfg = cfg;
  }
}

export class TrainingState {
  sampleCount: number;
  constructor() {
    this.sampleCount = 0;
  }
  reset(): void {
    this.sampleCount = 0;
  }
}

export class InferenceState {
  constructor() {}
}

export class ESNModel {
  readonly cfg: ESNRegressionConfig;
  readonly nFeatures: number;
  readonly nTargets: number;
  readonly outputDim: number;
  readonly zDim: number;

  readonly ring: RingBuffer;
  readonly normalizer: WelfordNormalizer;
  readonly reservoir: ESNReservoir;
  readonly readoutCfg: ReadoutConfig;
  readonly readoutParams: ReadoutParams;
  readonly readout: LinearReadout;

  readonly rlsState: RLSState;
  readonly rlsOpt: RLSOptimizer;

  readonly residualStats: ResidualStatsTracker;
  readonly outlier: OutlierDownweighter;

  readonly trainState: TrainingState;

  // scratch buffers (no per-call allocations)
  readonly xRaw: Float64Array; // [F]
  readonly xNorm: Float64Array; // [F]
  readonly z: Float64Array; // [zDim]
  readonly yHat: Float64Array; // [outputDim]
  readonly yTrue: Float64Array; // [outputDim]
  readonly residuals: Float64Array; // [outputDim]
  readonly rollState: Float64Array; // [N] scratch for recursive predict
  readonly rollXNorm: Float64Array; // [F]
  readonly rollZ: Float64Array; // [zDim]
  readonly rollYHat: Float64Array; // [nTargets] scratch for recursive steps

  constructor(cfg: ESNRegressionConfig, nFeatures: number, nTargets: number) {
    this.cfg = cfg;
    this.nFeatures = nFeatures | 0;
    this.nTargets = nTargets | 0;

    this.readoutCfg = new ReadoutConfig(cfg);

    const maxFutureSteps = cfg.maxFutureSteps | 0;
    const useDirect = cfg.useDirectMultiHorizon;
    this.outputDim = useDirect
      ? (this.nTargets * maxFutureSteps)
      : this.nTargets;

    const N = cfg.reservoirSize | 0;
    const F = this.nFeatures;
    const zDim = N + (cfg.useInputInReadout ? F : 0) +
      (cfg.useBiasInReadout ? 1 : 0);
    this.zDim = zDim;

    this.ring = new RingBuffer(cfg.maxSequenceLength | 0, F);
    this.normalizer = new WelfordNormalizer(
      F,
      cfg.normalizationEpsilon,
      cfg.normalizationWarmup | 0,
    );

    this.reservoir = new ESNReservoir(new ESNReservoirParams(cfg, F));
    this.readoutParams = new ReadoutParams(this.outputDim, zDim);
    this.readout = new LinearReadout(this.readoutParams);

    this.rlsState = new RLSState(zDim, cfg.rlsLambda, cfg.rlsDelta);
    this.rlsOpt = new RLSOptimizer(
      this.rlsState,
      this.outputDim,
      cfg.epsilon,
      cfg.l2Lambda,
      cfg.gradientClipNorm,
    );

    this.residualStats = new ResidualStatsTracker(
      cfg.residualWindowSize | 0,
      this.outputDim,
    );
    this.outlier = new OutlierDownweighter(
      cfg.outlierThreshold,
      cfg.outlierMinWeight,
      cfg.epsilon,
    );

    this.trainState = new TrainingState();

    this.xRaw = new Float64Array(F);
    this.xNorm = new Float64Array(F);
    this.z = new Float64Array(zDim);
    this.yHat = new Float64Array(this.outputDim);
    this.yTrue = new Float64Array(this.outputDim);
    this.residuals = new Float64Array(this.outputDim);

    this.rollState = new Float64Array(N);
    this.rollXNorm = new Float64Array(F);
    this.rollZ = new Float64Array(zDim);
    this.rollYHat = new Float64Array(this.nTargets);

    // ensure references
    this.rlsOpt.setZRef(this.z);
  }

  reset(): void {
    this.ring.reset();
    this.normalizer.reset();
    this.reservoir.resetState();
    TensorOps.fill(this.readoutParams.Wout, 0);
    this.rlsState.reset();
    this.residualStats.reset();
    this.trainState.reset();
  }

  assembleZFromCurrent(
    rState: Float64Array,
    xNorm: Float64Array,
    zOut: Float64Array,
  ): void {
    const N = this.cfg.reservoirSize | 0;
    let k = 0;

    // r
    for (let i = 0; i < N; i++) zOut[k++] = rState[i];

    // x
    if (this.cfg.useInputInReadout) {
      const F = this.nFeatures | 0;
      for (let j = 0; j < F; j++) zOut[k++] = xNorm[j];
    }

    // bias
    if (this.cfg.useBiasInReadout) zOut[k++] = 1.0;
  }
}

/** -------------- 8. Serialization -------------- */

export class SerializationHelper {
  static f64ToArray(x: Float64Array): number[] {
    const n = x.length | 0;
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) out[i] = x[i];
    return out;
  }

  static arrayToF64(arr: number[], out: Float64Array): void {
    const n = out.length | 0;
    if (arr.length !== n) {
      throw new Error("SerializationHelper.arrayToF64: length mismatch");
    }
    for (let i = 0; i < n; i++) out[i] = arr[i];
  }
}

/** -------------- 9. ESNRegression public API -------------- */

export class ESNRegression {
  readonly cfg: ESNRegressionConfig;

  private model: ESNModel | null;
  private initialized: boolean;

  private fitResult: FitResult;
  private predResult: PredictionResult;
  private predPredictions: number[][];
  private predLower: number[][];
  private predUpper: number[][];

  private metrics: MetricsAccumulator;

  constructor(config?: Partial<ESNRegressionConfig>) {
    const defaults: ESNRegressionConfig = {
      maxSequenceLength: 64,
      maxFutureSteps: 1,
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
      useDirectMultiHorizon: true,
      residualWindowSize: 100,
      uncertaintyMultiplier: 1.96,
      weightInitScale: 0.1,
      seed: 42,
      verbose: false,
    };
    this.cfg = Object.assign({}, defaults, config || {});
    if (!(this.cfg.maxSequenceLength > 0)) {
      throw new Error("maxSequenceLength must be > 0");
    }
    if (!(this.cfg.maxFutureSteps > 0)) {
      throw new Error("maxFutureSteps must be > 0");
    }
    if (!(this.cfg.reservoirSize > 0)) {
      throw new Error("reservoirSize must be > 0");
    }
    if (!(this.cfg.rlsLambda > 0 && this.cfg.rlsLambda <= 1.0)) {
      throw new Error("rlsLambda must be in (0,1]");
    }
    if (!(this.cfg.leakRate > 0 && this.cfg.leakRate <= 1.0)) {
      throw new Error("leakRate must be in (0,1]");
    }

    this.model = null;
    this.initialized = false;

    this.fitResult = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1,
    };

    // allocate empty prediction shells; fully allocated after init
    this.predPredictions = [];
    this.predLower = [];
    this.predUpper = [];
    this.predResult = {
      predictions: this.predPredictions,
      lowerBounds: this.predLower,
      upperBounds: this.predUpper,
      confidence: 0,
    };

    this.metrics = new MetricsAccumulator();
  }

  private ensureInitializedFromBatch(
    xCoordinates: number[][],
    yCoordinates: number[][],
  ): void {
    if (this.initialized) return;
    if (xCoordinates.length <= 0) throw new Error("fitOnline: empty batch");
    if (yCoordinates.length <= 0) throw new Error("fitOnline: empty batch");

    const nFeatures = xCoordinates[0].length | 0;
    const yLen = yCoordinates[0].length | 0;
    if (nFeatures <= 0) throw new Error("nFeatures must be > 0");
    if (yLen <= 0) throw new Error("nTargets must be > 0");

    let nTargets: number;
    if (this.cfg.useDirectMultiHorizon && (this.cfg.maxFutureSteps | 0) > 1) {
      const mfs = this.cfg.maxFutureSteps | 0;
      if (yLen % mfs !== 0) {
        throw new Error(
          "yCoordinates[0].length must be divisible by maxFutureSteps when useDirectMultiHorizon=true",
        );
      }
      nTargets = (yLen / mfs) | 0;
      if (nTargets <= 0) throw new Error("Invalid nTargets inferred");
    } else {
      nTargets = yLen | 0;
    }

    this.model = new ESNModel(this.cfg, nFeatures, nTargets);
    this.initialized = true;

    // allocate prediction arrays (reused per call)
    const steps = this.cfg.maxFutureSteps | 0;
    const T = nTargets | 0;
    this.predPredictions = new Array<number[]>(steps);
    this.predLower = new Array<number[]>(steps);
    this.predUpper = new Array<number[]>(steps);
    for (let s = 0; s < steps; s++) {
      const p = new Array<number>(T);
      const l = new Array<number>(T);
      const u = new Array<number>(T);
      for (let j = 0; j < T; j++) {
        p[j] = 0;
        l[j] = 0;
        u[j] = 0;
      }
      this.predPredictions[s] = p;
      this.predLower[s] = l;
      this.predUpper[s] = u;
    }
    this.predResult.predictions = this.predPredictions;
    this.predResult.lowerBounds = this.predLower;
    this.predResult.upperBounds = this.predUpper;
    this.predResult.confidence = 0;
  }

  /**
   * Online training (one sample at a time).
   * NOTE: fitOnline does NOT support observe-only; xCoordinates.length must equal yCoordinates.length.
   * The model always ingests X into internal RingBuffer BEFORE any training decision.
   *
   * @param param0 xCoordinates: number[][], yCoordinates: number[][]
   * @returns FitResult (reused object, no per-call allocations after init)
   *
   * @example
   * const esn = new ESNRegression({ maxFutureSteps: 3, useDirectMultiHorizon: true });
   * esn.fitOnline({ xCoordinates: [[1,2],[2,3],[3,4]], yCoordinates: [[10,11,12],[20,21,22],[30,31,32]] });
   */
  fitOnline(
    { xCoordinates, yCoordinates }: {
      xCoordinates: number[][];
      yCoordinates: number[][];
    },
  ): FitResult {
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        "fitOnline: xCoordinates.length must equal yCoordinates.length",
      );
    }
    const N = xCoordinates.length | 0;
    if (N === 0) {
      this.fitResult.samplesProcessed = 0;
      this.fitResult.averageLoss = 0;
      this.fitResult.gradientNorm = 0;
      this.fitResult.driftDetected = false;
      this.fitResult.sampleWeight = 1;
      return this.fitResult;
    }

    this.ensureInitializedFromBatch(xCoordinates, yCoordinates);
    const m = this.model as ESNModel;

    // Validate dimensions (strict)
    const F = m.nFeatures | 0;
    const maxFutureSteps = this.cfg.maxFutureSteps | 0;
    const expectedYLen = m.outputDim | 0;

    this.metrics.reset();
    let lastUpdateNorm = 0.0;
    let lastWeight = 1.0;

    for (let i = 0; i < N; i++) {
      const xRow = xCoordinates[i];
      const yRow = yCoordinates[i];

      if ((xRow.length | 0) !== F) {
        throw new Error("fitOnline: x row length mismatch");
      }
      if ((yRow.length | 0) !== expectedYLen) {
        if (this.cfg.useDirectMultiHorizon) {
          throw new Error(
            "fitOnline: y row length mismatch (expected nTargets*maxFutureSteps for direct multi-horizon)",
          );
        } else {
          throw new Error(
            "fitOnline: y row length mismatch (expected nTargets for single-step)",
          );
        }
      }

      // 1) Push X into ring buffer FIRST (critical requirement)
      m.ring.pushRow(xRow);

      // Copy latest X from ring to xRaw (authoritative internal latest-X)
      m.ring.copyLatestRow(m.xRaw);

      // 2) Normalization observe and normalize
      m.normalizer.observe(m.xRaw);
      m.normalizer.normalize(m.xRaw, m.xNorm);

      // 3) Reservoir update using latest normalized x
      m.reservoir.update(m.xNorm);

      // 4) Assemble z = [r; x; 1]
      m.assembleZFromCurrent(m.reservoir.r, m.xNorm, m.z);

      // 5) Forward (train)
      m.readout.forward(m.z, m.yHat);

      // yTrue copy (avoid allocations)
      for (let k = 0; k < expectedYLen; k++) m.yTrue[k] = yRow[k];

      // residuals
      for (let k = 0; k < expectedYLen; k++) {
        m.residuals[k] = m.yTrue[k] - m.yHat[k];
      }

      // 6) Loss
      const loss = LossFunction.mse(m.yTrue, m.yHat);

      // 7) Outlier weight (based on recent residual stats)
      const w = m.outlier.computeWeight(m.residuals, m.residualStats);
      lastWeight = w;

      // 8) RLS update
      m.rlsOpt.computeGainAndUpdateP();
      m.rlsOpt.applyWeightUpdate(m.readoutParams.Wout, m.residuals, w);
      lastUpdateNorm = m.rlsOpt.getLastUpdateNorm();

      // 9) Update residual stats after training step
      m.residualStats.addResiduals(m.residuals);

      // accounting
      m.trainState.sampleCount++;
      this.metrics.add(loss);
    }

    this.fitResult.samplesProcessed = N;
    this.fitResult.averageLoss = this.metrics.mean();
    this.fitResult.gradientNorm = lastUpdateNorm;
    this.fitResult.driftDetected = false;
    this.fitResult.sampleWeight = lastWeight;
    return this.fitResult;
  }

  /**
   * Predict multiple future steps.
   * Uses ONLY internal RingBuffer for the latest input row; never requires caller to pass latest X.
   * The input window ends at the most recently ingested X (no off-by-one).
   *
   * @param futureSteps number of steps to predict (1..maxFutureSteps)
   * @returns PredictionResult (reused object; predictions are [step][target])
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.initialized || !this.model) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }
    const fs = futureSteps | 0;
    const m = this.model;

    if (fs < 1 || fs > (this.cfg.maxFutureSteps | 0)) {
      throw new Error("predict: futureSteps out of range");
    }
    if (m.ring.length() <= 0) {
      throw new Error("predict: no input history available");
    }

    // Latest-X is authoritative and comes from ring buffer
    m.ring.copyLatestRow(m.xRaw);
    m.normalizer.normalize(m.xRaw, m.xNorm);

    const nTargets = m.nTargets | 0;
    const maxFutureSteps = this.cfg.maxFutureSteps | 0;
    const useDirect = this.cfg.useDirectMultiHorizon;

    // confidence based on recent residual stats
    const mse = m.residualStats.meanMSE(this.cfg.epsilon);
    const rmse = Math.sqrt(mse);
    let conf = 1.0 / (1.0 + rmse);
    if (!Number.isFinite(conf) || conf < 0) conf = 0;
    if (conf > 1) conf = 1;
    this.predResult.confidence = conf;

    const mult = this.cfg.uncertaintyMultiplier;

    if (useDirect) {
      // Use current live reservoir state (already includes latest ingested X during training),
      // but still assemble z using latest X from RingBuffer (critical behavior).
      m.assembleZFromCurrent(m.reservoir.r, m.xNorm, m.z);
      m.readout.forward(m.z, m.yHat);

      // yHat layout: [step0 targets..., step1 targets..., ...]
      for (let s = 0; s < fs; s++) {
        const base = s * nTargets;
        const rowP = this.predPredictions[s];
        const rowL = this.predLower[s];
        const rowU = this.predUpper[s];
        for (let t = 0; t < nTargets; t++) {
          const y = m.yHat[base + t];
          const sd = m.residualStats.std(base + t, this.cfg.epsilon);
          const d = mult * sd;
          rowP[t] = y;
          rowL[t] = y - d;
          rowU[t] = y + d;
        }
      }

      // For steps beyond requested, keep deterministic but do not modify (avoid extra loops)
      for (let s = fs; s < maxFutureSteps; s++) {
        const rowP = this.predPredictions[s];
        const rowL = this.predLower[s];
        const rowU = this.predUpper[s];
        for (let t = 0; t < nTargets; t++) {
          rowP[t] = rowP[t];
          rowL[t] = rowL[t];
          rowU[t] = rowU[t];
        }
      }
      return this.predResult;
    }

    // Recursive roll-forward (explicitly enabled by useDirectMultiHorizon=false).
    // This implementation does NOT feed predicted Y back into X, and does NOT mutate RingBuffer or live reservoir state.
    // It rolls the reservoir forward by repeatedly applying the same latest X (held constant).
    TensorOps.copy(m.rollState, m.reservoir.r);
    TensorOps.copy(m.rollXNorm, m.xNorm);

    for (let s = 0; s < fs; s++) {
      // update scratch state using constant x
      // We need an update that operates on scratch state; implement inline without allocating.
      const N = this.cfg.reservoirSize | 0;
      const F = m.nFeatures | 0;
      const a = this.cfg.leakRate;
      const oneMinusA = 1.0 - a;
      const inputScale = this.cfg.inputScale;
      const Win = m.reservoir.Win;
      const W = m.reservoir.W;
      const b = m.reservoir.bias;

      // reuse model's internal preAct buffer would mutate live; use rollZ temporarily? It is zDim.
      // We'll use rollZ[0..N) as preAct scratch, since zDim >= N.
      const pre = m.rollZ; // scratch
      for (let i = 0; i < N; i++) pre[i] = b[i];

      let idx = 0;
      for (let i = 0; i < N; i++) {
        let acc = pre[i];
        for (let j = 0; j < F; j++) {
          acc += Win[idx++] * (m.rollXNorm[j] * inputScale);
        }
        pre[i] = acc;
      }

      idx = 0;
      for (let i = 0; i < N; i++) {
        let acc = pre[i];
        for (let j = 0; j < N; j++) acc += W[idx++] * m.rollState[j];
        pre[i] = acc;
      }

      // activation
      if (this.cfg.activation === "tanh") {
        for (let i = 0; i < N; i++) pre[i] = Math.tanh(pre[i]);
      } else {
        for (let i = 0; i < N; i++) {
          const v = pre[i];
          pre[i] = v > 0 ? v : 0;
        }
      }

      for (let i = 0; i < N; i++) {
        m.rollState[i] = oneMinusA * m.rollState[i] + a * pre[i];
      }

      // assemble rollZ as z=[rollState; x; 1]
      m.assembleZFromCurrent(m.rollState, m.rollXNorm, m.rollZ);

      // forward readout (single-step): yHat size is nTargets
      // use model.yHat scratch with first nTargets to avoid allocations
      TensorOps.matVec(
        m.readoutParams.Wout,
        m.nTargets,
        m.zDim,
        m.rollZ,
        m.rollYHat,
      );

      const rowP = this.predPredictions[s];
      const rowL = this.predLower[s];
      const rowU = this.predUpper[s];
      for (let t = 0; t < nTargets; t++) {
        const y = m.rollYHat[t];
        const sd = m.residualStats.std(t, this.cfg.epsilon);
        const d = mult * sd;
        rowP[t] = y;
        rowL[t] = y - d;
        rowU[t] = y + d;
      }
    }

    return this.predResult;
  }

  getModelSummary(): ModelSummary {
    if (!this.initialized || !this.model) {
      return {
        totalParameters: 0,
        receptiveField: Math.max(1, this.cfg.maxSequenceLength | 0),
        spectralRadius: this.cfg.spectralRadius,
        reservoirSize: this.cfg.reservoirSize | 0,
        nFeatures: 0,
        nTargets: 0,
        maxSequenceLength: this.cfg.maxSequenceLength | 0,
        maxFutureSteps: this.cfg.maxFutureSteps | 0,
        sampleCount: 0,
        useDirectMultiHorizon: this.cfg.useDirectMultiHorizon,
      };
    }
    const m = this.model;
    const totalParameters = m.readoutParams.Wout.length | 0; // only trainable
    // approximate receptive field from leakRate: effective horizon ~ 1/(1-leak) capped by maxSequenceLength
    const a = this.cfg.leakRate;
    let rf = 1;
    if (a > 0 && a < 1) {
      rf = Math.min(
        this.cfg.maxSequenceLength | 0,
        Math.max(1, Math.round(1.0 / (1.0 - a))),
      );
    } else rf = Math.min(this.cfg.maxSequenceLength | 0, 1);

    return {
      totalParameters,
      receptiveField: rf,
      spectralRadius: this.cfg.spectralRadius,
      reservoirSize: this.cfg.reservoirSize | 0,
      nFeatures: m.nFeatures | 0,
      nTargets: m.nTargets | 0,
      maxSequenceLength: this.cfg.maxSequenceLength | 0,
      maxFutureSteps: this.cfg.maxFutureSteps | 0,
      sampleCount: m.trainState.sampleCount | 0,
      useDirectMultiHorizon: this.cfg.useDirectMultiHorizon,
    };
  }

  getWeights(): WeightInfo {
    if (!this.initialized || !this.model) return { weights: [] };
    const m = this.model;
    return {
      weights: [
        {
          name: "Win",
          shape: [this.cfg.reservoirSize | 0, m.nFeatures | 0],
          values: SerializationHelper.f64ToArray(m.reservoir.Win),
        },
        {
          name: "W",
          shape: [this.cfg.reservoirSize | 0, this.cfg.reservoirSize | 0],
          values: SerializationHelper.f64ToArray(m.reservoir.W),
        },
        {
          name: "bias",
          shape: [this.cfg.reservoirSize | 0],
          values: SerializationHelper.f64ToArray(m.reservoir.bias),
        },
        {
          name: "Wout",
          shape: [m.outputDim | 0, m.zDim | 0],
          values: SerializationHelper.f64ToArray(m.readoutParams.Wout),
        },
      ],
    };
  }

  getNormalizationStats(): NormalizationStats {
    if (!this.initialized || !this.model) {
      return { means: [], stds: [], count: 0, isActive: false };
    }
    const m = this.model;
    return {
      means: SerializationHelper.f64ToArray(m.normalizer.getMeansArray()),
      stds: SerializationHelper.f64ToArray(m.normalizer.getStdsArray()),
      count: m.normalizer.getCount(),
      isActive: m.normalizer.isActive(),
    };
  }

  reset(): void {
    if (this.model) this.model.reset();
  }

  save(): string {
    const m = this.model;
    const payload: any = {
      version: 1,
      cfg: this.cfg,
      initialized: this.initialized,
      state: null as any,
    };

    if (this.initialized && m) {
      payload.state = {
        nFeatures: m.nFeatures | 0,
        nTargets: m.nTargets | 0,
        outputDim: m.outputDim | 0,
        zDim: m.zDim | 0,

        ring: {
          data: SerializationHelper.f64ToArray(m.ring.getRawData()),
          head: m.ring.getHead(),
          size: m.ring.getSize(),
          totalPushed: m.ring.totalCount(),
        },

        normalizer: {
          count: m.normalizer.getCount(),
          active: m.normalizer.isActive(),
          means: SerializationHelper.f64ToArray(m.normalizer.getMeansArray()),
          stds: SerializationHelper.f64ToArray(m.normalizer.getStdsArray()),
          // m2 is internal; not exposed; reconstructing stds is enough for inference, but for determinism re-fit, store m2 too:
          m2: SerializationHelper.f64ToArray(
            (m as any).normalizer["m2"] as Float64Array,
          ),
        },

        reservoir: {
          r: SerializationHelper.f64ToArray(m.reservoir.r),
          Win: SerializationHelper.f64ToArray(m.reservoir.Win),
          W: SerializationHelper.f64ToArray(m.reservoir.W),
          bias: SerializationHelper.f64ToArray(m.reservoir.bias),
        },

        readout: {
          Wout: SerializationHelper.f64ToArray(m.readoutParams.Wout),
        },

        rls: {
          P: SerializationHelper.f64ToArray(m.rlsState.P),
        },

        residualStats: {
          buf: SerializationHelper.f64ToArray(
            (m as any).residualStats["buf"] as Float64Array,
          ),
          head: (m as any).residualStats["head"] as number,
          count: m.residualStats.getCount(),
          sum: SerializationHelper.f64ToArray(
            (m as any).residualStats["sum"] as Float64Array,
          ),
          sumsq: SerializationHelper.f64ToArray(
            (m as any).residualStats["sumsq"] as Float64Array,
          ),
        },

        trainState: {
          sampleCount: m.trainState.sampleCount | 0,
        },
      };
    }

    return JSON.stringify(payload);
  }

  load(w: string): void {
    const obj = JSON.parse(w);
    if (!obj || obj.version !== 1) throw new Error("load: invalid version");
    const cfg: ESNRegressionConfig = obj.cfg;
    // keep current config object, but validate compatibility
    // (We allow load to override internal cfg by exact match expectation.)
    // For strict determinism, require matching essential hyperparameters.
    const mustMatch: (keyof ESNRegressionConfig)[] = [
      "maxSequenceLength",
      "maxFutureSteps",
      "reservoirSize",
      "spectralRadius",
      "leakRate",
      "inputScale",
      "biasScale",
      "reservoirSparsity",
      "inputSparsity",
      "activation",
      "useInputInReadout",
      "useBiasInReadout",
      "readoutTraining",
      "rlsLambda",
      "rlsDelta",
      "epsilon",
      "l2Lambda",
      "gradientClipNorm",
      "normalizationEpsilon",
      "normalizationWarmup",
      "outlierThreshold",
      "outlierMinWeight",
      "useDirectMultiHorizon",
      "residualWindowSize",
      "uncertaintyMultiplier",
      "weightInitScale",
      "seed",
      "verbose",
    ];
    for (let i = 0; i < mustMatch.length; i++) {
      const k = mustMatch[i];
      if ((this.cfg as any)[k] !== (cfg as any)[k]) {
        throw new Error(`load: config mismatch on ${String(k)}`);
      }
    }

    if (!obj.initialized || !obj.state) {
      this.model = null;
      this.initialized = false;
      return;
    }

    const st = obj.state;
    const nFeatures = st.nFeatures | 0;
    const nTargets = st.nTargets | 0;

    this.model = new ESNModel(this.cfg, nFeatures, nTargets);
    this.initialized = true;

    const m = this.model;

    // ring
    SerializationHelper.arrayToF64(st.ring.data, m.ring.getRawData());
    m.ring.setInternal(
      m.ring.getRawData(),
      st.ring.head | 0,
      st.ring.size | 0,
      st.ring.totalPushed | 0,
    );

    // normalizer
    (m as any).normalizer["count"] = st.normalizer.count | 0;
    (m as any).normalizer["active"] = !!st.normalizer.active;
    SerializationHelper.arrayToF64(
      st.normalizer.means,
      m.normalizer.getMeansArray(),
    );
    SerializationHelper.arrayToF64(
      st.normalizer.stds,
      m.normalizer.getStdsArray(),
    );
    SerializationHelper.arrayToF64(
      st.normalizer.m2,
      (m as any).normalizer["m2"] as Float64Array,
    );

    // reservoir
    SerializationHelper.arrayToF64(st.reservoir.r, m.reservoir.r);
    SerializationHelper.arrayToF64(st.reservoir.Win, m.reservoir.Win);
    SerializationHelper.arrayToF64(st.reservoir.W, m.reservoir.W);
    SerializationHelper.arrayToF64(st.reservoir.bias, m.reservoir.bias);

    // readout
    SerializationHelper.arrayToF64(st.readout.Wout, m.readoutParams.Wout);

    // rls
    SerializationHelper.arrayToF64(st.rls.P, m.rlsState.P);

    // residual stats
    SerializationHelper.arrayToF64(
      st.residualStats.buf,
      (m as any).residualStats["buf"] as Float64Array,
    );
    (m as any).residualStats["head"] = st.residualStats.head | 0;
    (m as any).residualStats["count"] = st.residualStats.count | 0;
    SerializationHelper.arrayToF64(
      st.residualStats.sum,
      (m as any).residualStats["sum"] as Float64Array,
    );
    SerializationHelper.arrayToF64(
      st.residualStats.sumsq,
      (m as any).residualStats["sumsq"] as Float64Array,
    );

    // trainState
    m.trainState.sampleCount = st.trainState.sampleCount | 0;

    // prediction shells
    const steps = this.cfg.maxFutureSteps | 0;
    this.predPredictions = new Array<number[]>(steps);
    this.predLower = new Array<number[]>(steps);
    this.predUpper = new Array<number[]>(steps);
    for (let s = 0; s < steps; s++) {
      const p = new Array<number>(nTargets);
      const l = new Array<number>(nTargets);
      const u = new Array<number>(nTargets);
      for (let j = 0; j < nTargets; j++) {
        p[j] = 0;
        l[j] = 0;
        u[j] = 0;
      }
      this.predPredictions[s] = p;
      this.predLower[s] = l;
      this.predUpper[s] = u;
    }
    this.predResult.predictions = this.predPredictions;
    this.predResult.lowerBounds = this.predLower;
    this.predResult.upperBounds = this.predUpper;
    this.predResult.confidence = 0;
  }
}
