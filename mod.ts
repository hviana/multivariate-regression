/******************************************************
 * ESNRegression - Improved Echo State Network Library
 * Single-file TypeScript implementation
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
  maxSequenceLength: number;
  maxFutureSteps: number;
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
  useDirectMultiHorizon: boolean;
  residualWindowSize: number;
  uncertaintyMultiplier: number;
  weightInitScale: number;
  seed: number;
  verbose: boolean;
}

/* ============================================================
   1. Memory Infrastructure
   ============================================================ */

export class TensorShape {
  readonly dims: number[];
  readonly size: number;

  constructor(dims: number[]) {
    this.dims = dims.slice();
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
    this.strides = strides
      ? strides.slice()
      : TensorOps.computeRowMajorStrides(shape.dims);
  }
}

export class BufferPool {
  private pools: Map<number, Float64Array[]> = new Map();

  rent(size: number): Float64Array {
    const s = size | 0;
    const pool = this.pools.get(s);
    if (pool && pool.length > 0) return pool.pop()!;
    return new Float64Array(s);
  }

  release(buf: Float64Array): void {
    const s = buf.length;
    let pool = this.pools.get(s);
    if (!pool) {
      pool = [];
      this.pools.set(s, pool);
    }
    if (pool.length < 16) pool.push(buf);
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
    const o = this.offset;
    if (o + s > this.buffer.length) throw new Error("TensorArena exhausted");
    this.offset = o + s;
    return this.buffer.subarray(o, o + s);
  }
}

export class TensorOps {
  static computeRowMajorStrides(dims: number[]): number[] {
    const nd = dims.length;
    const strides = new Array<number>(nd);
    let acc = 1;
    for (let i = nd - 1; i >= 0; i--) {
      strides[i] = acc;
      acc *= dims[i] | 0;
    }
    return strides;
  }

  static fill(x: Float64Array, v: number): void {
    const n = x.length;
    for (let i = 0; i < n; i++) x[i] = v;
  }

  static copy(dst: Float64Array, src: Float64Array): void {
    const n = Math.min(dst.length, src.length);
    for (let i = 0; i < n; i++) dst[i] = src[i];
  }

  static dot(a: Float64Array, b: Float64Array): number {
    const n = a.length;
    let s = 0.0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }

  static norm2(a: Float64Array): number {
    const n = a.length;
    let s = 0.0;
    for (let i = 0; i < n; i++) s += a[i] * a[i];
    return Math.sqrt(s);
  }

  static axpy(y: Float64Array, a: number, x: Float64Array): void {
    const n = y.length;
    for (let i = 0; i < n; i++) y[i] += a * x[i];
  }

  static scale(x: Float64Array, a: number): void {
    const n = x.length;
    for (let i = 0; i < n; i++) x[i] *= a;
  }

  static matVec(
    A: Float64Array,
    m: number,
    n: number,
    x: Float64Array,
    y: Float64Array,
  ): void {
    let idx = 0;
    for (let i = 0; i < m; i++) {
      let s = 0.0;
      for (let j = 0; j < n; j++) s += A[idx++] * x[j];
      y[i] = s;
    }
  }

  static matVecAddBias(
    W: Float64Array,
    m: number,
    n: number,
    x: Float64Array,
    b: Float64Array | null,
    y: Float64Array,
  ): void {
    let idx = 0;
    for (let i = 0; i < m; i++) {
      let s = b ? b[i] : 0.0;
      for (let j = 0; j < n; j++) s += W[idx++] * x[j];
      y[i] = s;
    }
  }

  static clipInPlace(x: Float64Array, minVal: number, maxVal: number): void {
    const n = x.length;
    for (let i = 0; i < n; i++) {
      const v = x[i];
      x[i] = v < minVal ? minVal : (v > maxVal ? maxVal : v);
    }
  }

  static hasNonFinite(x: Float64Array): boolean {
    const n = x.length;
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(x[i])) return true;
    }
    return false;
  }

  static maxAbs(x: Float64Array): number {
    const n = x.length;
    let m = 0.0;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(x[i]);
      if (a > m) m = a;
    }
    return m;
  }
}

/* ============================================================
   2. Numerics: RNG, Activations, Normalizers
   ============================================================ */

export class ActivationOps {
  static applyInPlace(x: Float64Array, kind: "tanh" | "relu"): void {
    const n = x.length;
    if (kind === "tanh") {
      for (let i = 0; i < n; i++) x[i] = Math.tanh(x[i]);
    } else {
      for (let i = 0; i < n; i++) x[i] = x[i] > 0 ? x[i] : 0;
    }
  }
}

export class RandomGenerator {
  private state: number;

  constructor(seed: number) {
    this.state = ((seed | 0) >>> 0) || 0x9e3779b9;
  }

  nextUint32(): number {
    let x = this.state >>> 0;
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x >>>= 0;
    x ^= x << 5;
    x >>>= 0;
    this.state = x;
    return x >>> 0;
  }

  nextFloat(): number {
    return (this.nextUint32() >>> 0) / 4294967296.0;
  }

  nextSignedFloat(): number {
    return this.nextFloat() * 2.0 - 1.0;
  }

  nextGaussian(): number {
    let u = this.nextFloat();
    if (u < 1e-15) u = 1e-15;
    const v = this.nextFloat();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
}

export class WelfordAccumulator {
  count: number = 0;
  mean: number = 0;
  m2: number = 0;

  reset(): void {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }

  add(x: number): void {
    this.count++;
    const delta = x - this.mean;
    this.mean += delta / this.count;
    const delta2 = x - this.mean;
    this.m2 += delta * delta2;
  }

  variance(epsilon: number): number {
    if (this.count < 2) return epsilon;
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

  private count: number = 0;
  private means: Float64Array;
  private m2: Float64Array;
  private stds: Float64Array;
  private varFloor: number;
  private active: boolean = false;

  constructor(n: number, epsilon: number, warmup: number) {
    this.n = n | 0;
    this.epsilon = epsilon;
    this.warmup = warmup | 0;
    this.varFloor = epsilon * 100;
    this.means = new Float64Array(this.n);
    this.m2 = new Float64Array(this.n);
    this.stds = new Float64Array(this.n);
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
    const n = this.n;
    this.count++;
    const c = this.count;

    for (let i = 0; i < n; i++) {
      const xi = x[i];
      if (!Number.isFinite(xi)) continue;
      const delta = xi - this.means[i];
      this.means[i] += delta / c;
      const delta2 = xi - this.means[i];
      this.m2[i] += delta * delta2;
    }

    if (c >= this.warmup) {
      this.active = true;
      this.updateStds();
    }
  }

  private updateStds(): void {
    const n = this.n;
    const denom = this.count > 1 ? (this.count - 1) : 1;
    for (let i = 0; i < n; i++) {
      let v = this.m2[i] / denom;
      if (!Number.isFinite(v) || v < this.varFloor) v = this.varFloor;
      this.stds[i] = Math.sqrt(v);
    }
  }

  normalize(x: Float64Array, out: Float64Array): void {
    const n = this.n;
    if (!this.active) {
      for (let i = 0; i < n; i++) out[i] = x[i];
      return;
    }
    for (let i = 0; i < n; i++) {
      const s = this.stds[i] > this.epsilon ? this.stds[i] : this.epsilon;
      out[i] = (x[i] - this.means[i]) / s;
    }
  }

  denormalize(y: Float64Array, out: Float64Array): void {
    const n = Math.min(y.length, out.length, this.n);
    if (!this.active) {
      for (let i = 0; i < n; i++) out[i] = y[i];
      return;
    }
    for (let i = 0; i < n; i++) {
      out[i] = y[i] * this.stds[i] + this.means[i];
    }
  }

  denormalizeValue(y: number, idx: number): number {
    if (!this.active || idx >= this.n) return y;
    return y * this.stds[idx] + this.means[idx];
  }

  getStd(idx: number): number {
    return idx < this.n ? this.stds[idx] : 1.0;
  }

  getMean(idx: number): number {
    return idx < this.n ? this.means[idx] : 0.0;
  }

  getCount(): number {
    return this.count;
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
  getM2Array(): Float64Array {
    return this.m2;
  }

  setInternal(
    count: number,
    active: boolean,
    means: number[],
    stds: number[],
    m2: number[],
  ): void {
    this.count = count;
    this.active = active;
    for (let i = 0; i < this.n && i < means.length; i++) {
      this.means[i] = means[i];
    }
    for (let i = 0; i < this.n && i < stds.length; i++) this.stds[i] = stds[i];
    for (let i = 0; i < this.n && i < m2.length; i++) this.m2[i] = m2[i];
  }
}

export class LayerNormParams {
  gamma: Float64Array;
  beta: Float64Array;
  epsilon: number;

  constructor(size: number, epsilon: number) {
    this.gamma = new Float64Array(size);
    this.beta = new Float64Array(size);
    this.epsilon = epsilon;
    for (let i = 0; i < size; i++) this.gamma[i] = 1.0;
  }
}

export class LayerNormOps {
  static forward(
    x: Float64Array,
    params: LayerNormParams,
    out: Float64Array,
  ): void {
    const n = x.length;
    let mean = 0.0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let variance = 0.0;
    for (let i = 0; i < n; i++) {
      const d = x[i] - mean;
      variance += d * d;
    }
    variance /= n;
    const inv = 1.0 / Math.sqrt(variance + params.epsilon);
    for (let i = 0; i < n; i++) {
      out[i] = ((x[i] - mean) * inv) * params.gamma[i] + params.beta[i];
    }
  }
}

export class GradientAccumulator {
  grad: Float64Array;
  constructor(size: number) {
    this.grad = new Float64Array(size);
  }
  reset(): void {
    TensorOps.fill(this.grad, 0);
  }
}

/* ============================================================
   3. RLS Optimizer with Proper Regularization
   ============================================================ */

export class RLSState {
  readonly dim: number;
  readonly lambda: number;
  readonly delta: number;
  readonly l2Lambda: number;

  P: Float64Array;
  k: Float64Array;
  Pz: Float64Array;
  private updateCount: number = 0;
  private stabilizationInterval: number;

  constructor(dim: number, lambda: number, delta: number, l2Lambda: number) {
    this.dim = dim | 0;
    this.lambda = Math.max(0.9, Math.min(1.0, lambda));
    this.delta = Math.max(0.01, delta);
    this.l2Lambda = Math.max(0, l2Lambda);
    this.stabilizationInterval = Math.max(50, dim);

    this.P = new Float64Array(this.dim * this.dim);
    this.k = new Float64Array(this.dim);
    this.Pz = new Float64Array(this.dim);
    this.reset();
  }

  reset(): void {
    TensorOps.fill(this.P, 0);
    const inv = 1.0 / this.delta;
    const d = this.dim;
    for (let i = 0; i < d; i++) this.P[i * d + i] = inv;
    TensorOps.fill(this.k, 0);
    TensorOps.fill(this.Pz, 0);
    this.updateCount = 0;
  }

  incrementUpdateCount(): void {
    this.updateCount++;
  }

  shouldStabilize(): boolean {
    return this.updateCount > 0 &&
      (this.updateCount % this.stabilizationInterval) === 0;
  }

  stabilizeP(epsilon: number): void {
    const d = this.dim;
    const reg = this.l2Lambda * 0.1 + epsilon;
    for (let i = 0; i < d; i++) {
      const idx = i * d + i;
      this.P[idx] += reg;
      if (!Number.isFinite(this.P[idx]) || this.P[idx] > 1e10) {
        this.P[idx] = 1.0 / this.delta;
      }
    }
  }
}

export class RLSOptimizer {
  readonly dim: number;
  readonly outDim: number;
  readonly epsilon: number;
  readonly gradientClipNorm: number;

  private state: RLSState;
  private z: Float64Array;
  private lastUpdateNorm: number = 0;

  constructor(
    state: RLSState,
    outDim: number,
    epsilon: number,
    gradientClipNorm: number,
  ) {
    this.state = state;
    this.dim = state.dim;
    this.outDim = outDim | 0;
    this.epsilon = epsilon;
    this.gradientClipNorm = gradientClipNorm;
    this.z = new Float64Array(this.dim);
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

  computeGainAndUpdateP(): number {
    const d = this.dim;
    const P = this.state.P;
    const Pz = this.state.Pz;
    const z = this.z;
    const k = this.state.k;
    const lambda = this.state.lambda;
    const l2 = this.state.l2Lambda;

    // Pz = P * z
    for (let i = 0; i < d; i++) {
      let s = 0.0;
      const row = i * d;
      for (let j = 0; j < d; j++) s += P[row + j] * z[j];
      Pz[i] = s;
    }

    // denom = lambda + z^T P z + l2Lambda (Tikhonov regularization in denominator)
    let zTPz = 0.0;
    for (let i = 0; i < d; i++) zTPz += z[i] * Pz[i];

    let denom = lambda + zTPz + l2;
    if (!Number.isFinite(denom) || denom < this.epsilon) denom = this.epsilon;

    // k = Pz / denom
    const invDen = 1.0 / denom;
    for (let i = 0; i < d; i++) {
      k[i] = Pz[i] * invDen;
      if (!Number.isFinite(k[i])) k[i] = 0;
    }

    // P = (P - k * Pz^T) / lambda
    const invLam = 1.0 / lambda;
    for (let i = 0; i < d; i++) {
      const ki = k[i];
      const row = i * d;
      for (let j = 0; j < d; j++) {
        let v = (P[row + j] - ki * Pz[j]) * invLam;
        if (!Number.isFinite(v)) v = (i === j) ? (1.0 / this.state.delta) : 0;
        P[row + j] = v;
      }
    }

    this.state.incrementUpdateCount();
    if (this.state.shouldStabilize()) {
      this.state.stabilizeP(this.epsilon);
    }

    return denom;
  }

  applyWeightUpdate(
    Wout: Float64Array,
    errors: Float64Array,
    sampleWeight: number,
  ): void {
    const d = this.dim;
    const outD = this.outDim;
    const k = this.state.k;
    const clip = this.gradientClipNorm;

    // Compute gain norm for clipping
    let kNorm = 0.0;
    for (let i = 0; i < d; i++) kNorm += k[i] * k[i];
    kNorm = Math.sqrt(kNorm);

    let maxUpdate = 0.0;

    for (let o = 0; o < outD; o++) {
      let e = errors[o] * sampleWeight;
      if (!Number.isFinite(e)) e = 0;

      const updateMag = Math.abs(e) * kNorm;
      if (updateMag > maxUpdate) maxUpdate = updateMag;

      // Clip individual updates
      let scale = 1.0;
      if (clip > 0 && updateMag > clip) {
        scale = clip / (updateMag + this.epsilon);
      }
      e *= scale;

      const base = o * d;
      for (let j = 0; j < d; j++) {
        const upd = k[j] * e;
        if (Number.isFinite(upd)) Wout[base + j] += upd;
      }
    }

    this.lastUpdateNorm = maxUpdate;
  }

  applyL2Decay(Wout: Float64Array, decayRate: number): void {
    if (decayRate <= 0 || decayRate >= 1) return;
    const decay = 1.0 - decayRate;
    const n = Wout.length;
    for (let i = 0; i < n; i++) Wout[i] *= decay;
  }
}

/* ============================================================
   4. Reservoir: ESN Core with State Clipping
   ============================================================ */

export class ReservoirInitMask {
  readonly size: number;
  constructor(size: number) {
    this.size = size;
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
    // Power iteration with more iterations for better estimate
    for (let i = 0; i < n; i++) scratchV[i] = rng.nextSignedFloat();

    let vNorm = TensorOps.norm2(scratchV);
    if (vNorm < epsilon) vNorm = 1.0;
    for (let i = 0; i < n; i++) scratchV[i] /= vNorm;

    let est = 0.0;
    for (let t = 0; t < iters; t++) {
      // wv = W * v
      for (let i = 0; i < n; i++) {
        let s = 0.0;
        const row = i * n;
        for (let j = 0; j < n; j++) s += W[row + j] * scratchV[j];
        scratchWv[i] = s;
      }

      const wvNorm = TensorOps.norm2(scratchWv);
      if (!Number.isFinite(wvNorm) || wvNorm < epsilon) return est;

      est = wvNorm;
      const inv = 1.0 / wvNorm;
      for (let i = 0; i < n; i++) scratchV[i] = scratchWv[i] * inv;
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
    const iters = Math.max(50, n);
    const est = this.estimateSpectralRadius(
      W,
      n,
      iters,
      rng,
      epsilon,
      scratchV,
      scratchWv,
    );
    if (!Number.isFinite(est) || est < epsilon) {
      // Fallback: create identity-like with target radius
      TensorOps.fill(W, 0);
      for (let i = 0; i < n; i++) W[i * n + i] = targetRadius * 0.5;
      return targetRadius * 0.5;
    }
    const scale = targetRadius / est;
    TensorOps.scale(W, scale);
    return targetRadius;
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
  stateClip: number;

  constructor(cfg: ESNRegressionConfig, nFeatures: number) {
    this.reservoirSize = cfg.reservoirSize | 0;
    this.nFeatures = nFeatures | 0;
    this.leakRate = Math.max(0.01, Math.min(1.0, cfg.leakRate));
    this.spectralRadius = Math.max(0.1, Math.min(1.5, cfg.spectralRadius));
    this.inputScale = cfg.inputScale;
    this.biasScale = cfg.biasScale;
    this.reservoirSparsity = Math.max(0, Math.min(0.99, cfg.reservoirSparsity));
    this.inputSparsity = Math.max(0, Math.min(0.99, cfg.inputSparsity));
    this.activation = cfg.activation;
    this.weightInitScale = cfg.weightInitScale;
    this.seed = cfg.seed | 0;
    this.epsilon = cfg.epsilon;
    this.stateClip = 1.0;
  }
}

export class ESNReservoir {
  readonly params: ESNReservoirParams;
  readonly Win: Float64Array;
  readonly W: Float64Array;
  readonly bias: Float64Array;
  readonly r: Float64Array;
  private preAct: Float64Array;
  private actualSpectralRadius: number = 0;

  constructor(params: ESNReservoirParams) {
    this.params = params;
    const N = params.reservoirSize;
    const F = params.nFeatures;

    this.Win = new Float64Array(N * F);
    this.W = new Float64Array(N * N);
    this.bias = new Float64Array(N);
    this.r = new Float64Array(N);
    this.preAct = new Float64Array(N);

    this.initWeightsDeterministic();
  }

  private initWeightsDeterministic(): void {
    const N = this.params.reservoirSize;
    const F = this.params.nFeatures;
    const rng = new RandomGenerator(this.params.seed);
    const scale = this.params.weightInitScale;

    // Win: input weights with optional sparsity
    const inSparsity = this.params.inputSparsity;
    for (let i = 0; i < N * F; i++) {
      const keep = inSparsity <= 0 || rng.nextFloat() >= inSparsity;
      this.Win[i] = keep ? rng.nextGaussian() * scale : 0;
    }

    // W: reservoir weights with sparsity
    const sparsity = this.params.reservoirSparsity;
    for (let i = 0; i < N * N; i++) {
      const keep = sparsity >= 1 ? false : rng.nextFloat() >= sparsity;
      this.W[i] = keep ? rng.nextGaussian() * scale : 0;
    }

    // Bias
    for (let i = 0; i < N; i++) {
      this.bias[i] = rng.nextGaussian() * this.params.biasScale;
    }

    // Scale to target spectral radius
    const scratchV = new Float64Array(N);
    const scratchWv = new Float64Array(N);
    this.actualSpectralRadius = SpectralRadiusScaler.scaleToRadius(
      this.W,
      N,
      this.params.spectralRadius,
      rng,
      this.params.epsilon,
      scratchV,
      scratchWv,
    );
  }

  resetState(): void {
    TensorOps.fill(this.r, 0);
  }

  getActualSpectralRadius(): number {
    return this.actualSpectralRadius;
  }

  /**
   * Leaky integrator ESN update with state clipping:
   * r_t = (1-a)*r_{t-1} + a*act(Win*(s*x) + W*r_{t-1} + bias)
   * Then clip r_t to [-stateClip, stateClip]
   */
  update(xNorm: Float64Array): void {
    const N = this.params.reservoirSize;
    const F = this.params.nFeatures;
    const a = this.params.leakRate;
    const oneMinusA = 1.0 - a;
    const inputScale = this.params.inputScale;
    const clip = this.params.stateClip;

    const pre = this.preAct;

    // pre = bias + Win*(inputScale*x) + W*r
    for (let i = 0; i < N; i++) {
      let s = this.bias[i];

      // Win contribution
      const winRow = i * F;
      for (let j = 0; j < F; j++) {
        s += this.Win[winRow + j] * xNorm[j] * inputScale;
      }

      // W contribution
      const wRow = i * N;
      for (let j = 0; j < N; j++) {
        s += this.W[wRow + j] * this.r[j];
      }

      pre[i] = s;
    }

    // Activation
    ActivationOps.applyInPlace(pre, this.params.activation);

    // Leaky update with clipping
    for (let i = 0; i < N; i++) {
      let newR = oneMinusA * this.r[i] + a * pre[i];
      // Clip state to prevent explosion
      if (newR > clip) newR = clip;
      else if (newR < -clip) newR = -clip;
      if (!Number.isFinite(newR)) newR = 0;
      this.r[i] = newR;
    }
  }

  /** Update using external state buffer (for prediction rollforward) */
  updateWithState(
    xNorm: Float64Array,
    rIn: Float64Array,
    rOut: Float64Array,
    preActBuf: Float64Array,
  ): void {
    const N = this.params.reservoirSize;
    const F = this.params.nFeatures;
    const a = this.params.leakRate;
    const oneMinusA = 1.0 - a;
    const inputScale = this.params.inputScale;
    const clip = this.params.stateClip;

    for (let i = 0; i < N; i++) {
      let s = this.bias[i];
      const winRow = i * F;
      for (let j = 0; j < F; j++) {
        s += this.Win[winRow + j] * xNorm[j] * inputScale;
      }
      const wRow = i * N;
      for (let j = 0; j < N; j++) s += this.W[wRow + j] * rIn[j];
      preActBuf[i] = s;
    }

    ActivationOps.applyInPlace(preActBuf, this.params.activation);

    for (let i = 0; i < N; i++) {
      let newR = oneMinusA * rIn[i] + a * preActBuf[i];
      if (newR > clip) newR = clip;
      else if (newR < -clip) newR = -clip;
      if (!Number.isFinite(newR)) newR = 0;
      rOut[i] = newR;
    }
  }
}

/* ============================================================
   5. Readout Layer
   ============================================================ */

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
  Wout: Float64Array;

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
    TensorOps.matVec(
      this.params.Wout,
      this.params.outDim,
      this.params.zDim,
      z,
      yOut,
    );
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
  mask: Uint8Array;
  constructor(size: number) {
    this.mask = new Uint8Array(size);
  }
}

/* ============================================================
   6. Training Utilities
   ============================================================ */

export class ForwardContext {}
export class BackwardContext {}
export class GradientTape {}

export class RingBuffer {
  readonly capacity: number;
  readonly nFeatures: number;
  private data: Float64Array;
  private head: number = 0;
  private size: number = 0;
  private totalPushed: number = 0;

  constructor(capacity: number, nFeatures: number) {
    this.capacity = capacity | 0;
    this.nFeatures = nFeatures | 0;
    this.data = new Float64Array(this.capacity * this.nFeatures);
  }

  reset(): void {
    TensorOps.fill(this.data, 0);
    this.head = 0;
    this.size = 0;
    this.totalPushed = 0;
  }

  pushRow(x: number[] | Float64Array): void {
    const F = this.nFeatures;
    const base = this.head * F;
    for (let j = 0; j < F; j++) {
      const v = (x as any)[j];
      this.data[base + j] = Number.isFinite(v) ? v : 0;
    }
    this.head = (this.head + 1) % this.capacity;
    if (this.size < this.capacity) this.size++;
    this.totalPushed++;
  }

  length(): number {
    return this.size;
  }
  totalCount(): number {
    return this.totalPushed;
  }

  copyLatestRow(out: Float64Array): void {
    if (this.size <= 0) throw new Error("RingBuffer is empty");
    const F = this.nFeatures;
    const idx = (this.head - 1 + this.capacity) % this.capacity;
    const base = idx * F;
    for (let j = 0; j < F; j++) out[j] = this.data[base + j];
  }

  copyRowFromLatest(offsetFromLatest: number, out: Float64Array): void {
    const off = offsetFromLatest | 0;
    if (off < 0 || off >= this.size) {
      throw new Error("RingBuffer offset out of range");
    }
    const F = this.nFeatures;
    const idx = (this.head - 1 - off + this.capacity * 2) % this.capacity;
    const base = idx * F;
    for (let j = 0; j < F; j++) out[j] = this.data[base + j];
  }

  copyLastKRows(k: number, out: Float64Array): void {
    const K = Math.min(k | 0, this.size);
    const F = this.nFeatures;
    let outIdx = 0;
    for (let i = K - 1; i >= 0; i--) {
      const idx = (this.head - 1 - i + this.capacity * 2) % this.capacity;
      const base = idx * F;
      for (let j = 0; j < F; j++) out[outIdx++] = this.data[base + j];
    }
  }

  getRawData(): Float64Array {
    return this.data;
  }
  getHead(): number {
    return this.head;
  }
  getSize(): number {
    return this.size;
  }

  setInternal(
    data: Float64Array,
    head: number,
    size: number,
    totalPushed: number,
  ): void {
    TensorOps.copy(this.data, data);
    this.head = head | 0;
    this.size = size | 0;
    this.totalPushed = totalPushed | 0;
  }
}

export class ResidualStatsTracker {
  readonly windowSize: number;
  readonly n: number;

  private buf: Float64Array;
  private head: number = 0;
  private count: number = 0;
  private sum: Float64Array;
  private sumsq: Float64Array;

  constructor(windowSize: number, n: number) {
    this.windowSize = Math.max(10, windowSize | 0);
    this.n = n | 0;
    this.buf = new Float64Array(this.windowSize * this.n);
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
    const base = this.head * n;

    if (this.count === ws) {
      // Remove oldest
      for (let i = 0; i < n; i++) {
        const old = this.buf[base + i];
        this.sum[i] -= old;
        this.sumsq[i] -= old * old;
      }
    } else {
      this.count++;
    }

    for (let i = 0; i < n; i++) {
      const r = Number.isFinite(residuals[i]) ? residuals[i] : 0;
      this.buf[base + i] = r;
      this.sum[i] += r;
      this.sumsq[i] += r * r;
    }

    this.head = (this.head + 1) % ws;
  }

  getCount(): number {
    return this.count;
  }

  mean(i: number): number {
    if (this.count <= 0) return 0;
    return this.sum[i] / this.count;
  }

  variance(i: number, epsilon: number): number {
    if (this.count <= 1) return epsilon;
    const m = this.sum[i] / this.count;
    let v = this.sumsq[i] / this.count - m * m;
    // Bessel correction approximation for small samples
    v *= this.count / (this.count - 1);
    return Number.isFinite(v) && v > epsilon ? v : epsilon;
  }

  std(i: number, epsilon: number): number {
    return Math.sqrt(this.variance(i, epsilon));
  }

  meanMSE(epsilon: number): number {
    if (this.count <= 0) return epsilon;
    let s = 0.0;
    for (let i = 0; i < this.n; i++) {
      s += this.sumsq[i] / this.count;
    }
    const v = s / this.n;
    return Number.isFinite(v) && v > epsilon ? v : epsilon;
  }

  // For serialization
  getBuf(): Float64Array {
    return this.buf;
  }
  getHeadVal(): number {
    return this.head;
  }
  getSum(): Float64Array {
    return this.sum;
  }
  getSumsq(): Float64Array {
    return this.sumsq;
  }

  setInternal(
    buf: number[],
    head: number,
    count: number,
    sum: number[],
    sumsq: number[],
  ): void {
    for (let i = 0; i < this.buf.length && i < buf.length; i++) {
      this.buf[i] = buf[i];
    }
    this.head = head;
    this.count = count;
    for (let i = 0; i < this.n && i < sum.length; i++) this.sum[i] = sum[i];
    for (let i = 0; i < this.n && i < sumsq.length; i++) {
      this.sumsq[i] = sumsq[i];
    }
  }
}

export class OutlierDownweighter {
  readonly threshold: number;
  readonly minWeight: number;
  readonly epsilon: number;

  constructor(threshold: number, minWeight: number, epsilon: number) {
    this.threshold = Math.max(1.5, threshold);
    this.minWeight = Math.max(0.01, Math.min(0.5, minWeight));
    this.epsilon = epsilon;
  }

  computeWeight(residuals: Float64Array, stats: ResidualStatsTracker): number {
    const c = stats.getCount();
    if (c < 20) return 1.0; // Need enough samples for reliable stats

    const n = stats.n;
    let maxZ = 0.0;

    for (let i = 0; i < n; i++) {
      const mu = stats.mean(i);
      const sd = stats.std(i, this.epsilon);
      const z = Math.abs((residuals[i] - mu) / (sd + this.epsilon));
      if (z > maxZ) maxZ = z;
    }

    if (maxZ <= this.threshold) return 1.0;

    // Smooth downweight using Huber-like function
    let w = this.threshold / (maxZ + this.epsilon);
    w = w * w; // Quadratic penalty for large outliers
    if (w < this.minWeight) w = this.minWeight;
    return w;
  }
}

export class LossFunction {
  static mse(yTrue: Float64Array, yPred: Float64Array): number {
    const n = yTrue.length;
    if (n === 0) return 0;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const d = yTrue[i] - yPred[i];
      s += d * d;
    }
    return s / n;
  }

  static huber(
    yTrue: Float64Array,
    yPred: Float64Array,
    delta: number,
  ): number {
    const n = yTrue.length;
    if (n === 0) return 0;
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(yTrue[i] - yPred[i]);
      if (d <= delta) {
        s += 0.5 * d * d;
      } else {
        s += delta * (d - 0.5 * delta);
      }
    }
    return s / n;
  }
}

export class MetricsAccumulator {
  private sumLoss: number = 0;
  private count: number = 0;
  private minLoss: number = Infinity;
  private maxLoss: number = -Infinity;

  reset(): void {
    this.sumLoss = 0;
    this.count = 0;
    this.minLoss = Infinity;
    this.maxLoss = -Infinity;
  }

  add(loss: number): void {
    if (!Number.isFinite(loss)) return;
    this.sumLoss += loss;
    this.count++;
    if (loss < this.minLoss) this.minLoss = loss;
    if (loss > this.maxLoss) this.maxLoss = loss;
  }

  mean(): number {
    return this.count > 0 ? this.sumLoss / this.count : 0;
  }

  getCount(): number {
    return this.count;
  }
}

/* ============================================================
   7. ESNModel Assembly
   ============================================================ */

export class ESNModelConfig {
  cfg: ESNRegressionConfig;
  constructor(cfg: ESNRegressionConfig) {
    this.cfg = cfg;
  }
}

export class TrainingState {
  sampleCount: number = 0;
  warmupComplete: boolean = false;

  reset(): void {
    this.sampleCount = 0;
    this.warmupComplete = false;
  }
}

export class InferenceState {}

export class ESNModel {
  readonly cfg: ESNRegressionConfig;
  readonly nFeatures: number;
  readonly nTargets: number;
  readonly outputDim: number;
  readonly zDim: number;

  readonly ring: RingBuffer;
  readonly xNormalizer: WelfordNormalizer;
  readonly yNormalizer: WelfordNormalizer;
  readonly reservoir: ESNReservoir;
  readonly readoutCfg: ReadoutConfig;
  readonly readoutParams: ReadoutParams;
  readonly readout: LinearReadout;

  readonly rlsState: RLSState;
  readonly rlsOpt: RLSOptimizer;

  readonly residualStats: ResidualStatsTracker;
  readonly outlier: OutlierDownweighter;

  readonly trainState: TrainingState;

  // Scratch buffers
  readonly xRaw: Float64Array;
  readonly xNorm: Float64Array;
  readonly z: Float64Array;
  readonly yHatNorm: Float64Array;
  readonly yHatDenorm: Float64Array;
  readonly yTrueNorm: Float64Array;
  readonly yTrueRaw: Float64Array;
  readonly residuals: Float64Array;
  readonly rollState: Float64Array;
  readonly rollXNorm: Float64Array;
  readonly rollZ: Float64Array;
  readonly rollYHat: Float64Array;
  readonly rollPreAct: Float64Array;

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
    this.xNormalizer = new WelfordNormalizer(
      F,
      cfg.normalizationEpsilon,
      cfg.normalizationWarmup | 0,
    );
    this.yNormalizer = new WelfordNormalizer(
      this.outputDim,
      cfg.normalizationEpsilon,
      cfg.normalizationWarmup | 0,
    );

    this.reservoir = new ESNReservoir(new ESNReservoirParams(cfg, F));
    this.readoutParams = new ReadoutParams(this.outputDim, zDim);
    this.readout = new LinearReadout(this.readoutParams);

    this.rlsState = new RLSState(
      zDim,
      cfg.rlsLambda,
      cfg.rlsDelta,
      cfg.l2Lambda,
    );
    this.rlsOpt = new RLSOptimizer(
      this.rlsState,
      this.outputDim,
      cfg.epsilon,
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

    // Allocate scratch buffers
    this.xRaw = new Float64Array(F);
    this.xNorm = new Float64Array(F);
    this.z = new Float64Array(zDim);
    this.yHatNorm = new Float64Array(this.outputDim);
    this.yHatDenorm = new Float64Array(this.outputDim);
    this.yTrueNorm = new Float64Array(this.outputDim);
    this.yTrueRaw = new Float64Array(this.outputDim);
    this.residuals = new Float64Array(this.outputDim);

    this.rollState = new Float64Array(N);
    this.rollXNorm = new Float64Array(F);
    this.rollZ = new Float64Array(zDim);
    this.rollYHat = new Float64Array(this.nTargets);
    this.rollPreAct = new Float64Array(N);

    this.rlsOpt.setZRef(this.z);
  }

  reset(): void {
    this.ring.reset();
    this.xNormalizer.reset();
    this.yNormalizer.reset();
    this.reservoir.resetState();
    TensorOps.fill(this.readoutParams.Wout, 0);
    this.rlsState.reset();
    this.residualStats.reset();
    this.trainState.reset();
  }

  assembleZ(
    rState: Float64Array,
    xNorm: Float64Array,
    zOut: Float64Array,
  ): void {
    const N = this.cfg.reservoirSize | 0;
    let k = 0;

    for (let i = 0; i < N; i++) zOut[k++] = rState[i];

    if (this.cfg.useInputInReadout) {
      const F = this.nFeatures;
      for (let j = 0; j < F; j++) zOut[k++] = xNorm[j];
    }

    if (this.cfg.useBiasInReadout) zOut[k++] = 1.0;
  }
}

/* ============================================================
   8. Serialization Helper
   ============================================================ */

export class SerializationHelper {
  static f64ToArray(x: Float64Array): number[] {
    const out: number[] = new Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i];
    return out;
  }

  static arrayToF64(arr: number[], out: Float64Array): void {
    const n = Math.min(arr.length, out.length);
    for (let i = 0; i < n; i++) out[i] = arr[i];
  }
}

/* ============================================================
   9. ESNRegression Public API
   ============================================================ */

export class ESNRegression {
  readonly cfg: ESNRegressionConfig;

  private model: ESNModel | null = null;
  private initialized: boolean = false;

  private fitResult: FitResult;
  private predResult: PredictionResult;
  private predPredictions: number[][] = [];
  private predLower: number[][] = [];
  private predUpper: number[][] = [];

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

    // Validate config
    if (this.cfg.maxSequenceLength <= 0) {
      throw new Error("maxSequenceLength must be > 0");
    }
    if (this.cfg.maxFutureSteps <= 0) {
      throw new Error("maxFutureSteps must be > 0");
    }
    if (this.cfg.reservoirSize <= 0) {
      throw new Error("reservoirSize must be > 0");
    }
    if (this.cfg.rlsLambda <= 0 || this.cfg.rlsLambda > 1) {
      throw new Error("rlsLambda must be in (0,1]");
    }
    if (this.cfg.leakRate <= 0 || this.cfg.leakRate > 1) {
      throw new Error("leakRate must be in (0,1]");
    }

    this.fitResult = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1,
    };

    this.predResult = {
      predictions: this.predPredictions,
      lowerBounds: this.predLower,
      upperBounds: this.predUpper,
      confidence: 0,
    };

    this.metrics = new MetricsAccumulator();
  }

  private ensureInitialized(
    xCoordinates: number[][],
    yCoordinates: number[][],
  ): void {
    if (this.initialized) return;

    if (xCoordinates.length === 0 || yCoordinates.length === 0) {
      throw new Error("fitOnline: empty batch");
    }

    const nFeatures = xCoordinates[0].length | 0;
    const yLen = yCoordinates[0].length | 0;

    if (nFeatures <= 0) throw new Error("nFeatures must be > 0");
    if (yLen <= 0) throw new Error("nTargets must be > 0");

    let nTargets: number;
    if (this.cfg.useDirectMultiHorizon && this.cfg.maxFutureSteps > 1) {
      if (yLen % this.cfg.maxFutureSteps !== 0) {
        throw new Error(
          "yCoordinates[0].length must be divisible by maxFutureSteps for direct multi-horizon",
        );
      }
      nTargets = (yLen / this.cfg.maxFutureSteps) | 0;
    } else {
      nTargets = yLen;
    }

    this.model = new ESNModel(this.cfg, nFeatures, nTargets);
    this.initialized = true;

    // Allocate prediction arrays
    const steps = this.cfg.maxFutureSteps | 0;
    this.predPredictions = new Array(steps);
    this.predLower = new Array(steps);
    this.predUpper = new Array(steps);

    for (let s = 0; s < steps; s++) {
      this.predPredictions[s] = new Array(nTargets).fill(0);
      this.predLower[s] = new Array(nTargets).fill(0);
      this.predUpper[s] = new Array(nTargets).fill(0);
    }

    this.predResult.predictions = this.predPredictions;
    this.predResult.lowerBounds = this.predLower;
    this.predResult.upperBounds = this.predUpper;
  }

  /**
   * Online training: processes samples one at a time.
   *
   * CRITICAL: xCoordinates.length MUST equal yCoordinates.length (no observe-only).
   * For each sample:
   *   1. Push X into RingBuffer FIRST
   *   2. Normalize, update reservoir, forward, compute loss, update weights
   *
   * @param param0 Training data
   * @returns FitResult (reused object)
   *
   * @example
   * const esn = new ESNRegression({ maxFutureSteps: 3 });
   * esn.fitOnline({ xCoordinates: [[1,2],[2,3]], yCoordinates: [[10,11,12],[20,21,22]] });
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

    const N = xCoordinates.length;
    if (N === 0) {
      this.fitResult.samplesProcessed = 0;
      this.fitResult.averageLoss = 0;
      this.fitResult.gradientNorm = 0;
      this.fitResult.driftDetected = false;
      this.fitResult.sampleWeight = 1;
      return this.fitResult;
    }

    this.ensureInitialized(xCoordinates, yCoordinates);
    const m = this.model!;

    const F = m.nFeatures;
    const expectedYLen = m.outputDim;

    this.metrics.reset();
    let lastUpdateNorm = 0;
    let lastWeight = 1;

    for (let i = 0; i < N; i++) {
      const xRow = xCoordinates[i];
      const yRow = yCoordinates[i];

      if (xRow.length !== F) {
        throw new Error("fitOnline: x row length mismatch");
      }
      if (yRow.length !== expectedYLen) {
        throw new Error("fitOnline: y row length mismatch");
      }

      // 1) Push X into ring buffer FIRST (critical requirement)
      m.ring.pushRow(xRow);

      // Copy latest X from ring (authoritative)
      m.ring.copyLatestRow(m.xRaw);

      // 2) Observe and normalize X
      m.xNormalizer.observe(m.xRaw);
      m.xNormalizer.normalize(m.xRaw, m.xNorm);

      // 3) Copy Y and observe for normalization
      for (let k = 0; k < expectedYLen; k++) {
        m.yTrueRaw[k] = Number.isFinite(yRow[k]) ? yRow[k] : 0;
      }
      m.yNormalizer.observe(m.yTrueRaw);

      // 4) Update reservoir state
      m.reservoir.update(m.xNorm);

      // Skip training until normalizers are warmed up
      if (!m.xNormalizer.isActive() || !m.yNormalizer.isActive()) {
        m.trainState.sampleCount++;
        continue;
      }

      if (!m.trainState.warmupComplete) {
        m.trainState.warmupComplete = true;
      }

      // 5) Normalize Y for training
      m.yNormalizer.normalize(m.yTrueRaw, m.yTrueNorm);

      // 6) Assemble z = [r; x; 1]
      m.assembleZ(m.reservoir.r, m.xNorm, m.z);

      // 7) Forward pass (produces normalized predictions)
      m.readout.forward(m.z, m.yHatNorm);

      // 8) Compute residuals in normalized space
      for (let k = 0; k < expectedYLen; k++) {
        m.residuals[k] = m.yTrueNorm[k] - m.yHatNorm[k];
      }

      // 9) Compute loss (in normalized space for stable comparison)
      const loss = LossFunction.mse(m.yTrueNorm, m.yHatNorm);

      // 10) Outlier weight
      const w = m.outlier.computeWeight(m.residuals, m.residualStats);
      lastWeight = w;

      // 11) RLS update
      m.rlsOpt.computeGainAndUpdateP();
      m.rlsOpt.applyWeightUpdate(m.readoutParams.Wout, m.residuals, w);
      lastUpdateNorm = m.rlsOpt.getLastUpdateNorm();

      // 12) Update residual stats
      m.residualStats.addResiduals(m.residuals);

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
   * Uses ONLY internal RingBuffer for latest input; never requires caller to pass latest X.
   *
   * @param futureSteps Number of steps to predict (1..maxFutureSteps)
   * @returns PredictionResult (reused object)
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.initialized || !this.model) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }

    const fs = Math.max(1, Math.min(futureSteps | 0, this.cfg.maxFutureSteps));
    const m = this.model;

    if (m.ring.length() <= 0) {
      throw new Error("predict: no input history available");
    }

    // Get latest X from ring buffer (authoritative)
    m.ring.copyLatestRow(m.xRaw);
    m.xNormalizer.normalize(m.xRaw, m.xNorm);

    const nTargets = m.nTargets;
    const useDirect = this.cfg.useDirectMultiHorizon;

    // Compute confidence based on normalized residual stats
    const mse = m.residualStats.meanMSE(this.cfg.epsilon);
    const rmse = Math.sqrt(mse);
    let conf = 1.0 / (1.0 + rmse);
    if (!Number.isFinite(conf)) conf = 0;
    conf = Math.max(0, Math.min(1, conf));
    this.predResult.confidence = conf;

    const mult = this.cfg.uncertaintyMultiplier;

    if (useDirect) {
      // Direct multi-horizon prediction
      m.assembleZ(m.reservoir.r, m.xNorm, m.z);
      m.readout.forward(m.z, m.yHatNorm);

      // Denormalize and populate results
      for (let s = 0; s < fs; s++) {
        const base = s * nTargets;
        const rowP = this.predPredictions[s];
        const rowL = this.predLower[s];
        const rowU = this.predUpper[s];

        // Horizon-dependent confidence decay
        const horizonDecay = 1.0 + s * 0.1;

        for (let t = 0; t < nTargets; t++) {
          const outIdx = base + t;

          // Denormalize prediction
          const yNorm = m.yHatNorm[outIdx];
          const y = m.yNormalizer.denormalizeValue(yNorm, outIdx);

          // Compute uncertainty in original scale
          const sdNorm = m.residualStats.std(outIdx, this.cfg.epsilon);
          const yScale = m.yNormalizer.getStd(outIdx);
          const sd = sdNorm * yScale * horizonDecay;
          const d = mult * sd;

          rowP[t] = y;
          rowL[t] = y - d;
          rowU[t] = y + d;
        }
      }

      return this.predResult;
    }

    // Recursive roll-forward prediction (single-step readout)
    TensorOps.copy(m.rollState, m.reservoir.r);
    TensorOps.copy(m.rollXNorm, m.xNorm);

    for (let s = 0; s < fs; s++) {
      // Update scratch reservoir state
      m.reservoir.updateWithState(
        m.rollXNorm,
        m.rollState,
        m.rollState,
        m.rollPreAct,
      );

      // Assemble z for scratch state
      m.assembleZ(m.rollState, m.rollXNorm, m.rollZ);

      // Forward (single step)
      TensorOps.matVec(
        m.readoutParams.Wout,
        nTargets,
        m.zDim,
        m.rollZ,
        m.rollYHat,
      );

      const rowP = this.predPredictions[s];
      const rowL = this.predLower[s];
      const rowU = this.predUpper[s];

      const horizonDecay = 1.0 + s * 0.15;

      for (let t = 0; t < nTargets; t++) {
        const yNorm = m.rollYHat[t];
        const y = m.yNormalizer.denormalizeValue(yNorm, t);

        const sdNorm = m.residualStats.std(t, this.cfg.epsilon);
        const yScale = m.yNormalizer.getStd(t);
        const sd = sdNorm * yScale * horizonDecay;
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
        receptiveField: Math.max(1, this.cfg.maxSequenceLength),
        spectralRadius: this.cfg.spectralRadius,
        reservoirSize: this.cfg.reservoirSize,
        nFeatures: 0,
        nTargets: 0,
        maxSequenceLength: this.cfg.maxSequenceLength,
        maxFutureSteps: this.cfg.maxFutureSteps,
        sampleCount: 0,
        useDirectMultiHorizon: this.cfg.useDirectMultiHorizon,
      };
    }

    const m = this.model;
    const totalParams = m.readoutParams.Wout.length;

    // Effective receptive field from leak rate
    const a = this.cfg.leakRate;
    let rf = 1;
    if (a > 0 && a < 1) {
      rf = Math.min(
        this.cfg.maxSequenceLength,
        Math.max(1, Math.round(1.0 / (1.0 - a))),
      );
    }

    return {
      totalParameters: totalParams,
      receptiveField: rf,
      spectralRadius: m.reservoir.getActualSpectralRadius(),
      reservoirSize: this.cfg.reservoirSize,
      nFeatures: m.nFeatures,
      nTargets: m.nTargets,
      maxSequenceLength: this.cfg.maxSequenceLength,
      maxFutureSteps: this.cfg.maxFutureSteps,
      sampleCount: m.trainState.sampleCount,
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
          shape: [this.cfg.reservoirSize, m.nFeatures],
          values: SerializationHelper.f64ToArray(m.reservoir.Win),
        },
        {
          name: "W",
          shape: [this.cfg.reservoirSize, this.cfg.reservoirSize],
          values: SerializationHelper.f64ToArray(m.reservoir.W),
        },
        {
          name: "bias",
          shape: [this.cfg.reservoirSize],
          values: SerializationHelper.f64ToArray(m.reservoir.bias),
        },
        {
          name: "Wout",
          shape: [m.outputDim, m.zDim],
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
      means: SerializationHelper.f64ToArray(m.xNormalizer.getMeansArray()),
      stds: SerializationHelper.f64ToArray(m.xNormalizer.getStdsArray()),
      count: m.xNormalizer.getCount(),
      isActive: m.xNormalizer.isActive(),
    };
  }

  reset(): void {
    if (this.model) this.model.reset();
  }

  save(): string {
    const payload: any = {
      version: 2,
      cfg: this.cfg,
      initialized: this.initialized,
      state: null,
    };

    if (this.initialized && this.model) {
      const m = this.model;
      payload.state = {
        nFeatures: m.nFeatures,
        nTargets: m.nTargets,
        outputDim: m.outputDim,
        zDim: m.zDim,

        ring: {
          data: SerializationHelper.f64ToArray(m.ring.getRawData()),
          head: m.ring.getHead(),
          size: m.ring.getSize(),
          totalPushed: m.ring.totalCount(),
        },

        xNormalizer: {
          count: m.xNormalizer.getCount(),
          active: m.xNormalizer.isActive(),
          means: SerializationHelper.f64ToArray(m.xNormalizer.getMeansArray()),
          stds: SerializationHelper.f64ToArray(m.xNormalizer.getStdsArray()),
          m2: SerializationHelper.f64ToArray(m.xNormalizer.getM2Array()),
        },

        yNormalizer: {
          count: m.yNormalizer.getCount(),
          active: m.yNormalizer.isActive(),
          means: SerializationHelper.f64ToArray(m.yNormalizer.getMeansArray()),
          stds: SerializationHelper.f64ToArray(m.yNormalizer.getStdsArray()),
          m2: SerializationHelper.f64ToArray(m.yNormalizer.getM2Array()),
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
          buf: SerializationHelper.f64ToArray(m.residualStats.getBuf()),
          head: m.residualStats.getHeadVal(),
          count: m.residualStats.getCount(),
          sum: SerializationHelper.f64ToArray(m.residualStats.getSum()),
          sumsq: SerializationHelper.f64ToArray(m.residualStats.getSumsq()),
        },

        trainState: {
          sampleCount: m.trainState.sampleCount,
          warmupComplete: m.trainState.warmupComplete,
        },
      };
    }

    return JSON.stringify(payload);
  }

  load(w: string): void {
    const obj = JSON.parse(w);
    if (!obj || (obj.version !== 1 && obj.version !== 2)) {
      throw new Error("load: invalid version");
    }

    const cfg = obj.cfg as ESNRegressionConfig;

    // Validate config compatibility
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
    ];

    for (const k of mustMatch) {
      if ((this.cfg as any)[k] !== (cfg as any)[k]) {
        throw new Error(`load: config mismatch on ${k}`);
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

    // Ring buffer
    SerializationHelper.arrayToF64(st.ring.data, m.ring.getRawData());
    m.ring.setInternal(
      m.ring.getRawData(),
      st.ring.head,
      st.ring.size,
      st.ring.totalPushed,
    );

    // X normalizer
    m.xNormalizer.setInternal(
      st.xNormalizer.count,
      st.xNormalizer.active,
      st.xNormalizer.means,
      st.xNormalizer.stds,
      st.xNormalizer.m2,
    );

    // Y normalizer (handle v1 compatibility)
    if (st.yNormalizer) {
      m.yNormalizer.setInternal(
        st.yNormalizer.count,
        st.yNormalizer.active,
        st.yNormalizer.means,
        st.yNormalizer.stds,
        st.yNormalizer.m2,
      );
    }

    // Reservoir
    SerializationHelper.arrayToF64(st.reservoir.r, m.reservoir.r);
    SerializationHelper.arrayToF64(st.reservoir.Win, m.reservoir.Win);
    SerializationHelper.arrayToF64(st.reservoir.W, m.reservoir.W);
    SerializationHelper.arrayToF64(st.reservoir.bias, m.reservoir.bias);

    // Readout
    SerializationHelper.arrayToF64(st.readout.Wout, m.readoutParams.Wout);

    // RLS
    SerializationHelper.arrayToF64(st.rls.P, m.rlsState.P);

    // Residual stats
    m.residualStats.setInternal(
      st.residualStats.buf,
      st.residualStats.head,
      st.residualStats.count,
      st.residualStats.sum,
      st.residualStats.sumsq,
    );

    // Train state
    m.trainState.sampleCount = st.trainState.sampleCount | 0;
    m.trainState.warmupComplete = st.trainState.warmupComplete ?? true;

    // Reallocate prediction arrays
    const steps = this.cfg.maxFutureSteps;
    this.predPredictions = new Array(steps);
    this.predLower = new Array(steps);
    this.predUpper = new Array(steps);
    for (let s = 0; s < steps; s++) {
      this.predPredictions[s] = new Array(nTargets).fill(0);
      this.predLower[s] = new Array(nTargets).fill(0);
      this.predUpper[s] = new Array(nTargets).fill(0);
    }
    this.predResult.predictions = this.predPredictions;
    this.predResult.lowerBounds = this.predLower;
    this.predResult.upperBounds = this.predUpper;
    this.predResult.confidence = 0;
  }
}
