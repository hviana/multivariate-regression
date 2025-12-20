export interface ESNRegressionConfig {
  maxSequenceLength?: number;
  reservoirSize?: number;
  spectralRadius?: number;
  leakRate?: number;
  inputScale?: number;
  biasScale?: number;
  reservoirSparsity?: number;
  inputSparsity?: number;
  activation?: "tanh" | "relu";
  useInputInReadout?: boolean;
  useBiasInReadout?: boolean;
  readoutTraining?: "rls";
  rlsLambda?: number;
  rlsDelta?: number;
  epsilon?: number;
  l2Lambda?: number;
  gradientClipNorm?: number;
  normalizationEpsilon?: number;
  normalizationWarmup?: number;
  outlierThreshold?: number;
  outlierMinWeight?: number;
  residualWindowSize?: number;
  uncertaintyMultiplier?: number;
  weightInitScale?: number;
  seed?: number;
  verbose?: boolean;
  rollforwardMode?: "holdLastX" | "autoregressive";
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

class TensorShape {
  readonly dims: readonly number[];
  readonly size: number;
  readonly strides: readonly number[];

  constructor(dims: number[]) {
    const d = dims.slice();
    let size = 1;
    for (let i = 0; i < d.length; i++) size *= d[i];
    const strides: number[] = new Array(d.length);
    let stride = 1;
    for (let i = d.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= d[i];
    }
    this.dims = d;
    this.size = size;
    this.strides = strides;
  }

  static vector(n: number): TensorShape {
    return new TensorShape([n]);
  }

  static matrix(rows: number, cols: number): TensorShape {
    return new TensorShape([rows, cols]);
  }
}

class TensorView {
  readonly data: Float64Array;
  readonly offset: number;
  readonly shape: TensorShape;

  constructor(data: Float64Array, offset: number, shape: TensorShape) {
    this.data = data;
    this.offset = offset;
    this.shape = shape;
  }

  get(i: number): number {
    return this.data[this.offset + i];
  }

  set(i: number, v: number): void {
    this.data[this.offset + i] = v;
  }

  get2d(i: number, j: number): number {
    return this.data[this.offset + i * (this.shape.strides[0]) + j];
  }

  set2d(i: number, j: number, v: number): void {
    this.data[this.offset + i * (this.shape.strides[0]) + j] = v;
  }

  fill(v: number): void {
    const end = this.offset + this.shape.size;
    for (let i = this.offset; i < end; i++) {
      this.data[i] = v;
    }
  }

  copyFrom(src: Float64Array, srcOffset: number = 0): void {
    const len = this.shape.size;
    for (let i = 0; i < len; i++) {
      this.data[this.offset + i] = src[srcOffset + i];
    }
  }

  copyTo(dest: Float64Array, destOffset: number = 0): void {
    const len = this.shape.size;
    for (let i = 0; i < len; i++) {
      dest[destOffset + i] = this.data[this.offset + i];
    }
  }
}

class BufferPool {
  private pools: Map<number, Float64Array[]>;
  private maxPoolSize: number;

  constructor(maxPoolSize: number = 32) {
    this.pools = new Map();
    this.maxPoolSize = maxPoolSize;
  }

  acquire(size: number): Float64Array {
    const pool = this.pools.get(size);
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    return new Float64Array(size);
  }

  release(buffer: Float64Array): void {
    const size = buffer.length;
    let pool = this.pools.get(size);
    if (!pool) {
      pool = [];
      this.pools.set(size, pool);
    }
    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
    }
  }

  clear(): void {
    this.pools.clear();
  }
}

class TensorArena {
  private buffer: Float64Array;
  private offset: number;
  private capacity: number;

  constructor(capacity: number) {
    this.buffer = new Float64Array(capacity);
    this.offset = 0;
    this.capacity = capacity;
  }

  alloc(size: number): TensorView {
    if (this.offset + size > this.capacity) {
      throw new Error("TensorArena: out of memory");
    }
    const view = new TensorView(
      this.buffer,
      this.offset,
      TensorShape.vector(size),
    );
    this.offset += size;
    return view;
  }

  allocMatrix(rows: number, cols: number): TensorView {
    const size = rows * cols;
    if (this.offset + size > this.capacity) {
      throw new Error("TensorArena: out of memory");
    }
    const view = new TensorView(
      this.buffer,
      this.offset,
      TensorShape.matrix(rows, cols),
    );
    this.offset += size;
    return view;
  }

  reset(): void {
    this.offset = 0;
  }

  getUsed(): number {
    return this.offset;
  }
}

class TensorOps {
  static dot(
    a: Float64Array,
    aOff: number,
    b: Float64Array,
    bOff: number,
    len: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += a[aOff + i] * b[bOff + i];
    }
    return sum;
  }

  static matvec(
    mat: Float64Array,
    matOff: number,
    rows: number,
    cols: number,
    vec: Float64Array,
    vecOff: number,
    out: Float64Array,
    outOff: number,
  ): void {
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      const rowOff = matOff + i * cols;
      for (let j = 0; j < cols; j++) {
        sum += mat[rowOff + j] * vec[vecOff + j];
      }
      out[outOff + i] = sum;
    }
  }

  static axpy(
    a: number,
    x: Float64Array,
    xOff: number,
    y: Float64Array,
    yOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      y[yOff + i] += a * x[xOff + i];
    }
  }

  static scale(x: Float64Array, xOff: number, s: number, len: number): void {
    for (let i = 0; i < len; i++) {
      x[xOff + i] *= s;
    }
  }

  static copy(
    src: Float64Array,
    srcOff: number,
    dest: Float64Array,
    destOff: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dest[destOff + i] = src[srcOff + i];
    }
  }

  static fill(arr: Float64Array, off: number, len: number, val: number): void {
    for (let i = 0; i < len; i++) {
      arr[off + i] = val;
    }
  }

  static norm2(arr: Float64Array, off: number, len: number): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const v = arr[off + i];
      sum += v * v;
    }
    return Math.sqrt(sum);
  }
}

class ActivationOps {
  static tanh(x: number): number {
    return Math.tanh(x);
  }

  static relu(x: number): number {
    return x > 0 ? x : 0;
  }

  static apply(
    activation: "tanh" | "relu",
    arr: Float64Array,
    off: number,
    len: number,
  ): void {
    if (activation === "tanh") {
      for (let i = 0; i < len; i++) {
        arr[off + i] = Math.tanh(arr[off + i]);
      }
    } else {
      for (let i = 0; i < len; i++) {
        const v = arr[off + i];
        arr[off + i] = v > 0 ? v : 0;
      }
    }
  }
}

class RandomGenerator {
  private state: number;

  constructor(seed: number) {
    this.state = (seed >>> 0) || 1;
  }

  private xorshift32(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }

  random(): number {
    return this.xorshift32() / 4294967296;
  }

  gaussian(mean: number = 0, std: number = 1): number {
    const u1 = Math.max(1e-10, this.random());
    const u2 = this.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z * std + mean;
  }

  getState(): number {
    return this.state;
  }

  setState(s: number): void {
    this.state = (s >>> 0) || 1;
  }
}

class WelfordAccumulator {
  count: number;
  mean: number;
  m2: number;

  constructor() {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }

  update(x: number): void {
    this.count++;
    const delta = x - this.mean;
    this.mean += delta / this.count;
    const delta2 = x - this.mean;
    this.m2 += delta * delta2;
  }

  getVariance(): number {
    return this.count > 1 ? this.m2 / (this.count - 1) : 0;
  }

  getStd(epsilon: number): number {
    return Math.sqrt(Math.max(this.getVariance(), epsilon * epsilon));
  }

  reset(): void {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }
}

class WelfordNormalizer {
  private size: number;
  private count: number;
  private means: Float64Array;
  private m2s: Float64Array;

  constructor(size: number) {
    this.size = size;
    this.count = 0;
    this.means = new Float64Array(size);
    this.m2s = new Float64Array(size);
  }

  update(x: Float64Array, xOff: number = 0): void {
    this.count++;
    const n = this.count;
    for (let i = 0; i < this.size; i++) {
      const val = x[xOff + i];
      const delta = val - this.means[i];
      this.means[i] += delta / n;
      const delta2 = val - this.means[i];
      this.m2s[i] += delta * delta2;
    }
  }

  normalize(
    x: Float64Array,
    xOff: number,
    dest: Float64Array,
    destOff: number,
    epsilon: number,
  ): void {
    for (let i = 0; i < this.size; i++) {
      const variance = this.count > 1 ? this.m2s[i] / (this.count - 1) : 0;
      const std = Math.sqrt(Math.max(variance, 0));
      const denom = Math.max(std, epsilon);
      dest[destOff + i] = (x[xOff + i] - this.means[i]) / denom;
    }
  }

  getMeans(): Float64Array {
    return this.means;
  }

  getM2s(): Float64Array {
    return this.m2s;
  }

  getStds(epsilon: number): Float64Array {
    const stds = new Float64Array(this.size);
    for (let i = 0; i < this.size; i++) {
      const variance = this.count > 1 ? this.m2s[i] / (this.count - 1) : 0;
      stds[i] = Math.sqrt(Math.max(variance, epsilon * epsilon));
    }
    return stds;
  }

  computeStds(dest: Float64Array, destOff: number, epsilon: number): void {
    for (let i = 0; i < this.size; i++) {
      const variance = this.count > 1 ? this.m2s[i] / (this.count - 1) : 0;
      dest[destOff + i] = Math.sqrt(Math.max(variance, epsilon * epsilon));
    }
  }

  getCount(): number {
    return this.count;
  }

  getSize(): number {
    return this.size;
  }

  reset(): void {
    this.count = 0;
    this.means.fill(0);
    this.m2s.fill(0);
  }

  loadState(count: number, means: number[], m2s: number[]): void {
    this.count = count;
    for (let i = 0; i < this.size; i++) {
      this.means[i] = means[i] ?? 0;
      this.m2s[i] = m2s[i] ?? 0;
    }
  }
}

class RingBuffer {
  private data: Float64Array;
  private capacity: number;
  private width: number;
  private head: number;
  private count: number;

  constructor(capacity: number, width: number) {
    this.capacity = capacity;
    this.width = width;
    this.data = new Float64Array(capacity * width);
    this.head = 0;
    this.count = 0;
  }

  push(row: number[]): void {
    const offset = this.head * this.width;
    for (let i = 0; i < this.width; i++) {
      this.data[offset + i] = row[i];
    }
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  pushFloat64(row: Float64Array, rowOff: number = 0): void {
    const offset = this.head * this.width;
    for (let i = 0; i < this.width; i++) {
      this.data[offset + i] = row[rowOff + i];
    }
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  getLast(dest: Float64Array, destOff: number = 0): void {
    if (this.count === 0) {
      throw new Error("RingBuffer: empty");
    }
    const idx = (this.head - 1 + this.capacity) % this.capacity;
    const offset = idx * this.width;
    for (let i = 0; i < this.width; i++) {
      dest[destOff + i] = this.data[offset + i];
    }
  }

  getAt(index: number, dest: Float64Array, destOff: number = 0): void {
    if (index < 0 || index >= this.count) {
      throw new Error("RingBuffer: index out of bounds");
    }
    const actualIdx = (this.head - this.count + index + this.capacity) %
      this.capacity;
    const offset = actualIdx * this.width;
    for (let i = 0; i < this.width; i++) {
      dest[destOff + i] = this.data[offset + i];
    }
  }

  size(): number {
    return this.count;
  }

  getCapacity(): number {
    return this.capacity;
  }

  getWidth(): number {
    return this.width;
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }

  getData(): Float64Array {
    return this.data;
  }

  getHead(): number {
    return this.head;
  }

  loadState(head: number, count: number, data: number[]): void {
    this.head = head;
    this.count = count;
    for (let i = 0; i < data.length && i < this.data.length; i++) {
      this.data[i] = data[i];
    }
  }
}

class ResidualStatsTracker {
  private windowSize: number;
  private nTargets: number;
  private residuals: Float64Array;
  private head: number;
  private count: number;

  constructor(windowSize: number, nTargets: number) {
    this.windowSize = windowSize;
    this.nTargets = nTargets;
    this.residuals = new Float64Array(windowSize * nTargets);
    this.head = 0;
    this.count = 0;
  }

  update(residual: Float64Array, resOff: number = 0): void {
    const offset = this.head * this.nTargets;
    for (let i = 0; i < this.nTargets; i++) {
      this.residuals[offset + i] = residual[resOff + i];
    }
    this.head = (this.head + 1) % this.windowSize;
    if (this.count < this.windowSize) this.count++;
  }

  getStds(dest: Float64Array, destOff: number, epsilon: number): void {
    if (this.count === 0) {
      for (let t = 0; t < this.nTargets; t++) {
        dest[destOff + t] = epsilon;
      }
      return;
    }

    for (let t = 0; t < this.nTargets; t++) {
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < this.count; i++) {
        const v = this.residuals[i * this.nTargets + t];
        sum += v;
        sumSq += v * v;
      }
      const mean = sum / this.count;
      const variance = this.count > 1
        ? (sumSq - sum * mean) / (this.count - 1)
        : 0;
      dest[destOff + t] = Math.sqrt(Math.max(variance, epsilon * epsilon));
    }
  }

  getMeanAbsResidual(): number {
    if (this.count === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.count * this.nTargets; i++) {
      sum += Math.abs(this.residuals[i]);
    }
    return sum / (this.count * this.nTargets);
  }

  reset(): void {
    this.head = 0;
    this.count = 0;
  }

  getCount(): number {
    return this.count;
  }

  loadState(head: number, count: number, data: number[]): void {
    this.head = head;
    this.count = count;
    for (let i = 0; i < data.length && i < this.residuals.length; i++) {
      this.residuals[i] = data[i];
    }
  }

  getData(): Float64Array {
    return this.residuals;
  }

  getHead(): number {
    return this.head;
  }
}

class OutlierDownweighter {
  compute(
    residual: Float64Array,
    resOff: number,
    stds: Float64Array,
    stdsOff: number,
    len: number,
    threshold: number,
    minWeight: number,
    epsilon: number,
  ): number {
    let maxZ = 0;
    for (let i = 0; i < len; i++) {
      const std = Math.max(stds[stdsOff + i], epsilon);
      const z = Math.abs(residual[resOff + i]) / std;
      if (z > maxZ) maxZ = z;
    }
    if (maxZ <= threshold) return 1.0;
    const w = Math.exp(-0.5 * (maxZ - threshold));
    return Math.max(w, minWeight);
  }
}

class LossFunction {
  static mse(
    pred: Float64Array,
    predOff: number,
    target: number[],
    error: Float64Array,
    errorOff: number,
    len: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const e = target[i] - pred[predOff + i];
      error[errorOff + i] = e;
      sum += e * e;
    }
    return sum / len;
  }
}

class MetricsAccumulator {
  private count: number;
  private lossSum: number;
  private gradNormSum: number;

  constructor() {
    this.count = 0;
    this.lossSum = 0;
    this.gradNormSum = 0;
  }

  add(loss: number, gradNorm: number): void {
    this.count++;
    this.lossSum += loss;
    this.gradNormSum += gradNorm;
  }

  getAverageLoss(): number {
    return this.count > 0 ? this.lossSum / this.count : 0;
  }

  getAverageGradNorm(): number {
    return this.count > 0 ? this.gradNormSum / this.count : 0;
  }

  getCount(): number {
    return this.count;
  }

  reset(): void {
    this.count = 0;
    this.lossSum = 0;
    this.gradNormSum = 0;
  }
}

class ReservoirInitMask {
  static createSparseMask(
    rows: number,
    cols: number,
    sparsity: number,
    rng: RandomGenerator,
  ): boolean[] {
    const mask = new Array<boolean>(rows * cols);
    for (let i = 0; i < mask.length; i++) {
      mask[i] = rng.random() >= sparsity;
    }
    return mask;
  }
}

class SpectralRadiusScaler {
  static estimate(
    W: Float64Array,
    size: number,
    rng: RandomGenerator,
    maxIter: number = 100,
    tol: number = 1e-6,
  ): number {
    const v = new Float64Array(size);
    const w = new Float64Array(size);

    let norm = 0;
    for (let i = 0; i < size; i++) {
      v[i] = rng.random() - 0.5;
      norm += v[i] * v[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let i = 0; i < size; i++) v[i] /= norm;
    }

    let eigenvalue = 0;
    for (let iter = 0; iter < maxIter; iter++) {
      for (let i = 0; i < size; i++) {
        let sum = 0;
        for (let j = 0; j < size; j++) {
          sum += W[i * size + j] * v[j];
        }
        w[i] = sum;
      }

      norm = 0;
      for (let i = 0; i < size; i++) {
        norm += w[i] * w[i];
      }
      norm = Math.sqrt(norm);

      if (norm < 1e-10) return 0;

      if (Math.abs(norm - eigenvalue) < tol) {
        return norm;
      }
      eigenvalue = norm;

      for (let i = 0; i < size; i++) {
        v[i] = w[i] / norm;
      }
    }

    return eigenvalue;
  }

  static scale(
    W: Float64Array,
    currentRadius: number,
    targetRadius: number,
  ): void {
    if (currentRadius < 1e-10) return;
    const scaleFactor = targetRadius / currentRadius;
    for (let i = 0; i < W.length; i++) {
      W[i] *= scaleFactor;
    }
  }
}

interface ESNReservoirParams {
  reservoirSize: number;
  nFeatures: number;
  spectralRadius: number;
  leakRate: number;
  inputScale: number;
  biasScale: number;
  reservoirSparsity: number;
  inputSparsity: number;
  activation: "tanh" | "relu";
}

class ESNReservoir {
  private params: ESNReservoirParams;
  private Win: Float64Array;
  private W: Float64Array;
  private bias: Float64Array;
  private state: Float64Array;
  private preActivation: Float64Array;

  constructor(params: ESNReservoirParams, rng: RandomGenerator) {
    this.params = params;
    const {
      reservoirSize,
      nFeatures,
      inputScale,
      biasScale,
      reservoirSparsity,
      inputSparsity,
      spectralRadius,
    } = params;

    this.Win = new Float64Array(reservoirSize * nFeatures);
    this.W = new Float64Array(reservoirSize * reservoirSize);
    this.bias = new Float64Array(reservoirSize);
    this.state = new Float64Array(reservoirSize);
    this.preActivation = new Float64Array(reservoirSize);

    for (let i = 0; i < reservoirSize; i++) {
      for (let j = 0; j < nFeatures; j++) {
        if (rng.random() >= inputSparsity) {
          this.Win[i * nFeatures + j] = rng.gaussian() * inputScale;
        }
      }
    }

    for (let i = 0; i < reservoirSize; i++) {
      for (let j = 0; j < reservoirSize; j++) {
        if (rng.random() >= reservoirSparsity) {
          this.W[i * reservoirSize + j] = rng.gaussian();
        }
      }
    }

    const currentRadius = SpectralRadiusScaler.estimate(
      this.W,
      reservoirSize,
      rng,
    );
    SpectralRadiusScaler.scale(this.W, currentRadius, spectralRadius);

    for (let i = 0; i < reservoirSize; i++) {
      this.bias[i] = (rng.random() * 2 - 1) * biasScale;
    }
  }

  update(xNorm: Float64Array, xOff: number): void {
    const { reservoirSize, nFeatures, leakRate, activation } = this.params;

    for (let i = 0; i < reservoirSize; i++) {
      let sum = this.bias[i];
      const winRowOff = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        sum += this.Win[winRowOff + j] * xNorm[xOff + j];
      }
      const wRowOff = i * reservoirSize;
      for (let j = 0; j < reservoirSize; j++) {
        sum += this.W[wRowOff + j] * this.state[j];
      }
      this.preActivation[i] = sum;
    }

    const oneMinusLeak = 1 - leakRate;
    for (let i = 0; i < reservoirSize; i++) {
      let activated: number;
      if (activation === "tanh") {
        activated = Math.tanh(this.preActivation[i]);
      } else {
        activated = this.preActivation[i] > 0 ? this.preActivation[i] : 0;
      }
      this.state[i] = oneMinusLeak * this.state[i] + leakRate * activated;
    }
  }

  updateWithState(
    xNorm: Float64Array,
    xOff: number,
    stateIn: Float64Array,
    stateInOff: number,
    stateOut: Float64Array,
    stateOutOff: number,
    preActBuf: Float64Array,
    preActOff: number,
  ): void {
    const { reservoirSize, nFeatures, leakRate, activation } = this.params;

    for (let i = 0; i < reservoirSize; i++) {
      let sum = this.bias[i];
      const winRowOff = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        sum += this.Win[winRowOff + j] * xNorm[xOff + j];
      }
      const wRowOff = i * reservoirSize;
      for (let j = 0; j < reservoirSize; j++) {
        sum += this.W[wRowOff + j] * stateIn[stateInOff + j];
      }
      preActBuf[preActOff + i] = sum;
    }

    const oneMinusLeak = 1 - leakRate;
    for (let i = 0; i < reservoirSize; i++) {
      let activated: number;
      if (activation === "tanh") {
        activated = Math.tanh(preActBuf[preActOff + i]);
      } else {
        const v = preActBuf[preActOff + i];
        activated = v > 0 ? v : 0;
      }
      stateOut[stateOutOff + i] = oneMinusLeak * stateIn[stateInOff + i] +
        leakRate * activated;
    }
  }

  copyStateTo(dest: Float64Array, destOff: number): void {
    for (let i = 0; i < this.params.reservoirSize; i++) {
      dest[destOff + i] = this.state[i];
    }
  }

  copyStateFrom(src: Float64Array, srcOff: number): void {
    for (let i = 0; i < this.params.reservoirSize; i++) {
      this.state[i] = src[srcOff + i];
    }
  }

  getState(): Float64Array {
    return this.state;
  }

  getSize(): number {
    return this.params.reservoirSize;
  }

  resetState(): void {
    this.state.fill(0);
  }

  getWin(): Float64Array {
    return this.Win;
  }

  getW(): Float64Array {
    return this.W;
  }

  getBias(): Float64Array {
    return this.bias;
  }

  getNFeatures(): number {
    return this.params.nFeatures;
  }

  loadWin(data: number[]): void {
    for (let i = 0; i < data.length && i < this.Win.length; i++) {
      this.Win[i] = data[i];
    }
  }

  loadW(data: number[]): void {
    for (let i = 0; i < data.length && i < this.W.length; i++) {
      this.W[i] = data[i];
    }
  }

  loadBias(data: number[]): void {
    for (let i = 0; i < data.length && i < this.bias.length; i++) {
      this.bias[i] = data[i];
    }
  }

  loadState(data: number[]): void {
    for (let i = 0; i < data.length && i < this.state.length; i++) {
      this.state[i] = data[i];
    }
  }
}

interface ReadoutConfig {
  nTargets: number;
  zDim: number;
  initScale: number;
}

interface ReadoutParams {
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
  reservoirSize: number;
  nFeatures: number;
}

class LinearReadout {
  private nTargets: number;
  private zDim: number;
  private Wout: Float64Array;

  constructor(config: ReadoutConfig, rng: RandomGenerator) {
    this.nTargets = config.nTargets;
    this.zDim = config.zDim;
    this.Wout = new Float64Array(config.nTargets * config.zDim);

    for (let i = 0; i < this.Wout.length; i++) {
      this.Wout[i] = rng.gaussian() * config.initScale;
    }
  }

  forward(
    z: Float64Array,
    zOff: number,
    dest: Float64Array,
    destOff: number,
  ): void {
    for (let t = 0; t < this.nTargets; t++) {
      let sum = 0;
      const rowOff = t * this.zDim;
      for (let j = 0; j < this.zDim; j++) {
        sum += this.Wout[rowOff + j] * z[zOff + j];
      }
      dest[destOff + t] = sum;
    }
  }

  updateRow(
    targetIdx: number,
    gain: Float64Array,
    gainOff: number,
    error: number,
    l2Lambda: number,
  ): void {
    const offset = targetIdx * this.zDim;
    const decay = 1 - l2Lambda;
    for (let j = 0; j < this.zDim; j++) {
      this.Wout[offset + j] = decay * this.Wout[offset + j] +
        error * gain[gainOff + j];
    }
  }

  getWout(): Float64Array {
    return this.Wout;
  }

  getNTargets(): number {
    return this.nTargets;
  }

  getZDim(): number {
    return this.zDim;
  }

  loadWout(data: number[]): void {
    for (let i = 0; i < data.length && i < this.Wout.length; i++) {
      this.Wout[i] = data[i];
    }
  }
}

class RLSState {
  P: Float64Array;
  g: Float64Array;
  k: Float64Array;
  zDim: number;

  constructor(zDim: number, delta: number) {
    this.zDim = zDim;
    this.P = new Float64Array(zDim * zDim);
    this.g = new Float64Array(zDim);
    this.k = new Float64Array(zDim);

    for (let i = 0; i < zDim; i++) {
      this.P[i * zDim + i] = delta;
    }
  }

  loadP(data: number[]): void {
    for (let i = 0; i < data.length && i < this.P.length; i++) {
      this.P[i] = data[i];
    }
  }
}

class RLSOptimizer {
  private state: RLSState;
  private lambda: number;
  private epsilon: number;

  constructor(zDim: number, delta: number, lambda: number, epsilon: number) {
    this.state = new RLSState(zDim, delta);
    this.lambda = lambda;
    this.epsilon = epsilon;
  }

  computeGain(z: Float64Array, zOff: number): void {
    const { P, g, k, zDim } = this.state;

    for (let i = 0; i < zDim; i++) {
      let sum = 0;
      for (let j = 0; j < zDim; j++) {
        sum += P[i * zDim + j] * z[zOff + j];
      }
      g[i] = sum;
    }

    let denom = this.lambda;
    for (let i = 0; i < zDim; i++) {
      denom += z[zOff + i] * g[i];
    }

    const invDenom = 1 / Math.max(denom, this.epsilon);
    for (let i = 0; i < zDim; i++) {
      k[i] = g[i] * invDenom;
    }
  }

  updateP(): void {
    const { P, g, k, zDim } = this.state;
    const invLambda = 1 / this.lambda;

    for (let i = 0; i < zDim; i++) {
      for (let j = 0; j < zDim; j++) {
        P[i * zDim + j] = (P[i * zDim + j] - k[i] * g[j]) * invLambda;
      }
    }

    for (let i = 0; i < zDim; i++) {
      if (P[i * zDim + i] < this.epsilon) {
        P[i * zDim + i] = this.epsilon;
      }
    }
  }

  getGain(): Float64Array {
    return this.state.k;
  }

  getP(): Float64Array {
    return this.state.P;
  }

  getZDim(): number {
    return this.state.zDim;
  }

  loadP(data: number[]): void {
    this.state.loadP(data);
  }
}

interface SerializedState {
  initialized: boolean;
  config: Required<ESNRegressionConfig>;
  nFeatures: number;
  nTargets: number;
  zDim: number;
  sampleCount: number;
  rngState: number;
  reservoirState: number[];
  Win: number[];
  W: number[];
  bias: number[];
  Wout: number[];
  P: number[];
  normalizerCount: number;
  normalizerMeans: number[];
  normalizerM2s: number[];
  ringBufferHead: number;
  ringBufferCount: number;
  ringBufferData: number[];
  residualHead: number;
  residualCount: number;
  residualData: number[];
}

class SerializationHelper {
  static serialize(state: SerializedState): string {
    return JSON.stringify(state);
  }

  static deserialize(json: string): SerializedState {
    return JSON.parse(json) as SerializedState;
  }
}

/**
 * Echo State Network (ESN) for multivariate regression with online learning.
 * Uses RLS (Recursive Least Squares) for readout training and Welford z-score normalization.
 *
 * @example
 * ```typescript
 * const esn = new ESNRegression({ reservoirSize: 256, spectralRadius: 0.9 });
 * const result = esn.fitOnline({ xCoordinates: [[1, 2], [3, 4]], yCoordinates: [[10], [20]] });
 * const prediction = esn.predict(1);
 * ```
 */
export class ESNRegression {
  private config: Required<ESNRegressionConfig>;
  private initialized: boolean;
  private nFeatures: number;
  private nTargets: number;
  private zDim: number;
  private sampleCount: number;

  private rng: RandomGenerator;
  private initialRngState: number;

  private reservoir: ESNReservoir | null;
  private readout: LinearReadout | null;
  private rls: RLSOptimizer | null;
  private normalizer: WelfordNormalizer | null;
  private ringBuffer: RingBuffer | null;
  private residualStats: ResidualStatsTracker | null;
  private outlierWeighter: OutlierDownweighter;
  private metrics: MetricsAccumulator;

  private xScratch: Float64Array | null;
  private xNormScratch: Float64Array | null;
  private zScratch: Float64Array | null;
  private yHatScratch: Float64Array | null;
  private errorScratch: Float64Array | null;
  private residualStdsScratch: Float64Array | null;
  private scratchReservoirState: Float64Array | null;
  private preActScratch: Float64Array | null;

  private fitResultObj: FitResult;
  private predictResultObj: PredictionResult;
  private predictionsArray: number[][];
  private lowerBoundsArray: number[][];
  private upperBoundsArray: number[][];

  /**
   * Creates a new ESNRegression instance.
   * @param config Configuration options for the ESN model
   */
  constructor(config: ESNRegressionConfig = {}) {
    this.config = {
      maxSequenceLength: config.maxSequenceLength ?? 64,
      reservoirSize: config.reservoirSize ?? 256,
      spectralRadius: config.spectralRadius ?? 0.9,
      leakRate: config.leakRate ?? 0.3,
      inputScale: config.inputScale ?? 1.0,
      biasScale: config.biasScale ?? 0.1,
      reservoirSparsity: config.reservoirSparsity ?? 0.9,
      inputSparsity: config.inputSparsity ?? 0.0,
      activation: config.activation ?? "tanh",
      useInputInReadout: config.useInputInReadout ?? true,
      useBiasInReadout: config.useBiasInReadout ?? true,
      readoutTraining: config.readoutTraining ?? "rls",
      rlsLambda: config.rlsLambda ?? 0.999,
      rlsDelta: config.rlsDelta ?? 1.0,
      epsilon: config.epsilon ?? 1e-8,
      l2Lambda: config.l2Lambda ?? 0.0001,
      gradientClipNorm: config.gradientClipNorm ?? 1.0,
      normalizationEpsilon: config.normalizationEpsilon ?? 1e-8,
      normalizationWarmup: config.normalizationWarmup ?? 10,
      outlierThreshold: config.outlierThreshold ?? 3.0,
      outlierMinWeight: config.outlierMinWeight ?? 0.1,
      residualWindowSize: config.residualWindowSize ?? 100,
      uncertaintyMultiplier: config.uncertaintyMultiplier ?? 1.96,
      weightInitScale: config.weightInitScale ?? 0.1,
      seed: config.seed ?? 42,
      verbose: config.verbose ?? false,
      rollforwardMode: config.rollforwardMode ?? "holdLastX",
    };

    this.initialized = false;
    this.nFeatures = 0;
    this.nTargets = 0;
    this.zDim = 0;
    this.sampleCount = 0;

    this.rng = new RandomGenerator(this.config.seed);
    this.initialRngState = this.rng.getState();

    this.reservoir = null;
    this.readout = null;
    this.rls = null;
    this.normalizer = null;
    this.ringBuffer = null;
    this.residualStats = null;
    this.outlierWeighter = new OutlierDownweighter();
    this.metrics = new MetricsAccumulator();

    this.xScratch = null;
    this.xNormScratch = null;
    this.zScratch = null;
    this.yHatScratch = null;
    this.errorScratch = null;
    this.residualStdsScratch = null;
    this.scratchReservoirState = null;
    this.preActScratch = null;

    this.fitResultObj = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1.0,
    };

    this.predictResultObj = {
      predictions: [],
      lowerBounds: [],
      upperBounds: [],
      confidence: 0,
    };

    this.predictionsArray = [];
    this.lowerBoundsArray = [];
    this.upperBoundsArray = [];
  }

  private initialize(nFeatures: number, nTargets: number): void {
    this.nFeatures = nFeatures;
    this.nTargets = nTargets;

    this.zDim = this.config.reservoirSize;
    if (this.config.useInputInReadout) this.zDim += nFeatures;
    if (this.config.useBiasInReadout) this.zDim += 1;

    const reservoirParams: ESNReservoirParams = {
      reservoirSize: this.config.reservoirSize,
      nFeatures: nFeatures,
      spectralRadius: this.config.spectralRadius,
      leakRate: this.config.leakRate,
      inputScale: this.config.inputScale,
      biasScale: this.config.biasScale,
      reservoirSparsity: this.config.reservoirSparsity,
      inputSparsity: this.config.inputSparsity,
      activation: this.config.activation,
    };

    this.reservoir = new ESNReservoir(reservoirParams, this.rng);

    const readoutConfig: ReadoutConfig = {
      nTargets: nTargets,
      zDim: this.zDim,
      initScale: this.config.weightInitScale,
    };

    this.readout = new LinearReadout(readoutConfig, this.rng);

    this.rls = new RLSOptimizer(
      this.zDim,
      this.config.rlsDelta,
      this.config.rlsLambda,
      this.config.epsilon,
    );

    this.normalizer = new WelfordNormalizer(nFeatures);

    this.ringBuffer = new RingBuffer(this.config.maxSequenceLength, nFeatures);

    this.residualStats = new ResidualStatsTracker(
      this.config.residualWindowSize,
      nTargets,
    );

    this.xScratch = new Float64Array(nFeatures);
    this.xNormScratch = new Float64Array(nFeatures);
    this.zScratch = new Float64Array(this.zDim);
    this.yHatScratch = new Float64Array(nTargets);
    this.errorScratch = new Float64Array(nTargets);
    this.residualStdsScratch = new Float64Array(nTargets);
    this.scratchReservoirState = new Float64Array(this.config.reservoirSize);
    this.preActScratch = new Float64Array(this.config.reservoirSize);

    const maxSteps = this.config.maxSequenceLength;
    this.predictionsArray = new Array(maxSteps);
    this.lowerBoundsArray = new Array(maxSteps);
    this.upperBoundsArray = new Array(maxSteps);
    for (let i = 0; i < maxSteps; i++) {
      this.predictionsArray[i] = new Array(nTargets).fill(0);
      this.lowerBoundsArray[i] = new Array(nTargets).fill(0);
      this.upperBoundsArray[i] = new Array(nTargets).fill(0);
    }

    this.initialized = true;
  }

  private buildZ(
    reservoirState: Float64Array,
    rOff: number,
    xNorm: Float64Array,
    xOff: number,
    dest: Float64Array,
    destOff: number,
  ): void {
    let idx = destOff;

    for (let i = 0; i < this.config.reservoirSize; i++) {
      dest[idx++] = reservoirState[rOff + i];
    }

    if (this.config.useInputInReadout) {
      for (let i = 0; i < this.nFeatures; i++) {
        dest[idx++] = xNorm[xOff + i];
      }
    }

    if (this.config.useBiasInReadout) {
      dest[idx++] = 1.0;
    }
  }

  /**
   * Train the model online with new samples.
   * xCoordinates.length MUST equal yCoordinates.length.
   *
   * @param params Object containing xCoordinates and yCoordinates arrays
   * @returns FitResult with training metrics (reused object - copy if persistence needed)
   */
  fitOnline(
    params: { xCoordinates: number[][]; yCoordinates: number[][] },
  ): FitResult {
    const { xCoordinates, yCoordinates } = params;

    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `fitOnline: xCoordinates.length (${xCoordinates.length}) must equal yCoordinates.length (${yCoordinates.length})`,
      );
    }

    if (xCoordinates.length === 0) {
      this.fitResultObj.samplesProcessed = 0;
      this.fitResultObj.averageLoss = 0;
      this.fitResultObj.gradientNorm = 0;
      this.fitResultObj.driftDetected = false;
      this.fitResultObj.sampleWeight = 1.0;
      return this.fitResultObj;
    }

    if (!this.initialized) {
      const nf = xCoordinates[0].length;
      const nt = yCoordinates[0].length;
      if (nf === 0) throw new Error("fitOnline: nFeatures must be > 0");
      if (nt === 0) throw new Error("fitOnline: nTargets must be > 0");
      this.initialize(nf, nt);
    }

    for (let i = 0; i < xCoordinates.length; i++) {
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

    this.metrics.reset();
    let lastWeight = 1.0;

    for (let i = 0; i < xCoordinates.length; i++) {
      this.ringBuffer!.push(xCoordinates[i]);

      for (let j = 0; j < this.nFeatures; j++) {
        this.xScratch![j] = xCoordinates[i][j];
      }

      this.normalizer!.update(this.xScratch!, 0);

      this.normalizer!.normalize(
        this.xScratch!,
        0,
        this.xNormScratch!,
        0,
        this.config.normalizationEpsilon,
      );

      this.reservoir!.update(this.xNormScratch!, 0);

      this.buildZ(
        this.reservoir!.getState(),
        0,
        this.xNormScratch!,
        0,
        this.zScratch!,
        0,
      );

      this.readout!.forward(this.zScratch!, 0, this.yHatScratch!, 0);

      const loss = LossFunction.mse(
        this.yHatScratch!,
        0,
        yCoordinates[i],
        this.errorScratch!,
        0,
        this.nTargets,
      );

      let gradNorm = TensorOps.norm2(this.errorScratch!, 0, this.nTargets);

      if (
        gradNorm > this.config.gradientClipNorm &&
        gradNorm > this.config.epsilon
      ) {
        const scale = this.config.gradientClipNorm / gradNorm;
        TensorOps.scale(this.errorScratch!, 0, scale, this.nTargets);
        gradNorm = this.config.gradientClipNorm;
      }

      this.residualStats!.getStds(
        this.residualStdsScratch!,
        0,
        this.config.epsilon,
      );

      const weight = this.outlierWeighter.compute(
        this.errorScratch!,
        0,
        this.residualStdsScratch!,
        0,
        this.nTargets,
        this.config.outlierThreshold,
        this.config.outlierMinWeight,
        this.config.epsilon,
      );
      lastWeight = weight;

      this.rls!.computeGain(this.zScratch!, 0);
      const gain = this.rls!.getGain();

      for (let t = 0; t < this.nTargets; t++) {
        this.readout!.updateRow(
          t,
          gain,
          0,
          this.errorScratch![t] * weight,
          this.config.l2Lambda,
        );
      }

      this.rls!.updateP();

      for (let t = 0; t < this.nTargets; t++) {
        this.errorScratch![t] = yCoordinates[i][t] - this.yHatScratch![t];
      }
      this.residualStats!.update(this.errorScratch!, 0);

      this.metrics.add(loss, gradNorm);
      this.sampleCount++;
    }

    this.fitResultObj.samplesProcessed = xCoordinates.length;
    this.fitResultObj.averageLoss = this.metrics.getAverageLoss();
    this.fitResultObj.gradientNorm = this.metrics.getAverageGradNorm();
    this.fitResultObj.driftDetected = false;
    this.fitResultObj.sampleWeight = lastWeight;

    return this.fitResultObj;
  }

  /**
   * Predict future values for multiple steps ahead using roll-forward.
   * Uses ONLY internal RingBuffer state (no external latest-X required).
   *
   * @param futureSteps Number of steps to predict (1 to maxSequenceLength)
   * @returns PredictionResult with predictions and uncertainty bounds (reused object - copy if persistence needed)
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.initialized || this.ringBuffer!.size() === 0) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }

    if (!Number.isInteger(futureSteps) || futureSteps < 1) {
      throw new Error("predict: futureSteps must be an integer >= 1");
    }

    if (futureSteps > this.config.maxSequenceLength) {
      throw new Error(
        `predict: futureSteps (${futureSteps}) must be <= maxSequenceLength (${this.config.maxSequenceLength})`,
      );
    }

    this.reservoir!.copyStateTo(this.scratchReservoirState!, 0);

    this.ringBuffer!.getLast(this.xScratch!, 0);

    this.residualStats!.getStds(
      this.residualStdsScratch!,
      0,
      this.config.epsilon,
    );

    const useAutoregressive =
      this.config.rollforwardMode === "autoregressive" &&
      this.nFeatures === this.nTargets;

    for (let step = 0; step < futureSteps; step++) {
      this.normalizer!.normalize(
        this.xScratch!,
        0,
        this.xNormScratch!,
        0,
        this.config.normalizationEpsilon,
      );

      this.reservoir!.updateWithState(
        this.xNormScratch!,
        0,
        this.scratchReservoirState!,
        0,
        this.scratchReservoirState!,
        0,
        this.preActScratch!,
        0,
      );

      this.buildZ(
        this.scratchReservoirState!,
        0,
        this.xNormScratch!,
        0,
        this.zScratch!,
        0,
      );

      this.readout!.forward(this.zScratch!, 0, this.yHatScratch!, 0);

      const horizonFactor = Math.sqrt(step + 1);

      for (let t = 0; t < this.nTargets; t++) {
        const pred = this.yHatScratch![t];
        const sigma = this.residualStdsScratch![t] * horizonFactor;
        const margin = this.config.uncertaintyMultiplier * sigma;

        this.predictionsArray[step][t] = isFinite(pred) ? pred : 0;
        this.lowerBoundsArray[step][t] = isFinite(pred - margin)
          ? pred - margin
          : pred;
        this.upperBoundsArray[step][t] = isFinite(pred + margin)
          ? pred + margin
          : pred;
      }

      if (useAutoregressive && step < futureSteps - 1) {
        for (let t = 0; t < this.nTargets; t++) {
          this.xScratch![t] = this.yHatScratch![t];
        }
      }
    }

    let avgStd = 0;
    for (let t = 0; t < this.nTargets; t++) {
      avgStd += this.residualStdsScratch![t];
    }
    avgStd /= this.nTargets;

    let confidence = 1 / (1 + avgStd);
    confidence = Math.max(0, Math.min(1, confidence));
    if (!isFinite(confidence)) confidence = 0;

    this.predictResultObj.predictions = this.predictionsArray.slice(
      0,
      futureSteps,
    );
    this.predictResultObj.lowerBounds = this.lowerBoundsArray.slice(
      0,
      futureSteps,
    );
    this.predictResultObj.upperBounds = this.upperBoundsArray.slice(
      0,
      futureSteps,
    );
    this.predictResultObj.confidence = confidence;

    return this.predictResultObj;
  }

  /**
   * Get a summary of the model configuration and state.
   * @returns ModelSummary object
   */
  getModelSummary(): ModelSummary {
    const totalParams = this.initialized
      ? this.config.reservoirSize * this.nFeatures +
        this.config.reservoirSize * this.config.reservoirSize +
        this.config.reservoirSize +
        this.nTargets * this.zDim
      : 0;

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
   * Get the model weights for inspection or transfer.
   * @returns WeightInfo object with all weight matrices
   */
  getWeights(): WeightInfo {
    if (!this.initialized) {
      return { weights: [] };
    }

    return {
      weights: [
        {
          name: "Win",
          shape: [this.config.reservoirSize, this.nFeatures],
          values: Array.from(this.reservoir!.getWin()),
        },
        {
          name: "W",
          shape: [this.config.reservoirSize, this.config.reservoirSize],
          values: Array.from(this.reservoir!.getW()),
        },
        {
          name: "bias",
          shape: [this.config.reservoirSize],
          values: Array.from(this.reservoir!.getBias()),
        },
        {
          name: "Wout",
          shape: [this.nTargets, this.zDim],
          values: Array.from(this.readout!.getWout()),
        },
        {
          name: "P",
          shape: [this.zDim, this.zDim],
          values: Array.from(this.rls!.getP()),
        },
        {
          name: "reservoirState",
          shape: [this.config.reservoirSize],
          values: Array.from(this.reservoir!.getState()),
        },
      ],
    };
  }

  /**
   * Get normalization statistics for the input features.
   * @returns NormalizationStats object
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.initialized || !this.normalizer) {
      return { means: [], stds: [], count: 0, isActive: false };
    }

    return {
      means: Array.from(this.normalizer.getMeans()),
      stds: Array.from(
        this.normalizer.getStds(this.config.normalizationEpsilon),
      ),
      count: this.normalizer.getCount(),
      isActive: this.normalizer.getCount() >= this.config.normalizationWarmup,
    };
  }

  /**
   * Reset the model to its initial state (before any training).
   */
  reset(): void {
    this.rng.setState(this.initialRngState);
    this.initialized = false;
    this.nFeatures = 0;
    this.nTargets = 0;
    this.zDim = 0;
    this.sampleCount = 0;

    this.reservoir = null;
    this.readout = null;
    this.rls = null;
    this.normalizer = null;
    this.ringBuffer = null;
    this.residualStats = null;

    this.xScratch = null;
    this.xNormScratch = null;
    this.zScratch = null;
    this.yHatScratch = null;
    this.errorScratch = null;
    this.residualStdsScratch = null;
    this.scratchReservoirState = null;
    this.preActScratch = null;

    this.metrics.reset();

    this.fitResultObj.samplesProcessed = 0;
    this.fitResultObj.averageLoss = 0;
    this.fitResultObj.gradientNorm = 0;
    this.fitResultObj.driftDetected = false;
    this.fitResultObj.sampleWeight = 1.0;
  }

  /**
   * Serialize the model state to a JSON string.
   * @returns JSON string containing all model state
   */
  save(): string {
    if (!this.initialized) {
      const state: SerializedState = {
        initialized: false,
        config: this.config,
        nFeatures: 0,
        nTargets: 0,
        zDim: 0,
        sampleCount: 0,
        rngState: this.rng.getState(),
        reservoirState: [],
        Win: [],
        W: [],
        bias: [],
        Wout: [],
        P: [],
        normalizerCount: 0,
        normalizerMeans: [],
        normalizerM2s: [],
        ringBufferHead: 0,
        ringBufferCount: 0,
        ringBufferData: [],
        residualHead: 0,
        residualCount: 0,
        residualData: [],
      };
      return SerializationHelper.serialize(state);
    }

    const state: SerializedState = {
      initialized: true,
      config: this.config,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      zDim: this.zDim,
      sampleCount: this.sampleCount,
      rngState: this.rng.getState(),
      reservoirState: Array.from(this.reservoir!.getState()),
      Win: Array.from(this.reservoir!.getWin()),
      W: Array.from(this.reservoir!.getW()),
      bias: Array.from(this.reservoir!.getBias()),
      Wout: Array.from(this.readout!.getWout()),
      P: Array.from(this.rls!.getP()),
      normalizerCount: this.normalizer!.getCount(),
      normalizerMeans: Array.from(this.normalizer!.getMeans()),
      normalizerM2s: Array.from(this.normalizer!.getM2s()),
      ringBufferHead: this.ringBuffer!.getHead(),
      ringBufferCount: this.ringBuffer!.size(),
      ringBufferData: Array.from(this.ringBuffer!.getData()),
      residualHead: this.residualStats!.getHead(),
      residualCount: this.residualStats!.getCount(),
      residualData: Array.from(this.residualStats!.getData()),
    };

    return SerializationHelper.serialize(state);
  }

  /**
   * Load model state from a JSON string.
   * @param json JSON string from save()
   */
  load(json: string): void {
    const state = SerializationHelper.deserialize(json);

    this.config = state.config;

    if (!state.initialized) {
      this.reset();
      this.rng.setState(state.rngState);
      return;
    }

    this.initialize(state.nFeatures, state.nTargets);

    this.sampleCount = state.sampleCount;
    this.rng.setState(state.rngState);

    this.reservoir!.loadWin(state.Win);
    this.reservoir!.loadW(state.W);
    this.reservoir!.loadBias(state.bias);
    this.reservoir!.loadState(state.reservoirState);

    this.readout!.loadWout(state.Wout);

    this.rls!.loadP(state.P);

    this.normalizer!.loadState(
      state.normalizerCount,
      state.normalizerMeans,
      state.normalizerM2s,
    );

    this.ringBuffer!.loadState(
      state.ringBufferHead,
      state.ringBufferCount,
      state.ringBufferData,
    );

    this.residualStats!.loadState(
      state.residualHead,
      state.residualCount,
      state.residualData,
    );
  }
}
