export interface ESNRegressionConfig {
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
  uncertaintyMultiplier: number;
  weightInitScale: number;
  seed: number;
  verbose: boolean;
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

const DEFAULT_CONFIG: ESNRegressionConfig = {
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
  uncertaintyMultiplier: 1.96,
  weightInitScale: 0.1,
  seed: 42,
  verbose: false,
};

class TensorShape {
  readonly dims: readonly number[];
  readonly size: number;
  readonly strides: readonly number[];

  constructor(dims: number[]) {
    this.dims = Object.freeze([...dims]);
    let size = 1;
    for (let i = 0; i < dims.length; i++) {
      size *= dims[i];
    }
    this.size = size;
    const strides: number[] = new Array(dims.length);
    let stride = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= dims[i];
    }
    this.strides = Object.freeze(strides);
  }

  index(...indices: number[]): number {
    let idx = 0;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this.strides[i];
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
    this.offset = offset;
    this.shape = shape;
  }

  get(i: number): number {
    return this.data[this.offset + i];
  }

  set(i: number, v: number): void {
    this.data[this.offset + i] = v;
  }

  get2d(row: number, col: number): number {
    return this.data[this.offset + row * this.shape.strides[0] + col];
  }

  set2d(row: number, col: number, v: number): void {
    this.data[this.offset + row * this.shape.strides[0] + col] = v;
  }

  fill(v: number): void {
    const end = this.offset + this.shape.size;
    for (let i = this.offset; i < end; i++) {
      this.data[i] = v;
    }
  }

  copyFrom(src: Float64Array, srcOffset: number, length: number): void {
    for (let i = 0; i < length; i++) {
      this.data[this.offset + i] = src[srcOffset + i];
    }
  }

  copyTo(dst: Float64Array, dstOffset: number, length: number): void {
    for (let i = 0; i < length; i++) {
      dst[dstOffset + i] = this.data[this.offset + i];
    }
  }
}

class BufferPool {
  private pools: Map<number, Float64Array[]> = new Map();

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
    pool.push(buffer);
  }

  clear(): void {
    this.pools.clear();
  }
}

class TensorArena {
  private buffer: Float64Array;
  private offset: number = 0;
  private shapes: Map<string, TensorShape> = new Map();
  private views: Map<string, TensorView> = new Map();

  constructor(totalSize: number) {
    this.buffer = new Float64Array(totalSize);
  }

  allocate(name: string, dims: number[]): TensorView {
    let shape = this.shapes.get(name);
    if (!shape) {
      shape = new TensorShape(dims);
      this.shapes.set(name, shape);
    }
    const view = new TensorView(this.buffer, this.offset, shape);
    this.views.set(name, view);
    this.offset += shape.size;
    return view;
  }

  get(name: string): TensorView | undefined {
    return this.views.get(name);
  }

  reset(): void {
    this.offset = 0;
    this.views.clear();
    this.buffer.fill(0);
  }

  getBuffer(): Float64Array {
    return this.buffer;
  }
}

class TensorOps {
  static matVec(
    A: Float64Array,
    aOffset: number,
    rows: number,
    cols: number,
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
  ): void {
    for (let i = 0; i < rows; i++) {
      let sum = 0.0;
      const rowStart = aOffset + i * cols;
      for (let j = 0; j < cols; j++) {
        sum += A[rowStart + j] * x[xOffset + j];
      }
      y[yOffset + i] = sum;
    }
  }

  static dot(
    a: Float64Array,
    aOffset: number,
    b: Float64Array,
    bOffset: number,
    length: number,
  ): number {
    let sum = 0.0;
    for (let i = 0; i < length; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  static scale(
    x: Float64Array,
    xOffset: number,
    alpha: number,
    y: Float64Array,
    yOffset: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      y[yOffset + i] = alpha * x[xOffset + i];
    }
  }

  static add(
    a: Float64Array,
    aOffset: number,
    b: Float64Array,
    bOffset: number,
    y: Float64Array,
    yOffset: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      y[yOffset + i] = a[aOffset + i] + b[bOffset + i];
    }
  }

  static axpy(
    alpha: number,
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      y[yOffset + i] += alpha * x[xOffset + i];
    }
  }

  static copy(
    src: Float64Array,
    srcOffset: number,
    dst: Float64Array,
    dstOffset: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      dst[dstOffset + i] = src[srcOffset + i];
    }
  }

  static norm(x: Float64Array, offset: number, length: number): number {
    let sum = 0.0;
    for (let i = 0; i < length; i++) {
      const v = x[offset + i];
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  static fill(
    x: Float64Array,
    offset: number,
    length: number,
    value: number,
  ): void {
    for (let i = 0; i < length; i++) {
      x[offset + i] = value;
    }
  }

  static sparseMatVec(
    A: Float64Array,
    aOffset: number,
    rows: number,
    cols: number,
    mask: Uint8Array,
    maskOffset: number,
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
  ): void {
    for (let i = 0; i < rows; i++) {
      let sum = 0.0;
      const rowStart = aOffset + i * cols;
      const maskRowStart = maskOffset + i * cols;
      for (let j = 0; j < cols; j++) {
        if (mask[maskRowStart + j]) {
          sum += A[rowStart + j] * x[xOffset + j];
        }
      }
      y[yOffset + i] = sum;
    }
  }
}

class RandomGenerator {
  private state: number;

  constructor(seed: number) {
    this.state = (seed >>> 0) || 1;
    for (let i = 0; i < 20; i++) {
      this.next();
    }
  }

  private next(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }

  uniform(): number {
    return this.next() / 4294967296;
  }

  normal(): number {
    const u1 = Math.max(this.uniform(), 1e-15);
    const u2 = this.uniform();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 6.283185307179586 * u2;
    return r * Math.cos(theta);
  }

  range(min: number, max: number): number {
    return min + this.uniform() * (max - min);
  }

  getState(): number {
    return this.state;
  }

  setState(state: number): void {
    this.state = (state >>> 0) || 1;
  }
}

class ActivationOps {
  static tanh(
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      y[yOffset + i] = Math.tanh(x[xOffset + i]);
    }
  }

  static relu(
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      const v = x[xOffset + i];
      y[yOffset + i] = v > 0 ? v : 0;
    }
  }

  static apply(
    type: "tanh" | "relu",
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    length: number,
  ): void {
    if (type === "tanh") {
      ActivationOps.tanh(x, xOffset, y, yOffset, length);
    } else {
      ActivationOps.relu(x, xOffset, y, yOffset, length);
    }
  }
}

class WelfordAccumulator {
  count: number = 0;
  mean: number = 0;
  m2: number = 0;

  update(value: number): void {
    if (!Number.isFinite(value)) return;
    this.count++;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    const delta2 = value - this.mean;
    this.m2 += delta * delta2;
  }

  getVariance(): number {
    if (this.count < 2) return 0;
    return Math.max(0, this.m2 / (this.count - 1));
  }

  getStd(): number {
    return Math.sqrt(this.getVariance());
  }

  reset(): void {
    this.count = 0;
    this.mean = 0;
    this.m2 = 0;
  }
}

class WelfordNormalizer {
  private accumulators: WelfordAccumulator[];
  private nFeatures: number;
  private epsilon: number;
  private warmup: number;
  private cachedStds: Float64Array;
  private cachedMeans: Float64Array;

  constructor(nFeatures: number, epsilon: number, warmup: number) {
    this.nFeatures = nFeatures;
    this.epsilon = epsilon;
    this.warmup = warmup;
    this.accumulators = new Array(nFeatures);
    for (let i = 0; i < nFeatures; i++) {
      this.accumulators[i] = new WelfordAccumulator();
    }
    this.cachedStds = new Float64Array(nFeatures);
    this.cachedMeans = new Float64Array(nFeatures);
    this.cachedStds.fill(1.0);
  }

  update(input: Float64Array, offset: number): void {
    for (let i = 0; i < this.nFeatures; i++) {
      this.accumulators[i].update(input[offset + i]);
    }
    this.updateCache();
  }

  private updateCache(): void {
    const count = this.accumulators[0].count;
    for (let i = 0; i < this.nFeatures; i++) {
      this.cachedMeans[i] = this.accumulators[i].mean;
      const rawStd = this.accumulators[i].getStd();
      if (count < this.warmup) {
        this.cachedStds[i] = Math.max(rawStd, 1.0);
      } else {
        this.cachedStds[i] = Math.max(rawStd, this.epsilon);
      }
    }
  }

  normalize(
    input: Float64Array,
    inputOffset: number,
    output: Float64Array,
    outputOffset: number,
  ): void {
    for (let i = 0; i < this.nFeatures; i++) {
      output[outputOffset + i] =
        (input[inputOffset + i] - this.cachedMeans[i]) / this.cachedStds[i];
    }
  }

  denormalize(
    input: Float64Array,
    inputOffset: number,
    output: Float64Array,
    outputOffset: number,
  ): void {
    for (let i = 0; i < this.nFeatures; i++) {
      output[outputOffset + i] = input[inputOffset + i] * this.cachedStds[i] +
        this.cachedMeans[i];
    }
  }

  isActive(): boolean {
    return (
      this.accumulators.length > 0 &&
      this.accumulators[0].count >= this.warmup
    );
  }

  getCount(): number {
    return this.accumulators.length > 0 ? this.accumulators[0].count : 0;
  }

  getMeans(): number[] {
    const means: number[] = new Array(this.nFeatures);
    for (let i = 0; i < this.nFeatures; i++) {
      means[i] = this.cachedMeans[i];
    }
    return means;
  }

  getStds(): number[] {
    const stds: number[] = new Array(this.nFeatures);
    for (let i = 0; i < this.nFeatures; i++) {
      stds[i] = this.cachedStds[i];
    }
    return stds;
  }

  getCachedStds(): Float64Array {
    return this.cachedStds;
  }

  reset(): void {
    for (let i = 0; i < this.nFeatures; i++) {
      this.accumulators[i].reset();
    }
    this.cachedStds.fill(1.0);
    this.cachedMeans.fill(0);
  }

  serialize(): object {
    return {
      nFeatures: this.nFeatures,
      epsilon: this.epsilon,
      warmup: this.warmup,
      accumulators: this.accumulators.map((acc) => ({
        count: acc.count,
        mean: acc.mean,
        m2: acc.m2,
      })),
    };
  }

  static deserialize(data: any): WelfordNormalizer {
    const normalizer = new WelfordNormalizer(
      data.nFeatures,
      data.epsilon,
      data.warmup,
    );
    for (let i = 0; i < data.nFeatures; i++) {
      normalizer.accumulators[i].count = data.accumulators[i].count;
      normalizer.accumulators[i].mean = data.accumulators[i].mean;
      normalizer.accumulators[i].m2 = data.accumulators[i].m2;
    }
    normalizer.updateCache();
    return normalizer;
  }
}

class ResidualStatsTracker {
  private accumulators: WelfordAccumulator[];
  private nTargets: number;
  private cachedStds: Float64Array;

  constructor(nTargets: number) {
    this.nTargets = nTargets;
    this.accumulators = new Array(nTargets);
    for (let i = 0; i < nTargets; i++) {
      this.accumulators[i] = new WelfordAccumulator();
    }
    this.cachedStds = new Float64Array(nTargets);
  }

  update(
    prediction: Float64Array,
    predOffset: number,
    target: Float64Array,
    targetOffset: number,
  ): void {
    for (let i = 0; i < this.nTargets; i++) {
      const residual = prediction[predOffset + i] - target[targetOffset + i];
      this.accumulators[i].update(residual);
      this.cachedStds[i] = this.accumulators[i].getStd();
    }
  }

  getStds(): Float64Array {
    return this.cachedStds;
  }

  getStdAt(idx: number): number {
    return this.cachedStds[idx];
  }

  getMeans(): Float64Array {
    const means = new Float64Array(this.nTargets);
    for (let i = 0; i < this.nTargets; i++) {
      means[i] = this.accumulators[i].mean;
    }
    return means;
  }

  getCount(): number {
    return this.accumulators.length > 0 ? this.accumulators[0].count : 0;
  }

  reset(): void {
    for (let i = 0; i < this.nTargets; i++) {
      this.accumulators[i].reset();
      this.cachedStds[i] = 0;
    }
  }

  serialize(): object {
    return {
      nTargets: this.nTargets,
      accumulators: this.accumulators.map((acc) => ({
        count: acc.count,
        mean: acc.mean,
        m2: acc.m2,
      })),
    };
  }

  static deserialize(data: any): ResidualStatsTracker {
    const tracker = new ResidualStatsTracker(data.nTargets);
    for (let i = 0; i < data.nTargets; i++) {
      tracker.accumulators[i].count = data.accumulators[i].count;
      tracker.accumulators[i].mean = data.accumulators[i].mean;
      tracker.accumulators[i].m2 = data.accumulators[i].m2;
      tracker.cachedStds[i] = tracker.accumulators[i].getStd();
    }
    return tracker;
  }
}

class OutlierDownweighter {
  private threshold: number;
  private minWeight: number;

  constructor(threshold: number, minWeight: number) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }

  computeWeight(
    prediction: Float64Array,
    predOffset: number,
    target: Float64Array,
    targetOffset: number,
    residualStds: Float64Array,
    nTargets: number,
    epsilon: number,
  ): number {
    let maxZScore = 0;
    for (let i = 0; i < nTargets; i++) {
      const residual = Math.abs(
        prediction[predOffset + i] - target[targetOffset + i],
      );
      const std = residualStds[i] > epsilon ? residualStds[i] : 1.0;
      const zScore = residual / std;
      if (zScore > maxZScore) {
        maxZScore = zScore;
      }
    }
    if (maxZScore <= this.threshold) {
      return 1.0;
    }
    const excess = maxZScore - this.threshold;
    const weight = Math.exp(-0.5 * excess * excess);
    return Math.max(weight, this.minWeight);
  }
}

class LossFunction {
  static mse(
    prediction: Float64Array,
    predOffset: number,
    target: Float64Array,
    targetOffset: number,
    length: number,
  ): number {
    let sum = 0.0;
    for (let i = 0; i < length; i++) {
      const diff = prediction[predOffset + i] - target[targetOffset + i];
      sum += diff * diff;
    }
    return sum / length;
  }

  static computeError(
    prediction: Float64Array,
    predOffset: number,
    target: Float64Array,
    targetOffset: number,
    error: Float64Array,
    errorOffset: number,
    length: number,
  ): void {
    for (let i = 0; i < length; i++) {
      error[errorOffset + i] = prediction[predOffset + i] -
        target[targetOffset + i];
    }
  }
}

class MetricsAccumulator {
  private lossSum: number = 0;
  private lossCount: number = 0;
  private lastGradientNorm: number = 0;
  private lastSampleWeight: number = 1;

  addLoss(loss: number): void {
    if (Number.isFinite(loss)) {
      this.lossSum += loss;
      this.lossCount++;
    }
  }

  setGradientNorm(norm: number): void {
    this.lastGradientNorm = Number.isFinite(norm) ? norm : 0;
  }

  setSampleWeight(weight: number): void {
    this.lastSampleWeight = weight;
  }

  getAverageLoss(): number {
    if (this.lossCount === 0) return 0;
    return this.lossSum / this.lossCount;
  }

  getGradientNorm(): number {
    return this.lastGradientNorm;
  }

  getSampleWeight(): number {
    return this.lastSampleWeight;
  }

  getCount(): number {
    return this.lossCount;
  }

  reset(): void {
    this.lossSum = 0;
    this.lossCount = 0;
    this.lastGradientNorm = 0;
    this.lastSampleWeight = 1;
  }
}

class ReservoirInitMask {
  static generate(
    rows: number,
    cols: number,
    sparsity: number,
    rng: RandomGenerator,
  ): Uint8Array {
    const mask = new Uint8Array(rows * cols);
    const threshold = 1.0 - sparsity;
    for (let i = 0; i < rows * cols; i++) {
      mask[i] = rng.uniform() < threshold ? 1 : 0;
    }
    return mask;
  }
}

class SpectralRadiusScaler {
  static scaleToSpectralRadius(
    W: Float64Array,
    offset: number,
    size: number,
    mask: Uint8Array,
    maskOffset: number,
    targetRadius: number,
    rng: RandomGenerator,
    epsilon: number,
    iterations: number = 30,
  ): number {
    const v = new Float64Array(size);
    const vNew = new Float64Array(size);

    for (let i = 0; i < size; i++) {
      v[i] = rng.normal();
    }

    let norm = TensorOps.norm(v, 0, size);
    if (norm < epsilon) {
      for (let i = 0; i < size; i++) {
        v[i] = 1.0 / Math.sqrt(size);
      }
      norm = 1.0;
    } else {
      for (let i = 0; i < size; i++) {
        v[i] /= norm;
      }
    }

    let eigenvalue = 1.0;

    for (let iter = 0; iter < iterations; iter++) {
      TensorOps.sparseMatVec(
        W,
        offset,
        size,
        size,
        mask,
        maskOffset,
        v,
        0,
        vNew,
        0,
      );
      eigenvalue = TensorOps.norm(vNew, 0, size);
      if (eigenvalue < epsilon) {
        eigenvalue = epsilon;
        break;
      }
      const invEig = 1.0 / eigenvalue;
      for (let i = 0; i < size; i++) {
        v[i] = vNew[i] * invEig;
      }
    }

    if (eigenvalue > epsilon) {
      const scaleFactor = targetRadius / eigenvalue;
      const n = size * size;
      for (let i = 0; i < n; i++) {
        if (mask[maskOffset + i]) {
          W[offset + i] *= scaleFactor;
        }
      }
    }

    return targetRadius;
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
  seed: number;
  epsilon: number;
}

class ESNReservoir {
  private readonly params: ESNReservoirParams;
  private W: Float64Array;
  private Win: Float64Array;
  private bias: Float64Array;
  private Wmask: Uint8Array;
  private WinMask: Uint8Array;
  private state: Float64Array;
  private preActivation: Float64Array;
  private temp1: Float64Array;
  private temp2: Float64Array;

  constructor(params: ESNReservoirParams) {
    this.params = { ...params };
    const { reservoirSize, nFeatures } = params;
    this.W = new Float64Array(reservoirSize * reservoirSize);
    this.Win = new Float64Array(reservoirSize * nFeatures);
    this.bias = new Float64Array(reservoirSize);
    this.state = new Float64Array(reservoirSize);
    this.preActivation = new Float64Array(reservoirSize);
    this.temp1 = new Float64Array(reservoirSize);
    this.temp2 = new Float64Array(reservoirSize);
    this.Wmask = new Uint8Array(0);
    this.WinMask = new Uint8Array(0);
    this.initialize();
  }

  private initialize(): void {
    const {
      reservoirSize,
      nFeatures,
      spectralRadius,
      inputScale,
      biasScale,
      reservoirSparsity,
      inputSparsity,
      seed,
      epsilon,
    } = this.params;

    const rng = new RandomGenerator(seed);

    this.Wmask = ReservoirInitMask.generate(
      reservoirSize,
      reservoirSize,
      reservoirSparsity,
      rng,
    );
    this.WinMask = ReservoirInitMask.generate(
      reservoirSize,
      nFeatures,
      inputSparsity,
      rng,
    );

    for (let i = 0; i < reservoirSize * reservoirSize; i++) {
      if (this.Wmask[i]) {
        this.W[i] = rng.range(-1, 1);
      }
    }

    SpectralRadiusScaler.scaleToSpectralRadius(
      this.W,
      0,
      reservoirSize,
      this.Wmask,
      0,
      spectralRadius,
      rng,
      epsilon,
    );

    for (let i = 0; i < reservoirSize * nFeatures; i++) {
      if (this.WinMask[i]) {
        this.Win[i] = rng.range(-1, 1) * inputScale;
      }
    }

    for (let i = 0; i < reservoirSize; i++) {
      this.bias[i] = rng.range(-1, 1) * biasScale;
    }

    this.state.fill(0);
  }

  update(input: Float64Array, inputOffset: number): void {
    const { reservoirSize, nFeatures, leakRate, activation } = this.params;

    TensorOps.matVec(
      this.Win,
      0,
      reservoirSize,
      nFeatures,
      input,
      inputOffset,
      this.temp1,
      0,
    );
    TensorOps.sparseMatVec(
      this.W,
      0,
      reservoirSize,
      reservoirSize,
      this.Wmask,
      0,
      this.state,
      0,
      this.temp2,
      0,
    );

    for (let i = 0; i < reservoirSize; i++) {
      this.preActivation[i] = this.temp1[i] + this.temp2[i] + this.bias[i];
    }

    ActivationOps.apply(
      activation,
      this.preActivation,
      0,
      this.temp1,
      0,
      reservoirSize,
    );

    const oneMinusLeak = 1.0 - leakRate;
    for (let i = 0; i < reservoirSize; i++) {
      this.state[i] = oneMinusLeak * this.state[i] + leakRate * this.temp1[i];
    }
  }

  updateScratch(
    input: Float64Array,
    inputOffset: number,
    scratchState: Float64Array,
    scratchOffset: number,
  ): void {
    const { reservoirSize, nFeatures, leakRate, activation } = this.params;

    TensorOps.matVec(
      this.Win,
      0,
      reservoirSize,
      nFeatures,
      input,
      inputOffset,
      this.temp1,
      0,
    );
    TensorOps.sparseMatVec(
      this.W,
      0,
      reservoirSize,
      reservoirSize,
      this.Wmask,
      0,
      scratchState,
      scratchOffset,
      this.temp2,
      0,
    );

    for (let i = 0; i < reservoirSize; i++) {
      this.preActivation[i] = this.temp1[i] + this.temp2[i] + this.bias[i];
    }

    ActivationOps.apply(
      activation,
      this.preActivation,
      0,
      this.temp1,
      0,
      reservoirSize,
    );

    const oneMinusLeak = 1.0 - leakRate;
    for (let i = 0; i < reservoirSize; i++) {
      scratchState[scratchOffset + i] =
        oneMinusLeak * scratchState[scratchOffset + i] +
        leakRate * this.temp1[i];
    }
  }

  getState(): Float64Array {
    return this.state;
  }

  copyStateTo(dst: Float64Array, dstOffset: number): void {
    TensorOps.copy(this.state, 0, dst, dstOffset, this.params.reservoirSize);
  }

  getReservoirSize(): number {
    return this.params.reservoirSize;
  }

  resetState(): void {
    this.state.fill(0);
  }

  getW(): Float64Array {
    return this.W;
  }

  getWin(): Float64Array {
    return this.Win;
  }

  getBias(): Float64Array {
    return this.bias;
  }

  getWmask(): Uint8Array {
    return this.Wmask;
  }

  getWinMask(): Uint8Array {
    return this.WinMask;
  }

  serialize(): object {
    return {
      params: this.params,
      W: Array.from(this.W),
      Win: Array.from(this.Win),
      bias: Array.from(this.bias),
      Wmask: Array.from(this.Wmask),
      WinMask: Array.from(this.WinMask),
      state: Array.from(this.state),
    };
  }

  static deserialize(data: any): ESNReservoir {
    const reservoir = new ESNReservoir(data.params);
    reservoir.W = new Float64Array(data.W);
    reservoir.Win = new Float64Array(data.Win);
    reservoir.bias = new Float64Array(data.bias);
    reservoir.Wmask = new Uint8Array(data.Wmask);
    reservoir.WinMask = new Uint8Array(data.WinMask);
    reservoir.state = new Float64Array(data.state);
    return reservoir;
  }
}

class RLSState {
  P: Float64Array;
  zDim: number;

  constructor(zDim: number, delta: number) {
    this.zDim = zDim;
    this.P = new Float64Array(zDim * zDim);
    const invDelta = 1.0 / Math.max(delta, 1e-10);
    for (let i = 0; i < zDim; i++) {
      this.P[i * zDim + i] = invDelta;
    }
  }

  serialize(): object {
    return {
      zDim: this.zDim,
      P: Array.from(this.P),
    };
  }

  static deserialize(data: any): RLSState {
    const state = new RLSState(data.zDim, 1.0);
    state.P = new Float64Array(data.P);
    return state;
  }
}

class RLSOptimizer {
  private state: RLSState;
  private lambda: number;
  private l2Lambda: number;
  private epsilon: number;
  private Pz: Float64Array;
  private gain: Float64Array;

  constructor(
    zDim: number,
    lambda: number,
    delta: number,
    l2Lambda: number,
    epsilon: number,
  ) {
    this.state = new RLSState(zDim, delta);
    this.lambda = lambda;
    this.l2Lambda = l2Lambda;
    this.epsilon = epsilon;
    this.Pz = new Float64Array(zDim);
    this.gain = new Float64Array(zDim);
  }

  update(
    z: Float64Array,
    zOffset: number,
    prediction: Float64Array,
    predOffset: number,
    target: Float64Array,
    targetOffset: number,
    Wout: Float64Array,
    woutOffset: number,
    nTargets: number,
    sampleWeight: number,
  ): number {
    const zDim = this.state.zDim;
    const P = this.state.P;

    TensorOps.matVec(P, 0, zDim, zDim, z, zOffset, this.Pz, 0);

    let gamma = TensorOps.dot(z, zOffset, this.Pz, 0, zDim);
    let denom = this.lambda + gamma;

    if (Math.abs(denom) < this.epsilon) {
      denom = denom >= 0 ? this.epsilon : -this.epsilon;
    }

    const invDenom = sampleWeight / denom;
    for (let i = 0; i < zDim; i++) {
      this.gain[i] = this.Pz[i] * invDenom;
    }

    let gradNormSum = 0.0;
    for (let k = 0; k < nTargets; k++) {
      const error = target[targetOffset + k] - prediction[predOffset + k];
      gradNormSum += error * error;
      const rowOffset = woutOffset + k * zDim;
      for (let j = 0; j < zDim; j++) {
        Wout[rowOffset + j] += this.gain[j] * error;
      }
    }

    if (this.l2Lambda > 0) {
      const decay = 1.0 - this.l2Lambda * sampleWeight;
      const n = nTargets * zDim;
      for (let i = 0; i < n; i++) {
        Wout[woutOffset + i] *= decay;
      }
    }

    const invLambda = 1.0 / this.lambda;
    for (let i = 0; i < zDim; i++) {
      const gainI = this.gain[i];
      const rowOffset = i * zDim;
      for (let j = 0; j < zDim; j++) {
        P[rowOffset + j] = (P[rowOffset + j] - gainI * this.Pz[j]) * invLambda;
      }
    }

    return Math.sqrt(gradNormSum / nTargets);
  }

  getState(): RLSState {
    return this.state;
  }

  setState(state: RLSState): void {
    this.state = state;
  }

  reset(delta: number): void {
    const zDim = this.state.zDim;
    this.state.P.fill(0);
    const invDelta = 1.0 / Math.max(delta, 1e-10);
    for (let i = 0; i < zDim; i++) {
      this.state.P[i * zDim + i] = invDelta;
    }
  }

  serialize(): object {
    return {
      state: this.state.serialize(),
      lambda: this.lambda,
      l2Lambda: this.l2Lambda,
      epsilon: this.epsilon,
    };
  }

  static deserialize(data: any): RLSOptimizer {
    const zDim = data.state.zDim;
    const opt = new RLSOptimizer(
      zDim,
      data.lambda,
      1.0,
      data.l2Lambda,
      data.epsilon,
    );
    opt.state = RLSState.deserialize(data.state);
    return opt;
  }
}

interface ReadoutConfig {
  reservoirSize: number;
  nFeatures: number;
  nTargets: number;
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
}

class LinearReadout {
  private config: ReadoutConfig;
  private zDim: number;
  private Wout: Float64Array;
  private z: Float64Array;
  private output: Float64Array;

  constructor(
    config: ReadoutConfig,
    weightInitScale: number,
    rng: RandomGenerator,
  ) {
    this.config = { ...config };
    this.zDim = config.reservoirSize;
    if (config.useInputInReadout) {
      this.zDim += config.nFeatures;
    }
    if (config.useBiasInReadout) {
      this.zDim += 1;
    }
    this.Wout = new Float64Array(config.nTargets * this.zDim);
    for (let i = 0; i < this.Wout.length; i++) {
      this.Wout[i] = rng.normal() * weightInitScale;
    }
    this.z = new Float64Array(this.zDim);
    this.output = new Float64Array(config.nTargets);
  }

  buildExtendedState(
    reservoirState: Float64Array,
    stateOffset: number,
    input: Float64Array,
    inputOffset: number,
    output: Float64Array,
    outputOffset: number,
  ): void {
    const { reservoirSize, nFeatures, useInputInReadout, useBiasInReadout } =
      this.config;
    let idx = outputOffset;
    for (let i = 0; i < reservoirSize; i++) {
      output[idx++] = reservoirState[stateOffset + i];
    }
    if (useInputInReadout) {
      for (let i = 0; i < nFeatures; i++) {
        output[idx++] = input[inputOffset + i];
      }
    }
    if (useBiasInReadout) {
      output[idx] = 1.0;
    }
  }

  forward(
    z: Float64Array,
    zOffset: number,
    y: Float64Array,
    yOffset: number,
  ): void {
    TensorOps.matVec(
      this.Wout,
      0,
      this.config.nTargets,
      this.zDim,
      z,
      zOffset,
      y,
      yOffset,
    );
  }

  getZDim(): number {
    return this.zDim;
  }

  getWout(): Float64Array {
    return this.Wout;
  }

  getZ(): Float64Array {
    return this.z;
  }

  getOutput(): Float64Array {
    return this.output;
  }

  serialize(): object {
    return {
      config: this.config,
      zDim: this.zDim,
      Wout: Array.from(this.Wout),
    };
  }

  static deserialize(data: any): LinearReadout {
    const rng = new RandomGenerator(42);
    const readout = new LinearReadout(data.config, 0, rng);
    readout.Wout = new Float64Array(data.Wout);
    return readout;
  }
}

class SerializationHelper {
  static serialize(obj: any): string {
    return JSON.stringify(obj);
  }

  static deserialize(str: string): any {
    return JSON.parse(str);
  }
}

/**
 * ESNRegression: Echo State Network for Multivariate Autoregressive Regression
 * with RLS Online Learning and Welford Normalization
 *
 * @example
 * const model = new ESNRegression({ reservoirSize: 256 });
 * const result = model.fitOnline({ coordinates: [[1,2,3], [2,3,4], [3,4,5]] });
 * const prediction = model.predict(5);
 */
export class ESNRegression {
  private config: ESNRegressionConfig;
  private initialized: boolean = false;
  private nFeatures: number = 0;
  private sampleCount: number = 0;
  private reservoir: ESNReservoir | null = null;
  private readout: LinearReadout | null = null;
  private rlsOptimizer: RLSOptimizer | null = null;
  private normalizer: WelfordNormalizer | null = null;
  private residualTracker: ResidualStatsTracker | null = null;
  private outlierDownweighter: OutlierDownweighter;
  private metricsAccumulator: MetricsAccumulator;
  private latestCoordinates: Float64Array | null = null;
  private normalizedInput: Float64Array | null = null;
  private normalizedTarget: Float64Array | null = null;
  private scratchReservoirState: Float64Array | null = null;
  private scratchInput: Float64Array | null = null;
  private scratchZ: Float64Array | null = null;
  private scratchOutput: Float64Array | null = null;
  private tempTarget: Float64Array | null = null;
  private fitResult: FitResult = {
    samplesProcessed: 0,
    averageLoss: 0,
    gradientNorm: 0,
    driftDetected: false,
    sampleWeight: 1,
  };

  constructor(config?: Partial<ESNRegressionConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.outlierDownweighter = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );
    this.metricsAccumulator = new MetricsAccumulator();
  }

  private initializeModel(nFeatures: number): void {
    this.nFeatures = nFeatures;
    const rng = new RandomGenerator(this.config.seed);

    this.reservoir = new ESNReservoir({
      reservoirSize: this.config.reservoirSize,
      nFeatures: nFeatures,
      spectralRadius: this.config.spectralRadius,
      leakRate: this.config.leakRate,
      inputScale: this.config.inputScale,
      biasScale: this.config.biasScale,
      reservoirSparsity: this.config.reservoirSparsity,
      inputSparsity: this.config.inputSparsity,
      activation: this.config.activation,
      seed: this.config.seed,
      epsilon: this.config.epsilon,
    });

    this.readout = new LinearReadout(
      {
        reservoirSize: this.config.reservoirSize,
        nFeatures: nFeatures,
        nTargets: nFeatures,
        useInputInReadout: this.config.useInputInReadout,
        useBiasInReadout: this.config.useBiasInReadout,
      },
      this.config.weightInitScale,
      rng,
    );

    this.rlsOptimizer = new RLSOptimizer(
      this.readout.getZDim(),
      this.config.rlsLambda,
      this.config.rlsDelta,
      this.config.l2Lambda,
      this.config.epsilon,
    );

    this.normalizer = new WelfordNormalizer(
      nFeatures,
      this.config.normalizationEpsilon,
      this.config.normalizationWarmup,
    );

    this.residualTracker = new ResidualStatsTracker(nFeatures);

    this.latestCoordinates = new Float64Array(nFeatures);
    this.normalizedInput = new Float64Array(nFeatures);
    this.normalizedTarget = new Float64Array(nFeatures);
    this.scratchReservoirState = new Float64Array(this.config.reservoirSize);
    this.scratchInput = new Float64Array(nFeatures);
    this.scratchZ = new Float64Array(this.readout.getZDim());
    this.scratchOutput = new Float64Array(nFeatures);
    this.tempTarget = new Float64Array(nFeatures);

    this.initialized = true;
  }

  /**
   * Fit model on a batch of coordinate sequences (online learning)
   *
   * @param params Object containing coordinates array
   * @returns FitResult with training metrics
   *
   * @example
   * const result = model.fitOnline({ coordinates: [[1,2,3], [2,3,4], [3,4,5]] });
   */
  fitOnline(params: { coordinates: number[][] }): FitResult {
    const { coordinates } = params;

    if (!coordinates || !Array.isArray(coordinates)) {
      throw new Error("fitOnline: coordinates must be a non-empty array");
    }
    if (coordinates.length < 2) {
      throw new Error(
        "fitOnline: coordinates must have at least 2 rows to form input-target pairs",
      );
    }

    const firstRow = coordinates[0];
    if (!Array.isArray(firstRow) || firstRow.length === 0) {
      throw new Error("fitOnline: each row must be a non-empty array");
    }

    const nFeatures = firstRow.length;

    for (let i = 1; i < coordinates.length; i++) {
      if (
        !Array.isArray(coordinates[i]) ||
        coordinates[i].length !== nFeatures
      ) {
        throw new Error(`fitOnline: row ${i} has inconsistent dimension`);
      }
    }

    if (!this.initialized) {
      this.initializeModel(nFeatures);
    } else if (nFeatures !== this.nFeatures) {
      throw new Error(
        `fitOnline: feature dimension ${nFeatures} does not match initialized dimension ${this.nFeatures}`,
      );
    }

    this.metricsAccumulator.reset();

    const n = coordinates.length;

    for (let i = 0; i < n - 1; i++) {
      this.processSingleStep(coordinates[i], coordinates[i + 1]);
    }

    for (let j = 0; j < nFeatures; j++) {
      this.scratchInput![j] = coordinates[n - 1][j];
    }
    this.normalizer!.update(this.scratchInput!, 0);

    for (let j = 0; j < nFeatures; j++) {
      this.latestCoordinates![j] = coordinates[n - 1][j];
    }

    this.sampleCount += n - 1;

    this.fitResult.samplesProcessed = n - 1;
    this.fitResult.averageLoss = this.metricsAccumulator.getAverageLoss();
    this.fitResult.gradientNorm = this.metricsAccumulator.getGradientNorm();
    this.fitResult.driftDetected = false;
    this.fitResult.sampleWeight = this.metricsAccumulator.getSampleWeight();

    return this.fitResult;
  }

  private processSingleStep(input: number[], target: number[]): void {
    const nFeatures = this.nFeatures;

    for (let j = 0; j < nFeatures; j++) {
      this.scratchInput![j] = input[j];
    }

    this.normalizer!.normalize(this.scratchInput!, 0, this.normalizedInput!, 0);
    this.normalizer!.update(this.scratchInput!, 0);

    this.reservoir!.update(this.normalizedInput!, 0);

    this.readout!.buildExtendedState(
      this.reservoir!.getState(),
      0,
      this.normalizedInput!,
      0,
      this.scratchZ!,
      0,
    );

    this.readout!.forward(this.scratchZ!, 0, this.scratchOutput!, 0);

    for (let j = 0; j < nFeatures; j++) {
      this.tempTarget![j] = target[j];
    }
    this.normalizer!.normalize(this.tempTarget!, 0, this.normalizedTarget!, 0);

    const loss = LossFunction.mse(
      this.scratchOutput!,
      0,
      this.normalizedTarget!,
      0,
      nFeatures,
    );
    this.metricsAccumulator.addLoss(loss);

    const residualStds = this.residualTracker!.getStds();
    let sampleWeight = 1.0;

    if (this.residualTracker!.getCount() >= 5) {
      sampleWeight = this.outlierDownweighter.computeWeight(
        this.scratchOutput!,
        0,
        this.normalizedTarget!,
        0,
        residualStds,
        nFeatures,
        this.config.epsilon,
      );
    }
    this.metricsAccumulator.setSampleWeight(sampleWeight);

    const gradNorm = this.rlsOptimizer!.update(
      this.scratchZ!,
      0,
      this.scratchOutput!,
      0,
      this.normalizedTarget!,
      0,
      this.readout!.getWout(),
      0,
      nFeatures,
      sampleWeight,
    );
    this.metricsAccumulator.setGradientNorm(gradNorm);

    this.residualTracker!.update(
      this.scratchOutput!,
      0,
      this.normalizedTarget!,
      0,
    );
  }

  /**
   * Predict future steps using autoregressive roll-forward
   *
   * @param futureSteps Number of steps to predict (must be >= 1)
   * @returns PredictionResult containing predictions and uncertainty bounds
   *
   * @example
   * const result = model.predict(5);
   * console.log(result.predictions);
   */
  predict(futureSteps: number): PredictionResult {
    if (!this.initialized || !this.latestCoordinates) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }

    if (!Number.isInteger(futureSteps) || futureSteps < 1) {
      throw new Error("predict: futureSteps must be an integer >= 1");
    }

    const nFeatures = this.nFeatures;

    const predictions: number[][] = new Array(futureSteps);
    const lowerBounds: number[][] = new Array(futureSteps);
    const upperBounds: number[][] = new Array(futureSteps);

    for (let k = 0; k < futureSteps; k++) {
      predictions[k] = new Array(nFeatures);
      lowerBounds[k] = new Array(nFeatures);
      upperBounds[k] = new Array(nFeatures);
    }

    this.reservoir!.copyStateTo(this.scratchReservoirState!, 0);
    TensorOps.copy(this.latestCoordinates, 0, this.scratchInput!, 0, nFeatures);

    const residualStds = this.residualTracker!.getStds();
    const uncertaintyMultiplier = this.config.uncertaintyMultiplier;
    const normStds = this.normalizer!.getCachedStds();

    for (let step = 0; step < futureSteps; step++) {
      this.normalizer!.normalize(
        this.scratchInput!,
        0,
        this.normalizedInput!,
        0,
      );

      this.reservoir!.updateScratch(
        this.normalizedInput!,
        0,
        this.scratchReservoirState!,
        0,
      );

      this.readout!.buildExtendedState(
        this.scratchReservoirState!,
        0,
        this.normalizedInput!,
        0,
        this.scratchZ!,
        0,
      );

      this.readout!.forward(this.scratchZ!, 0, this.scratchOutput!, 0);
      this.normalizer!.denormalize(
        this.scratchOutput!,
        0,
        this.scratchInput!,
        0,
      );

      const horizonFactor = Math.sqrt(step + 1);

      for (let j = 0; j < nFeatures; j++) {
        const pred = this.scratchInput![j];
        predictions[step][j] = pred;
        const sigma = residualStds[j] * normStds[j] * horizonFactor;
        lowerBounds[step][j] = pred - uncertaintyMultiplier * sigma;
        upperBounds[step][j] = pred + uncertaintyMultiplier * sigma;
      }
    }

    const confidence = this.computeConfidence();

    return {
      predictions,
      lowerBounds,
      upperBounds,
      confidence,
    };
  }

  private computeConfidence(): number {
    if (!this.residualTracker || this.sampleCount < 2) {
      return 0;
    }

    const residualStds = this.residualTracker.getStds();
    let meanStd = 0;
    for (let i = 0; i < this.nFeatures; i++) {
      meanStd += residualStds[i];
    }
    meanStd /= this.nFeatures;

    const confidence = 1.0 / (1.0 + meanStd);
    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Get model summary statistics
   *
   * @returns ModelSummary object
   */
  getModelSummary(): ModelSummary {
    const reservoirSize = this.config.reservoirSize;
    const nFeatures = this.nFeatures || 0;
    const nTargets = nFeatures;

    let zDim = reservoirSize;
    if (this.config.useInputInReadout) zDim += nFeatures;
    if (this.config.useBiasInReadout) zDim += 1;

    const woutParams = nTargets * zDim;
    const winParams = reservoirSize * nFeatures;
    const wParams = reservoirSize * reservoirSize;
    const biasParams = reservoirSize;

    const sr = this.config.spectralRadius;
    const effectiveMemory = sr < 1.0 && sr > 0
      ? Math.ceil(-Math.log(0.01) / -Math.log(sr))
      : 1000;

    return {
      totalParameters: woutParams + winParams + wParams + biasParams,
      receptiveField: effectiveMemory,
      spectralRadius: this.config.spectralRadius,
      reservoirSize: reservoirSize,
      nFeatures: nFeatures,
      nTargets: nTargets,
      sampleCount: this.sampleCount,
    };
  }

  /**
   * Get all weight matrices
   *
   * @returns WeightInfo object
   */
  getWeights(): WeightInfo {
    const weights: Array<{ name: string; shape: number[]; values: number[] }> =
      [];

    if (this.readout) {
      const zDim = this.readout.getZDim();
      weights.push({
        name: "Wout",
        shape: [this.nFeatures, zDim],
        values: Array.from(this.readout.getWout()),
      });
    }

    if (this.reservoir) {
      weights.push({
        name: "Win",
        shape: [this.config.reservoirSize, this.nFeatures],
        values: Array.from(this.reservoir.getWin()),
      });

      weights.push({
        name: "W",
        shape: [this.config.reservoirSize, this.config.reservoirSize],
        values: Array.from(this.reservoir.getW()),
      });

      weights.push({
        name: "bias",
        shape: [this.config.reservoirSize],
        values: Array.from(this.reservoir.getBias()),
      });
    }

    return { weights };
  }

  /**
   * Get normalization statistics
   *
   * @returns NormalizationStats object
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.normalizer) {
      return {
        means: [],
        stds: [],
        count: 0,
        isActive: false,
      };
    }

    return {
      means: this.normalizer.getMeans(),
      stds: this.normalizer.getStds(),
      count: this.normalizer.getCount(),
      isActive: this.normalizer.isActive(),
    };
  }

  /**
   * Reset model to initial state
   */
  reset(): void {
    this.initialized = false;
    this.nFeatures = 0;
    this.sampleCount = 0;
    this.reservoir = null;
    this.readout = null;
    this.rlsOptimizer = null;
    this.normalizer = null;
    this.residualTracker = null;
    this.latestCoordinates = null;
    this.normalizedInput = null;
    this.normalizedTarget = null;
    this.scratchReservoirState = null;
    this.scratchInput = null;
    this.scratchZ = null;
    this.scratchOutput = null;
    this.tempTarget = null;
    this.metricsAccumulator.reset();
    this.fitResult = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1,
    };
  }

  /**
   * Save model state to JSON string
   *
   * @returns JSON string representation of model state
   */
  save(): string {
    const state = {
      config: this.config,
      initialized: this.initialized,
      nFeatures: this.nFeatures,
      sampleCount: this.sampleCount,
      reservoir: this.reservoir ? this.reservoir.serialize() : null,
      readout: this.readout ? this.readout.serialize() : null,
      rlsOptimizer: this.rlsOptimizer ? this.rlsOptimizer.serialize() : null,
      normalizer: this.normalizer ? this.normalizer.serialize() : null,
      residualTracker: this.residualTracker
        ? this.residualTracker.serialize()
        : null,
      latestCoordinates: this.latestCoordinates
        ? Array.from(this.latestCoordinates)
        : null,
    };
    return SerializationHelper.serialize(state);
  }

  /**
   * Load model state from JSON string
   *
   * @param str JSON string representation of model state
   */
  load(str: string): void {
    const state = SerializationHelper.deserialize(str);

    this.config = { ...DEFAULT_CONFIG, ...state.config };
    this.initialized = state.initialized;
    this.nFeatures = state.nFeatures;
    this.sampleCount = state.sampleCount;

    if (state.reservoir) {
      this.reservoir = ESNReservoir.deserialize(state.reservoir);
    }

    if (state.readout) {
      this.readout = LinearReadout.deserialize(state.readout);
    }

    if (state.rlsOptimizer) {
      this.rlsOptimizer = RLSOptimizer.deserialize(state.rlsOptimizer);
    }

    if (state.normalizer) {
      this.normalizer = WelfordNormalizer.deserialize(state.normalizer);
    }

    if (state.residualTracker) {
      this.residualTracker = ResidualStatsTracker.deserialize(
        state.residualTracker,
      );
    }

    if (state.latestCoordinates) {
      this.latestCoordinates = new Float64Array(state.latestCoordinates);
    }

    if (this.initialized && this.nFeatures > 0) {
      this.normalizedInput = new Float64Array(this.nFeatures);
      this.normalizedTarget = new Float64Array(this.nFeatures);
      this.scratchReservoirState = new Float64Array(this.config.reservoirSize);
      this.scratchInput = new Float64Array(this.nFeatures);
      this.scratchZ = new Float64Array(this.readout!.getZDim());
      this.scratchOutput = new Float64Array(this.nFeatures);
      this.tempTarget = new Float64Array(this.nFeatures);
    }

    this.outlierDownweighter = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );
  }
}

export default ESNRegression;
