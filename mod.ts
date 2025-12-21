// ============================================
// INTERFACES
// ============================================

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

// ============================================
// DEFAULT CONFIGURATION
// ============================================

const DEFAULT_CONFIG: ESNRegressionConfig = Object.freeze({
  reservoirSize: 256,
  spectralRadius: 0.9,
  leakRate: 0.3,
  inputScale: 1.0,
  biasScale: 0.1,
  reservoirSparsity: 0.9,
  inputSparsity: 0.0,
  activation: "tanh" as const,
  useInputInReadout: true,
  useBiasInReadout: true,
  readoutTraining: "rls" as const,
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
});

// ============================================
// TENSOR SHAPE
// ============================================

class TensorShape {
  readonly dims: readonly number[];
  readonly strides: readonly number[];
  readonly size: number;

  constructor(dims: number[]) {
    this.dims = Object.freeze([...dims]);
    const strides: number[] = new Array(dims.length);
    let size = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      strides[i] = size;
      size *= dims[i];
    }
    this.strides = Object.freeze(strides);
    this.size = size;
  }
}

// ============================================
// TENSOR VIEW
// ============================================

class TensorView {
  private readonly _data: Float64Array;
  private readonly _offset: number;
  private readonly _length: number;

  constructor(data: Float64Array, offset: number = 0, length?: number) {
    this._data = data;
    this._offset = offset;
    this._length = length ?? (data.length - offset);
  }

  get(i: number): number {
    return this._data[this._offset + i];
  }

  set(i: number, value: number): void {
    this._data[this._offset + i] = value;
  }

  fill(value: number): void {
    const end = this._offset + this._length;
    for (let i = this._offset; i < end; i++) {
      this._data[i] = value;
    }
  }

  get length(): number {
    return this._length;
  }

  get data(): Float64Array {
    return this._data;
  }

  get offset(): number {
    return this._offset;
  }
}

// ============================================
// BUFFER POOL
// ============================================

class BufferPool {
  private readonly _pools: Map<number, Float64Array[]> = new Map();
  private readonly _maxPoolSize: number;

  constructor(maxPoolSize: number = 32) {
    this._maxPoolSize = maxPoolSize;
  }

  acquire(size: number): Float64Array {
    const pool = this._pools.get(size);
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    return new Float64Array(size);
  }

  release(buffer: Float64Array): void {
    const size = buffer.length;
    let pool = this._pools.get(size);
    if (!pool) {
      pool = [];
      this._pools.set(size, pool);
    }
    if (pool.length < this._maxPoolSize) {
      buffer.fill(0);
      pool.push(buffer);
    }
  }

  clear(): void {
    this._pools.clear();
  }
}

// ============================================
// TENSOR ARENA
// ============================================

class TensorArena {
  private _buffer: Float64Array;
  private readonly _allocations: Map<
    string,
    { offset: number; length: number }
  > = new Map();
  private _nextOffset: number = 0;

  constructor(totalSize: number) {
    this._buffer = new Float64Array(totalSize);
  }

  allocate(name: string, length: number): TensorView {
    const existing = this._allocations.get(name);
    if (existing) {
      return new TensorView(this._buffer, existing.offset, existing.length);
    }

    if (this._nextOffset + length > this._buffer.length) {
      throw new Error(
        `TensorArena overflow: cannot allocate ${length} elements for '${name}'`,
      );
    }

    const offset = this._nextOffset;
    this._allocations.set(name, { offset, length });
    this._nextOffset += length;
    return new TensorView(this._buffer, offset, length);
  }

  get(name: string): TensorView | null {
    const alloc = this._allocations.get(name);
    if (!alloc) return null;
    return new TensorView(this._buffer, alloc.offset, alloc.length);
  }

  reset(): void {
    this._buffer.fill(0);
    this._allocations.clear();
    this._nextOffset = 0;
  }

  get totalSize(): number {
    return this._buffer.length;
  }

  get usedSize(): number {
    return this._nextOffset;
  }
}

// ============================================
// TENSOR OPERATIONS
// ============================================

class TensorOps {
  static matVec(
    A: Float64Array,
    x: Float64Array,
    y: Float64Array,
    rows: number,
    cols: number,
    yOffset: number = 0,
    xOffset: number = 0,
  ): void {
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      const rowStart = i * cols;
      for (let j = 0; j < cols; j++) {
        sum += A[rowStart + j] * x[xOffset + j];
      }
      y[yOffset + i] = sum;
    }
  }

  static dot(
    x: Float64Array,
    y: Float64Array,
    length: number,
    xOffset: number = 0,
    yOffset: number = 0,
  ): number {
    let sum = 0;
    for (let i = 0; i < length; i++) {
      sum += x[xOffset + i] * y[yOffset + i];
    }
    return sum;
  }

  static axpy(
    alpha: number,
    x: Float64Array,
    y: Float64Array,
    length: number,
    xOffset: number = 0,
    yOffset: number = 0,
  ): void {
    for (let i = 0; i < length; i++) {
      y[yOffset + i] += alpha * x[xOffset + i];
    }
  }

  static copy(
    src: Float64Array,
    dst: Float64Array,
    length: number,
    srcOffset: number = 0,
    dstOffset: number = 0,
  ): void {
    for (let i = 0; i < length; i++) {
      dst[dstOffset + i] = src[srcOffset + i];
    }
  }

  static norm(x: Float64Array, length: number, offset: number = 0): number {
    let sum = 0;
    for (let i = 0; i < length; i++) {
      const val = x[offset + i];
      sum += val * val;
    }
    return Math.sqrt(sum);
  }

  static fill(
    x: Float64Array,
    value: number,
    length: number,
    offset: number = 0,
  ): void {
    const end = offset + length;
    for (let i = offset; i < end; i++) {
      x[i] = value;
    }
  }

  static sparseMatVecByIndices(
    values: Float64Array,
    rowIndices: Int32Array,
    colIndices: Int32Array,
    x: Float64Array,
    y: Float64Array,
    yLength: number,
    xOffset: number = 0,
    yOffset: number = 0,
    clearOutput: boolean = true,
  ): void {
    if (clearOutput) {
      for (let i = 0; i < yLength; i++) {
        y[yOffset + i] = 0;
      }
    }
    const nnz = values.length;
    for (let k = 0; k < nnz; k++) {
      y[yOffset + rowIndices[k]] += values[k] * x[xOffset + colIndices[k]];
    }
  }

  static clampInPlace(
    x: Float64Array,
    min: number,
    max: number,
    length: number,
    offset: number = 0,
  ): void {
    const end = offset + length;
    for (let i = offset; i < end; i++) {
      const v = x[i];
      if (v < min) x[i] = min;
      else if (v > max) x[i] = max;
    }
  }

  static hasNonFinite(
    x: Float64Array,
    length: number,
    offset: number = 0,
  ): boolean {
    const end = offset + length;
    for (let i = offset; i < end; i++) {
      if (!Number.isFinite(x[i])) return true;
    }
    return false;
  }

  static sanitize(
    x: Float64Array,
    fallback: number,
    length: number,
    offset: number = 0,
  ): void {
    const end = offset + length;
    for (let i = offset; i < end; i++) {
      if (!Number.isFinite(x[i])) {
        x[i] = fallback;
      }
    }
  }
}

// ============================================
// RANDOM GENERATOR (Xorshift32)
// ============================================

class RandomGenerator {
  private _state: number;

  constructor(seed: number = 42) {
    this._state = seed >>> 0;
    if (this._state === 0) this._state = 1;
  }

  private _next(): number {
    let x = this._state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this._state = x >>> 0;
    return this._state;
  }

  uniform(): number {
    return this._next() / 4294967296;
  }

  normal(): number {
    const u1 = this.uniform() || 1e-10;
    const u2 = this.uniform();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  range(min: number, max: number): number {
    return min + this.uniform() * (max - min);
  }

  getState(): number {
    return this._state;
  }

  setState(state: number): void {
    this._state = state >>> 0;
    if (this._state === 0) this._state = 1;
  }
}

// ============================================
// ACTIVATION OPERATIONS
// ============================================

class ActivationOps {
  static tanh(x: number): number {
    if (x > 20) return 1;
    if (x < -20) return -1;
    return Math.tanh(x);
  }

  static relu(x: number): number {
    return x > 0 ? x : 0;
  }

  static apply(
    type: "tanh" | "relu",
    x: Float64Array,
    length: number,
    offset: number = 0,
  ): void {
    const end = offset + length;
    if (type === "tanh") {
      for (let i = offset; i < end; i++) {
        const v = x[i];
        if (v > 20) x[i] = 1;
        else if (v < -20) x[i] = -1;
        else x[i] = Math.tanh(v);
      }
    } else {
      for (let i = offset; i < end; i++) {
        if (x[i] < 0) x[i] = 0;
      }
    }
  }
}

// ============================================
// WELFORD ACCUMULATOR
// ============================================

class WelfordAccumulator {
  private _count: number = 0;
  private _mean: number = 0;
  private _m2: number = 0;

  update(value: number): void {
    this._count++;
    const delta = value - this._mean;
    this._mean += delta / this._count;
    const delta2 = value - this._mean;
    this._m2 += delta * delta2;
  }

  get count(): number {
    return this._count;
  }

  get mean(): number {
    return this._mean;
  }

  get variance(): number {
    return this._count > 1 ? this._m2 / (this._count - 1) : 0;
  }

  get std(): number {
    return Math.sqrt(this.variance);
  }

  reset(): void {
    this._count = 0;
    this._mean = 0;
    this._m2 = 0;
  }

  serialize(): { count: number; mean: number; m2: number } {
    return { count: this._count, mean: this._mean, m2: this._m2 };
  }

  static deserialize(
    data: { count: number; mean: number; m2: number },
  ): WelfordAccumulator {
    const acc = new WelfordAccumulator();
    acc._count = data.count;
    acc._mean = data.mean;
    acc._m2 = data.m2;
    return acc;
  }
}

// ============================================
// WELFORD NORMALIZER
// ============================================

class WelfordNormalizer {
  private _accumulators: WelfordAccumulator[];
  private readonly _nFeatures: number;
  private readonly _epsilon: number;
  private readonly _warmup: number;
  private _mins: Float64Array;
  private _maxs: Float64Array;

  constructor(nFeatures: number, epsilon: number = 1e-8, warmup: number = 10) {
    this._nFeatures = nFeatures;
    this._epsilon = epsilon;
    this._warmup = warmup;
    this._accumulators = [];
    for (let i = 0; i < nFeatures; i++) {
      this._accumulators.push(new WelfordAccumulator());
    }
    this._mins = new Float64Array(nFeatures);
    this._maxs = new Float64Array(nFeatures);
    this._mins.fill(Infinity);
    this._maxs.fill(-Infinity);
  }

  update(input: Float64Array, offset: number = 0): void {
    for (let i = 0; i < this._nFeatures; i++) {
      const val = input[offset + i];
      this._accumulators[i].update(val);
      if (val < this._mins[i]) this._mins[i] = val;
      if (val > this._maxs[i]) this._maxs[i] = val;
    }
  }

  get isActive(): boolean {
    return this._accumulators[0].count >= this._warmup;
  }

  get count(): number {
    return this._accumulators[0].count;
  }

  normalize(
    input: Float64Array,
    output: Float64Array,
    inputOffset: number = 0,
    outputOffset: number = 0,
  ): void {
    if (!this.isActive) {
      TensorOps.copy(input, output, this._nFeatures, inputOffset, outputOffset);
      return;
    }

    for (let i = 0; i < this._nFeatures; i++) {
      const mean = this._accumulators[i].mean;
      const std = this._accumulators[i].std;
      const denom = std > this._epsilon ? std : 1;
      let normalized = (input[inputOffset + i] - mean) / denom;
      if (normalized > 10) normalized = 10;
      else if (normalized < -10) normalized = -10;
      output[outputOffset + i] = normalized;
    }
  }

  denormalize(
    input: Float64Array,
    output: Float64Array,
    inputOffset: number = 0,
    outputOffset: number = 0,
  ): void {
    if (!this.isActive) {
      TensorOps.copy(input, output, this._nFeatures, inputOffset, outputOffset);
      return;
    }

    for (let i = 0; i < this._nFeatures; i++) {
      const mean = this._accumulators[i].mean;
      const std = this._accumulators[i].std;
      const denom = std > this._epsilon ? std : 1;
      output[outputOffset + i] = input[inputOffset + i] * denom + mean;
    }
  }

  getMeans(): number[] {
    return this._accumulators.map((a) => a.mean);
  }

  getStds(): number[] {
    return this._accumulators.map((a) => Math.max(a.std, this._epsilon));
  }

  getMins(): number[] {
    return Array.from(this._mins);
  }

  getMaxs(): number[] {
    return Array.from(this._maxs);
  }

  reset(): void {
    for (const acc of this._accumulators) {
      acc.reset();
    }
    this._mins.fill(Infinity);
    this._maxs.fill(-Infinity);
  }

  serialize(): object {
    return {
      nFeatures: this._nFeatures,
      epsilon: this._epsilon,
      warmup: this._warmup,
      accumulators: this._accumulators.map((a) => a.serialize()),
      mins: Array.from(this._mins),
      maxs: Array.from(this._maxs),
    };
  }

  static deserialize(data: any): WelfordNormalizer {
    const norm = new WelfordNormalizer(
      data.nFeatures,
      data.epsilon,
      data.warmup,
    );
    norm._accumulators = data.accumulators.map((a: any) =>
      WelfordAccumulator.deserialize(a)
    );
    norm._mins = new Float64Array(data.mins);
    norm._maxs = new Float64Array(data.maxs);
    return norm;
  }
}

// ============================================
// RESIDUAL STATS TRACKER
// ============================================

class ResidualStatsTracker {
  private _accumulators: WelfordAccumulator[];
  private readonly _nTargets: number;

  constructor(nTargets: number) {
    this._nTargets = nTargets;
    this._accumulators = [];
    for (let i = 0; i < nTargets; i++) {
      this._accumulators.push(new WelfordAccumulator());
    }
  }

  update(residuals: Float64Array, offset: number = 0): void {
    for (let i = 0; i < this._nTargets; i++) {
      this._accumulators[i].update(residuals[offset + i]);
    }
  }

  getStds(): Float64Array {
    const stds = new Float64Array(this._nTargets);
    for (let i = 0; i < this._nTargets; i++) {
      stds[i] = Math.max(this._accumulators[i].std, 1e-8);
    }
    return stds;
  }

  serialize(): object {
    return {
      nTargets: this._nTargets,
      accumulators: this._accumulators.map((a) => a.serialize()),
    };
  }

  static deserialize(data: any): ResidualStatsTracker {
    const tracker = new ResidualStatsTracker(data.nTargets);
    tracker._accumulators = data.accumulators.map((a: any) =>
      WelfordAccumulator.deserialize(a)
    );
    return tracker;
  }
}

// ============================================
// OUTLIER DOWNWEIGHTER
// ============================================

class OutlierDownweighter {
  private readonly _threshold: number;
  private readonly _minWeight: number;

  constructor(threshold: number = 3.0, minWeight: number = 0.1) {
    this._threshold = threshold;
    this._minWeight = minWeight;
  }

  computeWeight(
    residual: Float64Array,
    stds: Float64Array,
    epsilon: number,
  ): number {
    const nTargets = residual.length;
    let maxZScore = 0;

    for (let i = 0; i < nTargets; i++) {
      const absResidual = Math.abs(residual[i]);
      const std = stds[i] > epsilon ? stds[i] : epsilon;
      const zScore = absResidual / std;
      if (zScore > maxZScore) maxZScore = zScore;
    }

    if (maxZScore <= this._threshold) {
      return 1.0;
    }

    const excess = maxZScore - this._threshold;
    const weight = Math.exp(-0.5 * excess * excess);
    return Math.max(weight, this._minWeight);
  }

  serialize(): object {
    return { threshold: this._threshold, minWeight: this._minWeight };
  }

  static deserialize(data: any): OutlierDownweighter {
    return new OutlierDownweighter(data.threshold, data.minWeight);
  }
}

// ============================================
// LOSS FUNCTION
// ============================================

class LossFunction {
  static mse(
    pred: Float64Array,
    target: Float64Array,
    length: number,
    predOffset: number = 0,
    targetOffset: number = 0,
  ): number {
    let sum = 0;
    for (let i = 0; i < length; i++) {
      const diff = pred[predOffset + i] - target[targetOffset + i];
      sum += diff * diff;
    }
    return sum / length;
  }
}

// ============================================
// METRICS ACCUMULATOR
// ============================================

class MetricsAccumulator {
  private _lossSum: number = 0;
  private _count: number = 0;
  private _lastGradNorm: number = 0;

  addLoss(loss: number): void {
    this._lossSum += loss;
    this._count++;
  }

  setGradientNorm(norm: number): void {
    this._lastGradNorm = norm;
  }

  get averageLoss(): number {
    return this._count > 0 ? this._lossSum / this._count : 0;
  }

  get gradientNorm(): number {
    return this._lastGradNorm;
  }

  get count(): number {
    return this._count;
  }

  reset(): void {
    this._lossSum = 0;
    this._count = 0;
    this._lastGradNorm = 0;
  }
}

// ============================================
// SPARSE INDICES
// ============================================

class SparseIndices {
  readonly rowIndices: Int32Array;
  readonly colIndices: Int32Array;
  readonly nnz: number;

  constructor(rowIndices: Int32Array, colIndices: Int32Array) {
    if (rowIndices.length !== colIndices.length) {
      throw new Error("Row and column index arrays must have same length");
    }
    this.rowIndices = rowIndices;
    this.colIndices = colIndices;
    this.nnz = rowIndices.length;
  }

  static createRandom(
    rows: number,
    cols: number,
    sparsity: number,
    rng: RandomGenerator,
  ): SparseIndices {
    const totalElements = rows * cols;
    const nnz = Math.max(1, Math.floor(totalElements * (1 - sparsity)));

    const indices = new Set<number>();
    while (indices.size < nnz) {
      const idx = Math.floor(rng.uniform() * totalElements);
      indices.add(idx);
    }

    const rowIndices = new Int32Array(nnz);
    const colIndices = new Int32Array(nnz);
    let k = 0;
    for (const idx of indices) {
      rowIndices[k] = Math.floor(idx / cols);
      colIndices[k] = idx % cols;
      k++;
    }

    return new SparseIndices(rowIndices, colIndices);
  }

  serialize(): { rows: number[]; cols: number[] } {
    return {
      rows: Array.from(this.rowIndices),
      cols: Array.from(this.colIndices),
    };
  }

  static deserialize(data: { rows: number[]; cols: number[] }): SparseIndices {
    return new SparseIndices(
      new Int32Array(data.rows),
      new Int32Array(data.cols),
    );
  }
}

// ============================================
// SPECTRAL RADIUS SCALER
// ============================================

class SpectralRadiusScaler {
  static estimateSpectralRadius(
    values: Float64Array,
    indices: SparseIndices,
    size: number,
    rng: RandomGenerator,
    iterations: number = 50,
  ): number {
    const v = new Float64Array(size);
    const w = new Float64Array(size);

    for (let i = 0; i < size; i++) {
      v[i] = rng.normal();
    }

    let norm = TensorOps.norm(v, size);
    if (norm > 0) {
      for (let i = 0; i < size; i++) v[i] /= norm;
    }

    let eigenvalue = 0;
    for (let iter = 0; iter < iterations; iter++) {
      TensorOps.sparseMatVecByIndices(
        values,
        indices.rowIndices,
        indices.colIndices,
        v,
        w,
        size,
      );

      eigenvalue = TensorOps.dot(w, v, size);

      norm = TensorOps.norm(w, size);
      if (norm < 1e-10) break;

      for (let i = 0; i < size; i++) {
        v[i] = w[i] / norm;
      }
    }

    return Math.abs(eigenvalue);
  }

  static scaleToSpectralRadius(
    values: Float64Array,
    indices: SparseIndices,
    size: number,
    targetRadius: number,
    rng: RandomGenerator,
  ): void {
    const currentRadius = SpectralRadiusScaler.estimateSpectralRadius(
      values,
      indices,
      size,
      rng,
    );
    if (currentRadius > 1e-10) {
      const scale = targetRadius / currentRadius;
      for (let i = 0; i < values.length; i++) {
        values[i] *= scale;
      }
    }
  }
}

// ============================================
// ESN RESERVOIR
// ============================================

class ESNReservoir {
  private readonly _size: number;
  private readonly _inputSize: number;
  private readonly _leakRate: number;
  private readonly _activation: "tanh" | "relu";

  private _WValues: Float64Array;
  private _WIndices: SparseIndices;

  private _WinValues: Float64Array;
  private _WinIndices: SparseIndices | null;
  private _WinDense: Float64Array | null;

  private _bias: Float64Array;
  private _state: Float64Array;

  // Pre-allocated scratch buffers
  private _scratch1: Float64Array;
  private _scratch2: Float64Array;

  constructor(
    size: number,
    inputSize: number,
    spectralRadius: number,
    leakRate: number,
    inputScale: number,
    biasScale: number,
    reservoirSparsity: number,
    inputSparsity: number,
    activation: "tanh" | "relu",
    rng: RandomGenerator,
  ) {
    this._size = size;
    this._inputSize = inputSize;
    this._leakRate = leakRate;
    this._activation = activation;

    // Initialize reservoir weights (sparse)
    this._WIndices = SparseIndices.createRandom(
      size,
      size,
      reservoirSparsity,
      rng,
    );
    this._WValues = new Float64Array(this._WIndices.nnz);
    for (let i = 0; i < this._WIndices.nnz; i++) {
      this._WValues[i] = rng.range(-1, 1);
    }

    SpectralRadiusScaler.scaleToSpectralRadius(
      this._WValues,
      this._WIndices,
      size,
      spectralRadius,
      rng,
    );

    // Initialize input weights
    if (inputSparsity > 0) {
      this._WinIndices = SparseIndices.createRandom(
        size,
        inputSize,
        inputSparsity,
        rng,
      );
      this._WinValues = new Float64Array(this._WinIndices.nnz);
      for (let i = 0; i < this._WinIndices.nnz; i++) {
        this._WinValues[i] = rng.range(-inputScale, inputScale);
      }
      this._WinDense = null;
    } else {
      this._WinIndices = null;
      this._WinDense = new Float64Array(size * inputSize);
      for (let i = 0; i < this._WinDense.length; i++) {
        this._WinDense[i] = rng.range(-inputScale, inputScale);
      }
      this._WinValues = new Float64Array(0);
    }

    // Initialize bias
    this._bias = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      this._bias[i] = rng.range(-biasScale, biasScale);
    }

    // Initialize state and scratch buffers
    this._state = new Float64Array(size);
    this._scratch1 = new Float64Array(size);
    this._scratch2 = new Float64Array(size);
  }

  update(input: Float64Array, inputOffset: number = 0): void {
    this._computeNextState(
      input,
      this._state,
      this._scratch1,
      this._scratch2,
      inputOffset,
    );
    TensorOps.copy(this._scratch1, this._state, this._size);
  }

  updateScratch(
    input: Float64Array,
    state: Float64Array,
    scratch: Float64Array,
    inputOffset: number = 0,
  ): void {
    this._computeNextState(input, state, scratch, this._scratch2, inputOffset);
    TensorOps.copy(scratch, state, this._size);
  }

  private _computeNextState(
    input: Float64Array,
    currentState: Float64Array,
    nextState: Float64Array,
    tempBuffer: Float64Array,
    inputOffset: number,
  ): void {
    // nextState = W * currentState
    TensorOps.sparseMatVecByIndices(
      this._WValues,
      this._WIndices.rowIndices,
      this._WIndices.colIndices,
      currentState,
      nextState,
      this._size,
    );

    // Add Win * input
    if (this._WinDense) {
      TensorOps.matVec(
        this._WinDense,
        input,
        tempBuffer,
        this._size,
        this._inputSize,
        0,
        inputOffset,
      );
      TensorOps.axpy(1, tempBuffer, nextState, this._size);
    } else if (this._WinIndices) {
      TensorOps.sparseMatVecByIndices(
        this._WinValues,
        this._WinIndices.rowIndices,
        this._WinIndices.colIndices,
        input,
        nextState,
        this._size,
        inputOffset,
        0,
        false,
      );
    }

    // Add bias
    TensorOps.axpy(1, this._bias, nextState, this._size);

    // Apply activation
    ActivationOps.apply(this._activation, nextState, this._size);

    // Leaky integration
    for (let i = 0; i < this._size; i++) {
      nextState[i] = (1 - this._leakRate) * currentState[i] +
        this._leakRate * nextState[i];
    }

    // Sanitize
    TensorOps.sanitize(nextState, 0, this._size);
  }

  get state(): Float64Array {
    return this._state;
  }

  get size(): number {
    return this._size;
  }

  resetState(): void {
    this._state.fill(0);
  }

  copyState(dest: Float64Array, offset: number = 0): void {
    TensorOps.copy(this._state, dest, this._size, 0, offset);
  }

  serialize(): object {
    return {
      size: this._size,
      inputSize: this._inputSize,
      leakRate: this._leakRate,
      activation: this._activation,
      WValues: Array.from(this._WValues),
      WIndices: this._WIndices.serialize(),
      WinValues: Array.from(this._WinValues),
      WinIndices: this._WinIndices ? this._WinIndices.serialize() : null,
      WinDense: this._WinDense ? Array.from(this._WinDense) : null,
      bias: Array.from(this._bias),
      state: Array.from(this._state),
    };
  }

  static deserialize(data: any): ESNReservoir {
    const reservoir = Object.create(ESNReservoir.prototype);
    reservoir._size = data.size;
    reservoir._inputSize = data.inputSize;
    reservoir._leakRate = data.leakRate;
    reservoir._activation = data.activation;
    reservoir._WValues = new Float64Array(data.WValues);
    reservoir._WIndices = SparseIndices.deserialize(data.WIndices);
    reservoir._WinValues = new Float64Array(data.WinValues);
    reservoir._WinIndices = data.WinIndices
      ? SparseIndices.deserialize(data.WinIndices)
      : null;
    reservoir._WinDense = data.WinDense
      ? new Float64Array(data.WinDense)
      : null;
    reservoir._bias = new Float64Array(data.bias);
    reservoir._state = new Float64Array(data.state);
    reservoir._scratch1 = new Float64Array(data.size);
    reservoir._scratch2 = new Float64Array(data.size);
    return reservoir;
  }

  getWeights(): Array<{ name: string; shape: number[]; values: number[] }> {
    const weights: Array<{ name: string; shape: number[]; values: number[] }> =
      [];

    // Convert sparse W to dense for export
    const WDense = new Float64Array(this._size * this._size);
    for (let k = 0; k < this._WIndices.nnz; k++) {
      const row = this._WIndices.rowIndices[k];
      const col = this._WIndices.colIndices[k];
      WDense[row * this._size + col] = this._WValues[k];
    }
    weights.push({
      name: "W",
      shape: [this._size, this._size],
      values: Array.from(WDense),
    });

    // Input weights
    if (this._WinDense) {
      weights.push({
        name: "Win",
        shape: [this._size, this._inputSize],
        values: Array.from(this._WinDense),
      });
    } else if (this._WinIndices) {
      const WinDense = new Float64Array(this._size * this._inputSize);
      for (let k = 0; k < this._WinIndices.nnz; k++) {
        const row = this._WinIndices.rowIndices[k];
        const col = this._WinIndices.colIndices[k];
        WinDense[row * this._inputSize + col] = this._WinValues[k];
      }
      weights.push({
        name: "Win",
        shape: [this._size, this._inputSize],
        values: Array.from(WinDense),
      });
    }

    weights.push({
      name: "bias",
      shape: [this._size],
      values: Array.from(this._bias),
    });

    return weights;
  }
}

// ============================================
// RLS STATE
// ============================================

class RLSState {
  P: Float64Array;
  readonly extendedSize: number;
  updateCount: number;

  constructor(extendedSize: number, delta: number) {
    this.extendedSize = extendedSize;
    this.P = new Float64Array(extendedSize * extendedSize);
    this.updateCount = 0;

    for (let i = 0; i < extendedSize; i++) {
      this.P[i * extendedSize + i] = delta;
    }
  }

  serialize(): object {
    return {
      extendedSize: this.extendedSize,
      P: Array.from(this.P),
      updateCount: this.updateCount,
    };
  }

  static deserialize(data: any): RLSState {
    const state = new RLSState(data.extendedSize, 1);
    state.P = new Float64Array(data.P);
    state.updateCount = data.updateCount;
    return state;
  }
}

// ============================================
// RLS OPTIMIZER
// ============================================

class RLSOptimizer {
  private readonly _lambda: number;
  private readonly _delta: number;
  private readonly _l2Lambda: number;
  private readonly _gradientClipNorm: number;
  private readonly _epsilon: number;

  // Pre-allocated scratch buffers
  private _Pz: Float64Array;
  private _gain: Float64Array;

  constructor(
    extendedSize: number,
    lambda: number,
    delta: number,
    l2Lambda: number,
    gradientClipNorm: number,
    epsilon: number,
  ) {
    this._lambda = lambda;
    this._delta = delta;
    this._l2Lambda = l2Lambda;
    this._gradientClipNorm = gradientClipNorm;
    this._epsilon = epsilon;

    this._Pz = new Float64Array(extendedSize);
    this._gain = new Float64Array(extendedSize);
  }

  update(
    state: RLSState,
    extendedState: Float64Array,
    error: Float64Array,
    Wout: Float64Array,
    nTargets: number,
    sampleWeight: number = 1.0,
  ): number {
    const n = state.extendedSize;
    const P = state.P;

    // Pz = P * extendedState
    TensorOps.matVec(P, extendedState, this._Pz, n, n);

    // zTPz = extendedState' * Pz
    const zTPz = TensorOps.dot(extendedState, this._Pz, n);

    // gamma = lambda + zTPz
    const gamma = this._lambda + zTPz;

    // gain = Pz / gamma
    const invGamma = 1.0 / (gamma + this._epsilon);
    for (let i = 0; i < n; i++) {
      this._gain[i] = this._Pz[i] * invGamma;
    }

    // Apply gradient clipping
    let gradNorm = TensorOps.norm(this._gain, n);
    if (gradNorm > this._gradientClipNorm) {
      const scale = this._gradientClipNorm / gradNorm;
      for (let i = 0; i < n; i++) {
        this._gain[i] *= scale;
      }
      gradNorm = this._gradientClipNorm;
    }

    // Update weights: Wout += gain * error' * sampleWeight
    for (let t = 0; t < nTargets; t++) {
      const rowOffset = t * n;
      const err = error[t] * sampleWeight;
      for (let i = 0; i < n; i++) {
        Wout[rowOffset + i] += this._gain[i] * err;
      }
    }

    // Apply L2 regularization
    if (this._l2Lambda > 0) {
      const decay = 1 - this._l2Lambda;
      for (let i = 0; i < Wout.length; i++) {
        Wout[i] *= decay;
      }
    }

    // Update P: P = (P - gain * Pz') / lambda
    const invLambda = 1.0 / this._lambda;
    for (let i = 0; i < n; i++) {
      const gainI = this._gain[i];
      const rowOffset = i * n;
      for (let j = 0; j < n; j++) {
        P[rowOffset + j] = (P[rowOffset + j] - gainI * this._Pz[j]) * invLambda;
      }
    }

    // Periodically maintain numerical stability
    state.updateCount++;
    if (state.updateCount % 100 === 0) {
      this._symmetrizeP(P, n);
      this._checkPStability(state);
    }

    return gradNorm;
  }

  private _symmetrizeP(P: Float64Array, n: number): void {
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const avg = (P[i * n + j] + P[j * n + i]) * 0.5;
        P[i * n + j] = avg;
        P[j * n + i] = avg;
      }
    }
  }

  private _checkPStability(state: RLSState): void {
    const n = state.extendedSize;
    const P = state.P;

    let trace = 0;
    for (let i = 0; i < n; i++) {
      trace += P[i * n + i];
    }

    if (trace > 1e10 || trace < 1e-10 || !Number.isFinite(trace)) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          P[i * n + j] = i === j ? this._delta : 0;
        }
      }
    }
  }

  reinitializeScratch(extendedSize: number): void {
    this._Pz = new Float64Array(extendedSize);
    this._gain = new Float64Array(extendedSize);
  }

  serialize(): object {
    return {
      lambda: this._lambda,
      delta: this._delta,
      l2Lambda: this._l2Lambda,
      gradientClipNorm: this._gradientClipNorm,
      epsilon: this._epsilon,
    };
  }

  static deserialize(data: any, extendedSize: number): RLSOptimizer {
    return new RLSOptimizer(
      extendedSize,
      data.lambda,
      data.delta,
      data.l2Lambda,
      data.gradientClipNorm,
      data.epsilon,
    );
  }
}

// ============================================
// LINEAR READOUT
// ============================================

class LinearReadout {
  private readonly _extendedSize: number;
  private readonly _nTargets: number;
  private _Wout: Float64Array;
  private readonly _useInputInReadout: boolean;
  private readonly _useBiasInReadout: boolean;
  private readonly _reservoirSize: number;
  private readonly _inputSize: number;

  constructor(
    reservoirSize: number,
    inputSize: number,
    nTargets: number,
    useInputInReadout: boolean,
    useBiasInReadout: boolean,
    weightInitScale: number,
    rng: RandomGenerator,
  ) {
    this._reservoirSize = reservoirSize;
    this._inputSize = inputSize;
    this._nTargets = nTargets;
    this._useInputInReadout = useInputInReadout;
    this._useBiasInReadout = useBiasInReadout;

    this._extendedSize = reservoirSize;
    if (useInputInReadout) this._extendedSize += inputSize;
    if (useBiasInReadout) this._extendedSize += 1;

    this._Wout = new Float64Array(nTargets * this._extendedSize);
    for (let i = 0; i < this._Wout.length; i++) {
      this._Wout[i] = rng.normal() * weightInitScale;
    }
  }

  buildExtendedState(
    reservoirState: Float64Array,
    input: Float64Array,
    extendedState: Float64Array,
    reservoirOffset: number = 0,
    inputOffset: number = 0,
    extendedOffset: number = 0,
  ): void {
    let offset = extendedOffset;

    TensorOps.copy(
      reservoirState,
      extendedState,
      this._reservoirSize,
      reservoirOffset,
      offset,
    );
    offset += this._reservoirSize;

    if (this._useInputInReadout) {
      TensorOps.copy(
        input,
        extendedState,
        this._inputSize,
        inputOffset,
        offset,
      );
      offset += this._inputSize;
    }

    if (this._useBiasInReadout) {
      extendedState[offset] = 1.0;
    }
  }

  forward(
    extendedState: Float64Array,
    output: Float64Array,
    extendedOffset: number = 0,
    outputOffset: number = 0,
  ): void {
    for (let t = 0; t < this._nTargets; t++) {
      let sum = 0;
      const rowOffset = t * this._extendedSize;
      for (let i = 0; i < this._extendedSize; i++) {
        sum += this._Wout[rowOffset + i] * extendedState[extendedOffset + i];
      }
      output[outputOffset + t] = sum;
    }
  }

  get extendedSize(): number {
    return this._extendedSize;
  }

  get Wout(): Float64Array {
    return this._Wout;
  }

  get nTargets(): number {
    return this._nTargets;
  }

  serialize(): object {
    return {
      extendedSize: this._extendedSize,
      nTargets: this._nTargets,
      Wout: Array.from(this._Wout),
      useInputInReadout: this._useInputInReadout,
      useBiasInReadout: this._useBiasInReadout,
      reservoirSize: this._reservoirSize,
      inputSize: this._inputSize,
    };
  }

  static deserialize(data: any): LinearReadout {
    const readout = Object.create(LinearReadout.prototype);
    readout._extendedSize = data.extendedSize;
    readout._nTargets = data.nTargets;
    readout._Wout = new Float64Array(data.Wout);
    readout._useInputInReadout = data.useInputInReadout;
    readout._useBiasInReadout = data.useBiasInReadout;
    readout._reservoirSize = data.reservoirSize;
    readout._inputSize = data.inputSize;
    return readout;
  }

  getWeights(): Array<{ name: string; shape: number[]; values: number[] }> {
    return [
      {
        name: "Wout",
        shape: [this._nTargets, this._extendedSize],
        values: Array.from(this._Wout),
      },
    ];
  }
}

// ============================================
// SERIALIZATION HELPER
// ============================================

class SerializationHelper {
  static serialize(obj: any): string {
    return JSON.stringify(obj);
  }

  static deserialize(str: string): any {
    return JSON.parse(str);
  }
}

// ============================================
// ESN REGRESSION CLASS
// ============================================

export class ESNRegression {
  private readonly _config: ESNRegressionConfig;
  private _initialized: boolean = false;
  private _nFeatures: number = 0;
  private _sampleCount: number = 0;

  // Core components (lazily initialized)
  private _reservoir: ESNReservoir | null = null;
  private _readout: LinearReadout | null = null;
  private _rlsState: RLSState | null = null;
  private _rlsOptimizer: RLSOptimizer | null = null;
  private _normalizer: WelfordNormalizer | null = null;
  private _residualTracker: ResidualStatsTracker | null = null;
  private _outlierDownweighter: OutlierDownweighter;
  private _rng: RandomGenerator;
  private _bufferPool: BufferPool;

  // Pre-allocated scratch buffers for training
  private _normalizedInput: Float64Array | null = null;
  private _normalizedTarget: Float64Array | null = null;
  private _prediction: Float64Array | null = null;
  private _error: Float64Array | null = null;
  private _extendedState: Float64Array | null = null;
  private _latestCoordinates: Float64Array | null = null;

  // Pre-allocated scratch buffers for prediction
  private _predScratchState: Float64Array | null = null;
  private _predScratch: Float64Array | null = null;
  private _predInput: Float64Array | null = null;
  private _predNormInput: Float64Array | null = null;
  private _predOutput: Float64Array | null = null;
  private _predDenorm: Float64Array | null = null;
  private _predExtState: Float64Array | null = null;

  /**
   * Create a new ESN Regression model
   * @param config - Partial configuration to merge with defaults
   */
  constructor(config?: Partial<ESNRegressionConfig>) {
    this._config = { ...DEFAULT_CONFIG, ...config };
    this._rng = new RandomGenerator(this._config.seed);
    this._bufferPool = new BufferPool();
    this._outlierDownweighter = new OutlierDownweighter(
      this._config.outlierThreshold,
      this._config.outlierMinWeight,
    );
  }

  private _initialize(nFeatures: number): void {
    this._nFeatures = nFeatures;
    const reservoirSize = this._config.reservoirSize;

    this._reservoir = new ESNReservoir(
      reservoirSize,
      nFeatures,
      this._config.spectralRadius,
      this._config.leakRate,
      this._config.inputScale,
      this._config.biasScale,
      this._config.reservoirSparsity,
      this._config.inputSparsity,
      this._config.activation,
      this._rng,
    );

    this._readout = new LinearReadout(
      reservoirSize,
      nFeatures,
      nFeatures,
      this._config.useInputInReadout,
      this._config.useBiasInReadout,
      this._config.weightInitScale,
      this._rng,
    );

    const extendedSize = this._readout.extendedSize;
    this._rlsState = new RLSState(extendedSize, this._config.rlsDelta);
    this._rlsOptimizer = new RLSOptimizer(
      extendedSize,
      this._config.rlsLambda,
      this._config.rlsDelta,
      this._config.l2Lambda,
      this._config.gradientClipNorm,
      this._config.epsilon,
    );

    this._normalizer = new WelfordNormalizer(
      nFeatures,
      this._config.normalizationEpsilon,
      this._config.normalizationWarmup,
    );

    this._residualTracker = new ResidualStatsTracker(nFeatures);

    // Allocate training scratch buffers
    this._normalizedInput = new Float64Array(nFeatures);
    this._normalizedTarget = new Float64Array(nFeatures);
    this._prediction = new Float64Array(nFeatures);
    this._error = new Float64Array(nFeatures);
    this._extendedState = new Float64Array(extendedSize);
    this._latestCoordinates = new Float64Array(nFeatures);

    // Allocate prediction scratch buffers
    this._predScratchState = new Float64Array(reservoirSize);
    this._predScratch = new Float64Array(reservoirSize);
    this._predInput = new Float64Array(nFeatures);
    this._predNormInput = new Float64Array(nFeatures);
    this._predOutput = new Float64Array(nFeatures);
    this._predDenorm = new Float64Array(nFeatures);
    this._predExtState = new Float64Array(extendedSize);

    this._initialized = true;
  }

  /**
   * Train the model online with new coordinate data
   * @param params - Object containing coordinates array
   * @returns Training metrics
   */
  fitOnline(params: { coordinates: number[][] }): FitResult {
    const { coordinates } = params;

    if (!Array.isArray(coordinates) || coordinates.length < 2) {
      throw new Error(
        "fitOnline requires coordinates array with at least 2 rows",
      );
    }

    const nFeatures = coordinates[0].length;
    if (nFeatures === 0) {
      throw new Error("Coordinate vectors cannot be empty");
    }

    for (let i = 1; i < coordinates.length; i++) {
      if (coordinates[i].length !== nFeatures) {
        throw new Error(
          `Inconsistent dimension at row ${i}: expected ${nFeatures}, got ${
            coordinates[i].length
          }`,
        );
      }
    }

    if (!this._initialized) {
      this._initialize(nFeatures);
    } else if (this._nFeatures !== nFeatures) {
      throw new Error(
        `Feature dimension mismatch: model expects ${this._nFeatures}, got ${nFeatures}`,
      );
    }

    const metrics = new MetricsAccumulator();
    let lastSampleWeight = 1.0;
    let driftDetected = false;

    const nSamples = coordinates.length - 1;

    for (let i = 0; i < nSamples; i++) {
      const input = coordinates[i];
      const target = coordinates[i + 1];

      // Copy input to buffer
      for (let j = 0; j < nFeatures; j++) {
        this._normalizedInput![j] = input[j];
      }

      // Update normalizer statistics
      this._normalizer!.update(this._normalizedInput!);

      // Normalize input
      this._normalizer!.normalize(
        this._normalizedInput!,
        this._normalizedInput!,
      );

      // Update reservoir state
      this._reservoir!.update(this._normalizedInput!);

      // Build extended state
      this._readout!.buildExtendedState(
        this._reservoir!.state,
        this._normalizedInput!,
        this._extendedState!,
      );

      // Forward through readout
      this._readout!.forward(this._extendedState!, this._prediction!);

      // Copy and normalize target
      for (let j = 0; j < nFeatures; j++) {
        this._normalizedTarget![j] = target[j];
      }
      this._normalizer!.normalize(
        this._normalizedTarget!,
        this._normalizedTarget!,
      );

      // Compute error
      for (let j = 0; j < nFeatures; j++) {
        this._error![j] = this._normalizedTarget![j] - this._prediction![j];
      }

      // Compute loss
      const loss = LossFunction.mse(
        this._prediction!,
        this._normalizedTarget!,
        nFeatures,
      );
      metrics.addLoss(loss);

      // Compute sample weight for outlier handling
      const residualStds = this._residualTracker!.getStds();
      const sampleWeight = this._outlierDownweighter.computeWeight(
        this._error!,
        residualStds,
        this._config.epsilon,
      );
      lastSampleWeight = sampleWeight;

      // Detect drift
      if (sampleWeight < 0.5 && this._sampleCount > 100) {
        driftDetected = true;
      }

      // Update readout weights via RLS
      const gradNorm = this._rlsOptimizer!.update(
        this._rlsState!,
        this._extendedState!,
        this._error!,
        this._readout!.Wout,
        nFeatures,
        sampleWeight,
      );
      metrics.setGradientNorm(gradNorm);

      // Update residual statistics
      this._residualTracker!.update(this._error!);

      // Store latest coordinates
      for (let j = 0; j < nFeatures; j++) {
        this._latestCoordinates![j] = target[j];
      }

      this._sampleCount++;
    }

    return {
      samplesProcessed: nSamples,
      averageLoss: metrics.averageLoss,
      gradientNorm: metrics.gradientNorm,
      driftDetected,
      sampleWeight: lastSampleWeight,
    };
  }

  /**
   * Generate multi-step predictions
   * @param futureSteps - Number of future time steps to predict
   * @returns Predictions with confidence bounds
   */
  predict(futureSteps: number): PredictionResult {
    if (!this._initialized) {
      throw new Error(
        "Model must be initialized before prediction. Call fitOnline first.",
      );
    }

    if (!Number.isInteger(futureSteps) || futureSteps < 1) {
      throw new Error("futureSteps must be a positive integer");
    }

    const predictions: number[][] = [];
    const lowerBounds: number[][] = [];
    const upperBounds: number[][] = [];
    const nFeatures = this._nFeatures;

    // Copy current reservoir state
    this._reservoir!.copyState(this._predScratchState!);

    // Get statistics for uncertainty estimation
    const residualStds = this._residualTracker!.getStds();
    const normStds = this._normalizer!.getStds();
    const mins = this._normalizer!.getMins();
    const maxs = this._normalizer!.getMaxs();

    const ranges: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      ranges[j] = Math.max(maxs[j] - mins[j], 1e-6);
    }

    // Start with latest coordinates
    TensorOps.copy(this._latestCoordinates!, this._predInput!, nFeatures);

    let totalUncertainty = 0;

    for (let step = 0; step < futureSteps; step++) {
      // Normalize input
      this._normalizer!.normalize(this._predInput!, this._predNormInput!);
      TensorOps.clampInPlace(this._predNormInput!, -10, 10, nFeatures);

      // Update reservoir state using scratch
      this._reservoir!.updateScratch(
        this._predNormInput!,
        this._predScratchState!,
        this._predScratch!,
      );

      // Build extended state
      this._readout!.buildExtendedState(
        this._predScratchState!,
        this._predNormInput!,
        this._predExtState!,
      );

      // Forward through readout
      this._readout!.forward(this._predExtState!, this._predOutput!);

      // Clamp before denormalization
      TensorOps.clampInPlace(this._predOutput!, -100, 100, nFeatures);

      // Denormalize
      this._normalizer!.denormalize(this._predOutput!, this._predDenorm!);

      // Sanitize and clamp to observed range
      const lower: number[] = [];
      const upper: number[] = [];

      for (let j = 0; j < nFeatures; j++) {
        let val = this._predDenorm![j];

        if (!Number.isFinite(val)) {
          val = this._predInput![j];
        }

        const margin = ranges[j] * 0.5;
        const minClamp = mins[j] - margin;
        const maxClamp = maxs[j] + margin;
        if (val < minClamp) val = minClamp;
        if (val > maxClamp) val = maxClamp;

        this._predDenorm![j] = val;

        // Compute uncertainty bounds
        const horizonFactor = Math.sqrt(step + 1);
        const uncertaintyMult = this._config.uncertaintyMultiplier *
          horizonFactor;
        const std = this._normalizer!.isActive
          ? residualStds[j] * normStds[j]
          : residualStds[j];
        const uncertainty = std * uncertaintyMult;

        lower.push(val - uncertainty);
        upper.push(val + uncertainty);
        totalUncertainty += uncertainty;
      }

      predictions.push(Array.from(this._predDenorm!));
      lowerBounds.push(lower);
      upperBounds.push(upper);

      // Use prediction as next input
      TensorOps.copy(this._predDenorm!, this._predInput!, nFeatures);
    }

    // Compute confidence score
    const avgUncertainty = totalUncertainty / (futureSteps * nFeatures);
    const avgRange = ranges.reduce((a, b) => a + b, 0) / nFeatures;
    const confidence = Math.max(
      0,
      Math.min(1, 1 - avgUncertainty / (avgRange + this._config.epsilon)),
    );

    return {
      predictions,
      lowerBounds,
      upperBounds,
      confidence,
    };
  }

  /**
   * Get a summary of the model architecture
   * @returns Model summary object
   */
  getModelSummary(): ModelSummary {
    const reservoirSize = this._config.reservoirSize;
    const nFeatures = this._nFeatures || 0;
    const extendedSize = this._readout?.extendedSize || 0;

    let totalParams = 0;
    if (this._initialized && this._reservoir && this._readout) {
      // Reservoir weights
      totalParams += (this._reservoir as any)._WValues.length;

      // Input weights
      const winDense = (this._reservoir as any)._WinDense;
      const winValues = (this._reservoir as any)._WinValues;
      if (winDense) {
        totalParams += winDense.length;
      } else {
        totalParams += winValues.length;
      }

      // Bias
      totalParams += reservoirSize;

      // Output weights
      totalParams += nFeatures * extendedSize;
    }

    const receptiveField = Math.ceil(
      1 / (1 - this._config.spectralRadius + this._config.epsilon),
    );

    return {
      totalParameters: totalParams,
      receptiveField,
      spectralRadius: this._config.spectralRadius,
      reservoirSize,
      nFeatures,
      nTargets: nFeatures,
      sampleCount: this._sampleCount,
    };
  }

  /**
   * Get all weight matrices
   * @returns Weight information object
   */
  getWeights(): WeightInfo {
    if (!this._initialized || !this._readout || !this._reservoir) {
      return { weights: [] };
    }

    const weights: Array<{ name: string; shape: number[]; values: number[] }> =
      [];
    weights.push(...this._readout.getWeights());
    weights.push(...this._reservoir.getWeights());

    return { weights };
  }

  /**
   * Get normalization statistics
   * @returns Normalization stats object
   */
  getNormalizationStats(): NormalizationStats {
    if (!this._normalizer) {
      return { means: [], stds: [], count: 0, isActive: false };
    }

    return {
      means: this._normalizer.getMeans(),
      stds: this._normalizer.getStds(),
      count: this._normalizer.count,
      isActive: this._normalizer.isActive,
    };
  }

  /**
   * Reset the model to uninitialized state
   */
  reset(): void {
    this._initialized = false;
    this._nFeatures = 0;
    this._sampleCount = 0;
    this._reservoir = null;
    this._readout = null;
    this._rlsState = null;
    this._rlsOptimizer = null;
    this._normalizer = null;
    this._residualTracker = null;
    this._normalizedInput = null;
    this._normalizedTarget = null;
    this._prediction = null;
    this._error = null;
    this._extendedState = null;
    this._latestCoordinates = null;
    this._predScratchState = null;
    this._predScratch = null;
    this._predInput = null;
    this._predNormInput = null;
    this._predOutput = null;
    this._predDenorm = null;
    this._predExtState = null;
    this._rng = new RandomGenerator(this._config.seed);
    this._bufferPool.clear();
  }

  /**
   * Serialize model state to JSON string
   * @returns JSON string representation of model
   */
  save(): string {
    const state: any = {
      config: this._config,
      initialized: this._initialized,
      nFeatures: this._nFeatures,
      sampleCount: this._sampleCount,
      rngState: this._rng.getState(),
    };

    if (this._initialized) {
      state.reservoir = this._reservoir!.serialize();
      state.readout = this._readout!.serialize();
      state.rlsState = this._rlsState!.serialize();
      state.rlsOptimizer = this._rlsOptimizer!.serialize();
      state.normalizer = this._normalizer!.serialize();
      state.residualTracker = this._residualTracker!.serialize();
      state.outlierDownweighter = this._outlierDownweighter.serialize();
      state.latestCoordinates = Array.from(this._latestCoordinates!);
    }

    return SerializationHelper.serialize(state);
  }

  /**
   * Load model state from JSON string
   * @param str - JSON string to load
   */
  load(str: string): void {
    const state = SerializationHelper.deserialize(str);

    (this as any)._config = state.config;
    this._initialized = state.initialized;
    this._nFeatures = state.nFeatures;
    this._sampleCount = state.sampleCount;
    this._rng = new RandomGenerator(state.config.seed);
    this._rng.setState(state.rngState);

    if (state.initialized) {
      const reservoirSize = state.config.reservoirSize;
      const nFeatures = state.nFeatures;

      this._reservoir = ESNReservoir.deserialize(state.reservoir);
      this._readout = LinearReadout.deserialize(state.readout);
      this._rlsState = RLSState.deserialize(state.rlsState);
      this._rlsOptimizer = RLSOptimizer.deserialize(
        state.rlsOptimizer,
        this._readout.extendedSize,
      );
      this._normalizer = WelfordNormalizer.deserialize(state.normalizer);
      this._residualTracker = ResidualStatsTracker.deserialize(
        state.residualTracker,
      );
      this._outlierDownweighter = OutlierDownweighter.deserialize(
        state.outlierDownweighter,
      );
      this._latestCoordinates = new Float64Array(state.latestCoordinates);

      // Reinitialize scratch buffers
      const extendedSize = this._readout.extendedSize;
      this._normalizedInput = new Float64Array(nFeatures);
      this._normalizedTarget = new Float64Array(nFeatures);
      this._prediction = new Float64Array(nFeatures);
      this._error = new Float64Array(nFeatures);
      this._extendedState = new Float64Array(extendedSize);
      this._predScratchState = new Float64Array(reservoirSize);
      this._predScratch = new Float64Array(reservoirSize);
      this._predInput = new Float64Array(nFeatures);
      this._predNormInput = new Float64Array(nFeatures);
      this._predOutput = new Float64Array(nFeatures);
      this._predDenorm = new Float64Array(nFeatures);
      this._predExtState = new Float64Array(extendedSize);
    }
  }
}

export default ESNRegression;
