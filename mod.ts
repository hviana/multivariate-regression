/**
 * ESNRegression: Echo State Network for Multivariate Regression
 * with Incremental Online Learning using RLS Readout Training
 * and Welford Z-Score Normalization
 *
 * Single self-contained TypeScript module with no heavy runtime dependencies.
 */

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
// TENSOR SHAPE
// ============================================================================

class TensorShape {
  readonly dims: readonly number[];
  readonly size: number;
  readonly strides: readonly number[];

  constructor(dims: number[]) {
    this.dims = Object.freeze([...dims]);
    let s = 1;
    for (let i = 0; i < dims.length; i++) {
      s *= dims[i];
    }
    this.size = s;
    const strides: number[] = new Array(dims.length);
    let stride = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= dims[i];
    }
    this.strides = Object.freeze(strides);
  }

  index(indices: number[]): number {
    let idx = 0;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * this.strides[i];
    }
    return idx;
  }

  equals(other: TensorShape): boolean {
    if (this.dims.length !== other.dims.length) return false;
    for (let i = 0; i < this.dims.length; i++) {
      if (this.dims[i] !== other.dims[i]) return false;
    }
    return true;
  }
}

// ============================================================================
// TENSOR VIEW (Zero-copy view into Float64Array)
// ============================================================================

class TensorView {
  data: Float64Array;
  offset: number;
  shape: TensorShape;

  constructor(data: Float64Array, offset: number, shape: TensorShape) {
    this.data = data;
    this.offset = offset;
    this.shape = shape;
  }

  get(indices: number[]): number {
    return this.data[this.offset + this.shape.index(indices)];
  }

  set(indices: number[], value: number): void {
    this.data[this.offset + this.shape.index(indices)] = value;
  }

  getFlat(i: number): number {
    return this.data[this.offset + i];
  }

  setFlat(i: number, value: number): void {
    this.data[this.offset + i] = value;
  }

  fill(value: number): void {
    const end = this.offset + this.shape.size;
    for (let i = this.offset; i < end; i++) {
      this.data[i] = value;
    }
  }

  copyFrom(source: TensorView): void {
    const size = this.shape.size;
    for (let i = 0; i < size; i++) {
      this.data[this.offset + i] = source.data[source.offset + i];
    }
  }

  copyFromArray(arr: number[]): void {
    for (let i = 0; i < arr.length; i++) {
      this.data[this.offset + i] = arr[i];
    }
  }

  toArray(): number[] {
    const result: number[] = new Array(this.shape.size);
    for (let i = 0; i < this.shape.size; i++) {
      result[i] = this.data[this.offset + i];
    }
    return result;
  }
}

// ============================================================================
// BUFFER POOL (Object reuse for TensorView shells)
// ============================================================================

class BufferPool {
  private pool: TensorView[] = [];

  acquire(data: Float64Array, offset: number, shape: TensorShape): TensorView {
    let view = this.pool.pop();
    if (view) {
      view.data = data;
      view.offset = offset;
      view.shape = shape;
      return view;
    }
    return new TensorView(data, offset, shape);
  }

  release(view: TensorView): void {
    this.pool.push(view);
  }
}

// ============================================================================
// TENSOR ARENA (Preallocated contiguous memory)
// ============================================================================

class TensorArena {
  private buffer: Float64Array;
  private allocated: number = 0;

  constructor(totalSize: number) {
    this.buffer = new Float64Array(totalSize);
  }

  allocate(shape: TensorShape): TensorView {
    const offset = this.allocated;
    this.allocated += shape.size;
    if (this.allocated > this.buffer.length) {
      throw new Error("TensorArena: out of memory");
    }
    return new TensorView(this.buffer, offset, shape);
  }

  reset(): void {
    this.allocated = 0;
    this.buffer.fill(0);
  }

  getUsed(): number {
    return this.allocated;
  }

  getCapacity(): number {
    return this.buffer.length;
  }
}

// ============================================================================
// TENSOR OPS (Basic linear algebra, allocation-free)
// ============================================================================

class TensorOps {
  /**
   * Matrix-vector multiply: y = A * x
   * A: [rows, cols], x: [cols], y: [rows]
   */
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
      let sum = 0;
      const rowOffset = aOffset + i * cols;
      for (let j = 0; j < cols; j++) {
        sum += A[rowOffset + j] * x[xOffset + j];
      }
      y[yOffset + i] = sum;
    }
  }

  /**
   * Sparse matrix-vector multiply using mask
   */
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
      let sum = 0;
      const rowOffset = aOffset + i * cols;
      const maskRowOffset = maskOffset + i * cols;
      for (let j = 0; j < cols; j++) {
        if (mask[maskRowOffset + j]) {
          sum += A[rowOffset + j] * x[xOffset + j];
        }
      }
      y[yOffset + i] = sum;
    }
  }

  /**
   * Vector add: y = a + b
   */
  static vecAdd(
    a: Float64Array,
    aOffset: number,
    b: Float64Array,
    bOffset: number,
    y: Float64Array,
    yOffset: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      y[yOffset + i] = a[aOffset + i] + b[bOffset + i];
    }
  }

  /**
   * Vector scale: y = alpha * x
   */
  static vecScale(
    x: Float64Array,
    xOffset: number,
    alpha: number,
    y: Float64Array,
    yOffset: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      y[yOffset + i] = alpha * x[xOffset + i];
    }
  }

  /**
   * Vector dot product
   */
  static dot(
    a: Float64Array,
    aOffset: number,
    b: Float64Array,
    bOffset: number,
    len: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += a[aOffset + i] * b[bOffset + i];
    }
    return sum;
  }

  /**
   * Vector L2 norm
   */
  static norm(x: Float64Array, xOffset: number, len: number): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const v = x[xOffset + i];
      sum += v * v;
    }
    return Math.sqrt(sum);
  }

  /**
   * Outer product update: A += alpha * x * y^T
   * A: [m, n], x: [m], y: [n]
   */
  static outerUpdate(
    A: Float64Array,
    aOffset: number,
    m: number,
    n: number,
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    alpha: number,
  ): void {
    for (let i = 0; i < m; i++) {
      const xi = x[xOffset + i];
      const rowOffset = aOffset + i * n;
      for (let j = 0; j < n; j++) {
        A[rowOffset + j] += alpha * xi * y[yOffset + j];
      }
    }
  }

  /**
   * Copy vector
   */
  static vecCopy(
    src: Float64Array,
    srcOffset: number,
    dst: Float64Array,
    dstOffset: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOffset + i] = src[srcOffset + i];
    }
  }

  /**
   * Fill vector with value
   */
  static vecFill(
    dst: Float64Array,
    dstOffset: number,
    value: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      dst[dstOffset + i] = value;
    }
  }

  /**
   * Clamp vector values
   */
  static vecClamp(
    x: Float64Array,
    xOffset: number,
    minVal: number,
    maxVal: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      const idx = xOffset + i;
      if (x[idx] < minVal) x[idx] = minVal;
      else if (x[idx] > maxVal) x[idx] = maxVal;
    }
  }
}

// ============================================================================
// ACTIVATION OPS
// ============================================================================

class ActivationOps {
  static tanh(
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      y[yOffset + i] = Math.tanh(x[xOffset + i]);
    }
  }

  static relu(
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      const v = x[xOffset + i];
      y[yOffset + i] = v > 0 ? v : 0;
    }
  }

  static apply(
    activation: "tanh" | "relu",
    x: Float64Array,
    xOffset: number,
    y: Float64Array,
    yOffset: number,
    len: number,
  ): void {
    if (activation === "tanh") {
      ActivationOps.tanh(x, xOffset, y, yOffset, len);
    } else {
      ActivationOps.relu(x, xOffset, y, yOffset, len);
    }
  }
}

// ============================================================================
// RANDOM GENERATOR (Seeded PRNG)
// ============================================================================

class RandomGenerator {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
    if (this.state === 0) this.state = 1;
  }

  /**
   * Xorshift32 PRNG
   */
  next(): number {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }

  /**
   * Uniform random in [0, 1)
   */
  uniform(): number {
    return this.next() / 4294967296;
  }

  /**
   * Normal distribution using Box-Muller
   */
  normal(mean: number = 0, std: number = 1): number {
    const u1 = this.uniform();
    const u2 = this.uniform();
    const z0 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) *
      Math.cos(2 * Math.PI * u2);
    return mean + std * z0;
  }

  /**
   * Returns true with probability p
   */
  bernoulli(p: number): boolean {
    return this.uniform() < p;
  }

  getState(): number {
    return this.state;
  }

  setState(state: number): void {
    this.state = state >>> 0;
    if (this.state === 0) this.state = 1;
  }
}

// ============================================================================
// WELFORD ACCUMULATOR (Online mean/variance)
// ============================================================================

class WelfordAccumulator {
  count: number = 0;
  mean: number = 0;
  m2: number = 0;

  update(value: number): void {
    this.count++;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    const delta2 = value - this.mean;
    this.m2 += delta * delta2;
  }

  getVariance(): number {
    if (this.count < 2) return 0;
    return this.m2 / (this.count - 1);
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

// ============================================================================
// WELFORD NORMALIZER (Per-feature online normalization)
// ============================================================================

class WelfordNormalizer {
  private nFeatures: number;
  private counts: Float64Array;
  private means: Float64Array;
  private m2s: Float64Array;
  private epsilon: number;
  private warmup: number;

  constructor(nFeatures: number, epsilon: number, warmup: number) {
    this.nFeatures = nFeatures;
    this.counts = new Float64Array(nFeatures);
    this.means = new Float64Array(nFeatures);
    this.m2s = new Float64Array(nFeatures);
    this.epsilon = epsilon;
    this.warmup = warmup;
  }

  /**
   * Update statistics with new observation
   */
  update(x: Float64Array, xOffset: number): void {
    for (let j = 0; j < this.nFeatures; j++) {
      const value = x[xOffset + j];
      this.counts[j]++;
      const delta = value - this.means[j];
      this.means[j] += delta / this.counts[j];
      const delta2 = value - this.means[j];
      this.m2s[j] += delta * delta2;
    }
  }

  /**
   * Normalize input in-place
   */
  normalize(
    x: Float64Array,
    xOffset: number,
    out: Float64Array,
    outOffset: number,
  ): void {
    for (let j = 0; j < this.nFeatures; j++) {
      const mean = this.means[j];
      const variance = this.counts[j] > 1
        ? this.m2s[j] / (this.counts[j] - 1)
        : 0;
      const std = Math.sqrt(variance);
      const denom = Math.max(std, this.epsilon);
      out[outOffset + j] = (x[xOffset + j] - mean) / denom;
    }
  }

  isActive(): boolean {
    if (this.nFeatures === 0) return false;
    return this.counts[0] >= this.warmup;
  }

  getMeans(): number[] {
    return Array.from(this.means);
  }

  getStds(): number[] {
    const stds: number[] = new Array(this.nFeatures);
    for (let j = 0; j < this.nFeatures; j++) {
      const variance = this.counts[j] > 1
        ? this.m2s[j] / (this.counts[j] - 1)
        : 0;
      stds[j] = Math.sqrt(variance);
    }
    return stds;
  }

  getCount(): number {
    return this.nFeatures > 0 ? this.counts[0] : 0;
  }

  reset(): void {
    this.counts.fill(0);
    this.means.fill(0);
    this.m2s.fill(0);
  }

  // Serialization
  serialize(): { counts: number[]; means: number[]; m2s: number[] } {
    return {
      counts: Array.from(this.counts),
      means: Array.from(this.means),
      m2s: Array.from(this.m2s),
    };
  }

  deserialize(
    data: { counts: number[]; means: number[]; m2s: number[] },
  ): void {
    for (let i = 0; i < this.nFeatures; i++) {
      this.counts[i] = data.counts[i] || 0;
      this.means[i] = data.means[i] || 0;
      this.m2s[i] = data.m2s[i] || 0;
    }
  }
}

// ============================================================================
// RING BUFFER (Fixed-size circular buffer for time series)
// ============================================================================

class RingBuffer {
  private buffer: Float64Array;
  private head: number = 0;
  private count: number = 0;
  private capacity: number;
  private nFeatures: number;

  constructor(capacity: number, nFeatures: number) {
    this.capacity = capacity;
    this.nFeatures = nFeatures;
    this.buffer = new Float64Array(capacity * nFeatures);
  }

  /**
   * Push a new row into the buffer
   */
  push(row: number[] | Float64Array, rowOffset: number = 0): void {
    const writeIdx = this.head * this.nFeatures;
    for (let i = 0; i < this.nFeatures; i++) {
      this.buffer[writeIdx + i] = row instanceof Float64Array
        ? row[rowOffset + i]
        : row[i];
    }
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  /**
   * Get the most recent row (index 0 = newest)
   */
  getLatest(out: Float64Array, outOffset: number): boolean {
    if (this.count === 0) return false;
    const idx = ((this.head - 1 + this.capacity) % this.capacity) *
      this.nFeatures;
    for (let i = 0; i < this.nFeatures; i++) {
      out[outOffset + i] = this.buffer[idx + i];
    }
    return true;
  }

  /**
   * Get row by age (0 = newest, count-1 = oldest)
   */
  getByAge(age: number, out: Float64Array, outOffset: number): boolean {
    if (age >= this.count) return false;
    const idx = ((this.head - 1 - age + this.capacity * 2) % this.capacity) *
      this.nFeatures;
    for (let i = 0; i < this.nFeatures; i++) {
      out[outOffset + i] = this.buffer[idx + i];
    }
    return true;
  }

  /**
   * Get recent window as contiguous array (oldest to newest)
   * Returns actual length filled
   */
  getWindow(windowSize: number, out: Float64Array, outOffset: number): number {
    const actualSize = Math.min(windowSize, this.count);
    for (let age = actualSize - 1; age >= 0; age--) {
      const idx = ((this.head - 1 - age + this.capacity * 2) % this.capacity) *
        this.nFeatures;
      const outIdx = outOffset + (actualSize - 1 - age) * this.nFeatures;
      for (let i = 0; i < this.nFeatures; i++) {
        out[outIdx + i] = this.buffer[idx + i];
      }
    }
    return actualSize;
  }

  getCount(): number {
    return this.count;
  }

  getCapacity(): number {
    return this.capacity;
  }

  getNFeatures(): number {
    return this.nFeatures;
  }

  isEmpty(): boolean {
    return this.count === 0;
  }

  reset(): void {
    this.head = 0;
    this.count = 0;
    this.buffer.fill(0);
  }

  // Serialization
  serialize(): { buffer: number[]; head: number; count: number } {
    return {
      buffer: Array.from(this.buffer),
      head: this.head,
      count: this.count,
    };
  }

  deserialize(data: { buffer: number[]; head: number; count: number }): void {
    for (let i = 0; i < this.buffer.length; i++) {
      this.buffer[i] = data.buffer[i] || 0;
    }
    this.head = data.head;
    this.count = data.count;
  }
}

// ============================================================================
// RESIDUAL STATS TRACKER (Rolling window of residuals per target)
// ============================================================================

class ResidualStatsTracker {
  private windowSize: number;
  private nTargets: number;
  private buffers: Float64Array;
  private heads: Int32Array;
  private counts: Int32Array;
  private sums: Float64Array;
  private sumSqs: Float64Array;

  constructor(windowSize: number, nTargets: number) {
    this.windowSize = windowSize;
    this.nTargets = nTargets;
    this.buffers = new Float64Array(windowSize * nTargets);
    this.heads = new Int32Array(nTargets);
    this.counts = new Int32Array(nTargets);
    this.sums = new Float64Array(nTargets);
    this.sumSqs = new Float64Array(nTargets);
  }

  /**
   * Update residual stats with new residuals
   */
  update(residuals: Float64Array, offset: number): void {
    for (let t = 0; t < this.nTargets; t++) {
      const r = residuals[offset + t];
      const bufferStart = t * this.windowSize;
      const idx = bufferStart + this.heads[t];

      // If buffer is full, subtract old value
      if (this.counts[t] === this.windowSize) {
        const oldVal = this.buffers[idx];
        this.sums[t] -= oldVal;
        this.sumSqs[t] -= oldVal * oldVal;
      } else {
        this.counts[t]++;
      }

      // Add new value
      this.buffers[idx] = r;
      this.sums[t] += r;
      this.sumSqs[t] += r * r;

      // Advance head
      this.heads[t] = (this.heads[t] + 1) % this.windowSize;
    }
  }

  /**
   * Get standard deviation for target
   */
  getStd(targetIdx: number): number {
    const n = this.counts[targetIdx];
    if (n < 2) return 0;
    const mean = this.sums[targetIdx] / n;
    const variance = (this.sumSqs[targetIdx] / n) - mean * mean;
    return Math.sqrt(Math.max(0, variance));
  }

  /**
   * Get mean residual for target
   */
  getMean(targetIdx: number): number {
    const n = this.counts[targetIdx];
    if (n === 0) return 0;
    return this.sums[targetIdx] / n;
  }

  /**
   * Fill array with standard deviations
   */
  getStds(out: Float64Array, offset: number): void {
    for (let t = 0; t < this.nTargets; t++) {
      out[offset + t] = this.getStd(t);
    }
  }

  getCount(): number {
    return this.nTargets > 0 ? this.counts[0] : 0;
  }

  reset(): void {
    this.buffers.fill(0);
    this.heads.fill(0);
    this.counts.fill(0);
    this.sums.fill(0);
    this.sumSqs.fill(0);
  }

  // Serialization
  serialize(): {
    buffers: number[];
    heads: number[];
    counts: number[];
    sums: number[];
    sumSqs: number[];
  } {
    return {
      buffers: Array.from(this.buffers),
      heads: Array.from(this.heads),
      counts: Array.from(this.counts),
      sums: Array.from(this.sums),
      sumSqs: Array.from(this.sumSqs),
    };
  }

  deserialize(data: {
    buffers: number[];
    heads: number[];
    counts: number[];
    sums: number[];
    sumSqs: number[];
  }): void {
    for (let i = 0; i < this.buffers.length; i++) {
      this.buffers[i] = data.buffers[i] || 0;
    }
    for (let i = 0; i < this.nTargets; i++) {
      this.heads[i] = data.heads[i] || 0;
      this.counts[i] = data.counts[i] || 0;
      this.sums[i] = data.sums[i] || 0;
      this.sumSqs[i] = data.sumSqs[i] || 0;
    }
  }
}

// ============================================================================
// OUTLIER DOWNWEIGHTER
// ============================================================================

class OutlierDownweighter {
  private threshold: number;
  private minWeight: number;

  constructor(threshold: number, minWeight: number) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }

  /**
   * Compute sample weight based on residual z-scores
   * Uses soft downweighting for outliers
   */
  computeWeight(
    residuals: Float64Array,
    offset: number,
    stds: Float64Array,
    stdOffset: number,
    nTargets: number,
  ): number {
    let maxZ = 0;
    for (let t = 0; t < nTargets; t++) {
      const std = stds[stdOffset + t];
      if (std > 1e-10) {
        const z = Math.abs(residuals[offset + t]) / std;
        if (z > maxZ) maxZ = z;
      }
    }
    if (maxZ <= this.threshold) return 1.0;
    // Soft decay: weight = 1 / (1 + (z - threshold))
    const excess = maxZ - this.threshold;
    const weight = 1.0 / (1.0 + excess);
    return Math.max(weight, this.minWeight);
  }
}

// ============================================================================
// LOSS FUNCTION
// ============================================================================

class LossFunction {
  /**
   * Compute MSE loss
   */
  static mse(
    pred: Float64Array,
    predOffset: number,
    target: Float64Array,
    targetOffset: number,
    len: number,
  ): number {
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const diff = pred[predOffset + i] - target[targetOffset + i];
      sum += diff * diff;
    }
    return sum / len;
  }

  /**
   * Compute residuals: residuals = pred - target
   */
  static residuals(
    pred: Float64Array,
    predOffset: number,
    target: Float64Array,
    targetOffset: number,
    out: Float64Array,
    outOffset: number,
    len: number,
  ): void {
    for (let i = 0; i < len; i++) {
      out[outOffset + i] = pred[predOffset + i] - target[targetOffset + i];
    }
  }
}

// ============================================================================
// METRICS ACCUMULATOR
// ============================================================================

class MetricsAccumulator {
  private lossSum: number = 0;
  private lossCount: number = 0;
  private gradNormSum: number = 0;
  private gradNormCount: number = 0;
  private lastWeight: number = 1.0;

  update(loss: number, gradNorm: number, weight: number): void {
    this.lossSum += loss;
    this.lossCount++;
    this.gradNormSum += gradNorm;
    this.gradNormCount++;
    this.lastWeight = weight;
  }

  getAverageLoss(): number {
    return this.lossCount > 0 ? this.lossSum / this.lossCount : 0;
  }

  getAverageGradNorm(): number {
    return this.gradNormCount > 0 ? this.gradNormSum / this.gradNormCount : 0;
  }

  getLastWeight(): number {
    return this.lastWeight;
  }

  getCount(): number {
    return this.lossCount;
  }

  reset(): void {
    this.lossSum = 0;
    this.lossCount = 0;
    this.gradNormSum = 0;
    this.gradNormCount = 0;
    this.lastWeight = 1.0;
  }
}

// ============================================================================
// RESERVOIR INIT MASK (Sparse connectivity)
// ============================================================================

class ReservoirInitMask {
  /**
   * Generate sparse mask for W (reservoir x reservoir)
   */
  static generateReservoirMask(
    size: number,
    sparsity: number,
    rng: RandomGenerator,
  ): Uint8Array {
    const mask = new Uint8Array(size * size);
    const density = 1.0 - sparsity;
    for (let i = 0; i < size * size; i++) {
      mask[i] = rng.bernoulli(density) ? 1 : 0;
    }
    return mask;
  }

  /**
   * Generate sparse mask for Win (reservoir x features)
   */
  static generateInputMask(
    rows: number,
    cols: number,
    sparsity: number,
    rng: RandomGenerator,
  ): Uint8Array {
    const mask = new Uint8Array(rows * cols);
    const density = 1.0 - sparsity;
    for (let i = 0; i < rows * cols; i++) {
      mask[i] = rng.bernoulli(density) ? 1 : 0;
    }
    return mask;
  }
}

// ============================================================================
// SPECTRAL RADIUS SCALER (Power iteration)
// ============================================================================

class SpectralRadiusScaler {
  private static readonly MAX_ITERATIONS = 100;
  private static readonly TOLERANCE = 1e-6;

  /**
   * Estimate spectral radius using power iteration
   * Returns estimated largest eigenvalue magnitude
   */
  static estimate(
    W: Float64Array,
    wOffset: number,
    size: number,
    mask: Uint8Array,
    maskOffset: number,
    rng: RandomGenerator,
    scratch: Float64Array,
    scratchOffset: number,
  ): number {
    // Initialize random vector
    const v = scratch;
    const vOffset = scratchOffset;
    const u = scratch;
    const uOffset = scratchOffset + size;

    // Initialize with random unit vector
    let norm = 0;
    for (let i = 0; i < size; i++) {
      v[vOffset + i] = rng.normal();
      norm += v[vOffset + i] * v[vOffset + i];
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let i = 0; i < size; i++) {
        v[vOffset + i] /= norm;
      }
    }

    let eigenvalue = 0;
    for (let iter = 0; iter < SpectralRadiusScaler.MAX_ITERATIONS; iter++) {
      // u = W * v (sparse)
      TensorOps.sparseMatVec(
        W,
        wOffset,
        size,
        size,
        mask,
        maskOffset,
        v,
        vOffset,
        u,
        uOffset,
      );

      // Compute norm of u
      let uNorm = 0;
      for (let i = 0; i < size; i++) {
        uNorm += u[uOffset + i] * u[uOffset + i];
      }
      uNorm = Math.sqrt(uNorm);

      if (uNorm < 1e-10) {
        return 0; // Matrix is essentially zero
      }

      // Rayleigh quotient: lambda = v^T * u
      const newEigenvalue = TensorOps.dot(v, vOffset, u, uOffset, size);

      // Normalize u -> v
      for (let i = 0; i < size; i++) {
        v[vOffset + i] = u[uOffset + i] / uNorm;
      }

      // Check convergence
      if (
        Math.abs(newEigenvalue - eigenvalue) < SpectralRadiusScaler.TOLERANCE
      ) {
        return Math.abs(newEigenvalue);
      }
      eigenvalue = newEigenvalue;
    }

    return Math.abs(eigenvalue);
  }

  /**
   * Scale W so that spectral radius equals target
   */
  static scale(
    W: Float64Array,
    wOffset: number,
    size: number,
    currentRadius: number,
    targetRadius: number,
  ): void {
    if (currentRadius < 1e-10) return;
    const scaleFactor = targetRadius / currentRadius;
    const end = wOffset + size * size;
    for (let i = wOffset; i < end; i++) {
      W[i] *= scaleFactor;
    }
  }
}

// ============================================================================
// ESN RESERVOIR PARAMS
// ============================================================================

interface ESNReservoirParams {
  Win: Float64Array; // Input weights [reservoirSize x nFeatures]
  W: Float64Array; // Reservoir weights [reservoirSize x reservoirSize]
  b: Float64Array; // Bias [reservoirSize]
  WinMask: Uint8Array; // Input sparsity mask
  WMask: Uint8Array; // Reservoir sparsity mask
}

// ============================================================================
// ESN RESERVOIR
// ============================================================================

class ESNReservoir {
  private reservoirSize: number;
  private nFeatures: number;
  private leakRate: number;
  private activation: "tanh" | "relu";
  private inputScale: number;

  // Weights (fixed after init)
  private Win: Float64Array;
  private W: Float64Array;
  private b: Float64Array;
  private WinMask: Uint8Array;
  private WMask: Uint8Array;

  // State
  private state: Float64Array;

  // Scratch buffers (preallocated)
  private preActivation: Float64Array;
  private inputContrib: Float64Array;
  private recurrentContrib: Float64Array;

  constructor(
    reservoirSize: number,
    nFeatures: number,
    config: ESNRegressionConfig,
    rng: RandomGenerator,
  ) {
    this.reservoirSize = reservoirSize;
    this.nFeatures = nFeatures;
    this.leakRate = config.leakRate;
    this.activation = config.activation;
    this.inputScale = config.inputScale;

    // Allocate weights
    this.Win = new Float64Array(reservoirSize * nFeatures);
    this.W = new Float64Array(reservoirSize * reservoirSize);
    this.b = new Float64Array(reservoirSize);

    // Generate masks
    this.WinMask = ReservoirInitMask.generateInputMask(
      reservoirSize,
      nFeatures,
      config.inputSparsity,
      rng,
    );
    this.WMask = ReservoirInitMask.generateReservoirMask(
      reservoirSize,
      config.reservoirSparsity,
      rng,
    );

    // Initialize Win with random values
    for (let i = 0; i < this.Win.length; i++) {
      if (this.WinMask[i]) {
        this.Win[i] = rng.normal(0, config.weightInitScale);
      }
    }

    // Initialize W with random values
    for (let i = 0; i < this.W.length; i++) {
      if (this.WMask[i]) {
        this.W[i] = rng.normal(0, config.weightInitScale);
      }
    }

    // Initialize bias
    for (let i = 0; i < reservoirSize; i++) {
      this.b[i] = rng.normal(0, config.biasScale);
    }

    // Scale W to achieve target spectral radius
    const scratch = new Float64Array(reservoirSize * 2);
    const currentRadius = SpectralRadiusScaler.estimate(
      this.W,
      0,
      reservoirSize,
      this.WMask,
      0,
      rng,
      scratch,
      0,
    );
    SpectralRadiusScaler.scale(
      this.W,
      0,
      reservoirSize,
      currentRadius,
      config.spectralRadius,
    );

    // Initialize state
    this.state = new Float64Array(reservoirSize);

    // Allocate scratch buffers
    this.preActivation = new Float64Array(reservoirSize);
    this.inputContrib = new Float64Array(reservoirSize);
    this.recurrentContrib = new Float64Array(reservoirSize);
  }

  /**
   * Update reservoir state with new input
   * r_t = (1 - leakRate) * r_{t-1} + leakRate * activation( Win*x + W*r_{t-1} + b )
   */
  update(x: Float64Array, xOffset: number): void {
    const rs = this.reservoirSize;

    // Compute Win * (inputScale * x)
    for (let i = 0; i < rs; i++) {
      let sum = 0;
      const rowOffset = i * this.nFeatures;
      for (let j = 0; j < this.nFeatures; j++) {
        if (this.WinMask[rowOffset + j]) {
          sum += this.Win[rowOffset + j] * this.inputScale * x[xOffset + j];
        }
      }
      this.inputContrib[i] = sum;
    }

    // Compute W * r_{t-1}
    TensorOps.sparseMatVec(
      this.W,
      0,
      rs,
      rs,
      this.WMask,
      0,
      this.state,
      0,
      this.recurrentContrib,
      0,
    );

    // Compute pre-activation: Win*x + W*r + b
    for (let i = 0; i < rs; i++) {
      this.preActivation[i] = this.inputContrib[i] + this.recurrentContrib[i] +
        this.b[i];
    }

    // Apply activation
    if (this.activation === "tanh") {
      for (let i = 0; i < rs; i++) {
        this.preActivation[i] = Math.tanh(this.preActivation[i]);
      }
    } else {
      for (let i = 0; i < rs; i++) {
        this.preActivation[i] = this.preActivation[i] > 0
          ? this.preActivation[i]
          : 0;
      }
    }

    // Leaky integration
    const oneMinusLeak = 1.0 - this.leakRate;
    for (let i = 0; i < rs; i++) {
      this.state[i] = oneMinusLeak * this.state[i] +
        this.leakRate * this.preActivation[i];
    }
  }

  /**
   * Get current state (zero-copy view)
   */
  getState(): Float64Array {
    return this.state;
  }

  /**
   * Copy state to output
   */
  copyStateTo(out: Float64Array, outOffset: number): void {
    for (let i = 0; i < this.reservoirSize; i++) {
      out[outOffset + i] = this.state[i];
    }
  }

  /**
   * Set state from source
   */
  setStateFrom(src: Float64Array, srcOffset: number): void {
    for (let i = 0; i < this.reservoirSize; i++) {
      this.state[i] = src[srcOffset + i];
    }
  }

  /**
   * Reset state to zeros
   */
  resetState(): void {
    this.state.fill(0);
  }

  getReservoirSize(): number {
    return this.reservoirSize;
  }

  getNFeatures(): number {
    return this.nFeatures;
  }

  // Serialization
  getParams(): ESNReservoirParams {
    return {
      Win: this.Win,
      W: this.W,
      b: this.b,
      WinMask: this.WinMask,
      WMask: this.WMask,
    };
  }

  serialize(): {
    state: number[];
    Win: number[];
    W: number[];
    b: number[];
    WinMask: number[];
    WMask: number[];
  } {
    return {
      state: Array.from(this.state),
      Win: Array.from(this.Win),
      W: Array.from(this.W),
      b: Array.from(this.b),
      WinMask: Array.from(this.WinMask),
      WMask: Array.from(this.WMask),
    };
  }

  deserialize(data: {
    state: number[];
    Win: number[];
    W: number[];
    b: number[];
    WinMask: number[];
    WMask: number[];
  }): void {
    for (let i = 0; i < this.state.length; i++) {
      this.state[i] = data.state[i] || 0;
    }
    for (let i = 0; i < this.Win.length; i++) {
      this.Win[i] = data.Win[i] || 0;
    }
    for (let i = 0; i < this.W.length; i++) {
      this.W[i] = data.W[i] || 0;
    }
    for (let i = 0; i < this.b.length; i++) {
      this.b[i] = data.b[i] || 0;
    }
    for (let i = 0; i < this.WinMask.length; i++) {
      this.WinMask[i] = data.WinMask[i] || 0;
    }
    for (let i = 0; i < this.WMask.length; i++) {
      this.WMask[i] = data.WMask[i] || 0;
    }
  }
}

// ============================================================================
// READOUT CONFIG
// ============================================================================

interface ReadoutConfig {
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
}

// ============================================================================
// RLS STATE
// ============================================================================

interface RLSState {
  P: Float64Array; // Inverse covariance matrix [zDim x zDim]
  gain: Float64Array; // Gain vector [zDim]
  temp: Float64Array; // Temporary vector [zDim]
}

// ============================================================================
// RLS OPTIMIZER (Recursive Least Squares)
// ============================================================================

class RLSOptimizer {
  private zDim: number;
  private nTargets: number;
  private lambda: number;
  private delta: number;
  private l2Lambda: number;
  private epsilon: number;

  // State
  private P: Float64Array;
  private gain: Float64Array;
  private temp: Float64Array;
  private Pz: Float64Array;

  // Readout weights [nTargets x zDim]
  private Wout: Float64Array;

  constructor(
    zDim: number,
    nTargets: number,
    lambda: number,
    delta: number,
    l2Lambda: number,
    epsilon: number,
  ) {
    this.zDim = zDim;
    this.nTargets = nTargets;
    this.lambda = lambda;
    this.delta = delta;
    this.l2Lambda = l2Lambda;
    this.epsilon = epsilon;

    // Initialize P = delta * I
    this.P = new Float64Array(zDim * zDim);
    for (let i = 0; i < zDim; i++) {
      this.P[i * zDim + i] = delta;
    }

    // Scratch vectors
    this.gain = new Float64Array(zDim);
    this.temp = new Float64Array(zDim);
    this.Pz = new Float64Array(zDim);

    // Initialize Wout to zeros
    this.Wout = new Float64Array(nTargets * zDim);
  }

  /**
   * Perform RLS update
   * @param z Extended state vector [zDim]
   * @param y Target vector [nTargets]
   * @param weight Sample weight for outlier handling
   * @returns Gradient norm (approximation)
   */
  update(
    z: Float64Array,
    zOffset: number,
    y: Float64Array,
    yOffset: number,
    weight: number,
  ): number {
    const zDim = this.zDim;
    const nTargets = this.nTargets;

    // Compute Pz = P * z
    TensorOps.matVec(this.P, 0, zDim, zDim, z, zOffset, this.Pz, 0);

    // Compute denominator: lambda + z^T * P * z
    let denom = this.lambda;
    for (let i = 0; i < zDim; i++) {
      denom += z[zOffset + i] * this.Pz[i];
    }
    denom = Math.max(denom, this.epsilon);

    // Compute gain: k = Pz / denom
    for (let i = 0; i < zDim; i++) {
      this.gain[i] = this.Pz[i] / denom;
    }

    // Apply weight to gain
    for (let i = 0; i < zDim; i++) {
      this.gain[i] *= weight;
    }

    // Compute prediction errors and update Wout
    let gradNormSq = 0;
    for (let t = 0; t < nTargets; t++) {
      // Compute prediction: y_hat = Wout[t,:] * z
      let yHat = 0;
      const rowOffset = t * zDim;
      for (let i = 0; i < zDim; i++) {
        yHat += this.Wout[rowOffset + i] * z[zOffset + i];
      }

      // Compute error
      const error = y[yOffset + t] - yHat;
      gradNormSq += error * error;

      // Update Wout[t,:] += gain * error
      for (let i = 0; i < zDim; i++) {
        this.Wout[rowOffset + i] += this.gain[i] * error;
      }
    }

    // Update P: P = (P - k * z^T * P) / lambda
    // Simplified: P = (1/lambda) * (P - gain * Pz^T)
    const invLambda = 1.0 / this.lambda;
    for (let i = 0; i < zDim; i++) {
      const rowOffset = i * zDim;
      for (let j = 0; j < zDim; j++) {
        this.P[rowOffset + j] = invLambda *
          (this.P[rowOffset + j] - this.gain[i] * this.Pz[j]);
      }
    }

    // Apply L2 regularization to Wout (shrinkage)
    if (this.l2Lambda > 0) {
      const shrink = 1.0 - this.l2Lambda;
      for (let i = 0; i < this.Wout.length; i++) {
        this.Wout[i] *= shrink;
      }
    }

    return Math.sqrt(gradNormSq / nTargets);
  }

  /**
   * Predict: y = Wout * z
   */
  predict(
    z: Float64Array,
    zOffset: number,
    out: Float64Array,
    outOffset: number,
  ): void {
    TensorOps.matVec(
      this.Wout,
      0,
      this.nTargets,
      this.zDim,
      z,
      zOffset,
      out,
      outOffset,
    );
  }

  getWout(): Float64Array {
    return this.Wout;
  }

  getZDim(): number {
    return this.zDim;
  }

  getNTargets(): number {
    return this.nTargets;
  }

  // Serialization
  serialize(): {
    P: number[];
    Wout: number[];
  } {
    return {
      P: Array.from(this.P),
      Wout: Array.from(this.Wout),
    };
  }

  deserialize(data: { P: number[]; Wout: number[] }): void {
    for (let i = 0; i < this.P.length; i++) {
      this.P[i] = data.P[i] || 0;
    }
    for (let i = 0; i < this.Wout.length; i++) {
      this.Wout[i] = data.Wout[i] || 0;
    }
  }

  reset(): void {
    this.P.fill(0);
    for (let i = 0; i < this.zDim; i++) {
      this.P[i * this.zDim + i] = this.delta;
    }
    this.Wout.fill(0);
  }
}

// ============================================================================
// LINEAR READOUT
// ============================================================================

class LinearReadout {
  private reservoirSize: number;
  private nFeatures: number;
  private nTargets: number;
  private useInputInReadout: boolean;
  private useBiasInReadout: boolean;
  private zDim: number;

  private rls: RLSOptimizer;

  // Scratch buffers
  private z: Float64Array;
  private prediction: Float64Array;

  constructor(
    reservoirSize: number,
    nFeatures: number,
    nTargets: number,
    config: ESNRegressionConfig,
  ) {
    this.reservoirSize = reservoirSize;
    this.nFeatures = nFeatures;
    this.nTargets = nTargets;
    this.useInputInReadout = config.useInputInReadout;
    this.useBiasInReadout = config.useBiasInReadout;

    // Compute zDim = reservoirSize + (useInput ? nFeatures : 0) + (useBias ? 1 : 0)
    this.zDim = reservoirSize;
    if (this.useInputInReadout) this.zDim += nFeatures;
    if (this.useBiasInReadout) this.zDim += 1;

    // Initialize RLS optimizer
    this.rls = new RLSOptimizer(
      this.zDim,
      nTargets,
      config.rlsLambda,
      config.rlsDelta,
      config.l2Lambda,
      config.epsilon,
    );

    // Allocate scratch
    this.z = new Float64Array(this.zDim);
    this.prediction = new Float64Array(nTargets);
  }

  /**
   * Build extended state z = [r; x; 1]
   */
  buildZ(
    reservoirState: Float64Array,
    rsOffset: number,
    input: Float64Array,
    inputOffset: number,
    out: Float64Array,
    outOffset: number,
  ): void {
    let idx = outOffset;

    // Copy reservoir state
    for (let i = 0; i < this.reservoirSize; i++) {
      out[idx++] = reservoirState[rsOffset + i];
    }

    // Optionally append input
    if (this.useInputInReadout) {
      for (let i = 0; i < this.nFeatures; i++) {
        out[idx++] = input[inputOffset + i];
      }
    }

    // Optionally append bias
    if (this.useBiasInReadout) {
      out[idx] = 1.0;
    }
  }

  /**
   * Forward pass (prediction)
   */
  forward(
    reservoirState: Float64Array,
    rsOffset: number,
    input: Float64Array,
    inputOffset: number,
    out: Float64Array,
    outOffset: number,
  ): void {
    this.buildZ(reservoirState, rsOffset, input, inputOffset, this.z, 0);
    this.rls.predict(this.z, 0, out, outOffset);
  }

  /**
   * Training step
   */
  train(
    reservoirState: Float64Array,
    rsOffset: number,
    input: Float64Array,
    inputOffset: number,
    target: Float64Array,
    targetOffset: number,
    weight: number,
  ): number {
    this.buildZ(reservoirState, rsOffset, input, inputOffset, this.z, 0);
    return this.rls.update(this.z, 0, target, targetOffset, weight);
  }

  getZDim(): number {
    return this.zDim;
  }

  getNTargets(): number {
    return this.nTargets;
  }

  getWout(): Float64Array {
    return this.rls.getWout();
  }

  getRLS(): RLSOptimizer {
    return this.rls;
  }

  // Serialization
  serialize(): { rls: { P: number[]; Wout: number[] } } {
    return { rls: this.rls.serialize() };
  }

  deserialize(data: { rls: { P: number[]; Wout: number[] } }): void {
    this.rls.deserialize(data.rls);
  }

  reset(): void {
    this.rls.reset();
  }
}

// ============================================================================
// SERIALIZATION HELPER
// ============================================================================

class SerializationHelper {
  static serialize(model: ESNRegression): string {
    return JSON.stringify(model.getSerializableState());
  }

  static deserialize(model: ESNRegression, json: string): void {
    const state = JSON.parse(json);
    model.setSerializableState(state);
  }
}

// ============================================================================
// ESN REGRESSION (Main class)
// ============================================================================

/**
 * Echo State Network for Multivariate Regression with Online Learning
 *
 * @example
 * ```typescript
 * const model = new ESNRegression({ reservoirSize: 128 });
 *
 * const xTrain = [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]];
 * const yTrain = [[10, 11], [15, 16], [20, 21]];
 *
 * const fitResult = model.fitOnline({ xCoordinates: xTrain, yCoordinates: yTrain });
 * const prediction = model.predict(3);
 * ```
 */
export class ESNRegression {
  private config: ESNRegressionConfig;
  private initialized: boolean = false;
  private nFeatures: number = 0;
  private nTargets: number = 0;
  private sampleCount: number = 0;

  // Components (created on first fitOnline)
  private reservoir: ESNReservoir | null = null;
  private readout: LinearReadout | null = null;
  private normalizer: WelfordNormalizer | null = null;
  private ringBuffer: RingBuffer | null = null;
  private residualStats: ResidualStatsTracker | null = null;
  private outlierDownweighter: OutlierDownweighter | null = null;
  private metricsAccumulator: MetricsAccumulator | null = null;
  private rng: RandomGenerator;

  // Scratch buffers (preallocated after init)
  private scratchX: Float64Array | null = null;
  private scratchXNorm: Float64Array | null = null;
  private scratchY: Float64Array | null = null;
  private scratchPred: Float64Array | null = null;
  private scratchResiduals: Float64Array | null = null;
  private scratchStds: Float64Array | null = null;
  private scratchReservoirState: Float64Array | null = null;

  // Reusable result objects
  private fitResultObj: FitResult;
  private predictionResultObj: PredictionResult;

  constructor(config: Partial<ESNRegressionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.rng = new RandomGenerator(this.config.seed);

    // Initialize reusable result objects
    this.fitResultObj = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1.0,
    };

    // Initialize prediction result with max capacity
    const maxSteps = this.config.maxSequenceLength;
    this.predictionResultObj = {
      predictions: [],
      lowerBounds: [],
      upperBounds: [],
      confidence: 0,
    };
    // Pre-allocate arrays (will be resized/filled as needed)
    for (let i = 0; i < maxSteps; i++) {
      this.predictionResultObj.predictions.push([]);
      this.predictionResultObj.lowerBounds.push([]);
      this.predictionResultObj.upperBounds.push([]);
    }
  }

  /**
   * Initialize model components on first data
   */
  private initialize(nFeatures: number, nTargets: number): void {
    this.nFeatures = nFeatures;
    this.nTargets = nTargets;

    // Create reservoir
    this.reservoir = new ESNReservoir(
      this.config.reservoirSize,
      nFeatures,
      this.config,
      this.rng,
    );

    // Create readout
    this.readout = new LinearReadout(
      this.config.reservoirSize,
      nFeatures,
      nTargets,
      this.config,
    );

    // Create normalizer
    this.normalizer = new WelfordNormalizer(
      nFeatures,
      this.config.normalizationEpsilon,
      this.config.normalizationWarmup,
    );

    // Create ring buffer
    this.ringBuffer = new RingBuffer(this.config.maxSequenceLength, nFeatures);

    // Create residual stats tracker
    this.residualStats = new ResidualStatsTracker(
      this.config.residualWindowSize,
      nTargets,
    );

    // Create outlier downweighter
    this.outlierDownweighter = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );

    // Create metrics accumulator
    this.metricsAccumulator = new MetricsAccumulator();

    // Allocate scratch buffers
    this.scratchX = new Float64Array(nFeatures);
    this.scratchXNorm = new Float64Array(nFeatures);
    this.scratchY = new Float64Array(nTargets);
    this.scratchPred = new Float64Array(nTargets);
    this.scratchResiduals = new Float64Array(nTargets);
    this.scratchStds = new Float64Array(nTargets);
    this.scratchReservoirState = new Float64Array(this.config.reservoirSize);

    // Resize prediction result arrays for nTargets
    for (let i = 0; i < this.config.maxSequenceLength; i++) {
      this.predictionResultObj.predictions[i] = new Array(nTargets).fill(0);
      this.predictionResultObj.lowerBounds[i] = new Array(nTargets).fill(0);
      this.predictionResultObj.upperBounds[i] = new Array(nTargets).fill(0);
    }

    this.initialized = true;
  }

  /**
   * Online training with new samples
   *
   * @param xCoordinates Input features [N x nFeatures]
   * @param yCoordinates Target values [N x nTargets]
   * @returns FitResult with training metrics
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1.0, 2.0], [1.5, 2.5]],
   *   yCoordinates: [[10, 11], [15, 16]]
   * });
   * console.log(result.averageLoss);
   * ```
   */
  fitOnline({ xCoordinates, yCoordinates }: {
    xCoordinates: number[][];
    yCoordinates: number[][];
  }): FitResult {
    // CRITICAL: Validate lengths match BEFORE any ingestion
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `fitOnline: xCoordinates.length (${xCoordinates.length}) must equal ` +
          `yCoordinates.length (${yCoordinates.length})`,
      );
    }

    const N = xCoordinates.length;
    if (N === 0) {
      this.fitResultObj.samplesProcessed = 0;
      this.fitResultObj.averageLoss = 0;
      this.fitResultObj.gradientNorm = 0;
      this.fitResultObj.driftDetected = false;
      this.fitResultObj.sampleWeight = 1.0;
      return this.fitResultObj;
    }

    // Initialize on first non-empty call
    if (!this.initialized) {
      const nFeatures = xCoordinates[0].length;
      const nTargets = yCoordinates[0].length;
      if (nFeatures === 0) throw new Error("fitOnline: nFeatures must be > 0");
      if (nTargets === 0) throw new Error("fitOnline: nTargets must be > 0");
      this.initialize(nFeatures, nTargets);
    }

    // Validate all rows have correct dimensions
    for (let i = 0; i < N; i++) {
      if (xCoordinates[i].length !== this.nFeatures) {
        throw new Error(
          `fitOnline: xCoordinates[${i}].length (${xCoordinates[i].length}) ` +
            `must equal nFeatures (${this.nFeatures})`,
        );
      }
      if (yCoordinates[i].length !== this.nTargets) {
        throw new Error(
          `fitOnline: yCoordinates[${i}].length (${yCoordinates[i].length}) ` +
            `must equal nTargets (${this.nTargets})`,
        );
      }
    }

    // Reset metrics for this batch
    this.metricsAccumulator!.reset();

    // Process each sample
    for (let i = 0; i < N; i++) {
      // 1. FIRST: Push x into RingBuffer (Critical: must happen before any training)
      for (let j = 0; j < this.nFeatures; j++) {
        this.scratchX![j] = xCoordinates[i][j];
      }
      this.ringBuffer!.push(this.scratchX!, 0);

      // 2. Update normalization stats
      this.normalizer!.update(this.scratchX!, 0);

      // 3. Normalize x
      this.normalizer!.normalize(this.scratchX!, 0, this.scratchXNorm!, 0);

      // 4. Update reservoir state
      this.reservoir!.update(this.scratchXNorm!, 0);

      // 5. Copy target
      for (let j = 0; j < this.nTargets; j++) {
        this.scratchY![j] = yCoordinates[i][j];
      }

      // 6. Compute prediction
      this.readout!.forward(
        this.reservoir!.getState(),
        0,
        this.scratchXNorm!,
        0,
        this.scratchPred!,
        0,
      );

      // 7. Compute residuals
      for (let j = 0; j < this.nTargets; j++) {
        this.scratchResiduals![j] = this.scratchPred![j] - this.scratchY![j];
      }

      // 8. Get residual stds for outlier detection
      this.residualStats!.getStds(this.scratchStds!, 0);

      // 9. Compute sample weight
      const weight = this.outlierDownweighter!.computeWeight(
        this.scratchResiduals!,
        0,
        this.scratchStds!,
        0,
        this.nTargets,
      );

      // 10. Perform RLS update
      const gradNorm = this.readout!.train(
        this.reservoir!.getState(),
        0,
        this.scratchXNorm!,
        0,
        this.scratchY!,
        0,
        weight,
      );

      // 11. Compute loss
      let loss = 0;
      for (let j = 0; j < this.nTargets; j++) {
        loss += this.scratchResiduals![j] * this.scratchResiduals![j];
      }
      loss /= this.nTargets;

      // 12. Update residual stats (after computing loss)
      this.residualStats!.update(this.scratchResiduals!, 0);

      // 13. Update metrics
      this.metricsAccumulator!.update(loss, gradNorm, weight);

      this.sampleCount++;
    }

    // Fill result object (reused)
    this.fitResultObj.samplesProcessed = N;
    this.fitResultObj.averageLoss = this.metricsAccumulator!.getAverageLoss();
    this.fitResultObj.gradientNorm = this.metricsAccumulator!
      .getAverageGradNorm();
    this.fitResultObj.driftDetected = false; // ADWIN not implemented
    this.fitResultObj.sampleWeight = this.metricsAccumulator!.getLastWeight();

    return this.fitResultObj;
  }

  /**
   * Multi-step prediction
   *
   * @param futureSteps Number of steps to predict (1 to maxSequenceLength)
   * @returns PredictionResult with predictions and uncertainty bounds
   *
   * @example
   * ```typescript
   * const pred = model.predict(3);
   * console.log(pred.predictions[0]); // 1-step ahead
   * console.log(pred.predictions[2]); // 3-step ahead
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    // Validate model is initialized
    if (
      !this.initialized || !this.reservoir || !this.readout || !this.ringBuffer
    ) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }

    // Validate futureSteps
    if (!Number.isInteger(futureSteps) || futureSteps < 1) {
      throw new Error("predict: futureSteps must be a positive integer >= 1");
    }
    if (futureSteps > this.config.maxSequenceLength) {
      throw new Error(
        `predict: futureSteps (${futureSteps}) exceeds maxSequenceLength (${this.config.maxSequenceLength})`,
      );
    }

    // Check if we have data
    if (this.ringBuffer.isEmpty()) {
      throw new Error("predict: model not initialized (call fitOnline first)");
    }

    // Get latest x from RingBuffer
    this.ringBuffer.getLatest(this.scratchX!, 0);

    // Copy current reservoir state to scratch
    this.reservoir.copyStateTo(this.scratchReservoirState!, 0);

    // Get base residual stds for uncertainty
    this.residualStats!.getStds(this.scratchStds!, 0);

    // Roll forward for each step
    for (let step = 0; step < futureSteps; step++) {
      // Normalize current x
      this.normalizer!.normalize(this.scratchX!, 0, this.scratchXNorm!, 0);

      // Update scratch reservoir state
      this.updateScratchReservoir(this.scratchXNorm!, 0);

      // Compute prediction
      this.readout.forward(
        this.scratchReservoirState!,
        0,
        this.scratchXNorm!,
        0,
        this.scratchPred!,
        0,
      );

      // Store predictions with proper array bounds
      for (let t = 0; t < this.nTargets; t++) {
        this.predictionResultObj.predictions[step][t] = this.scratchPred![t];

        // Compute uncertainty bounds with horizon scaling
        // sigma_k = sigma_1 * sqrt(step + 1)
        const horizonScale = Math.sqrt(step + 1);
        const sigma = this.scratchStds![t] * horizonScale;
        const halfWidth = this.config.uncertaintyMultiplier * sigma;

        this.predictionResultObj.lowerBounds[step][t] = this.scratchPred![t] -
          halfWidth;
        this.predictionResultObj.upperBounds[step][t] = this.scratchPred![t] +
          halfWidth;
      }

      // Prepare x for next step based on rollforward mode
      if (
        this.config.rollforwardMode === "autoregressive" &&
        this.nFeatures === this.nTargets
      ) {
        // Use prediction as next input
        for (let j = 0; j < this.nFeatures; j++) {
          this.scratchX![j] = this.scratchPred![j];
        }
      }
      // else: holdLastX mode - scratchX remains unchanged
    }

    // Compute confidence (based on residual stats and prediction horizon)
    let avgStd = 0;
    for (let t = 0; t < this.nTargets; t++) {
      avgStd += this.scratchStds![t];
    }
    avgStd /= this.nTargets;

    // Confidence decreases with std and horizon
    // Base confidence from inverse of normalized uncertainty
    const baseConfidence = avgStd > 0 ? 1.0 / (1.0 + avgStd) : 1.0;
    // Apply horizon penalty
    const horizonPenalty = 1.0 / Math.sqrt(futureSteps);
    this.predictionResultObj.confidence = Math.max(
      0,
      Math.min(1, baseConfidence * horizonPenalty),
    );

    return this.predictionResultObj;
  }

  /**
   * Update scratch reservoir state (helper for roll-forward)
   * Uses the same dynamics as main reservoir but on scratch state
   */
  private updateScratchReservoir(xNorm: Float64Array, xOffset: number): void {
    const params = this.reservoir!.getParams();
    const rs = this.config.reservoirSize;
    const leakRate = this.config.leakRate;

    // Temporary storage (reuse scratchPred as temp since we overwrite it anyway)
    const preAct = new Float64Array(rs);
    const inputContrib = new Float64Array(rs);
    const recurrentContrib = new Float64Array(rs);

    // Compute Win * (inputScale * x)
    for (let i = 0; i < rs; i++) {
      let sum = 0;
      const rowOffset = i * this.nFeatures;
      for (let j = 0; j < this.nFeatures; j++) {
        if (params.WinMask[rowOffset + j]) {
          sum += params.Win[rowOffset + j] * this.config.inputScale *
            xNorm[xOffset + j];
        }
      }
      inputContrib[i] = sum;
    }

    // Compute W * r
    TensorOps.sparseMatVec(
      params.W,
      0,
      rs,
      rs,
      params.WMask,
      0,
      this.scratchReservoirState!,
      0,
      recurrentContrib,
      0,
    );

    // Compute pre-activation
    for (let i = 0; i < rs; i++) {
      preAct[i] = inputContrib[i] + recurrentContrib[i] + params.b[i];
    }

    // Apply activation
    if (this.config.activation === "tanh") {
      for (let i = 0; i < rs; i++) {
        preAct[i] = Math.tanh(preAct[i]);
      }
    } else {
      for (let i = 0; i < rs; i++) {
        preAct[i] = preAct[i] > 0 ? preAct[i] : 0;
      }
    }

    // Leaky integration
    const oneMinusLeak = 1.0 - leakRate;
    for (let i = 0; i < rs; i++) {
      this.scratchReservoirState![i] =
        oneMinusLeak * this.scratchReservoirState![i] + leakRate * preAct[i];
    }
  }

  /**
   * Get model summary
   */
  getModelSummary(): ModelSummary {
    const zDim = this.readout ? this.readout.getZDim() : 0;
    const totalParams = this.nTargets * zDim; // Only Wout is trainable

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
   * Get weight information
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
      const params = this.reservoir.getParams();
      weights.push({
        name: "Win",
        shape: [this.config.reservoirSize, this.nFeatures],
        values: Array.from(params.Win),
      });
      weights.push({
        name: "W",
        shape: [this.config.reservoirSize, this.config.reservoirSize],
        values: Array.from(params.W),
      });
      weights.push({
        name: "b",
        shape: [this.config.reservoirSize],
        values: Array.from(params.b),
      });
    }

    return { weights };
  }

  /**
   * Get normalization statistics
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
    if (this.reservoir) this.reservoir.resetState();
    if (this.readout) this.readout.reset();
    if (this.normalizer) this.normalizer.reset();
    if (this.ringBuffer) this.ringBuffer.reset();
    if (this.residualStats) this.residualStats.reset();
    if (this.metricsAccumulator) this.metricsAccumulator.reset();
    this.sampleCount = 0;
    this.rng = new RandomGenerator(this.config.seed);
  }

  /**
   * Save model state to JSON string
   */
  save(): string {
    return SerializationHelper.serialize(this);
  }

  /**
   * Load model state from JSON string
   */
  load(json: string): void {
    SerializationHelper.deserialize(this, json);
  }

  /**
   * Get serializable state (internal use)
   */
  getSerializableState(): object {
    return {
      config: this.config,
      initialized: this.initialized,
      nFeatures: this.nFeatures,
      nTargets: this.nTargets,
      sampleCount: this.sampleCount,
      rngState: this.rng.getState(),
      reservoir: this.reservoir ? this.reservoir.serialize() : null,
      readout: this.readout ? this.readout.serialize() : null,
      normalizer: this.normalizer ? this.normalizer.serialize() : null,
      ringBuffer: this.ringBuffer ? this.ringBuffer.serialize() : null,
      residualStats: this.residualStats ? this.residualStats.serialize() : null,
    };
  }

  /**
   * Set serializable state (internal use)
   */
  setSerializableState(state: any): void {
    this.config = { ...DEFAULT_CONFIG, ...state.config };
    this.initialized = state.initialized;
    this.nFeatures = state.nFeatures;
    this.nTargets = state.nTargets;
    this.sampleCount = state.sampleCount;
    this.rng.setState(state.rngState);

    if (state.initialized && state.nFeatures > 0 && state.nTargets > 0) {
      // Re-initialize components with correct dimensions
      this.initialize(state.nFeatures, state.nTargets);

      // Restore state
      if (state.reservoir) this.reservoir!.deserialize(state.reservoir);
      if (state.readout) this.readout!.deserialize(state.readout);
      if (state.normalizer) this.normalizer!.deserialize(state.normalizer);
      if (state.ringBuffer) this.ringBuffer!.deserialize(state.ringBuffer);
      if (state.residualStats) {
        this.residualStats!.deserialize(state.residualStats);
      }
    }
  }
}

// Export default
export default ESNRegression;
