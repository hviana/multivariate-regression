/**
 * ESNRegression: Echo State Network for Multivariate Regression
 * with Incremental Online Learning using RLS Readout Training
 * and Welford Z-Score Normalization
 *
 * @module ESNRegression
 */

// ============================================================
// RESULT TYPES AND INTERFACES
// ============================================================

/**
 * Result of fitOnline() call.
 */
export interface FitResult {
  /** Number of samples processed */
  samplesProcessed: number;
  /** Average loss over processed samples */
  averageLoss: number;
  /** Final readout update norm */
  gradientNorm: number;
  /** Whether concept drift was detected */
  driftDetected: boolean;
  /** Current sample weight (outlier-adjusted) */
  sampleWeight: number;
}

/**
 * Result of predict() call.
 */
export interface PredictionResult {
  /** Predictions for each future step: [step][target] */
  predictions: number[][];
  /** Lower confidence bounds: [step][target] */
  lowerBounds: number[][];
  /** Upper confidence bounds: [step][target] */
  upperBounds: number[][];
  /** Confidence score based on recent loss */
  confidence: number;
}

/**
 * Model summary information.
 */
export interface ModelSummary {
  /** Total number of trainable parameters */
  totalParameters: number;
  /** Effective memory horizon in timesteps (approx from config) */
  receptiveField: number;
  /** Reservoir spectral radius */
  spectralRadius: number;
  /** Reservoir size */
  reservoirSize: number;
  /** Input features */
  nFeatures: number;
  /** Output targets */
  nTargets: number;
  /** Maximum sequence length */
  maxSequenceLength: number;
  /** Maximum future steps */
  maxFutureSteps: number;
  /** Training samples seen */
  sampleCount: number;
  /** Whether direct multi-horizon is enabled */
  useDirectMultiHorizon: boolean;
}

/**
 * Weight information for inspection.
 */
export interface WeightInfo {
  /** Named weight tensors */
  weights: Array<{ name: string; shape: number[]; values: number[] }>;
}

/**
 * Normalization statistics.
 */
export interface NormalizationStats {
  /** Feature-wise means */
  means: number[];
  /** Feature-wise standard deviations */
  stds: number[];
  /** Number of samples used */
  count: number;
  /** Whether normalization is active (warmup complete) */
  isActive: boolean;
}

// ============================================================
// CONFIGURATION
// ============================================================

/**
 * Configuration for ESN Regression model.
 */
export interface ESNRegressionConfig {
  maxSequenceLength?: number;
  maxFutureSteps?: number;
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
  useDirectMultiHorizon?: boolean;
  residualWindowSize?: number;
  uncertaintyMultiplier?: number;
  weightInitScale?: number;
  seed?: number;
  verbose?: boolean;
}

const DEFAULT_CONFIG: Required<ESNRegressionConfig> = {
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

// ============================================================
// TENSOR INFRASTRUCTURE
// ============================================================

/**
 * Represents the shape of a tensor with row-major layout.
 * Immutable after construction.
 */
export class TensorShape {
  private readonly _dims: number[];
  private readonly _size: number;
  private readonly _strides: number[];

  constructor(dims: number[]) {
    this._dims = dims.slice();
    this._size = dims.length === 0 ? 0 : dims.reduce((a, b) => a * b, 1);

    // Compute row-major strides
    this._strides = new Array(dims.length);
    let stride = 1;
    for (let i = dims.length - 1; i >= 0; i--) {
      this._strides[i] = stride;
      stride *= dims[i];
    }
  }

  get dims(): readonly number[] {
    return this._dims;
  }

  get size(): number {
    return this._size;
  }

  get strides(): readonly number[] {
    return this._strides;
  }

  get rank(): number {
    return this._dims.length;
  }

  /**
   * Check shape equality.
   */
  equals(other: TensorShape): boolean {
    if (this._dims.length !== other._dims.length) return false;
    for (let i = 0; i < this._dims.length; i++) {
      if (this._dims[i] !== other._dims[i]) return false;
    }
    return true;
  }

  /**
   * Get dimension at index.
   */
  dim(index: number): number {
    return this._dims[index];
  }

  /**
   * Get stride at index.
   */
  stride(index: number): number {
    return this._strides[index];
  }
}

/**
 * A zero-copy view into a contiguous Float64Array buffer.
 * No allocations on access methods.
 */
export class TensorView {
  readonly data: Float64Array;
  readonly offset: number;
  readonly shape: TensorShape;

  constructor(data: Float64Array, offset: number, shape: TensorShape) {
    this.data = data;
    this.offset = offset;
    this.shape = shape;
  }

  /**
   * Get value at multi-dimensional indices.
   */
  get(indices: number[]): number {
    let idx = this.offset;
    const strides = this.shape.strides;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * strides[i];
    }
    return this.data[idx];
  }

  /**
   * Set value at multi-dimensional indices.
   */
  set(indices: number[], value: number): void {
    let idx = this.offset;
    const strides = this.shape.strides;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * strides[i];
    }
    this.data[idx] = value;
  }

  /**
   * Get value at linear index (offset from view start).
   */
  getLinear(index: number): number {
    return this.data[this.offset + index];
  }

  /**
   * Set value at linear index (offset from view start).
   */
  setLinear(index: number, value: number): void {
    this.data[this.offset + index] = value;
  }

  /**
   * Get 2D matrix element (row, col).
   */
  get2D(row: number, col: number): number {
    return this.data[this.offset + row * this.shape.strides[0] + col];
  }

  /**
   * Set 2D matrix element (row, col).
   */
  set2D(row: number, col: number, value: number): void {
    this.data[this.offset + row * this.shape.strides[0] + col] = value;
  }

  /**
   * Fill entire view with a value.
   */
  fill(value: number): void {
    const end = this.offset + this.shape.size;
    for (let i = this.offset; i < end; i++) {
      this.data[i] = value;
    }
  }

  /**
   * Copy data from another view.
   */
  copyFrom(src: TensorView): void {
    const size = this.shape.size;
    const srcData = src.data;
    const srcOffset = src.offset;
    const dstData = this.data;
    const dstOffset = this.offset;
    for (let i = 0; i < size; i++) {
      dstData[dstOffset + i] = srcData[srcOffset + i];
    }
  }

  /**
   * Copy data from a plain array.
   */
  copyFromArray(src: ArrayLike<number>, srcOffset: number = 0): void {
    const size = this.shape.size;
    const dstData = this.data;
    const dstOffset = this.offset;
    for (let i = 0; i < size; i++) {
      dstData[dstOffset + i] = src[srcOffset + i];
    }
  }

  /**
   * Copy data to a plain array.
   */
  copyToArray(dst: number[], dstOffset: number = 0): void {
    const size = this.shape.size;
    for (let i = 0; i < size; i++) {
      dst[dstOffset + i] = this.data[this.offset + i];
    }
  }

  /**
   * Convert to plain array (allocates).
   */
  toArray(): number[] {
    const result = new Array(this.shape.size);
    for (let i = 0; i < this.shape.size; i++) {
      result[i] = this.data[this.offset + i];
    }
    return result;
  }

  /**
   * Get a row slice as a new view (no allocation of underlying data).
   */
  row(index: number): TensorView {
    const rowOffset = this.offset + index * this.shape.strides[0];
    const rowShape = new TensorShape([this.shape.dim(1)]);
    return new TensorView(this.data, rowOffset, rowShape);
  }

  /**
   * Compute sum of all elements.
   */
  sum(): number {
    let s = 0;
    const end = this.offset + this.shape.size;
    for (let i = this.offset; i < end; i++) {
      s += this.data[i];
    }
    return s;
  }

  /**
   * Compute squared L2 norm.
   */
  squaredNorm(): number {
    let s = 0;
    const end = this.offset + this.shape.size;
    for (let i = this.offset; i < end; i++) {
      const v = this.data[i];
      s += v * v;
    }
    return s;
  }
}

/**
 * Size classes for buffer pooling (powers of 2).
 */
const SIZE_CLASSES = [
  16,
  32,
  64,
  128,
  256,
  512,
  1024,
  2048,
  4096,
  8192,
  16384,
  32768,
  65536,
  131072,
  262144,
  524288,
  1048576,
];

/**
 * Pool of reusable Float64Array buffers organized by size class.
 * Prevents hot-path allocations.
 */
export class BufferPool {
  private pools: Map<number, Float64Array[]>;
  private maxPoolSize: number;

  constructor(maxPoolSize: number = 16) {
    this.pools = new Map();
    this.maxPoolSize = maxPoolSize;
    for (const size of SIZE_CLASSES) {
      this.pools.set(size, []);
    }
  }

  /**
   * Get the smallest size class that fits the requested size.
   */
  private getSizeClass(size: number): number {
    for (const sc of SIZE_CLASSES) {
      if (sc >= size) return sc;
    }
    return size; // Larger than any class, use exact size
  }

  /**
   * Rent a buffer of at least the requested size.
   * @param minSize Minimum required size
   * @returns A Float64Array of at least minSize
   */
  rent(minSize: number): Float64Array {
    const sizeClass = this.getSizeClass(minSize);
    const pool = this.pools.get(sizeClass);
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    return new Float64Array(sizeClass);
  }

  /**
   * Return a buffer to the pool.
   * @param buffer The buffer to return
   */
  return(buffer: Float64Array): void {
    const size = buffer.length;
    const pool = this.pools.get(size);
    if (pool && pool.length < this.maxPoolSize) {
      // Zero the buffer before returning (optional, for safety)
      buffer.fill(0);
      pool.push(buffer);
    }
    // If pool is full or size doesn't match a class, let GC handle it
  }

  /**
   * Clear all pools.
   */
  clear(): void {
    for (const pool of this.pools.values()) {
      pool.length = 0;
    }
  }
}

/**
 * Arena allocator for tensor data with fixed capacity.
 * Provides zero-copy views into a single contiguous buffer.
 */
export class TensorArena {
  private buffer: Float64Array;
  private offset: number;
  private readonly capacity: number;
  private views: TensorView[];

  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Float64Array(capacity);
    this.offset = 0;
    this.views = [];
  }

  /**
   * Allocate a tensor view from the arena.
   * @param shape Shape of the tensor to allocate
   * @returns TensorView backed by arena memory
   */
  alloc(shape: TensorShape): TensorView {
    const size = shape.size;
    if (this.offset + size > this.capacity) {
      throw new Error(
        `TensorArena capacity exceeded: need ${size}, have ${
          this.capacity - this.offset
        }`,
      );
    }
    const view = new TensorView(this.buffer, this.offset, shape);
    this.offset += size;
    this.views.push(view);
    return view;
  }

  /**
   * Allocate a 1D vector.
   */
  allocVector(size: number): TensorView {
    return this.alloc(new TensorShape([size]));
  }

  /**
   * Allocate a 2D matrix.
   */
  allocMatrix(rows: number, cols: number): TensorView {
    return this.alloc(new TensorShape([rows, cols]));
  }

  /**
   * Reset the arena for reuse (does not deallocate underlying buffer).
   */
  reset(): void {
    this.offset = 0;
    this.views.length = 0;
  }

  /**
   * Get current usage.
   */
  get used(): number {
    return this.offset;
  }

  /**
   * Get remaining capacity.
   */
  get remaining(): number {
    return this.capacity - this.offset;
  }
}

// ============================================================
// TENSOR OPERATIONS
// ============================================================

/**
 * Low-level tensor operations with no allocations.
 * All operations write results to pre-allocated output buffers.
 */
export class TensorOps {
  /**
   * Matrix-vector multiplication: out = A * x
   * A: [m, n], x: [n], out: [m]
   *
   * Math: out_i = sum_j(A_ij * x_j)
   */
  static matvec(
    A: TensorView,
    x: TensorView,
    out: TensorView,
  ): void {
    const m = A.shape.dim(0);
    const n = A.shape.dim(1);
    const Ad = A.data;
    const Ao = A.offset;
    const xd = x.data;
    const xo = x.offset;
    const od = out.data;
    const oo = out.offset;

    for (let i = 0; i < m; i++) {
      let sum = 0;
      const rowOffset = Ao + i * n;
      for (let j = 0; j < n; j++) {
        sum += Ad[rowOffset + j] * xd[xo + j];
      }
      od[oo + i] = sum;
    }
  }

  /**
   * Matrix-vector multiplication with bias: out = A * x + b
   * A: [m, n], x: [n], b: [m], out: [m]
   */
  static matvecBias(
    A: TensorView,
    x: TensorView,
    b: TensorView,
    out: TensorView,
  ): void {
    const m = A.shape.dim(0);
    const n = A.shape.dim(1);
    const Ad = A.data;
    const Ao = A.offset;
    const xd = x.data;
    const xo = x.offset;
    const bd = b.data;
    const bo = b.offset;
    const od = out.data;
    const oo = out.offset;

    for (let i = 0; i < m; i++) {
      let sum = bd[bo + i];
      const rowOffset = Ao + i * n;
      for (let j = 0; j < n; j++) {
        sum += Ad[rowOffset + j] * xd[xo + j];
      }
      od[oo + i] = sum;
    }
  }

  /**
   * Vector addition: out = a + b
   */
  static add(a: TensorView, b: TensorView, out: TensorView): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    const od = out.data;
    const oo = out.offset;
    for (let i = 0; i < size; i++) {
      od[oo + i] = ad[ao + i] + bd[bo + i];
    }
  }

  /**
   * Vector subtraction: out = a - b
   */
  static sub(a: TensorView, b: TensorView, out: TensorView): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    const od = out.data;
    const oo = out.offset;
    for (let i = 0; i < size; i++) {
      od[oo + i] = ad[ao + i] - bd[bo + i];
    }
  }

  /**
   * Element-wise multiply: out = a * b
   */
  static mul(a: TensorView, b: TensorView, out: TensorView): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    const od = out.data;
    const oo = out.offset;
    for (let i = 0; i < size; i++) {
      od[oo + i] = ad[ao + i] * bd[bo + i];
    }
  }

  /**
   * Scale vector: out = a * scalar
   */
  static scale(a: TensorView, scalar: number, out: TensorView): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const od = out.data;
    const oo = out.offset;
    for (let i = 0; i < size; i++) {
      od[oo + i] = ad[ao + i] * scalar;
    }
  }

  /**
   * Scale in place: a *= scalar
   */
  static scaleInPlace(a: TensorView, scalar: number): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    for (let i = 0; i < size; i++) {
      ad[ao + i] *= scalar;
    }
  }

  /**
   * Add scaled: out = a + b * scalar
   */
  static addScaled(
    a: TensorView,
    b: TensorView,
    scalar: number,
    out: TensorView,
  ): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    const od = out.data;
    const oo = out.offset;
    for (let i = 0; i < size; i++) {
      od[oo + i] = ad[ao + i] + bd[bo + i] * scalar;
    }
  }

  /**
   * Add scaled in place: a += b * scalar
   */
  static addScaledInPlace(a: TensorView, b: TensorView, scalar: number): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    for (let i = 0; i < size; i++) {
      ad[ao + i] += bd[bo + i] * scalar;
    }
  }

  /**
   * Dot product: sum(a * b)
   */
  static dot(a: TensorView, b: TensorView): number {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    let sum = 0;
    for (let i = 0; i < size; i++) {
      sum += ad[ao + i] * bd[bo + i];
    }
    return sum;
  }

  /**
   * Outer product: out = a * b^T
   * a: [m], b: [n], out: [m, n]
   */
  static outer(a: TensorView, b: TensorView, out: TensorView): void {
    const m = a.shape.size;
    const n = b.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    const od = out.data;
    const oo = out.offset;

    for (let i = 0; i < m; i++) {
      const ai = ad[ao + i];
      const rowOffset = oo + i * n;
      for (let j = 0; j < n; j++) {
        od[rowOffset + j] = ai * bd[bo + j];
      }
    }
  }

  /**
   * Add outer product in place: out += a * b^T * scale
   */
  static addOuterScaledInPlace(
    out: TensorView,
    a: TensorView,
    b: TensorView,
    scale: number,
  ): void {
    const m = a.shape.size;
    const n = b.shape.size;
    const ad = a.data;
    const ao = a.offset;
    const bd = b.data;
    const bo = b.offset;
    const od = out.data;
    const oo = out.offset;

    for (let i = 0; i < m; i++) {
      const ai = ad[ao + i] * scale;
      const rowOffset = oo + i * n;
      for (let j = 0; j < n; j++) {
        od[rowOffset + j] += ai * bd[bo + j];
      }
    }
  }

  /**
   * L2 norm of vector.
   */
  static norm(a: TensorView): number {
    return Math.sqrt(a.squaredNorm());
  }

  /**
   * Copy vector.
   */
  static copy(src: TensorView, dst: TensorView): void {
    dst.copyFrom(src);
  }

  /**
   * Clip values to range [min, max] in place.
   */
  static clipInPlace(a: TensorView, min: number, max: number): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    for (let i = 0; i < size; i++) {
      const v = ad[ao + i];
      if (v < min) ad[ao + i] = min;
      else if (v > max) ad[ao + i] = max;
    }
  }

  /**
   * Check if all values are finite.
   */
  static allFinite(a: TensorView): boolean {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    for (let i = 0; i < size; i++) {
      if (!isFinite(ad[ao + i])) return false;
    }
    return true;
  }

  /**
   * Replace non-finite values with a default.
   */
  static sanitizeInPlace(a: TensorView, defaultValue: number = 0): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    for (let i = 0; i < size; i++) {
      if (!isFinite(ad[ao + i])) {
        ad[ao + i] = defaultValue;
      }
    }
  }
}

// ============================================================
// ACTIVATION OPERATIONS
// ============================================================

/**
 * Activation function operations with no allocations.
 */
export class ActivationOps {
  /**
   * Apply tanh in place.
   * Math: f(x) = tanh(x)
   */
  static tanhInPlace(a: TensorView): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    for (let i = 0; i < size; i++) {
      ad[ao + i] = Math.tanh(ad[ao + i]);
    }
  }

  /**
   * Apply ReLU in place.
   * Math: f(x) = max(0, x)
   */
  static reluInPlace(a: TensorView): void {
    const size = a.shape.size;
    const ad = a.data;
    const ao = a.offset;
    for (let i = 0; i < size; i++) {
      if (ad[ao + i] < 0) ad[ao + i] = 0;
    }
  }

  /**
   * Apply activation by name in place.
   */
  static applyInPlace(a: TensorView, activation: "tanh" | "relu"): void {
    if (activation === "tanh") {
      ActivationOps.tanhInPlace(a);
    } else {
      ActivationOps.reluInPlace(a);
    }
  }
}

// ============================================================
// RANDOM NUMBER GENERATOR
// ============================================================

/**
 * Deterministic pseudo-random number generator (xorshift128+).
 * Provides reproducible randomness from a seed.
 */
export class RandomGenerator {
  private s0: number;
  private s1: number;

  constructor(seed: number = 42) {
    // Initialize state from seed using splitmix64-like initialization
    let s = seed >>> 0;
    s = (s + 0x9e3779b9) >>> 0;
    this.s0 = this.splitmix32(s);
    this.s1 = this.splitmix32(s + 1);
  }

  private splitmix32(x: number): number {
    x = (x + 0x9e3779b9) >>> 0;
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
    return (x ^ (x >>> 16)) >>> 0;
  }

  /**
   * Get next random 32-bit unsigned integer.
   */
  nextUint32(): number {
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

  /**
   * Get uniform random in [0, 1).
   */
  uniform(): number {
    return this.nextUint32() / 4294967296;
  }

  /**
   * Get uniform random in [min, max).
   */
  uniformRange(min: number, max: number): number {
    return min + this.uniform() * (max - min);
  }

  /**
   * Get standard normal random (Box-Muller transform).
   * Math: Uses Z = sqrt(-2*ln(U1)) * cos(2*pi*U2)
   */
  normal(): number {
    const u1 = this.uniform();
    const u2 = this.uniform();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Get normal random with mean and std.
   */
  normalRange(mean: number, std: number): number {
    return mean + this.normal() * std;
  }

  /**
   * Fill tensor with uniform random values in [min, max).
   */
  fillUniform(tensor: TensorView, min: number = -1, max: number = 1): void {
    const size = tensor.shape.size;
    const data = tensor.data;
    const offset = tensor.offset;
    const range = max - min;
    for (let i = 0; i < size; i++) {
      data[offset + i] = min + this.uniform() * range;
    }
  }

  /**
   * Fill tensor with normal random values.
   */
  fillNormal(tensor: TensorView, mean: number = 0, std: number = 1): void {
    const size = tensor.shape.size;
    const data = tensor.data;
    const offset = tensor.offset;
    for (let i = 0; i < size; i++) {
      data[offset + i] = mean + this.normal() * std;
    }
  }

  /**
   * Apply sparsity mask: set fraction of values to zero.
   * @param tensor Tensor to sparsify
   * @param sparsity Fraction of values to zero (0 to 1)
   */
  applySparsity(tensor: TensorView, sparsity: number): void {
    const size = tensor.shape.size;
    const data = tensor.data;
    const offset = tensor.offset;
    for (let i = 0; i < size; i++) {
      if (this.uniform() < sparsity) {
        data[offset + i] = 0;
      }
    }
  }

  /**
   * Get state for serialization.
   */
  getState(): { s0: number; s1: number } {
    return { s0: this.s0, s1: this.s1 };
  }

  /**
   * Restore state from serialization.
   */
  setState(state: { s0: number; s1: number }): void {
    this.s0 = state.s0;
    this.s1 = state.s1;
  }
}

// ============================================================
// WELFORD ACCUMULATOR AND NORMALIZER
// ============================================================

/**
 * Welford's online algorithm for computing mean and variance.
 * Numerically stable single-pass computation.
 *
 * Math:
 * - M_n = M_{n-1} + (x_n - M_{n-1}) / n
 * - S_n = S_{n-1} + (x_n - M_{n-1}) * (x_n - M_n)
 * - Var = S_n / (n-1) for sample variance
 */
export class WelfordAccumulator {
  private count: number;
  private mean: Float64Array;
  private m2: Float64Array;
  private readonly size: number;
  private readonly epsilon: number;

  constructor(size: number, epsilon: number = 1e-8) {
    this.size = size;
    this.epsilon = epsilon;
    this.count = 0;
    this.mean = new Float64Array(size);
    this.m2 = new Float64Array(size);
  }

  /**
   * Update statistics with a new sample.
   * @param x New data point
   */
  update(x: TensorView): void {
    this.count++;
    const n = this.count;
    const data = x.data;
    const offset = x.offset;

    for (let i = 0; i < this.size; i++) {
      const xi = data[offset + i];
      const delta = xi - this.mean[i];
      this.mean[i] += delta / n;
      const delta2 = xi - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }

  /**
   * Update statistics from a plain array.
   */
  updateFromArray(x: ArrayLike<number>, offset: number = 0): void {
    this.count++;
    const n = this.count;

    for (let i = 0; i < this.size; i++) {
      const xi = x[offset + i];
      const delta = xi - this.mean[i];
      this.mean[i] += delta / n;
      const delta2 = xi - this.mean[i];
      this.m2[i] += delta * delta2;
    }
  }

  /**
   * Get current mean values.
   */
  getMean(): Float64Array {
    return this.mean;
  }

  /**
   * Get current variance values.
   */
  getVariance(): Float64Array {
    const variance = new Float64Array(this.size);
    if (this.count < 2) {
      variance.fill(1);
      return variance;
    }
    for (let i = 0; i < this.size; i++) {
      variance[i] = Math.max(this.m2[i] / (this.count - 1), this.epsilon);
    }
    return variance;
  }

  /**
   * Get current standard deviation values.
   */
  getStd(): Float64Array {
    const std = new Float64Array(this.size);
    if (this.count < 2) {
      std.fill(1);
      return std;
    }
    for (let i = 0; i < this.size; i++) {
      std[i] = Math.sqrt(Math.max(this.m2[i] / (this.count - 1), this.epsilon));
    }
    return std;
  }

  /**
   * Get sample count.
   */
  getCount(): number {
    return this.count;
  }

  /**
   * Reset accumulator state.
   */
  reset(): void {
    this.count = 0;
    this.mean.fill(0);
    this.m2.fill(0);
  }

  /**
   * Get state for serialization.
   */
  getState(): { count: number; mean: number[]; m2: number[] } {
    return {
      count: this.count,
      mean: Array.from(this.mean),
      m2: Array.from(this.m2),
    };
  }

  /**
   * Restore state from serialization.
   */
  setState(state: { count: number; mean: number[]; m2: number[] }): void {
    this.count = state.count;
    for (let i = 0; i < this.size; i++) {
      this.mean[i] = state.mean[i];
      this.m2[i] = state.m2[i];
    }
  }
}

/**
 * Welford-based z-score normalizer for online data normalization.
 * Handles warmup period before normalization is active.
 */
export class WelfordNormalizer {
  private accumulator: WelfordAccumulator;
  private readonly size: number;
  private readonly warmupCount: number;
  private readonly epsilon: number;
  private cachedMean: Float64Array;
  private cachedInvStd: Float64Array;
  private cachedStd: Float64Array;
  private cacheValid: boolean;

  constructor(
    size: number,
    warmupCount: number = 10,
    epsilon: number = 1e-8,
  ) {
    this.size = size;
    this.warmupCount = warmupCount;
    this.epsilon = epsilon;
    this.accumulator = new WelfordAccumulator(size, epsilon);
    this.cachedMean = new Float64Array(size);
    this.cachedInvStd = new Float64Array(size);
    this.cachedStd = new Float64Array(size);
    this.cachedInvStd.fill(1);
    this.cachedStd.fill(1);
    this.cacheValid = false;
  }

  /**
   * Update statistics with a new sample.
   */
  update(x: TensorView): void {
    this.accumulator.update(x);
    this.cacheValid = false;
  }

  /**
   * Update statistics from a plain array.
   */
  updateFromArray(x: ArrayLike<number>, offset: number = 0): void {
    this.accumulator.updateFromArray(x, offset);
    this.cacheValid = false;
  }

  /**
   * Update internal caches.
   */
  private updateCache(): void {
    if (this.cacheValid) return;

    const mean = this.accumulator.getMean();
    const std = this.accumulator.getStd();

    for (let i = 0; i < this.size; i++) {
      this.cachedMean[i] = mean[i];
      this.cachedStd[i] = std[i];
      this.cachedInvStd[i] = 1 / (std[i] + this.epsilon);
    }
    this.cacheValid = true;
  }

  /**
   * Whether normalization is active (warmup complete).
   */
  isActive(): boolean {
    return this.accumulator.getCount() >= this.warmupCount;
  }

  /**
   * Normalize a tensor in place.
   * Math: z = (x - mean) / std
   */
  normalizeInPlace(x: TensorView): void {
    if (!this.isActive()) return;

    this.updateCache();

    const data = x.data;
    const offset = x.offset;
    for (let i = 0; i < this.size; i++) {
      data[offset + i] = (data[offset + i] - this.cachedMean[i]) *
        this.cachedInvStd[i];
    }
  }

  /**
   * Normalize into output buffer.
   */
  normalize(x: TensorView, out: TensorView): void {
    if (!this.isActive()) {
      out.copyFrom(x);
      return;
    }

    this.updateCache();

    const xData = x.data;
    const xOffset = x.offset;
    const outData = out.data;
    const outOffset = out.offset;

    for (let i = 0; i < this.size; i++) {
      outData[outOffset + i] = (xData[xOffset + i] - this.cachedMean[i]) *
        this.cachedInvStd[i];
    }
  }

  /**
   * Denormalize a tensor in place.
   * Math: x = z * std + mean
   */
  denormalizeInPlace(z: TensorView): void {
    if (!this.isActive()) return;

    this.updateCache();

    const data = z.data;
    const offset = z.offset;
    for (let i = 0; i < this.size; i++) {
      data[offset + i] = data[offset + i] * this.cachedStd[i] +
        this.cachedMean[i];
    }
  }

  /**
   * Denormalize into output buffer.
   */
  denormalize(z: TensorView, out: TensorView): void {
    if (!this.isActive()) {
      out.copyFrom(z);
      return;
    }

    this.updateCache();

    const zData = z.data;
    const zOffset = z.offset;
    const outData = out.data;
    const outOffset = out.offset;

    for (let i = 0; i < this.size; i++) {
      outData[outOffset + i] = zData[zOffset + i] * this.cachedStd[i] +
        this.cachedMean[i];
    }
  }

  /**
   * Get normalization statistics.
   */
  getStats(): NormalizationStats {
    this.updateCache();
    return {
      means: Array.from(this.cachedMean),
      stds: Array.from(this.cachedStd),
      count: this.accumulator.getCount(),
      isActive: this.isActive(),
    };
  }

  /**
   * Get sample count.
   */
  getCount(): number {
    return this.accumulator.getCount();
  }

  /**
   * Reset normalizer state.
   */
  reset(): void {
    this.accumulator.reset();
    this.cachedMean.fill(0);
    this.cachedInvStd.fill(1);
    this.cachedStd.fill(1);
    this.cacheValid = false;
  }

  /**
   * Get state for serialization.
   */
  getState(): object {
    return {
      accumulator: this.accumulator.getState(),
      cacheValid: false,
    };
  }

  /**
   * Restore state from serialization.
   */
  setState(state: any): void {
    this.accumulator.setState(state.accumulator);
    this.cacheValid = false;
  }
}

// ============================================================
// LAYER NORMALIZATION
// ============================================================

/**
 * Parameters for layer normalization.
 */
export class LayerNormParams {
  readonly gamma: TensorView;
  readonly beta: TensorView;
  readonly size: number;

  constructor(arena: TensorArena, size: number) {
    this.size = size;
    this.gamma = arena.allocVector(size);
    this.beta = arena.allocVector(size);
    this.gamma.fill(1);
    this.beta.fill(0);
  }
}

/**
 * Layer normalization operations.
 * Math: y = gamma * (x - mean) / sqrt(var + eps) + beta
 */
export class LayerNormOps {
  /**
   * Apply layer normalization in place.
   */
  static normalizeInPlace(
    x: TensorView,
    params: LayerNormParams,
    epsilon: number,
    scratch: TensorView,
  ): void {
    const size = x.shape.size;
    const data = x.data;
    const offset = x.offset;
    const gamma = params.gamma.data;
    const gammaOffset = params.gamma.offset;
    const beta = params.beta.data;
    const betaOffset = params.beta.offset;

    // Compute mean
    let mean = 0;
    for (let i = 0; i < size; i++) {
      mean += data[offset + i];
    }
    mean /= size;

    // Compute variance
    let variance = 0;
    for (let i = 0; i < size; i++) {
      const d = data[offset + i] - mean;
      variance += d * d;
    }
    variance /= size;

    // Normalize
    const invStd = 1 / Math.sqrt(variance + epsilon);
    for (let i = 0; i < size; i++) {
      const normalized = (data[offset + i] - mean) * invStd;
      data[offset + i] = gamma[gammaOffset + i] * normalized +
        beta[betaOffset + i];
    }
  }
}

// ============================================================
// GRADIENT ACCUMULATOR
// ============================================================

/**
 * Accumulator for gradient statistics and clipping.
 */
export class GradientAccumulator {
  private sumSquared: number;
  private count: number;
  private readonly clipNorm: number;

  constructor(clipNorm: number = 1.0) {
    this.clipNorm = clipNorm;
    this.sumSquared = 0;
    this.count = 0;
  }

  /**
   * Get norm of a gradient vector.
   */
  computeNorm(grad: TensorView): number {
    return Math.sqrt(grad.squaredNorm());
  }

  /**
   * Clip gradient by norm in place.
   * Math: grad = grad * min(1, clip_norm / ||grad||)
   */
  clipByNormInPlace(grad: TensorView): number {
    const norm = this.computeNorm(grad);
    if (norm > this.clipNorm && norm > 0) {
      const scale = this.clipNorm / norm;
      TensorOps.scaleInPlace(grad, scale);
    }
    this.sumSquared += norm * norm;
    this.count++;
    return norm;
  }

  /**
   * Get average gradient norm.
   */
  getAverageNorm(): number {
    return this.count > 0 ? Math.sqrt(this.sumSquared / this.count) : 0;
  }

  /**
   * Reset accumulator.
   */
  reset(): void {
    this.sumSquared = 0;
    this.count = 0;
  }
}

// ============================================================
// RESERVOIR INITIALIZATION
// ============================================================

/**
 * Initialization mask for sparse reservoir connectivity.
 */
export class ReservoirInitMask {
  readonly mask: Float64Array;
  readonly nonZeroCount: number;
  readonly size: number;

  constructor(size: number, sparsity: number, rng: RandomGenerator) {
    this.size = size;
    this.mask = new Float64Array(size);
    let count = 0;
    for (let i = 0; i < size; i++) {
      if (rng.uniform() >= sparsity) {
        this.mask[i] = 1;
        count++;
      }
    }
    this.nonZeroCount = count;
  }

  /**
   * Apply mask to a tensor in place.
   */
  applyInPlace(tensor: TensorView): void {
    const data = tensor.data;
    const offset = tensor.offset;
    for (let i = 0; i < this.size; i++) {
      data[offset + i] *= this.mask[i];
    }
  }
}

/**
 * Spectral radius scaling for reservoir weight matrix.
 *
 * Scales W so that its spectral radius equals the target.
 * Uses power iteration to estimate the spectral radius.
 */
export class SpectralRadiusScaler {
  private readonly maxIterations: number;
  private readonly tolerance: number;

  constructor(maxIterations: number = 100, tolerance: number = 1e-6) {
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
  }

  /**
   * Estimate spectral radius using power iteration.
   * Math: Find largest eigenvalue magnitude via v_{k+1} = W * v_k / ||W * v_k||
   */
  estimateSpectralRadius(
    W: TensorView,
    scratch1: TensorView,
    scratch2: TensorView,
    rng: RandomGenerator,
  ): number {
    const n = W.shape.dim(0);

    // Initialize random vector
    rng.fillUniform(scratch1, -1, 1);

    // Normalize
    let norm = TensorOps.norm(scratch1);
    if (norm < 1e-10) {
      scratch1.data[scratch1.offset] = 1;
      norm = 1;
    }
    TensorOps.scaleInPlace(scratch1, 1 / norm);

    let lambda = 0;
    let prevLambda = 0;

    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Multiply: scratch2 = W * scratch1
      TensorOps.matvec(W, scratch1, scratch2);

      // Compute new eigenvalue estimate
      lambda = TensorOps.norm(scratch2);

      if (lambda < 1e-10) {
        return 0;
      }

      // Normalize
      TensorOps.scaleInPlace(scratch2, 1 / lambda);

      // Check convergence
      if (Math.abs(lambda - prevLambda) < this.tolerance) {
        break;
      }
      prevLambda = lambda;

      // Swap buffers
      const temp = scratch1;
      TensorOps.copy(scratch2, scratch1);
    }

    return lambda;
  }

  /**
   * Scale matrix to target spectral radius.
   */
  scaleToTarget(
    W: TensorView,
    targetRadius: number,
    scratch1: TensorView,
    scratch2: TensorView,
    rng: RandomGenerator,
  ): void {
    const currentRadius = this.estimateSpectralRadius(
      W,
      scratch1,
      scratch2,
      rng,
    );

    if (currentRadius > 1e-10) {
      const scale = targetRadius / currentRadius;
      TensorOps.scaleInPlace(W, scale);
    }
  }
}

// ============================================================
// ESN RESERVOIR
// ============================================================

/**
 * Parameters for ESN reservoir.
 */
export class ESNReservoirParams {
  readonly Win: TensorView; // Input weights [reservoirSize, nFeatures]
  readonly W: TensorView; // Reservoir weights [reservoirSize, reservoirSize]
  readonly bias: TensorView; // Bias [reservoirSize]
  readonly reservoirSize: number;
  readonly nFeatures: number;

  constructor(
    arena: TensorArena,
    reservoirSize: number,
    nFeatures: number,
  ) {
    this.reservoirSize = reservoirSize;
    this.nFeatures = nFeatures;
    this.Win = arena.allocMatrix(reservoirSize, nFeatures);
    this.W = arena.allocMatrix(reservoirSize, reservoirSize);
    this.bias = arena.allocVector(reservoirSize);
  }
}

/**
 * Echo State Network reservoir with leaky integrator dynamics.
 *
 * State update equation:
 * r_t = (1 - leakRate) * r_{t-1} + leakRate * activation(Win * x_t + W * r_{t-1} + bias)
 *
 * The reservoir weights (Win, W, bias) are fixed after initialization
 * and not trained - only the readout weights are learned.
 */
export class ESNReservoir {
  readonly params: ESNReservoirParams;
  readonly state: TensorView; // Current reservoir state [reservoirSize]
  private readonly preActivation: TensorView; // Scratch for pre-activation
  private readonly inputContrib: TensorView; // Scratch for Win * x
  private readonly stateContrib: TensorView; // Scratch for W * r

  readonly reservoirSize: number;
  readonly nFeatures: number;
  readonly leakRate: number;
  readonly inputScale: number;
  readonly activation: "tanh" | "relu";

  constructor(
    arena: TensorArena,
    config: Required<ESNRegressionConfig>,
    nFeatures: number,
    rng: RandomGenerator,
  ) {
    this.reservoirSize = config.reservoirSize;
    this.nFeatures = nFeatures;
    this.leakRate = config.leakRate;
    this.inputScale = config.inputScale;
    this.activation = config.activation;

    // Allocate parameters
    this.params = new ESNReservoirParams(
      arena,
      config.reservoirSize,
      nFeatures,
    );

    // Allocate state and scratch buffers
    this.state = arena.allocVector(config.reservoirSize);
    this.preActivation = arena.allocVector(config.reservoirSize);
    this.inputContrib = arena.allocVector(config.reservoirSize);
    this.stateContrib = arena.allocVector(config.reservoirSize);

    // Initialize weights
    this.initializeWeights(config, rng);
  }

  /**
   * Initialize reservoir weights with proper scaling.
   */
  private initializeWeights(
    config: Required<ESNRegressionConfig>,
    rng: RandomGenerator,
  ): void {
    const { reservoirSize, nFeatures } = this;

    // Initialize input weights Win with optional sparsity
    // Scale by inputScale / sqrt(nFeatures) for stable input contribution
    const inputStd = config.inputScale / Math.sqrt(nFeatures);
    rng.fillNormal(this.params.Win, 0, inputStd);
    if (config.inputSparsity > 0) {
      rng.applySparsity(this.params.Win, config.inputSparsity);
    }

    // Initialize reservoir weights W with sparsity
    rng.fillNormal(this.params.W, 0, 1);
    rng.applySparsity(this.params.W, config.reservoirSparsity);

    // Scale W to target spectral radius
    const scaler = new SpectralRadiusScaler();
    const scratch1 = new Float64Array(reservoirSize);
    const scratch2 = new Float64Array(reservoirSize);
    const scratch1View = new TensorView(
      scratch1,
      0,
      new TensorShape([reservoirSize]),
    );
    const scratch2View = new TensorView(
      scratch2,
      0,
      new TensorShape([reservoirSize]),
    );
    scaler.scaleToTarget(
      this.params.W,
      config.spectralRadius,
      scratch1View,
      scratch2View,
      rng,
    );

    // Initialize bias
    rng.fillNormal(this.params.bias, 0, config.biasScale);

    // Initialize state to zeros
    this.state.fill(0);
  }

  /**
   * Update reservoir state with new input.
   *
   * Math: r_t = (1 - α) * r_{t-1} + α * tanh(Win * x_t + W * r_{t-1} + b)
   * where α = leakRate
   *
   * @param x Input vector [nFeatures]
   */
  update(x: TensorView): void {
    const { reservoirSize, leakRate, activation } = this;
    const stateData = this.state.data;
    const stateOffset = this.state.offset;

    // Compute input contribution: inputContrib = Win * x
    TensorOps.matvec(this.params.Win, x, this.inputContrib);

    // Compute state contribution: stateContrib = W * r
    TensorOps.matvec(this.params.W, this.state, this.stateContrib);

    // Compute pre-activation: preAct = inputContrib + stateContrib + bias
    const preActData = this.preActivation.data;
    const preActOffset = this.preActivation.offset;
    const inputData = this.inputContrib.data;
    const inputOffset = this.inputContrib.offset;
    const stateContribData = this.stateContrib.data;
    const stateContribOffset = this.stateContrib.offset;
    const biasData = this.params.bias.data;
    const biasOffset = this.params.bias.offset;

    for (let i = 0; i < reservoirSize; i++) {
      preActData[preActOffset + i] = inputData[inputOffset + i] +
        stateContribData[stateContribOffset + i] +
        biasData[biasOffset + i];
    }

    // Apply activation
    ActivationOps.applyInPlace(this.preActivation, activation);

    // Leaky integration: state = (1 - α) * state + α * activated
    const oneMinusAlpha = 1 - leakRate;
    for (let i = 0; i < reservoirSize; i++) {
      stateData[stateOffset + i] = oneMinusAlpha * stateData[stateOffset + i] +
        leakRate * preActData[preActOffset + i];
    }
  }

  /**
   * Update reservoir state with input from plain array (no allocation).
   */
  updateFromArray(
    x: ArrayLike<number>,
    xOffset: number,
    scratchInput: TensorView,
  ): void {
    // Copy input to scratch
    const data = scratchInput.data;
    const offset = scratchInput.offset;
    for (let i = 0; i < this.nFeatures; i++) {
      data[offset + i] = x[xOffset + i];
    }
    this.update(scratchInput);
  }

  /**
   * Reset reservoir state to zeros.
   */
  resetState(): void {
    this.state.fill(0);
  }

  /**
   * Copy current state to another buffer.
   */
  copyStateTo(dst: TensorView): void {
    dst.copyFrom(this.state);
  }

  /**
   * Restore state from a buffer.
   */
  restoreStateFrom(src: TensorView): void {
    this.state.copyFrom(src);
  }

  /**
   * Get state for serialization.
   */
  getState(): { state: number[] } {
    return {
      state: this.state.toArray(),
    };
  }

  /**
   * Restore state from serialization.
   */
  setState(savedState: { state: number[] }): void {
    const data = this.state.data;
    const offset = this.state.offset;
    for (let i = 0; i < this.reservoirSize; i++) {
      data[offset + i] = savedState.state[i];
    }
  }

  /**
   * Get parameter state for serialization.
   */
  getParamsState(): object {
    return {
      Win: this.params.Win.toArray(),
      W: this.params.W.toArray(),
      bias: this.params.bias.toArray(),
    };
  }

  /**
   * Restore parameter state from serialization.
   */
  setParamsState(state: any): void {
    const WinData = this.params.Win.data;
    const WinOffset = this.params.Win.offset;
    for (let i = 0; i < state.Win.length; i++) {
      WinData[WinOffset + i] = state.Win[i];
    }

    const WData = this.params.W.data;
    const WOffset = this.params.W.offset;
    for (let i = 0; i < state.W.length; i++) {
      WData[WOffset + i] = state.W[i];
    }

    const biasData = this.params.bias.data;
    const biasOffset = this.params.bias.offset;
    for (let i = 0; i < state.bias.length; i++) {
      biasData[biasOffset + i] = state.bias[i];
    }
  }
}

// ============================================================
// RLS OPTIMIZER
// ============================================================

/**
 * State for Recursive Least Squares optimizer.
 * Maintains the inverse correlation matrix P.
 */
export class RLSState {
  readonly P: TensorView; // Inverse correlation matrix [inputDim, inputDim]
  readonly gain: TensorView; // Kalman gain vector [inputDim]
  readonly Pz: TensorView; // P * z scratch [inputDim]
  readonly inputDim: number;
  readonly lambda: number; // Forgetting factor
  readonly delta: number; // Initial P scale
  private initialized: boolean;

  constructor(
    arena: TensorArena,
    inputDim: number,
    lambda: number = 0.999,
    delta: number = 1.0,
  ) {
    this.inputDim = inputDim;
    this.lambda = lambda;
    this.delta = delta;
    this.P = arena.allocMatrix(inputDim, inputDim);
    this.gain = arena.allocVector(inputDim);
    this.Pz = arena.allocVector(inputDim);
    this.initialized = false;
    this.initialize();
  }

  /**
   * Initialize P to delta * I (identity scaled).
   */
  initialize(): void {
    this.P.fill(0);
    const data = this.P.data;
    const offset = this.P.offset;
    const n = this.inputDim;
    for (let i = 0; i < n; i++) {
      data[offset + i * n + i] = this.delta;
    }
    this.initialized = true;
  }

  /**
   * Check if RLS is initialized.
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get state for serialization.
   */
  getState(): object {
    return {
      P: this.P.toArray(),
      initialized: this.initialized,
    };
  }

  /**
   * Restore state from serialization.
   */
  setState(state: any): void {
    const data = this.P.data;
    const offset = this.P.offset;
    for (let i = 0; i < state.P.length; i++) {
      data[offset + i] = state.P[i];
    }
    this.initialized = state.initialized;
  }
}

/**
 * Recursive Least Squares optimizer for online readout training.
 *
 * RLS update equations:
 * 1. P_z = P * z
 * 2. denom = λ + z^T * P_z
 * 3. gain = P_z / denom
 * 4. error = target - prediction
 * 5. weights += gain * error^T
 * 6. P = (P - gain * z^T * P) / λ
 *
 * With optional L2 regularization integrated via modified P update.
 */
export class RLSOptimizer {
  private readonly state: RLSState;
  private readonly epsilon: number;
  private readonly l2Lambda: number;
  private readonly scratchRow: TensorView;

  constructor(
    state: RLSState,
    scratchArena: TensorArena,
    epsilon: number = 1e-8,
    l2Lambda: number = 0.0001,
  ) {
    this.state = state;
    this.epsilon = epsilon;
    this.l2Lambda = l2Lambda;
    this.scratchRow = scratchArena.allocVector(state.inputDim);
  }

  /**
   * Perform one RLS update step.
   *
   * @param z Extended state vector [inputDim]
   * @param target Target output vector [outputDim]
   * @param prediction Current prediction [outputDim]
   * @param weights Weight matrix [outputDim, inputDim]
   * @param sampleWeight Optional sample weight (for outlier downweighting)
   * @returns Error norm
   */
  update(
    z: TensorView,
    target: TensorView,
    prediction: TensorView,
    weights: TensorView,
    sampleWeight: number = 1.0,
  ): number {
    const { inputDim, lambda, P, gain, Pz } = this.state;
    const outputDim = target.shape.size;

    // Step 1: Compute P * z
    TensorOps.matvec(P, z, Pz);

    // Step 2: Compute denominator = λ + z^T * P * z
    const zTPz = TensorOps.dot(z, Pz);
    const denom = lambda + zTPz + this.epsilon;

    // Step 3: Compute gain = P_z / denom
    TensorOps.scale(Pz, 1 / denom, gain);

    // Compute error = target - prediction
    let errorNorm = 0;
    const targetData = target.data;
    const targetOffset = target.offset;
    const predData = prediction.data;
    const predOffset = prediction.offset;
    const wData = weights.data;
    const wOffset = weights.offset;
    const gainData = gain.data;
    const gainOffset = gain.offset;

    // Step 5: Update weights += sampleWeight * gain * error^T
    for (let o = 0; o < outputDim; o++) {
      const error = targetData[targetOffset + o] - predData[predOffset + o];
      errorNorm += error * error;

      const scaledError = sampleWeight * error;
      const rowOffset = wOffset + o * inputDim;
      for (let i = 0; i < inputDim; i++) {
        wData[rowOffset + i] += scaledError * gainData[gainOffset + i];
      }
    }

    // Step 6: Update P = (P - gain * z^T * P) / λ
    // This is the Sherman-Morrison update for the inverse
    // P_new = (I - gain * z^T) * P / λ
    this.updateP(z, gain);

    // Apply L2 regularization by shrinking weights slightly
    if (this.l2Lambda > 0) {
      const shrink = 1 - this.l2Lambda;
      const wSize = weights.shape.size;
      for (let i = 0; i < wSize; i++) {
        wData[wOffset + i] *= shrink;
      }
    }

    return Math.sqrt(errorNorm);
  }

  /**
   * Update P matrix using Sherman-Morrison formula.
   * P = (P - gain * (P * z)^T) / λ = (P - gain * Pz^T) / λ
   */
  private updateP(z: TensorView, gain: TensorView): void {
    const { inputDim, lambda, P, Pz } = this.state;
    const Pd = P.data;
    const Po = P.offset;
    const gainD = gain.data;
    const gainO = gain.offset;
    const PzD = Pz.data;
    const PzO = Pz.offset;

    const invLambda = 1 / lambda;

    // P = (P - gain * Pz^T) / λ
    for (let i = 0; i < inputDim; i++) {
      const gi = gainD[gainO + i];
      const rowOffset = Po + i * inputDim;
      for (let j = 0; j < inputDim; j++) {
        Pd[rowOffset + j] = (Pd[rowOffset + j] - gi * PzD[PzO + j]) * invLambda;
      }
    }

    // Ensure P remains symmetric (numerical stability)
    for (let i = 0; i < inputDim; i++) {
      for (let j = i + 1; j < inputDim; j++) {
        const avg = 0.5 *
          (Pd[Po + i * inputDim + j] + Pd[Po + j * inputDim + i]);
        Pd[Po + i * inputDim + j] = avg;
        Pd[Po + j * inputDim + i] = avg;
      }
    }
  }

  /**
   * Get RLS state.
   */
  getState(): RLSState {
    return this.state;
  }
}

// ============================================================
// READOUT CONFIGURATION AND PARAMS
// ============================================================

/**
 * Configuration for the linear readout layer.
 */
export interface ReadoutConfig {
  reservoirSize: number;
  nFeatures: number;
  nTargets: number;
  maxFutureSteps: number;
  useInputInReadout: boolean;
  useBiasInReadout: boolean;
  useDirectMultiHorizon: boolean;
}

/**
 * Parameters for the linear readout layer.
 */
export class ReadoutParams {
  readonly weights: TensorView; // [outputDim, inputDim]
  readonly inputDim: number; // reservoirSize + nFeatures(opt) + 1(opt)
  readonly outputDim: number; // nTargets * maxFutureSteps (direct) or nTargets (recursive)

  constructor(
    arena: TensorArena,
    config: ReadoutConfig,
  ) {
    // Compute input dimension
    let inputDim = config.reservoirSize;
    if (config.useInputInReadout) {
      inputDim += config.nFeatures;
    }
    if (config.useBiasInReadout) {
      inputDim += 1; // Bias term
    }
    this.inputDim = inputDim;

    // Compute output dimension
    if (config.useDirectMultiHorizon) {
      this.outputDim = config.nTargets * config.maxFutureSteps;
    } else {
      this.outputDim = config.nTargets;
    }

    this.weights = arena.allocMatrix(this.outputDim, inputDim);
    this.weights.fill(0);
  }
}

/**
 * Linear readout layer for ESN.
 * Maps extended state z = [r; x; 1] to output predictions.
 */
export class LinearReadout {
  readonly params: ReadoutParams;
  readonly extendedState: TensorView; // z = [r; x; 1]
  readonly output: TensorView; // Prediction output
  private readonly config: ReadoutConfig;

  constructor(
    arena: TensorArena,
    config: ReadoutConfig,
  ) {
    this.config = config;
    this.params = new ReadoutParams(arena, config);
    this.extendedState = arena.allocVector(this.params.inputDim);
    this.output = arena.allocVector(this.params.outputDim);
  }

  /**
   * Build extended state z from reservoir state and input.
   * z = [r; x; 1] (configurable components)
   */
  buildExtendedState(
    reservoirState: TensorView,
    input: TensorView | null,
  ): void {
    const { config } = this;
    const zData = this.extendedState.data;
    const zOffset = this.extendedState.offset;

    let idx = 0;

    // Copy reservoir state
    const rData = reservoirState.data;
    const rOffset = reservoirState.offset;
    for (let i = 0; i < config.reservoirSize; i++) {
      zData[zOffset + idx++] = rData[rOffset + i];
    }

    // Copy input if configured
    if (config.useInputInReadout && input !== null) {
      const xData = input.data;
      const xOffset = input.offset;
      for (let i = 0; i < config.nFeatures; i++) {
        zData[zOffset + idx++] = xData[xOffset + i];
      }
    }

    // Add bias term if configured
    if (config.useBiasInReadout) {
      zData[zOffset + idx] = 1;
    }
  }

  /**
   * Compute forward pass: output = W * z
   */
  forward(): void {
    TensorOps.matvec(this.params.weights, this.extendedState, this.output);
  }

  /**
   * Get output for specific future step and target.
   * For direct multi-horizon: index = step * nTargets + target
   */
  getOutput(step: number, target: number): number {
    if (this.config.useDirectMultiHorizon) {
      const idx = step * this.config.nTargets + target;
      return this.output.getLinear(idx);
    } else {
      return this.output.getLinear(target);
    }
  }

  /**
   * Get all outputs for a specific future step.
   */
  getStepOutputs(step: number, out: number[]): void {
    const { nTargets } = this.config;
    if (this.config.useDirectMultiHorizon) {
      const baseIdx = step * nTargets;
      for (let t = 0; t < nTargets; t++) {
        out[t] = this.output.getLinear(baseIdx + t);
      }
    } else {
      for (let t = 0; t < nTargets; t++) {
        out[t] = this.output.getLinear(t);
      }
    }
  }

  /**
   * Get state for serialization.
   */
  getState(): object {
    return {
      weights: this.params.weights.toArray(),
    };
  }

  /**
   * Restore state from serialization.
   */
  setState(state: any): void {
    const wData = this.params.weights.data;
    const wOffset = this.params.weights.offset;
    for (let i = 0; i < state.weights.length; i++) {
      wData[wOffset + i] = state.weights[i];
    }
  }
}

// ============================================================
// LINEAR LAYER (FOR POTENTIAL EXTENSIONS)
// ============================================================

/**
 * Parameters for a standard linear layer.
 */
export class LinearLayerParams {
  readonly weights: TensorView;
  readonly bias: TensorView;
  readonly inputDim: number;
  readonly outputDim: number;

  constructor(arena: TensorArena, inputDim: number, outputDim: number) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.weights = arena.allocMatrix(outputDim, inputDim);
    this.bias = arena.allocVector(outputDim);
  }
}

/**
 * Standard linear layer: y = Wx + b
 */
export class LinearLayer {
  readonly params: LinearLayerParams;
  readonly output: TensorView;

  constructor(arena: TensorArena, inputDim: number, outputDim: number) {
    this.params = new LinearLayerParams(arena, inputDim, outputDim);
    this.output = arena.allocVector(outputDim);
  }

  forward(input: TensorView): void {
    TensorOps.matvecBias(
      this.params.weights,
      input,
      this.params.bias,
      this.output,
    );
  }
}

/**
 * Dropout mask (placeholder for potential extensions).
 */
export class DropoutMask {
  readonly mask: Float64Array;
  readonly size: number;

  constructor(size: number) {
    this.size = size;
    this.mask = new Float64Array(size);
    this.mask.fill(1);
  }

  generate(rng: RandomGenerator, dropRate: number): void {
    const keepRate = 1 - dropRate;
    const scale = 1 / keepRate;
    for (let i = 0; i < this.size; i++) {
      this.mask[i] = rng.uniform() < keepRate ? scale : 0;
    }
  }

  applyInPlace(tensor: TensorView): void {
    const data = tensor.data;
    const offset = tensor.offset;
    for (let i = 0; i < this.size; i++) {
      data[offset + i] *= this.mask[i];
    }
  }
}

// ============================================================
// FORWARD AND BACKWARD CONTEXT
// ============================================================

/**
 * Context for forward pass (stores intermediate values if needed).
 */
export class ForwardContext {
  reservoirState: TensorView | null = null;
  extendedState: TensorView | null = null;
  output: TensorView | null = null;
  normalizedInput: TensorView | null = null;

  reset(): void {
    this.reservoirState = null;
    this.extendedState = null;
    this.output = null;
    this.normalizedInput = null;
  }
}

/**
 * Context for backward pass (placeholder - ESN doesn't need full backprop).
 */
export class BackwardContext {
  error: TensorView | null = null;

  reset(): void {
    this.error = null;
  }
}

/**
 * Gradient tape (minimal for ESN - only tracks readout updates).
 */
export class GradientTape {
  private updateNorm: number = 0;

  recordUpdate(norm: number): void {
    this.updateNorm = norm;
  }

  getLastUpdateNorm(): number {
    return this.updateNorm;
  }

  reset(): void {
    this.updateNorm = 0;
  }
}

// ============================================================
// RING BUFFER
// ============================================================

/**
 * Fixed-size ring buffer for storing history of input features.
 * Supports efficient append and window extraction without allocation.
 */
export class RingBuffer {
  private buffer: Float64Array;
  private readonly capacity: number;
  private readonly featureSize: number;
  private head: number; // Next write position
  private count: number; // Number of valid entries

  constructor(capacity: number, featureSize: number) {
    this.capacity = capacity;
    this.featureSize = featureSize;
    this.buffer = new Float64Array(capacity * featureSize);
    this.head = 0;
    this.count = 0;
  }

  /**
   * Push a new row into the buffer.
   * @param row Feature vector to push
   */
  push(row: ArrayLike<number>): void {
    const offset = this.head * this.featureSize;
    for (let i = 0; i < this.featureSize; i++) {
      this.buffer[offset + i] = row[i];
    }
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  /**
   * Push from a TensorView.
   */
  pushFromView(view: TensorView): void {
    const offset = this.head * this.featureSize;
    const srcData = view.data;
    const srcOffset = view.offset;
    for (let i = 0; i < this.featureSize; i++) {
      this.buffer[offset + i] = srcData[srcOffset + i];
    }
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  /**
   * Get the number of valid entries.
   */
  size(): number {
    return this.count;
  }

  /**
   * Check if buffer is empty.
   */
  isEmpty(): boolean {
    return this.count === 0;
  }

  /**
   * Check if buffer is full.
   */
  isFull(): boolean {
    return this.count === this.capacity;
  }

  /**
   * Get row at logical index (0 = oldest, count-1 = newest).
   * Writes to output array without allocation.
   */
  getRow(logicalIndex: number, out: number[]): void {
    if (logicalIndex < 0 || logicalIndex >= this.count) {
      throw new Error(`Index ${logicalIndex} out of bounds [0, ${this.count})`);
    }

    // Convert logical index to physical index
    // If buffer is not full, oldest is at 0
    // If buffer is full, oldest is at head
    let physicalIndex: number;
    if (this.count < this.capacity) {
      physicalIndex = logicalIndex;
    } else {
      physicalIndex = (this.head + logicalIndex) % this.capacity;
    }

    const offset = physicalIndex * this.featureSize;
    for (let i = 0; i < this.featureSize; i++) {
      out[i] = this.buffer[offset + i];
    }
  }

  /**
   * Get the most recent row (newest entry).
   */
  getLatest(out: number[]): void {
    if (this.count === 0) {
      throw new Error("Buffer is empty");
    }
    this.getRow(this.count - 1, out);
  }

  /**
   * Get row to a TensorView.
   */
  getRowToView(logicalIndex: number, view: TensorView): void {
    if (logicalIndex < 0 || logicalIndex >= this.count) {
      throw new Error(`Index ${logicalIndex} out of bounds [0, ${this.count})`);
    }

    let physicalIndex: number;
    if (this.count < this.capacity) {
      physicalIndex = logicalIndex;
    } else {
      physicalIndex = (this.head + logicalIndex) % this.capacity;
    }

    const srcOffset = physicalIndex * this.featureSize;
    const dstData = view.data;
    const dstOffset = view.offset;
    for (let i = 0; i < this.featureSize; i++) {
      dstData[dstOffset + i] = this.buffer[srcOffset + i];
    }
  }

  /**
   * Get the most recent row to a TensorView.
   */
  getLatestToView(view: TensorView): void {
    if (this.count === 0) {
      throw new Error("Buffer is empty");
    }
    this.getRowToView(this.count - 1, view);
  }

  /**
   * Copy a window of rows to output buffer.
   * Window ends at the most recent entry.
   * @param windowSize Number of rows to extract
   * @param out Output buffer [windowSize * featureSize]
   */
  getWindow(windowSize: number, out: Float64Array): void {
    const actualSize = Math.min(windowSize, this.count);
    const startIdx = this.count - actualSize;

    for (let w = 0; w < actualSize; w++) {
      let physicalIndex: number;
      if (this.count < this.capacity) {
        physicalIndex = startIdx + w;
      } else {
        physicalIndex = (this.head + startIdx + w) % this.capacity;
      }

      const srcOffset = physicalIndex * this.featureSize;
      const dstOffset = w * this.featureSize;
      for (let i = 0; i < this.featureSize; i++) {
        out[dstOffset + i] = this.buffer[srcOffset + i];
      }
    }

    // Zero-pad if window is larger than count
    if (actualSize < windowSize) {
      const startPad = actualSize * this.featureSize;
      const endPad = windowSize * this.featureSize;
      for (let i = startPad; i < endPad; i++) {
        out[i] = 0;
      }
    }
  }

  /**
   * Clear the buffer.
   */
  clear(): void {
    this.head = 0;
    this.count = 0;
    this.buffer.fill(0);
  }

  /**
   * Get state for serialization.
   */
  getState(): object {
    return {
      buffer: Array.from(this.buffer),
      head: this.head,
      count: this.count,
    };
  }

  /**
   * Restore state from serialization.
   */
  setState(state: any): void {
    for (let i = 0; i < state.buffer.length; i++) {
      this.buffer[i] = state.buffer[i];
    }
    this.head = state.head;
    this.count = state.count;
  }
}

// ============================================================
// RESIDUAL STATS TRACKER
// ============================================================

/**
 * Tracks residual statistics for uncertainty estimation.
 * Uses a sliding window approach with Welford accumulator.
 */
export class ResidualStatsTracker {
  private readonly windowSize: number;
  private readonly nTargets: number;
  private readonly maxFutureSteps: number;
  private residuals: Float64Array; // [windowSize, maxFutureSteps, nTargets]
  private squaredResiduals: Float64Array; // Running sum of squares per target/step
  private means: Float64Array;
  private variances: Float64Array;
  private head: number;
  private count: number;

  constructor(
    nTargets: number,
    maxFutureSteps: number,
    windowSize: number = 100,
  ) {
    this.nTargets = nTargets;
    this.maxFutureSteps = maxFutureSteps;
    this.windowSize = windowSize;

    const totalSize = windowSize * maxFutureSteps * nTargets;
    this.residuals = new Float64Array(totalSize);
    this.squaredResiduals = new Float64Array(maxFutureSteps * nTargets);
    this.means = new Float64Array(maxFutureSteps * nTargets);
    this.variances = new Float64Array(maxFutureSteps * nTargets);
    this.variances.fill(1); // Default variance
    this.head = 0;
    this.count = 0;
  }

  /**
   * Update residual statistics with new prediction error.
   * @param step Future step index
   * @param residual Error vector [nTargets]
   */
  update(step: number, residual: ArrayLike<number>): void {
    const { nTargets, maxFutureSteps, windowSize } = this;

    if (step >= maxFutureSteps) return;

    // Store new residuals
    for (let t = 0; t < nTargets; t++) {
      const idx = (this.head * maxFutureSteps + step) * nTargets + t;
      const oldVal = this.residuals[idx];
      const newVal = residual[t];
      this.residuals[idx] = newVal;

      // Update running sum of squares
      const statsIdx = step * nTargets + t;
      if (this.count >= windowSize) {
        // Remove old value's contribution
        this.squaredResiduals[statsIdx] -= oldVal * oldVal;
      }
      this.squaredResiduals[statsIdx] += newVal * newVal;
    }

    this.head = (this.head + 1) % windowSize;
    if (this.count < windowSize) {
      this.count++;
    }

    // Update variance estimates
    this.updateVariances();
  }

  /**
   * Update variance estimates from accumulated statistics.
   */
  private updateVariances(): void {
    const { nTargets, maxFutureSteps } = this;
    const n = Math.max(1, this.count);

    for (let s = 0; s < maxFutureSteps; s++) {
      for (let t = 0; t < nTargets; t++) {
        const idx = s * nTargets + t;
        this.variances[idx] = Math.max(1e-8, this.squaredResiduals[idx] / n);
      }
    }
  }

  /**
   * Get standard deviation for a specific step and target.
   */
  getStd(step: number, target: number): number {
    const idx = step * this.nTargets + target;
    return Math.sqrt(this.variances[idx]);
  }

  /**
   * Get all standard deviations for a step.
   */
  getStepStds(step: number, out: number[]): void {
    const baseIdx = step * this.nTargets;
    for (let t = 0; t < this.nTargets; t++) {
      out[t] = Math.sqrt(this.variances[baseIdx + t]);
    }
  }

  /**
   * Get average loss (mean squared residual).
   */
  getAverageLoss(): number {
    if (this.count === 0) return 0;

    let sum = 0;
    const total = this.maxFutureSteps * this.nTargets;
    for (let i = 0; i < total; i++) {
      sum += this.variances[i];
    }
    return sum / total;
  }

  /**
   * Get confidence score based on recent loss.
   * Lower variance = higher confidence.
   */
  getConfidence(): number {
    const avgLoss = this.getAverageLoss();
    // Sigmoid-like mapping: confidence decreases as loss increases
    return 1 / (1 + avgLoss);
  }

  /**
   * Reset tracker.
   */
  reset(): void {
    this.residuals.fill(0);
    this.squaredResiduals.fill(0);
    this.means.fill(0);
    this.variances.fill(1);
    this.head = 0;
    this.count = 0;
  }

  /**
   * Get state for serialization.
   */
  getState(): object {
    return {
      residuals: Array.from(this.residuals),
      squaredResiduals: Array.from(this.squaredResiduals),
      variances: Array.from(this.variances),
      head: this.head,
      count: this.count,
    };
  }

  /**
   * Restore state from serialization.
   */
  setState(state: any): void {
    for (let i = 0; i < state.residuals.length; i++) {
      this.residuals[i] = state.residuals[i];
    }
    for (let i = 0; i < state.squaredResiduals.length; i++) {
      this.squaredResiduals[i] = state.squaredResiduals[i];
    }
    for (let i = 0; i < state.variances.length; i++) {
      this.variances[i] = state.variances[i];
    }
    this.head = state.head;
    this.count = state.count;
  }
}

// ============================================================
// OUTLIER DOWNWEIGHTER
// ============================================================

/**
 * Computes sample weights based on outlier detection.
 * Uses z-score thresholding to downweight outlier samples.
 */
export class OutlierDownweighter {
  private readonly threshold: number;
  private readonly minWeight: number;

  constructor(threshold: number = 3.0, minWeight: number = 0.1) {
    this.threshold = threshold;
    this.minWeight = minWeight;
  }

  /**
   * Compute sample weight based on prediction error.
   * Large errors (potential outliers) get lower weights.
   *
   * Math: weight = max(minWeight, 1 - sigmoid((|error| / std - threshold) * 2))
   *
   * @param error Prediction error vector
   * @param std Standard deviation estimate per target
   * @returns Sample weight in [minWeight, 1]
   */
  computeWeight(error: ArrayLike<number>, std: ArrayLike<number>): number {
    const nTargets = error.length;
    let maxZScore = 0;

    for (let t = 0; t < nTargets; t++) {
      const zScore = Math.abs(error[t]) / (std[t] + 1e-8);
      if (zScore > maxZScore) {
        maxZScore = zScore;
      }
    }

    if (maxZScore <= this.threshold) {
      return 1.0;
    }

    // Smooth transition using sigmoid
    const excess = maxZScore - this.threshold;
    const weight = 1 / (1 + Math.exp(excess));
    return Math.max(this.minWeight, weight);
  }

  /**
   * Compute weight from scalar loss.
   */
  computeWeightFromLoss(loss: number, expectedLoss: number): number {
    if (expectedLoss < 1e-8) return 1.0;

    const ratio = loss / expectedLoss;
    if (ratio <= this.threshold) {
      return 1.0;
    }

    const excess = ratio - this.threshold;
    const weight = 1 / (1 + Math.exp(excess));
    return Math.max(this.minWeight, weight);
  }
}

// ============================================================
// LOSS FUNCTION
// ============================================================

/**
 * Loss function utilities.
 */
export class LossFunction {
  private readonly epsilon: number;

  constructor(epsilon: number = 1e-8) {
    this.epsilon = epsilon;
  }

  /**
   * Compute mean squared error.
   * Math: MSE = mean((pred - target)^2)
   */
  mse(prediction: TensorView, target: TensorView): number {
    const size = prediction.shape.size;
    const pData = prediction.data;
    const pOffset = prediction.offset;
    const tData = target.data;
    const tOffset = target.offset;

    let sum = 0;
    for (let i = 0; i < size; i++) {
      const diff = pData[pOffset + i] - tData[tOffset + i];
      sum += diff * diff;
    }
    return sum / size;
  }

  /**
   * Compute mean squared error from arrays.
   */
  mseArrays(prediction: ArrayLike<number>, target: ArrayLike<number>): number {
    const size = prediction.length;
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const diff = prediction[i] - target[i];
      sum += diff * diff;
    }
    return sum / size;
  }

  /**
   * Compute Huber loss (smooth L1).
   * Math: L = 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
   */
  huber(
    prediction: TensorView,
    target: TensorView,
    delta: number = 1.0,
  ): number {
    const size = prediction.shape.size;
    const pData = prediction.data;
    const pOffset = prediction.offset;
    const tData = target.data;
    const tOffset = target.offset;

    let sum = 0;
    for (let i = 0; i < size; i++) {
      const diff = Math.abs(pData[pOffset + i] - tData[tOffset + i]);
      if (diff <= delta) {
        sum += 0.5 * diff * diff;
      } else {
        sum += delta * (diff - 0.5 * delta);
      }
    }
    return sum / size;
  }

  /**
   * Compute error vector (prediction - target) into output.
   */
  computeError(
    prediction: TensorView,
    target: TensorView,
    out: TensorView,
  ): void {
    TensorOps.sub(prediction, target, out);
  }
}

// ============================================================
// METRICS ACCUMULATOR
// ============================================================

/**
 * Accumulates training metrics over time.
 */
export class MetricsAccumulator {
  private lossSum: number;
  private lossCount: number;
  private gradNormSum: number;
  private gradNormCount: number;

  constructor() {
    this.lossSum = 0;
    this.lossCount = 0;
    this.gradNormSum = 0;
    this.gradNormCount = 0;
  }

  /**
   * Add a loss value.
   */
  addLoss(loss: number): void {
    if (isFinite(loss)) {
      this.lossSum += loss;
      this.lossCount++;
    }
  }

  /**
   * Add a gradient norm.
   */
  addGradNorm(norm: number): void {
    if (isFinite(norm)) {
      this.gradNormSum += norm;
      this.gradNormCount++;
    }
  }

  /**
   * Get average loss.
   */
  getAverageLoss(): number {
    return this.lossCount > 0 ? this.lossSum / this.lossCount : 0;
  }

  /**
   * Get average gradient norm.
   */
  getAverageGradNorm(): number {
    return this.gradNormCount > 0 ? this.gradNormSum / this.gradNormCount : 0;
  }

  /**
   * Get sample count.
   */
  getCount(): number {
    return this.lossCount;
  }

  /**
   * Reset accumulator.
   */
  reset(): void {
    this.lossSum = 0;
    this.lossCount = 0;
    this.gradNormSum = 0;
    this.gradNormCount = 0;
  }
}

// ============================================================
// MODEL CONFIG AND STATE
// ============================================================

/**
 * Internal model configuration (resolved from user config).
 */
export class ESNModelConfig {
  readonly maxSequenceLength: number;
  readonly maxFutureSteps: number;
  readonly reservoirSize: number;
  readonly spectralRadius: number;
  readonly leakRate: number;
  readonly inputScale: number;
  readonly biasScale: number;
  readonly reservoirSparsity: number;
  readonly inputSparsity: number;
  readonly activation: "tanh" | "relu";
  readonly useInputInReadout: boolean;
  readonly useBiasInReadout: boolean;
  readonly readoutTraining: "rls";
  readonly rlsLambda: number;
  readonly rlsDelta: number;
  readonly epsilon: number;
  readonly l2Lambda: number;
  readonly gradientClipNorm: number;
  readonly normalizationEpsilon: number;
  readonly normalizationWarmup: number;
  readonly outlierThreshold: number;
  readonly outlierMinWeight: number;
  readonly useDirectMultiHorizon: boolean;
  readonly residualWindowSize: number;
  readonly uncertaintyMultiplier: number;
  readonly weightInitScale: number;
  readonly seed: number;
  readonly verbose: boolean;

  constructor(userConfig: ESNRegressionConfig = {}) {
    this.maxSequenceLength = userConfig.maxSequenceLength ??
      DEFAULT_CONFIG.maxSequenceLength;
    this.maxFutureSteps = userConfig.maxFutureSteps ??
      DEFAULT_CONFIG.maxFutureSteps;
    this.reservoirSize = userConfig.reservoirSize ??
      DEFAULT_CONFIG.reservoirSize;
    this.spectralRadius = userConfig.spectralRadius ??
      DEFAULT_CONFIG.spectralRadius;
    this.leakRate = userConfig.leakRate ?? DEFAULT_CONFIG.leakRate;
    this.inputScale = userConfig.inputScale ?? DEFAULT_CONFIG.inputScale;
    this.biasScale = userConfig.biasScale ?? DEFAULT_CONFIG.biasScale;
    this.reservoirSparsity = userConfig.reservoirSparsity ??
      DEFAULT_CONFIG.reservoirSparsity;
    this.inputSparsity = userConfig.inputSparsity ??
      DEFAULT_CONFIG.inputSparsity;
    this.activation = userConfig.activation ?? DEFAULT_CONFIG.activation;
    this.useInputInReadout = userConfig.useInputInReadout ??
      DEFAULT_CONFIG.useInputInReadout;
    this.useBiasInReadout = userConfig.useBiasInReadout ??
      DEFAULT_CONFIG.useBiasInReadout;
    this.readoutTraining = userConfig.readoutTraining ??
      DEFAULT_CONFIG.readoutTraining;
    this.rlsLambda = userConfig.rlsLambda ?? DEFAULT_CONFIG.rlsLambda;
    this.rlsDelta = userConfig.rlsDelta ?? DEFAULT_CONFIG.rlsDelta;
    this.epsilon = userConfig.epsilon ?? DEFAULT_CONFIG.epsilon;
    this.l2Lambda = userConfig.l2Lambda ?? DEFAULT_CONFIG.l2Lambda;
    this.gradientClipNorm = userConfig.gradientClipNorm ??
      DEFAULT_CONFIG.gradientClipNorm;
    this.normalizationEpsilon = userConfig.normalizationEpsilon ??
      DEFAULT_CONFIG.normalizationEpsilon;
    this.normalizationWarmup = userConfig.normalizationWarmup ??
      DEFAULT_CONFIG.normalizationWarmup;
    this.outlierThreshold = userConfig.outlierThreshold ??
      DEFAULT_CONFIG.outlierThreshold;
    this.outlierMinWeight = userConfig.outlierMinWeight ??
      DEFAULT_CONFIG.outlierMinWeight;
    this.useDirectMultiHorizon = userConfig.useDirectMultiHorizon ??
      DEFAULT_CONFIG.useDirectMultiHorizon;
    this.residualWindowSize = userConfig.residualWindowSize ??
      DEFAULT_CONFIG.residualWindowSize;
    this.uncertaintyMultiplier = userConfig.uncertaintyMultiplier ??
      DEFAULT_CONFIG.uncertaintyMultiplier;
    this.weightInitScale = userConfig.weightInitScale ??
      DEFAULT_CONFIG.weightInitScale;
    this.seed = userConfig.seed ?? DEFAULT_CONFIG.seed;
    this.verbose = userConfig.verbose ?? DEFAULT_CONFIG.verbose;
  }
}

/**
 * Training state for the ESN model.
 */
export class TrainingState {
  sampleCount: number;
  totalLoss: number;
  lastGradNorm: number;
  lastSampleWeight: number;

  constructor() {
    this.sampleCount = 0;
    this.totalLoss = 0;
    this.lastGradNorm = 0;
    this.lastSampleWeight = 1;
  }

  reset(): void {
    this.sampleCount = 0;
    this.totalLoss = 0;
    this.lastGradNorm = 0;
    this.lastSampleWeight = 1;
  }

  getState(): object {
    return {
      sampleCount: this.sampleCount,
      totalLoss: this.totalLoss,
      lastGradNorm: this.lastGradNorm,
      lastSampleWeight: this.lastSampleWeight,
    };
  }

  setState(state: any): void {
    this.sampleCount = state.sampleCount;
    this.totalLoss = state.totalLoss;
    this.lastGradNorm = state.lastGradNorm;
    this.lastSampleWeight = state.lastSampleWeight;
  }
}

/**
 * Inference state for the ESN model.
 */
export class InferenceState {
  // Preallocated result arrays
  predictions: number[][];
  lowerBounds: number[][];
  upperBounds: number[][];

  constructor(maxFutureSteps: number, nTargets: number) {
    this.predictions = [];
    this.lowerBounds = [];
    this.upperBounds = [];
    for (let s = 0; s < maxFutureSteps; s++) {
      this.predictions.push(new Array(nTargets).fill(0));
      this.lowerBounds.push(new Array(nTargets).fill(0));
      this.upperBounds.push(new Array(nTargets).fill(0));
    }
  }

  resize(maxFutureSteps: number, nTargets: number): void {
    while (this.predictions.length < maxFutureSteps) {
      this.predictions.push(new Array(nTargets).fill(0));
      this.lowerBounds.push(new Array(nTargets).fill(0));
      this.upperBounds.push(new Array(nTargets).fill(0));
    }
    for (let s = 0; s < maxFutureSteps; s++) {
      while (this.predictions[s].length < nTargets) {
        this.predictions[s].push(0);
        this.lowerBounds[s].push(0);
        this.upperBounds[s].push(0);
      }
    }
  }
}

// ============================================================
// ESN MODEL
// ============================================================

/**
 * Core ESN model combining reservoir and readout.
 */
export class ESNModel {
  readonly config: ESNModelConfig;
  readonly reservoir: ESNReservoir;
  readonly readout: LinearReadout;
  readonly rlsState: RLSState;
  readonly rlsOptimizer: RLSOptimizer;

  private readonly arena: TensorArena;
  private readonly scratchArena: TensorArena;
  private readonly rng: RandomGenerator;

  // Scratch buffers (preallocated)
  readonly scratchInput: TensorView;
  readonly scratchTarget: TensorView;
  private readonly scratchPrediction: TensorView;
  readonly scratchError: TensorView;
  private readonly scratchNormalizedInput: TensorView;
  private readonly scratchReservoirState: TensorView;

  // Dimensions (set on first data)
  nFeatures: number;
  nTargets: number;
  private initialized: boolean;

  constructor(config: ESNModelConfig) {
    this.config = config;
    this.nFeatures = 0;
    this.nTargets = 0;
    this.initialized = false;
    this.rng = new RandomGenerator(config.seed);

    // Estimate arena size
    // We'll initialize components lazily once we know nFeatures/nTargets
    const estimatedSize = this.estimateArenaSize(config, 64, 16); // placeholder dims
    this.arena = new TensorArena(estimatedSize);
    this.scratchArena = new TensorArena(estimatedSize);

    // Placeholder initialization - real init happens in initialize()
    this.reservoir = null as any;
    this.readout = null as any;
    this.rlsState = null as any;
    this.rlsOptimizer = null as any;
    this.scratchInput = null as any;
    this.scratchTarget = null as any;
    this.scratchPrediction = null as any;
    this.scratchError = null as any;
    this.scratchNormalizedInput = null as any;
    this.scratchReservoirState = null as any;
  }

  /**
   * Estimate arena size needed.
   */
  private estimateArenaSize(
    config: ESNModelConfig,
    nFeatures: number,
    nTargets: number,
  ): number {
    const rs = config.reservoirSize;
    const outputDim = config.useDirectMultiHorizon
      ? nTargets * config.maxFutureSteps
      : nTargets;

    let inputDim = rs;
    if (config.useInputInReadout) inputDim += nFeatures;
    if (config.useBiasInReadout) inputDim += 1;

    // Reservoir params: Win[rs, nFeatures] + W[rs, rs] + bias[rs] + state[rs] + scratches
    const reservoirSize = rs * nFeatures + rs * rs + rs + rs * 4;

    // Readout: weights[outputDim, inputDim] + extendedState[inputDim] + output[outputDim]
    const readoutSize = outputDim * inputDim + inputDim + outputDim;

    // RLS: P[inputDim, inputDim] + gain[inputDim] + Pz[inputDim]
    const rlsSize = inputDim * inputDim + inputDim * 3;

    // Scratch buffers
    const scratchSize = nFeatures * 4 + nTargets * config.maxFutureSteps * 4 +
      rs * 4;

    return (reservoirSize + readoutSize + rlsSize + scratchSize) * 2;
  }

  /**
   * Initialize model with actual dimensions.
   */
  initialize(nFeatures: number, nTargets: number): void {
    if (
      this.initialized && this.nFeatures === nFeatures &&
      this.nTargets === nTargets
    ) {
      return;
    }

    this.nFeatures = nFeatures;
    this.nTargets = nTargets;

    // Reallocate arenas with proper size
    const size = this.estimateArenaSize(this.config, nFeatures, nTargets);
    (this as any).arena = new TensorArena(size);
    (this as any).scratchArena = new TensorArena(size);

    // Initialize reservoir
    (this as any).reservoir = new ESNReservoir(
      this.arena,
      this.config as Required<ESNRegressionConfig>,
      nFeatures,
      this.rng,
    );

    // Initialize readout
    const readoutConfig: ReadoutConfig = {
      reservoirSize: this.config.reservoirSize,
      nFeatures,
      nTargets,
      maxFutureSteps: this.config.maxFutureSteps,
      useInputInReadout: this.config.useInputInReadout,
      useBiasInReadout: this.config.useBiasInReadout,
      useDirectMultiHorizon: this.config.useDirectMultiHorizon,
    };
    (this as any).readout = new LinearReadout(this.arena, readoutConfig);

    // Initialize RLS
    (this as any).rlsState = new RLSState(
      this.arena,
      this.readout.params.inputDim,
      this.config.rlsLambda,
      this.config.rlsDelta,
    );
    (this as any).rlsOptimizer = new RLSOptimizer(
      this.rlsState,
      this.scratchArena,
      this.config.epsilon,
      this.config.l2Lambda,
    );

    // Allocate scratch buffers
    const outputDim = this.config.useDirectMultiHorizon
      ? nTargets * this.config.maxFutureSteps
      : nTargets;

    (this as any).scratchInput = this.scratchArena.allocVector(nFeatures);
    (this as any).scratchTarget = this.scratchArena.allocVector(outputDim);
    (this as any).scratchPrediction = this.scratchArena.allocVector(outputDim);
    (this as any).scratchError = this.scratchArena.allocVector(outputDim);
    (this as any).scratchNormalizedInput = this.scratchArena.allocVector(
      nFeatures,
    );
    (this as any).scratchReservoirState = this.scratchArena.allocVector(
      this.config.reservoirSize,
    );

    this.initialized = true;
  }

  /**
   * Check if model is initialized.
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Update reservoir state with normalized input.
   */
  updateReservoir(normalizedInput: TensorView): void {
    this.reservoir.update(normalizedInput);
  }

  /**
   * Compute forward pass (readout only).
   */
  forward(input: TensorView): void {
    this.readout.buildExtendedState(this.reservoir.state, input);
    this.readout.forward();
  }

  /**
   * Perform RLS update.
   */
  updateReadout(target: TensorView, sampleWeight: number): number {
    return this.rlsOptimizer.update(
      this.readout.extendedState,
      target,
      this.readout.output,
      this.readout.params.weights,
      sampleWeight,
    );
  }

  /**
   * Get trainable parameter count.
   */
  getParameterCount(): number {
    if (!this.initialized) return 0;
    return this.readout.params.weights.shape.size;
  }

  /**
   * Reset model state (reservoir + training state, but keep weights).
   */
  resetState(): void {
    if (this.reservoir) {
      this.reservoir.resetState();
    }
  }

  /**
   * Reset all (including RLS P matrix).
   */
  resetAll(): void {
    this.resetState();
    if (this.rlsState) {
      this.rlsState.initialize();
    }
    if (this.readout) {
      this.readout.params.weights.fill(0);
    }
  }

  /**
   * Copy reservoir state to scratch.
   */
  saveReservoirState(): void {
    this.reservoir.copyStateTo(this.scratchReservoirState);
  }

  /**
   * Restore reservoir state from scratch.
   */
  restoreReservoirState(): void {
    this.reservoir.restoreStateFrom(this.scratchReservoirState);
  }
}

// ============================================================
// SERIALIZATION HELPER
// ============================================================

/**
 * Helper for model serialization/deserialization.
 */
export class SerializationHelper {
  /**
   * Serialize model state to JSON string.
   */
  static serialize(model: ESNRegression): string {
    const state = model.getFullState();
    return JSON.stringify(state);
  }

  /**
   * Deserialize model state from JSON string.
   */
  static deserialize(model: ESNRegression, json: string): void {
    const state = JSON.parse(json);
    model.setFullState(state);
  }
}

// ============================================================
// ESN REGRESSION (MAIN PUBLIC API)
// ============================================================

/**
 * Echo State Network for Multivariate Regression.
 *
 * Main public API class providing:
 * - Online learning via fitOnline()
 * - Multi-step prediction via predict()
 * - Model inspection and serialization
 *
 * @example
 * ```typescript
 * const model = new ESNRegression({
 *   maxFutureSteps: 5,
 *   reservoirSize: 128
 * });
 *
 * // Online training
 * for (const batch of trainingData) {
 *   const result = model.fitOnline({
 *     xCoordinates: batch.features,
 *     yCoordinates: batch.targets
 *   });
 *   console.log(`Loss: ${result.averageLoss}`);
 * }
 *
 * // Prediction
 * const prediction = model.predict(3);  // 3 steps ahead
 * console.log(prediction.predictions);
 * ```
 */
export class ESNRegression {
  private readonly config: ESNModelConfig;
  private model: ESNModel;
  private inputNormalizer: WelfordNormalizer | null;
  private outputNormalizer: WelfordNormalizer | null;
  private inputBuffer: RingBuffer | null;
  private residualTracker: ResidualStatsTracker | null;
  private outlierDownweighter: OutlierDownweighter;
  private lossFunction: LossFunction;
  private trainingState: TrainingState;
  private inferenceState: InferenceState | null;
  private metricsAccumulator: MetricsAccumulator;
  private gradientAccumulator: GradientAccumulator;

  // Reusable result object (prevents allocation in hot path)
  private fitResultCache: FitResult;
  private predictionResultCache: PredictionResult | null;

  // Scratch arrays for internal use
  private scratchRow: number[];
  private scratchTargetRow: number[];
  private scratchStds: number[];
  private scratchWindow: Float64Array | null;

  // Recursive rollforward scratch (if enabled)
  private rollforwardBuffer: RingBuffer | null;
  private rollforwardStateBuffer: Float64Array | null;

  constructor(config: ESNRegressionConfig = {}) {
    this.config = new ESNModelConfig(config);
    this.model = new ESNModel(this.config);
    this.inputNormalizer = null;
    this.outputNormalizer = null;
    this.inputBuffer = null;
    this.residualTracker = null;
    this.outlierDownweighter = new OutlierDownweighter(
      this.config.outlierThreshold,
      this.config.outlierMinWeight,
    );
    this.lossFunction = new LossFunction(this.config.epsilon);
    this.trainingState = new TrainingState();
    this.inferenceState = null;
    this.metricsAccumulator = new MetricsAccumulator();
    this.gradientAccumulator = new GradientAccumulator(
      this.config.gradientClipNorm,
    );

    // Initialize reusable result objects
    this.fitResultCache = {
      samplesProcessed: 0,
      averageLoss: 0,
      gradientNorm: 0,
      driftDetected: false,
      sampleWeight: 1,
    };
    this.predictionResultCache = null;

    // Scratch arrays (will be properly sized on first use)
    this.scratchRow = [];
    this.scratchTargetRow = [];
    this.scratchStds = [];
    this.scratchWindow = null;
    this.rollforwardBuffer = null;
    this.rollforwardStateBuffer = null;
  }

  /**
   * Initialize model with dimensions from data.
   */
  private ensureInitialized(nFeatures: number, nTargets: number): void {
    if (
      this.model.isInitialized() &&
      this.model.nFeatures === nFeatures &&
      this.model.nTargets === nTargets
    ) {
      return;
    }

    // Initialize model
    this.model.initialize(nFeatures, nTargets);

    // Initialize normalizers
    this.inputNormalizer = new WelfordNormalizer(
      nFeatures,
      this.config.normalizationWarmup,
      this.config.normalizationEpsilon,
    );
    this.outputNormalizer = new WelfordNormalizer(
      nTargets * this.config.maxFutureSteps,
      this.config.normalizationWarmup,
      this.config.normalizationEpsilon,
    );

    // Initialize input buffer
    this.inputBuffer = new RingBuffer(this.config.maxSequenceLength, nFeatures);

    // Initialize residual tracker
    this.residualTracker = new ResidualStatsTracker(
      nTargets,
      this.config.maxFutureSteps,
      this.config.residualWindowSize,
    );

    // Initialize inference state
    this.inferenceState = new InferenceState(
      this.config.maxFutureSteps,
      nTargets,
    );

    // Initialize prediction result cache
    this.predictionResultCache = {
      predictions: [],
      lowerBounds: [],
      upperBounds: [],
      confidence: 0,
    };
    for (let s = 0; s < this.config.maxFutureSteps; s++) {
      this.predictionResultCache.predictions.push(new Array(nTargets).fill(0));
      this.predictionResultCache.lowerBounds.push(new Array(nTargets).fill(0));
      this.predictionResultCache.upperBounds.push(new Array(nTargets).fill(0));
    }

    // Initialize scratch arrays
    this.scratchRow = new Array(nFeatures).fill(0);
    this.scratchTargetRow = new Array(nTargets).fill(0);
    this.scratchStds = new Array(nTargets).fill(1);
    this.scratchWindow = new Float64Array(
      this.config.maxSequenceLength * nFeatures,
    );

    // Initialize rollforward scratch if not using direct multi-horizon
    if (!this.config.useDirectMultiHorizon) {
      this.rollforwardBuffer = new RingBuffer(
        this.config.maxSequenceLength,
        nFeatures,
      );
      this.rollforwardStateBuffer = new Float64Array(this.config.reservoirSize);
    }
  }

  /**
   * Train the model online with new data samples.
   *
   * CRITICAL BEHAVIOR:
   * 1. Each xCoordinates row is pushed to internal RingBuffer FIRST
   * 2. Then training proceeds with normalization, reservoir update, readout update
   *
   * @param params Training data
   * @param params.xCoordinates Input features [nSamples, nFeatures]
   * @param params.yCoordinates Target values [nSamples, nTargets] or [nSamples, nTargets * maxFutureSteps]
   * @returns Training results
   * @throws Error if xCoordinates.length !== yCoordinates.length
   *
   * @example
   * ```typescript
   * const result = model.fitOnline({
   *   xCoordinates: [[1.0, 2.0], [1.1, 2.1]],
   *   yCoordinates: [[0.5], [0.6]]
   * });
   * ```
   */
  fitOnline(params: {
    xCoordinates: number[][];
    yCoordinates: number[][];
  }): FitResult {
    const { xCoordinates, yCoordinates } = params;

    // STRICT VALIDATION: lengths must match exactly
    if (xCoordinates.length !== yCoordinates.length) {
      throw new Error(
        `xCoordinates.length (${xCoordinates.length}) must equal yCoordinates.length (${yCoordinates.length})`,
      );
    }

    if (xCoordinates.length === 0) {
      this.fitResultCache.samplesProcessed = 0;
      this.fitResultCache.averageLoss = 0;
      this.fitResultCache.gradientNorm = 0;
      this.fitResultCache.driftDetected = false;
      this.fitResultCache.sampleWeight = 1;
      return this.fitResultCache;
    }

    // Infer dimensions from first sample
    const nFeatures = xCoordinates[0].length;
    const nTargets = yCoordinates[0].length;

    // Validate nTargets matches expected output dimension
    const expectedTargets = this.config.useDirectMultiHorizon
      ? this.config.maxFutureSteps
      : 1;

    // Allow both single-target and multi-horizon targets
    const actualNTargets = this.config.useDirectMultiHorizon
      ? Math.floor(nTargets / this.config.maxFutureSteps)
      : nTargets;

    if (actualNTargets === 0) {
      throw new Error(
        `yCoordinates[0].length (${nTargets}) is invalid for maxFutureSteps=${this.config.maxFutureSteps}`,
      );
    }

    // Initialize model on first call
    this.ensureInitialized(nFeatures, actualNTargets);

    // Reset metrics for this batch
    this.metricsAccumulator.reset();

    const nSamples = xCoordinates.length;
    let lastGradNorm = 0;
    let lastSampleWeight = 1;

    for (let i = 0; i < nSamples; i++) {
      const x = xCoordinates[i];
      const y = yCoordinates[i];

      // ========================================================
      // STEP 1: Push X into RingBuffer FIRST (CRITICAL)
      // ========================================================
      this.inputBuffer!.push(x);

      // ========================================================
      // STEP 2: Update input normalizer statistics
      // ========================================================
      this.inputNormalizer!.updateFromArray(x, 0);

      // ========================================================
      // STEP 3: Normalize input and update reservoir
      // ========================================================
      this.copyArrayToScratchInput(x);
      this.inputNormalizer!.normalizeInPlace(this.model.scratchInput);
      this.model.updateReservoir(this.model.scratchInput);

      // ========================================================
      // STEP 4: Build target tensor
      // ========================================================
      this.buildTargetTensor(y);

      // ========================================================
      // STEP 5: Forward pass
      // ========================================================
      this.model.forward(this.model.scratchInput);

      // ========================================================
      // STEP 6: Compute loss
      // ========================================================
      const loss = this.lossFunction.mse(
        this.model.readout.output,
        this.model.scratchTarget,
      );

      // ========================================================
      // STEP 7: Compute outlier weight
      // ========================================================
      this.computeErrorForOutlier();
      const sampleWeight = this.outlierDownweighter.computeWeightFromLoss(
        loss,
        this.metricsAccumulator.getAverageLoss() + this.config.epsilon,
      );
      lastSampleWeight = sampleWeight;

      // ========================================================
      // STEP 8: RLS readout update
      // ========================================================
      const gradNorm = this.model.updateReadout(
        this.model.scratchTarget,
        sampleWeight,
      );
      lastGradNorm = gradNorm;

      // ========================================================
      // STEP 9: Update residual statistics
      // ========================================================
      this.updateResidualStats(y);

      // ========================================================
      // STEP 10: Update metrics
      // ========================================================
      this.metricsAccumulator.addLoss(loss);
      this.metricsAccumulator.addGradNorm(gradNorm);
      this.trainingState.sampleCount++;
      this.trainingState.totalLoss += loss;
    }

    // Update cached result
    this.fitResultCache.samplesProcessed = nSamples;
    this.fitResultCache.averageLoss = this.metricsAccumulator.getAverageLoss();
    this.fitResultCache.gradientNorm = lastGradNorm;
    this.fitResultCache.driftDetected = false; // No ADWIN, always false
    this.fitResultCache.sampleWeight = lastSampleWeight;
    this.trainingState.lastGradNorm = lastGradNorm;
    this.trainingState.lastSampleWeight = lastSampleWeight;

    return this.fitResultCache;
  }

  /**
   * Copy input array to scratch tensor.
   */
  private copyArrayToScratchInput(x: ArrayLike<number>): void {
    const data = this.model.scratchInput.data;
    const offset = this.model.scratchInput.offset;
    for (let i = 0; i < this.model.nFeatures; i++) {
      data[offset + i] = x[i];
    }
  }

  /**
   * Build target tensor from y array.
   */
  private buildTargetTensor(y: ArrayLike<number>): void {
    const data = this.model.scratchTarget.data;
    const offset = this.model.scratchTarget.offset;
    const outputDim = this.model.scratchTarget.shape.size;

    // Copy available targets, zero-pad if needed
    const copyLen = Math.min(y.length, outputDim);
    for (let i = 0; i < copyLen; i++) {
      data[offset + i] = y[i];
    }
    for (let i = copyLen; i < outputDim; i++) {
      data[offset + i] = 0;
    }
  }

  /**
   * Compute prediction error for outlier detection.
   */
  private computeErrorForOutlier(): void {
    const predData = this.model.readout.output.data;
    const predOffset = this.model.readout.output.offset;
    const targetData = this.model.scratchTarget.data;
    const targetOffset = this.model.scratchTarget.offset;
    const errData = this.model.scratchError.data;
    const errOffset = this.model.scratchError.offset;

    const size = this.model.scratchError.shape.size;
    for (let i = 0; i < size; i++) {
      errData[errOffset + i] = predData[predOffset + i] -
        targetData[targetOffset + i];
    }
  }

  /**
   * Update residual statistics from prediction.
   */
  private updateResidualStats(y: ArrayLike<number>): void {
    if (!this.residualTracker) return;

    const nTargets = this.model.nTargets;
    const maxSteps = this.config.maxFutureSteps;

    if (this.config.useDirectMultiHorizon) {
      // Multi-horizon: y has shape [nTargets * maxFutureSteps]
      for (let s = 0; s < maxSteps; s++) {
        for (let t = 0; t < nTargets; t++) {
          const idx = s * nTargets + t;
          const pred = this.model.readout.output.getLinear(idx);
          const actual = idx < y.length ? y[idx] : 0;
          this.scratchTargetRow[t] = pred - actual;
        }
        this.residualTracker.update(s, this.scratchTargetRow);
      }
    } else {
      // Single-step: y has shape [nTargets]
      for (let t = 0; t < nTargets; t++) {
        const pred = this.model.readout.output.getLinear(t);
        const actual = t < y.length ? y[t] : 0;
        this.scratchTargetRow[t] = pred - actual;
      }
      this.residualTracker.update(0, this.scratchTargetRow);
    }
  }

  /**
   * Generate predictions for future time steps.
   *
   * CRITICAL BEHAVIOR:
   * - Uses ONLY the internal RingBuffer for input window
   * - Window ends at the most recently ingested X (via fitOnline)
   * - Caller should NOT pass the latest X separately
   *
   * @param futureSteps Number of steps ahead to predict (1 to maxFutureSteps)
   * @returns Predictions with confidence bounds
   * @throws Error if futureSteps is out of range or model not trained
   *
   * @example
   * ```typescript
   * // Predict 3 steps ahead
   * const result = model.predict(3);
   * console.log(result.predictions);  // [[target1_step1, target2_step1], ...]
   * console.log(result.confidence);   // 0.0 to 1.0
   * ```
   */
  predict(futureSteps: number): PredictionResult {
    // Validate futureSteps
    if (futureSteps < 1 || futureSteps > this.config.maxFutureSteps) {
      throw new Error(
        `futureSteps must be between 1 and ${this.config.maxFutureSteps}, got ${futureSteps}`,
      );
    }

    // Check if model is trained
    if (!this.model.isInitialized()) {
      throw new Error("Model not initialized. Call fitOnline() first.");
    }

    if (this.inputBuffer!.isEmpty()) {
      throw new Error("No input data available. Call fitOnline() first.");
    }

    const { nTargets, nFeatures } = this.model;

    // ========================================================
    // Get latest input from RingBuffer (CRITICAL: no caller-provided X)
    // ========================================================
    this.inputBuffer!.getLatest(this.scratchRow);

    // ========================================================
    // Normalize input
    // ========================================================
    this.copyArrayToScratchInput(this.scratchRow);
    this.inputNormalizer!.normalizeInPlace(this.model.scratchInput);

    // ========================================================
    // Run forward pass (reservoir state is already updated from training)
    // For prediction, we use the current reservoir state
    // ========================================================
    if (this.config.useDirectMultiHorizon) {
      // Direct multi-horizon: single forward pass gives all horizons
      this.model.forward(this.model.scratchInput);

      // Extract predictions for requested steps
      for (let s = 0; s < futureSteps; s++) {
        this.model.readout.getStepOutputs(
          s,
          this.predictionResultCache!.predictions[s],
        );

        // Compute confidence bounds
        this.residualTracker!.getStepStds(s, this.scratchStds);
        const mult = this.config.uncertaintyMultiplier;

        for (let t = 0; t < nTargets; t++) {
          const pred = this.predictionResultCache!.predictions[s][t];
          const std = this.scratchStds[t];
          this.predictionResultCache!.lowerBounds[s][t] = pred - mult * std;
          this.predictionResultCache!.upperBounds[s][t] = pred + mult * std;
        }
      }
    } else {
      // Recursive roll-forward (if enabled)
      this.predictRecursive(futureSteps);
    }

    // Compute confidence
    this.predictionResultCache!.confidence = this.residualTracker!
      .getConfidence();

    return this.predictionResultCache!;
  }

  /**
   * Recursive prediction (for non-direct multi-horizon mode).
   * Uses scratch buffers to avoid mutating main state.
   */
  private predictRecursive(futureSteps: number): void {
    const { nTargets, nFeatures } = this.model;

    // Save current reservoir state
    this.model.saveReservoirState();

    // Copy current input to rollforward scratch
    for (let i = 0; i < nFeatures; i++) {
      this.scratchRow[i] = this.model.scratchInput.getLinear(i);
    }

    for (let s = 0; s < futureSteps; s++) {
      // Forward pass
      this.copyArrayToScratchInput(this.scratchRow);
      this.model.forward(this.model.scratchInput);

      // Extract predictions
      for (let t = 0; t < nTargets; t++) {
        this.predictionResultCache!.predictions[s][t] = this.model.readout
          .output.getLinear(t);
      }

      // Compute confidence bounds
      this.residualTracker!.getStepStds(
        Math.min(s, this.config.maxFutureSteps - 1),
        this.scratchStds,
      );
      const mult = this.config.uncertaintyMultiplier;

      for (let t = 0; t < nTargets; t++) {
        const pred = this.predictionResultCache!.predictions[s][t];
        const std = this.scratchStds[t];
        this.predictionResultCache!.lowerBounds[s][t] = pred - mult * std;
        this.predictionResultCache!.upperBounds[s][t] = pred + mult * std;
      }

      // If more steps needed, update reservoir with prediction
      // (This is a simplification - in practice you'd need to map predictions back to features)
      if (s < futureSteps - 1) {
        // Update reservoir state for next step
        this.model.updateReservoir(this.model.scratchInput);
      }
    }

    // Restore original reservoir state (don't corrupt main state)
    this.model.restoreReservoirState();
  }

  /**
   * Get model summary information.
   *
   * @returns Summary of model architecture and state
   */
  getModelSummary(): ModelSummary {
    return {
      totalParameters: this.model.getParameterCount(),
      receptiveField: this.config.maxSequenceLength,
      spectralRadius: this.config.spectralRadius,
      reservoirSize: this.config.reservoirSize,
      nFeatures: this.model.nFeatures,
      nTargets: this.model.nTargets,
      maxSequenceLength: this.config.maxSequenceLength,
      maxFutureSteps: this.config.maxFutureSteps,
      sampleCount: this.trainingState.sampleCount,
      useDirectMultiHorizon: this.config.useDirectMultiHorizon,
    };
  }

  /**
   * Get model weights for inspection.
   *
   * @returns Weight information including readout weights
   */
  getWeights(): WeightInfo {
    const weights: Array<{ name: string; shape: number[]; values: number[] }> =
      [];

    if (this.model.isInitialized()) {
      // Readout weights
      weights.push({
        name: "readout.weights",
        shape: [
          this.model.readout.params.outputDim,
          this.model.readout.params.inputDim,
        ],
        values: this.model.readout.params.weights.toArray(),
      });

      // Reservoir weights (fixed, not trained)
      weights.push({
        name: "reservoir.Win",
        shape: [this.config.reservoirSize, this.model.nFeatures],
        values: this.model.reservoir.params.Win.toArray(),
      });

      weights.push({
        name: "reservoir.W",
        shape: [this.config.reservoirSize, this.config.reservoirSize],
        values: this.model.reservoir.params.W.toArray(),
      });

      weights.push({
        name: "reservoir.bias",
        shape: [this.config.reservoirSize],
        values: this.model.reservoir.params.bias.toArray(),
      });
    }

    return { weights };
  }

  /**
   * Get normalization statistics.
   *
   * @returns Current normalization parameters
   */
  getNormalizationStats(): NormalizationStats {
    if (!this.inputNormalizer) {
      return {
        means: [],
        stds: [],
        count: 0,
        isActive: false,
      };
    }
    return this.inputNormalizer.getStats();
  }

  /**
   * Reset model to initial state.
   * Clears all training history and resets reservoir state.
   */
  reset(): void {
    if (this.model.isInitialized()) {
      this.model.resetAll();
    }

    if (this.inputBuffer) {
      this.inputBuffer.clear();
    }

    if (this.inputNormalizer) {
      this.inputNormalizer.reset();
    }

    if (this.outputNormalizer) {
      this.outputNormalizer.reset();
    }

    if (this.residualTracker) {
      this.residualTracker.reset();
    }

    this.trainingState.reset();
    this.metricsAccumulator.reset();
    this.gradientAccumulator.reset();
  }

  /**
   * Save model state to JSON string.
   *
   * @returns JSON string containing full model state
   */
  save(): string {
    return SerializationHelper.serialize(this);
  }

  /**
   * Load model state from JSON string.
   *
   * @param json JSON string from save()
   */
  load(json: string): void {
    SerializationHelper.deserialize(this, json);
  }

  /**
   * Get full internal state for serialization.
   * @internal
   */
  getFullState(): object {
    const state: any = {
      config: { ...this.config },
      trainingState: this.trainingState.getState(),
      nFeatures: this.model.nFeatures,
      nTargets: this.model.nTargets,
      initialized: this.model.isInitialized(),
    };

    if (this.model.isInitialized()) {
      state.reservoir = {
        state: this.model.reservoir.getState(),
        params: this.model.reservoir.getParamsState(),
      };
      state.readout = this.model.readout.getState();
      state.rlsState = this.model.rlsState.getState();
      state.inputNormalizer = this.inputNormalizer!.getState();
      state.outputNormalizer = this.outputNormalizer!.getState();
      state.inputBuffer = this.inputBuffer!.getState();
      state.residualTracker = this.residualTracker!.getState();
    }

    return state;
  }

  /**
   * Restore full internal state from serialization.
   * @internal
   */
  setFullState(state: any): void {
    // Reinitialize with saved dimensions
    if (state.initialized && state.nFeatures > 0 && state.nTargets > 0) {
      this.ensureInitialized(state.nFeatures, state.nTargets);

      // Restore component states
      this.model.reservoir.setState(state.reservoir.state);
      this.model.reservoir.setParamsState(state.reservoir.params);
      this.model.readout.setState(state.readout);
      this.model.rlsState.setState(state.rlsState);
      this.inputNormalizer!.setState(state.inputNormalizer);
      this.outputNormalizer!.setState(state.outputNormalizer);
      this.inputBuffer!.setState(state.inputBuffer);
      this.residualTracker!.setState(state.residualTracker);
    }

    // Restore training state
    this.trainingState.setState(state.trainingState);
  }
}

// ============================================================
// DEFAULT EXPORT
// ============================================================

export default ESNRegression;
