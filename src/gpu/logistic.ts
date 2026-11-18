import type { Matrix } from '../core/matrix.js';
import { gpuDevice } from './index.js';
import { computePipeline, GPU_BUFFER, GPU_MAP_READ, gpuBufferLimit } from 'smaltkit/gpu';

/** Classes the GPU objective takes (two, as one score, up to this many); more stay on the CPU. */
export const GPU_MAX_CLASSES = 8;
const WG = 256;

const SHADER = /* wgsl */ `
struct Dims { n: u32, p: u32, k: u32, width: u32, intercept: u32, stride: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<u32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
const WG = ${WG}u;
const KMAX = ${GPU_MAX_CLASSES}u;
var<workgroup> r: array<f32, ${WG * GPU_MAX_CLASSES}>;
var<workgroup> losses: array<f32, ${WG}>;

fn log1pexp(z: f32) -> f32 {
  return max(z, 0.0) + log(1.0 + exp(-abs(z)));
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let i = gid.x;
  let t = lid.x;
  var loss = 0.0;
  for (var c = 0u; c < KMAX; c++) { r[t * KMAX + c] = 0.0; }
  if (i < dims.n) {
    var z: array<f32, ${GPU_MAX_CLASSES}>;
    for (var c = 0u; c < dims.k; c++) {
      var s = select(0.0, w[c * dims.width + dims.p], dims.intercept == 1u);
      for (var j = 0u; j < dims.p; j++) { s += X[j * dims.n + i] * w[c * dims.width + j]; }
      z[c] = s;
    }
    let yi = y[i];
    if (dims.k == 1u) {
      // Two classes: the second class's log-odds.
      let yf = f32(yi);
      loss = log1pexp(z[0]) - yf * z[0];
      r[t * KMAX] = 1.0 / (1.0 + exp(-z[0])) - yf;
    } else {
      var mx = z[0];
      for (var c = 1u; c < dims.k; c++) { mx = max(mx, z[c]); }
      var sum = 0.0;
      for (var c = 0u; c < dims.k; c++) { sum += exp(z[c] - mx); }
      let lse = mx + log(sum);
      loss = lse - z[yi];
      for (var c = 0u; c < dims.k; c++) { r[t * KMAX + c] = exp(z[c] - lse) - select(0.0, 1.0, c == yi); }
    }
  }
  losses[t] = loss;
  workgroupBarrier();
  // This workgroup's part of the gradient: entry (c, j) sums x_ij r_ic over its rows; j = p is the intercept.
  let row0 = wid.x * WG;
  let total = dims.k * dims.width;
  for (var idx = t; idx < total; idx += WG) {
    let c = idx / dims.width;
    let j = idx % dims.width;
    var s = 0.0;
    for (var q = 0u; q < WG; q++) {
      let row = row0 + q;
      if (row < dims.n) {
        let x = select(1.0, X[j * dims.n + row], j < dims.p);
        s += x * r[q * KMAX + c];
      }
    }
    out[wid.x * dims.stride + idx] = s;
  }
  if (t == 0u) {
    var s = 0.0;
    for (var q = 0u; q < WG; q++) { s += losses[q]; }
    out[wid.x * dims.stride + total] = s;
  }
}`;

/** The logistic objective's sums on the GPU, for one X and y kept there between evaluations. */
export interface GpuLogistic {
  /** The log loss summed over the rows (returned) and, in `grad`, the summed gradient: as binaryLogisticPart(). */
  sums(x: Float64Array, grad: Float64Array): Promise<number>;
  destroy(): void;
}

const { MAP_READ, COPY_SRC, COPY_DST, UNIFORM, STORAGE } = GPU_BUFFER;

/**
 * Uploads X (as 32-bit floats) and the class indexes, and returns the objective's sums on the GPU; null where there is
 * no GPU, too many classes, or X is larger than the GPU's buffers, so the fit runs on the CPU. Each workgroup of 256
 * rows sums in 32-bit floats; the workgroups' sums are added on the CPU in 64-bit floats.
 */
export async function logisticOnGpu(X: Matrix, y: Int32Array, K: number, intercept: boolean): Promise<GpuLogistic | null> {
  const d = await gpuDevice();
  if (!d || K > GPU_MAX_CLASSES) return null;
  const n = X.rows;
  const p = X.cols;
  const width = p + (intercept ? 1 : 0);
  const limit = gpuBufferLimit(d);
  const groups = Math.ceil(n / WG);
  const stride = K * width + 1;
  if (n * p * 4 > limit || groups * stride * 4 > limit || groups > d.limits.maxComputeWorkgroupsPerDimension) return null;
  const pipeline = computePipeline(d, SHADER);
  const make = (size: number, usage: number) => d.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
  const xBuf = make(n * p * 4, STORAGE | COPY_DST);
  d.queue.writeBuffer(xBuf, 0, Float32Array.from(X.data));
  const yBuf = make(n * 4, STORAGE | COPY_DST);
  d.queue.writeBuffer(yBuf, 0, Uint32Array.from(y));
  const dims = make(32, UNIFORM | COPY_DST);
  d.queue.writeBuffer(dims, 0, new Uint32Array([n, p, K, width, intercept ? 1 : 0, stride, 0, 0]));
  const wBuf = make(K * width * 4, STORAGE | COPY_DST);
  const outBuf = make(groups * stride * 4, STORAGE | COPY_SRC);
  const read = make(groups * stride * 4, MAP_READ | COPY_DST);
  const bind = d.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [dims, xBuf, yBuf, wBuf, outBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  return {
    async sums(x: Float64Array, grad: Float64Array): Promise<number> {
      d.queue.writeBuffer(wBuf, 0, Float32Array.from(x));
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(groups);
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, read, 0, groups * stride * 4);
      d.queue.submit([enc.finish()]);
      await read.mapAsync(GPU_MAP_READ);
      const parts = new Float32Array(read.getMappedRange().slice(0, groups * stride * 4));
      read.unmap();
      grad.fill(0);
      let loss = 0;
      for (let g = 0; g < groups; g++) {
        const o = g * stride;
        for (let k = 0; k < stride - 1; k++) grad[k]! += parts[o + k]!;
        loss += parts[o + stride - 1]!;
      }
      return loss;
    },
    destroy() {
      for (const b of [xBuf, yBuf, dims, wBuf, outBuf, read]) b.destroy();
    },
  };
}
