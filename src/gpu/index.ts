/**
 * WebGPU: the heavy steps on the GPU where the browser has one (ADR 0024, section 3). The CPU path is the product and
 * always works; the GPU is an accelerator. It computes in 32-bit floats, so its results agree with the CPU's within a
 * stated tolerance, not bit for bit.
 * @module
 */
import { fail } from '../core/errors.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';

/**
 * Where work runs: `'auto'` (the default) uses the GPU when the browser has WebGPU and the work is large enough to gain
 * from it, `'cpu'` always the CPU, `'gpu'` always the GPU (failing with ML_PARAM where there is none).
 */
export type Device = 'auto' | 'cpu' | 'gpu';

/** Options for a call that can run on the GPU. */
export interface DeviceOptions {
  device?: Device;
}

/**
 * The multiply-adds below which `'auto'` stays on the CPU: moving data to the GPU and back costs more than it saves on
 * small work. A guess until the benchmarks set it per model.
 */
export const GPU_MIN_WORK = 1 << 22;

let device: Promise<GPUDevice | null> | undefined;

/** The WebGPU device smaltml uses, or null where there is none (Node, or a browser without WebGPU). Asked for once. */
export function gpuDevice(): Promise<GPUDevice | null> {
  return (device ??= (async () => {
    const gpu = (globalThis as { navigator?: Navigator }).navigator?.gpu;
    if (!gpu) return null;
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter) return null;
      const d = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
      });
      // A lost device is asked for again next time.
      void d.lost.then(() => {
        device = undefined;
      });
      return d;
    } catch {
      return null;
    }
  })());
}

/** Whether this browser gives smaltml a WebGPU device. */
export async function gpuAvailable(): Promise<boolean> {
  return (await gpuDevice()) !== null;
}

/** Whether work of this many multiply-adds runs on the GPU, for this device choice. */
export async function useGpu(choice: Device | undefined, work: number): Promise<boolean> {
  const want = choice ?? 'auto';
  if (want !== 'auto' && want !== 'cpu' && want !== 'gpu') fail('ML_PARAM', `device is 'auto', 'cpu' or 'gpu', not ${String(want)}`);
  if (want === 'cpu') return false;
  if (want === 'auto' && work < GPU_MIN_WORK) return false;
  const available = await gpuAvailable();
  if (want === 'gpu' && !available) fail('ML_PARAM', "device 'gpu' needs WebGPU, which isn't available here: use 'auto' or 'cpu'");
  return available;
}

/**
 * The matrix product A · B: A is n × p, B is p × k, the result n × k. On the CPU in 64-bit floats, or on the GPU in
 * 32-bit floats (relative error around 1e-6 of the terms' size), as `device` chooses.
 */
export async function matMul(A: MatrixInput, B: MatrixInput, options: DeviceOptions = {}): Promise<Matrix> {
  const a = Matrix.from(A);
  const b = Matrix.from(B);
  if (a.cols !== b.rows) fail('ML_SHAPE', `can't multiply ${a.rows} × ${a.cols} by ${b.rows} × ${b.cols}: the inner sizes differ`);
  if (await useGpu(options.device, a.rows * a.cols * b.cols)) return gpuMatMul(a, b);
  return matMulSync(a, b);
}

/** A · B on the CPU, in 64-bit floats. */
export function matMulSync(A: MatrixInput, B: MatrixInput): Matrix {
  const a = Matrix.from(A);
  const b = Matrix.from(B);
  if (a.cols !== b.rows) fail('ML_SHAPE', `can't multiply ${a.rows} × ${a.cols} by ${b.rows} × ${b.cols}: the inner sizes differ`);
  const n = a.rows;
  const p = a.cols;
  const k = b.cols;
  const out = new Float64Array(n * k);
  for (let j = 0; j < k; j++) {
    const o = j * n;
    for (let l = 0; l < p; l++) {
      const w = b.data[j * p + l]!;
      if (w === 0) continue;
      const c = l * n;
      for (let i = 0; i < n; i++) out[o + i]! += a.data[c + i]! * w;
    }
  }
  return new Matrix(n, k, out);
}

// The kernel: a 16 × 16 tile of the output per workgroup, with the matching tiles of A and B staged in workgroup memory.
// Everything is column-major, as Matrix is.
const TILE = 16;
const SHADER = /* wgsl */ `
struct Dims { n: u32, p: u32, k: u32, pad: u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> ta: array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> tb: array<array<f32, ${TILE}>, ${TILE}>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let i = g.x;
  let j = g.y;
  var sum = 0.0;
  let tiles = (dims.p + ${TILE}u - 1u) / ${TILE}u;
  for (var t = 0u; t < tiles; t++) {
    let ca = t * ${TILE}u + l.y;
    let rb = t * ${TILE}u + l.x;
    ta[l.y][l.x] = select(0.0, a[ca * dims.n + i], i < dims.n && ca < dims.p);
    tb[l.y][l.x] = select(0.0, b[j * dims.p + rb], j < dims.k && rb < dims.p);
    workgroupBarrier();
    for (var q = 0u; q < ${TILE}u; q++) {
      sum += ta[q][l.x] * tb[l.y][q];
    }
    workgroupBarrier();
  }
  if (i < dims.n && j < dims.k) {
    out[j * dims.n + i] = sum;
  }
}`;

// The WebGPU spec's buffer usage and map mode flags, as numbers: TypeScript's DOM library types them but doesn't declare
// the GPUBufferUsage and GPUMapMode objects.
const MAP_READ = 0x0001;
const COPY_SRC = 0x0004;
const COPY_DST = 0x0008;
const UNIFORM = 0x0040;
const STORAGE = 0x0080;
const MAP_MODE_READ = 0x0001;

const pipelines = new WeakMap<GPUDevice, GPUComputePipeline>();

function pipelineFor(d: GPUDevice): GPUComputePipeline {
  let p = pipelines.get(d);
  if (!p) {
    p = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: SHADER }), entryPoint: 'main' } });
    pipelines.set(d, p);
  }
  return p;
}

/** A · B on the GPU, in 32-bit floats, in chunks of rows that fit the device's buffer limits. */
async function gpuMatMul(a: Matrix, b: Matrix): Promise<Matrix> {
  const d = await gpuDevice();
  if (!d) return fail('ML_PARAM', "device 'gpu' needs WebGPU, which isn't available here");
  const n = a.rows;
  const p = a.cols;
  const k = b.cols;
  const out = new Float64Array(n * k);
  if (n === 0 || k === 0) return new Matrix(n, k, out);
  if (p === 0) return new Matrix(n, k, out);
  const limit = Math.min(d.limits.maxStorageBufferBindingSize, d.limits.maxBufferSize);
  if (p * k * 4 > limit) fail('ML_SHAPE', `B (${p} × ${k}) is larger than this GPU's buffers`);
  const maxGroups = d.limits.maxComputeWorkgroupsPerDimension;
  const rowsPerChunk = Math.max(1, Math.min(Math.floor(limit / (4 * Math.max(p, k))), maxGroups * TILE));
  const pipeline = pipelineFor(d);
  const usage = STORAGE | COPY_DST;
  const bBuf = d.createBuffer({ size: p * k * 4, usage });
  d.queue.writeBuffer(bBuf, 0, Float32Array.from(b.data));
  try {
    for (let r0 = 0; r0 < n; r0 += rowsPerChunk) {
      const m = Math.min(rowsPerChunk, n - r0);
      // This chunk's rows of A, column-major.
      const chunk = new Float32Array(m * p);
      for (let c = 0; c < p; c++) chunk.set(a.data.subarray(c * n + r0, c * n + r0 + m), c * m);
      const dims = d.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
      d.queue.writeBuffer(dims, 0, new Uint32Array([m, p, k, 0]));
      const aBuf = d.createBuffer({ size: chunk.byteLength, usage });
      d.queue.writeBuffer(aBuf, 0, chunk);
      const size = m * k * 4;
      const oBuf = d.createBuffer({ size, usage: STORAGE | COPY_SRC });
      const read = d.createBuffer({ size, usage: MAP_READ | COPY_DST });
      const bind = d.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: dims } },
          { binding: 1, resource: { buffer: aBuf } },
          { binding: 2, resource: { buffer: bBuf } },
          { binding: 3, resource: { buffer: oBuf } },
        ],
      });
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(m / TILE), Math.ceil(k / TILE));
      pass.end();
      enc.copyBufferToBuffer(oBuf, 0, read, 0, size);
      d.queue.submit([enc.finish()]);
      await read.mapAsync(MAP_MODE_READ);
      const result = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      for (let j = 0; j < k; j++) {
        for (let i = 0; i < m; i++) out[j * n + r0 + i] = result[j * m + i]!;
      }
      for (const buf of [dims, aBuf, oBuf, read]) buf.destroy();
    }
  } finally {
    bBuf.destroy();
  }
  return new Matrix(n, k, out);
}
