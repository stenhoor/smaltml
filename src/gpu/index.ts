/**
 * WebGPU: the heavy steps on the GPU where the browser has one (ADR 0024, section 3). The CPU path is the product and
 * always works; the GPU is an accelerator. It computes in 32-bit floats, so its results agree with the CPU's within a
 * stated tolerance, not bit for bit.
 * @module
 */
import { fail } from '../core/errors.js';
import { kernelsReady } from '../core/kernel-set.js';
import { matMulKernel } from '../core/kernels.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import { gpuDevice as kitDevice, gpuMatMul as kitMatMul, gpuMatMulFits } from 'smaltkit/gpu';

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

/**
 * The WebGPU device smaltml uses, or null where there is none (Node, or a browser without WebGPU): smaltkit's compute
 * device (`smaltkit/gpu`), asked for once and again after it is lost.
 */
export function gpuDevice(): Promise<GPUDevice | null> {
  return kitDevice();
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
 * Whether a prediction runs on the GPU: only with `device: 'gpu'`. `'auto'` (the default) keeps predictions on the CPU,
 * which was faster for every prediction measured, at every size (ADR 0034, `npm run bench:gpu`). So on the CPU an
 * asynchronous prediction gives exactly what its `*Sync` twin gives, with or without WebGPU.
 */
export async function predictOnGpu(choice: Device | undefined): Promise<boolean> {
  if (choice === undefined || choice === 'auto') return false;
  return useGpu(choice, 0);
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
  await kernelsReady();
  return matMulSync(a, b);
}

/** A · B on the CPU, in 64-bit floats. */
export function matMulSync(A: MatrixInput, B: MatrixInput): Matrix {
  const a = Matrix.from(A);
  const b = Matrix.from(B);
  if (a.cols !== b.rows) fail('ML_SHAPE', `can't multiply ${a.rows} × ${a.cols} by ${b.rows} × ${b.cols}: the inner sizes differ`);
  return new Matrix(a.rows, b.cols, matMulKernel(a.rows, a.cols, b.cols, a.data, b.data));
}

/**
 * A · B on the GPU, in 32-bit floats: smaltkit's matrix product (`smaltkit/gpu`), a 16 × 16 tile per workgroup, A in
 * chunks of rows that fit the device's buffers.
 */
async function gpuMatMul(a: Matrix, b: Matrix): Promise<Matrix> {
  const d = await gpuDevice();
  if (!d) return fail('ML_PARAM', "device 'gpu' needs WebGPU, which isn't available here");
  const [n, p, k] = [a.rows, a.cols, b.cols];
  if (n > 0 && p > 0 && k > 0 && !gpuMatMulFits(d, p, k)) fail('ML_SHAPE', `B (${p} × ${k}) is larger than this GPU's buffers`);
  return new Matrix(n, k, await kitMatMul(d, n, p, k, a.data, b.data));
}
