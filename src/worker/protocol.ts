import { SmaltmlError, type ErrorCode } from '../core/errors.js';
import { Matrix } from '../core/matrix.js';
import type { Label, Labels, ModelDocument } from '../core/model.js';

/** The end of a message channel: a Worker, the worker's own scope, or a MessagePort. */
export interface Port {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
  start?(): void;
}

/** A Matrix as it travels: its numbers in a transferred buffer. */
export interface WireMatrix {
  rows: number;
  cols: number;
  data: Float64Array;
  names?: readonly string[];
}

export type Operation = 'fit' | 'predict' | 'predictProba' | 'transform';

export interface Request {
  id: number;
  op: Operation;
  model: ModelDocument;
  /** X as numbers, or, when it holds text, as named columns. */
  X: WireMatrix | { table: Record<string, Float64Array | (string | null)[]> };
  y?: Labels;
  device?: 'auto' | 'cpu' | 'gpu';
}

export type Result =
  | { kind: 'model'; model: ModelDocument }
  | { kind: 'matrix'; matrix: WireMatrix }
  | { kind: 'values'; values: Float64Array }
  | { kind: 'labels'; labels: Label[] };

export type Response = { id: number; ok: true; result: Result } | { id: number; ok: false; error: { code?: ErrorCode; message: string } };

/** A Matrix for postMessage: a copy of its numbers, whose buffer can be transferred without detaching the caller's. */
export function toWire(m: Matrix): [WireMatrix, ArrayBuffer] {
  const data = m.data.slice();
  return [{ rows: m.rows, cols: m.cols, data, ...(m.names ? { names: [...m.names] } : {}) }, data.buffer as ArrayBuffer];
}

export function fromWire(w: WireMatrix): Matrix {
  return new Matrix(w.rows, w.cols, w.data, w.names);
}

export function toError(e: { code?: ErrorCode; message: string }): Error {
  return e.code ? new SmaltmlError(e.code, e.message) : new Error(e.message);
}
