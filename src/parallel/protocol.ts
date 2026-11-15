/** What the main thread and the thread pool's workers say to each other. */

/** A matrix in shared memory: its numbers in a SharedArrayBuffer, column by column. */
export interface SharedMatrix {
  buffer: SharedArrayBuffer;
  rows: number;
  cols: number;
}

/** The work for one worker: an operation over a run of whole chunks of rows. */
export type Task =
  | { op: 'gram'; X: SharedMatrix; mean: Float64Array | null; weights: SharedArrayBuffer | null; ones: boolean; chunks: [number, number][] }
  | { op: 'xtv'; X: SharedMatrix; v: SharedArrayBuffer; mean: Float64Array | null; chunks: [number, number][] }
  | { op: 'xw'; X: SharedMatrix; w: Float64Array; b: number; mean: Float64Array | null; out: SharedArrayBuffer; chunks: [number, number][] }
  | { op: 'logistic'; X: SharedMatrix; y: SharedArrayBuffer; K: number; intercept: boolean; x: Float64Array; chunks: [number, number][] };

export interface Request {
  id: number;
  task: Task;
}

/** One result per chunk, in the order of the task's chunks: a vector, or a loss and its gradient. */
export type Response =
  | { id: number; ok: true; parts: { loss?: number; values: Float64Array }[] }
  | { id: number; ok: false; message: string };
