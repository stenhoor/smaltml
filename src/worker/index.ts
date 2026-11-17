/**
 * Fits and predictions in a Web Worker, so a long fit doesn't freeze the page (ADR 0024, section 5). The worker fits a
 * copy of your model and hands back an ordinary fitted model; predictions run there too and come back as values.
 * Code already off the page's main thread (a notebook kernel, Node) can call the models directly instead.
 * @module
 */
import { fail } from '../core/errors.js';
import { isText, numericMatrix, toTable, type TableInput } from '../core/table.js';
import { Matrix, type MatrixInput } from '../core/matrix.js';
import type { Label, Labels, Model } from '../core/model.js';
import type { Device } from '../gpu/index.js';
import { loadModel } from '../io/index.js';
import { fromWire, toError, toWire, type Operation, type Port, type Request, type Response, type Result } from './protocol.js';

export type { Port } from './protocol.js';
export { serve } from './runtime.js';

export interface PredictOptions {
  device?: Device;
}

/** smaltml in a worker: the same models, run off the calling thread. */
export interface MlWorker {
  /** Fits a copy of `model` in the worker and returns the fitted copy, of the same class. `model` itself is unchanged. */
  fit<M extends Model>(model: M, X: TableInput, y?: Labels, options?: { sampleWeight?: ArrayLike<number> }): Promise<M>;
  /** The fitted model's predictions, computed in the worker. */
  predict(model: Model, X: TableInput, options?: PredictOptions): Promise<Float64Array | Label[]>;
  /** The fitted classifier's probabilities, computed in the worker. */
  predictProba(model: Model, X: TableInput, options?: PredictOptions): Promise<Matrix>;
  /** The fitted transformer's output, computed in the worker. */
  transform(model: Model, X: TableInput): Promise<Matrix>;
  /** Stops the worker (when it is one this call started), and fails the requests still waiting. */
  terminate(): void;
}

/** Talks to smaltml served on `port` (a Worker running smaltml's worker script, or a port that serve() answers). */
export function connect(port: Port & { terminate?(): void }): MlWorker {
  let next = 1;
  const waiting = new Map<number, { resolve(r: Result): void; reject(e: Error): void }>();
  port.addEventListener('message', (event) => {
    const res = event.data as Response;
    const w = waiting.get(res?.id);
    if (!w) return;
    waiting.delete(res.id);
    if (res.ok) w.resolve(res.result);
    else w.reject(toError(res.error));
  });
  port.start?.();
  const call = (op: Operation, model: Model, X: TableInput, extra: Partial<Request> = {}): Promise<Result> => {
    const t = toTable(X as TableInput);
    // Numbers travel as a Matrix, a table with text as named columns; both copied by postMessage, never transferred.
    const wire: Request['X'] = t.columns.some(isText)
      ? { table: Object.fromEntries(t.columns.map((c, j) => [t.named ? t.names[j]! : `x${j}`, c])) }
      : toWire(numericMatrix(X as TableInput));
    const id = next++;
    const req: Request = { id, op, model: model.toJSON(), X: wire, ...extra };
    return new Promise((resolve, reject) => {
      waiting.set(id, { resolve, reject });
      port.postMessage(req);
    });
  };
  return {
    async fit<M extends Model>(model: M, X: TableInput, y?: Labels, options: { sampleWeight?: ArrayLike<number> } = {}): Promise<M> {
      const r = await call('fit', model, X, {
        ...(y === undefined ? {} : { y: Array.from(y as ArrayLike<number | string>) as Labels }),
        ...(options.sampleWeight === undefined ? {} : { sampleWeight: Array.from(options.sampleWeight) }),
      });
      if (r.kind !== 'model') fail('ML_FORMAT', 'the worker sent no model back');
      return loadModel(r.model) as M;
    },
    async predict(model, X, options = {}) {
      const r = await call('predict', model, X, options.device ? { device: options.device } : {});
      if (r.kind === 'values') return r.values;
      if (r.kind === 'labels') return r.labels;
      return fail('ML_FORMAT', 'the worker sent no predictions back');
    },
    async predictProba(model, X, options = {}) {
      const r = await call('predictProba', model, X, options.device ? { device: options.device } : {});
      return r.kind === 'matrix' ? fromWire(r.matrix) : fail('ML_FORMAT', 'the worker sent no probabilities back');
    },
    async transform(model, X) {
      const r = await call('transform', model, X);
      return r.kind === 'matrix' ? fromWire(r.matrix) : fail('ML_FORMAT', 'the worker sent no matrix back');
    },
    terminate() {
      port.terminate?.();
      for (const w of waiting.values()) w.reject(new Error('the smaltml worker was stopped'));
      waiting.clear();
    },
  };
}

/**
 * Starts smaltml's worker script and connects to it. `url` defaults to `smaltml-worker.js` beside this module, where
 * the package builds it; pass your bundler's URL for it (`new URL('@smaltworks/smaltml/worker-script', import.meta.url)`)
 * when it moves files.
 */
export function startWorker(url: string | URL = new URL('./smaltml-worker.js', import.meta.url)): MlWorker {
  if (typeof Worker === 'undefined') fail('ML_PARAM', 'Web Workers aren\'t available here: call the models directly');
  return connect(new Worker(url, { type: 'module', name: 'smaltml' }));
}
