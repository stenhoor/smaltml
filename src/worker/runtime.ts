/**
 * The worker side of `@smaltworks/smaltml/worker`: answers fit and predict requests on a port. The package's worker
 * script (`smaltml-worker.js`) calls serve(self); call it yourself to serve smaltml from a worker of your own.
 * @module
 */
import { SmaltmlError } from '../core/errors.js';
import type { MatrixInput } from '../core/matrix.js';
import { isClassifier, isPredictor, isTransformer, type Model } from '../core/model.js';
import { loadModel } from '../io/index.js';
import { fromWire, toWire, type Port, type Request, type Response, type Result } from './protocol.js';

/** Answers smaltml requests that arrive on `port`, one at a time, in order. */
export function serve(port: Port): void {
  let queue = Promise.resolve();
  port.addEventListener('message', (event) => {
    const req = event.data as Request;
    if (!req || typeof req !== 'object' || typeof req.id !== 'number') return;
    queue = queue.then(async () => {
      let response: Response;
      try {
        // Results are copied by postMessage, never transferred (protocol.ts says why).
        response = { id: req.id, ok: true, result: await handle(req) };
      } catch (e) {
        const err = e as Error;
        response = { id: req.id, ok: false, error: { ...(e instanceof SmaltmlError ? { code: e.code } : {}), message: err?.message ?? String(e) } };
      }
      port.postMessage(response);
    });
  });
  port.start?.();
}

async function handle(req: Request): Promise<Result> {
  const model: Model = loadModel(req.model);
  // A table with text reaches the model as named columns; models that need numbers say so with ML_VALUE.
  const X = ('table' in req.X ? req.X.table : fromWire(req.X)) as MatrixInput;
  const options = req.device ? { device: req.device } : {};
  switch (req.op) {
    case 'fit':
      await (model as unknown as { fit(X: unknown, y?: unknown, o?: object): Promise<unknown> }).fit(X, req.y, req.sampleWeight ? { sampleWeight: req.sampleWeight } : {});
      return { kind: 'model', model: model.toJSON() };
    case 'predict': {
      if (!isPredictor(model)) throw new SmaltmlError('ML_PARAM', `${model.kind} doesn't predict`);
      const out = await model.predict(X, options);
      if (out instanceof Float64Array) return { kind: 'values', values: out };
      return { kind: 'labels', labels: out };
    }
    case 'predictProba': {
      if (!isClassifier(model) && !(model.kind === 'Pipeline' && 'predictProba' in model)) {
        throw new SmaltmlError('ML_PARAM', `${model.kind} doesn't give probabilities`);
      }
      const matrix = toWire(await (model as unknown as { predictProba(X: unknown, o: unknown): Promise<import('../core/matrix.js').Matrix> }).predictProba(X, options));
      return { kind: 'matrix', matrix };
    }
    case 'transform': {
      if (!isTransformer(model)) throw new SmaltmlError('ML_PARAM', `${model.kind} doesn't transform`);
      const matrix = toWire(await model.transform(X));
      return { kind: 'matrix', matrix };
    }
    default:
      throw new SmaltmlError('ML_PARAM', `unknown operation ${String((req as { op: unknown }).op)}`);
  }
}
