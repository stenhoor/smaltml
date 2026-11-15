// The thread pool's worker script (dist/smaltml-threads.js): runs the kernels over the chunks of rows it is given, on
// matrices in shared memory, and sends back one part per chunk. In Node it runs under worker_threads; in a browser,
// as a module Worker on a cross-origin isolated page.
import { binaryLogisticPart, gramPart, multinomialLogisticPart, xtvPart, xw } from '../core/kernels.js';
import { Matrix } from '../core/matrix.js';
import type { Request, Response, SharedMatrix } from './protocol.js';

const nodeModule = (name: string): Promise<unknown> => import(/* @vite-ignore */ name);

interface Port {
  postMessage(message: unknown): void;
}

const views = new WeakMap<SharedArrayBuffer, Matrix>();
function matrix(m: SharedMatrix): Matrix {
  let v = views.get(m.buffer);
  if (!v || v.rows !== m.rows || v.cols !== m.cols) {
    v = new Matrix(m.rows, m.cols, new Float64Array(m.buffer));
    views.set(m.buffer, v);
  }
  return v;
}

function handle(req: Request): Response {
  const t = req.task;
  const X = matrix(t.X);
  const mean = 'mean' in t ? (t.mean ?? undefined) : undefined;
  switch (t.op) {
    case 'gram': {
      const w = t.weights ? new Float64Array(t.weights) : undefined;
      return { id: req.id, ok: true, parts: t.chunks.map(([a, b]) => ({ values: gramPart(X, mean, a, b, w, t.ones) })) };
    }
    case 'xtv': {
      const v = new Float64Array(t.v);
      return { id: req.id, ok: true, parts: t.chunks.map(([a, b]) => ({ values: xtvPart(X, v, mean, a, b) })) };
    }
    case 'xw': {
      const out = new Float64Array(t.out) as unknown as Float64Array<ArrayBuffer>;
      for (const [a, b] of t.chunks) xw(X, t.w, t.b, mean, out, a, b);
      return { id: req.id, ok: true, parts: [] };
    }
    case 'logistic': {
      const y = new Int32Array(t.y);
      const width = X.cols + (t.intercept ? 1 : 0);
      const parts = t.chunks.map(([a, b]) => {
        const g = new Float64Array(t.K === 1 ? width : t.K * width);
        const loss = t.K === 1 ? binaryLogisticPart(X, y, t.intercept, t.x, g, a, b) : multinomialLogisticPart(X, y, t.K, t.intercept, t.x, g, a, b);
        return { loss, values: g };
      });
      return { id: req.id, ok: true, parts };
    }
  }
}

function serve(port: Port, on: (listener: (data: Request) => void) => void): void {
  on((req) => {
    let res: Response;
    try {
      res = handle(req);
    } catch (e) {
      res = { id: req.id, ok: false, message: (e as Error)?.message ?? String(e) };
    }
    port.postMessage(res);
  });
}

const self_ = (globalThis as { self?: { postMessage?: unknown; addEventListener?: unknown } }).self;
if (self_ && typeof self_.postMessage === 'function' && typeof self_.addEventListener === 'function') {
  const scope = self_ as unknown as Port & { addEventListener(t: 'message', l: (e: { data: Request }) => void): void };
  serve(scope, (l) => scope.addEventListener('message', (e) => l(e.data)));
} else {
  // Node: worker_threads, imported only here, at run time, so the browser build never sees it.
  const wt = (await nodeModule('node:worker_threads')) as { parentPort: (Port & { on(e: 'message', l: (d: Request) => void): void }) | null };
  const port = wt.parentPort;
  if (port) serve(port, (l) => port.on('message', l));
}
