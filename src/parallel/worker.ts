// The thread pool's worker script (dist/smaltml-threads.js): runs the kernels over the chunks of rows it is given, on
// matrices in shared memory, and sends back one part per chunk. In Node it runs under worker_threads; in a browser,
// as a module Worker on a cross-origin isolated page.
import { binaryLogisticPart, binaryLogisticRows, gramPart, multinomialLogisticPart, multinomialLogisticRows, prepareLogistic, xtvPart, xw, type LogisticProducts } from '../core/kernels.js';
import { useKernels, type KernelSet } from '../core/kernel-set.js';
import { Matrix } from '../core/matrix.js';
import { cumulativeWeights, growTree } from '../ensemble/grow.js';
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

/**
 * The logistic products kernel's copy of each X, where the set gives one: kept while X is, so every evaluation of a
 * fit reads the same copy, and its chunks give what fitSync's do.
 */
const products = new WeakMap<Matrix, LogisticProducts | null>();
function productsFor(X: Matrix): LogisticProducts | null {
  if (!products.has(X)) products.set(X, prepareLogistic(X));
  return products.get(X)!;
}

/**
 * Imports a kernel set's module (smaltml's SIMD kernels, or a plugin's) and uses the set it gives, as the main thread
 * does (ADR 0035, item 8).
 */
async function loadKernels(id: number, module: string, options: unknown): Promise<Response> {
  const mod = (await import(/* @vite-ignore */ module)) as { default?: (options: unknown) => KernelSet | Promise<KernelSet> };
  if (typeof mod.default !== 'function') throw new Error(`${module} has no default export that gives a kernel set`);
  const used = useKernels(await mod.default(options));
  return { id, ok: true, parts: [], kernels: used.kernels };
}

function handle(req: Request): Response | Promise<Response> {
  const t = req.task;
  if (t.op === 'kernels') return loadKernels(req.id, t.module, t.options);
  if (t.op === 'trees') {
    // X with its column names, as fitSync()'s trees see it, so the trees save the same documents.
    const X = new Matrix(t.X.rows, t.X.cols, new Float64Array(t.X.buffer), t.names ?? undefined);
    const job = { classifier: t.classifier, X, y: t.y, weights: t.weights, params: t.params, bootstrap: t.bootstrap, draws: t.draws, oob: t.oob };
    const cumulative = t.bootstrap ? cumulativeWeights(t.weights) : null;
    const trees = t.streams.map((s) => {
      const g = growTree(job, cumulative, s);
      return { doc: g.tree.toJSON(), outOfBag: g.outOfBag, oob: g.oob };
    });
    return { id: req.id, ok: true, parts: [], trees };
  }
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
      const w = t.weights ? new Float64Array(t.weights) : undefined;
      const width = X.cols + (t.intercept ? 1 : 0);
      const prepared = productsFor(X);
      const parts = t.chunks.map(([a, b]) => {
        const g = new Float64Array(t.K === 1 ? width : t.K * width);
        const loss = prepared
          ? t.K === 1
            ? prepared.binaryPart(y, t.intercept, t.x, g, a, b, w, binaryLogisticRows)
            : prepared.multinomialPart(y, t.K, t.intercept, t.x, g, a, b, w, multinomialLogisticRows)
          : t.K === 1
            ? binaryLogisticPart(X, y, t.intercept, t.x, g, a, b, w)
            : multinomialLogisticPart(X, y, t.K, t.intercept, t.x, g, a, b, w);
        return { loss, values: g };
      });
      return { id: req.id, ok: true, parts };
    }
  }
}

function serve(port: Port, on: (listener: (data: Request) => void) => void): void {
  // Tasks run in the order they arrive: none starts before the kernels asked for ahead of it are loaded.
  let queue: Promise<void> = Promise.resolve();
  on((req) => {
    queue = queue.then(async () => {
      let res: Response;
      try {
        res = await handle(req);
      } catch (e) {
        res = { id: req.id, ok: false, message: (e as Error)?.message ?? String(e) };
      }
      port.postMessage(res);
    });
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
