/**
 * A forest's trees on the thread pool (ADR 0039, item 5): each worker grows a run of whole trees from their streams,
 * with the same code as fitSync() (grow.ts), and sends them back as documents, which load to the same trees.
 */
import type { Label, ModelDocument } from '../core/model.js';
import { share, threadPool, type Threads } from '../parallel/index.js';
import { DecisionTreeClassifier, DecisionTreeRegressor } from '../tree/index.js';
import type { Grown, GrowJob } from './grow.js';

/**
 * The trees grown on the pool, in stream order, or `null` where they should grow on this thread: one thread asked
 * for, no shared memory, too little work, or no pool.
 */
export async function growOnThreads(threads: Threads | undefined, job: GrowJob, streams: number[]): Promise<Grown<DecisionTreeRegressor | DecisionTreeClassifier>[] | null> {
  // The work grows with every tree: X's values once per tree.
  const pool = await threadPool(threads, job.X.rows * job.X.cols * streams.length);
  if (!pool) return null;
  const X = share(job.X);
  const runs: number[][] = [];
  const per = Math.ceil(streams.length / pool.size);
  for (let i = 0; i < streams.length; i += per) runs.push(streams.slice(i, i + per));
  const replies = await Promise.all(
    runs.map((run, i) =>
      pool.run(i, {
        op: 'trees',
        X,
        names: job.X.names ? [...job.X.names] : null,
        classifier: job.classifier,
        y: job.classifier ? [...(job.y as readonly Label[])] : (job.y as Float64Array),
        weights: job.weights,
        params: job.params as Record<string, unknown>,
        bootstrap: job.bootstrap,
        draws: job.draws,
        oob: job.oob,
        streams: run,
      }),
    ),
  );
  const load = (doc: ModelDocument) => (job.classifier ? DecisionTreeClassifier.fromJSON(doc) : DecisionTreeRegressor.fromJSON(doc));
  return replies.flatMap((r) => r.trees!.map((t) => ({ tree: load(t.doc), outOfBag: t.outOfBag, oob: t.oob })));
}
