// smaltml's worker script (dist/smaltml-worker.js): serves fits and predictions to startWorker() on the page.
import { serve } from './runtime.js';
import type { Port } from './protocol.js';

serve(self as unknown as Port);
