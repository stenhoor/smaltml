// The setup for `npm run test:simd` (ADR 0040, carrying ADR 0035's item 7): the compare tests against scikit-learn,
// the parallel tests and the hardware tests, with smaltml's SIMD kernels on, all of them, row steps included. Every
// compare case must pass within its tolerance, threaded fits must equal fitSync, and matMulSync must give its tested
// results. SMALTML_SIMD_FMA=1 turns fused multiply-add on too.
//
// The thread pool's workers load the kernel set from one file (dist/smaltml-simd-kernels.js in the package); here it
// is bundled from the source, as the parallel tests bundle the workers' script.
import * as esbuild from 'esbuild';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { kernelsInUse } from '../src/core/kernel-set.js';
import { useSimd } from '../src/simd/index.js';

const file = join(mkdtempSync(join(tmpdir(), 'smaltml-simd-')), 'smaltml-simd-kernels.js');
await esbuild.build({ entryPoints: ['src/simd/kernels.ts'], bundle: true, format: 'esm', platform: 'neutral', outfile: file, logLevel: 'silent' });
const fusedMultiplyAdd = process.env.SMALTML_SIMD_FMA === '1';
if (!(await useSimd({ fusedMultiplyAdd, logistic: true, kernelsModule: pathToFileURL(file).href }))) {
  throw new Error('the SIMD kernels did not turn on: WebAssembly SIMD is missing, or a kernel failed the check at start');
}
const used = kernelsInUse();
if (used.kernels.join() !== 'gramPart,matMul,binaryLogisticRows,multinomialLogisticRows,logisticProducts,xtvPart,columnSums') {
  throw new Error(`unexpected kernels in use: ${JSON.stringify(used)}`);
}
