// Every test file starts on smaltml's own TypeScript kernels: where the engine has WebAssembly SIMD, smaltml would
// otherwise turn its SIMD kernels on at the first asynchronous fit (kernelsReady, ADR 0040). `npm run test:simd` runs
// the tests that matter with SIMD on instead (test/simd-setup.ts). A useKernels() call is the app's choice, kept by
// kernelsReady(); tests that load smaltml afresh (vi.resetModules) start from the default again.
import { useKernels } from '../src/core/kernel-set.js';

useKernels(null);
