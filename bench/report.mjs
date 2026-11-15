// Writes the results tables of docs/research/benchmarks.md from bench/results/*.json, between the page's
// <!-- bench:results --> markers, and the WebGPU tables from bench/results/gpu.json (npm run bench:gpu), where there
// is one, between its <!-- bench:gpu --> markers. The page's method and notes stay as written.
//   node bench/report.mjs
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';

const res = (f) => JSON.parse(readFileSync(new URL(`./results/${f}`, import.meta.url), 'utf8'));
const files = readdirSync(new URL('./results', import.meta.url));
const runs = (prefix) => files.filter((f) => f.startsWith(prefix)).map(res).sort((a, b) => a.threads - b.threads);
// smaltml.json (one thread) and smaltml-N.json (fits on up to N threads); scikit-learn-N.json.
const mls = runs('smaltml');
const ml = mls[0];
const sk = runs('scikit-learn-');
const fmt = (v) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const ratio = (a, b) => `${(a / b).toFixed(1)}×`;

const lines = [];
for (const size of Object.keys(ml.milliseconds)) {
  lines.push(`### ${size}`, '');
  const t = (r) => `${r.threads} ${r.threads === 1 ? 'thread' : 'threads'}`;
  // Each smaltml run against scikit-learn on the same number of threads.
  const pairs = mls.map((m) => [m, sk.find((s) => s.threads === m.threads)]).filter(([, s]) => s);
  const head = ['Operation', ...mls.map((m) => `smaltml, ${t(m)}`), ...sk.map((s) => `scikit-learn, ${t(s)}`), ...pairs.map(([m]) => `smaltml ÷ scikit-learn, ${t(m)}`)];
  lines.push(`| ${head.join(' | ')} |`, `| --- | ${head.slice(1).map(() => '---:').join(' | ')} |`);
  for (const op of Object.keys(ml.milliseconds[size])) {
    if (op === 'iterations') continue;
    const cell = (r) => (r.milliseconds[size][op] === undefined ? '' : fmt(r.milliseconds[size][op]));
    const rat = ([m, s]) => (m.milliseconds[size][op] === undefined || s.milliseconds[size][op] === undefined ? '' : ratio(m.milliseconds[size][op], s.milliseconds[size][op]));
    lines.push(`| ${[op, ...mls.map(cell), ...sk.map(cell), ...pairs.map(rat)].join(' | ')} |`);
  }
  const it = ml.milliseconds[size].iterations;
  const its = sk[0]?.milliseconds[size].iterations;
  lines.push('', `Iterations: Lasso ${it.lasso} (scikit-learn ${its?.lasso}), logistic regression ${it.logistic2} and ${it.logistic3} (scikit-learn ${its?.logistic2} and ${its?.logistic3}).`, '');
}
lines.push(`Run ${ml.date}: ${ml.library} on ${ml.runtime}; ${sk.map((s) => `${s.library} on ${s.runtime} (${s.blas})`)[0]}. CPU: ${ml.cpu}. smaltml's runs on several threads use fit() with threads 'auto' (up to that many); predictions and StandardScaler run on one thread in every smaltml column.`);

// WebGPU against the CPU, in the browser: bench/gpu.mjs.
const gpuLines = [];
if (files.includes('gpu.json')) {
  const g = res('gpu.json');
  const work = (v) => v.toLocaleString('en-US');
  gpuLines.push('### matMul', '', `\`'auto'\` runs on the GPU from \`GPU_MIN_WORK\` (${work(g.gpuMinWork)}) multiply-adds, n · p · k. **slower** marks the cases where that picks the slower device.`, '');
  gpuLines.push('| A · B | Multiply-adds | CPU (ms) | GPU (ms) | GPU ÷ CPU | `auto` picks |', '| --- | ---: | ---: | ---: | ---: | --- |');
  for (const c of g.matMul) {
    const wrong = (c.auto === 'gpu') !== c.gpu < c.cpu;
    gpuLines.push(`| ${work(c.n)} × ${c.p} · ${c.p} × ${c.k} | ${work(c.work)} | ${fmt(c.cpu)} | ${fmt(c.gpu)} | ${ratio(c.gpu, c.cpu)} | ${c.auto}${wrong ? ', **slower**' : ''} |`);
  }
  gpuLines.push('');
  for (const [size, row] of Object.entries(g.models)) {
    gpuLines.push(`### ${size}, the CPU benchmark's data and parameters`, '');
    gpuLines.push('| Operation | CPU, 1 thread (ms) | CPU, 4 threads (ms) | GPU (ms) | GPU ÷ CPU, 1 thread | Iterations, CPU / GPU | Run to the end: iterations, CPU / GPU | Run to the end: largest probability difference |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const [op, t] of Object.entries(row)) {
      const its = t.iterations ? `${t.iterations.cpu} / ${t.iterations.gpu}` : '';
      const end = t.converged ? `${t.converged.iterations.cpu} / ${t.converged.iterations.gpu}` : '';
      const diff = t.converged ? t.converged.probabilityDifference.toExponential(1) : '';
      gpuLines.push(`| ${op} | ${fmt(t.cpu)} | ${t.threads4 === undefined ? '' : fmt(t.threads4)} | ${fmt(t.gpu)} | ${ratio(t.gpu, t.cpu)} | ${its} | ${end} | ${diff} |`);
    }
    gpuLines.push('', 'The timed fits stop at the default 100 iterations, before they converge on this data; the last two columns fit again with `maxIter: 20000`, and compare the answers once both have finished.', '');
  }
  gpuLines.push('### LogisticRegression fits by size, default parameters, columns of one scale', '');
  gpuLines.push(`The CPU column runs as \`'auto'\` would, on up to every core (${g.cores} here). Work is rows × columns × classes (1 for two classes): the multiply-adds of one evaluation.`, '');
  gpuLines.push('| Rows | Columns | Classes | Work | CPU, every core (ms) | GPU (ms) | GPU speed-up | Iterations, CPU / GPU |', '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
  for (const f of [...g.fits].sort((a, b) => a.work - b.work)) gpuLines.push(`| ${work(f.n)} | ${f.p} | ${f.classes} | ${work(f.work)} | ${fmt(f.cpuAll)} | ${fmt(f.gpu)} | ${(f.cpuAll / f.gpu).toFixed(1)}× | ${f.iterations.cpu} / ${f.iterations.gpu} |`);
  const a = g.adapter;
  gpuLines.push('', `Run ${g.date}: ${g.library} in ${g.browser}. GPU: WebGPU adapter ${[a.vendor, a.architecture, a.description].filter(Boolean).join(' ')}. CPU: ${g.cpu}. Not the machine of the tables above.`);
}

const page = new URL('../docs/research/benchmarks.md', import.meta.url);
let text = readFileSync(page, 'utf8');
/** The page with `body` between its <!-- name --> and <!-- /name --> markers. */
const between = (name, body) => {
  const start = `<!-- ${name} -->`;
  const end = `<!-- /${name} -->`;
  const a = text.indexOf(start);
  const b = text.indexOf(end);
  if (a < 0 || b < 0) throw new Error(`docs/research/benchmarks.md needs the ${name} markers`);
  text = `${text.slice(0, a + start.length)}\n\n${body}\n\n${text.slice(b)}`;
};
between('bench:results', lines.join('\n'));
if (gpuLines.length) between('bench:gpu', gpuLines.join('\n'));
writeFileSync(page, text);
console.log(lines.join('\n'));
if (gpuLines.length) console.log('\n' + gpuLines.join('\n'));
