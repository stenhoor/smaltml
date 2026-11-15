// Writes the results tables of docs/research/benchmarks.md from bench/results/*.json, between the page's
// <!-- bench:results --> markers. The page's method and notes stay as written.
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

const page = new URL('../docs/research/benchmarks.md', import.meta.url);
const text = readFileSync(page, 'utf8');
const start = '<!-- bench:results -->';
const end = '<!-- /bench:results -->';
const a = text.indexOf(start);
const b = text.indexOf(end);
if (a < 0 || b < 0) throw new Error('docs/research/benchmarks.md needs the bench:results markers');
writeFileSync(page, `${text.slice(0, a + start.length)}\n\n${lines.join('\n')}\n\n${text.slice(b)}`);
console.log(lines.join('\n'));
