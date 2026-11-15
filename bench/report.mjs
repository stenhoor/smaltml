// Writes the results tables of docs/research/benchmarks.md from bench/results/*.json, between the page's
// <!-- bench:results --> markers. The page's method and notes stay as written.
//   node bench/report.mjs
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';

const res = (f) => JSON.parse(readFileSync(new URL(`./results/${f}`, import.meta.url), 'utf8'));
const ml = res('smaltml.json');
const sk = readdirSync(new URL('./results', import.meta.url))
  .filter((f) => f.startsWith('scikit-learn-'))
  .map(res)
  .sort((a, b) => a.threads - b.threads);
const fmt = (v) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const ratio = (a, b) => `${(a / b).toFixed(1)}×`;

const lines = [];
for (const size of Object.keys(ml.milliseconds)) {
  lines.push(`### ${size}`, '');
  const head = ['Operation', 'smaltml', ...sk.map((s) => `scikit-learn, ${s.threads} ${s.threads === 1 ? 'thread' : 'threads'}`), ...sk.map((s) => `smaltml ÷ ${s.threads} ${s.threads === 1 ? 'thread' : 'threads'}`)];
  lines.push(`| ${head.join(' | ')} |`, `| --- | ${head.slice(1).map(() => '---:').join(' | ')} |`);
  for (const [op, v] of Object.entries(ml.milliseconds[size])) {
    if (op === 'iterations') continue;
    const others = sk.map((s) => s.milliseconds[size][op]);
    lines.push(`| ${op} | ${fmt(v)} | ${others.map(fmt).join(' | ')} | ${others.map((o) => ratio(v, o)).join(' | ')} |`);
  }
  const it = ml.milliseconds[size].iterations;
  const its = sk[0]?.milliseconds[size].iterations;
  lines.push('', `Iterations: Lasso ${it.lasso} (scikit-learn ${its?.lasso}), logistic regression ${it.logistic2} and ${it.logistic3} (scikit-learn ${its?.logistic2} and ${its?.logistic3}).`, '');
}
lines.push(`Run ${ml.date}: ${ml.library} on ${ml.runtime}, ${ml.threads} thread; ${sk.map((s) => `${s.library} on ${s.runtime} (${s.blas})`)[0]}. CPU: ${ml.cpu}.`);

const page = new URL('../docs/research/benchmarks.md', import.meta.url);
const text = readFileSync(page, 'utf8');
const start = '<!-- bench:results -->';
const end = '<!-- /bench:results -->';
const a = text.indexOf(start);
const b = text.indexOf(end);
if (a < 0 || b < 0) throw new Error('docs/research/benchmarks.md needs the bench:results markers');
writeFileSync(page, `${text.slice(0, a + start.length)}\n\n${lines.join('\n')}\n\n${text.slice(b)}`);
console.log(lines.join('\n'));
