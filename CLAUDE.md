# CLAUDE.md

smaltml (repository `stenhoor/smaltml`, npm `@smaltworks/smaltml`) is machine learning for the smalt family, in
TypeScript: classic models on tables, matched against scikit-learn, on the CPU everywhere and WebGPU where it helps.
It is a kit, between the languages and the user tools. Its scope and architecture are
[ADR 0024](docs/adr/0024-smaltml-scope-and-architecture.md). It is part of the smalt family: smaltdb's CLAUDE.md has
the family's shared working rules, and they apply here too.

## Brand: read smalt-brand first

Before changing the site, the README or how the product is described, read smalt-brand's `BRAND.md` and
`DECISIONS.md`. smaltml's brand is set (DECISIONS.md, "smaltml", 2026-11-14): "Small. Alternative. Predict.",
"Machine learning for smaltdb, on the hardware you have.", said smalt-M-L, the **Hardware** mark, in the Kits layer's
jade (the hero's `suffix kit`). The logo files in `docs/public/` come from smalt-brand's `logos/smaltml/`, unchanged;
a change starts there. Copy the site kit (`docs/.vitepress/theme/{shiki.ts,smaltdb.css,smaltworks.ts,stage.ts,playground.ts,agents.ts}`)
and the badges unchanged from smalt-brand.

## Workflow

The family's rules: one task per branch and pull request; open it once CI passes; the owner merges. Before pushing,
`npm run check`, and `npm run test:browser` when the docs or the runner change.

The site (https://stenhoor.github.io/smaltml/) deploys from `.github/workflows/docs.yml` only: the repository's
Settings → Pages → Source must be **GitHub Actions**.

## Commands

| Command | What it does |
| --- | --- |
| `npm ci` | Installs dependencies |
| `npm run check` | Typecheck, then the tests (vitest): the compare cases, every docs example, the entry points. Before every commit |
| `npm run test:browser` | In headless Chromium: Run on every playground of the site (`test/browser/docs.mjs`), then WebGPU and the worker against the CPU (`test/browser/hardware.mjs`, WebGPU on SwiftShader). CI runs it |
| `npm run bench` / `bench:python` / `bench:report` | Times smaltml and scikit-learn on the same data; the report writes docs/research/benchmarks.md. Rerun after performance work |
| `npm run bench:gpu` | Times WebGPU against the CPU in Chromium on a real GPU (refuses SwiftShader); needs a machine with one. `-- --browser PATH` for a Chromium other than Playwright's |
| `npm run compare` | Reruns compare/run.py through scikit-learn and rewrites fixtures/compare.json |
| `npm run docs:dev` / `docs:build` | The site |
| `npm run build` | `dist/` and type declarations |

Playwright is a dev dependency, pinned to CI's version (1.64.1). Its Chromium is preinstalled in the cloud containers; don't run `playwright install` there.

## Rules

- **scikit-learn is the reference, not the source** (ADR 0024, section 2). Same models, parameters, defaults and
  results, in camelCase (`maxIter`; fitted attributes drop the `_`; `random_state` is `seed`). Write our own code from
  the documentation and the papers; never copy or translate scikit-learn's, XGBoost's, PyTorch's or TensorFlow's.
- **Compare tests** (section 7): a new model or option gets cases in `compare/run.py`, with their tolerance; run
  `npm run compare` and commit `fixtures/compare.json`. CI checks the fixtures without Python. The Python packages are
  pinned in `compare/requirements.txt` (scikit-learn and NumPy, approved 2026-11-14); adding one is outside code: ask.
- **Unsupported fails, never answers differently.** A parameter or case smaltml doesn't support yet fails with a
  `SmaltmlError` and a code; list it in `docs/reference/divergences.md`. A deliberate difference is a `DIVERGES` entry
  in `test/compare.test.ts` too.
- **The CPU path is the product** (section 3): 64-bit floats, every browser, Node. WebGPU is an accelerator, checked
  against the CPU within a stated tolerance (`test/browser/hardware.mjs`). On the CPU, an asynchronous call gives
  exactly its `*Sync` twin's result.
- **Asynchronous API over a synchronous core** (section 5): every `fit`, `predict`, `predictProba` and `transform` has
  a `*Sync` twin; the core touches neither the DOM nor Node.
- **Entry points** (section 9): each area of `src/` is one, listed in `package.json` `exports`, `scripts/build.mjs`
  and the table in `test/modules.test.ts`, which says which area may import which. A new dependency between areas
  changes that table on purpose. A new entry point also goes in the guide's table (`docs/guide/getting-started.md`).
- **Models save as JSON** (section 6): a new model gets `toJSON`/`fromJSON`, a loader in `src/io/index.ts`, and a round
  trip in `test/core.test.ts`. A change to a saved document's shape raises `MODEL_FORMAT_VERSION`, and older documents
  still load.
- **Docs** follow smalt-brand's documentation standard: Quick start, Getting started, User guide and Reference. A
  reference page per class or function, in the standard's shape (signature, parameters, returns, errors, notes,
  examples, working demo, see also).
- **Every ```` ```ts ```` block in `docs/` runs where it stands**, in the kit's playground with smaltml's runner
  (`docs/.vitepress/theme/smaltml.ts`). A page's examples share one scope, in order; the last line's value is the result.
  Each is also type-checked by smaltty (through `smaltkit/typechecker`, as the smalt apps check code) against smaltml's
  source, so a misspelt method or option fails with tsc's message.
  The table under an example is its tested result: `test/docs-examples.test.ts` runs each as the playground does
  (`UPDATE_DOCS=1 npx vitest run test/docs-examples.test.ts` writes them). `<!-- docs:error CODE -->` marks one that
  must fail; `<!-- docs:skip -->` marks a block that isn't run and isn't setup for those below (signatures, sketches).
  `noplay` blocks still run as setup, so don't use them for code that can't run.
- **No runtime dependencies.** smaltkit is a dev dependency, pinned to a commit, for the site's playgrounds. Ask the
  owner before adding any other package.
- No fixed sleeps in tests. Probe tests (`test/zz-*.test.ts`) are scratch: never commit them.
- A significant decision gets an ADR in `docs/adr/`.
