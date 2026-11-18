import { defineConfig } from 'vitepress'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { smaltLight, smaltDark } from './theme/shiki'
import { smaltworksMenu, smaltworksFooter } from './theme/smaltworks'
import { playgroundPlugin } from './theme/playground'
import { agentDocs, packageVersion } from './theme/agents'

const here = path.dirname(fileURLToPath(import.meta.url))
const REPO = 'https://github.com/stenhoor/smaltml'
// GitHub Pages serves the site under the repository's name.
const BASE = process.env.SMALTML_BASE ?? '/smaltml/'

export default defineConfig({
  title: 'smaltml',
  description: 'Machine learning for smaltdb, on the hardware you have.',
  base: BASE,
  cleanUrls: true,
  rewrites: { 'adr/README.md': 'adr/index.md' },
  lastUpdated: false,
  // Markdown for AI agents beside the pages: a twin of each page, /llms.txt and /agents.md (BRAND.md, Documentation).
  buildEnd: (config) => {
    agentDocs(config, {
      product: 'smaltml',
      version: packageVersion(path.resolve(here, '../../package.json')),
      url: `https://stenhoor.github.io${BASE}`,
      intro: 'guide/agents.md',
      guide: ['reference/'],
    })
  },
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${BASE}favicon-32.svg` }],
    ['meta', { property: 'og:title', content: 'smaltml' }],
    ['meta', { property: 'og:description', content: 'Small. Alternative. Predict. Machine learning for smaltdb, on the hardware you have.' }],
    ['meta', { property: 'og:image', content: 'https://raw.githubusercontent.com/stenhoor/smalt-brand/main/social/smaltml.png' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Geist:wght@400;500;700&family=Geist+Mono:wght@400;500&family=Martian+Mono:wght@400;600&display=swap' }],
  ],
  vite: {
    resolve: {
      alias: [
        { find: /^@smaltworks\/smaltml$/, replacement: path.resolve(here, '../../src/index.ts') },
        { find: /^@smaltworks\/smaltml\/(.*)$/, replacement: path.resolve(here, '../../src/$1/index.ts') },
      ],
    },
    worker: { format: 'es' },
  },
  markdown: {
    theme: { light: smaltLight, dark: smaltDark },
    config(md) {
      // Every ts example runs where it stands (BRAND.md, Documentation, rule 7; theme/smaltml.ts).
      md.use(playgroundPlugin, { languages: ['ts'] })
      // Links that leave docs/ point at the file on GitHub.
      const fallback = md.renderer.rules.link_open ?? ((t, i, o, _e, s) => s.renderToken(t, i, o))
      md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
        const href = tokens[idx].attrGet('href')
        if (href && !/^[a-z]+:|^#|^\//i.test(href) && env?.relativePath) {
          const target = path.posix.normalize(path.posix.join(path.posix.dirname(env.relativePath), href))
          if (target.startsWith('../')) {
            tokens[idx].attrSet('href', `${REPO}/blob/HEAD/${target.slice(3)}`)
            tokens[idx].attrSet('target', '_blank')
            tokens[idx].attrSet('rel', 'noreferrer')
          }
        }
        return fallback(tokens, idx, options, env, self)
      }
    },
  },
  themeConfig: {
    logo: { light: '/lockup.svg', dark: '/lockup-on-dark.svg', alt: 'smaltml' },
    siteTitle: false,
    // The four sections of every smalt site (BRAND.md, Documentation, rule 1).
    nav: [
      { text: 'Quick start', link: '/guide/quick-start' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Reference', link: '/reference/' },
      { text: 'Playground', link: '/playground' },
      smaltworksMenu(),
    ],
    sidebar: [
      { text: 'Quick start', items: [{ text: 'Quick start', link: '/guide/quick-start' }] },
      {
        text: 'Getting started',
        items: [
          { text: 'Installation', link: '/guide/getting-started' },
          { text: 'For AI agents', link: '/guide/agents' },
        ],
      },
      {
        text: 'User guide',
        items: [
          { text: 'Preparing data', link: '/guide/preparing-data' },
          { text: 'Training models', link: '/guide/training' },
          { text: 'Evaluating models', link: '/guide/evaluating' },
          { text: 'Saving and loading', link: '/guide/saving' },
          { text: 'Predicting by name', link: '/guide/predicting' },
          { text: 'The CPU, WebGPU and workers', link: '/guide/hardware' },
          { text: 'Plugins', link: '/guide/plugins' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'Overview', link: '/reference/' },
          { text: 'Matrix', link: '/reference/matrix' },
          {
            text: 'Preprocessing',
            collapsed: false,
            items: [
              { text: 'StandardScaler', link: '/reference/preprocessing/standard-scaler' },
              { text: 'MinMaxScaler', link: '/reference/preprocessing/min-max-scaler' },
              { text: 'RobustScaler', link: '/reference/preprocessing/robust-scaler' },
              { text: 'SimpleImputer', link: '/reference/preprocessing/simple-imputer' },
              { text: 'OneHotEncoder', link: '/reference/preprocessing/one-hot-encoder' },
              { text: 'OrdinalEncoder', link: '/reference/preprocessing/ordinal-encoder' },
            ],
          },
          {
            text: 'Linear models',
            collapsed: false,
            items: [
              { text: 'LinearRegression', link: '/reference/linear/linear-regression' },
              { text: 'Ridge', link: '/reference/linear/ridge' },
              { text: 'Lasso', link: '/reference/linear/lasso' },
              { text: 'ElasticNet', link: '/reference/linear/elastic-net' },
              { text: 'LogisticRegression', link: '/reference/linear/logistic-regression' },
            ],
          },
          {
            text: 'Trees',
            collapsed: false,
            items: [
              { text: 'DecisionTreeRegressor', link: '/reference/tree/decision-tree-regressor' },
              { text: 'DecisionTreeClassifier', link: '/reference/tree/decision-tree-classifier' },
              { text: 'RandomForestRegressor', link: '/reference/ensemble/random-forest-regressor' },
              { text: 'RandomForestClassifier', link: '/reference/ensemble/random-forest-classifier' },
            ],
          },
          {
            text: 'Model selection',
            collapsed: false,
            items: [
              { text: 'trainTestSplit', link: '/reference/selection/train-test-split' },
              { text: 'KFold', link: '/reference/selection/k-fold' },
              { text: 'StratifiedKFold', link: '/reference/selection/stratified-k-fold' },
              { text: 'GroupKFold', link: '/reference/selection/group-k-fold' },
              { text: 'crossValScore', link: '/reference/selection/cross-val-score' },
              { text: 'crossValidate', link: '/reference/selection/cross-validate' },
              { text: 'GridSearchCV', link: '/reference/selection/grid-search-cv' },
            ],
          },
          {
            text: 'Pipelines',
            collapsed: false,
            items: [
              { text: 'pipeline', link: '/reference/pipeline/pipeline' },
              { text: 'ColumnTransformer', link: '/reference/pipeline/column-transformer' },
            ],
          },
          {
            text: 'Several targets',
            collapsed: false,
            items: [{ text: 'MultiOutputRegressor', link: '/reference/multioutput/multi-output-regressor' }],
          },
          {
            text: 'Metrics',
            collapsed: false,
            link: '/reference/metrics/',
            items: [
              { text: 'meanSquaredError', link: '/reference/metrics/mean-squared-error' },
              { text: 'meanAbsoluteError', link: '/reference/metrics/mean-absolute-error' },
              { text: 'r2Score', link: '/reference/metrics/r2-score' },
              { text: 'accuracyScore', link: '/reference/metrics/accuracy-score' },
              { text: 'precisionScore', link: '/reference/metrics/precision-score' },
              { text: 'recallScore', link: '/reference/metrics/recall-score' },
              { text: 'f1Score', link: '/reference/metrics/f1-score' },
              { text: 'rocAucScore', link: '/reference/metrics/roc-auc-score' },
              { text: 'confusionMatrix', link: '/reference/metrics/confusion-matrix' },
              { text: 'logLoss', link: '/reference/metrics/log-loss' },
            ],
          },
          {
            text: 'Saving and loading',
            collapsed: false,
            items: [
              { text: 'saveModel', link: '/reference/io/save-model' },
              { text: 'loadModel', link: '/reference/io/load-model' },
            ],
          },
          {
            text: 'Model registry',
            collapsed: false,
            items: [
              { text: 'predict', link: '/reference/registry/predict' },
              { text: 'ModelRegistry', link: '/reference/registry/model-registry' },
              { text: 'MemoryStore', link: '/reference/registry/memory-store' },
              { text: 'TableStore', link: '/reference/registry/table-store' },
            ],
          },
          {
            text: 'Hardware',
            collapsed: false,
            items: [
              { text: 'matMul', link: '/reference/gpu/mat-mul' },
              { text: 'gpuAvailable', link: '/reference/gpu/gpu-available' },
              { text: 'configureThreads', link: '/reference/parallel/configure-threads' },
              { text: 'useKernels', link: '/reference/use-kernels' },
              { text: 'useSimd', link: '/reference/simd/use-simd' },
              { text: 'simdSupport', link: '/reference/simd/simd-support' },
              { text: 'startWorker', link: '/reference/worker/start-worker' },
              { text: 'connect', link: '/reference/worker/connect' },
            ],
          },
          { text: 'Errors', link: '/reference/errors' },
          { text: 'Where smaltml differs', link: '/reference/divergences' },
          {
            text: 'Internals',
            collapsed: true,
            items: [
              { text: 'Benchmarks', link: '/research/benchmarks' },
              { text: 'Decisions', link: '/adr/' },
              { text: '0024 Scope and architecture', link: '/adr/0024-smaltml-scope-and-architecture' },
              { text: '0034 Where auto runs', link: '/adr/0034-where-auto-runs' },
              { text: '0035 SIMD in a plugin', link: '/adr/0035-simd-kernels-in-a-plugin' },
              { text: '0037 Several targets', link: '/adr/0037-several-targets' },
              { text: '0039 Trees and forests', link: '/adr/0039-trees-and-forests' },
              { text: '0040 SIMD in smaltml', link: '/adr/0040-simd-kernels-in-smaltml' },
            ],
          },
        ],
      },
    ],
    socialLinks: [{ icon: 'github', link: REPO }],
    search: { provider: 'local' },
    outline: { level: [2, 3] },
    footer: smaltworksFooter(BASE),
  },
})
