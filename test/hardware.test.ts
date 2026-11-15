import { MessageChannel } from 'node:worker_threads';
import { describe, expect, test } from 'vitest';
import {
  ColumnTransformer,
  LinearRegression,
  OneHotEncoder,
  LogisticRegression,
  Matrix,
  SmaltmlError,
  StandardScaler,
  gpuAvailable,
  matMul,
  matMulSync,
  pipeline,
  predictOnGpu,
  useGpu,
} from '../src/index.js';
import { connect, serve, type Port } from '../src/worker/index.js';

// Node has no WebGPU: these check the CPU path and the device choice. test/browser/hardware.mjs checks the GPU itself,
// in Chromium, against these CPU results.
describe('device choice without WebGPU', () => {
  test('no GPU here; auto and cpu run on the CPU; gpu fails', async () => {
    expect(await gpuAvailable()).toBe(false);
    expect(await useGpu('auto', 1e12)).toBe(false);
    expect(await useGpu('cpu', 1e12)).toBe(false);
    await expect(useGpu('gpu', 1)).rejects.toMatchObject({ code: 'ML_PARAM' });
    await expect(useGpu('tpu' as 'gpu', 1)).rejects.toMatchObject({ code: 'ML_PARAM' });
  });

  test("predictions: 'auto' and 'cpu' on the CPU, whatever the size; 'gpu' fails without WebGPU", async () => {
    expect(await predictOnGpu(undefined)).toBe(false);
    expect(await predictOnGpu('auto')).toBe(false);
    expect(await predictOnGpu('cpu')).toBe(false);
    await expect(predictOnGpu('gpu')).rejects.toMatchObject({ code: 'ML_PARAM' });
    await expect(predictOnGpu('tpu' as 'gpu')).rejects.toMatchObject({ code: 'ML_PARAM' });
  });

  test('matMul on the CPU', async () => {
    const A = [[1, 2], [3, 4], [5, 6]];
    const B = [[1, 0, 2], [0, 1, 3]];
    expect((await matMul(A, B)).toRows()).toEqual([[1, 2, 8], [3, 4, 18], [5, 6, 28]]);
    expect(matMulSync(A, B).toRows()).toEqual([[1, 2, 8], [3, 4, 18], [5, 6, 28]]);
    await expect(matMul(A, A)).rejects.toMatchObject({ code: 'ML_SHAPE' });
  });

  test('a fit on the GPU fails where there is none', async () => {
    await expect(new LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1], { device: 'gpu' })).rejects.toMatchObject({ code: 'ML_PARAM' });
  });

  test('async predictions equal the sync ones on the CPU', async () => {
    const X = [[1, 2], [2, 1], [3, 5], [4, 3], [5, 6], [6, 4]];
    const lin = new LinearRegression().fitSync(X, [3, 4, 9, 8, 12, 11]);
    expect(Array.from(await lin.predict(X))).toEqual(Array.from(lin.predictSync(X)));
    const log = new LogisticRegression().fitSync(X, ['a', 'a', 'b', 'a', 'b', 'b']);
    expect((await log.predictProba(X, { device: 'cpu' })).data).toEqual(log.predictProbaSync(X).data);
    expect((await log.predictProba(X)).data).toEqual(log.predictProbaSync(X).data);
    expect(await log.predict(X)).toEqual(log.predictSync(X));
  });
});

describe('the worker protocol', () => {
  const setup = () => {
    const { port1, port2 } = new MessageChannel();
    serve(port1 as unknown as Port);
    const ml = connect(port2 as unknown as Port);
    return { ml, close: () => (port1.close(), port2.close()) };
  };
  const X = [[1, 20], [2, 18], [3, 25], [4, 30], [5, 28], [6, 40], [7, 35], [8, 45]];
  const labels = ['no', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'yes'];

  test('fit returns a fitted copy that matches a local fit, bit for bit', async () => {
    const { ml, close } = setup();
    try {
      const model = new LogisticRegression({ C: 0.5 });
      const fitted = await ml.fit(model, X, labels);
      expect(model.fitted).toBe(false);
      expect(fitted).toBeInstanceOf(LogisticRegression);
      const local = new LogisticRegression({ C: 0.5 }).fitSync(X, labels);
      expect(fitted.coef.data).toEqual(local.coef.data);
      expect(await ml.predict(fitted, X)).toEqual(local.predictSync(X));
      expect((await ml.predictProba(fitted, X)).data).toEqual(local.predictProbaSync(X).data);
    } finally {
      close();
    }
  });

  test('pipelines, regression values and transforms come back', async () => {
    const { ml, close } = setup();
    try {
      const p = await ml.fit(pipeline(new StandardScaler(), new LinearRegression()), X, [1, 2, 3, 4, 5, 6, 7, 8]);
      const values = await ml.predict(p, X);
      expect(values).toBeInstanceOf(Float64Array);
      expect(Array.from(values as Float64Array)).toEqual(Array.from(p.predictSync(X) as Float64Array));
      const scaler = await ml.fit(new StandardScaler(), Matrix.from({ a: [1, 2, 3], b: [2, 4, 9] }));
      const z = await ml.transform(scaler, { a: [2], b: [4] });
      expect(z.names).toEqual(['a', 'b']);
      expect(Array.from(z.data)).toEqual(Array.from(scaler.transformSync({ a: [2], b: [4] }).data));
    } finally {
      close();
    }
  });

  test('tables with text reach the worker as columns', async () => {
    const { ml, close } = setup();
    try {
      const people = { city: ['Leeds', 'York', 'Leeds', 'Bath'], age: [31, 45, 27, 52] };
      const ct = new ColumnTransformer([['city', new OneHotEncoder(), ['city']]], { remainder: 'passthrough' });
      const fitted = await ml.fit(ct, people);
      const local = new ColumnTransformer([['city', new OneHotEncoder(), ['city']]], { remainder: 'passthrough' }).fitSync(people);
      const out = await ml.transform(fitted, people);
      expect(out.names).toEqual(local.getFeatureNamesOut());
      expect(Array.from(out.data)).toEqual(Array.from(local.transformSync(people).data));
    } finally {
      close();
    }
  });

    test('errors come back as SmaltmlErrors with their codes; the caller\'s data stays usable', async () => {
    const { ml, close } = setup();
    try {
      const data = Matrix.from(X);
      await expect(ml.fit(new LinearRegression(), data, [1, 2])).rejects.toMatchObject({ code: 'ML_SHAPE' });
      await expect(ml.predict(new LinearRegression(), data)).rejects.toBeInstanceOf(SmaltmlError);
      expect(data.data.length).toBe(16);
    } finally {
      close();
    }
  });
});
