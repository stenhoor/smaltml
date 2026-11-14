import { DatabaseSync } from 'node:sqlite';
import { describe, expect, test } from 'vitest';
import {
  LinearRegression,
  LogisticRegression,
  MemoryStore,
  ModelRegistry,
  StandardScaler,
  TableStore,
  defaultRegistry,
  pipeline,
  predict,
  predictProba,
  useRegistry,
  type ModelStore,
  type SqlDatabase,
} from '../src/index.js';

/** Node's built-in SQLite as a SqlDatabase: the same SQL and `?` parameters smaltdb with smaltql takes. */
function sqlite(): SqlDatabase & { raw: DatabaseSync } {
  const raw = new DatabaseSync(':memory:');
  return {
    raw,
    async query(sql, params = []) {
      return raw.prepare(sql).all(...(params as never[])) as Record<string, unknown>[];
    },
    async exec(sql, params = []) {
      return raw.prepare(sql).run(...(params as never[]));
    },
  };
}

const X = [[1, 20], [2, 18], [3, 25], [4, 30], [5, 28], [6, 40], [7, 35], [8, 45]];
const labels = ['no', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'yes'];
const y = [10, 11, 15, 19, 20, 27, 27, 33];

describe.each([
  ['MemoryStore', () => new MemoryStore()],
  ['TableStore on SQLite', () => new TableStore(sqlite())],
] as [string, () => ModelStore][])('%s', (_name, make) => {
  test('versions count up per name; get returns the latest or the asked version', async () => {
    const store = make();
    const a = new LinearRegression().fitSync(X, y).toJSON();
    const b = new LinearRegression({ fitIntercept: false }).fitSync(X, y).toJSON();
    expect(await store.put('rent', a)).toBe(1);
    expect(await store.put('rent', b)).toBe(2);
    expect(await store.put('churn', new LogisticRegression().fitSync(X, labels).toJSON())).toBe(1);
    expect((await store.get('rent'))?.params).toEqual({ fitIntercept: false });
    expect((await store.get('rent', 1))?.params).toEqual({ fitIntercept: true });
    expect(await store.get('rent', 3)).toBeUndefined();
    expect(await store.get('nothing')).toBeUndefined();
    const list = await store.list();
    expect(list.map((e) => [e.name, e.version, e.model])).toEqual([
      ['churn', 1, 'LogisticRegression'],
      ['rent', 1, 'LinearRegression'],
      ['rent', 2, 'LinearRegression'],
    ]);
    expect(Number.isNaN(Date.parse(list[0]!.savedAt))).toBe(false);
    expect(await store.delete('rent', 1)).toBe(1);
    expect((await store.list()).length).toBe(2);
    expect(await store.delete('rent')).toBe(1);
    expect(await store.put('rent', a)).toBe(1);
  });

  test('a registry saves, loads and predicts by name', async () => {
    const registry = new ModelRegistry(make());
    const model = pipeline(new StandardScaler(), new LogisticRegression()).fitSync(X, labels);
    expect(await registry.save('churn', model)).toBe(1);
    expect(await registry.predict('churn', X)).toEqual(model.predictSync(X));
    expect((await registry.predictProba('churn', X)).data).toEqual(model.predictProbaSync(X).data);
    expect(await predict('churn', X, { registry })).toEqual(model.predictSync(X));
    const v2 = new LogisticRegression({ C: 0.01 }).fitSync(X, labels);
    expect(await registry.save('churn', v2)).toBe(2);
    expect((await predictProba('churn', X, { registry })).data).toEqual(v2.predictProbaSync(X).data);
    expect((await predictProba('churn', X, { registry, version: 1 })).data).toEqual(model.predictProbaSync(X).data);
  });
});

describe('the registry', () => {
  test('errors: unfitted models, unknown names, bad names', async () => {
    const registry = new ModelRegistry();
    await expect(registry.save('m', new LinearRegression())).rejects.toMatchObject({ code: 'ML_NOT_FITTED' });
    await expect(registry.load('missing')).rejects.toMatchObject({ code: 'ML_FORMAT' });
    await expect(registry.save('', new LinearRegression().fitSync(X, y))).rejects.toMatchObject({ code: 'ML_PARAM' });
    await registry.save('scaler', new StandardScaler().fitSync(X));
    await expect(registry.predict('scaler', X)).rejects.toMatchObject({ code: 'ML_PARAM' });
    expect(() => new TableStore(sqlite(), { table: 'x; DROP TABLE y' })).toThrow(expect.objectContaining({ code: 'ML_PARAM' }));
  });

  test('predict takes a model too; names use the default registry', async () => {
    const model = new LinearRegression().fitSync(X, y);
    expect(Array.from((await predict(model, X)) as Float64Array)).toEqual(Array.from(model.predictSync(X)));
    const registry = new ModelRegistry();
    useRegistry(registry);
    expect(defaultRegistry()).toBe(registry);
    await registry.save('rent', model);
    expect(Array.from((await predict('rent', [[9, 50]])) as Float64Array)).toEqual(Array.from(model.predictSync([[9, 50]])));
  });

  test('a TableStore keeps its rows in the table, readable by SQL', async () => {
    const db = sqlite();
    const registry = new ModelRegistry(new TableStore(db, { table: 'models' }));
    await registry.save('rent', new LinearRegression().fitSync(X, y));
    const rows = db.raw.prepare('SELECT name, version, model, smaltml FROM models').all();
    expect(rows).toEqual([{ name: 'rent', version: 1, model: 'LinearRegression', smaltml: '0.1.0' }]);
    // A second registry on the same table sees the model.
    const again = new ModelRegistry(new TableStore(db, { table: 'models' }));
    expect((await again.load('rent')).kind).toBe('LinearRegression');
  });
});
