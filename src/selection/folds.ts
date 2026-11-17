import { fail } from '../core/errors.js';
import { Random } from '../core/random.js';
import type { Label, Labels } from '../core/model.js';

/** One split: the rows to train on and the rows to test on, by position. */
export interface Fold {
  train: Uint32Array;
  test: Uint32Array;
}

/** Splits rows into folds. */
export interface Splitter {
  readonly nSplits: number;
  /**
   * The folds for n rows (and their labels, for splitters that keep classes in proportion, and their groups, for
   * splitters that keep each group together).
   */
  split(n: number, y?: Labels, groups?: Groups): Fold[];
  /** Whether split() uses groups: a splitter that doesn't is never given them (crossValScore fails instead). */
  readonly usesGroups?: boolean;
}

/** Each row's group, such as a patient or a customer: rows of one group are never split between training and test. */
export type Groups = ArrayLike<Label>;

export interface KFoldParams {
  /** How many folds. Default 5. */
  nSplits?: number;
  /** Shuffle the rows before cutting them into folds. Default false. */
  shuffle?: boolean;
  /** The shuffle's seed. */
  seed?: number;
}

function checkParams(p: KFoldParams): Required<Omit<KFoldParams, 'seed'>> & { seed: number | undefined } {
  const nSplits = p.nSplits ?? 5;
  if (!Number.isInteger(nSplits) || nSplits < 2) fail('ML_PARAM', `nSplits must be a whole number of at least 2, not ${nSplits}`);
  if (p.seed !== undefined && !p.shuffle) fail('ML_PARAM', 'seed only applies with shuffle: true');
  return { nSplits, shuffle: p.shuffle ?? false, seed: p.seed };
}

function order(n: number, shuffle: boolean, seed: number | undefined): Uint32Array {
  return shuffle ? new Random(seed ?? Date.now()).permutation(n) : Uint32Array.from({ length: n }, (_, i) => i);
}

/** Every fold's test rows, as train/test pairs. */
function folds(n: number, tests: number[][]): Fold[] {
  return tests.map((test) => {
    const inTest = new Uint8Array(n);
    for (const i of test) inTest[i] = 1;
    const train: number[] = [];
    for (let i = 0; i < n; i++) if (!inTest[i]) train.push(i);
    return { train: Uint32Array.from(train), test: Uint32Array.from(test).sort() };
  });
}

/**
 * K folds of consecutive rows: the first n % k folds have one row more. Each fold is the test set once, the rest the
 * training set. scikit-learn's `KFold`: without shuffling, the same folds; with it, smaltml's own seeded shuffle.
 */
export class KFold implements Splitter {
  readonly nSplits: number;
  readonly shuffle: boolean;
  readonly seed: number | undefined;

  constructor(params: KFoldParams = {}) {
    const p = checkParams(params);
    this.nSplits = p.nSplits;
    this.shuffle = p.shuffle;
    this.seed = p.seed;
  }

  split(n: number): Fold[] {
    if (this.nSplits > n) fail('ML_PARAM', `${this.nSplits} folds need at least ${this.nSplits} rows; there are ${n}`);
    const idx = order(n, this.shuffle, this.seed);
    const tests: number[][] = [];
    let start = 0;
    for (let k = 0; k < this.nSplits; k++) {
      const size = Math.floor(n / this.nSplits) + (k < n % this.nSplits ? 1 : 0);
      tests.push(Array.from(idx.subarray(start, start + size)));
      start += size;
    }
    return folds(n, tests);
  }
}

/**
 * K folds that keep each class's share of the rows: scikit-learn's `StratifiedKFold`, with its documented guarantees
 * (each test fold has each class in proportion, test folds differ in size by at most one, the folds don't depend on
 * what the labels are called, and without shuffling each class's rows keep their order). Its allocation is smaltml's
 * own, so the folds can differ from scikit-learn's.
 */
export class StratifiedKFold implements Splitter {
  readonly nSplits: number;
  readonly shuffle: boolean;
  readonly seed: number | undefined;

  constructor(params: KFoldParams = {}) {
    const p = checkParams(params);
    this.nSplits = p.nSplits;
    this.shuffle = p.shuffle;
    this.seed = p.seed;
  }

  split(n: number, y?: Labels): Fold[] {
    if (!y) fail('ML_PARAM', 'StratifiedKFold needs y, the labels to keep in proportion');
    if (y.length !== n) fail('ML_SHAPE', `y has ${y.length} values; X has ${n} rows`);
    const k = this.nSplits;
    // Classes in the order they first appear, so the folds don't depend on the labels' names.
    const byClass = new Map<Label, number[]>();
    const rows = order(n, this.shuffle, this.seed);
    for (const i of rows) {
      const label = y[i] as Label;
      let list = byClass.get(label);
      if (!list) byClass.set(label, (list = []));
      list.push(i);
    }
    // A class with fewer rows than folds is spread as evenly as it can be (scikit-learn warns there).
    if (k > n) fail('ML_PARAM', `${k} folds need at least ${k} rows; there are ${n}`);
    const tests: number[][] = Array.from({ length: k }, () => []);
    const sizes = new Array<number>(k).fill(0);
    let next = 0;
    for (const list of byClass.values()) {
      const base = Math.floor(list.length / k);
      let extra = list.length % k;
      // Each fold takes `base` rows of this class; the remainder goes to the folds that are smallest so far, starting
      // where the last class's remainder stopped, so test folds stay within one row of each other.
      const counts = new Array<number>(k).fill(base);
      const byFill = Array.from({ length: k }, (_, j) => (next + j) % k).sort((a, b) => sizes[a]! - sizes[b]!);
      for (const f of byFill) {
        if (extra === 0) break;
        counts[f]! += 1;
        extra--;
        next = (f + 1) % k;
      }
      let at = 0;
      for (let f = 0; f < k; f++) {
        // A loop, not push(...slice): a class can have more rows than a call can take as arguments.
        for (let i = at; i < at + counts[f]!; i++) tests[f]!.push(list[i]!);
        at += counts[f]!;
        sizes[f]! += counts[f]!;
      }
    }
    return folds(n, tests);
  }
}

export interface GroupKFoldParams {
  /** How many folds. Default 5. There must be at least as many groups. */
  nSplits?: number;
}

/**
 * K folds that keep each group's rows together: a group is in one test fold, and in the training rows of the others.
 * scikit-learn's `GroupKFold`, with the same folds: the groups, largest first, each go to the test fold with the
 * fewest rows so far, so the test folds are about the same size. Of two groups as large, the one sorted later goes
 * first; of two folds as full, the lower one takes it.
 */
export class GroupKFold implements Splitter {
  readonly nSplits: number;
  readonly usesGroups = true;

  constructor(params: GroupKFoldParams = {}) {
    const extra = Object.keys(params).filter((k) => k !== 'nSplits');
    if (extra.length) fail('ML_PARAM', `GroupKFold takes nSplits only; not ${extra.join(', ')} (scikit-learn's shuffle isn't supported yet)`);
    this.nSplits = checkParams({ nSplits: params.nSplits }).nSplits;
  }

  split(n: number, _y?: Labels, groups?: Groups): Fold[] {
    if (!groups) fail('ML_PARAM', 'GroupKFold needs groups: one per row');
    if (groups.length !== n) fail('ML_SHAPE', `groups has ${groups.length} values; X has ${n} rows`);
    const k = this.nSplits;
    // The groups in sorted order (numbers in order, text by code unit), and each row's.
    const keys = [...new Set(Array.from(groups))];
    const text = keys.some((g) => typeof g === 'string');
    if (text && keys.some((g) => typeof g !== 'string')) fail('ML_LABELS', 'groups mix numbers and text');
    keys.sort(text ? undefined : (a, b) => (a as number) - (b as number));
    if (k > keys.length) fail('ML_PARAM', `${k} folds need at least ${k} groups; there are ${keys.length}`);
    const index = new Map(keys.map((g, i) => [g, i]));
    const sizes = new Array<number>(keys.length).fill(0);
    for (let i = 0; i < n; i++) sizes[index.get(groups[i]!)!]! += 1;
    const order = keys.map((_, g) => g).sort((a, b) => sizes[b]! - sizes[a]! || b - a);
    const load = new Array<number>(k).fill(0);
    const foldOf = new Int32Array(keys.length);
    for (const g of order) {
      let f = 0;
      for (let j = 1; j < k; j++) if (load[j]! < load[f]!) f = j;
      foldOf[g] = f;
      load[f]! += sizes[g]!;
    }
    const tests: number[][] = Array.from({ length: k }, () => []);
    for (let i = 0; i < n; i++) tests[foldOf[index.get(groups[i]!)!]!]!.push(i);
    return folds(n, tests);
  }
}
