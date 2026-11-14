/**
 * Seeded random numbers: xoshiro128**, seeded through splitmix32. The same seed gives the same numbers on every
 * platform. Python's random numbers and these differ by design (ADR 0024, section 5), so a seeded split or shuffle
 * matches itself, not scikit-learn's.
 */
export class Random {
  #s0: number;
  #s1: number;
  #s2: number;
  #s3: number;

  constructor(seed: number = Date.now()) {
    let x = Math.floor(seed) >>> 0;
    const next = () => {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.#s0 = next();
    this.#s1 = next();
    this.#s2 = next();
    this.#s3 = next();
  }

  /** A whole number in [0, 2³²). */
  uint32(): number {
    const result = Math.imul(rotl(Math.imul(this.#s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (this.#s1 << 9) >>> 0;
    this.#s2 ^= this.#s0;
    this.#s3 ^= this.#s1;
    this.#s1 ^= this.#s2;
    this.#s0 ^= this.#s3;
    this.#s2 ^= t;
    this.#s3 = rotl(this.#s3, 11);
    this.#s0 >>>= 0;
    this.#s1 >>>= 0;
    this.#s2 >>>= 0;
    return result;
  }

  /** A number in [0, 1), with 53 random bits. */
  next(): number {
    const hi = this.uint32() >>> 5;
    const lo = this.uint32() >>> 6;
    return (hi * 67108864 + lo) / 9007199254740992;
  }

  /** A whole number in [0, n), without bias. */
  int(n: number): number {
    if (n <= 0) return 0;
    if (n > 0x100000000) return Math.floor(this.next() * n);
    const limit = 0x100000000 - (0x100000000 % n);
    let v: number;
    do v = this.uint32();
    while (v >= limit);
    return v % n;
  }

  /** The numbers 0 … n − 1 in a random order (Fisher–Yates). */
  permutation(n: number): Uint32Array {
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) out[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const t = out[i]!;
      out[i] = out[j]!;
      out[j] = t;
    }
    return out;
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}
