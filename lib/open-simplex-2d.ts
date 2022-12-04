// https://github.com/joshforisha/open-simplex-noise-js
//
// This is free and unencumbered software released into the public domain

import shuffleSeed from "./shuffle-seed.js";

const NORM_2D = 1.0 / 47.0;
const SQUISH_2D = (Math.sqrt(2 + 1) - 1) / 2;
const STRETCH_2D = (1 / Math.sqrt(2 + 1) - 1) / 2;

export type Noise2D = (x: number, y: number) => number;

export function makeNoise2D(clientSeed: number): Noise2D {
  const perm = new Uint8Array(256);
  const source = new Uint8Array(256);
  for (let i = 0; i < 256; i++) source[i] = i;
  let seed = new Uint32Array(1);
  seed[0] = clientSeed;
  seed = shuffleSeed(shuffleSeed(shuffleSeed(seed)));
  for (let i = 255; i >= 0; i--) {
    seed = shuffleSeed(seed);
    const r = new Uint32Array(1);
    r[0] = (seed[0] + 31) % (i + 1);
    if (r[0] < 0) r[0] += i + 1;
    perm[i] = source[r[0]];
    source[r[0]] = source[i];
  }

  return (x: number, y: number): number => {
    const stretchOffset = (x + y) * STRETCH_2D;

    const xs = x + stretchOffset;
    const ys = y + stretchOffset;

    const xsb = Math.floor(xs);
    const ysb = Math.floor(ys);

    const squishOffset = (xsb + ysb) * SQUISH_2D;

    const dx0 = x - (xsb + squishOffset);
    const dy0 = y - (ysb + squishOffset);

    const xins = xs - xsb;
    const yins = ys - ysb;

    const inSum = xins + yins;
    const hash = (xins - yins + 1) |
                 (inSum << 1) |
                 ((inSum + yins) << 2) |
                 ((inSum + xins) << 4);

    let value = 0;
    let c = kLookup[hash];

    for (let i = 0; i < kContributionsPerHash; i++, c += kContributionsStride) {
      const cdx = kContributions[c + 0];
      const cdy = kContributions[c + 1];

      const dx = dx0 + cdx;
      const dy = dy0 + cdy;

      const attn = 2 - dx * dx - dy * dy;
      if (attn > 0) {
        const cxsb = kContributions[c + 2];
        const cysb = kContributions[c + 3];

        const px = xsb + cxsb;
        const py = ysb + cysb;

        const indexPart = perm[px & 0xff];
        const index = perm[(indexPart + py) & 0xff] & 0x0e;

        const absGradientX = (index & 2) ? 2 : 5;
        const absGradientY = 7 - absGradientX;
        const gradientX = (index & 4) ? -absGradientX : absGradientX;
        const gradientY = (index & 8) ? -absGradientY : absGradientY;

        const valuePart = gradientX * dx + gradientY * dy;
        value += attn * attn * attn * attn * valuePart;
      }
    }

    return value * NORM_2D;
  };
}

const base2D = [
  [[1, 1, 0], [1, 0, 1], [0, 0, 0]],
  [[1, 1, 0], [1, 0, 1], [2, 1, 1]],
];

const lookupPairs2D = [
  [0,  1],
  [1,  0],
  [4,  1],
  [17, 0],
  [20, 2],
  [21, 2],
  [22, 5],
  [23, 5],
  [26, 4],
  [39, 3],
  [42, 4],
  [43, 3],
];

const p2D = [
  [0, 0, 1, -1],
  [0, 0, -1, 1],
  [0, 2, 1, 1],
  [1, 2, 2, 0],
  [1, 2, 0, 2],
  [1, 0, 0, 0],
];

const kContributionsStride = 4;
const kContributionsPerHash = 4;

// Stored packed: accessed at query time
const [kContributions, kLookup] = (() => {
  const n = p2D.length * kContributionsStride * kContributionsPerHash;
  const contributions = new Float64Array(n);
  let i = 0;

  const append = (multiplier: number, xsb: number, ysb: number) => {
    if (!(i + 4 <= contributions.length)) throw new Error();
    contributions[i + 0] = -xsb - multiplier * SQUISH_2D;
    contributions[i + 1] = -ysb - multiplier * SQUISH_2D;
    contributions[i + 2] = xsb;
    contributions[i + 3] = ysb;
    i += 4;
  };

  for (const [baseIndex, multiplier, dx, dy] of p2D) {
    for (const [bm, bx, by] of base2D[baseIndex]) {
      append(bm, bx, by);
    }
    append(multiplier, dx, dy);
  }
  if (i !== contributions.length) throw new Error();

  const lookup = new Int8Array(128);
  lookup.fill(-1);
  for (const [i, j] of lookupPairs2D) {
    lookup[i] = kContributionsStride * kContributionsPerHash * j;
  }
  return [contributions, lookup];
})();
