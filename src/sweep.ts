type Point = [number, number, number];

type Check = (x: Point) => boolean;

const kSweepShift = 8;
const kSweepResolution = 1 << kSweepShift;
const kSweepMask = kSweepResolution - 1;

const kSpeeds = [0, 0, 0, 0];
const kDistances = [0, 0, 0, kSweepResolution];
const kCellCur: Point = [0, 0, 0];
const kCellEnd: Point = [0, 0, 0];

const sweep = (min: Point, max: Point, delta: Point, impacts: Point, check: Check) => {
  for (let i = 0; i < 3; i++) {
    min[i] = (min[i] * kSweepResolution) | 0;
    max[i] = (max[i] * kSweepResolution) | 0;
    delta[i] = (delta[i] * kSweepResolution) | 0;
    impacts[i] = 0;
  }

  while (delta[0] || delta[1] || delta[2]) {
    let best = 3;
    let bounded = true;

    for (let i = 0; i < 3; i++) {
      const step = delta[i];
      const speed = Math.abs(step);
      const place = step > 0 ? max[i] : -min[i];
      const distance = kSweepResolution - ((place - 1) & kSweepMask);
      kSpeeds[i] = speed;
      kDistances[i] = distance;

      bounded = bounded && speed < distance;
      const better = speed * kDistances[best] > kSpeeds[best] * distance;
      if (better) best = i;
    }

    if (bounded) {
      for (let i = 0; i < 3; i++) {
        min[i] += delta[i];
        max[i] += delta[i];
        delta[i] = 0;
      }
      break;
    }

    const factor = kDistances[best] / kSpeeds[best];
    for (let i = 0; i < 3; i++) {
      const speed = kSpeeds[i];
      const distance = kDistances[i];
      const move = i !== best
        ? Math.min(distance - 1, (speed * factor) | 0)
        : distance;
      const step = move * Math.sign(delta[i]);
      min[i] += step;
      max[i] += step;
      delta[i] -= step;
    }

    {
      const i = best;
      kCellCur[i] = (delta[i] > 0 ? max[i] : min[i] - 1) >> kSweepShift;

      const j = i < 2 ? i + 1 : i - 2;
      const k = i < 1 ? i + 2 : i - 1;
      kCellCur[j] = min[j] >> kSweepShift;
      kCellEnd[j] = (max[j] - 1) >> kSweepShift;
      kCellCur[k] = min[k] >> kSweepShift;
      kCellEnd[k] = (max[k] - 1) >> kSweepShift;

      let done = false;
      for (; !done && kCellCur[j] <= kCellEnd[j]; kCellCur[j]++) {
        for (; !done && kCellCur[k] <= kCellEnd[k]; kCellCur[k]++) {
          if (check(kCellCur)) continue;
          const step =  delta[i] > 0 ? -1 : 1;
          impacts[i] = -step;
          min[i] += step;
          max[i] += step;
          delta[i] = 0;
          done = true;
        }
      }
    }
  }

  for (let i = 0; i < 3; i++) {
    min[i] = min[i] / kSweepResolution;
    max[i] = max[i] / kSweepResolution;
  }
};

export {Point, sweep};
