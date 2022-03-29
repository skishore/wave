type int = number;

const assert = (x: boolean, message?: () => string) => {
  if (x) return;
  throw new Error(message ? message() : 'Assertion failed!');
};

const drop = <T>(xs: T[], x: T): void => {
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] !== x) continue;
    xs[i] = xs[xs.length - 1];
    xs.pop();
    return;
  }
};

const nonnull = <T>(x: T | null, message?: () => string): T => {
  if (x !== null) return x;
  throw new Error(message ? message() : 'Unexpected null!');
};

//////////////////////////////////////////////////////////////////////////////

type Vec3 = [number, number, number];

const Vec3 = {
  create: (): Vec3 => [0, 0, 0],
  from: (x: number, y: number, z: number): Vec3 => [x, y, z],
  copy: (d: Vec3, a: Vec3) => {
    d[0] = a[0];
    d[1] = a[1];
    d[2] = a[2];
  },
  set: (d: Vec3, x: number, y: number, z: number) => {
    d[0] = x;
    d[1] = y;
    d[2] = z;
  },
  add: (d: Vec3, a: Vec3, b: Vec3) => {
    d[0] = a[0] + b[0];
    d[1] = a[1] + b[1];
    d[2] = a[2] + b[2];
  },
  sub: (d: Vec3, a: Vec3, b: Vec3) => {
    d[0] = a[0] - b[0];
    d[1] = a[1] - b[1];
    d[2] = a[2] - b[2];
  },
  rotateX: (d: Vec3, a: Vec3, r: number) => {
    const sin = Math.sin(r);
    const cos = Math.cos(r);
    const ax = a[0], ay = a[1], az = a[2];
    d[0] = ax;
    d[1] = ay * cos - az * sin;
    d[2] = ay * sin + az * cos;
  },
  rotateY: (d: Vec3, a: Vec3, r: number) => {
    const sin = Math.sin(r);
    const cos = Math.cos(r);
    const ax = a[0], ay = a[1], az = a[2];
    d[0] = az * sin + ax * cos;
    d[1] = ay;
    d[2] = az * cos - ax * sin;
  },
  rotateZ: (d: Vec3, a: Vec3, r: number) => {
    const sin = Math.sin(r);
    const cos = Math.cos(r);
    const ax = a[0], ay = a[1], az = a[2];
    d[0] = ax * cos - ay * sin;
    d[1] = ax * sin + ay * cos;
    d[2] = az;
  },
  scale: (d: Vec3, a: Vec3, k: number) => {
    d[0] = a[0] * k;
    d[1] = a[1] * k;
    d[2] = a[2] * k;
  },
  scaleAndAdd: (d: Vec3, a: Vec3, b: Vec3, k: number) => {
    d[0] = a[0] + b[0] * k;
    d[1] = a[1] + b[1] * k;
    d[2] = a[2] + b[2] * k;
  },
  length: (a: Vec3) => {
    const x = a[0], y = a[1], z = a[2];
    return Math.sqrt(x * x + y * y + z * z);
  },
  normalize: (d: Vec3, a: Vec3) => {
    const length = Vec3.length(a);
    if (length !== 0) Vec3.scale(d, a, 1 / length);
  },
};

class Tensor3 {
  data: Uint32Array;
  shape: [int, int, int];
  stride: [int, int, int];

  constructor(x: int, y: int, z: int) {
    this.data = new Uint32Array(x * y * z);
    this.shape = [x, y, z];
    this.stride = [1, x * z, x];
  }

  get(x: int, y: int, z: int): int {
    return this.data[this.index(x, y, z)];
  }

  set(x: int, y: int, z: int, value: int) {
    this.data[this.index(x, y, z)] = value;
  }

  index(x: int, y: int, z: int): int {
    return x * this.stride[0] + y * this.stride[1] + z * this.stride[2];
  }
};

//////////////////////////////////////////////////////////////////////////////

export {assert, drop, int, nonnull, Tensor3, Vec3};
