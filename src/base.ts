type int = number;
type Color = [number, number, number, number];

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

const only = <T>(xs: T[]): T => {
  assert(xs.length === 1);
  return xs[0];
};

//////////////////////////////////////////////////////////////////////////////

interface Vec3 extends Float64Array {__type__: 'Vec3'};

const Vec3 = {
  create: (): Vec3 => new Float64Array(3) as Vec3,
  from: (x: number, y: number, z: number): Vec3 => {
    const result = new Float64Array(3) as Vec3;
    result[0] = x;
    result[1] = y;
    result[2] = z;
    return result;
  },
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

interface Mat4 extends Float32Array {__type__: 'Mat4'};

const Mat4 = {
  create: (): Mat4 => new Float32Array(16) as Mat4,
  identity: (d: Mat4) => {
    d[0]  = 1; d[1]  = 0; d[2]  = 0; d[3]  = 0;
    d[4]  = 0; d[5]  = 1; d[6]  = 0; d[7]  = 0;
    d[8]  = 0; d[9]  = 0; d[10] = 1; d[11] = 0;
    d[12] = 0; d[13] = 0; d[14] = 0; d[15] = 1;
  },
  multiply: (d: Mat4, a: Mat4, b: Mat4) => {
    const a00 = a[0];  const a01 = a[1];  const a02 = a[2];  const a03 = a[3];
    const a10 = a[4];  const a11 = a[5];  const a12 = a[6];  const a13 = a[7];
    const a20 = a[8];  const a21 = a[9];  const a22 = a[10]; const a23 = a[11];
    const a30 = a[12]; const a31 = a[13]; const a32 = a[14]; const a33 = a[15];

    let b0 = b[0]; let b1 = b[1]; let b2 = b[2]; let b3 = b[3];
    d[0]  = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    d[1]  = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    d[2]  = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    d[3]  = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    d[4]  = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    d[5]  = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    d[6]  = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    d[7]  = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    d[8]  = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    d[9]  = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    d[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    d[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    d[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    d[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    d[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    d[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  },
  multiplyVec3: (d: Vec3, a: Mat4, b: Vec3) => {
    const b0 = b[0]; const b1 = b[1]; const b2 = b[2];
    const w = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15];
    if (w) {
      const l = 1 / w;
      const x = a[0] * b0 + a[4] * b1 + a[8]  * b2 + a[12];
      const y = a[1] * b0 + a[5] * b1 + a[9]  * b2 + a[13];
      const z = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14];
      d[0] = x * l; d[1] = y * l; d[2] = z * l;
    } else {
      d[0] = 0; d[1] = 0; d[2] = 0;
    }
  },
  perspective: (d: Mat4, fov: number, aspect: number,
                near: number, far?: number) => {
    const f = 1 / Math.tan(fov / 2);
    const g = f * aspect;
    let x = 1;
    let y = -2 * near;
    if (far) {
      const n = 1 / (far - near);
      x = (far + near) * n;
      y = -(far * near * 2) * n;
    }
    d[0]  = f; d[1]  = 0; d[2]  = 0; d[3]  = 0;
    d[4]  = 0; d[5]  = g; d[6]  = 0; d[7]  = 0;
    d[8]  = 0; d[9]  = 0; d[10] = x; d[11] = 1;
    d[12] = 0; d[13] = 0; d[14] = y; d[15] = 0;
  },
  view: (d: Mat4, pos: Vec3, dir: Vec3) => {
    // Assumes dir is already normalized.
    // Assumes that "up" is (0, 1, 0).
    const p0 = pos[0]; const p1 = pos[1]; const p2 = pos[2];
    const z0 = dir[0]; const z1 = dir[1]; const z2 = dir[2];

    // x = normalize(cross(up, z));
    let x0 = z2; let x1 = 0; let x2 = -z0;
    let xl = Math.hypot(x0, x1, x2);
    if (xl) {
      xl = 1 / xl;
      x0 *= xl; x1 *= xl; x2 *= xl;
    } else {
      x0 = 0; x1 = 0; x2 = 0;
    }

    // y = normalize(cross(dir, x));
    let y0 = z1 * x2 - z2 * x1;
    let y1 = z2 * x0 - z0 * x2;;
    let y2 = z0 * x1 - z1 * x0;;
    let yl = Math.hypot(y0, y1, y2);
    if (yl) {
      yl = 1 / yl;
      y0 *= yl; y1 *= yl; y2 *= yl;
    } else {
      y0 = 0; y1 = 0; y2 = 0;
    }

    d[0] = x0; d[1] = y0; d[2]  = z0; d[3]  = 0;
    d[4] = x1; d[5] = y1; d[6]  = z1; d[7]  = 0;
    d[8] = x2; d[9] = y2; d[10] = z2; d[11] = 0;
    d[12] = -(x0 * p0 + x1 * p1 + x2 * p2);
    d[13] = -(y0 * p0 + y1 * p1 + y2 * p2);
    d[14] = -(z0 * p0 + z1 * p1 + z2 * p2);
    d[15] = 1;
  },
};

class Tensor2 {
  data: Int16Array;
  shape: [int, int];
  stride: [int, int];

  constructor(x: int, y: int) {
    this.data = new Int16Array(x * y);
    this.shape = [x, y];
    this.stride = [1, x];
  }

  get(x: int, y: int): int {
    return this.data[this.index(x, y)];
  }

  set(x: int, y: int, value: int) {
    this.data[this.index(x, y)] = value;
  }

  index(x: int, y: int): int {
    return x * this.stride[0] + y * this.stride[1];
  }
};

class Tensor3 {
  data: Int16Array;
  shape: [int, int, int];
  stride: [int, int, int];

  constructor(x: int, y: int, z: int) {
    this.data = new Int16Array(x * y * z);
    this.shape = [x, y, z];
    this.stride = [y, 1, x * y];
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

export {assert, drop, int, nonnull, only, Color, Mat4, Tensor2, Tensor3, Vec3};
