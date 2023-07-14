import {assert, int, nonnull, Vec3} from './base.js';
import {sweep} from './sweep.js';

//////////////////////////////////////////////////////////////////////////////

class Point {
  readonly x: int;
  readonly y: int;
  readonly z: int;

  static origin = new Point(0, 0, 0);

  constructor(x: int, y: int, z: int) {
    this.x = x; this.y = y; this.z = z;
  }

  add(o: Point): Point {
    return new Point(int(this.x + o.x), int(this.y + o.y), int(this.z + o.z));
  }

  sub(o: Point): Point {
    return new Point(int(this.x - o.x), int(this.y - o.y), int(this.z - o.z));
  }

  distanceL2(o: Point): number {
    return Math.sqrt(this.distanceSquared(o));
  }

  distanceSquared(o: Point): int {
    const dx = this.x - o.x;
    const dy = this.y - o.y;
    const dz = this.z - o.z;
    return int(dx * dx + dy * dy + dz * dz);
  }

  equal(o: Point): boolean {
    return this.x === o.x && this.y === o.y && this.z === o.z;
  }

  toString(): string { return `Point(${this.x}, ${this.y}, ${this.z})`; }
};

//////////////////////////////////////////////////////////////////////////////

class Direction extends Point {
  static none = new Direction( 0,  0,  0);
  static n    = new Direction( 0,  0, -1);
  static ne   = new Direction( 1,  0, -1);
  static e    = new Direction( 1,  0,  0);
  static se   = new Direction( 1,  0,  1);
  static s    = new Direction( 0,  0,  1);
  static sw   = new Direction(-1,  0,  1);
  static w    = new Direction(-1,  0,  0);
  static nw   = new Direction(-1,  0, -1);
  static up   = new Direction( 0,  1,  0);
  static down = new Direction( 0, -1,  0);

  static all = [Direction.n, Direction.ne, Direction.e, Direction.se,
                Direction.s, Direction.sw, Direction.w, Direction.nw];

  static cardinal = [Direction.n, Direction.e, Direction.s, Direction.w];

  static diagonal = [Direction.ne, Direction.se, Direction.sw, Direction.nw];

  private constructor(x: int, y: int, z: int) { super(x, y, z); }

  static assert(point: Point): Direction {
    if (point.equal(Direction.none)) return Direction.none;
    return nonnull(Direction.all.filter(x => x.equal(point))[0]);
  }
};

//////////////////////////////////////////////////////////////////////////////

class PathNode extends Point {
  readonly jump: boolean;

  constructor(p: Point, jump: boolean) {
    super(p.x, p.y, p.z);
    this.jump = jump;
  }
};

//////////////////////////////////////////////////////////////////////////////

const precomputeDiagonal = (x: int, z: int): Point[] => {
  const result: Point[] = [];
  result.push(Point.origin);

  const check = (x: int, y: int, z: int) => {
    result.push(new Point(x, y, z));
    return true;
  };
  const min = Vec3.from(0, 0, 0);
  const max = Vec3.from(1, 1, 1);
  const delta = Vec3.from(x, 0, z);
  const impacts = Vec3.create();

  sweep(min, max, delta, impacts, check);

  result.forEach(p => {
    assert(p.x >= 0);
    assert(p.y === 0);
    assert(p.z >= 0);
  });
  return result;
};

const kDiagonalDistance = 5;
const kDiagonalArea = kDiagonalDistance * kDiagonalDistance;
const kDiagonalScratch: boolean[] = new Array(kDiagonalArea).fill(true);
const kDiagonalChecks: Point[][] = [];

for (let x = int(1); x < kDiagonalDistance; x++) {
  for (let z = int(1); z < kDiagonalDistance; z++) {
    if (x === 1 && z === 1) continue;
    if (x * x + z * z > kDiagonalArea) continue;
    kDiagonalChecks.push(precomputeDiagonal(x, z));
  }
}

const kSweepDistance = 16;
const kSweeps: Point[][] = [];

for (let z = int(0); z < kSweepDistance; z++) {
  for (let x = int(0); x < kSweepDistance; x++) {
    kSweeps.push(precomputeDiagonal(x, z));
  }
}

const canAttack = (source: Point, target: Point, check: Check): boolean => {
  if (!hasDirectPath(source, target, check)) return false;

  const sx = source.x;
  const sz = source.z;
  const dx = int(target.x - sx);
  const dz = int(target.z - sz);
  const ax = int(Math.abs(dx));
  const az = int(Math.abs(dz));
  if (ax * ax + az * az > kDiagonalArea) return false;
  if (ax <= 1 && az <= 1) return true;

  const index = ax + az * kSweepDistance;
  const sweep = kSweeps[index];
  const limit = sweep.length - 1;

  const y = int(source.y + 1);
  for (let i = 0; i < limit; i++) {
    const p = sweep[i];
    const x = dx > 0 ? int(sx + p.x) : int(sx - p.x);
    const z = dz > 0 ? int(sz + p.z) : int(sz - p.z);
    if (!check(new Point(x, y, z))) return false;
  }
  return true;
};

const hasDirectPath = (source: Point, target: Point, check: Check): boolean => {
  if (source.y < target.y) return false;

  const sx = source.x;
  const sz = source.z;
  const dx = int(target.x - sx);
  const dz = int(target.z - sz);
  const ax = int(Math.abs(dx));
  const az = int(Math.abs(dz));
  if (ax >= kSweepDistance || az >= kSweepDistance) return false;

  const index = ax + az * kSweepDistance;
  const sweep = kSweeps[index];
  const limit = sweep.length - 1;

  const last = sweep[limit];
  assert(last.x === ax);
  assert(last.z === az);

  let y = source.y;
  for (let i = 1; i < limit; i++) {
    const p = sweep[i];
    const x = dx > 0 ? int(sx + p.x) : int(sx - p.x);
    const z = dz > 0 ? int(sz + p.z) : int(sz - p.z);
    if (!check(new Point(x, y, z))) return false;
    while (y >= target.y && check(new Point(x, int(y - 1), z))) y--;
    if (y < target.y) return false;
  }
  return true;
};

//////////////////////////////////////////////////////////////////////////////

class AStarNode extends Point {
  public index: int | null = null;
  constructor(p: Point, public parent: AStarNode | null,
              public distance: number, public score: number) {
    super(p.x, p.y, p.z);
  }
}

// Min-heap implementation on lists of A* nodes. Nodes track indices as well.
type AStarHeap = AStarNode[];

const AStarHeapCheckInvariants = (heap: AStarHeap): void => {
  if (1 === 1) return; // Comment this line out to enable debug checks.
  heap.map(x => `(${x.index}, ${x.score})`).join('; ');
  heap.forEach((node, index) => {
    const debug = (label: string) => {
      const contents = heap.map(x => `(${x.index}, ${x.score})`).join('; ');
      return `Violated ${label} at ${index}: ${contents}`;
    };
    assert(node.index === index, () => debug('index'));
    if (index === 0) return;
    const parent_index = Math.floor((index - 1) / 2);
    assert(heap[parent_index]!.score <= node.score, () => debug('ordering'));
  });
};

const AStarHeapPush = (heap: AStarHeap, node: AStarNode): void => {
  assert(node.index === null);
  heap.push(node);
  AStarHeapify(heap, node, int(heap.length - 1));
};

const AStarHeapify = (heap: AStarHeap, node: AStarNode, index: int): void => {
  assert(0 <= index && index < heap.length);
  const score = node.score;

  while (index > 0) {
    const parent_index = int((index - 1) / 2);
    const parent = heap[parent_index]!;
    if (parent.score <= score) break;

    heap[index] = parent;
    parent.index = index;
    index = parent_index;
  }

  heap[index] = node;
  node.index = index;
  AStarHeapCheckInvariants(heap);
};

const AStarHeapExtractMin = (heap: AStarHeap): AStarNode => {
  assert(heap.length > 0);
  const result = heap[0]!;
  const node = heap.pop()!;
  result.index = null;

  if (!heap.length) return result;

  let index = int(0);
  while (2 * index + 1 < heap.length) {
    const c1 = heap[2 * index + 1]!;
    const c2 = heap[2 * index + 2] || c1;
    if (node.score <= Math.min(c1.score, c2.score)) break;

    const child_index = int(2 * index + (c1.score > c2.score ? 2 : 1));
    const child = (c1.score > c2.score ? c2 : c1);
    heap[index] = child;
    child.index = index;
    index = child_index;
  }

  heap[index] = node;
  node.index = index;
  AStarHeapCheckInvariants(heap);
  return result;
};

//////////////////////////////////////////////////////////////////////////////

type Check = (p: Point) => boolean;

const AStarDiagonalPenalty = 1;
const AStarJumpPenalty = 2;
const AStarUnitCost = 16;
const AStarUpCost   = 64;
const AStarDownCost = 4;
const AStarLimit = int(256);

const AStarKey = (p: Point, source: Point): int => {
  const result = (((p.x - source.x) & 0x3ff) << 0) |
                 (((p.y - source.y) & 0x3ff) << 10) |
                 (((p.z - source.z) & 0x3ff) << 20);
  return result as int;
};

const AStarHeuristic = (source: Point, target: Point) => {
  let dx = target.x - source.x;
  let dy = target.y - source.y;
  let dz = target.z - source.z;
  if (dx !== 0 || dy !== 0 || dz !== 0) {
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dx /= length;
    dy /= length;
    dz /= length;
  }

  return (p: Point): number => {
    const px = p.x - target.x;
    const py = p.y - target.y;
    const pz = p.z - target.z;

    const dot = px * dx + py * dy + pz * dz;
    const ox = px - dot * dx;
    const oy = py - dot * dy;
    const oz = pz - dot * dz;
    const off = Math.sqrt(ox * ox + oy * oy + oz * oz);

    const ax = Math.abs(px), az = Math.abs(pz);
    const base = Math.max(ax, az) * AStarUnitCost +
                 Math.min(ax, az) * AStarDiagonalPenalty;
    return base + off + py * (py > 0 ? AStarDownCost : -AStarUpCost);
  };
};

const AStarHeight =
    (source: Point, target: Point, check: Check): int | null => {
  if (!check(target)) {
    const up = Direction.up;
    const jump = check(source.add(up)) && check(target.add(up));
    return jump ? int(target.y + 1) : null;
  }
  return AStarDrop(target, check);
};

const AStarDropTmpPoint = new Point(0, 0, 0);

const AStarDrop = (p: Point, check: Check): int => {
  const floor = AStarDropTmpPoint;
  (floor as any).x = p.x;
  (floor as any).y = p.y - 1;
  (floor as any).z = p.z;
  while (floor.y >= 0 && check(floor)) {
    (floor as any).y--;
  }
  return int(floor.y + 1);
};

const AStarAdjust = (p: Point, y: int): Point => {
  return y === p.y ? p : new Point(p.x, y, p.z);
};

const AStarNeighbors =
    (source: Point, check: Check, first: boolean): Point[] => {
  const result = [];
  const {up, down} = Direction;

  // The source may not be a grounded block. Every other node in the path
  // will be grounded, due to the logic in this neighbor lookup function.
  const y = first ? AStarDrop(source, check) : source.y;
  const grounded = y === source.y;
  if (!grounded) result.push(new Point(source.x, y, source.z));

  let blocked = 0;
  const directions = Direction.all;

  for (let i = 0; i < 8; i++) {
    const diagonal = i & 4;
    const index = ((i & 3) << 1) | (diagonal ? 1 : 0);
    if (diagonal && (blocked & (1 << index))) continue;
    const dir = directions[index];

    const next = source.add(dir);
    const ny = AStarHeight(source, next, check);
    if (!diagonal && (ny === null || ny > next.y)) {
      blocked |= 1 << ((index - 1) & 7);
      blocked |= 1 << ((index + 1) & 7);
    }
    if (ny === null) continue;

    result.push(AStarAdjust(next, ny));

    if (ny < next.y && grounded &&
        check(source.add(up)) && check(next.add(up))) {
      const flat_limit = 4;
      const jump_limit = 3;
      if (!diagonal) {
        for (let j = 0, jump = next; j < flat_limit; j++) {
          jump = jump.add(dir);
          const jump_up = jump.add(up);
          if (!check(jump_up)) break;
          if (!(j < jump_limit || check(jump))) break;
          const jy = AStarDrop(jump_up, check);
          result.push(AStarAdjust(jump, jy));
          if (jy > source.y) break;
        }
      } else {
        const d = kDiagonalDistance;
        const scratch = kDiagonalScratch;
        const check_two = (p: Point) => check(p) && check(p.add(up));

        for (let i = 1; i < kDiagonalDistance; i++) {
          const x = int(i * dir.x), z = int(i * dir.z);
          scratch[i * 1] = check_two(source.add(new Point(x, 0, 0)));
          scratch[i * d] = check_two(source.add(new Point(0, 0, z)));
        }

        for (const path of kDiagonalChecks) {
          let okay = true;
          const limit = path.length - 1;
          for (let j = 1; j < limit; j++) {
            const value = path[j];
            const index = value.x + value.z * d;
            if (scratch[index]) continue;
            okay = false;
            break;
          }
          const value = path[limit];
          const index = value.x + value.z * d;
          if (!okay) {
            scratch[index] = false;
          } else {
            const dx = int(value.x * dir.x), dz = int(value.z * dir.z);
            const jump = source.add(new Point(dx, 0, dz));
            okay = check_two(jump);
            scratch[index] = okay;
            if (okay) result.push(AStarAdjust(jump, AStarDrop(jump, check)));
          }
        }
      }
    }
  }
  return result;
};

const AStarCore = (source: Point, target: Point, check: Check): Point[] => {
  let count = int(0);
  const limit = AStarLimit;

  const sy = AStarDrop(source, check);
  source = sy >= source.y - 1 ? AStarAdjust(source, sy) : source;
  const ty = AStarDrop(target, check);
  const drop = target.y - ty;
  target = AStarAdjust(target, ty);

  let best: AStarNode | null = null;
  const map: Map<int, AStarNode> = new Map();
  const heap: AStarHeap = [];

  const heuristic = AStarHeuristic(source, target);
  const score = heuristic(source);
  const node = new AStarNode(source, null, 0, score);
  map.set(AStarKey(source, source), node);
  AStarHeapPush(heap, node);

  while (count < limit && heap.length > 0) {
    const cur = AStarHeapExtractMin(heap);
    //console.log(`  ${count}: ${cur.toString()}: ` +
    //            `distance = ${cur.distance}, ` +
    //            `score = ${cur.score}`);
    count = int(count + 1);

    if (cur.equal(target)) {
      best = cur;
      break;
    }

    for (const next of AStarNeighbors(cur, check, count === 1)) {
      const dy = next.y - cur.y;
      const ax = Math.abs(next.x - cur.x);
      const az = Math.abs(next.z - cur.z);
      const distance = cur.distance +
                       Math.max(ax, az) * AStarUnitCost +
                       Math.min(ax, az) * AStarDiagonalPenalty +
                       ((ax > 1 || az > 1) ? AStarJumpPenalty : 0) +
                       dy * (dy > 0 ? AStarUpCost : -AStarDownCost);
      const key = AStarKey(next, source);
      const existing = map.get(key);

      // index !== null is a check to see if we've already popped this node
      // from the heap. We need it because our heuristic is not admissible.
      //
      // Using such a heuristic substantially speeds up search in easy cases,
      // with the downside that we don't always find an optimal path.
      if (existing && existing.index !== null && existing.distance > distance) {
        existing.score += distance - existing.distance;
        existing.distance = distance;
        existing.parent = cur;
        AStarHeapify(heap, existing, existing.index);
      } else if (!existing) {
        const score = distance + heuristic(next);
        const created = new AStarNode(next, cur, distance, score);
        AStarHeapPush(heap, created);
        map.set(key, created);
      }
    }
  }

  if (best === null) {
    const heuristic = (x: AStarNode) => x.score - x.distance;
    for (const node of map.values()) {
      if (!best || heuristic(node) < heuristic(best)) best = node;
    }
  }

  assert(best !== null);
  const result: Point[] = [];
  while (best) {
    result.push(best);
    best = best.parent;
  }
  return result.reverse();
};


const AStar = (source: Point, target: Point, check: Check): PathNode[] => {
  //console.log(`AStar: ${source.toString()} -> ${target.toString()}`);

  const sy = AStarDrop(source, check);
  const sdrop = sy === source.y - 1 ? 1 : 0;
  source = AStarAdjust(source, int(source.y - sdrop));

  const ty = AStarDrop(target, check);
  const tdrop = target.y - ty;
  target = AStarAdjust(target, int(target.y - tdrop));

  const path = AStarCore(source, target, check);
  assert(path.length > 0);

  if (sdrop) {
    const original = path[0].add(Direction.up);
    if (path.length > 1 && path[1].y > path[0].y) {
      path[0] = original;
    } else {
      path.unshift(original);
    }
  }

  if (tdrop > 1) {
    for (let i = 0; i < path.length - 1; i++) {
      if (path[i].y - path[i + 1].y > 1) return [];
    }
  }

  const nodes = path.map((p, i) => {
    if (i === 0) return new PathNode(p, false);
    const prev = path[i - 1];
    const jump = Math.abs(prev.x - p.x) > 1 || Math.abs(prev.z - p.z) > 1;
    return new PathNode(p, jump);
  });

  // NOTE: We could use dynamic programming here for further simplification.
  assert(nodes.length > 0);
  const result: PathNode[] = [nodes[0]];
  for (let i = int(2); i < nodes.length; i++) {
    const prev = result[result.length - 1], next = nodes[i];
    if (!prev.jump && !next.jump && hasDirectPath(prev, next, check)) continue;
    result.push(nodes[i - 1]);
  }
  if (nodes.length > 1) result.push(nodes[nodes.length - 1]);

  //console.log(`Found ${result.length}-node path:`);
  //for (const step of result) {
  //  console.log(`  ${step.toString()}`);
  //}
  return result;
};

//////////////////////////////////////////////////////////////////////////////

export {AStar, Check, Direction, PathNode, Point, canAttack};
