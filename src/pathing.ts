import {assert, int, nonnull} from './base.js';

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

  // An injection from Z x Z x Z -> Z suitable for use as a Map key.
  key(): int {
    const {x, y, z} = this;
    const ax = x < 0 ? -2 * x + 1 : 2 * x;
    const ay = y < 0 ? -2 * y + 1 : 2 * y;
    const az = z < 0 ? -2 * z + 1 : 2 * z;

    const a = ax + ay + az;
    const b = ax + ay;
    const c = ax;

    return int(a * (a + 1) * (a + 2) / 6 + b * (b + 1) / 2 + c);
  }

  toString(): string { return `Point(${this.x}, ${this.y}, {this.z})`; }
};

//////////////////////////////////////////////////////////////////////////////

class Direction extends Point {
  static none = new Direction(0,  0, 0);
  static n    = new Direction(0,  0, -1);
  static ne   = new Direction(1,  0, -1);
  static e    = new Direction(1,  0, 0);
  static se   = new Direction(1,  0, 1);
  static s    = new Direction(0,  0, 1);
  static sw   = new Direction(-1, 0, 1);
  static w    = new Direction(-1, 0, 0);
  static nw   = new Direction(-1, 0, -1);

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
  return; // Comment this line out to enable debug checks.
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

const AStarUnitCost = 16;
const AStarDiagonalPenalty = 2;
const AStarLOSDeltaPenalty = 1;
const AStarOccupiedPenalty = 64;

enum Status { FREE, BLOCKED, OCCUPIED };

const AStarHeuristic = (p: Point): number => {
  return 0;
};

const AStar = (source: Point, target: Point, check: (p: Point) => Status,
               record?: Point[]): Point[] | null => {
  const map: Map<int, AStarNode> = new Map();
  const heap: AStarHeap = [];

  const score = AStarHeuristic(source);
  const node = new AStarNode(source, null, 0, score);
  AStarHeapPush(heap, node);
  map.set(Point.origin.key(), node);

  while (heap.length > 0) {
    const cur = AStarHeapExtractMin(heap);
    if (record) record.push(cur);

    if (cur.equal(target)) {
      let current = cur;
      const result: Point[] = [];
      while (current.parent) {
        result.push(current);
        current = current.parent;
      }
      return result.reverse();
    }

    for (const direction of Direction.all) {
      const next = cur.add(direction);
      const test = next.equal(target) ? Status.FREE : check(next);
      if (test === Status.BLOCKED) continue;

      const diagonal = direction.x !== 0 && direction.y !== 0;
      const occupied = test === Status.OCCUPIED;
      const distance = cur.distance + AStarUnitCost +
                       (diagonal ? AStarDiagonalPenalty : 0) +
                       (occupied ? AStarOccupiedPenalty : 0);

      const key = next.sub(source).key();
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
        const score = distance + AStarHeuristic(next);
        const created = new AStarNode(next, cur, distance, score);
        AStarHeapPush(heap, created);
        map.set(key, created);
      }
    }
  }

  return null;
};

//////////////////////////////////////////////////////////////////////////////

export {AStar, Point, Status};
