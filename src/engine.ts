import {assert, drop, int, nonnull} from './base.js';
import {Color, Tensor2, Tensor3, Vec3} from './base.js';
import {EntityComponentSystem} from './ecs.js';
import {HighlightMesh, InstancedMesh, Geometry, Mesh} from './renderer.js';
import {Instance, LightTexture, Renderer, Texture, VoxelMesh} from './renderer.js';
import {TerrainMesher} from './mesher.js';
import {kSweepResolution, sweep} from './sweep.js';

//////////////////////////////////////////////////////////////////////////////

type Input = 'up' | 'left' | 'down' | 'right' | 'hover' | 'call' |
             'mouse0' | 'mouse1' | 'space' | 'pointer';

interface KeyBinding {input: Input, handled: boolean};

class Container {
  element: Element;
  canvas: HTMLCanvasElement;
  stats: Element | null;
  bindings: Map<int, KeyBinding>;
  inputs: Record<Input, boolean>;
  deltas: {x: int, y: int, scroll: int};

  constructor(id: string) {
    this.element = nonnull(document.getElementById(id), () => id);
    this.canvas = nonnull(this.element.querySelector('canvas'));
    this.stats = document.getElementById('stats');
    this.inputs = {
      up: false,
      left: false,
      down: false,
      right: false,
      hover: false,
      call: false,
      space: false,
      mouse0: false,
      mouse1: false,
      pointer: false,
    };
    this.deltas = {x: 0, y: 0, scroll: 0};

    this.bindings = new Map();
    this.addBinding('W', 'up');
    this.addBinding('A', 'left');
    this.addBinding('S', 'down');
    this.addBinding('D', 'right');
    this.addBinding('E', 'hover');
    this.addBinding('Q', 'call');
    this.addBinding(' ', 'space');

    const canvas = this.canvas;
    const target = nonnull(this.canvas.parentElement);
    target.addEventListener('click', (e: Event) => {
      if (this.inputs.pointer) return;
      this.onMimicPointerLock(e, true);
      this.insistOnPointerLock();
    });

    document.addEventListener('keydown', e => this.onKeyInput(e, true));
    document.addEventListener('keyup', e => this.onKeyInput(e, false));
    document.addEventListener('mousedown', e => this.onMouseDown(e));
    document.addEventListener('mousemove', e => this.onMouseMove(e));
    document.addEventListener('touchmove', e => this.onMouseMove(e));
    document.addEventListener('pointerlockchange', e => this.onPointerInput(e));
    document.addEventListener('wheel', e => this.onMouseWheel(e));
  }

  displayStats(stats: string): void {
    if (this.stats) this.stats.textContent = stats;
  }

  private addBinding(key: string, input: Input): void {
    assert(key.length === 1);
    this.bindings.set(int(key.charCodeAt(0)), {input, handled: false});
  }

  private insistOnPointerLock(): void {
    if (!this.inputs.pointer) return;
    if (document.pointerLockElement === this.canvas) return;
    this.canvas.requestPointerLock();
    setTimeout(() => this.insistOnPointerLock(), 100);
  }

  private onKeyInput(e: Event, down: boolean): void {
    if (!this.inputs.pointer) return;
    const keycode = int((e as KeyboardEvent).keyCode);
    if (keycode === 27) return this.onMimicPointerLock(e, false);
    const binding = this.bindings.get(keycode);
    if (!binding || binding.handled === down) return;
    this.onInput(e, binding.input, down);
    binding.handled = down;
  }

  private onMouseDown(e: Event): void {
    if (!this.inputs.pointer) return;
    const button = (e as MouseEvent).button;
    if (button === 0) this.inputs.mouse0 = true;
    if (button !== 0) this.inputs.mouse1 = true;
  }

  private onMouseMove(e: Event): void {
    if (!this.inputs.pointer) return;
    this.deltas.x += (e as MouseEvent).movementX;
    this.deltas.y += (e as MouseEvent).movementY;
  }

  private onMouseWheel(e: Event): void {
    if (!this.inputs.pointer) return;
    this.deltas.scroll += (e as any).deltaY;
  }

  private onMimicPointerLock(e: Event, locked: boolean): void {
    if (locked) this.element.classList.remove('paused');
    if (!locked) this.element.classList.add('paused');
    this.onInput(e, 'pointer', locked);
  }

  private onPointerInput(e: Event): void {
    const locked = document.pointerLockElement === this.canvas;
    this.onMimicPointerLock(e, locked);
  }

  private onInput(e: Event, input: Input, state: boolean): void {
    this.inputs[input] = state;
    e.stopPropagation();
    e.preventDefault();
  }
};

//////////////////////////////////////////////////////////////////////////////

type BlockId = int & {__type__: 'BlockId'};
type MaterialId = int & {__type__: 'MaterialId'};
type MaybeMaterialId = MaterialId | 0;

interface Material {
  liquid: boolean,
  texture: Texture,
  textureIndex: int,
};

interface BlockSprite {
  url: string,
  x: int,
  y: int,
  w: int,
  h: int,
};

const kBlack: Color = [0, 0, 0, 1];
const kWhite: Color = [1, 1, 1, 1];

const kNoMaterial = 0 as 0;

const kEmptyBlock = 0 as BlockId;
const kUnknownBlock = 1 as BlockId;

class Registry {
  // If a block's light value is -1, then the block is opaque and it always
  // has a computed light level of 0.
  //
  // Otherwise, the block casts a light equal to this value (but its computed
  // light may be greater than that, due to light from its neighbors).
  //
  light: int[];
  opaque: boolean[];
  solid: boolean[];
  private faces: MaybeMaterialId[];
  private meshes: (InstancedMesh | null)[];
  private materials: Material[];
  private ids: Map<string, MaterialId>;
  private helper: WasmHelper;
  private renderer: Renderer;

  constructor(helper: WasmHelper, renderer: Renderer) {
    this.helper = helper;
    this.renderer = renderer;

    this.opaque = [false, false];
    this.solid = [false, true];
    this.light = [0, 0];
    this.faces = []
    for (let i = 0; i < 12; i++) {
      this.faces.push(kNoMaterial);
    }
    this.meshes = [null, null];
    this.materials = [];
    this.ids = new Map();

    this.registerBlock(kEmptyBlock);
    this.registerBlock(kUnknownBlock);
  }

  addBlock(xs: string[], solid: boolean, light: int = 0): BlockId {
    type Materials = [string, string, string, string, string, string];
    const materials = ((): Materials => {
      switch (xs.length) {
        // All faces for this block use same material.
        case 1: return [xs[0], xs[0], xs[0], xs[0], xs[0], xs[0]];
        // xs specifies [top/bottom, sides]
        case 2: return [xs[1], xs[1], xs[0], xs[0], xs[1], xs[1]];
        // xs specifies [top, bottom, sides]
        case 3: return [xs[2], xs[2], xs[0], xs[1], xs[2], xs[2]];
        // xs specifies [+x, -x, +y, -y, +z, -z]
        case 6: return xs as Materials;
        // Uninterpretable case.
        default: throw new Error(`Unexpected materials: ${JSON.stringify(xs)}`);
      }
    })();

    let opaque = true;
    materials.forEach(x => {
      const id = this.ids.get(x);
      if (id === undefined) throw new Error(`Unknown material: ${x}`);
      const material = id + 1 as MaterialId;
      this.faces.push(material);

      const texture = this.getMaterialData(material).texture;
      const alphaBlend = texture.color[3] < 1;
      const alphaTest  = texture.alphaTest;
      if (alphaBlend || alphaTest) opaque = false;
    });

    light = opaque && light === 0 ? -1 : light;
    const result = this.opaque.length as BlockId;
    this.opaque.push(opaque);
    this.solid.push(solid);
    this.light.push(light);
    this.meshes.push(null);
    this.registerBlock(result);
    return result;
  }

  addBlockMesh(mesh: InstancedMesh, solid: boolean, light: int = 0): BlockId {
    const result = this.opaque.length as BlockId;
    for (let i = 0; i < 6; i++) this.faces.push(kNoMaterial);
    this.meshes.push(mesh);
    this.opaque.push(false);
    this.solid.push(solid);
    this.light.push(light);
    this.registerBlock(result);
    return result;
  }

  addMaterial(name: string, texture: Texture, liquid: boolean = false) {
    assert(name.length > 0, () => 'Empty material name!');
    assert(!this.ids.has(name), () => `Duplicate material: ${name}`);
    const id = this.materials.length as MaterialId;
    const textureIndex = this.renderer.addTexture(texture);
    this.ids.set(name, id);
    this.materials.push({liquid, texture, textureIndex});
    this.registerMaterial(id);
  }

  // faces has 6 elements for each block type: [+x, -x, +y, -y, +z, -z]
  getBlockFaceMaterial(id: BlockId, face: int): MaybeMaterialId {
    return this.faces[id * 6 + face];
  }

  getBlockMesh(id: BlockId): InstancedMesh | null {
    return this.meshes[id];
  }

  getMaterialData(id: MaterialId): Material {
    assert(0 < id && id <= this.materials.length);
    return this.materials[id - 1];
  }

  private registerBlock(id: BlockId): void {
    assert(0 <= id && id < this.opaque.length);
    const b = 6 * id;
    const faces = this.faces;
    this.helper.module.asm.registerBlock(
        id, this.opaque[id], this.solid[id], this.light[id],
        faces[b + 0], faces[b + 1], faces[b + 2],
        faces[b + 3], faces[b + 4], faces[b + 5]);
  }

  private registerMaterial(id: MaterialId): void {
    assert(0 <= id && id < this.materials.length);
    const material = this.materials[id]
    const [r, g, b, a] = material.texture.color;
    this.helper.module.asm.registerMaterial(
        id, material.liquid, material.texture.alphaTest,
        material.textureIndex, r, g, b, a);
  }
};

//////////////////////////////////////////////////////////////////////////////

class Performance {
  private now: any;
  private index: int;
  private ticks: int[];
  private last: number;
  private sum: int;

  constructor(now: any, samples: int) {
    assert(samples > 0);
    this.now = now;
    this.index = 0;
    this.ticks = new Array(samples).fill(0);
    this.last = 0;
    this.sum = 0;
  }

  begin() {
    this.last = this.now.now();
  }

  end() {
    const index = this.index;
    const next_index = index + 1;
    this.index = int(next_index < this.ticks.length ? next_index : 0);
    const tick = int(Math.round(1000 * (this.now.now() - this.last)));
    this.sum += tick - this.ticks[index];
    this.ticks[index] = tick;
  }

  frame(): int {
    return this.index;
  }

  max(): number {
    return Math.max.apply(null, this.ticks);
  }

  mean(): number {
    return this.sum / this.ticks.length;
  }
};

//////////////////////////////////////////////////////////////////////////////

const kTickResolution = 4;
const kTicksPerFrame = 4;
const kTicksPerSecond = 60;

type Callback = (dt: number) => void;

class Timing {
  private now: any;
  private remesh: Callback;
  private render: Callback;
  private update: Callback;
  private renderBinding: () => void;
  private updateDelay: number;
  private updateLimit: number;
  private lastRender: int;
  private lastUpdate: int;
  remeshPerf: Performance;
  renderPerf: Performance;
  updatePerf: Performance;

  constructor(remesh: Callback, render: Callback, update: Callback) {
    this.now = performance || Date;
    this.remesh = remesh;
    this.render = render;
    this.update = update;

    const now = this.now.now();
    this.lastRender = now;
    this.lastUpdate = now;

    this.renderBinding = this.renderHandler.bind(this);
    requestAnimationFrame(this.renderBinding);

    this.updateDelay = 1000 / kTicksPerSecond;
    this.updateLimit = this.updateDelay * kTicksPerFrame;
    const updateInterval = this.updateDelay / kTickResolution;
    setInterval(this.updateHandler.bind(this), updateInterval);

    this.remeshPerf = new Performance(this.now, 60);
    this.renderPerf = new Performance(this.now, 60);
    this.updatePerf = new Performance(this.now, 60);
  }

  renderHandler() {
    requestAnimationFrame(this.renderBinding);
    this.updateHandler();

    const now = this.now.now();
    const dt = (now - this.lastRender) / 1000;
    this.lastRender = now;

    try {
      this.remeshPerf.begin();
      this.remesh(dt);
      this.remeshPerf.end();
      this.renderPerf.begin();
      this.render(dt);
      this.renderPerf.end();
    } catch (e) {
      this.onError(e);
    }
  }

  private updateHandler() {
    let now = this.now.now();
    const delay = this.updateDelay;
    const limit = now + this.updateLimit;

    while (this.lastUpdate + delay < now) {
      try {
        this.updatePerf.begin();
        this.update(delay / 1000);
        this.updatePerf.end();
      } catch (e) {
        this.onError(e);
      }
      this.lastUpdate += delay;
      now = this.now.now();

      if (now > limit) {
        this.lastUpdate = now;
        break;
      }
    }
  }

  private onError(e: any) {
    this.remesh = this.render = this.update = () => {};
    console.error(e);
  }
};

//////////////////////////////////////////////////////////////////////////////

type Loader = (x: int, z: int, column: Column) => void;

class Column {
  private decorations: int[];
  private data: Int16Array;
  private last: int = 0;
  private size: int = 0;

  // For computing "chunk equi-levels" efficiently. An "equi-level" is a
  // height in a chunk at which all columns have the same block.
  private mismatches: Int16Array;
  private reference_data: Int16Array;
  private reference_size: int = 0;

  constructor() {
    this.decorations = [];
    this.data = new Int16Array(2 * kWorldHeight);
    this.mismatches = new Int16Array(kWorldHeight);
    this.reference_data = new Int16Array(2 * kWorldHeight);
  }

  clear(): void {
    this.decorations.length = 0;
    this.last = 0;
    this.size = 0;
  }

  fillChunk(x: int, z: int, chunk: Chunk, first: boolean): void {
    let last = int(0);
    for (let i = 0; i < this.size; i++) {
      const offset = 2 * i;
      const block = this.data[offset + 0] as BlockId;
      const level = int(this.data[offset + 1]);
      chunk.setColumn(x, z, last, int(level - last), block);
      last = level;
    }
    for (let i = 0; i < this.decorations.length; i += 2) {
      const block = this.decorations[i + 0] as BlockId
      const level = this.decorations[i + 1];
      chunk.setColumn(x, z, level, 1, block);
    }
    this.detectEquiLevelChanges(first);
  }

  fillEquilevels(equilevels: Int8Array): void {
    let current = 0;
    const mismatches = this.mismatches;
    for (let i = 0; i < kWorldHeight; i++) {
      current += mismatches[i];
      equilevels[i] = (current === 0 ? 1 : 0);
    }
  }

  overwrite(block: BlockId, y: int) {
    if (!(0 <= y && y < kWorldHeight)) return;
    this.decorations.push(block);
    this.decorations.push(y);
  }

  push(block: BlockId, height: int): void {
    height = int(Math.min(height, kWorldHeight));
    if (height <= this.last) return;
    this.last = height;
    const offset = 2 * this.size;
    this.data[offset + 0] = block;
    this.data[offset + 1] = this.last;
    this.size++;
  }

  getNthBlock(n: int, bedrock: BlockId): BlockId {
    return n < 0 ? bedrock : this.data[2 * n + 0] as BlockId;
  }

  getNthLevel(n: int): int {
    return n < 0 ? 0 : int(this.data[2 * n + 1]);
  }

  getSize(): int {
    return this.size;
  }

  private detectEquiLevelChanges(first: boolean): void {
    if (this.last < kWorldHeight) {
      const offset = 2 * this.size;
      this.data[offset + 0] = kEmptyBlock;
      this.data[offset + 1] = kWorldHeight;
      this.size++;
    }

    if (first) this.mismatches.fill(0);

    for (let i = 0; i < this.decorations.length; i += 2) {
      const level = this.decorations[i + 1];
      this.mismatches[level]++;
      if (level + 1 < kWorldHeight) this.mismatches[level + 1]--;
    }

    if (first) {
      for (let i = 0; i < 2 * this.size; i++) {
        this.reference_data[i] = this.data[i];
      }
      this.reference_size = this.size;
      return;
    }

    let matched = true;
    let di = 0, ri = 0;
    let d_start = 0, r_start = 0;
    while (di < this.size && ri < this.reference_size) {
      const d_offset = 2 * di;
      const d_block = this.data[d_offset + 0];
      const d_limit = this.data[d_offset + 1];

      const r_offset = 2 * ri;
      const r_block = this.reference_data[r_offset + 0];
      const r_limit = this.reference_data[r_offset + 1];

      if (matched !== (d_block === r_block)) {
        const height = Math.max(d_start, r_start);
        this.mismatches[height] += matched ? 1 : -1;
        matched = !matched;
      }

      if (d_limit <= r_limit) {
        d_start = d_limit;
        di++;
      }
      if (r_limit <= d_limit) {
        r_start = r_limit;
        ri++;
      }
    }

    assert(di === this.size);
    assert(ri === this.reference_size);
    assert(d_start === kWorldHeight);
    assert(r_start === kWorldHeight);
  }
};

//////////////////////////////////////////////////////////////////////////////

interface CircleElement {
  cx: int;
  cz: int;
  dispose(): void;
};

class Circle<T extends CircleElement> {
  private center_x: int = 0;
  private center_z: int = 0;
  private deltas: Int32Array;
  private points: Int32Array;
  private elements: (T | null)[];
  private shift: int;
  private mask: int;

  constructor(radius: number) {
    const bound = radius * radius;
    const floor = Math.floor(radius);

    const points = [];
    for (let i = -floor; i <= floor; i++) {
      for (let j = -floor; j <= floor; j++) {
        const distance = i * i + j * j;
        if (distance > bound) continue;
        points.push({i, j, distance});
      }
    }
    points.sort((a, b) => a.distance - b.distance);

    let current = 0;
    this.deltas = new Int32Array(floor + 1);
    this.points = new Int32Array(2 * points.length);
    for (const {i, j} of points) {
      this.points[current++] = i;
      this.points[current++] = j;
      const ai = Math.abs(i), aj = Math.abs(j);
      this.deltas[ai] = Math.max(this.deltas[ai], aj);
    }
    assert(current === this.points.length);

    let shift = 0;
    while ((1 << shift) < 2 * floor + 1) shift++;
    this.elements = new Array(1 << (2 * shift)).fill(null);
    this.shift = int(shift);
    this.mask = int((1 << shift) - 1);
  }

  center(center_x: int, center_z: int): void {
    if (center_x === this.center_x && center_z === this.center_z) return;
    this.each((cx: int, cz: int): boolean => {
      const ax = Math.abs(cx - center_x);
      const az = Math.abs(cz - center_z);
      if (az <= this.deltas[ax]) return false;

      const index = this.index(cx, cz);
      const value = this.elements[index];
      if (value === null) return false;
      value.dispose();
      this.elements[index] = null;
      return false;
    });
    this.center_x = center_x;
    this.center_z = center_z;
  }

  each(fn: (cx: int, cz: int) => boolean) {
    const {center_x, center_z, points} = this;
    const length = points.length;
    for (let i = 0; i < length; i += 2) {
      const done = fn(int(points[i] + center_x), int(points[i + 1] + center_z));
      if (done) break;
    }
  }

  get(cx: int, cz: int): T | null {
    const value = this.elements[this.index(cx, cz)];
    return value && value.cx === cx && value.cz === cz ? value : null;
  }

  set(cx: int, cz: int, value: T): void {
    const index = this.index(cx, cz);
    assert(this.elements[index] === null);
    this.elements[index] = value;
  }

  private index(cx: int, cz: int): int {
    const {mask, shift} = this;
    return int(((cz & mask) << shift) | (cx & mask));
  }
};

//////////////////////////////////////////////////////////////////////////////

const kChunkBits   = int(4);
const kChunkWidth  = int(1 << kChunkBits);
const kChunkMask   = int(kChunkWidth - 1);
const kHeightBits  = int(8);
const kWorldHeight = int(1 << kHeightBits);

const kChunkShiftX = kHeightBits;
const kChunkShiftZ = kHeightBits + kChunkBits;

const kChunkRadius = 12;

const kNumChunksToLoadPerFrame = 1;
const kNumChunksToMeshPerFrame = 1;
const kNumChunksToLightPerFrame = 4;
const kNumLODChunksToMeshPerFrame = 1;

const kFrontierLOD = 2;
const kFrontierRadius = 8;
const kFrontierLevels = 0;

// Enable debug assertions for the equi-levels optimization.
const kCheckEquilevels = false;

// List of neighboring chunks to include when meshing.
type Point = [int, int, int];
const kNeighborOffsets = ((): [Point, Point, Point, Point][] => {
  const W = kChunkWidth;
  const H = kWorldHeight;
  const L = int(W - 1);
  const N = int(W + 1);
  return [
    [[ 0,  0,  0], [1, 1, 1], [0, 0, 0], [W, H, W]],
    [[-1,  0,  0], [0, 1, 1], [L, 0, 0], [1, H, W]],
    [[ 1,  0,  0], [N, 1, 1], [0, 0, 0], [1, H, W]],
    [[ 0,  0, -1], [1, 1, 0], [0, 0, L], [W, H, 1]],
    [[ 0,  0,  1], [1, 1, N], [0, 0, 0], [W, H, 1]],
    [[-1,  0,  1], [0, 1, N], [L, 0, 0], [1, H, 1]],
    [[ 1,  0,  1], [N, 1, N], [0, 0, 0], [1, H, 1]],
    [[-1,  0, -1], [0, 1, 0], [L, 0, L], [1, H, 1]],
    [[ 1,  0, -1], [N, 1, 0], [0, 0, L], [1, H, 1]],
  ];
})();
const kZone = kNeighborOffsets.map(x => ({x: x[0][0], z: x[0][2]}));
const kNeighbors = kZone.slice(1);

class LightSpread {
  constructor(public diff: int, public mask: int, public test: int) {}
}
const kSpread : LightSpread[] = [
  new LightSpread(int(-1 << 8),  int(0x0f00), int(0x0000)),
  new LightSpread(int(+1 << 8),  int(0x0f00), int(0x0f00)),
  new LightSpread(int(-1 << 12), int(0xf000), int(0x0000)),
  new LightSpread(int(+1 << 12), int(0xf000), int(0xf000)),
  new LightSpread(int(-1 << 0),  int(0x00ff), int(0x0000)),
  new LightSpread(int(+1 << 0),  int(0x00ff), int(0x00ff)),
];
const kSunlightLevel = 0xf;

// If the light at a cell changes from `prev` to `next`, what range
// of lights in neighboring cells may need updating? The bounds are
// inclusive on both sides.
//
// These equations are tricky. We do some casework to derive them:
//
//   - If the light value in a cell drops 8 -> 4, then adjacent cells
//     with lights in {4, 5, 6, 7} may also drop. 8 is too big, since
//     an adjacent cell with the same light has a different source.
//     But 3 is too small: we can cast a light of value 3.
//
//   - If the light value increases from 4 -> 8, then adjacent cells
//     with lights in {3, 4, 5, 6} may increase. 7 is too big, since
//     we can't raise the adjacent light to 8.
//
//   - As a special case, a cell in full sunlight can raise a neighbor
//     (the one right below) to full sunlight, so we include it here.
//     `max - (max < kSunlightLevel ? 1 : 0)` is the max we can cast.
//
// If we allow for blocks that filter more than one light level at a
// time, then the lower bounds fail, but the upper bounds still hold.
//
const maxUpdatedNeighborLight = (next: int, prev: int): int => {
  const max = int(Math.max(next, prev));
  return int(max - (max < kSunlightLevel ? 1 : 0) - (next > prev ? 1 : 0));
};
const minUpdatedNeighborLight = (next: int, prev: int): int => {
  const min = int(Math.min(next, prev));
  return int(min - (next > prev ? 1 : 0));
};

class Chunk {
  cx: int;
  cz: int;
  private dirty: boolean = false;
  private ready: boolean = false;
  private neighbors: int = 0;
  private instances: Map<int, Instance>;
  private solid: VoxelMesh | null = null;
  private water: VoxelMesh | null = null;
  private light: LightTexture | null = null;
  private point_lights: Map<int, int>;
  private world: World;
  private voxels: Tensor3;
  private heightmap: Tensor2;
  private equilevels: Int8Array;

  // Cellular automaton lighting. The main trick here is to get lighting to
  // work across multiple chunks. We rely on the fact that the maximum light
  // level is smaller than a chunk's width.
  //
  // When we traverse the voxel graph to propagate lighting values, we always
  // track voxels by their index in a chunk. The index is just a 16-bit int,
  // and we can extract (x, y, z) coordinates from it or compute neighboring
  // indices with simple arithmetic.
  //
  // Stage 1 lighting is chunk-local. It assumes that all neighboring chunks
  // are completely dark and propagates sunlight within this chunk. When we
  // edit blocks in a chunk, we only need to recompute its stage 1 lighting -
  // never any neighbors'. We use an incremental algorithm to compute these
  // values, tracking a list of dirty sources when we edit the chunk. We store
  // stage 1 lights in a dense array.
  //
  // When we update stage 1 lighting, we also keep track of "edges": blocks on
  // the x-z boundary of the chunk that could shine light into neighbors in
  // other chunks. The edge map is sparse: it only includes edge voxels with
  // light values x where 1 < x < kSunlightMax. The vast majority of the edge
  // voxels have light values equal to kSunlightMax, and these are implicit in
  // the heightmap, so we save memory by skipping those.
  //
  // Stage 2 lighting includes neighboring voxels. To compute it for a given
  // chunk, we load the chunk and its neighbors and propagate the neighbors'
  // edge lighting (including the implicit lights implied by the heightmap).
  // We store stage 2 lights sparsely, as a delta on stage 1 lights.
  //
  private stage1_dirty: int[];
  private stage1_edges: Set<int>;
  private stage1_lights: Tensor3;
  private stage2_dirty: boolean = false;
  private stage2_lights: Map<int, int>;

  constructor(cx: int, cz: int, world: World, loader: Loader) {
    this.cx = cx;
    this.cz = cz;
    this.world = world;
    this.instances = new Map();
    this.voxels = new Tensor3(kChunkWidth, kWorldHeight, kChunkWidth);
    this.heightmap = new Tensor2(kChunkWidth, kChunkWidth);
    this.equilevels = new Int8Array(kWorldHeight);
    this.point_lights = new Map();

    this.stage1_dirty = [];
    this.stage1_edges = new Set();
    this.stage1_lights = new Tensor3(kChunkWidth, kWorldHeight, kChunkWidth);
    this.stage2_lights = new Map();

    this.load(loader);

    // Check the invariants we use to optimize getBlock.
    const [sx, sy, sz] = this.voxels.stride;
    assert(sx === (1 << kChunkShiftX));
    assert(sz === (1 << kChunkShiftZ));
    assert(sy === 1);
  }

  dispose(): void {
    this.dropMeshes();
    this.eachNeighbor(x => x.notifyNeighborDisposed());
  }

  getLightLevel(x: int, y: int, z: int): int {
    const xm = int(x & kChunkMask), zm = int(z & kChunkMask);
    const index = int((xm << kChunkShiftX) | y | (zm << kChunkShiftZ));
    const light = this.stage2_lights.get(index);
    const base = light !== undefined ? light : this.stage1_lights.data[index];

    const registry = this.world.registry;
    const block = this.voxels.data[index] as BlockId;
    const mesh = registry.getBlockMesh(block);
    return int(Math.min(base + (mesh ? 1 : 0), kSunlightLevel));
  }

  getBlock(x: int, y: int, z: int): BlockId {
    const xm = int(x & kChunkMask), zm = int(z & kChunkMask);
    const index = (xm << kChunkShiftX) | y | (zm << kChunkShiftZ);
    return this.voxels.data[index] as BlockId;
  }

  setBlock(x: int, y: int, z: int, block: BlockId): void {
    const voxels = this.voxels;
    const xm = int(x & kChunkMask), zm = int(z & kChunkMask);

    const old = voxels.get(xm, y, zm) as BlockId;
    if (old === block) return;
    const index = int((xm << kChunkShiftX) | y | (zm << kChunkShiftZ));
    voxels.data[index] = block;

    this.dirty = true;
    this.stage1_dirty.push(index);
    this.stage2_dirty = true;

    this.updateHeightmap(xm, zm, index, y, 1, block);
    this.equilevels[y] = 0;

    const neighbor = (x: int, y: int, z: int) => {
      const {cx, cz} = this;
      const chunk = this.world.chunks.get(int(x + cx), int(z + cz));
      if (chunk) chunk.dirty = true;
    };
    if (xm === 0) neighbor(-1, 0, 0);
    if (zm === 0) neighbor(0, 0, -1);
    if (xm === kChunkMask) neighbor(1, 0, 0);
    if (zm === kChunkMask) neighbor(0, 0, 1);
    if (xm === 0 && zm === 0) neighbor(-1, 0, -1);
    if (xm === 0 && zm === kChunkMask) neighbor(-1, 0, 1);
    if (xm === kChunkMask && zm === 0) neighbor(1, 0, -1);
    if (xm === kChunkMask && zm === kChunkMask) neighbor(1, 0, 1);
  }

  setColumn(x: int, z: int, start: int, count: int, block: BlockId): void {
    const voxels = this.voxels;
    const xm = int(x & kChunkMask), zm = int(z & kChunkMask);

    assert(voxels.stride[1] === 1);
    const index = voxels.index(xm, start, zm);
    voxels.data.fill(block, index, index + count);

    const light = this.world.registry.light[block];
    if (light > 0) {
      for (let i = 0; i < count; i++) {
        this.stage1_dirty.push(int(index + i));
      }
    }

    this.updateHeightmap(xm, zm, index, start, count, block);
  }

  setPointLight(x: int, y: int, z: int, level: int): void {
    const xm = int(x & kChunkMask), zm = int(z & kChunkMask);
    const index = int((xm << kChunkShiftX) | y | (zm << kChunkShiftZ));
    level > 0 ? this.point_lights.set(index, level)
              : this.point_lights.delete(index);
    this.stage1_dirty.push(index);
    this.stage2_dirty = true;
  }

  hasMesh(): boolean {
    return !!(this.solid || this.water);
  }

  needsRelight(): boolean {
    return this.stage2_dirty && this.ready && this.hasMesh();
  }

  needsRemesh(): boolean {
    return this.dirty && this.ready;
  }

  relightChunk(): void {
    // Called from remeshChunk to set the meshes' light textures, even if
    // !this.needsRelight(). Each step checks a dirty flag so that's okay.
    this.eachNeighbor(x => x.lightingStage1());
    this.lightingStage1();
    this.lightingStage2();
    this.setLightTexture();
  }

  remeshChunk(): void {
    assert(this.needsRemesh());
    this.remeshSprites();
    this.remeshTerrain();
    this.relightChunk();
    this.dirty = false;
  }

  private load(loader: Loader): void {
    const {cx, cz, world} = this;
    const column = world.column;
    const dx = cx << kChunkBits;
    const dz = cz << kChunkBits;
    for (let x = 0; x < kChunkWidth; x++) {
      for (let z = 0; z < kChunkWidth; z++) {
        const first = x + z === 0;
        const ax = int(x + dx), az =  int(z + dz);
        loader(ax, az, column);
        column.fillChunk(ax, az, this, first);
        column.clear();
      }
    }
    column.fillEquilevels(this.equilevels);
    this.lightingInit();

    if (kCheckEquilevels) {
      for (let y = int(0); y < kWorldHeight; y++) {
        if (this.equilevels[y] === 0) continue;
        const base = this.voxels.get(0, y, 0);
        for (let x = int(0); x < kChunkWidth; x++) {
          for (let z = int(0); z < kChunkWidth; z++) {
            assert(this.voxels.get(x, y, z) === base);
          }
        }
      }
    }

    this.eachNeighbor(chunk => {
      chunk.notifyNeighborLoaded();
      this.neighbors++;
    });
    this.dirty = true;
    this.ready = this.checkReady();
  }

  private checkReady(): boolean {
    return this.neighbors === kNeighbors.length;
  }

  private dropMeshes(): void {
    this.dropInstancedMeshes();
    if (this.hasMesh()) {
      this.world.frontier.markDirty(0);
    }
    this.light?.dispose();
    this.solid?.dispose();
    this.water?.dispose();
    this.light = null;
    this.solid = null;
    this.water = null;
    this.dirty = true;
  }

  private dropInstancedMeshes(): void {
    const instances = this.instances;
    for (const mesh of instances.values()) mesh.dispose();
    instances.clear();
  }

  private eachNeighbor(fn: (chunk: Chunk) => void) {
    const {cx, cz} = this;
    const chunks = this.world.chunks;
    for (const {x, z} of kNeighbors) {
      const chunk = chunks.get(int(x + cx), int(z + cz));
      if (chunk) fn(chunk);
    }
  }

  private lightingInit(): void {
    const {heightmap, voxels} = this;
    const opaque = this.world.registry.opaque;
    const lights = this.stage1_lights;
    const dirty = this.stage1_dirty;
    this.stage2_dirty = true;

    // Use for fast bitwise index propagation below.
    assert(lights.stride[0] === kSpread[1].diff);
    assert(lights.stride[1] === kSpread[5].diff);
    assert(lights.stride[2] === kSpread[3].diff);
    assert(heightmap.stride[0] === (kSpread[1].diff >> 8));
    assert(heightmap.stride[1] === (kSpread[3].diff >> 8));

    const data = lights.data;
    data.fill(kSunlightLevel);

    for (let x = int(0); x < kChunkWidth; x++) {
      for (let z = int(0); z < kChunkWidth; z++) {
        const height = heightmap.get(x, z);
        const index = (x << 8) | (z << 12);

        for (let i = 0; i < 4; i++) {
          const spread = kSpread[i];
          if ((index & spread.mask) === spread.test) continue;

          const neighbor_index = int(index + spread.diff);
          const neighbor = heightmap.data[neighbor_index >> 8];
          for (let y = height; y < neighbor; y++) {
            dirty.push(int(neighbor_index + y));
          }
        }

        if (height > 0) {
          const below = int(index + height - 1);
          if (!opaque[voxels.data[below]]) dirty.push(below);
          data.fill(0, index, index + height);
        }
      }
    }
  }

  private lightingStage1(): void {
    if (this.stage1_dirty.length === 0) return;

    const heightmap_data = this.heightmap.data;
    const voxels_data = this.voxels.data;
    const block_light = this.world.registry.light;
    const lights = this.stage1_lights;
    const edges = this.stage1_edges;
    const data = lights.data;

    assert(lights.shape[0] === (1 << 4));
    assert(lights.shape[2] === (1 << 4));
    assert(lights.stride[0] === (1 << 8));
    assert(lights.stride[2] === (1 << 12));

    // Stage 1 lighting operates on "index" values, which are (x, y, z)
    // coordinates represented as indices into our {lights, voxel} Tensor3.
    let prev = this.stage1_dirty;
    let next: int[] = [];

    // Returns true if the given index is on an x-z edge of the chunk.
    const edge = (index: int): boolean => {
      const x_edge = (((index >> 8)  + 1) & 0xf) < 2;
      const z_edge = (((index >> 12) + 1) & 0xf) < 2;
      return x_edge || z_edge;
    };

    // Returns the updated lighting value at the given index. Note that we
    // can never use the `prev` light value in this computation: it can be
    // arbitrarily out-of-date since the chunk contents can change.
    const query = (index: int): int => {
      const from_block = block_light[voxels_data[index]];
      if (from_block < 0) return 0;

      const from_point = this.point_lights.get(index) || 0;
      const base = Math.max(from_block, from_point);

      const height = heightmap_data[index >> 8];
      if ((index & 0xff) >= height) return kSunlightLevel;

      let max_neighbor = base + 1;
      for (const spread of kSpread) {
        if ((index & spread.mask) === spread.test) continue;
        const neighbor_index = int(index + spread.diff);
        const neighbor = data[neighbor_index];
        if (neighbor > max_neighbor) max_neighbor = neighbor;
      }
      return int(max_neighbor - 1);
    };

    // Enqueues new indices that may be affected by the given change.
    const enqueue = (index: int, hi: int, lo: int): void => {
      for (const spread of kSpread) {
        if ((index & spread.mask) === spread.test) continue;
        const neighbor_index = int(index + spread.diff);
        const neighbor = data[neighbor_index];
        if (lo <= neighbor && neighbor <= hi) next.push(neighbor_index);
      }
    };

    while (prev.length > 0) {
      for (const index of prev) {
        const prev = int(data[index]);
        const next = query(index);
        if (next === prev) continue;

        data[index] = next;

        if (edge(index)) {
          // The edge lights map only contains cells on the edge that are not
          // at full sunlight, since the heightmap takes care of the rest.
          const next_in_map = 1 < next && next < kSunlightLevel;
          const prev_in_map = 1 < prev && prev < kSunlightLevel;
          if (next_in_map !== prev_in_map) {
            if (next_in_map) edges.add(index);
            else edges.delete(index);
          }
        }

        const hi = maxUpdatedNeighborLight(next, prev);
        const lo = minUpdatedNeighborLight(next, prev);
        enqueue(index, hi, lo);
      }
      [prev, next] = [next, prev];
      next.length = 0;
    }

    assert(this.stage1_dirty.length === 0);
    this.eachNeighbor(x => x.stage2_dirty = true);
  }

  private lightingStage2(): void {
    if (!this.ready || !this.stage2_dirty) return;

    const getIndex = (x: int, z: int): int => {
      return ((x + 1) | ((z + 1) << 2)) as int;
    };

    const {cx, cz} = this;
    const chunks = this.world.chunks;
    const zone: (Chunk | null)[] = new Array(16).fill(null);
    const zone_lights: (Uint8Array | null)[] = new Array(16).fill(null);
    const zone_voxels: (Uint8Array | null)[] = new Array(16).fill(null);
    for (const {x, z} of kZone) {
      const index = getIndex(x, z);
      const chunk = nonnull(chunks.get(int(x + cx), int(z + cz)));
      zone_lights[index] = chunk.stage1_lights.data;
      zone_voxels[index] = chunk.voxels.data;
      zone[index] = chunk;
    }

    // Stage 1 lighting tracks nodes by "index", where an index can be used
    // to look up a chunk (x, y, z) coordinate in a Tensor3. Stage 2 lighting
    // deals with multiple chunks, so we deal with "locations". The first 16
    // bits of a location are an index; bits 16:18 are a chunk x coordinate,
    // and bits 18:20 are a chunk z coordinate.
    //
    // To keep the cellular automaton as fast as possible, we update stage 1
    // lighting in place. We must undo these changes at the end of this call,
    // so we track a list of (location, previous value) pairs in `deltas` as
    // we make the updates.
    //
    // To avoid needing Theta(n) heap allocations, we flatten `deltas`.
    const deltas: int[] = [];
    const opaque = this.world.registry.opaque;

    // Cells at a light level of i appear in stage[kSunlightLevel - i - 1].
    // Cells at a light level of {0, 1} don't propagate, so we drop them.
    assert(kSunlightLevel > 2);
    const stages: int[][] = Array(kSunlightLevel - 2).fill(null).map(_ => []);
    const stage0 = stages[0];

    for (const {x, z} of kZone) {
      const chunk = nonnull(zone[getIndex(x, z)]);
      const edges = Array.from(chunk.stage1_edges);
      const light = chunk.stage1_lights.data;
      const heightmap = chunk.heightmap.data;

      for (let i = 0; i < 4; i++) {
        const {diff, mask, test} = kSpread[i];
        assert(mask === 0x0f00 || mask === 0xf000);
        const dx = mask === 0x0f00 ? diff >> 8  : 0;
        const dz = mask === 0xf000 ? diff >> 12 : 0;
        const nx = int(x + dx), nz = int(z + dz);
        if (!(-1 <= nx && nx <= 1 && -1 <= nz && nz <= 1)) continue;

        const ni = getIndex(nx, nz);
        const neighbor_union = ni << 16;
        const neighbor_chunk = nonnull(zone[ni]);
        const neighbor_light = neighbor_chunk.stage1_lights.data;

        for (const index of edges) {
          if ((index & mask) !== test) continue;

          const level = light[index] - 1;
          const neighbor_index = index ^ mask;
          const neighbor_level = int(neighbor_light[neighbor_index]);
          if (level <= neighbor_level) continue;

          const neighbor_location = (neighbor_index | neighbor_union) as int;
          neighbor_light[neighbor_index] = level;
          deltas.push(neighbor_location);
          deltas.push(neighbor_level);

          if (level <= 1) continue;
          stages[kSunlightLevel - level - 1].push(neighbor_location);
        }

        let offset = 0;
        const source = test;
        const target = source ^ mask;
        const stride = mask === 0x0f00 ? 0x1000 : 0x0100;
        const neighbor_heightmap = neighbor_chunk.heightmap.data;
        const level = kSunlightLevel - 1;

        for (let j = 0; j < kChunkWidth; j++, offset += stride) {
          const height = heightmap[(source + offset) >> 8];
          const neighbor_height = neighbor_heightmap[(target + offset) >> 8];
          for (let y = height; y < neighbor_height; y++) {
            const neighbor_index = target + offset + y;
            const neighbor_level = int(neighbor_light[neighbor_index]);
            if (level <= neighbor_level) continue;

            const neighbor_location = (neighbor_index | neighbor_union) as int;
            neighbor_light[neighbor_index] = level;
            deltas.push(neighbor_location);
            deltas.push(neighbor_level);

            stage0.push(neighbor_location);
          }
        }
      }
    }

    // Returns the taxicab distance from the location to the center chunk.
    const distance = (location: int): int => {
      const cx = (location >> 16) & 0x3;
      const x  = (location >> 8 ) & 0xf;
      const dx = cx === 0 ? 16 - x : cx === 1 ? 0 : x - 31;

      const cz = (location >> 18) & 0x3;
      const z  = (location >> 12) & 0xf;
      const dz = cz === 0 ? 16 - z : cz === 1 ? 0 : z - 31;

      return int(dx + dz);
    };

    // Returns the given location, shifted by the delta. If the shift is out
    // of bounds any direction, it'll return -1.
    const shift = (location: int, spread: LightSpread): int => {
      const {diff, mask, test} = spread;
      if ((location & mask) !== test) return int(location + diff);
      switch (mask) {
        case 0x00ff: return -1;
        case 0x0f00: {
          const x = int(((location >> 16) & 0x3) + (diff >> 8));
          const z = int(((location >> 18) & 0x3));
          if (!(0 <= x && x <= 2)) return -1;
          return int(((location & 0xffff) ^ mask) | (x << 16) | (z << 18));
        }
        case 0xf000: {
          const x = int(((location >> 16) & 0x3));
          const z = int(((location >> 18) & 0x3) + (diff >> 12));
          if (!(0 <= z && z <= 2)) return -1;
          return int(((location & 0xffff) ^ mask) | (x << 16) | (z << 18));
        }
        default: assert(false);
      }
      return -1;
    };

    for (let level = int(kSunlightLevel - 2); level > 0; level--) {
      const prev = stages[kSunlightLevel - level - 2];
      const next = level > 1 ? stages[kSunlightLevel - level - 1] : null;
      const prev_level = level + 1;

      for (const location of prev) {
        if (distance(location) > level) continue;
        const current_level = zone_lights[location >> 16]![location & 0xffff];
        if (current_level != prev_level) continue;

        // TODO(skishore): If the index is too far from the center given its
        // current light value, don't enqueue its neighbors here.
        for (const spread of kSpread) {
          const neighbor_location = shift(location, spread);
          if (neighbor_location < 0) continue;

          const neighbor_union = neighbor_location >> 16;
          const neighbor_index = neighbor_location & 0xffff;
          const neighbor_light = zone_lights[neighbor_union]!;
          const neighbor_level = int(neighbor_light[neighbor_index]);
          if (level <= neighbor_level) continue;

          if (neighbor_level === 0) {
            const voxels = zone_voxels[neighbor_union]!;
            if (opaque[voxels[neighbor_index]]) continue;
          }

          neighbor_light[neighbor_index] = level;
          deltas.push(neighbor_location);
          deltas.push(neighbor_level);

          if (next === null) continue;
          next.push(neighbor_location);
        }
      }
    }

    assert(getIndex(0, 0) === 5);
    const input = this.stage1_lights.data;
    const output = this.stage2_lights;
    output.clear();

    for (let i = 0; i < deltas.length; i += 2) {
      const location = deltas[i + 0];
      if ((location >> 16) !== 5) continue;
      const index = int(location & 0xffff);
      output.set(index, int(input[index]));
    }
    for (let i = deltas.length - 2; i >= 0; i -= 2) {
      const location = deltas[i + 0];
      zone_lights[location >> 16]![location & 0xffff] = deltas[i + 1];
    }
    this.stage2_dirty = false;
  }

  private notifyNeighborDisposed(): void {
    assert(this.neighbors > 0);
    this.neighbors--;
    const old = this.ready;
    this.ready = this.checkReady();
    if (old && !this.ready) this.dropMeshes();
  }

  private notifyNeighborLoaded(): void {
    assert(this.neighbors < kNeighbors.length);
    this.neighbors++;
    this.ready = this.checkReady();
  }

  private remeshSprites(): void {
    this.dropInstancedMeshes();
    const {equilevels, instances, voxels, world} = this;
    const {registry, renderer} = world;
    const {data, stride} = voxels;

    const bx = this.cx << kChunkBits;
    const bz = this.cz << kChunkBits;

    assert(stride[1] === 1);
    for (let y = int(0); y < kWorldHeight; y++) {
      const block = data[y] as BlockId;
      if (equilevels[y] && !registry.getBlockMesh(block)) continue;
      for (let x = int(0); x < kChunkWidth; x++) {
        for (let z = int(0); z < kChunkWidth; z++) {
          const index = voxels.index(x, y, z);
          const mesh = registry.getBlockMesh(data[index] as BlockId);
          if (!mesh) continue;

          const item = mesh.addInstance();
          item.setPosition(bx + x + 0.5, y, bz + z + 0.5);
          instances.set(index, item);
        }
      }
    }
  }

  private remeshTerrain(): void {
    const {cx, cz, world} = this;
    const {bedrock, buffer, heightmap, equilevels} = world;
    equilevels.set(this.equilevels, 1);
    for (const offset of kNeighborOffsets) {
      const [c, dstPos, srcPos, size] = offset;
      const chunk = world.chunks.get(int(cx + c[0]), int(cz + c[2]));
      const delta = int(dstPos[1] - srcPos[1]);
      assert(delta === 1);
      if (chunk) {
        this.copyHeightmap(heightmap, dstPos, chunk.heightmap, srcPos, size);
        this.copyVoxels(buffer, dstPos, chunk.voxels, srcPos, size);
      } else {
        this.zeroHeightmap(heightmap, dstPos, size, delta);
        this.zeroVoxels(buffer, dstPos, size);
      }
      if (chunk !== this) {
        this.copyEquilevels(equilevels, chunk, srcPos, size, delta);
      }
    }

    if (kCheckEquilevels) {
      for (let y = int(0); y < buffer.shape[1]; y++) {
        if (equilevels[y] === 0) continue;
        const base = buffer.get(1, y, 1);
        for (let x = int(0); x < buffer.shape[0]; x++) {
          for (let z = int(0); z < buffer.shape[2]; z++) {
            if ((x !== 0 && x !== buffer.shape[0] - 1) ||
                (z !== 0 && z !== buffer.shape[2] - 1)) {
              assert(buffer.get(x, y, z) === base);
            }
          }
        }
      }
    }

    const x = cx << kChunkBits, z = cz << kChunkBits;
    const meshed = world.mesher.meshChunk(
        buffer, heightmap, equilevels, this.solid, this.water);
    const [solid, water] = meshed;
    solid?.setPosition(x, 0, z);
    water?.setPosition(x, 0, z);
    this.solid = solid;
    this.water = water;

    this.dropMeshes();
  }

  private setLightTexture(): void {
    if (!this.hasMesh()) return;

    // TODO(skishore): Share a texture between the two meshes.
    const saved = new Map();
    const {stage1_lights, stage2_lights} = this;
    const lights = stage1_lights.data;
    for (const [index, value] of stage2_lights.entries()) {
      saved.set(index, lights[index]);
      lights[index] = value;
    }

    this.light?.dispose();
    this.light = this.world.renderer.addLightTexture(lights);
    this.solid?.setLight(this.light);
    this.water?.setLight(this.light);

    for (const [index, instance] of this.instances.entries()) {
      const level = int(Math.min(lights[index] + 1, kSunlightLevel));
      instance.setLight(lighting(level));
    }

    for (const [index, value] of saved.entries()) {
      lights[index] = value;
    }
  }

  private updateHeightmap(xm: int, zm: int, index: int,
                          start: int, count: int, block: BlockId): void {
    const end = start + count;
    const offset = this.heightmap.index(xm, zm);
    const height = this.heightmap.data[offset];
    const voxels = this.voxels;
    assert(voxels.stride[1] === 1);

    if (block === kEmptyBlock && start < height && height <= end) {
      let i = 0;
      for (; i < start; i++) {
        if (voxels.data[index - i - 1] !== kEmptyBlock) break;
      }
      this.heightmap.data[offset] = start - i;
    } else if (block !== kEmptyBlock && height <= end) {
      this.heightmap.data[offset] = end;
    }
  }

  private copyEquilevels(dst: Int8Array, chunk: Chunk | null,
                         srcPos: Point, size: Point, delta: int): void {
    assert(this.voxels.stride[1] === 1);
    const data = this.voxels.data;

    if (chunk === null) {
      for (let i = 0; i < kWorldHeight; i++) {
        if (dst[i + delta] === 0) continue;
        if (data[i] !== kEmptyBlock) dst[i + delta] = 0;
      }
      return;
    }

    assert(chunk.voxels.stride[1] === 1);
    assert(size[0] === 1 || size[2] === 1);
    const stride = chunk.voxels.stride[size[0] === 1 ? 2 : 0];
    const index = chunk.voxels.index(srcPos[0], srcPos[1], srcPos[2]);
    const limit = stride * (size[0] === 1 ? size[2] : size[0]);

    const chunk_equilevels = chunk.equilevels;
    const chunk_data = chunk.voxels.data;

    for (let i = 0; i < kWorldHeight; i++) {
      if (dst[i + delta] === 0) continue;
      const base = data[i];
      if (chunk_equilevels[i] === 1 && chunk_data[i] === base) continue;
      for (let offset = 0; offset < limit; offset += stride) {
        if (chunk_data[index + offset + i] === base) continue;
        dst[i + delta] = 0;
        break;
      }
    }
  }

  private copyHeightmap(dst: Tensor2, dstPos: Point,
                        src: Tensor2, srcPos: Point, size: Point): void {
    const ni = size[0], nk = size[2];
    const di = dstPos[0], dk = dstPos[2];
    const si = srcPos[0], sk = srcPos[2];
    const offset = dstPos[1] - srcPos[1];

    for (let i = 0; i < ni; i++) {
      for (let k = 0; k < nk; k++) {
        const sindex = src.index(int(si + i), int(sk + k));
        const dindex = dst.index(int(di + i), int(dk + k));
        dst.data[dindex] = src.data[sindex] + offset;
      }
    }
  }

  private copyVoxels(dst: Tensor3, dstPos: Point,
                     src: Tensor3, srcPos: Point, size: Point): void {
    const [ni, nj, nk] = size;
    const [di, dj, dk] = dstPos;
    const [si, sj, sk] = srcPos;
    assert(dst.stride[1] === 1);
    assert(src.stride[1] === 1);

    for (let i = 0; i < ni; i++) {
      for (let k = 0; k < nk; k++) {
        const sindex = src.index(int(si + i), sj, int(sk + k));
        const dindex = dst.index(int(di + i), dj, int(dk + k));
        dst.data.set(src.data.subarray(sindex, sindex + nj), dindex);
      }
    }
  }

  private zeroHeightmap(
      dst: Tensor2, dstPos: Point, size: Point, delta: int): void {
    const ni = size[0], nk = size[2];
    const di = dstPos[0], dk = dstPos[2];

    for (let i = 0; i < ni; i++) {
      for (let k = 0; k < nk; k++) {
        dst.set(int(di + i), int(dk + k), delta);
      }
    }
  }

  private zeroVoxels(dst: Tensor3, dstPos: Point, size: Point): void {
    const [ni, nj, nk] = size;
    const [di, dj, dk] = dstPos;
    const dsj = dst.stride[1];
    for (let i = 0; i < ni; i++) {
      for (let k = 0; k < nk; k++) {
        // Unroll along the y-axis, since it's the longest chunk dimension.
        let dindex = dst.index(int(di + i), dj, int(dk + k));
        for (let j = 0; j < nj; j++, dindex += dsj) {
          dst.data[dindex] = kEmptyBlock;
        }
      }
    }
  }
};

//////////////////////////////////////////////////////////////////////////////

const kMultiMeshBits = int(2);
const kMultiMeshSide = int(1 << kMultiMeshBits);
const kMultiMeshArea = int(kMultiMeshSide * kMultiMeshSide);
const kLODSingleMask = int((1 << 4) - 1);

class LODMultiMesh {
  solid: VoxelMesh | null;
  water: VoxelMesh | null;
  meshed: boolean[];

  private mask: Int32Array;
  private visible: int = 0;
  private enabled: boolean[];

  constructor() {
    this.solid = null;
    this.water = null;
    this.meshed = new Array(kMultiMeshArea).fill(false);
    this.enabled = new Array(kMultiMeshArea).fill(false);
    this.mask = new Int32Array(2);
    this.mask[0] = this.mask[1] = -1;
  }

  disable(index: int): void {
    if (!this.enabled[index]) return;
    this.setMask(index, kLODSingleMask);
    this.enabled[index] = false;
    if (this.enabled.some(x => x)) return;

    for (let i = 0; i < this.meshed.length; i++) this.meshed[i] = false;
    this.solid?.dispose();
    this.water?.dispose();
    this.solid = null;
    this.water = null;
    this.mask[0] = this.mask[1] = -1;
  }

  index(chunk: FrontierChunk): int {
    const mask = kMultiMeshSide - 1;
    return int(((chunk.cz & mask) << kMultiMeshBits) | (chunk.cx & mask));
  }

  show(index: int, mask: int): void {
    assert(this.meshed[index]);
    this.setMask(index, mask);
    this.enabled[index] = true;
  }

  private setMask(index: int, mask: int): void {
    const mask_index = index >> 3;
    const mask_shift = (index & 7) * 4;
    this.mask[mask_index] &= ~(kLODSingleMask << mask_shift);
    this.mask[mask_index] |= mask << mask_shift;

    const shown = (this.mask[0] & this.mask[1]) !== -1;
    this.solid?.show(this.mask, shown);
    this.water?.show(this.mask, shown);
  }
};

class FrontierChunk {
  cx: int;
  cz: int;
  index: int;
  level: int;
  mesh: LODMultiMesh;
  frontier: Frontier;

  constructor(cx: int, cz: int, level: int,
              mesh: LODMultiMesh, frontier: Frontier) {
    this.cx = cx;
    this.cz = cz;
    this.level = level;
    this.mesh = mesh;
    this.frontier = frontier;
    this.index = mesh.index(this);
  }

  dispose() {
    if (this.hasMesh()) {
      this.frontier.markDirty(int(this.level + 1));
    }
    this.mesh.disable(this.index);
  }

  hasMesh() {
    return this.mesh.meshed[this.index];
  }
};

class Frontier {
  private world: World;
  private dirty: boolean[];
  private levels: Circle<FrontierChunk>[];
  private meshes: Map<int, LODMultiMesh>;
  private solid_heightmap: Uint32Array;
  private water_heightmap: Uint32Array;
  private side: int;

  constructor(world: World) {
    this.world = world;
    this.meshes = new Map();

    this.dirty = [];
    this.levels = [];
    let radius = (kChunkRadius | 0) + 0.5;
    for (let i = 0; i < kFrontierLevels; i++) {
      radius = (radius + kFrontierRadius) / 2;
      this.levels.push(new Circle(radius));
      this.dirty.push(true);
    }

    assert(kChunkWidth % kFrontierLOD === 0);
    const side = int(kChunkWidth / kFrontierLOD);
    const size = int(2 * (side + 2) * (side + 2));
    this.solid_heightmap = new Uint32Array(size);
    this.water_heightmap = new Uint32Array(size);
    this.side = side;
  }

  center(cx: int, cz: int) {
    for (const level of this.levels) {
      cx = int(cx >> 1);
      cz = int(cz >> 1);
      level.center(cx, cz);
    }
  }

  markDirty(level: int) {
    if (level < this.dirty.length) this.dirty[level] = true;
  }

  remeshFrontier() {
    for (let i = int(0); i < kFrontierLevels; i++) {
      this.computeLODAtLevel(i);
    }
  }

  private computeLODAtLevel(l: int) {
    if (!this.dirty[l]) return;
    const world = this.world;
    const level = this.levels[l];

    const meshed = (dx: int, dz: int): boolean => {
      if (l > 0) {
        const chunk = this.levels[l - 1].get(dx, dz);
        return chunk !== null && chunk.hasMesh();
      } else {
        const chunk = world.chunks.get(dx, dz);
        return chunk !== null && chunk.hasMesh();
      }
    };

    let counter = 0;
    let skipped = false;
    level.each((cx: int, cz: int): boolean => {
      let mask = int(0);
      for (let i = 0; i < 4; i++) {
        const dx = int((cx << 1) + (i & 1 ? 1 : 0));
        const dz = int((cz << 1) + (i & 2 ? 1 : 0));
        if (meshed(dx, dz)) mask = int(mask | (1 << i));
      }

      const shown = mask !== 15;
      const extra = counter < kNumLODChunksToMeshPerFrame;
      const create = shown && (extra || mask !== 0);
      if (shown && !create) skipped = true;

      const existing = level.get(cx, cz);
      if (!existing && !create) return false;

      const lod = (() => {
        if (existing) return existing;
        const created = this.createFrontierChunk(cx, cz, l);
        level.set(cx, cz, created);
        return created;
      })();

      if (shown && !lod.hasMesh()) {
        this.createLODMeshes(lod);
        this.markDirty(int(l + 1));
        counter++;
      }
      lod.mesh.show(lod.mesh.index(lod), mask);
      return false;
    });
    this.dirty[l] = skipped;
  }

  private createLODMeshes(chunk: FrontierChunk): void {
    const {side, world} = this;
    const {cx, cz, level, mesh} = chunk;
    const {bedrock, column, loadFrontier, registry} = world;
    const {solid_heightmap, water_heightmap} = this;
    if (!loadFrontier) return;

    assert(kFrontierLOD % 2 === 0);
    assert(registry.solid[bedrock]);
    const lshift = kChunkBits + level;
    const lod = int(kFrontierLOD << level);
    const x = (2 * cx + 1) << lshift;
    const z = (2 * cz + 1) << lshift;

    // The (x, z) position of the center of the multimesh for this mesh.
    const multi = kMultiMeshSide;
    const mx = (2 * (cx & ~(multi - 1)) + multi) << lshift;
    const mz = (2 * (cz & ~(multi - 1)) + multi) << lshift;

    for (let k = 0; k < 4; k++) {
      const dx = (k & 1 ? 0 : -1 << lshift);
      const dz = (k & 2 ? 0 : -1 << lshift);
      const ax = x + dx + lod / 2;
      const az = z + dz + lod / 2;

      for (let i = 0; i < side; i++) {
        for (let j = 0; j < side; j++) {
          loadFrontier(int(ax + i * lod), int(az + j * lod), column);
          const offset = 2 * ((i + 1) + (j + 1) * (side + 2));

          const size = column.getSize();
          const last_block = column.getNthBlock(int(size - 1), bedrock);
          const last_level = column.getNthLevel(int(size - 1));

          if (registry.solid[last_block]) {
            solid_heightmap[offset + 0] = last_block;
            solid_heightmap[offset + 1] = last_level;
            water_heightmap[offset + 0] = 0;
            water_heightmap[offset + 1] = 0;
          } else {
            water_heightmap[offset + 0] = last_block;
            water_heightmap[offset + 1] = last_level;

            for (let i = size; i > 0; i--) {
              const block = column.getNthBlock(int(i - 2), bedrock);
              const level = column.getNthLevel(int(i - 2));
              if (!registry.solid[block]) continue;
              solid_heightmap[offset + 0] = block;
              solid_heightmap[offset + 1] = level;
              break;
            }
          }
          column.clear();
        }
      }

      const n = int(side + 2);
      const px = int(x + dx - mx - lod);
      const pz = int(z + dz - mz - lod);
      const mask = int(k + 4 * mesh.index(chunk));
      mesh.solid = this.world.mesher.meshFrontier(
          solid_heightmap, mask, px, pz, n, n, lod, mesh.solid, true);
      mesh.water = this.world.mesher.meshFrontier(
          water_heightmap, mask, px, pz, n, n, lod, mesh.water, false);
    }

    mesh.solid?.setPosition(mx, 0, mz);
    mesh.water?.setPosition(mx, 0, mz);
    mesh.meshed[mesh.index(chunk)] = true;
  }

  private createFrontierChunk(cx: int, cz: int, level: int): FrontierChunk {
    const bits = kMultiMeshBits;
    const mesh = this.getOrCreateMultiMesh(
        int(cx >> bits), int(cz >> bits), level);
    const result = new FrontierChunk(cx, cz, level, mesh, this);
    // A FrontierChunk's mesh is just a fragment of data in its LODMultiMesh.
    // That means that we may already have a mesh when we construct the chunk,
    // if we previously disposed it without discarding data in the multi-mesh.
    // We count this case as meshing a chunk and mark l + 1 dirty.
    if (result.hasMesh()) this.markDirty(int(level + 1));
    return result;
  }

  private getOrCreateMultiMesh(cx: int, cz: int, level: int): LODMultiMesh {
    const shift = 12;
    const mask = (1 << shift) - 1;
    const base = ((cz & mask) << shift) | (cx & mask);
    const key = int(base * kFrontierLevels + level);

    const result = this.meshes.get(key);
    if (result) return result;
    const created = new LODMultiMesh();
    this.meshes.set(key, created);
    return created;
  }
};

//////////////////////////////////////////////////////////////////////////////

class World {
  chunks: Circle<Chunk>;
  column: Column;
  renderer: Renderer;
  registry: Registry;
  frontier: Frontier;
  mesher: TerrainMesher;
  bedrock: BlockId;
  loadChunk: Loader | null;
  loadFrontier: Loader | null;
  buffer: Tensor3;
  heightmap: Tensor2;
  equilevels: Int8Array;

  constructor(registry: Registry, renderer: Renderer) {
    const radius = (kChunkRadius | 0) + 0.5;
    this.chunks = new Circle(radius);
    this.column = new Column();
    this.renderer = renderer;
    this.registry = registry;
    this.frontier = new Frontier(this);
    this.mesher = new TerrainMesher(registry, renderer);
    this.loadChunk = null;
    this.loadFrontier = null;
    this.bedrock = kEmptyBlock;

    // Add a one-block-wide plane of extra space on each side of our voxels,
    // so that we can include adjacent chunks and use their contents for AO.
    //
    // We add a two-block-wide plane below our voxel data, so that we also
    // have room for a plane of bedrock blocks below this chunk (in case we
    // dig all the way to y = 0).
    const w = int(kChunkWidth + 2);
    const h = int(kWorldHeight + 2);
    this.buffer = new Tensor3(w, h, w);
    this.heightmap = new Tensor2(w, w);
    this.equilevels = new Int8Array(h);
    this.equilevels[0] = this.equilevels[h - 1] = 1;
  }

  getLight(x: int, y: int, z: int): number {
    return lighting(this.getLightLevel(x, y, z));
  }

  getBlock(x: int, y: int, z: int): BlockId {
    if (y < 0) return this.bedrock;
    if (y >= kWorldHeight) return kEmptyBlock;
    const cx = int(x >> kChunkBits), cz = int(z >> kChunkBits);
    const chunk = this.chunks.get(cx, cz);
    return chunk ? chunk.getBlock(x, y, z) : kUnknownBlock;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    if (!(0 <= y && y < kWorldHeight)) return;
    const cx = int(x >> kChunkBits), cz = int(z >> kChunkBits);
    const chunk = this.chunks.get(cx, cz);
    chunk?.setBlock(x, y, z, block);
  }

  setLoader(bedrock: BlockId, loadChunk: Loader, loadFrontier?: Loader) {
    this.bedrock = bedrock;
    this.loadChunk = loadChunk;
    this.loadFrontier = loadFrontier || loadChunk;

    const buffer = this.buffer;
    for (let x = int(0); x < buffer.shape[0]; x++) {
      for (let z = int(0); z < buffer.shape[2]; z++) {
        buffer.set(x, 0, z, bedrock);
      }
    }
  }

  setPointLight(x: int, y: int, z: int, level: int): void {
    if (!(0 <= y && y < kWorldHeight)) return;
    const cx = int(x >> kChunkBits), cz = int(z >> kChunkBits);
    const chunk = this.chunks.get(cx, cz);
    // We can't support a block light of kSunlightLevel until we have separate
    // channels for block light and sunlight.
    chunk?.setPointLight(x, y, z, int(Math.min(level, kSunlightLevel - 1)));
  }

  recenter(ix: int, iz: int) {
    const {chunks, frontier, loadChunk} = this;
    const cx = int(ix >> kChunkBits);
    const cz = int(iz >> kChunkBits);
    chunks.center(cx, cz);
    frontier.center(cx, cz);

    if (!loadChunk) return;

    let loaded = 0;
    chunks.each((cx: int, cz: int): boolean => {
      const existing = chunks.get(cx, cz);
      if (existing) return false;
      const chunk = new Chunk(cx, cz, this, loadChunk);
      chunks.set(cx, cz, chunk);
      return (++loaded) === kNumChunksToLoadPerFrame;
    });
  }

  remesh() {
    const {chunks, frontier} = this;
    let lit = 0, meshed = 0, total = 0;
    chunks.each((cx: int, cz: int): boolean => {
      total++;
      const canRelight = lit < kNumChunksToLightPerFrame;
      const canRemesh = total <= 9 || meshed < kNumChunksToMeshPerFrame;
      if (!(canRelight || canRemesh)) return true;

      const chunk = chunks.get(cx, cz);
      if (!chunk) return false;

      if (canRemesh && chunk.needsRemesh()) {
        if (!chunk.hasMesh()) frontier.markDirty(0);
        chunk.remeshChunk();
        meshed++;
      } else if (canRelight && chunk.needsRelight()) {
        chunk.relightChunk();
        lit++;
      }
      return false;
    });
    frontier.remeshFrontier();
  }

  private distance(cx: int, cz: int, x: number, z: number) {
    const half = kChunkWidth / 2;
    const dx = (cx << kChunkBits) + half - x;
    const dy = (cz << kChunkBits) + half - z;
    return dx * dx + dy * dy;
  }

  private getLightLevel(x: int, y: int, z: int): int {
    if (y < 0) return 0;
    if (y >= kWorldHeight) return kSunlightLevel;
    const cx = int(x >> kChunkBits), cz = int(z >> kChunkBits);
    const chunk = this.chunks.get(cx, cz);
    return chunk ? chunk.getLightLevel(x, y, z) : kSunlightLevel;
  }
};

//////////////////////////////////////////////////////////////////////////////

const kTmpPos     = Vec3.create();
const kTmpMin     = Vec3.create();
const kTmpMax     = Vec3.create();
const kTmpDelta   = Vec3.create();
const kTmpImpacts = Vec3.create();

const kMinZLowerBound = 0.001;
const kMinZUpperBound = 0.1;

const lighting = (x: int): number => Math.pow(0.8, kSunlightLevel - x);

class Env {
  entities: EntityComponentSystem;
  registry: Registry;
  renderer: Renderer;
  private helper: WasmHelper;
  private cameraColor: Color;
  private cameraMaterial: MaybeMaterialId;
  private container: Container;
  private highlight: HighlightMesh;
  private highlightSide: int = -1;
  private highlightPosition: Vec3;
  private timing: Timing;
  private frame: number = 0;
  private world: World;

  constructor(id: string) {
    this.container = new Container(id);
    this.entities = new EntityComponentSystem();
    this.renderer = new Renderer(this.container.canvas);

    this.helper = nonnull(helper);
    this.helper.renderer = this.renderer;
    this.helper.initializeWorld(kChunkRadius);

    this.registry = new Registry(this.helper, this.renderer);
    this.world = new World(this.registry, this.renderer);
    this.highlight = this.renderer.addHighlightMesh();
    this.highlightPosition = Vec3.create();

    this.cameraColor = kWhite.slice() as Color;
    this.cameraMaterial = kNoMaterial;

    const remesh = this.remesh.bind(this);
    const render = this.render.bind(this);
    const update = this.update.bind(this);
    this.timing = new Timing(remesh, render, update);
  }

  getBlock(x: int, y: int, z: int): BlockId {
    return this.helper.getBlock(x, y, z);
  }

  getLight(x: int, y: int, z: int): number {
    return this.world.getLight(x, y, z);
  }

  getMutableInputs(): Record<Input, boolean> {
    return this.container.inputs;
  }

  getTargetedBlock(): Vec3 | null {
    return this.highlightSide < 0 ? null : this.highlightPosition;
  }

  getTargetedBlockSide(): int {
    return this.highlightSide;
  }

  setBlock(x: int, y: int, z: int, block: BlockId): void {
    this.helper.setBlock(x, y, z, block);
    this.world.setBlock(x, y, z, block);
  }

  setCameraTarget(x: number, y: number, z: number): void {
    this.renderer.camera.setTarget(x, y, z);
    this.setSafeZoomDistance();
  }

  setLoader(bedrock: BlockId, loadChunk: Loader, loadFrontier?: Loader) {
    this.world.setLoader(bedrock, loadChunk, loadFrontier);
  }

  setPointLight(x: int, y: int, z: int, level: int): void {
    this.world.setPointLight(x, y, z, level);
  }

  recenter(x: number, y: number, z: number): void {
    const ix = int(Math.round(x)), iz = int(Math.round(z));
    this.helper.recenterWorld(ix, iz);
    this.world.recenter(ix, iz);
  }

  refresh(): void {
    const saved = this.container.inputs.pointer;
    this.container.inputs.pointer = true;
    this.update(0);
    this.render(0);
    this.container.inputs.pointer = saved;
  }

  remesh(): void {
    this.helper.remeshWorld();
    this.world.remesh();
  }

  render(dt: number): void {
    if (!this.container.inputs.pointer) return;

    const old_frame = this.frame;
    this.frame = old_frame + 60 * dt;
    if (this.frame > 0xffff) this.frame -= 0xffff;
    const pos = this.frame / 256;
    const rad = 2 * Math.PI * pos;
    const move = 0.25 * (Math.cos(rad) * 0.5 + pos);
    const wave = 0.05 * (Math.sin(rad) + 3);

    const camera = this.renderer.camera;
    const deltas = this.container.deltas;
    camera.applyInputs(deltas.x, deltas.y, deltas.scroll);
    deltas.x = deltas.y = deltas.scroll = 0;

    this.entities.render(dt);
    this.updateHighlightMesh();
    this.updateOverlayColor(wave);
    const sparkle = int(old_frame) !== int(this.frame);
    const renderer_stats = this.renderer.render(move, wave, sparkle);

    const timing = this.timing;
    if (timing.renderPerf.frame() % 20 !== 0) return;
    const stats = `Update: ${this.formatStat(timing.updatePerf)}\r\n` +
                  `Remesh: ${this.formatStat(timing.remeshPerf)}\r\n` +
                  `Render: ${this.formatStat(timing.renderPerf)}\r\n` +
                  renderer_stats;
    this.container.displayStats(stats);
  }

  update(dt: number): void {
    if (!this.container.inputs.pointer) return;
    this.entities.update(dt);
  }

  private formatStat(perf: Performance): string {
    const format = (x: number) => (x / 1000).toFixed(2);
    return `${format(perf.mean())}ms / ${format(perf.max())}ms`;
  }

  private getRenderBlock(x: int, y: int, z: int): BlockId {
    const result = this.world.getBlock(x, y, z);
    if (result === kEmptyBlock || result === kUnknownBlock ||
        this.registry.getBlockFaceMaterial(result, 3) === kNoMaterial) {
      return kEmptyBlock;
    }
    return result;
  }

  private setSafeZoomDistance(): void {
    const camera = this.renderer.camera;
    const {direction, target, zoom} = camera;
    const [x, y, z] = target;

    const check = (x: int, y: int, z: int) => {
      const block = this.world.getBlock(x, y, z);
      return !this.registry.opaque[block];
    };

    const shift_target = (delta: Vec3, bump: number) => {
      const buffer = kMinZUpperBound;
      Vec3.set(kTmpMin, x - buffer, y - buffer + bump, z - buffer);
      Vec3.set(kTmpMax, x + buffer, y + buffer + bump, z + buffer);
      sweep(kTmpMin, kTmpMax, kTmpDelta, kTmpImpacts, check, true);

      Vec3.add(kTmpDelta, kTmpMin, kTmpMax);
      Vec3.scale(kTmpDelta, kTmpDelta, 0.5);
      Vec3.sub(kTmpDelta, kTmpDelta, target);
      return Vec3.length(kTmpDelta);
    };

    const safe_zoom_at = (bump: number) => {
      Vec3.scale(kTmpDelta, direction, -zoom);
      return shift_target(kTmpDelta, bump);
    };

    const max_bump = () => {
      Vec3.set(kTmpDelta, 0, 0.5, 0);
      return shift_target(kTmpDelta, 0);
    };

    let limit = 1;
    let best_bump = -1;
    let best_zoom = -1;
    const step_size = 1 / 64;

    for (let i = 0; i < limit; i++) {
      const bump_at = i * step_size;
      const zoom_at = safe_zoom_at(bump_at) - bump_at;
      if (zoom_at < best_zoom) continue;
      best_bump = bump_at;
      best_zoom = zoom_at;
      if (zoom_at > zoom - bump_at - step_size) break;
      if (i === 0) limit = Math.floor(max_bump() / step_size);
    }
    camera.setSafeZoomDistance(best_bump, best_zoom);
  }

  private updateHighlightMesh(): void {
    const camera = this.renderer.camera;
    const {direction, target, zoom} = camera;

    let move = false;
    this.highlight.mask = int((1 << 6) - 1);
    this.highlightSide = -1;

    const check = (x: int, y: int, z: int) => {
      const block = this.world.getBlock(x, y, z);
      if (!this.registry.solid[block]) return true;

      let mask = 0;
      const pos = kTmpPos;
      Vec3.set(pos, x, y, z);
      for (let d = 0; d < 3; d++) {
        pos[d] += 1;
        const b0 = this.world.getBlock(int(pos[0]), int(pos[1]), int(pos[2]));
        if (this.registry.opaque[b0]) mask |= (1 << (2 * d + 0));
        pos[d] -= 2;
        const b1 = this.world.getBlock(int(pos[0]), int(pos[1]), int(pos[2]));
        if (this.registry.opaque[b1]) mask |= (1 << (2 * d + 1));
        pos[d] += 1;
      }
      move = pos[0] !== this.highlightPosition[0] ||
             pos[1] !== this.highlightPosition[1] ||
             pos[2] !== this.highlightPosition[2];
      this.highlight.mask = int(mask);
      Vec3.copy(this.highlightPosition, pos);
      return false;
    };

    const buffer = 1 / kSweepResolution;
    const x = Math.floor(target[0] * kSweepResolution) / kSweepResolution;
    const y = Math.floor(target[1] * kSweepResolution) / kSweepResolution;
    const z = Math.floor(target[2] * kSweepResolution) / kSweepResolution;

    Vec3.set(kTmpMin, x - buffer, y - buffer, z - buffer);
    Vec3.set(kTmpMax, x + buffer, y + buffer, z + buffer);
    Vec3.scale(kTmpDelta, direction, 10);
    sweep(kTmpMin, kTmpMax, kTmpDelta, kTmpImpacts, check, true);

    for (let i = 0; i < 3; i++) {
      const impact = kTmpImpacts[i];
      if (impact === 0) continue;
      this.highlightSide = int(2 * i + (impact < 0 ? 0 : 1));
      break;
    }

    if (move) {
      const pos = this.highlightPosition;
      this.highlight.setPosition(pos[0], pos[1], pos[2]);
    }
  }

  private updateOverlayColor(wave: number): void {
    const [x, y, z] = this.renderer.camera.position;
    const xi = int(Math.floor(x));
    const yi = int(Math.floor(y));
    const zi = int(Math.floor(z));
    let boundary = 1;

    // We should only apply wave if the block above a liquid is an air block.
    const new_block = ((): BlockId => {
      const below = this.getRenderBlock(xi, yi, zi);
      if (below === kEmptyBlock) return below;
      const above = this.getRenderBlock(xi, int(yi + 1), zi);
      if (above !== kEmptyBlock) return below;
      const delta = y + wave - yi - 1;
      boundary = Math.abs(delta);
      return delta > 0 ? kEmptyBlock : below;
    })();

    const new_material = ((): MaybeMaterialId => {
      if (new_block === kEmptyBlock) return kNoMaterial;
      return this.registry.getBlockFaceMaterial(new_block, 3);
    })();

    const old_material = this.cameraMaterial;
    this.cameraMaterial = new_material;

    const max = kMinZUpperBound;
    const min = kMinZLowerBound;
    const minZ = Math.max(Math.min(boundary / 2, max), min);
    this.renderer.camera.setMinZ(minZ);

    if (new_material === kNoMaterial) {
      const changed = new_material !== old_material;
      if (changed) this.renderer.setOverlayColor(kWhite);
      return;
    }

    const color = this.registry.getMaterialData(new_material).texture.color;
    const light = this.world.getLight(xi, yi, zi);
    const saved = this.cameraColor;
    saved[0] = color[0] * light;
    saved[1] = color[1] * light;
    saved[2] = color[2] * light;
    saved[3] = color[3];
    this.renderer.setOverlayColor(saved);
  }
};

//////////////////////////////////////////////////////////////////////////////

type WasmCharPtr   = int & {__cpp_type__: 'char*'};
type WasmNoise2D   = int & {__cpp_type__: 'voxels::Noise2D*'};
type WasmHeightmap = int & {__cpp_type__: 'voxels::Heightmap*'};

interface WasmModule {
  HEAP8:   Int8Array,
  HEAP16:  Int16Array,
  HEAP32:  Int32Array,
  HEAPF32: Float32Array,
  HEAPF64: Float32Array,
  HEAPU8:  Uint8Array,
  HEAPU16: Uint16Array,
  HEAPU32: Uint32Array,
  asm: {
    free:   (WasmCharPtr: int) => void,
    malloc: (bytes: int) => WasmCharPtr,

    noise: (x: int, y: int) => number,
    heightmap: (x: int, y: int) => WasmHeightmap,
    loadChunk: (cx: int, cz: int) => WasmCharPtr,

    createNoise2D: (seed: int) => WasmNoise2D,
    queryNoise2D: (noise: WasmNoise2D, x: number, y: number) => number,

    initializeWorld: (radius: number) => void,
    recenterWorld: (x: int, z: int) => void,
    remeshWorld: () => void,

    getBlock: (x: int, y: int, z: int) => BlockId,
    setBlock: (x: int, y: int, z: int, block: BlockId) => void,

    registerBlock: any,
    registerMaterial: any,
  },
};

class WasmHandle<T> {
  entries: (T | null)[];
  freeList: int[];

  constructor() {
    this.entries = [];
    this.freeList = [];
  }

  allocate(value: T): int {
    const free = this.freeList.pop();
    if (free !== undefined) {
      this.entries[free] = value;
      return free;
    }
    const result = int(this.entries.length);
    this.entries.push(value);
    return result;
  }

  free(index: int): T {
    const value = nonnull(this.entries[index]);
    this.entries[index] = null;
    this.freeList.push(index);
    return value;
  }

  get(index: int): T {
    return nonnull(this.entries[index]);
  }
};

class WasmHelper {
  module: WasmModule;

  // Bindings to call C++ from JavaScript.

  initializeWorld: (radius: number) => void;
  recenterWorld: (x: int, z: int) => void;
  remeshWorld: () => void;

  getBlock: (x: int, y: int, z: int) => BlockId;
  setBlock: (x: int, y: int, z: int, block: BlockId) => void;

  // Bindings to call JavaScript from C++.

  meshes: WasmHandle<VoxelMesh>;
  renderer: Renderer | null = null;

  constructor(module: WasmModule) {
    this.module = module;
    this.initializeWorld = module.asm.initializeWorld;
    this.recenterWorld = module.asm.recenterWorld;
    this.remeshWorld = module.asm.remeshWorld;
    this.getBlock = module.asm.getBlock;
    this.setBlock = module.asm.setBlock;
    this.meshes = new WasmHandle();
  }
};

let loaded = false;
let helper: WasmHelper | null = null;
let on_start_callbacks: (() => void)[] = [];

const checkReady = () => {
  if (!(loaded && helper)) return;
  on_start_callbacks.forEach(x => x());
};

const js_AddVoxelMesh = (data: int, size: int, phase: int) => {
  const h = nonnull(helper);
  const r = nonnull(h.renderer);
  const offset = data >> 2;
  const buffer = h.module.HEAP32.slice(offset, offset + size);
  const geo = new Geometry(buffer, int(size / Geometry.StrideInInt32));
  const result = h.meshes.allocate(r.addVoxelMesh(geo, phase));
  return result;
};

const js_FreeVoxelMesh = (handle: int): void => {
  nonnull(helper).meshes.free(handle).dispose();
};

const js_SetVoxelMeshGeometry = (handle: int, data: int, size: int): void => {
  const h = nonnull(helper);
  const offset = data >> 2;
  const buffer = h.module.HEAP32.slice(offset, offset + size);
  const geo = new Geometry(buffer, int(size / Geometry.StrideInInt32));
  h.meshes.get(handle).setGeometry(geo);
};

const js_SetVoxelMeshPosition = (handle: int, x: int, y: int, z: int): void => {
  nonnull(helper).meshes.get(handle).setPosition(x, y, z);
};

const init = (fn: () => void) => on_start_callbacks.push(fn);

window.onload = () => { loaded = true; checkReady(); };
(window as any).beforeWasmCompile = (env: any) => {
  env.js_AddVoxelMesh  = js_AddVoxelMesh;
  env.js_FreeVoxelMesh = js_FreeVoxelMesh;
  env.js_SetVoxelMeshGeometry = js_SetVoxelMeshGeometry;
  env.js_SetVoxelMeshPosition = js_SetVoxelMeshPosition;
};
(window as any).onWasmCompile =
  (m: WasmModule) => { helper = new WasmHelper(m); checkReady(); };

//////////////////////////////////////////////////////////////////////////////

export {BlockId, MaterialId, Column, Env, init};
export {kChunkWidth, kEmptyBlock, kNoMaterial, kWorldHeight};
