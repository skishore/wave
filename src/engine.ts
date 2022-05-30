import {assert, drop, int, nonnull, Color, Tensor3, Vec3} from './base.js';
import {EntityComponentSystem} from './ecs.js';
import {Mesh, Renderer, Texture} from './renderer.js';
import {TerrainMesher} from './mesher.js';

//////////////////////////////////////////////////////////////////////////////
// The game engine:

type Input = 'up' | 'left' | 'down' | 'right' | 'space' | 'pointer';

class Container {
  element: Element;
  canvas: HTMLCanvasElement;
  stats: Element;
  bindings: Map<int, Input>;
  inputs: Record<Input, boolean>;
  deltas: {x: int, y: int, scroll: int};

  constructor(id: string) {
    this.element = nonnull(document.getElementById(id), () => id);
    this.canvas = nonnull(this.element.querySelector('canvas'));
    this.stats = nonnull(this.element.querySelector('#stats'));
    this.inputs = {
      up: false,
      left: false,
      down: false,
      right: false,
      space: false,
      pointer: false,
    };
    this.deltas = {x: 0, y: 0, scroll: 0};

    this.bindings = new Map();
    this.bindings.set('W'.charCodeAt(0), 'up');
    this.bindings.set('A'.charCodeAt(0), 'left');
    this.bindings.set('S'.charCodeAt(0), 'down');
    this.bindings.set('D'.charCodeAt(0), 'right');
    this.bindings.set(' '.charCodeAt(0), 'space');

    const element = this.element;
    element.addEventListener('click', () => element.requestPointerLock());
    document.addEventListener('keydown', e => this.onKeyInput(e, true));
    document.addEventListener('keyup', e => this.onKeyInput(e, false));
    document.addEventListener('mousemove', e => this.onMouseMove(e));
    document.addEventListener('touchmove', e => this.onMouseMove(e));
    document.addEventListener('pointerlockchange', e => this.onPointerInput(e));
    document.addEventListener('wheel', e => this.onMouseWheel(e));
  }

  displayStats(stats: string) {
    this.stats.textContent = stats;
  }

  onKeyInput(e: Event, down: boolean) {
    if (!this.inputs.pointer) return;
    const input = this.bindings.get((e as any).keyCode);
    if (input) this.onInput(e, input, down);
  }

  onMouseMove(e: Event) {
    if (!this.inputs.pointer) return;
    this.deltas.x += (e as any).movementX;
    this.deltas.y += (e as any).movementY;
  }

  onMouseWheel(e: Event) {
    if (!this.inputs.pointer) return;
    this.deltas.scroll += (e as any).deltaY;
  }

  onPointerInput(e: Event) {
    const locked = document.pointerLockElement === this.element;
    this.onInput(e, 'pointer', locked);
  }

  onInput(e: Event, input: Input, state: boolean) {
    this.inputs[input] = state;
    e.stopPropagation();
    e.preventDefault();
  }
};

//////////////////////////////////////////////////////////////////////////////

type BlockId = int & {__type__: 'BlockId'};
type MaterialId = int & {__type__: 'MaterialId'};

interface Material {
  color: Color,
  liquid: boolean,
  texture: Texture | null,
  textureIndex: int,
};

const kBlack: Color = [0, 0, 0, 1];
const kWhite: Color = [1, 1, 1, 1];

const kNoMaterial = 0 as MaterialId;

const kEmptyBlock = 0 as BlockId;
const kUnknownBlock = 1 as BlockId;

class Registry {
  opaque: boolean[];
  solid: boolean[];
  private faces: MaterialId[];
  private materials: Material[];
  private ids: Map<string, MaterialId>;

  constructor() {
    this.opaque = [false, false];
    this.solid = [false, true];
    this.faces = []
    for (let i = 0; i < 12; i++) {
      this.faces.push(kNoMaterial);
    }
    this.materials = [];
    this.ids = new Map();
  }

  addBlock(xs: string[], solid: boolean): BlockId {
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

      const data = this.getMaterialData(material);
      const alphaBlend = data.color[3] < 1;
      const alphaTest = data.texture && data.texture.alphaTest;
      if (alphaBlend || alphaTest) opaque = false;
    });

    const result = this.opaque.length as BlockId;
    this.opaque.push(opaque);
    this.solid.push(solid);
    return result;
  }

  addMaterialOfColor(name: string, color: Color, liquid: boolean = false) {
    this.addMaterialHelper(name, color, liquid, null);
  }

  addMaterialOfTexture(name: string, texture: Texture,
                       color: Color = kWhite, liquid: boolean = false) {
    this.addMaterialHelper(name, color, liquid, texture);
  }

  // faces has 6 elements for each block type: [+x, -x, +y, -y, +z, -z]
  getBlockFaceMaterial(id: BlockId, face: int): MaterialId {
    return this.faces[id * 6 + face];
  }

  getMaterialData(id: MaterialId): Material {
    assert(0 < id && id <= this.materials.length);
    return this.materials[id - 1];
  }

  private addMaterialHelper(
      name: string, color: Color, liquid: boolean, texture: Texture | null) {
    assert(name.length > 0, () => 'Empty material name!');
    assert(!this.ids.has(name), () => `Duplicate material: ${name}`);
    this.ids.set(name, this.materials.length as MaterialId);
    this.materials.push({color, liquid, texture, textureIndex: 0});
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
    this.index = next_index < this.ticks.length ? next_index : 0;
    const tick = Math.round(1000 * (this.now.now() - this.last));
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
const kTicksPerSecond = 30;

class Timing {
  now: any;
  render: (dt: int, fraction: number) => void;
  update: (dt: int) => void;
  renderBinding: () => void;
  updateDelay: number;
  updateLimit: number;
  lastRender: int;
  lastUpdate: int;
  renderPerf: Performance;
  updatePerf: Performance;

  constructor(render: (dt: int, fraction: number) => void,
              update: (dt: int) => void) {
    this.now = performance || Date;
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

    this.renderPerf = new Performance(this.now, 60);
    this.updatePerf = new Performance(this.now, 60);
  }

  renderHandler() {
    requestAnimationFrame(this.renderBinding);
    this.updateHandler();

    const now = this.now.now();
    const dt = now - this.lastRender;
    this.lastRender = now;

    const fraction = (now - this.lastUpdate) / this.updateDelay;
    try {
      this.renderPerf.begin();
      this.render(dt, fraction);
      this.renderPerf.end();
    } catch (e) {
      this.render = () => {};
      console.error(e);
    }
  }

  private updateHandler() {
    let now = this.now.now();
    const delay = this.updateDelay;
    const limit = now + this.updateLimit;

    while (this.lastUpdate + delay < now) {
      try {
        this.updatePerf.begin();
        this.update(delay);
        this.updatePerf.end();
      } catch (e) {
        this.update = () => {};
        console.error(e);
      }
      this.lastUpdate += delay;
      now = this.now.now();

      if (now > limit) {
        this.lastUpdate = now;
        break;
      }
    }
  }
};

//////////////////////////////////////////////////////////////////////////////

type Loader = (x: int, z: int, column: Column) => void;

class Column {
  private data: Uint16Array;
  private last: int;
  private size: int;

  constructor() {
    this.data = new Uint16Array(2 * kWorldHeight);
    this.last = 0;
    this.size = 0;
  }

  clear(): void {
    this.last = 0;
    this.size = 0;
  }

  fillChunk(x: int, z: int, chunk: Chunk): void {
    let last = 0;
    for (let i = 0; i < this.size; i++) {
      const offset = 2 * i;
      const block = this.data[offset + 0] as BlockId;
      const level = this.data[offset + 1];
      for (let y = last; y < level; y++) {
        chunk.setBlock(x, y, z, block);
      }
      last = level;
    }
  }

  push(block: BlockId, height: int): void {
    if (height <= this.last) return;
    this.last = Math.min(height, kWorldHeight);
    const offset = 2 * this.size;
    this.data[offset + 0] = block;
    this.data[offset + 1] = this.last;
    this.size++;
  }

  getNthBlock(n: int, bedrock: BlockId): BlockId {
    return n < 0 ? bedrock : this.data[2 * n + 0] as BlockId;
  }

  getNthLevel(n: int): int {
    return n < 0 ? 0 : this.data[2 * n + 1];
  }

  getSize(): int {
    return this.size;
  }
};

//////////////////////////////////////////////////////////////////////////////

const kChunkBits = 4;
const kChunkWidth = 1 << kChunkBits;
const kChunkMask = kChunkWidth - 1;
const kWorldHeight = 256;

const kChunkKeyBits = 12;
const kChunkKeySize = 1 << kChunkKeyBits;
const kChunkKeyMask = kChunkKeySize - 1;

const kChunkRadius = 12;
const kNeighbors = (kChunkRadius ? 4 : 0);

const kNumChunksToLoadPerFrame = 1;
const kNumChunksToMeshPerFrame = 1;
const kNumLODChunksToMeshPerFrame = 1;

const kFrontierLOD = 2;
const kFrontierRadius = 4;
const kFrontierLevels = 8;

// List of neighboring chunks to include when meshing.
type Point = [number, number, number];
const kNeighborOffsets = ((): [Point, Point, Point, Point][] => {
  const W = kChunkWidth;
  const H = kWorldHeight;
  const L = W - 1;
  const N = W + 1;
  return [
    [[ 0,  0,  0], [1, 2, 1], [0, 0, 0], [W, H, W]],
    [[-1,  0,  0], [0, 2, 1], [L, 0, 0], [1, H, W]],
    [[ 1,  0,  0], [N, 2, 1], [0, 0, 0], [1, H, W]],
    [[ 0,  0, -1], [1, 2, 0], [0, 0, L], [W, H, 1]],
    [[ 0,  0,  1], [1, 2, N], [0, 0, 0], [W, H, 1]],
  ];
})();

class Chunk {
  cx: int;
  cz: int;
  world: World;
  active: boolean;
  enabled: boolean;
  finished: boolean;
  requested: boolean;
  distance: number;
  private neighbors: int;
  private dirty: boolean;
  private solid: Mesh | null;
  private water: Mesh | null;
  private voxels: Tensor3 | null;

  constructor(world: World, cx: int, cz: int) {
    this.cx = cx;
    this.cz = cz;
    this.world = world;
    this.active = false;
    this.enabled = false;
    this.finished = false;
    this.requested = false;
    this.distance = 0;
    this.neighbors = kNeighbors;
    this.dirty = true;
    this.solid = null;
    this.water = null;
    this.voxels = null;
  }

  disable() {
    if (!this.enabled) return;
    this.world.enabled.delete(this);

    if (this.solid) this.solid.dispose();
    if (this.water) this.water.dispose();
    this.solid = null;
    this.water = null;

    this.active = false;
    this.enabled = false;
    this.dirty = true;
  }

  enable() {
    this.world.enabled.add(this);
    this.enabled = true;
    this.active = this.checkActive();
  }

  load(loader: Loader) {
    assert(!this.voxels);
    this.voxels = new Tensor3(kChunkWidth, kWorldHeight, kChunkWidth);

    const {cx, cz} = this;
    const dx = cx << kChunkBits;
    const dz = cz << kChunkBits;
    const column = new Column();
    for (let x = 0; x < kChunkWidth; x++) {
      for (let z = 0; z < kChunkWidth; z++) {
        loader(x + dx, z + dz, column);
        column.fillChunk(x + dx, z + dz, this);
        column.clear();
      }
    }

    this.finish();
  }

  finish() {
    assert(!this.finished);
    this.finished = true;

    const {cx, cz} = this;
    const neighbor = (x: int, z: int) => {
      const chunk = this.world.getChunk(x + cx, z + cz, false);
      if (!(chunk && chunk.finished)) return;
      chunk.notifyNeighborFinished();
      this.neighbors--;
    };
    neighbor(1, 0); neighbor(-1, 0);
    neighbor(0, 1); neighbor(0, -1);

    this.active = this.checkActive();
    this.dirty = !!this.voxels;
  }

  getBlock(x: int, y: int, z: int): BlockId {
    const voxels = this.voxels;
    if (!voxels) return kEmptyBlock;
    const xm = x & kChunkMask, zm = z & kChunkMask;
    return voxels.get(xm, y, zm) as BlockId;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    const voxels = this.voxels;
    if (!voxels) return;
    const xm = x & kChunkMask, zm = z & kChunkMask;
    if (!this.finished) return voxels.set(xm, y, zm, block);

    const old = voxels.get(xm, y, zm) as BlockId;
    if (old === block) return;
    voxels.set(xm, y, zm, block);
    this.dirty = true;

    const neighbor = (x: int, y: int, z: int) => {
      const {cx, cz} = this;
      const chunk = this.world.getChunk(x + cx, z + cz, false);
      if (chunk) chunk.dirty = true;
    };
    if (xm === 0) neighbor(-1, 0, 0);
    if (xm === kChunkMask) neighbor(1, 0, 0);
    if (zm === 0) neighbor(0, 0, -1);
    if (zm === kChunkMask) neighbor(0, 0, 1);
  }

  hasMesh(): boolean {
    return !!(this.solid || this.water);
  }

  needsRemesh(): boolean {
    return this.active && this.dirty;
  }

  remeshChunk() {
    assert(this.dirty);
    this.remeshTerrain();
    this.dirty = false;
  }

  private checkActive(): boolean {
    return this.enabled && this.finished && this.neighbors === 0;
  }

  private notifyNeighborFinished() {
    assert(this.neighbors > 0);
    this.neighbors--;
    this.active = this.checkActive();
  }

  private remeshTerrain() {
    const {cx, cz, world} = this;
    const {bedrock, buffer} = world;
    for (const offset of kNeighborOffsets) {
      const [c, dstPos, srcPos, size] = offset;
      const chunk = world.getChunk(cx + c[0], cz + c[2], false);
      chunk && chunk.voxels
        ? this.copyVoxels(buffer, dstPos, chunk.voxels, srcPos, size)
        : this.zeroVoxels(buffer, dstPos, size);
    }

    const x = cx << kChunkBits, z = cz << kChunkBits;
    const meshed = world.mesher.meshChunk(buffer, this.solid, this.water);
    const [solid, water] = meshed;
    if (solid) solid.setPosition(x, -1, z);
    if (water) water.setPosition(x, -1, z);
    this.solid = solid;
    this.water = water;
  }

  private copyVoxels(dst: Tensor3, dstPos: [number, number, number],
                     src: Tensor3, srcPos: [number, number, number],
                     size: [number, number, number]) {
    const [ni, nj, nk] = size;
    const [di, dj, dk] = dstPos;
    const [si, sj, sk] = srcPos;
    const dsj = dst.stride[1];
    const ssj = src.stride[1];
    for (let i = 0; i < ni; i++) {
      for (let k = 0; k < nk; k++) {
        // Unroll along the y-axis, since it's the longest chunk dimension.
        let sindex = src.index(si + i, sj, sk + k);
        let dindex = dst.index(di + i, dj, dk + k);
        for (let j = 0; j < nj; j++, dindex += dsj, sindex += ssj) {
          dst.data[dindex] = src.data[sindex];
        }
      }
    }
  }

  private zeroVoxels(dst: Tensor3, dstPos: [number, number, number],
                     size: [number, number, number]) {
    const [ni, nj, nk] = size;
    const [di, dj, dk] = dstPos;
    const dsj = dst.stride[1];
    for (let i = 0; i < ni; i++) {
      for (let k = 0; k < nk; k++) {
        // Unroll along the y-axis, since it's the longest chunk dimension.
        let dindex = dst.index(di + i, dj, dk + k);
        for (let j = 0; j < nj; j++, dindex += dsj) {
          dst.data[dindex] = kEmptyBlock;
        }
      }
    }
  }
};

class Counters {
  private values: Map<int, int>;

  constructor() {
    this.values = new Map();
  }

  bounds(): [int, int] {
    let min = Infinity, max = -Infinity;
    for (const value of this.values.keys()) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    return [min, max + 1];
  }

  dec(value: int): void {
    const count = this.values.get(value) || 0;
    if (count > 1) {
      this.values.set(value, count - 1);
    } else {
      assert(count === 1);
      this.values.delete(value);
    }
  }

  inc(value: int): void {
    const count = this.values.get(value) || 0;
    this.values.set(value, count + 1);
  }
};

const kLODSingleMask = (1 << 4) - 1;
const kLODMultiMask = (1 << 16) - 1;

class LODMultiMesh {
  solid: Mesh | null;
  water: Mesh | null;
  meshed: boolean[];

  private mask: int;
  private enabled: boolean[];

  constructor() {
    this.solid = null;
    this.water = null;
    this.mask = 0;
    this.meshed = new Array(4).fill(false);
    this.enabled = new Array(4).fill(false);
  }

  disable(index: int): void {
    if (!this.enabled[index]) return;
    this.setMask(index, kLODSingleMask);
    this.enabled[index] = false;
    if (this.enabled.some(x => x)) return;

    for (let i = 0; i < this.meshed.length; i++) this.meshed[i] = false;
    if (this.solid) this.solid.dispose();
    if (this.water) this.water.dispose();
    this.solid = null;
    this.water = null;
  }

  index(chunk: FrontierChunk): int {
    return ((chunk.cz & 1) << 1) | (chunk.cx & 1);
  }

  show(index: int, mask: int): void {
    assert(this.meshed[index]);
    this.setMask(index, mask);
    this.enabled[index] = true;
  }

  private setMask(index: int, mask: int): void {
    this.mask &= ~(kLODSingleMask << (index * 4));
    this.mask |= mask << (index * 4);
    const shown = this.mask !== kLODMultiMask;
    if (this.solid) this.solid.show(this.mask, shown);
    if (this.water) this.water.show(this.mask, shown);
  }
};

interface FrontierChunk {
  cx: int,
  cz: int,
  level: int,
  mesh: LODMultiMesh;
};

interface FrontierLevel {
  ax: int,
  az: int,
  bx: int,
  bz: int,
};

class Frontier {
  private xs: Counters;
  private zs: Counters;
  private world: World;
  private column: Column;
  private chunks: Map<int, FrontierChunk>;
  private meshes: Map<int, LODMultiMesh>;
  private levels: FrontierLevel[];
  private solid_heightmap: Uint32Array;
  private water_heightmap: Uint32Array;
  private side: int;

  constructor(world: World) {
    this.xs = new Counters();
    this.zs = new Counters();
    this.world = world;
    this.column = new Column();
    this.chunks = new Map();
    this.meshes = new Map();

    this.levels = [];
    for (let i = 0; i <= kFrontierLevels; i++) {
      this.levels.push({ax: 0, az: 0, bx: 0, bz: 0});
    }

    assert(kChunkWidth % kFrontierLOD === 0);
    const side = kChunkWidth / kFrontierLOD;
    const size = (side + 2) * (side + 2) * 2;
    this.solid_heightmap = new Uint32Array(size);
    this.water_heightmap = new Uint32Array(size);
    this.side = side;
  }

  chunkHidden(chunk: Chunk) {
    if (!chunk.hasMesh()) return;
    this.xs.dec(chunk.cx);
    this.zs.dec(chunk.cz);
  }

  chunkShown(chunk: Chunk) {
    if (chunk.hasMesh()) return;
    this.xs.inc(chunk.cx);
    this.zs.inc(chunk.cz);
  }

  remeshFrontier() {
    if (!kFrontierLevels) return;
    const [ax, bx] = this.xs.bounds();
    const [az, bz] = this.zs.bounds();
    if (ax > bx || az > bz) return;

    const bounds = this.levels[0];
    bounds.ax = ax; bounds.az = az; bounds.bx = bx; bounds.bz = bz;
    for (let i = 0; i < kFrontierLevels; i++) {
      this.computeLODAtLevel(i);
    }
    this.disableFarawayMeshes();
  }

  private computeLODAtLevel(level: int) {
    const prev = this.levels[level];
    const pax = prev.ax; const pbx = prev.bx;
    const paz = prev.az; const pbz = prev.bz;

    const r = kFrontierRadius;
    const ax = (pax - r) >> 1, bx = (pbx + r + 1) >> 1;
    const az = (paz - r) >> 1, bz = (pbz + r + 1) >> 1;

    const meshed = (dx: int, dz: int) => {
      if (!(pax <= dx && dx < pbx)) return false;
      if (!(paz <= dz && dz < pbz)) return false;
      if (level > 0) {
        const chunk = this.getFrontierChunk(dx, dz, level - 1);
        return chunk && chunk.mesh.meshed[chunk.mesh.index(chunk)];
      } else {
        const chunk = this.world.getChunk(dx, dz, false);
        return chunk && chunk.hasMesh();
      }
    };

    const required: [FrontierChunk, int][] = [];
    const optional: [FrontierChunk, int][] = [];

    const world = this.world;
    for (let cx = ax; cx < bx; cx++) {
      for (let cz = az; cz < bz; cz++) {
        const lod = this.getFrontierChunk(cx, cz, level);

        let mask = 0;
        for (let i = 0; i < 4; i++) {
          const dx = (cx << 1) + (i & 1 ? 1 : 0);
          const dz = (cz << 1) + (i & 2 ? 1 : 0);
          if (meshed(dx, dz)) mask |= (1 << i);
        }

        const shown = mask !== 15;
        const index = lod.mesh.index(lod);
        if (shown && !lod.mesh.meshed[index]) {
          (mask ? required : optional).push([lod, mask]);
        } else {
          lod.mesh.show(index, mask);
        }
      }
    }

    const mesh = (x: [FrontierChunk, int]) => {
      const [lod, mask] = x;
      this.createLODMeshes(lod);
      lod.mesh.show(lod.mesh.index(lod), mask);
    };

    const extra = kNumLODChunksToMeshPerFrame - required.length;
    const count = Math.min(optional.length, extra);

    if (0 < count && count < optional.length) {
      const hx = (ax + bx - 1) >> 1;
      const hz = (az + bz - 1) >> 1;
      const distance = (lod: FrontierChunk): int =>
          Math.abs(lod.cx - hx) + Math.abs(lod.cz - hz);
      optional.sort((a, b) => distance(a[0]) - distance(b[0]));
    }

    for (const lod of required) mesh(lod);
    for (let i = 0; i < count; i++) mesh(optional[i]);

    const bounds = this.levels[level + 1];
    bounds.ax = ax; bounds.az = az; bounds.bx = bx; bounds.bz = bz;
  }

  private createLODMeshes(chunk: FrontierChunk): void {
    const {column, side, world} = this;
    const {cx, cz, level, mesh} = chunk;
    const {bedrock, loadFrontier, registry} = world;
    const {solid_heightmap, water_heightmap} = this;
    if (!loadFrontier) return;

    assert(kFrontierLOD % 2 === 0);
    assert(registry.solid[bedrock]);
    const lshift = kChunkBits + level;
    const lod = kFrontierLOD << level;
    const x = (2 * cx + 1) << lshift;
    const z = (2 * cz + 1) << lshift;

    for (let k = 0; k < 4; k++) {
      const dx = (k & 1 ? 0 : -1 << lshift);
      const dz = (k & 2 ? 0 : -1 << lshift);
      const ax = x + dx + lod / 2;
      const az = z + dz + lod / 2;

      for (let i = 0; i < side; i++) {
        for (let j = 0; j < side; j++) {
          loadFrontier(ax + i * lod, az + j * lod, column);
          const offset = 2 * ((i + 1) + (j + 1) * (side + 2));

          const size = column.getSize();
          const last_block = column.getNthBlock(size - 1, bedrock);
          const last_level = column.getNthLevel(size - 1);

          if (registry.solid[last_block]) {
            solid_heightmap[offset + 0] = last_block;
            solid_heightmap[offset + 1] = last_level;
            water_heightmap[offset + 0] = 0;
            water_heightmap[offset + 1] = 0;
          } else {
            water_heightmap[offset + 0] = last_block;
            water_heightmap[offset + 1] = last_level;

            for (let i = size; i > 0; i--) {
              const block = column.getNthBlock(i - 2, bedrock);
              const level = column.getNthLevel(i - 2);
              if (!registry.solid[block]) continue;
              solid_heightmap[offset + 0] = block;
              solid_heightmap[offset + 1] = level;
              break;
            }
          }
          column.clear();
        }
      }

      const n = side + 2;
      const px = dx - lod - ((cx & 1 ? 1 : 3) << lshift);
      const pz = dz - lod - ((cz & 1 ? 1 : 3) << lshift);
      const mask = (1 << (k + 4 * mesh.index(chunk)));
      mesh.solid = this.world.mesher.meshFrontier(
          solid_heightmap, mask, px, pz, n, n, lod, mesh.solid, true);
      mesh.water = this.world.mesher.meshFrontier(
          water_heightmap, mask, px, pz, n, n, lod, mesh.water, false);
    }

    const px = x + ((cx & 1 ? 1 : 3) << lshift);
    const pz = z + ((cz & 1 ? 1 : 3) << lshift);
    if (mesh.solid) mesh.solid.setPosition(px, 0, pz);
    if (mesh.water) mesh.water.setPosition(px, 0, pz);
    mesh.meshed[mesh.index(chunk)] = true;
  }

  private disableFarawayMeshes() {
    const {chunks, levels} = this;
    for (const lod of chunks.values()) {
      const {cx, cz, level, mesh} = lod;
      const {ax, az, bx, bz} = levels[level + 1];
      const disable = !(ax <= cx && cx < bx && az <= cz && cz < bz);
      if (disable) mesh.disable(mesh.index(lod));
    }
  }

  private getFrontierChunk(cx: int, cz: int, level: int): FrontierChunk {
    const key = this.world.getChunkKey(cx, cz) * kFrontierLevels + level;
    const result = this.chunks.get(key);
    if (result) return result;
    const mesh = this.getMultiMesh(cx >> 1, cz >> 1, level + 1);
    const created = {cx, cz, level, mesh};
    this.chunks.set(key, created);
    return created;
  }

  private getMultiMesh(cx: int, cz: int, level: int): LODMultiMesh {
    const key = this.world.getChunkKey(cx, cz) * kFrontierLevels + level;
    const result = this.meshes.get(key);
    if (result) return result;
    const created = new LODMultiMesh();
    this.meshes.set(key, created);
    return created;
  }
};

class World {
  chunks: Map<int, Chunk>;
  enabled: Set<Chunk>;
  renderer: Renderer;
  registry: Registry;
  frontier: Frontier;
  mesher: TerrainMesher;
  bedrock: BlockId;
  loadChunk: Loader | null;
  loadFrontier: Loader | null;
  buffer: Tensor3;

  constructor(registry: Registry, renderer: Renderer) {
    this.chunks = new Map();
    this.enabled = new Set();
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
    const w = kChunkWidth + 2;
    const h = kWorldHeight + 3;
    this.buffer = new Tensor3(w, h, w);
  }

  getBlock(x: int, y: int, z: int): BlockId {
    if (y < 0) return this.bedrock;
    if (y >= kWorldHeight) return kEmptyBlock;
    const cx = x >> kChunkBits, cz = z >> kChunkBits;
    const chunk = this.getChunk(cx, cz, false);
    return chunk && chunk.finished ? chunk.getBlock(x, y, z) : kUnknownBlock;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    if (!(0 <= y && y < kWorldHeight)) return;
    const cx = x >> kChunkBits, cz = z >> kChunkBits;
    const chunk = this.getChunk(cx, cz, false);
    if (chunk && chunk.active) chunk.setBlock(x, y, z, block);
  }

  getChunk(cx: int, cz: int, add: boolean): Chunk | null {
    const key = this.getChunkKey(cx, cz);
    const result = this.chunks.get(key);
    if (result) return result;
    if (!add) return null;
    const chunk = new Chunk(this, cx, cz);
    this.chunks.set(key, chunk);
    return chunk;
  }

  getChunkKey(cx: int, cz: int): int {
    return (cx & kChunkKeyMask) | ((cz & kChunkKeyMask) << kChunkKeyBits);
  }

  setLoader(bedrock: BlockId, loadChunk: Loader, loadFrontier?: Loader) {
    this.bedrock = bedrock;
    this.loadChunk = loadChunk;
    this.loadFrontier = loadFrontier || loadChunk;

    const buffer = this.buffer;
    for (let x = 0; x < buffer.shape[0]; x++) {
      for (let z = 0; z < buffer.shape[2]; z++) {
        buffer.set(x, 1, z, bedrock);
      }
    }
  }

  recenter(x: number, y: number, z: number) {
    const dx = (x >> kChunkBits);
    const dz = (z >> kChunkBits);

    const area = kChunkWidth * kChunkWidth;
    const base = kChunkRadius * kChunkRadius;
    const lo = (base + 1) * area;
    const hi = (base + 9) * area;
    const limit = kChunkRadius + 1;
    const frontier = this.frontier;

    for (const chunk of this.enabled) {
      const {cx, cz} = chunk;
      const ax = Math.abs(cx - dx);
      const az = Math.abs(cz - dz);
      if (ax + az <= 1) continue;
      const disable = ax > limit || az > limit ||
                      this.distance(cx, cz, x, z) > hi;
      if (!disable) continue;
      frontier.chunkHidden(chunk);
      chunk.disable();
    }

    const loader = this.loadChunk;
    if (!loader) return;

    const requests = [];
    for (let i = dx - kChunkRadius; i <= dx + kChunkRadius; i++) {
      const ax = Math.abs(i - dx);
      for (let k = dz - kChunkRadius; k <= dz + kChunkRadius; k++) {
        const az = Math.abs(k - dz);
        const distance = this.distance(i, k, x, z);
        if (ax + az > 1 && distance > lo) continue;
        const chunk = nonnull(this.getChunk(i, k, true));
        if (!chunk.requested) requests.push(chunk);
        chunk.distance = distance;
        chunk.enable();
      }
    }

    const n = kNumChunksToLoadPerFrame;
    const m = Math.min(requests.length, n);
    if (requests.length > n) {
      requests.sort((x, y) => x.distance - y.distance);
    }
    for (let i = 0; i < m; i++) {
      const chunk = requests[i];
      chunk.requested = true;
      chunk.load(loader);
    }
  }

  remesh() {
    const queued = [];
    for (const chunk of this.chunks.values()) {
      if (chunk.needsRemesh()) queued.push(chunk);
    }
    const n = kNumChunksToMeshPerFrame;
    const m = Math.min(queued.length, kNumChunksToMeshPerFrame);
    if (queued.length > n) queued.sort((x, y) => x.distance - y.distance);

    const frontier = this.frontier;
    for (let i = 0; i < m; i++) {
      const chunk = queued[i];
      frontier.chunkShown(chunk);
      chunk.remeshChunk();
    }

    frontier.remeshFrontier();
  }

  private distance(cx: int, cz: int, x: number, z: number) {
    const half = kChunkWidth / 2;
    const dx = (cx << kChunkBits) + half - x;
    const dy = (cz << kChunkBits) + half - z;
    return dx * dx + dy * dy;
  }
};

//////////////////////////////////////////////////////////////////////////////

class Env {
  container: Container;
  entities: EntityComponentSystem;
  registry: Registry;
  renderer: Renderer;
  world: World;
  private cameraAlpha = 0;
  private cameraBlock = kEmptyBlock;
  private cameraColor = kWhite;
  private timing: Timing;
  private frame: int = 0;

  constructor(id: string) {
    this.container = new Container(id);
    this.entities = new EntityComponentSystem();
    this.registry = new Registry();
    this.renderer = new Renderer(this.container.canvas);
    this.world = new World(this.registry, this.renderer);
    this.timing = new Timing(this.render.bind(this), this.update.bind(this));
  }

  refresh() {
    const saved = this.container.inputs.pointer;
    this.container.inputs.pointer = true;
    this.update(0);
    this.render(0);
    this.container.inputs.pointer = saved;
  }

  render(dt: int) {
    if (!this.container.inputs.pointer) return;

    this.frame += 1;
    if (this.frame === 65536) this.frame = 0;
    const pos = this.frame / 256;
    const rad = 2 * Math.PI * pos;
    const move = 0.25 * (Math.cos(rad) * 0.5 + pos);
    const wave = 0.05 * (Math.sin(rad) + 3);

    const camera = this.renderer.camera;
    const deltas = this.container.deltas;
    camera.applyInputs(deltas.x, deltas.y, deltas.scroll);
    deltas.x = deltas.y = deltas.scroll = 0;

    this.entities.render(dt);
    this.updateOverlayColor(wave);
    const renderer_stats = this.renderer.render(move, wave);

    const timing = this.timing;
    if (timing.updatePerf.frame() % 10 !== 0) return;
    const stats = `Update: ${this.formatStat(timing.updatePerf)}\r\n` +
                  `Render: ${this.formatStat(timing.renderPerf)}\r\n` +
                  renderer_stats;
    this.container.displayStats(stats);
  }

  update(dt: int) {
    if (!this.container.inputs.pointer) return;
    this.entities.update(dt);
    this.world.remesh();
  }

  private formatStat(perf: Performance): string {
    const format = (x: number) => (x / 1000).toFixed(2);
    return `${format(perf.mean())}ms / ${format(perf.max())}ms`;
  }

  private getRenderBlock(x: int, y: int, z: int): BlockId {
    const result = this.world.getBlock(x, y, z);
    return result === kUnknownBlock ? kEmptyBlock : result;
  }

  private updateOverlayColor(wave: int) {
    const [x, y, z] = this.renderer.camera.position;
    const xi = Math.floor(x), zi = Math.floor(z);
    const yi = Math.floor(y + wave);
    const yf = y + wave - yi;

    const old_block = this.cameraBlock;
    const new_block = this.getRenderBlock(xi, yi, zi);
    this.cameraBlock = new_block;

    const focus = new_block === kEmptyBlock && yf < 0.2 &&
                  this.getRenderBlock(xi, yi - 1, zi) !== kEmptyBlock;
    this.renderer.camera.setMinZ(focus ? Math.max(yf / 2, 0.001) : 0.1);

    if (new_block === kEmptyBlock) {
      const changed = new_block !== old_block;
      if (changed) this.renderer.setOverlayColor(kWhite);
      return;
    }

    if (new_block !== old_block) {
      const material = this.registry.getBlockFaceMaterial(new_block, 3);
      const color = this.registry.getMaterialData(material).color;
      this.cameraColor = color.slice() as Color;
      this.cameraAlpha = color[3];
    }

    const falloff = (() => {
      const max = 2, step = 32;
      const limit = max * step;
      for (let i = 1; i < limit; i++) {
        const other = this.world.getBlock(xi, yi + i, zi);
        if (other === kEmptyBlock) return Math.pow(2, i / step);
      }
      return Math.pow(2, max);
    })();
    this.cameraColor[3] = 1 - (1 - this.cameraAlpha) / falloff;
    this.renderer.setOverlayColor(this.cameraColor);
  }
};

//////////////////////////////////////////////////////////////////////////////

export {BlockId, MaterialId, Column, Env, kEmptyBlock, kWorldHeight};
