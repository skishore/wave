import {assert, drop, int, nonnull, Tensor3, Vec3} from './base.js';
import {EntityComponentSystem} from './ecs.js';
import {Mesh, Renderer} from './renderer.js';
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

type Color = [number, number, number, number];

interface Material {
  color: Color,
  texture: string | null,
  textureIndex: int,
};

const kBlack: Color = [0, 0, 0, 1];
const kWhite: Color = [1, 1, 1, 1];

const kNoMaterial = 0 as MaterialId;

const kEmptyBlock = 0 as BlockId;
const kUnknownBlock = 1 as BlockId;

class Registry {
  _opaque: boolean[];
  _solid: boolean[];
  _faces: MaterialId[];
  _materials: Material[];
  _ids: Map<string, MaterialId>;
  rust: RustHelper;

  constructor(rust: RustHelper) {
    this._opaque = [false, false];
    this._solid = [false, true];
    this._faces = []
    for (let i = 0; i < 12; i++) {
      this._faces.push(kNoMaterial);
    }
    this._materials = [];
    this._ids = new Map();
    this.rust = rust;

    this.registerBlockWithRust(0);
    this.registerBlockWithRust(1);
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

    const result = this._opaque.length as BlockId;
    this._opaque.push(solid);
    this._solid.push(solid);
    materials.forEach(x => {
      const material = this._ids.get(x);
      if (material === undefined) throw new Error(`Unknown material: ${x}`);
      this._faces.push(material + 1 as MaterialId);
    });

    this.registerBlockWithRust(result);
    return result;
  }

  addMaterialOfColor(name: string, color: Color) {
    this.addMaterialHelper(name, color, null);
  }

  addMaterialOfTexture(name: string, texture: string) {
    this.addMaterialHelper(name, kWhite, texture);
  }

  // faces has 6 elements for each block type: [+x, -x, +y, -y, +z, -z]
  getBlockFaceMaterial(id: BlockId, face: int): MaterialId {
    return this._faces[id * 6 + face];
  }

  getMaterialData(id: MaterialId): Material {
    assert(0 < id && id <= this._materials.length);
    return this._materials[id - 1];
  }

  private addMaterialHelper(
      name: string, color: Color, texture: string | null) {
    assert(name.length > 0, () => 'Empty material name!');
    assert(!this._ids.has(name), () => `Duplicate material: ${name}`);
    this._ids.set(name, this._materials.length as MaterialId);
    this._materials.push({color, texture, textureIndex: 0});
    this.registerMaterialWithRust(color, texture);
  }

  private registerBlockWithRust(id: int) {
    this.rust.register_block(
      this._faces[6 * id + 0],
      this._faces[6 * id + 1],
      this._faces[6 * id + 2],
      this._faces[6 * id + 3],
      this._faces[6 * id + 4],
      this._faces[6 * id + 5],
      this._opaque[id],
      this._solid[id],
    );
  }

  private registerMaterialWithRust(color: Color, texture: string | null) {
    this.rust.register_facet(color[0], color[1], color[2], color[3], 0);
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
    const tick = Math.round(1000 * (performance.now() - this.last));
    this.sum += tick - this.ticks[index];
    this.ticks[index] = tick;
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

  clear() {
    this.last = 0;
    this.size = 0;
  }

  fill(x: int, z: int, chunk: Chunk) {
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

  push(block: BlockId, count: int) {
    if (count <= 0) return;
    const offset = 2 * this.size;
    const last = Math.min(this.last + count, kWorldHeight);
    this.data[offset + 0] = block;
    this.data[offset + 1] = last;
    this.last = last;
    this.size++;
  }
};

//////////////////////////////////////////////////////////////////////////////

const kChunkBits = 4;
const kChunkWidth = 1 << kChunkBits;
const kChunkMask = kChunkWidth - 1;
const kWorldHeight = 256;

const kChunkKeyBits = 16;
const kChunkKeySize = 1 << kChunkKeyBits;
const kChunkKeyMask = kChunkKeySize - 1;

const kChunkRadius = 12;
const kNeighbors = (kChunkRadius ? 4 : 0);

const kNumChunksToLoadPerFrame = 1;
const kNumChunksToMeshPerFrame = 1;

// List of neighboring chunks to include when meshing.
type Point = [number, number, number];
const kNeighborOffsets = ((): [Point, Point, Point, Point][] => {
  const W = kChunkWidth;
  const H = kWorldHeight;
  const L = W - 1;
  const N = W + 1;
  return [
    [[ 0,  0,  0], [1, 1, 1], [0, 0, 0], [W, H, W]],
    [[-1,  0,  0], [0, 1, 1], [L, 0, 0], [1, H, W]],
    [[ 1,  0,  0], [N, 1, 1], [0, 0, 0], [1, H, W]],
    [[ 0,  0, -1], [1, 1, 0], [0, 0, L], [W, H, 1]],
    [[ 0,  0,  1], [1, 1, N], [0, 0, 0], [W, H, 1]],
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
  private mesh: Mesh | null;
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
    this.mesh = null;
    this.voxels = null;
  }

  disable() {
    if (!this.enabled) return;
    this.world.enabled.delete(this);

    if (this.mesh) this.mesh.dispose();
    this.mesh = null;

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
        column.fill(x + dx, z + dz, this);
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
    if (!this.voxels) return kEmptyBlock;
    if (!(0 <= y && y < kWorldHeight)) return kEmptyBlock;
    const mask = kChunkMask;
    return this.voxels.get(x & mask, y, z & mask) as BlockId;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    if (!this.voxels) return;
    if (!(0 <= y && y < kWorldHeight)) return;
    const xm = x & kChunkMask;
    const zm = z & kChunkMask;
    const voxels = nonnull(this.voxels);
    const old = voxels.get(xm, y, zm) as BlockId;
    if (old === block) return;

    voxels.set(xm, y, zm, block);

    this.dirty = true;
    if (!this.finished) return;

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

  needsRemesh() {
    return this.active && this.dirty;
  }

  remesh() {
    assert(this.dirty);
    this.refreshTerrain();
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

  private refreshTerrain() {
    if (this.mesh) this.mesh.dispose();
    const {cx, cz, voxels} = this;
    const w = kChunkWidth + 2;
    const h = kWorldHeight + 2;
    const expanded = new Tensor3(w, h, w);
    for (const offset of kNeighborOffsets) {
      const [c, dstPos, srcPos, size] = offset;
      const chunk = this.world.getChunk(cx + c[0], cz + c[2], false);
      if (!(chunk && chunk.voxels)) continue;
      this.copyVoxels(expanded, dstPos, chunk.voxels, srcPos, size);
    }
    const dx = cx << kChunkBits;
    const dz = cz << kChunkBits;
    (() => {
      const rs_start = Date.now();
      for (let i = 0; i < 100; i++) {
        this.world.rust.mesh();
      }
      const rs_time = Date.now() - rs_start;
      const js_start = Date.now();
      for (let i = 0; i < 100; i++) {
        this.world.mesher.countQuads(expanded);
      }
      const js_time = Date.now() - js_start;
      console.log(rs_time, js_time);
    })();
    const rs_quads = this.world.rust.mesh();
    const js_quads = this.world.mesher.countQuads(expanded);
    assert(rs_quads === js_quads);
    const mesh = voxels ? this.world.mesher.mesh(expanded) : null;
    if (mesh) mesh.setPosition(dx, 0, dz);
    this.mesh = mesh;
  }

  private copyVoxels(dst: Tensor3, dstPos: [number, number, number],
                     src: Tensor3, srcPos: [number, number, number],
                     size: [number, number, number]) {
    const ni = size[0], nj = size[1], nk = size[2];
    const di = dstPos[0], dj = dstPos[1], dk = dstPos[2];
    const si = srcPos[0], sj = srcPos[1], sk = srcPos[2];

    const y_offset = kChunkWidth + 2;
    const z_offset = y_offset * (kWorldHeight + 2);
    const memory = this.world.rust.memory;
    const raw = new Uint32Array(memory.buffer, this.world.data);

    for (let i = 0; i < ni; i++) {
      for (let j = 0; j < nj; j++) {
        for (let k = 0; k < nk; k++) {
          const tile = src.get(si + i, sj + j, sk + k);
          dst.set(di + i, dj + j, dk + k, tile);
          raw[di + i + (dj + j) * y_offset + (dk + k) * z_offset] = tile;
        }
      }
    }

    assert(raw.buffer === memory.buffer);
  }
};

class World {
  chunks: Map<int, Chunk>;
  enabled: Set<Chunk>;
  renderer: Renderer;
  registry: Registry;
  mesher: TerrainMesher;
  loader: Loader | null;
  rust: RustHelper;
  data: int;

  constructor(registry: Registry, renderer: Renderer, rust: RustHelper) {
    this.chunks = new Map();
    this.enabled = new Set();
    this.renderer = renderer;
    this.registry = registry;
    this.mesher = new TerrainMesher(registry, renderer);
    this.loader = null;
    this.rust = rust;

    const w = kChunkWidth + 2;
    const h = kWorldHeight + 2;
    this.data = rust.allocate_voxels(w, h, w);
  }

  getBlock(x: int, y: int, z: int): BlockId {
    const bits = kChunkBits;
    const chunk = this.getChunk(x >> bits, z >> bits, false);
    return chunk && chunk.finished ? chunk.getBlock(x, y, z) : kUnknownBlock;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    const bits = kChunkBits;
    const chunk = this.getChunk(x >> bits, z >> bits, false);
    if (chunk && chunk.active) chunk.setBlock(x, y, z, block);
  }

  getChunk(cx: int, cz: int, add: boolean): Chunk | null {
    const key = (cx & kChunkKeyMask) | ((cz & kChunkKeyMask) << kChunkKeyBits);
    const result = this.chunks.get(key);
    if (result) return result;
    if (!add) return null;
    const chunk = new Chunk(this, cx, cz);
    this.chunks.set(key, chunk);
    return chunk;
  }

  setLoader(loader: Loader | null) {
    this.loader = loader;
  }

  recenter(x: number, y: number, z: number) {
    const dx = (x >> kChunkBits);
    const dz = (z >> kChunkBits);

    const area = kChunkWidth * kChunkWidth;
    const base = kChunkRadius * kChunkRadius;
    const lo = (base + 1) * area;
    const hi = (base + 9) * area;
    const limit = kChunkRadius + 1;

    for (const chunk of this.enabled) {
      const {cx, cz} = chunk;
      const ax = Math.abs(cx - dx);
      const az = Math.abs(cz - dz);
      if (ax + az <= 1) continue;
      const disable = ax > limit || az > limit ||
                      this.distance(cx, cz, x, z) > hi;
      if (disable) chunk.disable();
    }

    const loader = this.loader;
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
    for (let i = 0; i < m; i++) queued[i].remesh();
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
  timing: Timing;
  world: World;

  constructor(id: string, rust: RustHelper) {
    this.container = new Container(id);
    this.entities = new EntityComponentSystem();
    this.registry = new Registry(rust);
    this.renderer = new Renderer(this.container.canvas);
    this.timing = new Timing(this.render.bind(this), this.update.bind(this));
    this.world = new World(this.registry, this.renderer, rust);
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

    const camera = this.renderer.camera;
    const deltas = this.container.deltas;
    camera.applyInputs(deltas.x, deltas.y, deltas.scroll);
    deltas.x = deltas.y = deltas.scroll = 0;

    this.entities.render(dt);
    this.renderer.render();

    const timing = this.timing;
    const stats = `Update: ${this.formatStat(timing.updatePerf)}\r\n` +
                  `Render: ${this.formatStat(timing.renderPerf)}`;
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
};

//////////////////////////////////////////////////////////////////////////////

interface RustHelper {
  memory: WebAssembly.Memory,
  register_block: (f0: int, f1: int, f2: int, f3: int, f4: int, f5: int,
                   opaque: boolean, solid: boolean) => null,
  register_facet: (c0: number, c1: number, c2: number, c3: number,
                   texture: int) => null,
  allocate_voxels: (x: int, y: int, z: int) => int,
  mesh: () => int,
};

//////////////////////////////////////////////////////////////////////////////

export {BlockId, MaterialId, Column, Env, RustHelper, kWorldHeight};
