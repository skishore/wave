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

type SpriteMesh = never;

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
  _meshes: (SpriteMesh | null)[];
  _materials: Material[];
  _ids: Map<string, MaterialId>;

  constructor() {
    this._opaque = [false, false];
    this._solid = [false, true];
    this._meshes = [null, null];
    this._faces = []
    for (let i = 0; i < 12; i++) {
      this._faces.push(kNoMaterial);
    }
    this._materials = [];
    this._ids = new Map();
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
    this._meshes.push(null);
    materials.forEach(x => {
      const material = this._ids.get(x);
      if (material === undefined) throw new Error(`Unknown material: ${x}`);
      this._faces.push(material + 1 as MaterialId);
    });

    return result;
  }

  addBlockSprite(mesh: SpriteMesh, solid: boolean): BlockId {
    const result = this._opaque.length as BlockId;
    this._opaque.push(false);
    this._solid.push(solid);
    this._meshes.push(mesh);
    for (let i = 0; i < 6; i++) this._faces.push(kNoMaterial);
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

const kChunkBits = 4;
const kChunkWidth = 1 << kChunkBits;
const kChunkMask = kChunkWidth - 1;
const kChunkHeight = 256;

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
  const H = kChunkHeight;
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
  world: World;
  active: boolean;
  enabled: boolean;
  finished: boolean;
  requested: boolean;
  neighbors: int;
  spritesDirty: boolean;
  terrainDirty: boolean;
  mesh: Mesh | null;
  voxels: Tensor3 | null;
  distance: number;
  cx: int;
  cz: int;

  constructor(world: World, cx: int, cz: int) {
    this.world = world;
    this.active = false;
    this.enabled = false;
    this.finished = false;
    this.requested = false;
    this.neighbors = kNeighbors;
    this.spritesDirty = true;
    this.terrainDirty = true;
    this.mesh = null;
    this.voxels = null;
    this.distance = 0;
    this.cx = cx;
    this.cz = cz;
  }

  disable() {
    if (!this.enabled) return;
    this.world.enabled.delete(this);

    if (!this.spritesDirty) this.refreshSprites(false);
    if (this.mesh) this.mesh.dispose();
    this.mesh = null;

    this.active = false;
    this.enabled = false;
    this.spritesDirty = true;
    this.terrainDirty = true;
  }

  enable() {
    this.world.enabled.add(this);
    this.enabled = true;
    this.active = this.checkActive();
  }

  init() {
    assert(!this.voxels);
    this.voxels = new Tensor3(kChunkWidth, kChunkHeight, kChunkWidth);
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

    if (!this.voxels) {
      this.spritesDirty = false;
      this.terrainDirty = false;
    }

    this.active = this.checkActive();
  }

  getBlock(x: int, y: int, z: int): BlockId {
    if (!this.voxels) return kEmptyBlock;
    if (!(0 <= y && y < kChunkHeight)) return kEmptyBlock;
    const mask = kChunkMask;
    return this.voxels.get(x & mask, y, z & mask) as BlockId;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    if (!this.voxels) return;
    if (!(0 <= y && y < kChunkHeight)) return;
    const xm = x & kChunkMask;
    const zm = z & kChunkMask;
    const voxels = nonnull(this.voxels);
    const old = voxels.get(xm, y, zm) as BlockId;
    if (old === block) return;

    const old_mesh = this.world.registry._meshes[old];
    const new_mesh = this.world.registry._meshes[block];
    if (!this.spritesDirty) {
      // TODO(skishore): Restore TerrainSprites.
      //if (old_mesh) this.world.sprites.remove(x, y, z, old);
      //if (new_mesh) this.world.sprites.add(x, y, z, block, new_mesh);
    }

    voxels.set(xm, y, zm, block);
    if (old_mesh && new_mesh) return;

    this.terrainDirty = true;
    if (!this.finished) return;

    const neighbor = (x: int, y: int, z: int) => {
      const {cx, cz} = this;
      const chunk = this.world.getChunk(x + cx, z + cz, false);
      if (chunk) chunk.terrainDirty = true;
    };
    if (xm === 0) neighbor(-1, 0, 0);
    if (xm === kChunkMask) neighbor(1, 0, 0);
    if (zm === 0) neighbor(0, 0, -1);
    if (zm === kChunkMask) neighbor(0, 0, 1);
  }

  needsRemesh() {
    return this.active && (this.spritesDirty || this.terrainDirty);
  }

  remesh() {
    if (this.spritesDirty) {
      this.refreshSprites(true);
      this.spritesDirty = false;
    }
    if (this.terrainDirty) {
      this.refreshTerrain();
      this.terrainDirty = false;
    }
  }

  private checkActive(): boolean {
    return this.enabled && this.finished && this.neighbors === 0;
  }

  private notifyNeighborFinished() {
    assert(this.neighbors > 0);
    this.neighbors--;
    this.active = this.checkActive();
  }

  private refreshSprites(enabled: boolean) {
    if (!this.voxels) return;
    const dx = this.cx << kChunkBits;
    const dz = this.cz << kChunkBits;

    // TODO(skishore): Restore TerrainSprites.
    //const sprites = this.world.sprites;
    //for (let x = 0; x < kChunkWidth; x++) {
    //  for (let y = 0; y < kChunkHeight; y++) {
    //    for (let z = 0; z < kChunkWidth; z++) {
    //      const cell = this.voxels.get(x, y, z) as BlockId;
    //      const mesh = this.world.registry._meshes[cell];
    //      if (!mesh) continue;
    //      enabled ? sprites.add(x + dx, y, z + dz, cell, mesh)
    //              : sprites.remove(x + dx, y, z + dz, cell);
    //    }
    //  }
    //}
  }

  private refreshTerrain() {
    if (this.mesh) this.mesh.dispose();
    const {cx, cz, voxels} = this;
    const w = kChunkWidth + 2;
    const h = kChunkHeight + 2;
    const expanded = new Tensor3(w, h, w);
    for (const offset of kNeighborOffsets) {
      const [c, dstPos, srcPos, size] = offset;
      const chunk = this.world.getChunk(cx + c[0], cz + c[2], false);
      if (!(chunk && chunk.voxels)) continue;
      this.copyVoxels(expanded, dstPos, chunk.voxels, srcPos, size);
    }
    const dx = cx << kChunkBits;
    const dz = cz << kChunkBits;
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
    for (let i = 0; i < ni; i++) {
      for (let j = 0; j < nj; j++) {
        for (let k = 0; k < nk; k++) {
          const tile = src.get(si + i, sj + j, sk + k);
          dst.set(di + i, dj + j, dk + k, tile);
        }
      }
    }
  }
};

class World {
  chunks: Map<int, Chunk>;
  enabled: Set<Chunk>;
  renderer: Renderer;
  registry: Registry;
  mesher: TerrainMesher;

  constructor(registry: Registry, renderer: Renderer) {
    this.chunks = new Map();
    this.enabled = new Set();
    this.renderer = renderer;
    this.registry = registry;
    this.mesher = new TerrainMesher(registry, renderer);
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

  recenter(x: number, y: number, z: number): Chunk[] {
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
    let result = requests;
    if (requests.length > n) {
      requests.sort((x, y) => x.distance - y.distance);
      result = requests.slice(0, n);
    }
    result.forEach(x => x.requested = true);
    return result;
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

  constructor(id: string) {
    this.container = new Container(id);
    this.entities = new EntityComponentSystem();
    this.registry = new Registry();
    this.renderer = new Renderer(this.container.canvas);
    this.timing = new Timing(this.render.bind(this), this.update.bind(this));
    this.world = new World(this.registry, this.renderer);
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

    // TODO(skishore): Restore TerrainSprites.
    //this.world.sprites.updateBillboards(camera.heading);
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

export {Chunk, Env, kChunkWidth, kChunkHeight, kEmptyBlock};
