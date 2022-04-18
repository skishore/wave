import {assert, drop, int, nonnull, Tensor3, Vec3} from './base.js';
import {EntityComponentSystem} from './ecs.js';
import {Mesh, Renderer} from './renderer.js';
import {TerrainMesher} from './mesher.js';

//////////////////////////////////////////////////////////////////////////////
// The game engine:

const Constants = {
  CHUNK_KEY_BITS: 8,
  TICK_RESOLUTION: 4,
  TICKS_PER_FRAME: 4,
  TICKS_PER_SECOND: 30,
};

//////////////////////////////////////////////////////////////////////////////

type Input = 'up' | 'left' | 'down' | 'right' | 'space' | 'pointer';

class Container {
  element: Element;
  canvas: HTMLCanvasElement;
  bindings: Map<int, Input>;
  inputs: Record<Input, boolean>;
  deltas: {x: int, y: int, scroll: int};

  constructor(id: string) {
    this.element = nonnull(document.getElementById(id), () => id);
    this.canvas = nonnull(this.element.querySelector('canvas'));
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

class Timing {
  now: any;
  render: (dt: int, fraction: number) => void;
  update: (dt: int) => void;
  renderBinding: () => void;
  updateDelay: number;
  updateLimit: number;
  lastRender: int;
  lastUpdate: int;

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

    this.updateDelay = 1000 / Constants.TICKS_PER_SECOND;
    this.updateLimit = this.updateDelay * Constants.TICKS_PER_FRAME;
    const updateInterval = this.updateDelay / Constants.TICK_RESOLUTION;
    setInterval(this.updateHandler.bind(this), updateInterval);
  }

  renderHandler() {
    requestAnimationFrame(this.renderBinding);
    this.updateHandler();

    const now = this.now.now();
    const dt = now - this.lastRender;
    this.lastRender = now;

    const fraction = (now - this.lastUpdate) / this.updateDelay;
    try {
      this.render(dt, fraction);
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
        this.update(delay);
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
const kChunkSize = 1 << kChunkBits;
const kChunkMask = kChunkSize - 1;

const kChunkKeyBits = 8;
const kChunkKeySize = 1 << kChunkKeyBits;
const kChunkKeyMask = kChunkKeySize - 1;

const kChunkRadius = 0;
const kNeighbors = (kChunkRadius ? 6 : 0);

const kNumChunksToLoadPerFrame = 1;
const kNumChunksToMeshPerFrame = 1;

const kSpriteKeyBits = 10;
const kSpriteKeySize = 1 << kSpriteKeyBits;
const kSpriteKeyMask = kSpriteKeySize - 1;

// These conditions ensure that we'll dispose of a sprite before allocating
// a new sprite at a key that collides with the old one.
assert((1 << kSpriteKeyBits) > (kChunkSize * (2 * kChunkRadius + 1)));

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
  cy: int;
  cz: int;

  constructor(world: World, cx: int, cy: int, cz: int) {
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
    this.cy = cy;
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
    this.voxels = new Tensor3(kChunkSize, kChunkSize, kChunkSize);
  }

  finish() {
    assert(!this.finished);
    this.finished = true;

    const {cx, cy, cz} = this;
    const neighbor = (x: int, y: int, z: int) => {
      const chunk = this.world.getChunk(x + cx, y + cy, z + cz, false);
      if (!(chunk && chunk.finished)) return;
      chunk.notifyNeighborFinished();
      this.neighbors--;
    };
    neighbor(1, 0, 0); neighbor(-1, 0, 0);
    neighbor(0, 1, 0); neighbor(0, -1, 0);
    neighbor(0, 0, 1); neighbor(0, 0, -1);

    if (!this.voxels) {
      this.spritesDirty = false;
      this.terrainDirty = false;
    }

    this.active = this.checkActive();
  }

  getBlock(x: int, y: int, z: int): BlockId {
    const mask = kChunkMask;
    if (!this.voxels) return kEmptyBlock;
    return this.voxels.get(x & mask, y & mask, z & mask) as BlockId;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    const xm = x & kChunkMask;
    const ym = y & kChunkMask;
    const zm = z & kChunkMask;
    const voxels = nonnull(this.voxels);
    const old = voxels.get(xm, ym, zm) as BlockId;
    if (old === block) return;

    const old_mesh = this.world.registry._meshes[old];
    const new_mesh = this.world.registry._meshes[block];
    if (!this.spritesDirty) {
      // TODO(skishore): Restore TerrainSprites.
      //if (old_mesh) this.world.sprites.remove(x, y, z, old);
      //if (new_mesh) this.world.sprites.add(x, y, z, block, new_mesh);
    }

    voxels.set(xm, ym, zm, block);
    if (old_mesh && new_mesh) return;

    this.terrainDirty = true;
    const neighbor = (x: int, y: int, z: int) => {
      const {cx, cy, cz} = this;
      const chunk = this.world.getChunk(x + cx, y + cy, z + cz, false);
      if (!(chunk && chunk.finished)) return;
      chunk.terrainDirty = true;
    };
    if (xm === 0) neighbor(-1, 0, 0);
    if (xm === kChunkMask) neighbor(1, 0, 0);
    if (ym === 0) neighbor(0, -1, 0);
    if (ym === kChunkMask) neighbor(0, 1, 0);
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
    const dy = this.cy << kChunkBits;
    const dz = this.cz << kChunkBits;

    // TODO(skishore): Restore TerrainSprites.
    //const sprites = this.world.sprites;
    //for (let x = 0; x < kChunkSize; x++) {
    //  for (let y = 0; y < kChunkSize; y++) {
    //    for (let z = 0; z < kChunkSize; z++) {
    //      const cell = this.voxels.get(x, y, z) as BlockId;
    //      const mesh = this.world.registry._meshes[cell];
    //      if (!mesh) continue;
    //      enabled ? sprites.add(x + dx, y + dy, z + dz, cell, mesh)
    //              : sprites.remove(x + dx, y + dy, z + dz, cell);
    //    }
    //  }
    //}
  }

  private refreshTerrain() {
    if (this.mesh) this.mesh.dispose();
    const {cx, cy, cz, voxels} = this;
    const mesh = voxels ? this.world.mesher.mesh(voxels) : null;
    if (mesh) mesh.setPosition(1, 1, 1);
    this.mesh = mesh;
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
    const chunk = this.getChunk(x >> bits, y >> bits, z >> bits, false);
    return chunk && chunk.finished ? chunk.getBlock(x, y, z) : kUnknownBlock;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    const bits = kChunkBits;
    const chunk = this.getChunk(x >> bits, y >> bits, z >> bits, false);
    if (chunk && chunk.active) chunk.setBlock(x, y, z, block);
  }

  getChunk(cx: int, cy: int, cz: int, add: boolean): Chunk | null {
    const key = (cx & kChunkKeyMask) << (0 * kChunkKeyBits) |
                (cy & kChunkKeyMask) << (1 * kChunkKeyBits) |
                (cz & kChunkKeyMask) << (2 * kChunkKeyBits);
    const result = this.chunks.get(key);
    if (result) return result;
    if (!add) return null;
    const chunk = new Chunk(this, cx, cy, cz);
    this.chunks.set(key, chunk);
    return chunk;
  }

  recenter(x: number, y: number, z: number): Chunk[] {
    const dx = (x >> kChunkBits);
    const dy = (y >> kChunkBits);
    const dz = (z >> kChunkBits);

    const lo = (kChunkRadius * kChunkRadius + 1) * kChunkSize * kChunkSize;
    const hi = (kChunkRadius * kChunkRadius + 9) * kChunkSize * kChunkSize;
    const limit = kChunkRadius + 1;

    const disabled = [];
    for (const chunk of this.enabled) {
      const {cx, cy, cz} = chunk;
      const ax = Math.abs(cx - dx);
      const ay = Math.abs(cy - dy);
      const az = Math.abs(cz - dz);
      if (ax + ay + az <= 1) continue;
      const disable = ax > limit || ay > limit || az > limit ||
                      this.distance(cx, cy, cz, x, y, z) > hi;
      if (disable) disabled.push(chunk);
    }
    for (const chunk of disabled) chunk.disable();

    const requests = [];
    for (let i = dx - kChunkRadius; i <= dx + kChunkRadius; i++) {
      const ax = Math.abs(i - dx);
      for (let j = dy - kChunkRadius; j <= dy + kChunkRadius; j++) {
        const ay = Math.abs(j - dy);
        for (let k = dz - kChunkRadius; k <= dz + kChunkRadius; k++) {
          const az = Math.abs(k - dz);
          const distance = this.distance(i, j, k, x, y, z);
          if (ax + ay + az > 1 && distance > lo) continue;
          const chunk = nonnull(this.getChunk(i, j, k, true));
          if (!chunk.requested) requests.push(chunk);
          chunk.distance = distance;
          chunk.enable();
        }
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

  private distance(cx: int, cy: int, cz: int, x: number, y: number, z: number) {
    const half = kChunkSize / 2;
    const i = (cx << kChunkBits) + half - x;
    const j = (cy << kChunkBits) + half - y;
    const k = (cz << kChunkBits) + half - z;
    return i * i + j * j + k * k;
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
  }

  update(dt: int) {
    if (!this.container.inputs.pointer) return;
    this.entities.update(dt);
    this.world.remesh();
  }
};

//////////////////////////////////////////////////////////////////////////////

export {Chunk, Env, kChunkBits, kChunkSize, kEmptyBlock};
