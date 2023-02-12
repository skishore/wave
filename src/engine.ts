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

    this.helper = helper;
    this.renderer = renderer;
    this.helper.block_to_instance = this.meshes;

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
        id, !!this.meshes[id], this.opaque[id], this.solid[id], this.light[id],
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

const kTmpPos     = Vec3.create();
const kTmpMin     = Vec3.create();
const kTmpMax     = Vec3.create();
const kTmpDelta   = Vec3.create();
const kTmpImpacts = Vec3.create();

const kMinZLowerBound = 0.001;
const kMinZUpperBound = 0.1;

const kChunkWidth  = 16;
const kWorldHeight = 256;

const kChunkRadius = 12;
const kFrontierRadius = 8;
const kFrontierLevels = 6;

const kSunlightLevel = 0xf;

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

  constructor(id: string) {
    this.container = new Container(id);
    this.entities = new EntityComponentSystem();
    this.renderer = new Renderer(this.container.canvas);

    this.helper = nonnull(helper);
    this.helper.renderer = this.renderer;
    this.helper.initializeWorld(kChunkRadius, kFrontierRadius, kFrontierLevels);

    this.registry = new Registry(this.helper, this.renderer);
    this.highlight = this.renderer.addHighlightMesh();
    this.highlightPosition = Vec3.create();

    this.cameraColor = kWhite.slice() as Color;
    this.cameraMaterial = kNoMaterial;

    const remesh = this.remesh.bind(this);
    const render = this.render.bind(this);
    const update = this.update.bind(this);
    this.timing = new Timing(remesh, render, update);
  }

  getBaseHeight(x: int, z: int): int {
    return this.helper.module.asm.getBaseHeight(x, z);
  }

  getBlock(x: int, y: int, z: int): BlockId {
    return this.helper.getBlock(x, y, z);
  }

  getLight(x: int, y: int, z: int): number {
    return lighting(this.helper.getLightLevel(x, y, z));
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
  }

  setCameraTarget(x: number, y: number, z: number): void {
    this.renderer.camera.setTarget(x, y, z);
    this.setSafeZoomDistance();
  }

  setPointLight(x: int, y: int, z: int, level: int): void {
    this.helper.setPointLight(x, y, z, level);
  }

  recenter(x: number, y: number, z: number): void {
    const ix = int(Math.round(x)), iz = int(Math.round(z));
    this.helper.recenterWorld(ix, iz);
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
    const result = this.helper.getBlock(x, y, z);
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
      const block = this.helper.getBlock(x, y, z);
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
      const block = this.helper.getBlock(x, y, z);
      if (!this.registry.solid[block]) return true;

      let mask = 0;
      const pos = kTmpPos;
      Vec3.set(pos, x, y, z);
      for (let d = 0; d < 3; d++) {
        pos[d] += 1;
        const b0 = this.helper.getBlock(int(pos[0]), int(pos[1]), int(pos[2]));
        if (this.registry.opaque[b0]) mask |= (1 << (2 * d + 0));
        pos[d] -= 2;
        const b1 = this.helper.getBlock(int(pos[0]), int(pos[1]), int(pos[2]));
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
    const light = this.getLight(xi, yi, zi);
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

    initializeWorld: (chunkRadius: int, frontierRadius: int, frontierLevels: int) => void;
    recenterWorld: (x: int, z: int) => void,
    remeshWorld: () => void,

    getBaseHeight: (x: int, z: int) => int,
    getBlock: (x: int, y: int, z: int) => BlockId,
    setBlock: (x: int, y: int, z: int, block: BlockId) => void,
    getLightLevel: (x: int, y: int, z: int) => int,
    setPointLight: (x: int, y: int, z: int, level: int) => void,

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

  initializeWorld: (chunkRadius: int, frontierRadius: int, frontierLevels: int) => void;
  recenterWorld: (x: int, z: int) => void;
  remeshWorld: () => void;

  getBlock: (x: int, y: int, z: int) => BlockId;
  setBlock: (x: int, y: int, z: int, block: BlockId) => void;
  getLightLevel: (x: int, y: int, z: int) => int;
  setPointLight: (x: int, y: int, z: int, level: int) => void;

  // Bindings to call JavaScript from C++.

  instances: WasmHandle<Instance>;
  lights: WasmHandle<LightTexture>;
  meshes: WasmHandle<VoxelMesh>;
  renderer: Renderer | null = null;
  block_to_instance: (InstancedMesh | null)[];

  constructor(module: WasmModule) {
    this.module = module;
    this.initializeWorld = module.asm.initializeWorld;
    this.recenterWorld = module.asm.recenterWorld;
    this.remeshWorld = module.asm.remeshWorld;
    this.getBlock = module.asm.getBlock;
    this.setBlock = module.asm.setBlock;
    this.getLightLevel = module.asm.getLightLevel;
    this.setPointLight = module.asm.setPointLight;

    this.instances = new WasmHandle();
    this.lights = new WasmHandle();
    this.meshes = new WasmHandle();
    this.block_to_instance = [];
  }
};

let loaded = false;
let helper: WasmHelper | null = null;
let on_start_callbacks: (() => void)[] = [];

const checkReady = () => {
  if (!(loaded && helper)) return;
  on_start_callbacks.forEach(x => x());
};

const js_AddLightTexture = (data: int, size: int) => {
  const h = nonnull(helper);
  const r = nonnull(h.renderer);
  const buffer = h.module.HEAPU8.subarray(data, data + size);
  return h.lights.allocate(r.addLightTexture(buffer));
};

const js_FreeLightTexture = (handle: int) => {
  nonnull(helper).lights.free(handle).dispose();
};

const js_AddInstancedMesh = (block: BlockId, x: int, y: int, z: int) => {
  const h = nonnull(helper);
  const instance = nonnull(h.block_to_instance[block]).addInstance();
  instance.setPosition(x + 0.5, y, z + 0.5);
  return h.instances.allocate(instance);
};

const js_FreeInstancedMesh = (handle: int): void => {
  nonnull(helper).instances.free(handle).dispose();
};

const js_SetInstancedMeshLight = (handle: int, level: int) => {
  const h = nonnull(helper);
  h.instances.get(handle).setLight(lighting(level));
};

const js_AddVoxelMesh = (data: int, size: int, phase: int) => {
  const h = nonnull(helper);
  const r = nonnull(h.renderer);
  const offset = data >> 2;
  const buffer = h.module.HEAP32.slice(offset, offset + size);
  const geo = new Geometry(buffer, int(size / Geometry.StrideInInt32));
  return h.meshes.allocate(r.addVoxelMesh(geo, phase));
};

const js_FreeVoxelMesh = (handle: int): void => {
  nonnull(helper).meshes.free(handle).dispose();
};

const js_AddVoxelMeshGeometry = (handle: int, data: int, size: int): void => {
  const h = nonnull(helper);
  const mesh = h.meshes.get(handle);
  const geo = mesh.getGeometry();
  const old_num_quads = geo.num_quads;

  const offset = data >> 2;
  const buffer = h.module.HEAP32.subarray(offset, offset + size);
  geo.allocateQuads(int(old_num_quads + (size / Geometry.StrideInInt32)));
  geo.quads.set(buffer, old_num_quads * Geometry.StrideInInt32);
  geo.dirty = true;

  mesh.setGeometry(geo);
};

const js_SetVoxelMeshGeometry = (handle: int, data: int, size: int): void => {
  const h = nonnull(helper);
  const offset = data >> 2;
  const buffer = h.module.HEAP32.slice(offset, offset + size);
  const geo = new Geometry(buffer, int(size / Geometry.StrideInInt32));
  h.meshes.get(handle).setGeometry(geo);
};

const js_SetVoxelMeshLight = (handle: int, texture: int) => {
  const h = nonnull(helper);
  h.meshes.get(handle).setLight(h.lights.get(texture));
};

const js_SetVoxelMeshMask = (handle: int, m0: int, m1: int, shown: int) => {
  const h = nonnull(helper);
  h.meshes.get(handle).show(m0, m1, !!shown);
};

const js_SetVoxelMeshPosition = (handle: int, x: int, y: int, z: int): void => {
  nonnull(helper).meshes.get(handle).setPosition(x, y, z);
};

const init = (fn: () => void) => on_start_callbacks.push(fn);

window.onload = () => { loaded = true; checkReady(); };
(window as any).beforeWasmCompile = (env: any) => {
  env.js_AddLightTexture  = js_AddLightTexture;
  env.js_FreeLightTexture = js_FreeLightTexture;
  env.js_AddInstancedMesh      = js_AddInstancedMesh;
  env.js_FreeInstancedMesh     = js_FreeInstancedMesh;
  env.js_SetInstancedMeshLight = js_SetInstancedMeshLight;
  env.js_AddVoxelMesh  = js_AddVoxelMesh;
  env.js_FreeVoxelMesh = js_FreeVoxelMesh;
  env.js_AddVoxelMeshGeometry = js_AddVoxelMeshGeometry;
  env.js_SetVoxelMeshGeometry = js_SetVoxelMeshGeometry;
  env.js_SetVoxelMeshLight    = js_SetVoxelMeshLight;
  env.js_SetVoxelMeshMask     = js_SetVoxelMeshMask;
  env.js_SetVoxelMeshPosition = js_SetVoxelMeshPosition;
};
(window as any).onWasmCompile =
  (m: WasmModule) => { helper = new WasmHelper(m); checkReady(); };

//////////////////////////////////////////////////////////////////////////////

export {BlockId, MaterialId, Env, init};
export {kChunkWidth, kEmptyBlock, kNoMaterial, kWorldHeight};
