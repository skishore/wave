import {assert, drop, int, nonnull, Tensor3, Vec3} from './base.js';
import {ECS} from './ecs.js';

//////////////////////////////////////////////////////////////////////////////
// The game engine:

const Constants = {
  CHUNK_KEY_BITS: 8,
  TICK_RESOLUTION: 4,
  TICKS_PER_FRAME: 4,
  TICKS_PER_SECOND: 30,
  CAMERA_SENSITIVITY: 10,
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
    element.addEventListener('keydown', e => this.onKeyInput(e, true));
    element.addEventListener('keyup', e => this.onKeyInput(e, false));

    element.addEventListener('click', () => element.requestPointerLock());
    document.addEventListener('pointerlockchange', e => this.onPointerInput(e));
    document.addEventListener('mousemove', e => this.onMouseMove(e));
    document.addEventListener('wheel', e => this.onMouseWheel(e));
  }

  onKeyInput(e: Event, down: boolean) {
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

type Color = [number, number, number];

interface Material {
  alpha: number,
  color: Color,
  texture: string | null,
  textureAlpha: boolean,
};

const kBlack: Color = [0, 0, 0];
const kWhite: Color = [1, 1, 1];

const kNoMaterial = 0 as MaterialId;

const kEmptyBlock = 0 as BlockId;
const kUnknownBlock = 1 as BlockId;

class Registry {
  _opaque: boolean[];
  _solid: boolean[];
  _faces: MaterialId[];
  _meshes: (BABYLON.Mesh | null)[];
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

  addBlockSprite(mesh: BABYLON.Mesh, solid: boolean): BlockId {
    const result = this._opaque.length as BlockId;
    this._opaque.push(false);
    this._solid.push(solid);
    this._meshes.push(mesh);
    for (let i = 0; i < 6; i++) this._faces.push(kNoMaterial);
    mesh.setEnabled(false);
    return result;
  }

  addMaterialOfColor(name: string, color: Color, alpha: number = 1.0) {
    this.addMaterialHelper(name, alpha, color, null, false);
  }

  addMaterialOfTexture(name: string, texture: string,
                       textureAlpha: boolean = false) {
    this.addMaterialHelper(name, 1, kWhite, texture, textureAlpha);
  }

  // faces has 6 elements for each block type: [+x, -x, +y, -y, +z, -z]
  getBlockFaceMaterial(id: BlockId, face: int): MaterialId {
    return this._faces[id * 6 + face];
  }

  getMaterial(id: MaterialId): Material {
    assert(0 < id && id <= this._materials.length);
    return this._materials[id - 1];
  }

  private addMaterialHelper(name: string, alpha: number, color: Color,
                            texture: string | null, textureAlpha: boolean) {
    assert(name.length > 0, () => 'Empty material name!');
    assert(!this._ids.has(name), () => `Duplicate material: ${name}`);
    this._ids.set(name, this._materials.length as MaterialId);
    this._materials.push({alpha, color, texture, textureAlpha});
  }
};

//////////////////////////////////////////////////////////////////////////////

const kTmpPos = Vec3.create();

class Camera {
  camera: BABYLON.FreeCamera;
  holder: BABYLON.TransformNode;
  direction: Vec3;
  heading: number; // In radians: [0, 2π)
  pitch: number;   // In radians: (-π/2, π/2)
  zoom: number;
  last_dx: number;
  last_dy: number;

  constructor(scene: BABYLON.Scene) {
    const origin = new BABYLON.Vector3(0, 0, 0);
    this.holder = new BABYLON.TransformNode('holder', scene);
    this.camera = new BABYLON.FreeCamera('camera', origin, scene);
    this.camera.parent = this.holder;
    this.camera.minZ = 0.01;

    this.pitch = 0;
    this.heading = 0;
    this.zoom = 0;
    this.direction = Vec3.create();

    this.last_dx = 0;
    this.last_dy = 0;
  }

  applyInputs(dx: number, dy: number, dscroll: number) {
    // Smooth out large mouse-move inputs.
    const jerkx = Math.abs(dx) > 400 && Math.abs(dx / (this.last_dx || 1)) > 4;
    const jerky = Math.abs(dy) > 400 && Math.abs(dy / (this.last_dy || 1)) > 4;
    if (jerkx || jerky) {
      const saved_x = this.last_dx;
      const saved_y = this.last_dy;
      this.last_dx = (dx + this.last_dx) / 2;
      this.last_dy = (dy + this.last_dy) / 2;
      dx = saved_x;
      dy = saved_y;
    } else {
      this.last_dx = dx;
      this.last_dy = dy;
    }

    let pitch = this.holder.rotation.x;
    let heading = this.holder.rotation.y;

    // Overwatch uses the same constant values to do this conversion.
    const conversion = 0.0066 * Math.PI / 180;
    dx = dx * Constants.CAMERA_SENSITIVITY * conversion;
    dy = dy * Constants.CAMERA_SENSITIVITY * conversion;

    this.heading += dx;
    const T = 2 * Math.PI;
    while (this.heading < 0) this.heading += T;
    while (this.heading > T) this.heading -= T;

    const U = Math.PI / 2 - 0.01;
    this.pitch = Math.max(-U, Math.min(U, this.pitch + dy));

    this.holder.rotation.x = this.pitch;
    this.holder.rotation.y = this.heading;

    const dir = this.direction;
    Vec3.set(dir, 0, 0, 1);
    Vec3.rotateX(dir, dir, this.pitch);
    Vec3.rotateY(dir, dir, this.heading);

    // Scrolling is trivial to apply: add and clamp.
    if (dscroll === 0) return;
    this.zoom = Math.max(0, Math.min(10, this.zoom + Math.sign(dscroll)));
  }

  setTarget(x: number, y: number, z: number) {
    Vec3.set(kTmpPos, x, y, z);
    Vec3.scaleAndAdd(kTmpPos, kTmpPos, this.direction, -this.zoom);
    this.holder.position.copyFromFloats(kTmpPos[0], kTmpPos[1], kTmpPos[2]);
  }
};

//////////////////////////////////////////////////////////////////////////////

class Renderer {
  camera: Camera;
  engine: BABYLON.Engine;
  light: BABYLON.Light;
  scene: BABYLON.Scene;

  constructor(container: Container) {
    const antialias = true;
    const options = {preserveDrawingBuffer: true};
    this.engine = new BABYLON.Engine(container.canvas, antialias, options);
    this.scene = new BABYLON.Scene(this.engine);

    const source = new BABYLON.Vector3(0.1, 1.0, 0.3);
    this.light = new BABYLON.HemisphericLight('light', source, this.scene);
    this.scene.clearColor = new BABYLON.Color4(0.8, 0.9, 1.0);
    this.scene.ambientColor = new BABYLON.Color3(1, 1, 1);
    this.light.diffuse = new BABYLON.Color3(1, 1, 1);
    this.light.specular = new BABYLON.Color3(1, 1, 1);

    const scene = this.scene;
    scene.detachControl();
    scene.skipPointerMovePicking = true;
    this.camera = new Camera(scene);
  }

  makeSprite(url: string): BABYLON.Mesh {
    const scene = this.scene;
    const wrap = BABYLON.Texture.CLAMP_ADDRESSMODE;
    const mode = BABYLON.Texture.NEAREST_NEAREST_MIPNEAREST;
    const texture = new BABYLON.Texture(url, scene, false, true, mode);
    texture.wrapU = texture.wrapV = wrap;
    texture.hasAlpha = true;

    const material = new BABYLON.StandardMaterial(`material-${url}`, scene);
    material.specularColor.copyFromFloats(0, 0, 0);
    material.emissiveColor.copyFromFloats(1, 1, 1);
    material.backFaceCulling = false;
    material.diffuseTexture = texture;

    const mesh = BABYLON.Mesh.CreatePlane(`block-${url}`, 1, scene);
    mesh.cullingStrategy = BABYLON.AbstractMesh.CULLINGSTRATEGY_STANDARD;
    mesh.material = material;
    return mesh;
  }

  makeStandardMaterial(name: string): BABYLON.StandardMaterial {
    const result = new BABYLON.StandardMaterial(name, this.scene);
    result.specularColor.copyFromFloats(0, 0, 0);
    result.ambientColor.copyFromFloats(1, 1, 1);
    result.diffuseColor.copyFromFloats(1, 1, 1);
    return result;
  }

  render() {
    this.engine.beginFrame();
    this.scene.render();
    this.engine.endFrame();
  }

  startInstrumentation() {
    const perf = new BABYLON.SceneInstrumentation(this.scene);
    perf.captureActiveMeshesEvaluationTime = true;
    perf.captureRenderTargetsRenderTime = true;
    perf.captureCameraRenderTime = true;
    perf.captureRenderTime = true;
    let frame = 0;

    this.scene.onAfterRenderObservable.add(() => {
      frame = (frame + 1) % 60;
      if (frame !== 0) return;

      console.log(`
activeMeshesEvaluationTime: ${perf.activeMeshesEvaluationTimeCounter.average}
   renderTargetsRenderTime: ${perf.renderTargetsRenderTimeCounter.average}
          cameraRenderTime: ${perf.cameraRenderTimeCounter.average}
          drawCallsCounter: ${perf.drawCallsCounter.lastSecAverage}
                renderTime: ${perf.renderTimeCounter.average}
      `.trim());
    });
  }
};

//////////////////////////////////////////////////////////////////////////////

declare const NoaTerrainMesher: any;

class TerrainMesher {
  mesher: any;
  flatMaterial: BABYLON.Material;
  registry: Registry;
  requests: int;
  world: World;

  constructor(world: World) {
    const registry = world.registry;
    const renderer = world.renderer;
    this.flatMaterial = renderer.makeStandardMaterial('flat-material');
    this.flatMaterial.freeze();
    this.registry = registry;
    this.requests = 0;
    this.world = world;

    const shim = {
      registry: {
        _solidityLookup: registry._solid,
        _opacityLookup: registry._opaque,
        getBlockFaceMaterial: registry.getBlockFaceMaterial.bind(registry),
        getMaterialData: (x: MaterialId) => registry.getMaterial(x),
        getMaterialTexture: (x: MaterialId) => registry.getMaterial(x).texture,
        _getMaterialVertexColor: (x: MaterialId) => registry.getMaterial(x).color,
      },
      rendering: {
        useAO: true,
        aoVals: [0.93, 0.8, 0.5],
        revAoVal: 1.0,
        flatMaterial: this.flatMaterial,
        addMeshToScene: () => {},
        makeStandardMaterial: renderer.makeStandardMaterial.bind(renderer),
        getScene: () => renderer.scene,
      },
    };
    this.mesher = new NoaTerrainMesher(shim);
  }

  mesh(cx: int, cy: int, cz: int, voxels: Tensor3): BABYLON.Mesh | null {
    const requestID = this.requests++;
    const meshes: BABYLON.Mesh[] = [];
    const chunk = {
      voxels,
      requestID,
      pos: null,
      _isFull: false,
      _isEmpty: false,
      _terrainMeshes: meshes,
      _neighbors: {get: (x: int, y: int, z: int) => {
        const chunk = this.world.getChunk(x + cx, y + cy, z + cz, false);
        return chunk && chunk.voxels ? chunk : null;
      }},
    };

    this.mesher.meshChunk(chunk);
    assert(meshes.length <= 1, () => `Unexpected: ${meshes.length} meshes`);
    return meshes.length === 1 ? meshes[0] : null;
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

interface TerrainSprite {
  dirty: boolean;
  mesh: BABYLON.Mesh;
  buffer: Float32Array;
  index: Map<int, int>;
  capacity: int;
  size: int;
};

const kMinCapacity = 4;
const kSpriteRadius = 1 / 2 + 1 / 256;

const kTmpBillboard = new Float32Array([
  0, -kSpriteRadius, 0,
  0, -kSpriteRadius, 0,
  0,  kSpriteRadius, 0,
  0,  kSpriteRadius, 0,
]);

const kTmpTransform = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

const kSpriteKeyBits = 10;
const kSpriteKeySize = 1 << kSpriteKeyBits;
const kSpriteKeyMask = kSpriteKeySize - 1;

class TerrainSprites {
  renderer: Renderer;
  kinds: Map<BlockId, TerrainSprite>;
  root: BABYLON.TransformNode;
  billboards: BABYLON.Mesh[];

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.kinds = new Map();
    this.root = new BABYLON.TransformNode('sprites', renderer.scene);
    this.root.position.copyFromFloats(0.5, 0.5, 0.5);
    this.billboards = [];
  }

  add(x: int, y: int, z: int, block: BlockId, mesh: BABYLON.Mesh) {
    let data = this.kinds.get(block);
    if (!data) {
      const capacity = kMinCapacity;
      const buffer = new Float32Array(capacity * 16);
      data = {dirty: false, mesh, buffer, index: new Map(), capacity, size: 0};
      this.kinds.set(block, data);

      mesh.parent = this.root;
      mesh.position.setAll(0);
      mesh.alwaysSelectAsActiveMesh = true;
      mesh.doNotSyncBoundingInfo = true;
      mesh.freezeWorldMatrix();
      mesh.thinInstanceSetBuffer('matrix', buffer);
      mesh.setVerticesData('position', kTmpBillboard, true);
      if (mesh.material) mesh.material.freeze();
    }

    const key = this.key(x, y, z);
    if (data.index.has(key)) return;

    if (data.size === data.capacity) {
      this.reallocate(data, data.capacity * 2);
    }

    kTmpTransform[12] = x;
    kTmpTransform[13] = y;
    kTmpTransform[14] = z;
    this.copy(kTmpTransform, 0, data.buffer, data.size);
    data.index.set(key, data.size);

    data.size++;
    data.dirty = true;
  }

  remove(x: int, y: int, z: int, block: BlockId) {
    const data = this.kinds.get(block);
    if (!data) return;

    const buffer = data.buffer;
    const key = this.key(x, y, z);
    const index = data.index.get(key);
    if (index === undefined) return;

    const last = data.size - 1;
    if (index !== last) {
      const b = 16 * last + 12;
      const other = this.key(buffer[b + 0], buffer[b + 1], buffer[b + 2]);
      assert(data.index.get(other) === last);
      this.copy(buffer, last, buffer, index);
      data.index.set(other, index);
    }
    data.index.delete(key);

    data.size--;
    if (data.capacity > Math.max(kMinCapacity, 4 * data.size)) {
      this.reallocate(data, data.capacity / 2);
    }
    data.dirty = true;
  }

  registerBillboard(mesh: BABYLON.Mesh) {
    this.billboards.push(mesh);
    mesh.setVerticesData('position', kTmpBillboard, true);
  }

  unregisterBillboard(mesh: BABYLON.Mesh) {
    drop(this.billboards, mesh);
  }

  updateBillboards(heading: number) {
    const cos = kSpriteRadius * Math.cos(heading);
    const sin = kSpriteRadius * Math.sin(heading);

    kTmpBillboard[0] = kTmpBillboard[9]  = -cos;
    kTmpBillboard[2] = kTmpBillboard[11] = sin;
    kTmpBillboard[3] = kTmpBillboard[6]  = cos;
    kTmpBillboard[5] = kTmpBillboard[8]  = -sin;

    for (const mesh of this.billboards) {
      const geo = mesh.geometry;
      if (geo) geo.updateVerticesDataDirectly('position', kTmpBillboard, 0);
    }

    for (const data of this.kinds.values()) {
      if (data.size !== 0) {
        const geo = data.mesh.geometry;
        if (geo) geo.updateVerticesDataDirectly('position', kTmpBillboard, 0);
      }
      if (!data.dirty) continue;
      data.mesh.thinInstanceCount = data.size;
      data.mesh.thinInstanceBufferUpdated('matrix');
      data.mesh.setEnabled(data.size > 0);
      data.dirty = false;
    }
  }

  private copy(src: Float32Array, srcOff: int, dst: Float32Array, dstOff: int) {
    srcOff *= 16;
    dstOff *= 16;
    for (let i = 0; i < 16; i++) {
      dst[dstOff + i] = src[srcOff + i];
    }
  }

  private key(x: int, y: int, z: int): int {
    return (x & kSpriteKeyMask) << (0 * kSpriteKeyBits) |
           (y & kSpriteKeyMask) << (1 * kSpriteKeyBits) |
           (z & kSpriteKeyMask) << (2 * kSpriteKeyBits);
  }

  private reallocate(data: TerrainSprite, capacity: int) {
    data.capacity = capacity;
    const buffer = new Float32Array(capacity * 16);
    for (let i = 0; i < data.size * 16; i++) buffer[i] = data.buffer[i];
    data.mesh.thinInstanceSetBuffer('matrix', buffer);
    data.buffer = buffer;
  }
};

//////////////////////////////////////////////////////////////////////////////

const kChunkBits = 6;
const kChunkSize = 1 << kChunkBits;
const kChunkMask = kChunkSize - 1;

const kChunkKeyBits = 8;
const kChunkKeySize = 1 << kChunkKeyBits;
const kChunkKeyMask = kChunkKeySize - 1;

const kChunkRadius = 4;
const kNeighbors = (kChunkRadius ? 6 : 0);

const kNumChunksToLoadPerFrame = 1;
const kNumChunksToMeshPerFrame = 1;

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
  mesh: BABYLON.Mesh | null;
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
      if (old_mesh) this.world.sprites.remove(x, y, z, old);
      if (new_mesh) this.world.sprites.add(x, y, z, block, new_mesh);
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

    const sprites = this.world.sprites;
    for (let x = 0; x < kChunkSize; x++) {
      for (let y = 0; y < kChunkSize; y++) {
        for (let z = 0; z < kChunkSize; z++) {
          const cell = this.voxels.get(x, y, z) as BlockId;
          const mesh = this.world.registry._meshes[cell];
          if (!mesh) continue;
          enabled ? sprites.add(x + dx, y + dy, z + dz, cell, mesh)
                  : sprites.remove(x + dx, y + dy, z + dz, cell);
        }
      }
    }
  }

  private refreshTerrain() {
    if (this.mesh) this.mesh.dispose();
    const {cx, cy, cz, voxels} = this;
    const mesh = voxels ? this.world.mesher.mesh(cx, cy, cz, voxels) : null;
    if (mesh) {
      mesh.position.copyFromFloats(
        cx << kChunkBits, cy << kChunkBits, cz << kChunkBits);
      mesh.cullingStrategy = BABYLON.AbstractMesh.CULLINGSTRATEGY_STANDARD;
      mesh.doNotSyncBoundingInfo = true;
      mesh.freezeWorldMatrix();
      mesh.freezeNormals();
    }
    this.mesh = mesh;
  }
};

class World {
  chunks: Map<int, Chunk>;
  enabled: Set<Chunk>;
  renderer: Renderer;
  registry: Registry;
  mesher: TerrainMesher;
  sprites: TerrainSprites;

  constructor(registry: Registry, renderer: Renderer) {
    this.chunks = new Map();
    this.enabled = new Set();
    this.renderer = renderer;
    this.registry = registry;
    this.mesher = new TerrainMesher(this);
    this.sprites = new TerrainSprites(renderer);
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
  entities: ECS;
  registry: Registry;
  renderer: Renderer;
  timing: Timing;
  world: World;

  constructor(id: string) {
    this.container = new Container(id);
    this.entities = new ECS();
    this.registry = new Registry();
    this.renderer = new Renderer(this.container);
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

    this.world.sprites.updateBillboards(camera.heading);
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
