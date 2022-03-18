import {sweep} from './sweep.js';

//////////////////////////////////////////////////////////////////////////////
// Utilities and math:

type int = number;

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

//////////////////////////////////////////////////////////////////////////////

type Vec3 = [number, number, number];

const Vec3 = {
  create: (): Vec3 => [0, 0, 0],
  from: (x: number, y: number, z: number): Vec3 => [x, y, z],
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

class Tensor3 {
  data: Uint32Array;
  shape: [int, int, int];
  stride: [int, int, int];

  constructor(x: int, y: int, z: int) {
    this.data = new Uint32Array(x * y * z);
    this.shape = [x, y, z];
    this.stride = [1, x, x * y];
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
    const mode = BABYLON.Texture.NEAREST_SAMPLINGMODE;
    const wrap = BABYLON.Texture.CLAMP_ADDRESSMODE;
    const texture = new BABYLON.Texture(url, scene, true, true, mode);
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

type EntityId = int & {__type__: 'EntityId'};

interface ComponentState {id: EntityId, index: int};

const kNoEntity: EntityId = 0 as EntityId;

interface Component<T extends ComponentState = ComponentState> {
  init: () => T,
  order?: number,
  onAdd?: (state: T) => void,
  onRemove?: (state: T) => void,
  onRender?: (dt: number, states: T[]) => void,
  onUpdate?: (dt: number, states: T[]) => void,
};

class ComponentStore<T extends ComponentState = ComponentState> {
  component: string;
  definition: Component<T>;
  lookup: Map<EntityId, T>;
  states: T[];

  constructor(component: string, definition: Component<T>) {
    this.component = component;
    this.definition = definition;
    this.lookup = new Map();
    this.states = [];
  }

  get(entity: EntityId): T | null {
    const result = this.lookup.get(entity);
    return result ? result : null;
  }

  getX(entity: EntityId): T {
    const result = this.lookup.get(entity);
    if (!result) throw new Error(`${entity} missing ${this.component}`);
    return result;
  }

  add(entity: EntityId): T {
    if (this.lookup.has(entity)) {
      throw new Error(`Duplicate for ${entity}: ${this.component}`);
    }

    const index = this.states.length;
    const state = this.definition.init();
    state.id = entity;
    state.index = index;

    this.lookup.set(entity, state);
    this.states.push(state);

    const callback = this.definition.onAdd;
    if (callback) callback(state);
    return state;
  }

  remove(entity: EntityId) {
    const state = this.lookup.get(entity);
    if (!state) return;

    this.lookup.delete(entity);
    const popped = this.states.pop() as T;
    assert(popped.index === this.states.length);
    if (popped.id === entity) return;

    const index = state.index;
    assert(index < this.states.length);
    this.states[index] = popped;
    popped.index = index;

    const callback = this.definition.onRemove;
    if (callback) callback(state);
  }

  render(dt: int) {
    const callback = this.definition.onRender;
    if (!callback) throw new Error(`render called: ${this.component}`);
    callback(dt, this.states);
  }

  update(dt: int) {
    const callback = this.definition.onUpdate;
    if (!callback) throw new Error(`update called: ${this.component}`);
    callback(dt, this.states);
  }
};

class EntityComponentSystem {
  last: EntityId;
  components: Map<string, ComponentStore<any>>;
  onRenders: ComponentStore<any>[];
  onUpdates: ComponentStore<any>[];

  constructor() {
    this.last = 0 as EntityId;
    this.components = new Map();
    this.onRenders = [];
    this.onUpdates = [];
  }

  addEntity(): EntityId {
    return this.last = (this.last + 1) as EntityId;
  }

  removeEntity(entity: EntityId) {
    this.components.forEach(x => x.remove(entity));
  }

  registerComponent<T extends ComponentState>(
      component: string, definition: Component<T>): ComponentStore<T> {
    const exists = this.components.has(component);
    if (exists) throw new Error(`Duplicate component: ${component}`);
    const store = new ComponentStore(component, definition);
    this.components.set(component, store);

    if (definition.onRender) this.onRenders.push(store);
    if (definition.onUpdate) this.onUpdates.push(store);
    return store;
  }

  render(dt: int) {
    for (const store of this.onRenders) store.render(dt);
  }

  update(dt: int) {
    for (const store of this.onUpdates) store.update(dt);
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

  update(heading: number) {
    const cos = kSpriteRadius * Math.cos(heading);
    const sin = kSpriteRadius * Math.sin(heading);

    kTmpBillboard[0] = kTmpBillboard[9]  = -cos;
    kTmpBillboard[2] = kTmpBillboard[11] = sin;
    kTmpBillboard[3] = kTmpBillboard[6]  = cos;
    kTmpBillboard[5] = kTmpBillboard[8]  = -sin;

    for (const mesh of this.billboards) {
      mesh.setVerticesData('position', kTmpBillboard);
    }

    for (const data of this.kinds.values()) {
      if (data.size !== 0) {
        data.mesh.setVerticesData('position', kTmpBillboard);
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

const kChunkBits = 5;
const kChunkSize = 1 << kChunkBits;
const kChunkMask = kChunkSize - 1;

const kChunkKeyBits = 8;
const kChunkKeySize = 1 << kChunkKeyBits;
const kChunkKeyMask = kChunkKeySize - 1;

const kChunkRadiusX = 8;
const kChunkRadiusY = 0;
const kNeighbors = (kChunkRadiusX ? 4 : 0) + (kChunkRadiusY ? 2 : 0);

const kNumChunksToLoadPerFrame = 1;
const kNumChunksToMeshPerFrame = 1;

// These conditions ensure that we'll dispose of a sprite before allocating
// a new sprite at a key that collides with the old one.
assert((1 << kSpriteKeyBits) > (kChunkSize * (2 * kChunkRadiusX + 1)));
assert((1 << kSpriteKeyBits) > (kChunkSize * (2 * kChunkRadiusY + 1)));

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
    if (!this.voxels) return kUnknownBlock;
    return this.voxels.get(x & mask, y & mask, z & mask) as BlockId;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    const mask = kChunkMask;
    const size = kChunkSize;
    if (!this.voxels) this.voxels = new Tensor3(size, size, size);
    const old = this.voxels.get(x & mask, y & mask, z & mask) as BlockId;
    if (old === block) return;

    const old_mesh = this.world.registry._meshes[old];
    const new_mesh = this.world.registry._meshes[block];
    if (!this.spritesDirty) {
      if (old_mesh) this.world.sprites.remove(x, y, z, old);
      if (new_mesh) this.world.sprites.add(x, y, z, block, new_mesh);
    }

    this.voxels.set(x & mask, y & mask, z & mask, block);
    this.terrainDirty ||= !(old_mesh || old === 0) ||
                          !(new_mesh || block === 0);
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
    return chunk && chunk.active ? chunk.getBlock(x, y, z) : kUnknownBlock;
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

    const lo = (kChunkRadiusX * kChunkRadiusX + 1) * kChunkSize * kChunkSize;
    const hi = (kChunkRadiusX * kChunkRadiusX + 9) * kChunkSize * kChunkSize;

    const disabled = [];
    for (const chunk of this.enabled) {
      const {cx, cy, cz} = chunk;
      const disable = Math.abs(cx - dx) > kChunkRadiusX + 1 ||
                      Math.abs(cy - dy) > kChunkRadiusY + 1 ||
                      Math.abs(cz - dz) > kChunkRadiusX + 1 ||
                      this.distance(cx, cy, cz, x, y, z) > hi;
      if (disable) disabled.push(chunk);
    }
    for (const chunk of disabled) chunk.disable();

    const result = [];
    for (let i = dx - kChunkRadiusX; i <= dx + kChunkRadiusX; i++) {
      for (let j = dy - kChunkRadiusY; j <= dy + kChunkRadiusY; j++) {
        for (let k = dz - kChunkRadiusX; k <= dz + kChunkRadiusX; k++) {
          if (this.distance(i, j, k, x, y, z) > lo) continue;
          const chunk = nonnull(this.getChunk(i, j, k, true));
          chunk.enable();

          if (chunk.requested) continue;
          result.push(chunk);
          chunk.requested = true;
          if (result.length === kNumChunksToLoadPerFrame) break;
        }
      }
    }
    return result;
  }

  remesh() {
    const queued = [];
    for (const chunk of this.chunks.values()) {
      if (chunk.needsRemesh()) queued.push(chunk);
      if (queued.length === kNumChunksToMeshPerFrame) break;
    }
    for (const chunk of queued) chunk.remesh();
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

    this.world.sprites.update(camera.heading);
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
// The game code:

class TypedEnv extends Env {
  position: ComponentStore<PositionState>;
  movement: ComponentStore<MovementState>;
  physics: ComponentStore<PhysicsState>;
  mesh: ComponentStore<MeshState>;
  shadow: ComponentStore<ShadowState>;
  target: ComponentStore;
  center: ComponentStore;

  constructor(id: string) {
    super(id);
    const ents = this.entities;
    this.position = ents.registerComponent('position', Position);
    this.movement = ents.registerComponent('movement', Movement(this));
    this.physics = ents.registerComponent('physics', Physics(this));
    this.mesh = ents.registerComponent('mesh', Mesh(this));
    this.shadow = ents.registerComponent('shadow', Shadow(this));
    this.target = ents.registerComponent('camera-target', CameraTarget(this));
    this.center= ents.registerComponent('recenter-world', RecenterWorld(this));
  }
};

// An entity with a position is an axis-aligned bounding box (AABB) centered
// at (x, y, z), with x- and z-extents equal to w and y-extent equal to h.

interface PositionState {
  id: EntityId,
  index: int,
  x: number,
  y: number,
  z: number,
  h: number,
  w: number,
};

const Position: Component<PositionState> = {
  init: () => ({id: kNoEntity, index: 0, x: 0, y: 0, z: 0, h: 0, w: 0}),
};

// An entity's physics state tracks its location and velocity, and allows
// other systems to apply forces and impulses to it. It updates the entity's
// AABB and keeps its position in sync.

interface PhysicsState {
  id: EntityId,
  index: int,
  min: Vec3,
  max: Vec3,
  vel: Vec3,
  forces: Vec3,
  impulses: Vec3,
  resting: Vec3,
  friction: number,
  mass: number,
};

const kTmpGravity = Vec3.from(0, -40, 0);
const kTmpAcceleration = Vec3.create();
const kTmpFriction = Vec3.create();
const kTmpDelta = Vec3.create();
const kTmpSize = Vec3.create();
const kTmpPush = Vec3.create();
const kTmpPos = Vec3.create();

const setPhysicsFromPosition = (a: PositionState, b: PhysicsState) => {
  Vec3.set(kTmpPos, a.x, a.y, a.z);
  Vec3.set(kTmpSize, a.w / 2, a.h / 2, a.w / 2);
  Vec3.sub(b.min, kTmpPos, kTmpSize);
  Vec3.add(b.max, kTmpPos, kTmpSize);
};

const setPositionFromPhysics = (a: PositionState, b: PhysicsState) => {
  a.x = (b.min[0] + b.max[0]) / 2;
  a.y = (b.min[1] + b.max[1]) / 2;
  a.z = (b.min[2] + b.max[2]) / 2;
};

const applyFriction = (axis: int, state: PhysicsState, dv: Vec3) => {
  const resting = state.resting[axis];
  if (resting === 0 || resting * dv[axis] <= 0) return;

  Vec3.copy(kTmpFriction, state.vel);
  kTmpFriction[axis] = 0;
  const length = Vec3.length(kTmpFriction);
  if (length === 0) return;

  const loss = Math.abs(state.friction * dv[axis]);
  const scale = length < loss ? 0 : (length - loss) / length;
  state.vel[(axis + 1) % 3] *= scale;
  state.vel[(axis + 2) % 3] *= scale;
};

const runPhysics = (env: TypedEnv, dt: int, state: PhysicsState) => {
  if (state.mass <= 0) return;

  dt = dt / 1000;
  Vec3.scale(kTmpAcceleration, state.forces, 1 / state.mass);
  Vec3.add(kTmpAcceleration, kTmpAcceleration, kTmpGravity);
  Vec3.scale(kTmpDelta, kTmpAcceleration, dt);
  Vec3.scaleAndAdd(kTmpDelta, kTmpDelta, state.impulses, 1 / state.mass);
  if (state.friction) {
    applyFriction(0, state, kTmpDelta);
    applyFriction(1, state, kTmpDelta);
    applyFriction(2, state, kTmpDelta);
  }

  // Update our state based on the computations above.
  Vec3.add(state.vel, state.vel, kTmpDelta);
  Vec3.scale(kTmpDelta, state.vel, dt);
  sweep(state.min, state.max, kTmpDelta, state.resting, (p: Vec3) => {
    const block = env.world.getBlock(p[0], p[1], p[2]);
    return !env.registry._solid[block];
  });
  Vec3.set(state.forces, 0, 0, 0);
  Vec3.set(state.impulses, 0, 0, 0);

  for (let i = 0; i < 3; i++) {
    if (state.resting[i] !== 0) state.vel[i] = 0;
  }
};

const Physics = (env: TypedEnv): Component<PhysicsState> => ({
  init: () => ({
    id: kNoEntity,
    index: 0,
    min: Vec3.create(),
    max: Vec3.create(),
    vel: Vec3.create(),
    forces: Vec3.create(),
    impulses: Vec3.create(),
    resting: Vec3.create(),
    friction: 0,
    mass: 1,
  }),
  onAdd: (state: PhysicsState) => {
    setPhysicsFromPosition(env.position.getX(state.id), state);
  },
  onRemove: (state: PhysicsState) => {
    setPositionFromPhysics(env.position.getX(state.id), state);
  },
  onRender: (dt: int, states: PhysicsState[]) => {
    for (const state of states) {
      setPositionFromPhysics(env.position.getX(state.id), state);
    }
  },
  onUpdate: (dt: int, states: PhysicsState[]) => {
    for (const state of states) runPhysics(env, dt, state);
  },
});

// Movement allows an entity to process inputs and attempt to move.

interface MovementState {
  id: EntityId,
  index: int,
  heading: number,
  running: boolean,
  jumping: boolean,
  maxSpeed: number,
  moveForce: number,
  responsiveness: number,
  runningFriction: number,
  standingFriction: number,
  airMoveMultiplier: number,
  airJumps: number,
  jumpTime: number,
  jumpForce: number,
  jumpImpulse: number,
  _jumped: boolean,
  _jumpCount: number,
  _jumpTimeLeft: number,
};

const handleJumping =
    (dt: int, state: MovementState, body: PhysicsState, grounded: boolean) => {
  if (state._jumped) {
    if (state._jumpTimeLeft <= 0) return;
    const delta = state._jumpTimeLeft <= dt ? state._jumpTimeLeft / dt : 1;
    const force = state.jumpForce * delta;
    Vec3.add(body.forces, body.forces, [0, force, 0]);
    return;
  }

  const hasAirJumps = state._jumpCount < state.airJumps;
  const canJump = grounded || hasAirJumps;
  if (!canJump) return;

  state._jumped = true;
  state._jumpTimeLeft = state.jumpTime;
  Vec3.add(body.impulses, body.impulses, [0, state.jumpImpulse, 0]);
  if (grounded) return;

  body.vel[1] = Math.max(body.vel[1], 0);
  state._jumpCount++;
};

const handleRunning =
    (dt: int, state: MovementState, body: PhysicsState, grounded: boolean) => {
  const speed = state.maxSpeed;
  Vec3.set(kTmpDelta, 0, 0, speed);
  Vec3.rotateY(kTmpDelta, kTmpDelta, state.heading);

  Vec3.sub(kTmpPush, kTmpDelta, body.vel);
  kTmpPush[1] = 0;
  const length = Vec3.length(kTmpPush);
  if (length === 0) return;

  const bound = state.moveForce * (grounded ? 1 : state.airMoveMultiplier);
  const input = state.responsiveness * length;
  Vec3.scale(kTmpPush, kTmpPush, Math.min(bound, input) / length);
  Vec3.add(body.forces, body.forces, kTmpPush);
};

const runMovement = (env: TypedEnv, dt: int, state: MovementState) => {
  dt = dt / 1000;

  // Process the inputs to get a heading, running, and jumping state.
  const inputs = env.container.inputs;
  const fb = (inputs.up ? 1 : 0) - (inputs.down ? 1 : 0);
  const lr = (inputs.right ? 1 : 0) - (inputs.left ? 1 : 0);
  state.running = fb !== 0 || lr !== 0;
  state.jumping = inputs.space;

  if (state.running) {
    let heading = env.renderer.camera.heading;
    if (fb) {
      if (fb === -1) heading += Math.PI;
      heading += fb * lr * Math.PI / 4;
    } else {
      heading += lr * Math.PI / 2;
    }
    state.heading = heading;
  }

  // All inputs processed; update the entity's PhysicsState.
  const body = env.physics.getX(state.id);
  const grounded = body.resting[1] < 0;
  if (grounded) state._jumpCount = 0;

  if (state.jumping) {
    handleJumping(dt, state, body, grounded);
  } else {
    state._jumped = false;
  }

  if (state.running) {
    handleRunning(dt, state, body, grounded);
    body.friction = state.runningFriction;
  } else {
    body.friction = state.standingFriction;
  }
};

const Movement = (env: TypedEnv): Component<MovementState> => ({
  init: () => ({
    id: kNoEntity,
    index: 0,
    heading: 0,
    running: false,
    jumping: false,
    maxSpeed: 10,
    moveForce: 30,
    responsiveness: 15,
    runningFriction: 0,
    standingFriction: 2,
    airMoveMultiplier: 0.5,
    airJumps: 9999,
    jumpTime: 500,
    jumpForce: 15,
    jumpImpulse: 10,
    _jumped: false,
    _jumpCount: 0,
    _jumpTimeLeft: 0,
  }),
  onUpdate: (dt: int, states: MovementState[]) => {
    for (const state of states) runMovement(env, dt, state);
  }
});

// Mesh signifies that an entity should be rendered with some provided mesh.

interface MeshState {
  id: EntityId,
  index: int,
  mesh: BABYLON.Mesh | null,
  texture: BABYLON.Texture | null,
  frame: number,
};

const setMesh = (env: TypedEnv, state: MeshState, mesh: BABYLON.Mesh) => {
  if (state.mesh) state.mesh.dispose();

  const billboards = env.world.sprites.billboards;
  mesh.onDisposeObservable.add(() => drop(billboards, mesh));
  billboards.push(mesh);

  const texture = (() => {
    const material = mesh.material;
    if (!(material instanceof BABYLON.StandardMaterial)) return null;
    const texture = (material as BABYLON.StandardMaterial).diffuseTexture;
    if (!(texture instanceof BABYLON.Texture)) return null;
    return texture as BABYLON.Texture;
  })();
  if (texture) {
    const fudge = 1 - 1 / 256;
    texture.uScale = fudge / 3;
    texture.vScale = fudge / 4;
    texture.vOffset = 0.75;
  }

  state.mesh = mesh;
  state.texture = texture;
};

const Mesh = (env: TypedEnv): Component<MeshState> => ({
  init: () => ({id: kNoEntity, index: 0, mesh: null, texture: null, frame: 0}),
  onRemove: (state: MeshState) => {
    if (state.mesh) state.mesh.dispose();
  },
  onRender: (dt: int, states: MeshState[]) => {
    for (const state of states) {
      if (!state.mesh) continue;
      const {x, y, z, h} = env.position.getX(state.id);
      const dy = (state.mesh.scaling.y - h) / 2;
      state.mesh.position.copyFromFloats(x, y + dy, z);
    }
  },
  onUpdate: (dt: int, states: MeshState[]) => {
    for (const state of states) {
      if (!state.texture) return;
      const body = env.physics.get(state.id);
      if (!body) return;

      const setting = (() => {
        if (!body.resting[1]) return 1;
        const speed = Vec3.length(body.vel);
        state.frame = speed ? (state.frame + 0.025 * speed) % 4 : 0;
        if (!speed) return 0;
        const value = Math.floor(state.frame);
        return value & 1 ? 0 : (value + 2) >> 1;
      })();
      state.texture.uOffset = state.texture.uScale * setting;
    }
  },
});

// Shadow places a shadow underneath the entity.

interface ShadowState {
  id: EntityId,
  index: int,
  mesh: BABYLON.Mesh | null,
  extent: number,
  height: number,
};

const Shadow = (env: TypedEnv): Component<ShadowState> => {
  const material = env.renderer.makeStandardMaterial('shadow-material');
  material.ambientColor.copyFromFloats(0, 0, 0);
  material.diffuseColor.copyFromFloats(0, 0, 0);
  material.alpha = 0.5;

  const scene = env.renderer.scene;
  const option = {radius: 1, tessellation: 16};
  const shadow = BABYLON.CreateDisc('shadow', option, scene);
  shadow.cullingStrategy = BABYLON.AbstractMesh.CULLINGSTRATEGY_STANDARD;
  scene.removeMesh(shadow);

  shadow.material = material;
  shadow.rotation.x = Math.PI / 2;
  shadow.setEnabled(false);

  return {
    init: () => ({id: kNoEntity, index: 0, mesh: null, extent: 8, height: 0}),
    onAdd: (state: ShadowState) => {
      const instance = shadow.createInstance('shadow-instance');
      state.mesh = instance as any as BABYLON.Mesh;
    },
    onRemove: (state: ShadowState) => {
      if (state.mesh) state.mesh.dispose();
    },
    onRender: (dt: int, states: ShadowState[]) => {
      for (const state of states) {
        if (!state.mesh) continue;
        const {x, y, z, w} = env.position.getX(state.id);
        state.mesh.position.copyFromFloats(x, state.height + 0.05, z);
        const fraction = 1 - (y - state.height) / state.extent;
        const scale = w * Math.max(0, Math.min(1, fraction)) / 2;
        state.mesh.scaling.copyFromFloats(scale, scale, scale);
      }
    },
    onUpdate: (dt: int, states: ShadowState[]) => {
      for (const state of states) {
        const position = env.position.getX(state.id);
        const x = Math.floor(position.x);
        const y = Math.floor(position.y);
        const z = Math.floor(position.z);
        state.height = (() => {
          for (let i = 0; i < state.extent; i++) {
            const h = y - i;
            if (env.world.getBlock(x, h - 1, z) !== kEmptyBlock) return h;
          }
          return y - state.extent;
        })();
      }
    },
  };
};

// CameraTarget signifies that the camera will follow an entity.

const CameraTarget = (env: TypedEnv): Component => ({
  init: () => ({id: kNoEntity, index: 0}),
  onRender: (dt: int, states: ComponentState[]) => {
    for (const state of states) {
      const {x, y, z, h} = env.position.getX(state.id);
      env.renderer.camera.setTarget(x, y + h / 3, z);
    }
  },
});

// RecenterWorld signifies that we'll load the world around an entity.

let loadChunkData = (chunk: Chunk) => {};

const RecenterWorld = (env: TypedEnv): Component => ({
  init: () => ({id: kNoEntity, index: 0}),
  onUpdate: (dt: int, states: ComponentState[]) => {
    for (const state of states) {
      const position = env.position.getX(state.id);
      const chunks = env.world.recenter(position.x, position.y, position.z);
      chunks.forEach(x => { loadChunkData(x); x.finish(); });
      break;
    }
  },
});

// Putting it all together:

const main = () => {
  const env = new TypedEnv('container');
  const sprite = (x: string) => env.renderer.makeSprite(`images/${x}.png`);
  //env.renderer.startInstrumentation();

  const player = env.entities.addEntity();
  const position = env.position.add(player);
  position.x = 2;
  position.y = 5;
  position.z = 2;
  position.w = 0.6;
  position.h = 0.8;

  env.physics.add(player);
  env.movement.add(player);
  env.shadow.add(player);
  env.target.add(player);
  env.center.add(player);

  const mesh = env.mesh.add(player);
  setMesh(env, mesh, sprite('player'));

  const registry = env.registry;
  const scene = env.renderer.scene;
  const textures = ['dirt', 'grass', 'ground', 'wall'];
  for (const texture of textures) {
    registry.addMaterialOfTexture(texture, `images/${texture}.png`);
  }
  const wall = registry.addBlock(['wall'], true);
  const grass = registry.addBlock(['grass', 'dirt', 'dirt'], true);
  const ground = registry.addBlock(['ground', 'dirt', 'dirt'], true);

  const rock = registry.addBlockSprite(sprite('rock'), true);
  const tree = registry.addBlockSprite(sprite('tree'), true);
  const tree0 = registry.addBlockSprite(sprite('tree0'), true);
  const tree1 = registry.addBlockSprite(sprite('tree1'), true);

  loadChunkData = (chunk: Chunk) => {
    if (chunk.cx < 0 || chunk.cz < 0) return null;
    if (chunk.cy !== 0) return null;

    const size = kChunkSize;
    const pl = size / 4;
    const pr = 3 * size / 4;
    const layers = [ground, ground, grass, wall, tree0, tree1];
    const dx = chunk.cx << kChunkBits;
    const dy = chunk.cy << kChunkBits;
    const dz = chunk.cz << kChunkBits;

    for (let x = 0; x < size; x++) {
      for (let z = 0; z < size; z++) {
        const edge = (chunk.cx === 0 && x === 0) ||
                     (chunk.cz === 0 && z === 0);
        const height = Math.min(edge ? layers.length : 3, size);
        for (let y = 0; y < height; y++) {
          chunk.setBlock(x + dx, y + dy, z + dz, layers[y]);
        }
        if (edge) continue;
        const test = Math.random();
        const limit = 0.05;
        if (test < 1 * limit) {
          chunk.setBlock(x + dx, 3 + dy, z + dz, rock);
        } else if (test < 2 * limit) {
          chunk.setBlock(x + dx, 3 + dy, z + dz, tree);
        } else if (test < 3 * limit) {
          chunk.setBlock(x + dx, 3 + dy, z + dz, wall);
        } else if (test < 4 * limit) {
          chunk.setBlock(x + dx, 3 + dy, z + dz, tree0);
          chunk.setBlock(x + dx, 4 + dy, z + dz, tree1);
        }
      }
    }
  };

  env.refresh();
};

window.onload = main;

export {};
