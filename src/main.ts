import {sweep} from './sweep';

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
  CHUNK_SIZE: 16,
  CHUNK_KEY_BITS: 8,
  TICK_RESOLUTION: 4,
  TICKS_PER_FRAME: 4,
  TICKS_PER_SECOND: 30,
  CAMERA_SENSITIVITY: 10,
};

//////////////////////////////////////////////////////////////////////////////

type Input = 'up' | 'left' | 'down' | 'right' | 'pointer';

class Container {
  element: Element;
  canvas: HTMLCanvasElement;
  bindings: Map<int, Input>;
  inputs: Record<Input, boolean>;
  deltas: {x: int, y: int};

  constructor(id: string) {
    this.element = nonnull(document.getElementById(id), () => id);
    this.canvas = nonnull(this.element.querySelector('canvas'));
    this.inputs = {up: false, left: false, down: false, right: false, pointer: false};
    this.deltas = {x: 0, y: 0};

    this.bindings = new Map();
    this.bindings.set('W'.charCodeAt(0), 'up');
    this.bindings.set('A'.charCodeAt(0), 'left');
    this.bindings.set('S'.charCodeAt(0), 'down');
    this.bindings.set('D'.charCodeAt(0), 'right');

    const element = this.element;
    element.addEventListener('keydown', e => this.onKeyInput(e, true));
    element.addEventListener('keyup', e => this.onKeyInput(e, false));

    element.addEventListener('click', () => element.requestPointerLock());
    document.addEventListener('pointerlockchange', e => this.onPointerInput(e));
    document.addEventListener('mousemove', e => this.onMouseMove(e));
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

class Registry {
  _opaque: boolean[];
  _solid: boolean[];
  _faces: MaterialId[];
  _materials: Material[];
  _ids: Map<string, MaterialId>;

  constructor() {
    this._opaque = [false];
    this._solid = [false];
    const none = 0 as MaterialId;
    this._faces = [none, none, none, none, none, none];
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
    materials.forEach(x => {
      const material = this._ids.get(x);
      if (material === undefined) throw new Error(`Unknown material: ${x}`);
      this._faces.push(material + 1 as MaterialId);
    });

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
  direction: BABYLON.Vector3;
  heading: number; // In radians: [0, 2π)
  pitch: number;   // In radians: (-π/2, π/2)

  constructor(scene: BABYLON.Scene) {
    const origin = new BABYLON.Vector3(0, 0, 0);
    this.holder = new BABYLON.TransformNode('holder', scene);
    this.camera = new BABYLON.FreeCamera('camera', origin, scene);
    this.camera.parent = this.holder;
    this.camera.minZ = 0.01;

    this.pitch = 0;
    this.heading = 0;
    this.direction = new BABYLON.Vector3(0, 0, 1);
  }

  applyInputs(dx: number, dy: number) {
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

    // Compute the direction: rotate around the X-axis, then the Y-axis.
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const ch = Math.cos(this.heading);
    const sh = Math.sin(this.heading);
    this.direction.copyFromFloats(sh * cp, -sp, ch * cp);
  }

  setPosition(x: number, y: number, z: number) {
    this.holder.position.copyFromFloats(x, y, z);
  }
};

//////////////////////////////////////////////////////////////////////////////

type Octree = BABYLON.Octree<BABYLON.Mesh>;
type OctreeBlock = BABYLON.OctreeBlock<BABYLON.Mesh>;

class Renderer {
  camera: Camera;
  engine: BABYLON.Engine;
  light: BABYLON.Light;
  scene: BABYLON.Scene;
  octree: Octree;
  blocks: Map<int, OctreeBlock>;

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
    scene._addComponent(new BABYLON.OctreeSceneComponent(scene));
    this.camera = new Camera(scene);
    this.octree = new BABYLON.Octree(() => {});
    this.octree.blocks = [];
    scene._selectionOctree = this.octree;
    this.blocks = new Map();
  }

  addMesh(mesh: BABYLON.Mesh, dynamic: boolean) {
    if (dynamic) {
      const meshes = this.octree.dynamicContent;
      mesh.onDisposeObservable.add(() => drop(meshes, mesh));
      meshes.push(mesh);
      return;
    }

    const key = this.getMeshKey(mesh);
    const block = this.getMeshBlock(mesh, key);
    mesh.onDisposeObservable.add(() => {
      drop(block.entries, mesh);
      if (block.entries.length) return;
      drop(this.octree.blocks, block);
      this.blocks.delete(key);
    });
    block.entries.push(mesh);

    mesh.alwaysSelectAsActiveMesh = true;
    mesh.freezeWorldMatrix();
    mesh.freezeNormals();
  }

  render() {
    this.engine.beginFrame();
    this.scene.render();
    this.engine.endFrame();
  }

  private getMeshKey(mesh: BABYLON.Mesh): int {
    assert(!mesh.parent);
    const pos = mesh.position;
    const mod = Constants.CHUNK_SIZE;
    assert(pos.x % mod === 0);
    assert(pos.y % mod === 0);
    assert(pos.z % mod === 0);

    const bits = Constants.CHUNK_KEY_BITS;
    const mask = (1 << bits) - 1;
    return (((pos.x / mod) & mask) << (0 * bits)) |
           (((pos.y / mod) & mask) << (1 * bits)) |
           (((pos.z / mod) & mask) << (2 * bits));
  }

  private getMeshBlock(mesh: BABYLON.Mesh, key: int): OctreeBlock {
    const cached = this.blocks.get(key);
    if (cached) return cached;

    const pos = mesh.position;
    const mod = Constants.CHUNK_SIZE;
    const min = new BABYLON.Vector3(pos.x, pos.y, pos.z);
    const max = new BABYLON.Vector3(pos.x + mod, pos.y + mod, pos.z + mod);

    const block: OctreeBlock =
      new BABYLON.OctreeBlock(min, max, 0, 0, 0, () => {});
    this.octree.blocks.push(block);
    this.blocks.set(key, block);
    return block;
  }
};

//////////////////////////////////////////////////////////////////////////////

declare const NoaTerrainMesher: any;

class TerrainMesher {
  mesher: any;
  scene: BABYLON.Scene;
  flatMaterial: BABYLON.Material;
  registry: Registry;
  requests: int;

  constructor(scene: BABYLON.Scene, registry: Registry) {
    this.scene = scene;
    this.flatMaterial = this.makeStandardMaterial('flat-material');
    this.registry = registry;
    this.requests = 0;

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
        makeStandardMaterial: this.makeStandardMaterial.bind(this),
        getScene: () => scene,
      },
    };
    this.mesher = new NoaTerrainMesher(shim);
  }

  makeStandardMaterial(name: string): BABYLON.Material {
    const result = new BABYLON.StandardMaterial(name, this.scene);
    result.specularColor.copyFromFloats(0, 0, 0);
    result.ambientColor.copyFromFloats(1, 1, 1);
    result.diffuseColor.copyFromFloats(1, 1, 1);
    return result;
  }

  mesh(voxels: Tensor3): BABYLON.Mesh | null {
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
        const self = x === 0 && y === 0 && z === 0;
        return self ? {voxels} : null;
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

  add(entity: EntityId) {
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

  addEntity(components: string[]): EntityId {
    this.last = (this.last + 1) as EntityId;
    const result = this.last;
    components.forEach(x => this.addComponent(result, x));
    return result;
  }

  addComponent(entity: EntityId, component: string) {
    const store = this.components.get(component);
    if (!store) throw new Error(`Unknown component: ${component}`);
    store.add(entity);
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

class Env {
  container: Container;
  entities: EntityComponentSystem;
  registry: Registry;
  renderer: Renderer;
  mesher: TerrainMesher;
  timing: Timing;
  voxels: Tensor3;
  _dirty: boolean;
  _mesh: BABYLON.Mesh | null;

  constructor(id: string) {
    this.container = new Container(id);
    this.entities = new EntityComponentSystem();
    this.registry = new Registry();
    this.renderer = new Renderer(this.container);
    this.mesher = new TerrainMesher(this.renderer.scene, this.registry);
    this.timing = new Timing(this.render.bind(this), this.update.bind(this));

    const size = Constants.CHUNK_SIZE;
    this.voxels = new Tensor3(size, size, size);
    this._dirty = true;
    this._mesh = null;
  }

  getBlock(x: int, y: int, z: int): BlockId {
    return this.voxels.get(x, y, z) as BlockId;
  }

  setBlock(x: int, y: int, z: int, block: BlockId) {
    this.voxels.set(x, y, z, block);
    this._dirty = true;
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

    const deltas = this.container.deltas;
    this.renderer.camera.applyInputs(deltas.x, deltas.y);
    deltas.x = deltas.y = 0;
    this.entities.render(dt);
    this.renderer.render();
  }

  update(dt: int) {
    if (!this.container.inputs.pointer) return;

    if (this._dirty) {
      if (this._mesh) this._mesh.dispose();
      this._mesh = this.mesher.mesh(this.voxels);
      if (this._mesh) this.renderer.addMesh(this._mesh, false);
      this._dirty = false;
    }

    this.entities.update(dt);
  }
};

//////////////////////////////////////////////////////////////////////////////
// The game code:

class TypedEnv extends Env {
  position: ComponentStore<PositionState>;
  target: ComponentStore<ComponentState>;

  constructor(id: string) {
    super(id);
    const ents = this.entities;
    this.position = ents.registerComponent('position', Position);
    this.target = ents.registerComponent('camera-target', CameraTarget(this));
  }
};

// Position tracks (x, y, z) coordinates for an entity.

interface PositionState {
  id: EntityId,
  index: int,
  x: number,
  y: number,
  z: number,
};

const Position: Component<PositionState> = {
  init: () => ({id: kNoEntity, index: 0, x: 0, y: 0, z: 0}),
};

// CameraTarget signifies that the camera will follow an entity.

const CameraTarget = (env: TypedEnv): Component => ({
  init: () => ({id: kNoEntity, index: 0}),
  onRender: (dt: int, states: ComponentState[]) => {
    for (const state of states) {
      const position = env.position.getX(state.id);
      env.renderer.camera.setPosition(position.x, position.y, position.z);
    }
  },
  onUpdate: (dt: int, states: ComponentState[]) => {
    const inputs = env.container.inputs;
    const ud = (inputs.up ? 1 : 0) - (inputs.down ? 1 : 0);
    const speed = 0.5 * ud;

    const camera = env.renderer.camera;
    const direction = camera.direction;

    for (const state of states) {
      const position = env.position.getX(state.id);
      position.x += speed * direction.x;
      position.y += speed * direction.y;
      position.z += speed * direction.z;
    }
  },
});

// Putting it all together:

const main = () => {
  const env = new TypedEnv('container');
  const player = env.entities.addEntity(['position', 'camera-target']);
  const position = env.position.getX(player);
  position.x = 8;
  position.y = 4;
  position.z = 1.5;

  const registry = env.registry;
  registry.addMaterialOfColor('grass', [0.2, 0.8, 0.2]);
  registry.addMaterialOfColor('water', [0.4, 0.4, 0.8], 0.6);
  const grass = registry.addBlock(['grass'], true);
  const water = registry.addBlock(['water'], false);

  const size = Constants.CHUNK_SIZE;
  const pl = size / 4;
  const pr = 3 * size / 4;
  for (let x = 0; x < size; x++) {
    for (let z = 0; z < size; z++) {
      const wall = x === 0 || x === size - 1 || z === 0 || z === size - 1;
      const pool = (pl <= x && x < pr && 4 && pl <= z && z < pr);
      const height = Math.min(wall ? 7 : 3, size);
      for (let y = 0; y < height; y++) {
        assert(env.getBlock(x, y, z) === 0);
        const tile = y > 0 && pool ? water : grass;
        env.setBlock(x, y, z, tile);
      }
    }
  }

  env.refresh();
};

window.onload = main;

export {};
