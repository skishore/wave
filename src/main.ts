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
  data: Uint8Array;
  shape: [int, int, int];
  stride: [int, int, int];

  constructor(x: int, y: int, z: int) {
    this.data = new Uint8Array(x * y * z);
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

type Octree = BABYLON.Octree<BABYLON.Mesh>;
type OctreeBlock = BABYLON.OctreeBlock<BABYLON.Mesh>;

class Renderer {
  camera: BABYLON.Camera;
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
    this.scene.detachControl();

    const source = new BABYLON.Vector3(0.1, 1.0, 0.3);
    this.light = new BABYLON.HemisphericLight('light', source, this.scene);
    this.scene.clearColor = new BABYLON.Color4(0.8, 0.9, 1.0);
    this.scene.ambientColor = new BABYLON.Color3(1, 1, 1);
    this.light.diffuse = new BABYLON.Color3(1, 1, 1);
    this.light.specular = new BABYLON.Color3(1, 1, 1);

    const origin = new BABYLON.Vector3(8, 4, 1.5);
    this.camera = new BABYLON.FreeCamera('camera', origin, this.scene);
    this.camera.minZ = 0.01;

    const scene = this.scene;
    scene._addComponent(new BABYLON.OctreeSceneComponent(scene));
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

class Engine {
  container: Container;
  registry: Registry;
  renderer: Renderer;
  mesher: TerrainMesher;

  constructor(id: string) {
    this.container = new Container(id);
    this.registry = new Registry();
    this.renderer = new Renderer(this.container);
    this.mesher = new TerrainMesher(this.renderer.scene, this.registry);
  }
};

//////////////////////////////////////////////////////////////////////////////
// The game code:

const main = () => {
  const engine = new Engine('container');
  const registry = engine.registry;

  registry.addMaterialOfColor('grass', [0.2, 0.8, 0.2]);
  registry.addMaterialOfColor('water', [0.4, 0.4, 0.8], 0.6);
  const grass = registry.addBlock(['grass'], true);
  const water = registry.addBlock(['water'], false);

  const size = Constants.CHUNK_SIZE;
  const pl = size / 4;
  const pr = 3 * size / 4;
  const voxels = new Tensor3(size, size, size);
  for (let x = 0; x < size; x++) {
    for (let z = 0; z < size; z++) {
      const wall = x === 0 || x === size - 1 || z === 0 || z === size - 1;
      const pool = (pl <= x && x < pr && 4 && pl <= z && z < pr);
      const height = Math.min(wall ? 7 : 3, size);
      for (let y = 0; y < height; y++) {
        assert(voxels.get(x, y, z) === 0);
        const tile = y > 0 && pool ? water : grass;
        voxels.set(x, y, z, tile);
      }
    }
  }

  const renderer = engine.renderer;
  const mesh = engine.mesher.mesh(voxels);
  if (mesh) renderer.addMesh(mesh, false);
  renderer.render();
};

window.onload = main;
