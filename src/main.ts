import {assert, int, nonnull, Color, Vec3} from './base.js';
import {BlockId, Env, init} from './engine.js';
import {kEmptyBlock, kNoMaterial, kSunlightLevel, kWorldHeight} from './engine.js';
import {Component, ComponentState, ComponentStore} from './ecs.js';
import {EntityId, kNoEntity} from './ecs.js';
import {AStar, Check, PathNode, Point as AStarPoint} from './pathing.js';
import {ItemGeometry, Texture} from './renderer.js';
import {ItemMesh, SpriteMesh, ShadowMesh} from './renderer.js';
import {sweep} from './sweep.js';

//////////////////////////////////////////////////////////////////////////////

const TAU = 2 * Math.PI;
const kNumParticles = 16;
const kMaxNumParticles = 64;

const kWaterDelay = 200;
const kWaterDisplacements = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [-1, 0, 0],
  [0, 0, -1],
];

type Point = [int, int, int];
type Position = [number, number, number];

interface Blocks {
  bedrock: BlockId,
  bush:    BlockId,
  dirt:    BlockId,
  fungi:   BlockId,
  grass:   BlockId,
  rock:    BlockId,
  sand:    BlockId,
  snow:    BlockId,
  stone:   BlockId,
  trunk:   BlockId,
  water:   BlockId,
};

//////////////////////////////////////////////////////////////////////////////

class WebUI {
  private cursor: HTMLElement;
  private cursor_shown = false;

  constructor() {
    this.cursor = nonnull(document.getElementById("cursor"));
  }

  showCursor(shown: boolean): void {
    if (shown === this.cursor_shown) return;
    if (!shown) this.cursor.classList.add('hidden');
    if (shown) this.cursor.classList.remove('hidden');
    this.cursor_shown = shown;
  }
};

//////////////////////////////////////////////////////////////////////////////

class TypedEnv extends Env {
  particles: int = 0;
  blocks: Blocks | null = null;
  point_lights: Map<string, PointLight>;
  callback: ComponentStore<CallbackState>;
  position: ComponentStore<PositionState>;
  movement: ComponentStore<MovementState>;
  pathing: ComponentStore<PathingState>;
  physics: ComponentStore<PhysicsState>;
  meshes: ComponentStore<MeshState>;
  shadow: ComponentStore<ShadowState>;
  inputs: ComponentStore<InputState>;
  lights: ComponentStore<LightState>;
  target: ComponentStore;
  ui: WebUI;

  constructor(id: string) {
    super(id);
    const ents = this.entities;
    this.point_lights = new Map();
    this.callback = ents.registerComponent('callback', Callback);
    this.position = ents.registerComponent('position', Position);
    this.inputs = ents.registerComponent('inputs', Inputs(this));
    this.pathing = ents.registerComponent('pathing', Pathing(this));
    this.movement = ents.registerComponent('movement', Movement(this));
    this.physics = ents.registerComponent('physics', Physics(this));
    this.meshes = ents.registerComponent('meshes', Meshes(this));
    this.shadow = ents.registerComponent('shadow', Shadow(this));
    this.lights = ents.registerComponent('lights', Lights(this));
    this.target = ents.registerComponent('camera-target', CameraTarget(this));
    this.ui = new WebUI();
  }
};

const hasWaterNeighbor = (env: TypedEnv, water: BlockId, p: Point) => {
  for (const d of kWaterDisplacements) {
    const x = int(d[0] + p[0]), y = int(d[1] + p[1]), z = int(d[2] + p[2]);
    const block = env.getBlock(x, y, z);
    if (block === water) return true;
  }
  return false;
};

const flowWater = (env: TypedEnv, water: BlockId, points: Point[]) => {
  const next: Point[] = [];
  const visited: Set<string> = new Set();

  for (const p of points) {
    const block = env.getBlock(p[0], p[1], p[2]);
    if (block !== kEmptyBlock || !hasWaterNeighbor(env, water, p)) continue;
    env.setBlock(p[0], p[1], p[2], water);
    for (const d of kWaterDisplacements) {
      const n: Point = [int(p[0] - d[0]), int(p[1] - d[1]), int(p[2] - d[2])];
      const key = `${n[0]},${n[1]},${n[2]}`;
      if (visited.has(key)) continue;
      visited.add(key);
      next.push(n);
    }
  }

  if (next.length === 0) return;
  setTimeout(() => flowWater(env, water, next), kWaterDelay);
};

//////////////////////////////////////////////////////////////////////////////

// An entity with a callback calls it on each onRender and onUpdate call.

interface CallbackState {
  id: EntityId,
  index: int,
  onRender: ((dt: number) => void) | null,
  onUpdate: ((dt: number) => void) | null,
};

const Callback: Component<CallbackState> = {
  init: () => ({id: kNoEntity, index: 0, onRender: null, onUpdate: null}),
  onRender: (dt: number, states: CallbackState[]) => {
    for (const state of states) {
      if (state.onRender) state.onRender(dt);
    }
  },
  onUpdate: (dt: number, states: CallbackState[]) => {
    for (const state of states) {
      if (state.onUpdate) state.onUpdate(dt);
    }
  },
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
  inFluid: boolean,
  inGrass: boolean,
  friction: number,
  restitution: number,
  mass: number,
  autoStep: number,
  autoStepMax: number,
};

const kTmpGravity = Vec3.from(0, -40, 0);
const kTmpAcceleration = Vec3.create();
const kTmpFriction = Vec3.create();
const kTmpDelta = Vec3.create();
const kTmpSize = Vec3.create();
const kTmpPush = Vec3.create();
const kTmpMax = Vec3.create();
const kTmpMin = Vec3.create();
const kTmpPos = Vec3.create();
const kTmpResting = Vec3.create();

const setPhysicsFromPosition = (a: PositionState, b: PhysicsState) => {
  Vec3.set(kTmpPos, a.x, a.y, a.z);
  Vec3.set(kTmpSize, 0.5 * a.w, 0.5 * a.h, 0.5 * a.w);
  Vec3.sub(b.min, kTmpPos, kTmpSize);
  Vec3.add(b.max, kTmpPos, kTmpSize);
};

const setPositionFromPhysics = (a: PositionState, b: PhysicsState) => {
  a.x = 0.5 * (b.min[0] + b.max[0]);
  a.y = 0.5 * (b.min[1] + b.max[1]);
  a.z = 0.5 * (b.min[2] + b.max[2]);
};

const applyFriction = (axis: int, state: PhysicsState, dv: Vec3) => {
  const resting = state.resting[axis];
  if (resting === 0) return;

  Vec3.copy(kTmpFriction, state.vel);
  kTmpFriction[axis] = 0;
  const length = Vec3.length(kTmpFriction);
  if (length === 0) return;

  const loss = Math.abs(state.friction * dv[axis]);
  const scale = length < loss ? 0 : (length - loss) / length;
  state.vel[(axis + 1) % 3] *= scale;
  state.vel[(axis + 2) % 3] *= scale;
};

const tryAutoStepping =
    (env: TypedEnv, dt: number, state: PhysicsState, min: Vec3, max: Vec3,
     check: (x: int, y: int, z: int) => boolean) => {
  if (state.resting[1] > 0 && !state.inFluid) return;

  const {resting, vel} = state;
  const {opaque, solid} = env.registry;

  const threshold = 16;
  const speed_x = Math.abs(vel[0]);
  const speed_z = Math.abs(vel[2]);

  const step_x = (() => {
    if (resting[0] === 0) return false;
    if (threshold * speed_x <= speed_z) return false;
    const x = int(Math.floor(vel[0] > 0 ? max[0] + 0.5 : min[0] - 0.5));
    const y = int(Math.floor(min[1]));
    const z = int(Math.floor(0.5 * (min[2] + max[2])));
    const block = env.getBlock(x, y, z);
    return opaque[block] && solid[block];
  })();
  const step_z = (() => {
    if (resting[2] === 0) return false;
    if (threshold * speed_z <= speed_x) return false;
    const x = int(Math.floor(0.5 * (min[0] + max[0])));
    const y = int(Math.floor(min[1]));
    const z = int(Math.floor(vel[2] > 0 ? max[2] + 0.5 : min[2] - 0.5));
    const block = env.getBlock(x, y, z);
    return opaque[block] && solid[block];
  })();
  if (!step_x && !step_z) return;

  const height = 1 - min[1] + Math.floor(min[1]);
  if (height > state.autoStepMax) return;

  Vec3.set(kTmpDelta, 0, height, 0);
  sweep(min, max, kTmpDelta, kTmpResting, check);
  if (kTmpResting[1] !== 0) return;

  Vec3.scale(kTmpDelta, state.vel, dt);
  kTmpDelta[1] = 0;
  sweep(min, max, kTmpDelta, kTmpResting, check);
  if (min[0] === state.min[0] && min[2] === state.min[2]) return;

  if (height > state.autoStep) {
    Vec3.set(kTmpDelta, 0, state.autoStep, 0);
    sweep(state.min, state.max, kTmpDelta, state.resting, check);
    if (!step_x) state.vel[0] = 0;
    if (!step_z) state.vel[2] = 0;
    state.vel[1] = 0;
    return;
  }

  Vec3.copy(state.min, min);
  Vec3.copy(state.max, max);
  Vec3.copy(state.resting, kTmpResting);
};

const runPhysics = (env: TypedEnv, dt: number, state: PhysicsState) => {
  if (state.mass <= 0) return;

  const check = (x: int, y: int, z: int) => {
    const block = env.getBlock(x, y, z);
    return !env.registry.solid[block];
  };

  const {min, max} = state;
  const x = int(Math.floor(0.5 * (min[0] + max[0])));
  const z = int(Math.floor(0.5 * (min[2] + max[2])));
  const y = int(Math.floor(min[1]));

  const block = env.getBlock(x, y, z);
  const mesh = env.registry.getBlockMesh(block);
  state.inFluid = block !== kEmptyBlock && mesh === null;
  state.inGrass = block === nonnull(env.blocks).bush;

  const drag = state.inFluid ? 2 : 0;
  const left = Math.max(1 - drag * dt, 0);
  const gravity = state.inFluid ? 0.25 : 1;
  const inverse_mass = 1 / state.mass;

  Vec3.scale(kTmpAcceleration, state.forces, inverse_mass);
  Vec3.scaleAndAdd(kTmpAcceleration, kTmpAcceleration, kTmpGravity, gravity);
  Vec3.scale(kTmpDelta, kTmpAcceleration, dt);
  Vec3.scaleAndAdd(kTmpDelta, kTmpDelta, state.impulses, inverse_mass);
  if (state.friction) {
    Vec3.add(kTmpAcceleration, kTmpDelta, state.vel);
    applyFriction(0, state, kTmpAcceleration);
    applyFriction(1, state, kTmpAcceleration);
    applyFriction(2, state, kTmpAcceleration);
  }

  if (state.autoStep) {
    Vec3.copy(kTmpMax, state.max);
    Vec3.copy(kTmpMin, state.min);
  }

  // Update our state based on the computations above.
  Vec3.add(state.vel, state.vel, kTmpDelta);
  Vec3.scale(state.vel, state.vel, left);
  Vec3.scale(kTmpDelta, state.vel, dt);
  sweep(state.min, state.max, kTmpDelta, state.resting, check);
  Vec3.set(state.forces, 0, 0, 0);
  Vec3.set(state.impulses, 0, 0, 0);

  if (state.autoStep) {
    tryAutoStepping(env, dt, state, kTmpMin, kTmpMax, check);
  }

  for (let i = 0; i < 3; i++) {
    if (state.resting[i] === 0) continue;
    state.vel[i] = -state.restitution * state.vel[i];
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
    inFluid: false,
    inGrass: false,
    friction: 0,
    restitution: 0,
    mass: 1,
    autoStep: 0,
    autoStepMax: 0,
  }),
  onAdd: (state: PhysicsState) => {
    setPhysicsFromPosition(env.position.getX(state.id), state);
  },
  onRemove: (state: PhysicsState) => {
    const position = env.position.get(state.id);
    if (position) setPositionFromPhysics(position, state);
  },
  onUpdate: (dt: number, states: PhysicsState[]) => {
    for (const state of states) {
      runPhysics(env, dt, state);
      setPositionFromPhysics(env.position.getX(state.id), state);
    }
  },
});

// Movement allows an entity to process inputs and attempt to move.

interface MovementState {
  id: EntityId,
  index: int,
  inputX: number,
  inputZ: number,
  jumping: boolean,
  maxSpeed: number,
  moveForce: number,
  grassPenalty: number,
  waterPenalty: number,
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

const movementPenalty = (state: MovementState, body: PhysicsState): number => {
  return body.inFluid ? state.waterPenalty :
         body.inGrass ? state.grassPenalty : 1;
}

const handleJumping = (dt: number, state: MovementState,
                       body: PhysicsState, grounded: boolean) => {
  if (state._jumped) {
    if (state._jumpTimeLeft <= 0) return;
    const delta = state._jumpTimeLeft <= dt ? state._jumpTimeLeft / dt : 1;
    const force = state.jumpForce * delta;
    state._jumpTimeLeft -= dt;
    body.forces[1] += force;
    return;
  }

  const hasAirJumps = state._jumpCount < state.airJumps;
  const canJump = grounded || body.inFluid || hasAirJumps;
  if (!canJump) return;

  const height = body.min[1];
  const factor = height / kWorldHeight;
  const density = factor > 1 ? Math.exp(1 - factor) : 1;
  const penalty = density * (body.inFluid ? state.waterPenalty : 1);

  state._jumped = true;
  state._jumpTimeLeft = state.jumpTime;
  body.impulses[1] += state.jumpImpulse * penalty;
  if (grounded) return;

  body.vel[1] = Math.max(body.vel[1], 0);
  state._jumpCount++;
};

const handleRunning = (dt: number, state: MovementState,
                       body: PhysicsState, grounded: boolean) => {
  const penalty = movementPenalty(state, body);
  const speed = penalty * state.maxSpeed;
  Vec3.set(kTmpDelta, state.inputX * speed, 0, state.inputZ * speed);
  Vec3.sub(kTmpPush, kTmpDelta, body.vel);
  kTmpPush[1] = 0;
  const length = Vec3.length(kTmpPush);
  if (length === 0) return;

  const bound = state.moveForce * (grounded ? 1 : state.airMoveMultiplier);
  const input = state.responsiveness * length;
  Vec3.scale(kTmpPush, kTmpPush, Math.min(bound, input) / length);
  Vec3.add(body.forces, body.forces, kTmpPush);
};

const generateParticles =
    (env: TypedEnv, block: BlockId, x: int, y: int, z: int, side: int) => {
  const texture = (() => {
    const mesh = env.registry.getBlockMesh(block);
    if (mesh) {
      const {frame, sprite: {url, x: w, y: h}} = mesh;
      const x = frame % w, y = Math.floor(frame / w);
      return {alphaTest: true, sparkle: false, url, x, y, w, h};
    }
    const adjusted = side === 2 || side === 3 ? 0 : side;
    const material = env.registry.getBlockFaceMaterial(block, adjusted);
    if (material === kNoMaterial) return;
    return env.registry.getMaterialData(material).texture;
  })();
  if (!texture) return;

  const count = Math.min(kNumParticles, kMaxNumParticles - env.particles);
  env.particles += count;

  for (let i = 0; i < count; i++) {
    const particle = env.entities.addEntity();
    const position = env.position.add(particle);

    const size = Math.floor(3 * Math.random() + 1) / 16;
    position.x = x + (1 - size) * Math.random() + size / 2;
    position.y = y + (1 - size) * Math.random() + size / 2;
    position.z = z + (1 - size) * Math.random() + size / 2;
    position.w = position.h = size;

    const kParticleSpeed = 8;
    const body = env.physics.add(particle);
    body.impulses[0] = kParticleSpeed * (Math.random() - 0.5);
    body.impulses[1] = kParticleSpeed * Math.random();
    body.impulses[2] = kParticleSpeed * (Math.random() - 0.5);
    body.restitution = 0.5;
    body.friction = 1;

    const mesh = env.meshes.add(particle);
    const sprite = {url: texture.url, x: texture.w, y: texture.h};
    mesh.mesh = env.renderer.addSpriteMesh(size / texture.w, sprite);
    mesh.mesh.frame = int(texture.x + texture.y * texture.w);

    const epsilon = 0.01;
    const s = Math.floor(16 * (1 - size) * Math.random()) / 16;
    const t = Math.floor(16 * (1 - size) * Math.random()) / 16;
    const uv = size - 2 * epsilon;
    mesh.mesh.setSTUV(s + epsilon, t + epsilon, uv, uv);

    let lifetime = 1.0 * Math.random() + 0.5;
    env.callback.add(particle).onUpdate = dt => {
      if ((lifetime -= dt) > 0) return;
      env.entities.removeEntity(particle);
      env.particles--;
    };
  }
};

const modifyBlock = (env: TypedEnv, x: int, y: int, z: int,
                     block: BlockId, side: int): void => {
  const old_block = env.getBlock(x, y, z);
  env.setBlock(x, y, z, block);
  const new_block = env.getBlock(x, y, z);

  if (env.blocks) {
    const water = env.blocks.water;
    setTimeout(() => flowWater(env, water, [[x, y, z]]), kWaterDelay);
  }

  if (old_block !== kEmptyBlock && old_block !== new_block &&
      !(env.blocks && old_block === env.blocks.water)) {
    generateParticles(env, old_block, x, y, z, side);
  }
};

const tryToModifyBlock =
    (env: TypedEnv, body: PhysicsState, add: boolean) => {
  const target = env.getTargetedBlock();
  if (target === null) return;

  const side = env.getTargetedBlockSide();
  Vec3.copy(kTmpPos, target);

  if (add) {
    kTmpPos[side >> 1] += (side & 1) ? -1 : 1;
    const intersect = env.movement.some(state => {
      const body = env.physics.get(state.id);
      if (!body) return false;
      const {max, min} = body;
      for (let i = 0; i < 3; i++) {
        const pos = kTmpPos[i];
        if (pos < max[i] && min[i] < pos + 1) continue;
        return false;
      }
      return true;
    });
    if (intersect) return;
  }

  const x = int(kTmpPos[0]), y = int(kTmpPos[1]), z = int(kTmpPos[2]);
  const block = add && env.blocks ? env.blocks.dirt : kEmptyBlock;
  modifyBlock(env, x, y, z, block, side);

  if (block === kEmptyBlock) {
    for (let dy = 1; dy < 8; dy++) {
      const above = env.getBlock(x, int(y + dy), z);
      if (env.registry.getBlockMesh(above) === null) break;
      modifyBlock(env, x, int(y + dy), z, block, side);
    }
  }
};

const runMovement = (env: TypedEnv, dt: number, state: MovementState) => {
  const body = env.physics.getX(state.id);
  const grounded = body.resting[1] < 0;
  if (grounded) state._jumpCount = 0;

  if (state.jumping) {
    handleJumping(dt, state, body, grounded);
    state.jumping = false;
  } else {
    state._jumped = false;
  }

  if (state.inputX || state.inputZ) {
    handleRunning(dt, state, body, grounded);
    body.friction = state.runningFriction;
    state.inputX = state.inputZ = 0;
  } else {
    body.friction = state.standingFriction;
  }
};

const Movement = (env: TypedEnv): Component<MovementState> => ({
  init: () => ({
    id: kNoEntity,
    index: 0,
    inputX: 0,
    inputZ: 0,
    jumping: false,
    maxSpeed: 7.5,
    moveForce: 30,
    grassPenalty: 0.5,
    waterPenalty: 0.5,
    responsiveness: 15,
    runningFriction: 0,
    standingFriction: 2,
    airMoveMultiplier: 0.5,
    airJumps: 0,
    jumpTime: 0.2,
    jumpForce: 10,
    jumpImpulse: 7.5,
    _jumped: false,
    _jumpCount: 0,
    _jumpTimeLeft: 0,
  }),
  onUpdate: (dt: number, states: MovementState[]) => {
    for (const state of states) runMovement(env, dt, state);
  }
});

// An entity with an input component processes inputs.

const kSwingTime = 0.06;
const kSwingHalfLife = 0.5 * kSwingTime;
const kSwingPullback = -0.15;
const kSwingForwards = 0.30;
const kMaxEffortTime = 0.75;

type ItemType = 'ball' | 'shovel';

interface MonsterData {outside: boolean};

interface BaseItem {mesh: ItemMesh, range: number, type: ItemType};

type Item = BaseItem & {type: 'ball', extra: MonsterData} |
            BaseItem & {type: 'shovel', extra: null};

interface CurItem {
  item: Item,
  readied: boolean,
  effort: number,
  swing: number,
};

interface InputState {
  id: EntityId,
  index: int,
  lastHeading: number,
  curItem: CurItem | null,
  items: Item[],
};

const itemEnabled = (item: Item) => {
  return !(item.type === 'ball' && item.extra.outside);
};

const throwBall = (env: TypedEnv, effort: number, source: Item, vel: Vec3) => {
  if (source.type !== 'ball') throw new Error();
  if (source.extra.outside) return false;
  source.extra.outside = true;

  const friction = 4;
  const restitution = 0.25;
  const size = 0.375;
  const speed = 50;

  const camera = env.renderer.camera;
  const ball = env.entities.addEntity();
  const position = env.position.add(ball);
  const [px, py, pz] = source.mesh.getCenter(camera, 1.5 / size);
  const [vx, vy, vz] = camera.direction;

  position.x = px;
  position.y = py;
  position.z = pz;
  position.w = position.h = size;

  const body = env.physics.add(ball);
  body.impulses[0] = effort * speed * vx + vel[0];
  body.impulses[1] = effort * speed * vy + vel[1];
  body.impulses[2] = effort * speed * vz + vel[2];
  body.restitution = restitution;
  body.friction = friction;

  const sprite = kBallSprite;
  const mesh = env.meshes.add(ball);
  const shadow = env.shadow.add(ball);
  mesh.mesh = env.renderer.addSpriteMesh(size / sprite.x, sprite);
  mesh.type = 'ball';
  mesh.cols = 8;

  const kLifetime = 0.50;
  const light = env.lights.add(ball);
  let lifetime = kLifetime;
  let airborne = true;

  env.callback.add(ball).onUpdate = dt => {
    if (airborne) {
      if (body.resting[1] >= 0) return;
      const kStoppedSpeed = 0.01;
      const vx = body.vel[0], vz = body.vel[2];
      const speed_squared = vx * vx + vz * vz;
      if (speed_squared >= kStoppedSpeed * kStoppedSpeed) return;
      if (mesh.mesh) mesh.mesh.frame = int(mesh.cols + 1);
      tryToAddMonster(env, body);
      airborne = false;
    } else {
      lifetime -= dt;
      if (lifetime < 0) return env.entities.removeEntity(ball);
      const intensity = Math.min((1.5 * lifetime / kLifetime), 1);
      light.level = int(Math.round(kSunlightLevel * intensity));
    }
  };
};

const runInputs = (env: TypedEnv, dt: number, state: InputState) => {
  const movement = env.movement.get(state.id);
  if (!movement) return;

  // Process the inputs to get a heading, running, and jumping state.
  const inputs = env.getMutableInputs();
  const fb = (inputs.up ? 1 : 0) - (inputs.down ? 1 : 0);
  const lr = (inputs.right ? 1 : 0) - (inputs.left ? 1 : 0);
  movement.jumping = inputs.space;

  if (fb || lr) {
    let heading = env.renderer.camera.heading;
    if (fb) {
      if (fb === -1) heading += 0.5 * TAU;
      heading += fb * lr * 0.125 * TAU;
    } else {
      heading += lr * 0.25 * TAU;
    }
    movement.inputX = Math.sin(heading);
    movement.inputZ = Math.cos(heading);
    state.lastHeading = heading;

    const mesh = env.meshes.get(state.id);
    if (mesh) {
      const row = mesh.row;
      const option_a = fb > 0 ? 0 : fb < 0 ? 2 : -1;
      const option_b = lr > 0 ? 3 : lr < 0 ? 1 : -1;
      if (row !== option_a && row !== option_b) {
        mesh.row = int(Math.max(option_a, option_b));
      }
    }
  }

  // Call any followers.
  const body = env.physics.get(state.id);
  if (body) {
    const {min, max} = body;
    let heading = state.lastHeading;
    let multiplier = (fb || lr) ? 1.5 : 2.0;
    if (state.curItem) {
      heading += 0.45 * TAU;
      multiplier = (fb || lr ? 6.0 : 5.5);
    }
    const kFollowDistance = multiplier * (max[0] - min[0]);
    const x = 0.5 * (min[0] + max[0]) - kFollowDistance * Math.sin(heading);
    const z = 0.5 * (min[2] + max[2]) - kFollowDistance * Math.cos(heading);
    const y = (min[1] + body.autoStepMax);

    const ix = int(Math.floor(x));
    const iy = int(Math.floor(y));
    const iz = int(Math.floor(z));

    env.pathing.each(other => {
      other.request = {
        target: [ix, iy, iz],
        softTarget: [x, y, z],
        finalHeading: state.lastHeading,
      };
    });
  }

  // Use the item buttons to choose an item.
  const setCurItem = (item: Item | null) => {
    const mesh = state.curItem?.item.mesh;
    if (mesh) mesh.enabled = false;
    state.curItem = item ? {item, readied: false, effort: 0, swing: 0} : null;
  };
  if (inputs.item0 || inputs.item1 || inputs.quit) {
    const index = inputs.item0 ? 0 : inputs.item1 ? 1 : -1;
    setCurItem(state.items[index] || null);
    inputs.item0 = inputs.item1 = false;
  }
  inputs.item0 = inputs.item1 = inputs.quit = false;

  // Use the left mouse button to use the item.
  const curItem = state.curItem;
  if (curItem) {
    if (inputs.mouse1) {
      curItem.readied = false;
      inputs.mouse0 = false;
    }
    if (inputs.mouse0) curItem.readied = true;

    if (curItem.swing > 0) {
      curItem.swing += dt;
      if (curItem.swing >= kSwingTime) {
        const body = env.physics.get(state.id);
        if (body) tryToModifyBlock(env, body, false);
        curItem.swing = 0;
      }
    } else if (curItem.readied && !inputs.mouse0) {
      const base_effort = curItem.effort / kMaxEffortTime;
      const safe_effort = Math.max(0, Math.min(1, base_effort));
      const effort = 0.25 + 0.75 * safe_effort;
      curItem.readied = false;
      curItem.effort = 0;
      curItem.swing += dt;
      if (curItem.item.type === 'ball') {
        const vel = body?.vel || Vec3.create();
        throwBall(env, effort, curItem.item, vel);
      }
    } else if (curItem.readied) {
      curItem.effort += dt;
    }
  }
  inputs.mouse1 = false;
};

const Inputs = (env: TypedEnv): Component<InputState> => ({
  init: () => ({id: kNoEntity, index: 0, lastHeading: 0, curItem: null, items: []}),
  onUpdate: (dt: number, states: InputState[]) => {
    for (const state of states) runInputs(env, dt, state);
  }
});

// An entity with PathingState computes a path to a target and moves along it.

interface Path {
  index: int,
  steps: PathNode[],
  stepNeedsPrecision: boolean[],
  softTarget: Position | null,
  finalHeading: number | null,
};

interface PathRequest {
  target: Point,
  softTarget: Position | null,
  finalHeading: number | null,
};

interface PathingState {
  id: EntityId,
  index: int,
  path: Path | null,
  request: PathRequest | null,
};

const solid = (env: TypedEnv, x: int, y: int, z: int): boolean => {
  const block = env.getBlock(x, y, z);
  return env.registry.solid[block];
};

const findPath = (env: TypedEnv, body: PhysicsState,
                  state: PathingState, request: PathRequest): void => {
  const grounded = body.resting[1] < 0;
  if (inMidJump(state.path, grounded)) return;

  const {min, max} = body;
  const sx = int(Math.floor(0.5 * (min[0] + max[0])));
  const sz = int(Math.floor(0.5 * (min[2] + max[2])));
  const sy = int(Math.floor(min[1]));
  const [tx, ty, tz] = request.target;

  const source = new AStarPoint(sx, sy, sz);
  const target = new AStarPoint(tx, ty, tz);
  const check = (p: AStarPoint) => !solid(env, p.x, p.y, p.z);

  const steps = AStar(source, target, check);
  if (steps.length === 0) return;

  const last_index = steps.length - 1;
  const last = steps[last_index];
  const use_soft = last.x === tx && last.z === tz;

  const path = {
    index: 0 as int,
    steps: steps,
    stepNeedsPrecision: steps.map(_ => false),
    softTarget: use_soft ? request.softTarget : null,
    finalHeading: request.finalHeading,
  };
  state.path = path;
  state.request = null;
  path.stepNeedsPrecision[last_index] = true;

  //console.log(JSON.stringify(path.steps.map(x => [x.x, x.y, x.z])));
};

const PIDController =
    (error: number, derror: number, grounded: boolean): number => {
  const dfactor = grounded ? 1.00 : 2.00;
  return 20.00 * error + dfactor * derror;
};

const getSoftTarget = (path: Path, grounded: boolean): Position | null => {
  const {index, softTarget, steps} = path;
  const okay = index === steps.length - 1 && (grounded || !steps[index].jump);
  return okay ? softTarget : null;
};

const inMidJump = (path: Path | null, grounded: boolean): boolean => {
  if (grounded || path === null) return false;
  const {index, steps} = path;
  const step = steps[index];
  return step.jump || (index > 0 && step.y > steps[index - 1].y);
};

const nextPathStep = (env: TypedEnv, body: PhysicsState, path: Path): boolean => {
  const grounded = body.resting[1] < 0;
  if (inMidJump(path, grounded)) return false;

  const {min, max} = body;
  const {index, steps, stepNeedsPrecision} = path;
  const final_path_step = index === steps.length - 1;
  const needs_precision = stepNeedsPrecision[index];
  const soft_target = getSoftTarget(path, grounded);
  const step = steps[index];

  const x = soft_target ? soft_target[0] - 0.5 : step.x;
  const z = soft_target ? soft_target[2] - 0.5 : step.z;
  const y = step.y;

  const E = (() => {
    const width = max[0] - min[0];
    const final_path_step = index === steps.length - 1;
    if (final_path_step) return 0.4 * (1 - width);
    if (needs_precision) return 0.1 * (1 - width);
    return (index === 0 ? -0.6 : -0.4) * width;
  })();

  // Note that we use min[1] instead of max[1] for the upper bound here.
  // That means that we can trigger nextPathStep when we're still in the air
  // above the target block. That's okay, because we only run this code when
  // we're falling, never jumping - see the inMidJump check above.
  const y_okay = needs_precision ? y <= min[1] && max[1] <= y + 1
                                 : y <= min[1] && min[1] < y + 1;
  const result = x + E <= min[0] && max[0] <= x + 1 - E &&
                 z + E <= min[2] && max[2] <= z + 1 - E &&
                 y_okay;

  if (result && !needs_precision && !final_path_step) {
    const blocked = (() => {
      const check = (x: int, y: int, z: int) => {
        const block = env.getBlock(x, y, z);
        return !env.registry.solid[block];
      };

      const check_move = (x: number, y: number, z: number) => {
        Vec3.set(kTmpDelta, x, y, z);
        sweep(kTmpMin, kTmpMax, kTmpDelta, kTmpResting, check, true);
        return kTmpResting[0] || kTmpResting[1] || kTmpResting[2];
      };

      Vec3.copy(kTmpMax, body.max);
      Vec3.copy(kTmpMin, body.min);

      const prev = steps[index];
      const next = steps[index + 1];
      const dx = next.x + 0.5 - 0.5 * (body.min[0] + body.max[0]);
      const dz = next.z + 0.5 - 0.5 * (body.min[2] + body.max[2]);
      const dy = next.y - body.min[1];

      // TODO(shaunak): When applied to step 0, this check can result in a
      // kind of infinite loop that prevents path following. It occurs if the
      // pathfinding algorithm returns a path where the first step (from the
      // sprite's original cell to the next cell) includes a collision. A-star
      // isn't supposed to do that, but that's a fragile invariant to rely on.
      //
      // When the invariant is broken, then path_needs_precision will be set
      // for path_index 0. We'll move to the center of that cell, then move
      // to the next path step. But as soon as we move away from the center,
      // the next call to A-star will still have the same origin, and will
      // re-start the loop.
      //
      // How can we fix this issue safely? We could double-check here that if
      // the current path is blocked, then the path from the center of the cell
      // is not blocked (i.e. double-check the supposed invariant). If it fails,
      // then pathing to the block center is useless and we'll skip it.
      return (dy > 0 && check_move(0, dy, 0)) ||
             ((dx || dz) && check_move(dx, 0, dz));
    })();

    if (!blocked) return true;
    stepNeedsPrecision[index] = true;
    return false;
  }

  return result;
};

const followPath = (env: TypedEnv, body: PhysicsState,
                    state: PathingState, path: Path): void => {
  const mesh = env.meshes.get(state.id);
  const movement = env.movement.get(state.id);
  if (!movement) return;

  const steps = path.steps;
  assert(path.index < steps.length);
  if (nextPathStep(env, body, path)) path.index++;

  if (path.index === steps.length) {
    if (mesh && path.finalHeading) mesh.heading = path.finalHeading;
    state.path = null;
    return;
  }

  const index = path.index;
  const step = steps[index];
  const grounded = body.resting[1] < 0;
  const soft_target = getSoftTarget(path, grounded);
  const in_mid_jump = inMidJump(path, grounded);

  const cx = 0.5 * (body.min[0] + body.max[0]);
  const cz = 0.5 * (body.min[2] + body.max[2]);
  const dx = (soft_target ? soft_target[0] : step.x + 0.5) - cx;
  const dz = (soft_target ? soft_target[2] : step.z + 0.5) - cz;

  const penalty = movementPenalty(movement, body);
  const speed = penalty * movement.maxSpeed;
  const inverse_speed = speed ? 1 / speed : 1;

  let inputX = PIDController(dx, -body.vel[0], grounded) * inverse_speed;
  let inputZ = PIDController(dz, -body.vel[2], grounded) * inverse_speed;
  const length = Math.sqrt(inputX * inputX + inputZ * inputZ);
  const normalization = length > 1 ? 1 / length : 1;
  movement.inputX = inputX * normalization;
  movement.inputZ = inputZ * normalization;

  if (grounded) movement._jumped = false;
  movement.jumping = (() => {
    if (step.y > body.min[1]) return true;
    if (!grounded) return false;
    if (!step.jump) return false;

    if (index === 0) return false;
    const prev = steps[index - 1];
    if (Math.floor(cx) !== prev.x) return false;
    if (Math.floor(cz) !== prev.z) return false;

    const fx = cx - prev.x;
    const fz = cz - prev.z;
    return (dx > 1 && fx > 0.5) || (dx < -1 && fx < 0.5) ||
           (dz > 1 && fz > 0.5) || (dz < -1 && fz < 0.5);
  })();

  if (!mesh) return;
  const use_dx = (!in_mid_jump && soft_target) || index === 0;
  const vx = use_dx ? dx : step.x - steps[index - 1].x;
  const vz = use_dx ? dz : step.z - steps[index - 1].z;
  mesh.heading = Math.atan2(vx, vz);
};

const runPathing = (env: TypedEnv, state: PathingState): void => {
  if (!state.path && !state.request) return;
  const body = env.physics.get(state.id);
  if (!body) return;

  if (state.request) findPath(env, body, state, state.request);
  if (state.path) followPath(env, body, state, state.path);
};

const Pathing = (env: TypedEnv): Component<PathingState> => ({
  init: () => ({id: kNoEntity, index: 0, path: null, request: null}),
  onUpdate: (dt: number, states: PathingState[]) => {
    for (const state of states) runPathing(env, state);
  }
});

// An entity with a MeshState keeps a renderer mesh at its position.

type MeshType = 'ball' | 'walk';

interface MeshState {
  id: EntityId,
  index: int,
  mesh: SpriteMesh | null,
  heading: number | null,
  col: int,
  row: int,
  cols: int,
  rows: int,
  frame: number,
  offset: number,
  type: MeshType,
};

const Meshes = (env: TypedEnv): Component<MeshState> => ({
  init: () => ({
    id: kNoEntity,
    index: 0,
    mesh: null,
    heading: null,
    col: 0,
    row: 0,
    cols: 0,
    rows: 0,
    frame: 0,
    offset: 0,
    type: 'walk',
  }),
  onRemove: (state: MeshState) => state.mesh?.dispose(),
  onRender: (dt: number, states: MeshState[]) => {
    const camera = env.renderer.camera;
    let cx = camera.position[0], cz = camera.position[2];
    env.target.each(state => {
      const {x, y, z, h, w} = env.position.getX(state.id);
      cx = x - camera.zoom_value * Math.sin(camera.heading);
      cz = z - camera.zoom_value * Math.cos(camera.heading);
    });

    for (const state of states) {
      const {mesh, offset} = state;
      if (!mesh) continue;

      const {x, y, z, h, w} = env.position.getX(state.id);
      const light = env.getLight(x, y, z);
      mesh.setPosition(x, y - 0.5 * h - offset, z);
      mesh.height = h + 2 * offset;
      mesh.light = light;

      if (state.heading !== null) {
        const camera_heading = Math.atan2(x - cx, z - cz);
        const delta = state.heading - camera_heading;
        state.row = int(Math.floor(20.5 - 8 * delta / TAU) & 7);
        mesh.frame = int(state.col + state.row * state.cols);
      }
    }
  },
  onUpdate: (dt: number, states: MeshState[]) => {
    const lookup: int[][] = [
      [0, 0, 0, 0],
      [0, 0, 1, 1],
      [0, 1, 0, 2],
    ];

    for (const state of states) {
      if (!state.mesh || !state.cols) continue;
      const body = env.physics.get(state.id);
      if (!body) continue;

      if (state.type === 'ball') {
        if (state.mesh.frame >= state.cols) continue;
        const count = state.cols;
        const lower = (x: number): number => {
          while (x < 0) x += count;
          while (x >= count) x -= count;
          return x;
        };
        state.col = ((): int => {
          const heading = env.renderer.camera.heading;
          const vx = Math.sin(heading);
          const vz = Math.cos(heading);
          const velocity = body.vel[0] * vx + body.vel[2] * vz;
          state.frame = lower(state.frame + 0.01 * count * velocity);
          return int(lower(Math.floor(state.frame) + 1));
        })();
        state.mesh.frame = int(state.col + state.row * state.cols);
        continue;
      }

      const index = state.cols - 1;
      const count = index < lookup.length ? lookup[index].length : state.cols;

      const lower = (x: number): number => {
        while (x >= count) x -= count;
        return x;
      };

      const frame = ((): int => {
        if (body.resting[1] >= 0) return 1;
        const distance = dt * Vec3.length(body.vel);
        if (!distance) return state.frame = 0;
        state.frame = lower(state.frame + 0.1875 * count * distance);
        return int(lower(Math.floor(state.frame) + 1));
      })();
      state.col = index < lookup.length ? lookup[index][frame] : frame;
      state.mesh.frame = int(state.col + state.row * state.cols);
    }
  },
});

// An entity with a ShadowState casts a discrete shadow.

interface ShadowState {
  id: EntityId,
  index: int,
  mesh: ShadowMesh | null,
  extent: number,
  height: number,
};

const Shadow = (env: TypedEnv): Component<ShadowState> => ({
  init: () => ({id: kNoEntity, index: 0, mesh: null, extent: 16, height: 0}),
  onRemove: (state: ShadowState) => state.mesh?.dispose(),
  onRender: (dt: number, states: ShadowState[]) => {
    for (const state of states) {
      if (!state.mesh) state.mesh = env.renderer.addShadowMesh();
      const {x, y, z, w, h} = env.position.getX(state.id);
      const fraction = 1 - (y - 0.5 * h - state.height) / state.extent;
      const size = 0.5 * w * Math.max(0, Math.min(1, fraction));
      state.mesh.setPosition(x, state.height + 0.01, z);
      state.mesh.setSize(size);
    }
  },
  onUpdate: (dt: number, states: ShadowState[]) => {
    for (const state of states) {
      const position = env.position.getX(state.id);
      const x = int(Math.floor(position.x));
      const y = int(Math.floor(position.y));
      const z = int(Math.floor(position.z));
      state.height = (() => {
        for (let i = 0; i < state.extent; i++) {
          const h = y - i;
          if (solid(env, x, int(h - 1), z)) return h;
        }
        return 0;
      })();
    }
  },
});

// An entity with a LightState casts light.

interface LightState {
  id: EntityId,
  index: int,
  level: int,
};

interface PointLight {
  x: int,
  y: int,
  z: int,
  level: int,
};

const Lights = (env: TypedEnv): Component<LightState> => ({
  init: () => ({id: kNoEntity, index: 0, level: 0}),
  onRemove: (state: LightState) => {},
  onUpdate: (dt: number, states: LightState[]) => {
    const old_lights = env.point_lights;
    const new_lights: Map<string, PointLight> = new Map();

    for (const state of states) {
      if (state.level === 0) continue;
      const position = env.position.getX(state.id);
      const x = int(Math.floor(position.x));
      const y = int(Math.floor(position.y));
      const z = int(Math.floor(position.z));

      const fx = position.x - x;
      const fz = position.z - z;
      const ax = fx <  0.25 ? int(x - 1) : x;
      const bx = fx >= 0.75 ? int(x + 1) : x;
      const az = fz <  0.25 ? int(z - 1) : z;
      const bz = fz >= 0.75 ? int(z + 1) : z;

      for (let x = ax; x <= bx; x++) {
        for (let z = az; z <= bz; z++) {
          const key = `${x},${y},${z}`;
          const new_value = new_lights.get(key);
          if (new_value === undefined) {
            new_lights.set(key, {x, y, z, level: state.level});
          } else {
            new_value.level = int(Math.max(new_value.level, state.level));
          }
        }
      }
    }

    for (const [key, old_value] of old_lights.entries()) {
      if (new_lights.get(key)?.level !== old_value.level) {
        const {x, y, z} = old_value;
        env.setPointLight(x, y, z, 0);
      }
    }
    for (const [key, new_value] of new_lights.entries()) {
      if (old_lights.get(key)?.level !== new_value.level) {
        const {x, y, z, level} = new_value;
        env.setPointLight(x, y, z, level);
      }
    }

    env.point_lights = new_lights;
  },
});

// CameraTarget signifies that the camera will follow an entity.

const CameraTarget = (env: TypedEnv): Component => ({
  init: () => ({id: kNoEntity, index: 0}),
  onRender: (dt: number, states: ComponentState[]) => {
    for (const state of states) {
      const curItem = env.inputs.get(state.id)?.curItem;
      const {x, y, z, h, w} = env.position.getX(state.id);
      env.setCameraTarget(x, y + h / 3, z, !!curItem);
      env.setHighlightRange(curItem?.item.range || 0);

      const item = curItem?.item.mesh;
      const mesh = env.meshes.get(state.id)?.mesh;
      const near = env.renderer.camera.zoom_value < 2 * w;
      if (mesh) mesh.enabled = !near;
      if (item) item.enabled = near && itemEnabled(curItem.item);
      env.ui.showCursor(near);

      if (item && near) {
        // ln(2) ~ 0.6931
        const decay = Math.exp((-0.6931 / kSwingHalfLife) * dt);
        const target = curItem.swing ? kSwingForwards :
                       curItem.readied ? kSwingPullback : 0;
        item.offset = decay * item.offset + (1 - decay) * target;
        item.light = env.getLight(x, y + h / 3, z);
      }
    }
  },
  onUpdate: (dt: number, states: ComponentState[]) => {
    for (const state of states) {
      const {x, y, z} = env.position.getX(state.id);
      env.recenter(x, y, z);
    }
  },
});

// Putting it all together:

const kBallSprite    = {url: 'images/ball.png',   x: int(16), y: int(24)};
const kItemSprite    = {url: `images/items.png`,  x: int(24), y: int(24)};
const kMonsterSprite = {url: `images/0025.png`,   x: int(32), y: int(40)};
const kPlayerSprite  = {url: `images/player.png`, x: int(32), y: int(32)};

const kSprites = [kBallSprite, kItemSprite, kMonsterSprite, kPlayerSprite];

interface Pos {x: number, y: number, z: number, w: number, h: number};

const getSafeHeight = (env: Env, position: PositionState): number => {
  const radius = 0.5 * (position.w + 1);
  const ax = Math.floor(position.x - radius);
  const az = Math.floor(position.z - radius);
  const bx = Math.ceil(position.x + radius);
  const bz = Math.ceil(position.z + radius);

  let height = 0;
  for (let x = int(ax); x <= bx; x++) {
    for (let z = int(az); z <= bz; z++) {
      height = Math.max(height, env.getBaseHeight(x, z));
    }
  }
  return height + 0.5 * (position.h + 1);
};

const addEntity = (env: TypedEnv, pos: Pos, safeHeight: boolean,
                   maxSpeed: number, moveForceFactor: number,
                   jumpForce: number, jumpImpulse: number): EntityId => {
  const entity = env.entities.addEntity();
  const position = env.position.add(entity);
  position.x = pos.x;
  position.y = pos.y;
  position.z = pos.z;
  position.w = pos.w;
  position.h = pos.h;

  if (safeHeight) position.y = getSafeHeight(env, position);

  const movement = env.movement.add(entity);
  movement.maxSpeed = maxSpeed;
  movement.moveForce = maxSpeed * moveForceFactor;
  movement.jumpForce = jumpForce;
  movement.jumpImpulse = jumpImpulse;

  const body = env.physics.add(entity);
  body.autoStep =  0.0625;
  body.autoStepMax = 0.5;

  env.shadow.add(entity);
  return entity;
};

const addMonster = (env: TypedEnv, x: number, y: number, z: number): void => {
  const sprite = kMonsterSprite;
  const [height, width] = [0.75, 0.75];
  const pos = {x, y: y + 0.5 * height + 0.05, z, w: width, h: height};
  const monster = addEntity(env, pos, false, 12, 8, 15, 10);
  const scale = 2 * width / sprite.y;
  const mesh = env.meshes.add(monster);
  mesh.mesh = env.renderer.addSpriteMesh(scale, sprite);
  mesh.cols = 4;
  mesh.rows = 8;
  mesh.offset = scale * (sprite.y - 24);
  env.pathing.add(monster);
};

const tryToAddMonster = (env: TypedEnv, body: PhysicsState): void => {
  const x = 0.5 * (body.min[0] + body.max[0]);
  const z = 0.5 * (body.min[2] + body.max[2]);
  const y = body.min[1];
  addMonster(env, x, y, z);
};

const main = () => {
  const env = new TypedEnv('container');
  for (const sprite of kSprites) env.renderer.preloadSprite(sprite);

  const [x, z] = [-1.5, 2.5];
  const sprite = kPlayerSprite;
  const [width, height] = [0.75, 1.5];
  const pos = {x, y: 0, z, w: width, h: height};
  const player = addEntity(env, pos, true, 8, 4, 10, 7.5);
  const scale = 2 * width / sprite.y;
  const mesh = env.meshes.add(player);
  mesh.mesh = env.renderer.addSpriteMesh(scale, sprite);
  mesh.cols = 3;
  mesh.rows = 4;
  env.inputs.add(player);
  env.target.add(player);

  const inputs = env.inputs.getX(player);
  const item_base = (mesh: ItemMesh, range: number, type: ItemType, extra: any): Item => {
    return {mesh, range, type, extra};
  };
  const item_ball = (mesh: ItemMesh, extra: MonsterData): Item => {
    return item_base(mesh, 0, 'ball', extra);
  };
  const item_shovel = (mesh: ItemMesh, range: number): Item => {
    return item_base(mesh, range, 'shovel', null);
  };

  const shovel = env.renderer.addItemMesh(
    kItemSprite, 0,
    new ItemGeometry()
      .scale(0.25)
      .rotateX(0.20 * TAU)
      .rotateY(0.08 * TAU)
      .translate(0.15, -0.10, 0.25),
  );
  inputs.items.push(item_shovel(shovel, 4));

  const ball = env.renderer.addItemMesh(
    kItemSprite, 1,
    new ItemGeometry()
      .scale(0.10)
      .rotateY(0.05 * TAU)
      .translate(0.10, -0.09, 0.25),
  );
  inputs.items.push(item_ball(ball, {outside: false}));

  const white: Color = [1, 1, 1, 1];
  const texture = (x: int, y: int, alphaTest: boolean = false,
                   color: Color = white, sparkle: boolean = false): Texture => {
    const url = 'images/frlg.png';
    return {alphaTest, color, sparkle, url, x, y, w: 16, h: 16};
  };

  const block = (x: int, y: int) => {
    const url = 'images/frlg.png';
    const frame = int(x + 16 * y);
    return env.renderer.addInstancedMesh(frame, {url, x: 16, y: 16});
  };

  const registry = env.registry;
  registry.addMaterial(
      'blue', texture(12, 0, false, [0.1, 0.1, 0.4, 0.6]), true);
  registry.addMaterial(
      'water', texture(11, 0, false, [1, 1, 1, 0.8], true), true);
  const textures: [string, int, int][] = [
    ['bedrock', 6, 0],
    ['dirt', 2, 0],
    ['grass', 0, 0],
    ['grass-side', 3, 0],
    ['stone', 1, 0],
    ['sand', 4, 0],
    ['snow', 5, 0],
    ['trunk', 8, 0],
    ['trunk-side', 7, 0],
  ];
  for (const [name, x, y] of textures) {
    registry.addMaterial(name, texture(x, y));
  }

  const blocks = {
    bedrock: registry.addBlock(['bedrock'], true),
    bush:    registry.addBlockMesh(block(10, 0), false),
    dirt:    registry.addBlock(['dirt'], true),
    fungi:   registry.addBlockMesh(block(13, 0), false, 9),
    grass:   registry.addBlock(['grass', 'dirt', 'grass-side'], true),
    rock:    registry.addBlockMesh(block(9, 0), true),
    sand:    registry.addBlock(['sand'], true),
    snow:    registry.addBlock(['snow'], true),
    stone:   registry.addBlock(['stone'], true),
    trunk:   registry.addBlock(['trunk', 'trunk-side'], true),
    water:   registry.addBlock(['water', 'blue', 'blue'], false),
  };

  env.blocks = blocks;
  env.refresh();
};

//////////////////////////////////////////////////////////////////////////////

init(main);

export {};
