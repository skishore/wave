import {assert, int, nonnull, Color, Vec3} from './base.js';
import {BlockId, Env, init} from './engine.js';
import {kEmptyBlock, kNoMaterial, kWorldHeight} from './engine.js';
import {Component, ComponentState, ComponentStore} from './ecs.js';
import {EntityId, kNoEntity} from './ecs.js';
import {AStar, Check, PathNode, Point as AStarPoint} from './pathing.js';
import {SpriteMesh, ShadowMesh, Texture} from './renderer.js';
import {sweep} from './sweep.js';

//////////////////////////////////////////////////////////////////////////////

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

class TypedEnv extends Env {
  particles: int = 0;
  blocks: Blocks | null = null;
  point_lights: Map<string, PointLight>;
  lifetime: ComponentStore<LifetimeState>;
  position: ComponentStore<PositionState>;
  movement: ComponentStore<MovementState>;
  pathing: ComponentStore<PathingState>;
  physics: ComponentStore<PhysicsState>;
  meshes: ComponentStore<MeshState>;
  shadow: ComponentStore<ShadowState>;
  inputs: ComponentStore<InputState>;
  lights: ComponentStore<LightState>;
  target: ComponentStore;

  constructor(id: string) {
    super(id);
    const ents = this.entities;
    this.point_lights = new Map();
    this.lifetime = ents.registerComponent('lifetime', Lifetime);
    this.position = ents.registerComponent('position', Position);
    this.inputs = ents.registerComponent('inputs', Inputs(this));
    this.pathing = ents.registerComponent('pathing', Pathing(this));
    this.movement = ents.registerComponent('movement', Movement(this));
    this.physics = ents.registerComponent('physics', Physics(this));
    this.meshes = ents.registerComponent('meshes', Meshes(this));
    this.shadow = ents.registerComponent('shadow', Shadow(this));
    this.lights = ents.registerComponent('lights', Lights(this));
    this.target = ents.registerComponent('camera-target', CameraTarget(this));
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

// An entity with a lifetime calls cleanup() at the end of its life.

interface LifetimeState {
  id: EntityId,
  index: int,
  lifetime: number,
  cleanup: (() => void) | null,
};

const Lifetime: Component<LifetimeState> = {
  init: () => ({id: kNoEntity, index: 0, lifetime: 0, cleanup: null}),
  onUpdate: (dt: number, states: LifetimeState[]) => {
    for (const state of states) {
      state.lifetime -= dt;
      if (state.lifetime < 0 && state.cleanup) state.cleanup();
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
    const z = int(Math.floor((min[2] + max[2]) / 2));
    const block = env.getBlock(x, y, z);
    return opaque[block] && solid[block];
  })();
  const step_z = (() => {
    if (resting[2] === 0) return false;
    if (threshold * speed_z <= speed_x) return false;
    const x = int(Math.floor((min[0] + max[0]) / 2));
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
  const x = int(Math.floor((min[0] + max[0]) / 2));
  const y = int(Math.floor(min[1]));
  const z = int(Math.floor((min[2] + max[2]) / 2));

  const block = env.getBlock(x, y, z);
  const mesh = env.registry.getBlockMesh(block);
  state.inFluid = block !== kEmptyBlock && mesh === null;
  state.inGrass = block === nonnull(env.blocks).bush;

  const drag = state.inFluid ? 2 : 0;
  const left = Math.max(1 - drag * dt, 0);
  const gravity = state.inFluid ? 0.25 : 1;

  Vec3.scale(kTmpAcceleration, state.forces, 1 / state.mass);
  Vec3.scaleAndAdd(kTmpAcceleration, kTmpAcceleration, kTmpGravity, gravity);
  Vec3.scale(kTmpDelta, kTmpAcceleration, dt);
  Vec3.scaleAndAdd(kTmpDelta, kTmpDelta, state.impulses, 1 / state.mass);
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
    autoStep: 0.0625,
    autoStepMax: 0.5,
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
  hovering: boolean,
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
  hoverFallForce: number,
  hoverRiseForce: number,
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
    body.friction = 10;
    body.restitution = 0.5;

    const mesh = env.meshes.add(particle);
    const sprite = {url: texture.url, x: texture.w, y: texture.h};
    mesh.mesh = env.renderer.addSpriteMesh(size / texture.w, sprite);
    mesh.mesh.frame = int(texture.x + texture.y * texture.w);

    const epsilon = 0.01;
    const s = Math.floor(16 * (1 - size) * Math.random()) / 16;
    const t = Math.floor(16 * (1 - size) * Math.random()) / 16;
    const uv = size - 2 * epsilon;
    mesh.mesh.setSTUV(s + epsilon, t + epsilon, uv, uv);

    const lifetime = env.lifetime.add(particle);
    lifetime.lifetime = 1.0 * Math.random() + 0.5;
    lifetime.cleanup = () => {
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

  if (state.hovering) {
    const force = body.vel[1] < 0 ? state.hoverFallForce : state.hoverRiseForce;
    body.forces[1] += force;
  }

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
    hovering: false,
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
    hoverFallForce: 160,
    hoverRiseForce: 80,
  }),
  onUpdate: (dt: number, states: MovementState[]) => {
    for (const state of states) runMovement(env, dt, state);
  }
});

// An entity with an input component processes inputs.

interface InputState {
  id: EntityId,
  index: int,
  lastHeading: number;
};

const runInputs = (env: TypedEnv, state: InputState) => {
  const movement = env.movement.get(state.id);
  if (!movement) return;

  // Process the inputs to get a heading, running, and jumping state.
  const inputs = env.getMutableInputs();
  const fb = (inputs.up ? 1 : 0) - (inputs.down ? 1 : 0);
  const lr = (inputs.right ? 1 : 0) - (inputs.left ? 1 : 0);
  movement.jumping = inputs.space;
  movement.hovering = inputs.hover;

  if (fb || lr) {
    let heading = env.renderer.camera.heading;
    if (fb) {
      if (fb === -1) heading += Math.PI;
      heading += fb * lr * Math.PI / 4;
    } else {
      heading += lr * Math.PI / 2;
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
  if (body && (inputs.call || true)) {
    const {min, max} = body;
    const heading = state.lastHeading;
    const multiplier = (fb || lr) ? 1.5 : 2.0;
    const kFollowDistance = multiplier * (max[0] - min[0]);
    const x = (min[0] + max[0]) / 2 - kFollowDistance * Math.sin(heading);
    const z = (min[2] + max[2]) / 2 - kFollowDistance * Math.cos(heading);
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
  inputs.call = false;

  // Turn mouse inputs into actions.
  if (inputs.mouse0 || inputs.mouse1) {
    const body = env.physics.get(state.id);
    if (body) tryToModifyBlock(env, body, !inputs.mouse0);
    inputs.mouse0 = false;
    inputs.mouse1 = false;
  }
};

const Inputs = (env: TypedEnv): Component<InputState> => ({
  init: () => ({id: kNoEntity, index: 0, lastHeading: 0}),
  onUpdate: (dt: number, states: InputState[]) => {
    for (const state of states) runInputs(env, state);
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
  const sx = int(Math.floor((min[0] + max[0]) / 2));
  const sy = int(Math.floor(min[1]));
  const sz = int(Math.floor((min[2] + max[2]) / 2));
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

  const cx = (body.min[0] + body.max[0]) / 2;
  const cz = (body.min[2] + body.max[2]) / 2;
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
      const light = env.getLight(
          int(Math.floor(x)), int(Math.floor(y)), int(Math.floor(z)));
      mesh.setPosition(x, y - h / 2 - offset, z);
      mesh.height = h + 2 * offset;
      mesh.light = light;

      if (state.heading !== null) {
        const camera_heading = Math.atan2(x - cx, z - cz);
        const delta = state.heading - camera_heading;
        state.row = int(Math.floor(20.5 - 8 * delta / (2 * Math.PI)) & 7);
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
      if (!state.mesh || !state.cols) return;
      const body = env.physics.get(state.id);
      if (!body) return;

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
      const {x, y, z, h, w} = env.position.getX(state.id);
      env.setCameraTarget(x, y + h / 3, z);
      const mesh = env.meshes.get(state.id);
      const zoom = env.renderer.camera.zoom_value;
      if (mesh && mesh.mesh) mesh.mesh.enabled = zoom > 2 * w;
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

const safeHeight = (env: Env, position: PositionState): number => {
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

const addEntity = (env: TypedEnv, x: number, z: number, h: number, w: number,
                   maxSpeed: number, moveForceFactor: number,
                   jumpForce: number, jumpImpulse: number): EntityId => {
  const entity = env.entities.addEntity();
  const position = env.position.add(entity);
  position.x = x + 0.5;
  position.z = z + 0.5;
  position.w = w;
  position.h = h;
  position.y = safeHeight(env, position);

  const movement = env.movement.add(entity);
  movement.maxSpeed = maxSpeed;
  movement.moveForce = maxSpeed * moveForceFactor;
  movement.jumpForce = jumpForce;
  movement.jumpImpulse = jumpImpulse;

  env.physics.add(entity);
  env.shadow.add(entity);
  return entity;
};

const main = () => {
  const env = new TypedEnv('container');

  const [x, z] = [-2, 2];
  const player_height = 1.5;
  const player_sprite = {url: `images/player.png`, x: int(32), y: int(32)};
  const player = addEntity(env, x, z, player_height, 0.75, 8, 4, 10, 7.5);
  const player_scale = player_height / player_sprite.y;
  const player_mesh = env.meshes.add(player);
  player_mesh.mesh = env.renderer.addSpriteMesh(player_scale, player_sprite);
  player_mesh.cols = 3;
  player_mesh.rows = 4;
  env.inputs.add(player);
  env.target.add(player);

  const [cy, sx, sy] = [int(24), int(32), int(40)];
  const follower = addEntity(env, x, z, 0.75, 0.75, 12, 8, 15, 10);
  const follower_scale = player_scale * 0.8;
  const follower_mesh = env.meshes.add(follower);
  const follower_sprite = {url: `images/0025.png`, x: sx, y: sy};
  follower_mesh.mesh = env.renderer.addSpriteMesh(follower_scale, follower_sprite);
  follower_mesh.cols = 4;
  follower_mesh.rows = 8;
  follower_mesh.heading = 0;
  follower_mesh.offset = follower_scale * (sy - cy);
  env.lights.add(follower).level = 15;
  env.pathing.add(follower);

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
