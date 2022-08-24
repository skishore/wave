import {makeNoise2D} from '../lib/open-simplex-2d.js';
import {assert, int, nonnull, Tensor3, Vec3} from './base.js';
import {BlockId, Column, Env} from './engine.js';
import {kChunkWidth, kEmptyBlock, kWorldHeight} from './engine.js';
import {Component, ComponentState, ComponentStore} from './ecs.js';
import {EntityId, kNoEntity} from './ecs.js';
import {SpriteMesh, ShadowMesh, Texture} from './renderer.js';
import {sweep} from './sweep.js';
import {Blocks, getHeight, loadChunk, loadFrontier} from './worldgen.js';

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

//////////////////////////////////////////////////////////////////////////////

class TypedEnv extends Env {
  particles: int = 0;
  blocks: Blocks | null = null;
  lifetime: ComponentStore<LifetimeState>;
  position: ComponentStore<PositionState>;
  movement: ComponentStore<MovementState>;
  pathing: ComponentStore<PathingState>;
  physics: ComponentStore<PhysicsState>;
  meshes: ComponentStore<MeshState>;
  shadow: ComponentStore<ShadowState>;
  inputs: ComponentStore;
  target: ComponentStore;

  constructor(id: string) {
    super(id);
    const ents = this.entities;
    this.lifetime = ents.registerComponent('lifetime', Lifetime);
    this.position = ents.registerComponent('position', Position);
    this.inputs = ents.registerComponent('inputs', Inputs(this));
    this.pathing = ents.registerComponent('pathing', Pathing(this));
    this.movement = ents.registerComponent('movement', Movement(this));
    this.physics = ents.registerComponent('physics', Physics(this));
    this.meshes = ents.registerComponent('meshes', Meshes(this));
    this.shadow = ents.registerComponent('shadow', Shadow(this));
    this.target = ents.registerComponent('camera-target', CameraTarget(this));
  }
};

const hasWaterNeighbor = (env: TypedEnv, water: BlockId, p: Point) => {
  for (const d of kWaterDisplacements) {
    const block = env.world.getBlock(d[0] + p[0], d[1] + p[1], d[2] + p[2]);
    if (block === water) return true;
  }
  return false;
};

const flowWater = (env: TypedEnv, water: BlockId, points: Point[]) => {
  const next: Point[] = [];
  const visited: Set<string> = new Set();

  for (const p of points) {
    const block = env.world.getBlock(p[0], p[1], p[2]);
    if (block !== kEmptyBlock || !hasWaterNeighbor(env, water, p)) continue;
    env.world.setBlock(p[0], p[1], p[2], water);
    for (const d of kWaterDisplacements) {
      const n: Point = [p[0] - d[0], p[1] - d[1], p[2] - d[2]];
      const key = `${n[0]}-${n[1]}-${n[2]}`;
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
  onUpdate: (dt: int, states: LifetimeState[]) => {
    dt = dt / 1000;
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
  friction: number,
  restitution: number,
  mass: number,
  autoStep: number,
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
    (dt: int, state: PhysicsState, min: Vec3, max: Vec3,
     check: (pos: Vec3) => boolean) => {
  if (state.resting[1] > 0 && !state.inFluid) return;

  const threshold = 4;
  const speed_x = Math.abs(state.vel[0]);
  const speed_z = Math.abs(state.vel[2]);
  const step_x = (state.resting[0] !== 0 && threshold * speed_x > speed_z);
  const step_z = (state.resting[2] !== 0 && threshold * speed_z > speed_x);
  if (!step_x && !step_z) return;

  const height = 1 - min[1] + Math.floor(min[1]);
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

const runPhysics = (env: TypedEnv, dt: int, state: PhysicsState) => {
  if (state.mass <= 0) return;

  const check = (pos: Vec3) => {
    const block = env.world.getBlock(pos[0], pos[1], pos[2]);
    return !env.registry.solid[block];
  };

  const [x, y, z] = state.min;
  const block = env.world.getBlock(
      Math.floor(x), Math.floor(y), Math.floor(z));
  state.inFluid = block !== kEmptyBlock;

  dt = dt / 1000;
  const drag = state.inFluid ? 2 : 0;
  const left = Math.max(1 - drag * dt, 0);
  const gravity = state.inFluid ? 0.25 : 1;

  Vec3.scale(kTmpAcceleration, state.forces, 1 / state.mass);
  Vec3.scaleAndAdd(kTmpAcceleration, kTmpAcceleration, kTmpGravity, gravity);
  Vec3.scale(kTmpDelta, kTmpAcceleration, dt);
  Vec3.scaleAndAdd(kTmpDelta, kTmpDelta, state.impulses, 1 / state.mass);
  if (state.friction) {
    applyFriction(0, state, kTmpDelta);
    applyFriction(1, state, kTmpDelta);
    applyFriction(2, state, kTmpDelta);
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
    tryAutoStepping(dt, state, kTmpMin, kTmpMax, check);
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
    friction: 0,
    restitution: 0,
    mass: 1,
    autoStep: 0,
  }),
  onAdd: (state: PhysicsState) => {
    setPhysicsFromPosition(env.position.getX(state.id), state);
  },
  onRemove: (state: PhysicsState) => {
    const position = env.position.get(state.id);
    if (position) setPositionFromPhysics(position, state);
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
  inputX: number,
  inputZ: number,
  jumping: boolean,
  hovering: boolean,
  maxSpeed: number,
  moveForce: number,
  swimPenalty: number,
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

const handleJumping =
    (dt: int, state: MovementState, body: PhysicsState, grounded: boolean) => {
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
  const penalty = body.inFluid ? state.swimPenalty : density;

  state._jumped = true;
  state._jumpTimeLeft = state.jumpTime;
  body.impulses[1] += state.jumpImpulse * penalty;
  if (grounded) return;

  body.vel[1] = Math.max(body.vel[1], 0);
  state._jumpCount++;
};

const handleRunning =
    (dt: int, state: MovementState, body: PhysicsState, grounded: boolean) => {
  const penalty = body.inFluid ? state.swimPenalty : 1;
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
  const adjusted = side === 2 || side === 3 ? 0 : side;
  const material = env.registry.getBlockFaceMaterial(block, adjusted);
  const data = env.registry.getMaterialData(material);
  if (!data.texture) return;

  const count = Math.min(kNumParticles, kMaxNumParticles - env.particles);
  env.particles += count;

  for (let i = 0; i < count; i++) {
    const particle = env.entities.addEntity();
    const position = env.position.add(particle);

    const side = Math.floor(3 * Math.random() + 1) / 16;
    position.x = x + (1 - side) * Math.random() + side / 2;
    position.y = y + (1 - side) * Math.random() + side / 2;
    position.z = z + (1 - side) * Math.random() + side / 2;
    position.w = position.h = side;

    const kParticleSpeed = 8;
    const body = env.physics.add(particle);
    body.impulses[0] = kParticleSpeed * (Math.random() - 0.5);
    body.impulses[1] = kParticleSpeed * Math.random();
    body.impulses[2] = kParticleSpeed * (Math.random() - 0.5);
    body.friction = 10;
    body.restitution = 0.5;

    const size = position.h;
    const mesh = env.meshes.add(particle);
    const sprite = {url: 'images/rhodox-edited.png', size, x: 16, y: 16};
    mesh.mesh = env.renderer.addSpriteMesh(sprite);
    mesh.mesh.setFrame(data.texture.x + 16 * data.texture.y);

    const epsilon = 0.01;
    const s = Math.floor(16 * (1 - side) * Math.random()) / 16;
    const t = Math.floor(16 * (1 - side) * Math.random()) / 16;
    const uv = side - 2 * epsilon;
    mesh.mesh.setSTUV(s + epsilon, t + epsilon, uv, uv);

    const lifetime = env.lifetime.add(particle);
    lifetime.lifetime = 1.0 * Math.random() + 0.5;
    lifetime.cleanup = () => {
      env.entities.removeEntity(particle);
      env.particles--;
    };
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

  const x = kTmpPos[0], y = kTmpPos[1], z = kTmpPos[2];
  const old_block = add ? kEmptyBlock : env.world.getBlock(x, y, z);
  const block = add && env.blocks ? env.blocks.dirt : kEmptyBlock;
  env.world.setBlock(x, y, z, block);
  const new_block = add ? kEmptyBlock : env.world.getBlock(x, y, z);

  if (env.blocks) {
    const water = env.blocks.water;
    setTimeout(() => flowWater(env, water, [[x, y, z]]), kWaterDelay);
  }

  if (old_block !== kEmptyBlock && old_block !== new_block) {
    generateParticles(env, old_block, x, y, z, side);
  }
};

const runMovement = (env: TypedEnv, dt: int, state: MovementState) => {
  dt = dt / 1000;
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
    swimPenalty: 0.5,
    responsiveness: 15,
    runningFriction: 0,
    standingFriction: 2,
    airMoveMultiplier: 0.5,
    airJumps: 0,
    jumpTime: 0.2,
    jumpForce: 15,
    jumpImpulse: 10,
    _jumped: false,
    _jumpCount: 0,
    _jumpTimeLeft: 0,
    hoverFallForce: 160,
    hoverRiseForce: 80,
  }),
  onUpdate: (dt: int, states: MovementState[]) => {
    for (const state of states) runMovement(env, dt, state);
  }
});

// An entity with an input component processes inputs.

const runInputs = (env: TypedEnv, id: EntityId) => {
  const state = env.movement.get(id);
  if (!state) return;

  // Process the inputs to get a heading, running, and jumping state.
  const inputs = env.container.inputs;
  const fb = (inputs.up ? 1 : 0) - (inputs.down ? 1 : 0);
  const lr = (inputs.right ? 1 : 0) - (inputs.left ? 1 : 0);
  state.jumping = inputs.space;
  state.hovering = inputs.hover;

  if (fb || lr) {
    let heading = env.renderer.camera.heading;
    if (fb) {
      if (fb === -1) heading += Math.PI;
      heading += fb * lr * Math.PI / 4;
    } else {
      heading += lr * Math.PI / 2;
    }
    state.inputX = Math.sin(heading);
    state.inputZ = Math.cos(heading);

    const mesh = env.meshes.get(id);
    if (mesh) {
      const row = mesh.row;
      const option_a = fb > 0 ? 0 : fb < 0 ? 2 : -1;
      const option_b = lr > 0 ? 3 : lr < 0 ? 1 : -1;
      if (row !== option_a && row !== option_b) {
        mesh.row = Math.max(option_a, option_b);
      }
    }
  }

  // Call any followers.
  if (inputs.call) {
    const body = env.physics.get(id);
    if (body) {
      const x = Math.floor((body.min[0] + body.max[0]) / 2);
      const y = Math.floor((body.min[1] + body.max[1]) / 2);
      const z = Math.floor((body.min[2] + body.max[2]) / 2);
      env.pathing.each(other => other.target = [x, y, z]);
    }
    inputs.call = false;
  }

  // Turn mouse inputs into actions.
  if (inputs.mouse0 || inputs.mouse1) {
    const body = env.physics.get(id);
    if (body) tryToModifyBlock(env, body, !inputs.mouse0);
    inputs.mouse0 = false;
    inputs.mouse1 = false;
  }
};

const Inputs = (env: TypedEnv): Component => ({
  init: () => ({id: kNoEntity, index: 0}),
  onUpdate: (dt: int, states: ComponentState[]) => {
    for (const state of states) runInputs(env, state.id);
  }
});

// An entity with PathingState computes a path to a target and moves along it.

interface PathingState {
  id: EntityId,
  index: int,
  path_index: int,
  path: Point[] | null,
  target: Point | null,
};

const solid = (env: TypedEnv, x: int, y: int, z: int): boolean => {
  const block = env.world.getBlock(x, y, z);
  return env.registry.solid[block];
};

const hasDirectPath = (env: TypedEnv, start: Point, end: Point): boolean => {
  if (start[1] !== end[1]) return false;
  if (start[0] === end[0] && start[2] === end[2]) return true;

  const [sx, y, sz] = start;
  const [ex, _, ez] = end;
  const dx = Math.abs(ex - sx);
  const dz = Math.abs(ez - sz);
  const elements: Point[] = [];

  if (dx <= dz) {
    const extra = dx === 0 ? 1 : 0;
    for (let i = 0; i < dz; i++) {
      const a = Math.floor(i * dx / dz);
      const b = Math.ceil((i + 1) * dx / dz) + extra;
      for (let j = a; j < b; j++) {
        const x = ex >= sx ? sx + j : sx - j - 1;
        const z = ez >= sz ? sz + i : sz - i - 1;
        elements.push([x, y, z]);
      }
    }
  } else {
    const extra = dz === 0 ? 1 : 0;
    for (let i = 0; i < dx; i++) {
      const a = Math.floor(i * dz / dx);
      const b = Math.ceil((i + 1) * dz / dx) + extra;
      for (let j = a; j < b; j++) {
        const x = ex >= sx ? sx + i : sx - i - 1;
        const z = ez >= sz ? sz + j : sz - j - 1;
        elements.push([x, y, z]);
      }
    }
  }

  const n = dx && dz ? 4 : 1;
  for (const element of elements) {
    const [x, y, z] = element;
    for (let i = 0; i < n; i++) {
      const ix = x + (i & 1);
      const iz = z + ((i >> 1) & 1);
      if (solid(env, ix, y, iz) || !solid(env, ix, y - 1, iz)) return false;
    }
  }
  return true;
};

const findPath = (env: TypedEnv, state: PathingState,
                  body: PhysicsState): void => {
  const min = body.min;
  const sx = Math.floor(min[0]);
  const sy = Math.floor(min[1]);
  const sz = Math.floor(min[2]);
  const [tx, ty, tz] = nonnull(state.target);

  console.log(`Pathing: (${sx}, ${sy}, ${sz}) -> (${tx}, ${ty}, ${tz})`);

  const limit = 64;
  let prev: Point[] = [[sx, sy, sz]];
  let next: Point[] = [];
  let done: boolean = false;

  let shift = 1;
  while ((1 << shift) < 2 * limit) shift++;
  const mask = (1 << shift) - 1;

  const compute_key = (x: int, y: int, z: int): int => {
    return (y << (2 * shift)) |
           (((x - sx) & mask) << shift) |
           (((z - sz) & mask));
  };

  const visited: Map<int, Point | null> = new Map();
  visited.set(compute_key(sx, sy, sz), null);

  for (let i = 0; i < limit; i++) {
    for (const pp of prev) {
      const npY = pp[1];
      for (let dir = 0; dir < 4; dir++) {
        const npX = pp[0] + ((dir & 1) ? dir - 2 : 0);
        const npZ = pp[2] + ((dir & 1) ? 0 : dir - 1);
        const key = compute_key(npX, npY, npZ);
        if (visited.has(key)) continue;

        if (solid(env, npX, npY, npZ)) continue;
        if (!solid(env, npX, npY - 1, npZ)) continue;

        visited.set(key, pp);
        next.push([npX, npY, npZ]);
        if (npX === tx && npY === ty && npZ === tz) done = true;
      }
    }
    if (done) break;
    [prev, next] = [next, prev];
    next.length === 0;
  }

  if (!done) return;

  const full: Point[] = [];
  let cur: Point | null | void = [tx, ty, tz];
  while (cur) {
    full.push(cur);
    cur = visited.get(compute_key(cur[0], cur[1], cur[2]));
  }
  full.reverse();

  const result = [full[0]];
  for (let i = 2; i < full.length; i++) {
    const last = result[result.length - 1];
    if (hasDirectPath(env, last, full[i])) continue;
    result.push(full[i - 1]);
  }
  if (full.length > 1) result.push(full[full.length - 1]);

  state.path = result;
  state.path_index = 0;
  state.target = null;
  console.log(state.path);
};

const PIDController = (error: number, derror: number): number => {
  return 1.00 * error + 0.05 * derror;
};

const followPath = (env: TypedEnv, state: PathingState,
                    body: PhysicsState): void => {
  const path = nonnull(state.path);
  if (state.path_index === path.length) { state.path = null; return; }
  const movement = env.movement.get(state.id);
  if (!movement) return;

  const cx = (body.min[0] + body.max[0]) / 2;
  const cz = (body.min[2] + body.max[2]) / 2;

  const node = path[state.path_index];
  const E = state.path_index + 1 === path.length
    ? 0.4 * (1 - (body.max[0] - body.min[0]))
    : 0;
  if (node[0] + E <= body.min[0] && body.max[0] <= node[0] + 1 - E &&
      node[1] + 0 <= body.min[1] && body.max[1] <= node[1] + 1 - 0 &&
      node[2] + E <= body.min[2] && body.max[2] <= node[2] + 1 - E) {
    state.path_index += 1;
  }
  if (state.path_index === path.length) { state.path = null; return; }

  const cur = path[state.path_index];
  let inputX = PIDController(cur[0] + 0.5 - cx, -body.vel[0]);
  let inputZ = PIDController(cur[2] + 0.5 - cz, -body.vel[2]);
  const length = Math.sqrt(inputX * inputX + inputZ * inputZ);
  const normalization = length > 1 ? 1 / length : 1;
  movement.inputX = inputX * normalization;
  movement.inputZ = inputZ * normalization;

  const mesh = env.meshes.get(state.id);
  if (!mesh) return;
  const dx = state.path_index > 0
    ? cur[0] - path[state.path_index - 1][0]
    : cur[0] + 0.5 - cx;
  const dz = state.path_index > 0
    ? cur[2] - path[state.path_index - 1][2]
    : cur[2] + 0.5 - cz;
  mesh.heading = Math.atan2(dx, dz);
};

const runPathing = (env: TypedEnv, state: PathingState): void => {
  if (!state.path && !state.target) return;
  const body = env.physics.get(state.id);
  if (!body) return;

  if (state.target) return findPath(env, state, body);
  if (state.path) return followPath(env, state, body);
};

const Pathing = (env: TypedEnv): Component<PathingState> => ({
  init: () => ({
    id: kNoEntity,
    index: 0,
    path: null,
    path_index: 0,
    target: null,
  }),
  onUpdate: (dt: int, states: PathingState[]) => {
    for (const state of states) runPathing(env, state);
  }
});

// An entity with a MeshState keeps a renderer mesh at its position.

interface MeshState {
  id: EntityId,
  index: int,
  mesh: SpriteMesh | null,
  heading: number | null,
  columns: number,
  column: number,
  frame: number,
  row: number,
};

const Meshes = (env: TypedEnv): Component<MeshState> => ({
  init: () => ({
    id: kNoEntity,
    index: 0,
    mesh: null,
    heading: null,
    columns: 0,
    column: 0,
    frame: 0,
    row: 0,
  }),
  onRemove: (state: MeshState) => { if (state.mesh) state.mesh.dispose(); },
  onRender: (dt: int, states: MeshState[]) => {
    for (const state of states) {
      if (!state.mesh) continue;
      const {x, y, z, h} = env.position.getX(state.id);
      const lit = env.world.isBlockLit(
          Math.floor(x), Math.floor(y), Math.floor(z));
      state.mesh.setPosition(x, y - h / 2, z);
      state.mesh.setLight(lit ? 1 : 0.64);

      if (state.heading !== null) {
        const pos = env.renderer.camera.position;
        const camera_heading = Math.atan2(x - pos[0], z - pos[2]);
        const delta = state.heading - camera_heading;
        state.row = Math.floor(8.5 - 2 * delta / Math.PI) & 3;
        state.mesh.setFrame(state.column + state.row * state.columns);
      }
    }
  },
  onUpdate: (dt: int, states: MeshState[]) => {
    for (const state of states) {
      if (!state.mesh || !state.columns) return;
      const body = env.physics.get(state.id);
      if (!body) return;

      state.column = (() => {
        if (!body.resting[1]) return 1;
        const speed = Vec3.length(body.vel);
        state.frame = speed ? (state.frame + 0.025 * speed) % 4 : 0;
        if (!speed) return 0;
        const value = Math.floor(state.frame);
        return value & 1 ? 0 : (value + 2) >> 1;
      })();
      state.mesh.setFrame(state.column + state.row * state.columns);
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
  onRemove: (state: ShadowState) => { if (state.mesh) state.mesh.dispose(); },
  onRender: (dt: int, states: ShadowState[]) => {
    for (const state of states) {
      if (!state.mesh) state.mesh = env.renderer.addShadowMesh();
      const {x, y, z, w, h} = env.position.getX(state.id);
      const fraction = 1 - (y - 0.5 * h - state.height) / state.extent;
      const size = 0.5 * w * Math.max(0, Math.min(1, fraction));
      state.mesh.setPosition(x, state.height + 0.01, z);
      state.mesh.setSize(size);
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
          if (solid(env, x, h - 1, z)) return h;
        }
        return 0;
      })();
    }
  },
});

// CameraTarget signifies that the camera will follow an entity.

const CameraTarget = (env: TypedEnv): Component => ({
  init: () => ({id: kNoEntity, index: 0}),
  onRender: (dt: int, states: ComponentState[]) => {
    for (const state of states) {
      const {x, y, z, h, w} = env.position.getX(state.id);
      env.setCameraTarget(x, y + h / 3, z);
      const mesh = env.meshes.get(state.id);
      const zoom = env.renderer.camera.safe_zoom;
      if (mesh && mesh.mesh) mesh.mesh.enabled = zoom > 2 * w;
    }
  },
  onUpdate: (dt: int, states: ComponentState[]) => {
    for (const state of states) {
      const {x, y, z} = env.position.getX(state.id);
      env.world.recenter(x, y, z);
    }
  },
});

// Putting it all together:

const safeHeight = (position: PositionState): number => {
  const radius = 0.5 * (position.w + 1);
  const ax = Math.floor(position.x - radius);
  const az = Math.floor(position.z - radius);
  const bx = Math.ceil(position.x + radius);
  const bz = Math.ceil(position.z + radius);

  let height = 0;
  for (let x = ax; x <= bx; x++) {
    for (let z = az; z <= bz; z++) {
      height = Math.max(height, getHeight(x, z));
    }
  }
  return height + 0.5 * (position.h + 1);
};

const addEntity = (env: TypedEnv, image: string,
                   x: number, z: number, h: number, w: number): EntityId => {
  const entity = env.entities.addEntity();
  const position = env.position.add(entity);
  position.x = x + 0.5;
  position.z = z + 0.5;
  position.w = w;
  position.h = h;
  position.y = safeHeight(position);

  const mesh = env.meshes.add(entity);
  const sprite = {url: `images/${image}.png`, size: 1, x: 32, y: 32};
  mesh.mesh = env.renderer.addSpriteMesh(sprite);
  mesh.columns = 3;

  env.physics.add(entity);
  env.movement.add(entity);
  env.shadow.add(entity);
  return entity;
};

const main = () => {
  const env = new TypedEnv('container');

  const player = addEntity(env, 'player', 1, 1, 0.8, 0.6);
  env.inputs.add(player);
  env.target.add(player);

  const follower = addEntity(env, 'follower', 1, 1, 0.5, 0.5);
  env.meshes.getX(follower).heading = 0;
  env.pathing.add(follower);

  const texture = (x: int, y: int, alphaTest: boolean = false,
                   sparkle: boolean = false): Texture => {
    const url = 'images/rhodox-edited.png';
    return {alphaTest, sparkle, url, x, y, w: 16, h: 16};
  };

  const registry = env.registry;
  registry.addMaterialOfColor('blue', [0.1, 0.1, 0.4, 0.6], true);
  registry.addMaterialOfTexture(
    'water', texture(13, 12, false, true), [1, 1, 1, 0.8], true);
  registry.addMaterialOfTexture('leaves', texture(4, 3, true));
  const textures: [string, int, int][] = [
    ['bedrock', 1, 1],
    ['dirt', 2, 0],
    ['grass', 0, 0],
    ['grass-side', 3, 0],
    ['rock', 1, 0],
    ['sand', 0, 11],
    ['snow', 2, 4],
    ['trunk', 5, 1],
    ['trunk-side', 4, 1],
  ];
  for (const [name, x, y] of textures) {
    registry.addMaterialOfTexture(name, texture(x, y));
  }

  const blocks = {
    bedrock: registry.addBlock(['bedrock'], true),
    dirt:    registry.addBlock(['dirt'], true),
    grass:   registry.addBlock(['grass', 'dirt', 'grass-side'], true),
    leaves:  registry.addBlock(['leaves'], true),
    rock:    registry.addBlock(['rock'], true),
    sand:    registry.addBlock(['sand'], true),
    snow:    registry.addBlock(['snow'], true),
    trunk:   registry.addBlock(['trunk', 'trunk-side'], true),
    water:   registry.addBlock(['water', 'blue', 'blue'], false),
  };

  env.blocks = blocks;
  const loadChunkFn = loadChunk(blocks);
  const loadFrontierFn = loadFrontier(blocks);
  env.world.setLoader(blocks.bedrock, loadChunkFn, loadFrontierFn);
  env.refresh();
};

window.onload = main;

export {};
