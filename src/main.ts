import {int, Tensor3, Vec3} from './base.js';
import {Chunk, Env, kChunkBits, kChunkSize, kEmptyBlock} from './engine.js';
import {Component, ComponentState, ComponentStore, EntityId, kNoEntity} from './ecs.js';
import {sweep} from './sweep.js';

//////////////////////////////////////////////////////////////////////////////
// The game code:

class TypedEnv extends Env {
  position: ComponentStore<PositionState>;
  movement: ComponentStore<MovementState>;
  physics: ComponentStore<PhysicsState>;
  target: ComponentStore;
  center: ComponentStore;

  constructor(id: string) {
    super(id);
    const ents = this.entities;
    this.position = ents.registerComponent('position', Position);
    this.movement = ents.registerComponent('movement', Movement(this));
    this.physics = ents.registerComponent('physics', Physics(this));
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
    body.forces[1] += force;
    return;
  }

  const hasAirJumps = state._jumpCount < state.airJumps;
  const canJump = grounded || hasAirJumps;
  if (!canJump) return;

  state._jumped = true;
  state._jumpTimeLeft = state.jumpTime;
  body.impulses[1] += state.jumpImpulse;
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

// Perlin noise implementation:

const perlin2D = () => {
  const getPermutation = (x: int): int[] => {
    const result = [];
    for (let i = 0; i < x; i++) {
      result.push(i);
      const idx = Math.floor(Math.random() * result.length);
      result[result.length - 1] = result[idx];
      result[idx] = i;
    }
    return result;
  };

  const count = 256;
  const table = getPermutation(count);
  table.slice().forEach(x => table.push(x));

  const gradients: [int, int][] = [];
  for (let i = 0; i < count; i++) {
    const angle = 2 * Math.PI * i / count;
    gradients.push([Math.cos(angle), Math.sin(angle)]);
  }

  const dot = (gradient: [int, int], x: number, y: number) => {
    return gradient[0] * x + gradient[1] * y;
  };

  const fade = (x: number): number => {
    return x * x * x * (x * (x * 6 - 15) + 10);
  };

  const lerp = (x: number, a: number, b: number): number => {
    return a + x * (b - a);
  };

  const noise = (x: number, y: number): number => {
    let ix = Math.floor(x);
    let iy = Math.floor(y);
    x -= ix;
    y -= iy;
    ix &= 255;
    iy &= 255;

    const g00 = table[ix +     table[iy    ]];
    const g10 = table[ix + 1 + table[iy    ]];
    const g01 = table[ix +     table[iy + 1]];
    const g11 = table[ix + 1 + table[iy + 1]];

    const n00 = dot(gradients[g00], x,     y    );
    const n10 = dot(gradients[g10], x - 1, y    );
    const n01 = dot(gradients[g01], x,     y - 1);
    const n11 = dot(gradients[g11], x - 1, y - 1);

    const fx = fade(x);
    const fy = fade(y);
    const y1 = lerp(fx, n00, n10);
    const y2 = lerp(fx, n01, n11);
    return lerp(fy, y1, y2);
  };

  return noise;
};

// Putting it all together:

const main = () => {
  const env = new TypedEnv('container');
  //env.renderer.startInstrumentation();

  const player = env.entities.addEntity();
  const position = env.position.add(player);
  position.x = 2;
  position.y = 12;
  position.z = 2;
  position.w = 0.6;
  position.h = 0.8;

  env.physics.add(player);
  env.movement.add(player);
  env.target.add(player);
  env.center.add(player);

  const registry = env.registry;
  const textures = ['dirt', 'grass', 'ground', 'wall'];
  for (const texture of textures) {
    registry.addMaterialOfTexture(texture, `images/${texture}.png`);
  }
  const wall = registry.addBlock(['wall'], true);
  const dirt = registry.addBlock(['dirt'], true);
  const grass = registry.addBlock(['grass', 'dirt', 'dirt'], true);
  const ground = registry.addBlock(['ground', 'dirt', 'dirt'], true);

  const noise = perlin2D();

  loadChunkData = (chunk: Chunk) => {
    if (chunk.cx || chunk.cy || chunk.cz) return;
    chunk.init();
    for (let x = 1; x < kChunkSize - 1; x++) {
      for (let z = 1; z < kChunkSize - 1; z++) {
        if (4 <= x && x < 12 && 4 <= z && z < 12) continue;
        chunk.setBlock(x, 1, z, wall);
        chunk.setBlock(x, 2, z, wall);
      }
    }
    return;

    if (chunk.cy > 0) return;

    chunk.init();

    const size = kChunkSize;
    const pl = size / 4;
    const pr = 3 * size / 4;
    const dx = chunk.cx << kChunkBits;
    const dy = chunk.cy << kChunkBits;
    const dz = chunk.cz << kChunkBits;

    for (let x = 0; x < size; x++) {
      for (let z = 0; z < size; z++) {
        const fx = (x + dx) / 32;
        const fz = (z + dz) / 32;
        const height = 8 * noise(fx, fz);
        const edge = (x === 0) || (z === 0);
        for (let y = 0; y < Math.min(height - dy, size); y++) {
          const h = y + dy;
          const tile = h < -1 ? wall : h < 0 ? dirt : h < 1 ? ground : grass;
          chunk.setBlock(x + dx, y + dy, z + dz, tile);
        }
      }
    }
  };

  env.refresh();
};

window.onload = main;

export {};
