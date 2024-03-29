import {assert, int} from './base.js';

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
  private component: string;
  private definition: Component<T>;
  private lookup: Map<EntityId, T>;
  private states: T[];

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

  each(fn: (state: T) => void): void {
    this.states.forEach(fn);
  }

  every(fn: (state: T) => boolean): boolean {
    return this.states.every(fn);
  }

  some(fn: (state: T) => boolean): boolean {
    return this.states.some(fn);
  }

  add(entity: EntityId): T {
    if (this.lookup.has(entity)) {
      throw new Error(`Duplicate for ${entity}: ${this.component}`);
    }

    const index = int(this.states.length);
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
    const popped = this.states.pop()!;
    assert(popped.index === this.states.length);

    if (popped.id !== entity) {
      const index = state.index;
      assert(index < this.states.length);
      this.states[index] = popped;
      popped.index = index;
    }

    const callback = this.definition.onRemove;
    if (callback) callback(state);
  }

  render(dt: number) {
    const callback = this.definition.onRender;
    if (!callback) throw new Error(`render called: ${this.component}`);
    callback(dt, this.states);
  }

  update(dt: number) {
    const callback = this.definition.onUpdate;
    if (!callback) throw new Error(`update called: ${this.component}`);
    callback(dt, this.states);
  }
};

class EntityComponentSystem {
  private last: EntityId;
  private reusable: EntityId[];
  private components: Map<string, ComponentStore<any>>;
  private onRenders: ComponentStore<any>[];
  private onUpdates: ComponentStore<any>[];

  constructor() {
    this.last = kNoEntity;
    this.reusable = [];
    this.components = new Map();
    this.onRenders = [];
    this.onUpdates = [];
  }

  addEntity(): EntityId {
    if (this.reusable.length > 0) {
      return this.reusable.pop()!;
    }
    return this.last = (this.last + 1) as EntityId;
  }

  removeEntity(entity: EntityId) {
    this.components.forEach(x => x.remove(entity));
    this.reusable.push(entity);
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

  render(dt: number) {
    for (const store of this.onRenders) store.render(dt);
  }

  update(dt: number) {
    for (const store of this.onUpdates) store.update(dt);
  }
};

//////////////////////////////////////////////////////////////////////////////

export {Component, ComponentState, ComponentStore, EntityComponentSystem, EntityId, kNoEntity};
