#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

constexpr int32_t kChunkWidth = 16;
constexpr int32_t kWorldHeight = 258;

struct Point {
  int32_t distanceSquared() const {
    return x * x + z * z;
  }

  Point operator+(const Point& o) const {
    return {x + o.x, z + o.z};
  }

  Point operator-(const Point& o) const {
    return {x - o.x, z - o.z};
  }

  bool operator==(const Point& o) const {
    return x == o.x && z == o.z;
  }

  int32_t x;
  int32_t z;
};

struct Chunk {
  void create() {
    ready = false;
    dirty = true;
  }

  void destroy() {}

  bool ready;
  bool dirty;
  Point point;
};

template <typename T>
struct Circle {
  Circle(double radius) {
    const auto bound = radius * radius;
    const auto floor = static_cast<int32_t>(radius);

    for (auto i = -floor; i <= floor; i++) {
      for (auto j = -floor; j <= floor; j++) {
        const auto point = Point{i, j};
        if (point.distanceSquared() > bound) continue;
        points.push_back(point);
      }
    }
    std::sort(points.begin(), points.end(), [](Point a, Point b) {
      return a.distanceSquared() - b.distanceSquared();
    });

    deltas.resize(floor + 1);
    for (const auto& point : points) {
      const auto ax = abs(point.x);
      const auto az = abs(point.z);
      deltas[ax] = std::max(deltas[ax], az);
    }

    storage.resize(points.size());
    freeList.reserve(points.size());
    for (auto& x : storage) {
      freeList.push_back(&x);
    }

    while ((1 << shift) < (2 * floor + 1)) shift++;
    lookup.resize(1 << (2 * shift));
    mask = (1 << shift) - 1;
  }

  void recenter(Point p) {
    if (center == p) return;
    each([&](const Point& point) {
      const auto diff = point - p;
      if (abs(diff.z) <= deltas[abs(diff.x)]) return false;

      const auto index = getIndex(point);
      const auto value = lookup[index];
      if (value == nullptr) return false;

      assert(used > 0);
      value->destroy();
      lookup[index] = nullptr;
      freeList[--used] = value;
      return false;
    });
    center = p;
  }

  template <typename Fn>
  void each(Fn fn) {
    const auto center = this->center;
    for (const auto& point : points) {
      if (fn(point + center)) break;
    }
  }

  T* get(Point p) const {
    const auto result = lookup[getIndex(p)];
    return result && result->point == p ? result : nullptr;
  }

  void set(Point p) {
    auto& result = lookup[getIndex(p)];
    assert(result == nullptr);
    assert(used < freeList.size());
    result = freeList[used++];
    result->create();
  }

 private:
  int32_t getIndex(Point p) {
    return ((p.z & mask) << shift) | (p.x & mask);
  }

  Point center;
  int32_t used = 0;
  int32_t mask = 0;
  int32_t shift = 0;
  std::vector<T> storage;
  std::vector<T*> lookup;
  std::vector<T*> freeList;
  std::vector<Point> points;
  std::vector<int32_t> deltas;
};

struct World {
  World(double radius) : chunks(radius) {}

  void recenter(Point p) {
    const auto c = Point{p.x / kChunkWidth, p.z / kChunkWidth};
    chunks.recenter(c);
  }

 private:
  Circle<Chunk> chunks;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels

//////////////////////////////////////////////////////////////////////////////

#define WASM_EXPORT(X) __attribute__((export_name(#X))) extern "C"

WASM_EXPORT(create_world)
voxels::World* create_world(double radius) {
  return new voxels::World(radius);
}

WASM_EXPORT(update_world)
void update_world(voxels::World* world, int x, int z) {
  world->recenter({x, z});
}

#undef WASM_EXPORT
