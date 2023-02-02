#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

#include "base.h"
#include "mesher.h"
#include "worldgen.h"

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

constexpr int kNumChunksToLoadPerFrame  = 1;
constexpr int kNumChunksToMeshPerFrame  = 1;
constexpr int kNumChunksToLightPerFrame = 4;

// Require a layer of air blocks at the top of the world. Doing so simplifies
// our data structures and shaders (for example, a height fits in a uint8_t).
constexpr int kBuildHeight = kWorldHeight - 1;

// Used to mask x- and z-values into a chunk's index range.
constexpr int kChunkMask = kChunkWidth - 1;

constexpr size_t kNumNeighbors = 8;
constexpr Point kNeighbors[kNumNeighbors] = {
  {-1,  0}, { 1,  0}, { 0, -1}, { 0,  1},
  {-1, -1}, {-1,  1}, { 1, -1}, { 1,  1},
};

// Debugging helper for the equilevels optimization.
template <typename T, typename U>
void checkEquilevels(const T& equilevels, const U& voxels) {
  if constexpr (true) return;

  auto count = 0;
  const auto [sx, sy, sz] = voxels.shape;
  for (auto y = 0; y < sy; y++) {
    if (!equilevels[y]) continue;
    const auto base = voxels.get(0, y, 0);
    for (auto x = 0; x < sx; x++) {
      for (auto z = 0; z < sz; z++) {
        assert(voxels.get(x, y, z) == base);
      }
    }
    count++;
  }
  const auto fraction = count / static_cast<double>(sy);
  printf("equilevels: %d/%lu (%f%%)\n", count, sy, fraction);
}

struct World;

struct Chunk {
  Chunk() = default;

  void create(Point p, World* w) {
    assert(w != nullptr);

    point = p;
    world = w;
    neighbors = 0;

    load();

    eachNeighbor([&](Chunk* chunk) {
      chunk->notifyNeighborLoaded();
      neighbors++;
    });
    dirty = stage1_dirty = stage2_dirty = true;
    ready = checkReady();
  }

  void destroy() {
    dropMeshes();
    eachNeighbor([](Chunk* chunk) {
      chunk->notifyNeighborDisposed();
    });
  }

  Block getBlock(int x, int y, int z) {
    assert(0 <= x && x < kChunkWidth);
    assert(0 <= z && z < kChunkWidth);
    assert(0 <= y && y < kBuildHeight);

    return voxels.get(x, y, z);
  }

  bool hasMesh() const {
    return solid || water;
  }

  bool needsRelight() const {
    return stage2_dirty && ready && hasMesh();
  }

  bool needsRemesh() const {
    return dirty && ready;
  }

  void relightChunk() {
    // Called from remeshChunk to set the meshes' light textures, even if
    // !this.needsRelight(). Each step checks a dirty flag, so that's okay.
    stage1_dirty = stage2_dirty = false;
  }

  void remeshChunk() {
    assert(needsRemesh());
    remeshTerrain();
    relightChunk();
    dirty = false;
  }

  void setBlock(int x, int y, int z, Block block) {
    assert(0 <= x && x < kChunkWidth);
    assert(0 <= z && z < kChunkWidth);
    assert(0 <= y && y < kBuildHeight);

    const auto index = voxels.index(x, y, z);
    if (voxels.data[index] == block) return;
    voxels.data[index] = block;

    dirty = stage1_dirty = stage2_dirty = true;
    updateHeightmap(x, z, y, 1, block, index);
    equilevels[y] = 0;

    constexpr auto M = kChunkMask;
    const auto neighbor = [&](int dx, int dz) {
      const auto neighbor = getNeighbor({dx, dz});
      if (neighbor) neighbor->dirty = true;
    };
    if (x == 0) neighbor(-1,  0);
    if (x == M) neighbor( 1,  0);
    if (z == 0) neighbor( 0, -1);
    if (z == M) neighbor( 0,  1);
    if (x == 0 && z == 0) neighbor(-1, -1);
    if (x == 0 && z == M) neighbor(-1,  1);
    if (x == M && z == 0) neighbor( 1, -1);
    if (x == M && z == M) neighbor( 1,  1);
  }

 private:
  struct ChunkItem { Block block; uint8_t index; };

  template <typename T> friend struct Circle;

  Mesher& getMesher() const;
  Chunk* getNeighbor(Point delta) const;

  bool checkReady() const {
    return neighbors == kNumNeighbors;
  }

  void dropMeshes() {
    solid = std::nullopt;
    water = std::nullopt;
  }

  template <typename Fn>
  void eachNeighbor(Fn fn) {
    for (const auto& delta : kNeighbors) {
      const auto neighbor = getNeighbor(delta);
      if (neighbor) fn(neighbor);
    }
  }

  void load() {
    std::array<int, kWorldHeight> mismatches;
    heightmap.data.fill(0);
    mismatches.fill(0);

    static_assert(alignof(ChunkItem) == 1);
    constexpr auto size = sizeof(ChunkItem);
    const auto data = loadChunkData(point.x, point.z);
    const uint8_t* cur = data.start;
    const auto base = reinterpret_cast<const ChunkItem*>(cur);

    for (auto z = 0; z < kChunkWidth; z++) {
      for (auto x = 0; x < kChunkWidth; x++) {
        const auto test = reinterpret_cast<const ChunkItem*>(cur);
        detectMismatches(base, test, mismatches);
        for (auto start = 0; start < kBuildHeight; cur += size) {
          const auto item = *reinterpret_cast<const ChunkItem*>(cur);
          assert(item.index > start);
          setColumn(x, z, start, item.index - start, item.block);
          start = item.index;
        }
        const auto decorations = *(cur++);
        const auto end = cur + size * decorations;
        for (; cur != end; cur += size) {
          const auto item = *reinterpret_cast<const ChunkItem*>(cur);
          setColumn(x, z, item.index, 1, item.block);
          mismatches[item.index + 0]++;
          mismatches[item.index + 1]--;
        }
      }
    }
    assert(cur == data.end);

    auto cur_mismatches = 0;
    for (auto i = 0; i < kWorldHeight; i++) {
      cur_mismatches += mismatches[i];
      assert(cur_mismatches >= 0);
      equilevels[i] = cur_mismatches == 0;
    }
    assert(cur_mismatches == 0);

    checkEquilevels(equilevels, voxels);
  }

  void detectMismatches(const ChunkItem* base, const ChunkItem* test,
                        std::array<int, kWorldHeight>& mismatches) {
    auto matched = true;
    auto base_start = 0;
    auto test_start = 0;

    while (base_start < kBuildHeight) {
      if (matched != (base->block == test->block)) {
        const auto height = std::max(base_start, test_start);
        mismatches[height] += matched ? 1 : -1;
        matched = !matched;
      }

      const auto base_limit = base->index;
      const auto test_limit = test->index;
      if (base_limit <= test_limit) {
        base_start = base_limit;
        base++;
      }
      if (test_limit <= base_limit) {
        test_start = test_limit;
        test++;
      }
    }

    if (!matched) mismatches[kBuildHeight]--;

    assert(base_start == test_start);
  }

  void notifyNeighborDisposed() {
    assert(neighbors > 0);
    neighbors--;
    const auto old = ready;
    ready = checkReady();
    if (old && !ready) dropMeshes();
  }

  void notifyNeighborLoaded() {
    assert(neighbors < kNumNeighbors);
    neighbors++;
    ready = checkReady();
  }

  void remeshTerrain() {
    auto& mesher = getMesher();

    static_assert(sizeof(equilevels[0]) == 1);
    static_assert(sizeof(mesher.equilevels[0]) == 1);
    std::memcpy(&mesher.equilevels[1], &equilevels[0], equilevels.size());

    for (const auto& [delta, dstPos, srcPos, size] : kMesherOffsets) {
      const auto chunk = getNeighbor(delta);
      if (chunk) {
        copyHeightmap(mesher.heightmap, dstPos, chunk->heightmap, srcPos, size);
        copyVoxels(mesher.voxels, dstPos, chunk->voxels, srcPos, size);
      } else {
        zeroHeightmap(mesher.heightmap, dstPos, size);
        zeroVoxels(mesher.voxels, dstPos, size);
      }
      if (chunk != this) {
        copyEquilevels(mesher.equilevels, chunk, srcPos, size);
      }
    }

    checkEquilevels(mesher.equilevels, mesher.voxels);

    mesher.meshChunk();
  }

  void copyHeightmap(MeshTensor2<uint8_t>& dst, Point dstPos,
                     ChunkTensor2<uint8_t>& src, Point srcPos, Point size) {
    for (auto x = 0; x < size.x; x++) {
      for (auto z = 0; z < size.z; z++) {
        const auto sindex = src.index(srcPos.x + x, srcPos.z + z);
        const auto dindex = dst.index(dstPos.x + x, dstPos.z + z);
        dst.data[dindex] = src.data[sindex];
      }
    }
  }

  void copyVoxels(MeshTensor3<Block>& dst, Point dstPos,
                  ChunkTensor3<Block>& src, Point srcPos, Point size) {
    static_assert(sizeof(Block) == 1);
    static_assert(std::remove_reference_t<decltype(src)>::stride[1] == 1);
    static_assert(std::remove_reference_t<decltype(dst)>::stride[1] == 1);
    constexpr size_t bytes = ChunkTensor3<Block>::shape[1];

    for (auto x = 0; x < size.x; x++) {
      for (auto z = 0; z < size.z; z++) {
        const auto sindex = src.index(srcPos.x + x, 0, srcPos.z + z);
        const auto dindex = dst.index(dstPos.x + x, 1, dstPos.z + z);
        memcpy(&dst.data[dindex], &src.data[sindex], bytes);
      }
    }
  }

  void zeroHeightmap(MeshTensor2<uint8_t>& dst, Point dstPos, Point size) {
    for (auto x = 0; x < size.x; x++) {
      for (auto z = 0; z < size.z; z++) {
        dst.set(dstPos.x + x, dstPos.z + z, 0);
      }
    }
  }

  void zeroVoxels(MeshTensor3<Block>& dst, Point dstPos, Point size) {
    static_assert(sizeof(Block) == 1);
    static_assert(ChunkTensor3<Block>::stride[1] == 1);
    constexpr size_t bytes = ChunkTensor3<Block>::shape[1];

    for (auto x = 0; x < size.x; x++) {
      for (auto z = 0; z < size.z; z++) {
        const auto dindex = dst.index(dstPos.x + x, 1, dstPos.z + z);
        memset(&dst.data[dindex], static_cast<int>(Block::Air), bytes);
      }
    }
  }

  void copyEquilevels(MeshTensor1<uint8_t>& dst, Chunk* chunk,
                      Point srcPos, Point size) {
    static_assert(ChunkTensor3<Block>::stride[1] == 1);

    if (chunk == nullptr) {
      for (auto i = 0; i < kWorldHeight; i++) {
        if (dst[i + 1] == 0) continue;
        if (voxels.data[i] != Block::Air) dst[i + 1] = 0;
      }
      return;
    }

    assert(size.x == 1 || size.z == 1);
    const auto stride = chunk->voxels.stride[size.x == 1 ? 2 : 0];
    const auto index  = chunk->voxels.index(srcPos.x, 0, srcPos.z);
    const auto limit  = stride * (size.x == 1 ? size.z : size.x);

    for (auto i = 0; i < kWorldHeight; i++) {
      if (dst[i + 1] == 0) continue;
      const auto base = voxels.data[i];
      if (chunk->equilevels[i] == 1 && chunk->voxels.data[i] == base) continue;
      for (auto offset = 0; offset < limit; offset += stride) {
        if (chunk->voxels.data[index + offset + i] == base) continue;
        dst[i + 1] = 0;
        break;
      }
    }
  }

  void setColumn(int x, int z, int start, int count, Block block) {
    static_assert(sizeof(block) == 1);
    assert(0 <= x && x < kChunkWidth);
    assert(0 <= z && z < kChunkWidth);
    assert(0 <= start && start < kBuildHeight);

    const auto index = voxels.index(x, start, z);
    memset(&voxels.data[index], static_cast<uint8_t>(block), count);
    updateHeightmap(x, z, start, count, block, index);
  }

  void updateHeightmap(int x, int z, int start, int count,
                       Block block, int index) {
    const auto end = start + count;
    const auto offset = heightmap.index(x, z);
    const auto height = heightmap.data[index];
    using T = decltype(height);

    if (block == Block::Air && start < height && height <= end) {
      auto i = 0;
      for (; i < start; i++) {
        if (voxels.data[index - i - 1] != Block::Air) break;
      }
      heightmap.data[offset] = static_cast<T>(start - i);
    } else if (block != Block::Air && height < end) {
      heightmap.data[offset] = static_cast<T>(end);
    }
  }

  // Chunk data layout follows.
  bool dirty;
  bool ready;
  bool stage1_dirty;
  bool stage2_dirty;
  Point point;
  World* world;
  int neighbors;
  std::optional<int> solid;
  std::optional<int> water;
  ChunkTensor1<uint8_t> equilevels;
  ChunkTensor2<uint8_t> heightmap;
  ChunkTensor3<uint8_t> lights;
  ChunkTensor3<Block> voxels;

  DISALLOW_COPY_AND_ASSIGN(Chunk);
};

template <typename T>
struct Circle {
  Circle(double radius) {
    const auto bound = radius * radius;
    const auto floor = static_cast<int>(radius);

    for (auto i = -floor; i <= floor; i++) {
      for (auto j = -floor; j <= floor; j++) {
        const auto point = Point{i, j};
        if (point.normSquared() > bound) continue;
        total++;
      }
    }

    points = std::make_unique<Point[]>(total);
    unused = std::make_unique<T*[]>(total);
    storage = std::make_unique<T[]>(total);
    for (auto i = 0; i < total; i++) {
      unused[i] = &storage[i];
    }

    auto current = 0;
    for (auto i = -floor; i <= floor; i++) {
      for (auto j = -floor; j <= floor; j++) {
        const auto point = Point{i, j};
        if (point.normSquared() > bound) continue;
        points[current++] = point;
      }
    }
    std::stable_sort(&points[0], &points[total], [](Point a, Point b) {
      return a.normSquared() < b.normSquared();
    });

    numDeltas = floor + 1;
    deltas = allocate_array<int>(numDeltas, 0);
    for (auto i = 0; i < total; i++) {
      const auto& point = points[i];
      const auto ax = abs(point.x);
      const auto az = abs(point.z);
      assert(0 <= ax && ax <= floor);
      deltas[ax] = std::max(deltas[ax], az);
    }

    while ((1 << shift) < (2 * floor + 1)) shift++;
    lookup = allocate_array<T*>(1 << (2 * shift), nullptr);
    mask = (1 << shift) - 1;
  }

  void recenter(Point p) {
    if (center == p) return;
    each([&](Point point) {
      const auto diff = point - p;
      const auto ax = abs(diff.x), az = abs(diff.z);
      if (ax < numDeltas && az <= deltas[ax]) return false;

      const auto index = getIndex(point);
      const auto value = lookup[index];
      if (value == nullptr) return false;

      assert(used > 0);
      value->destroy();
      lookup[index] = nullptr;
      unused[--used] = value;
      return false;
    });
    center = p;
  }

  template <typename Fn>
  void each(Fn fn) {
    const auto center = this->center;
    for (auto i = 0; i < total; i++) {
      if (fn(points[i] + center)) break;
    }
  }

  T* get(Point p) const {
    const auto result = lookup[getIndex(p)];
    return result && result->point == p ? result : nullptr;
  }

  template <typename U>
  void set(Point p, const U& context) {
    auto& result = lookup[getIndex(p)];
    assert(result == nullptr);
    assert(used < total);
    result = unused[used++];
    result->create(p, context);
  }

 private:
  int getIndex(Point p) const {
    return ((p.z & mask) << shift) | (p.x & mask);
  }

  Point center;
  int used = 0;
  int mask = 0;
  int shift = 0;
  int total = 0;
  int numDeltas = 0;
  std::unique_ptr<T[]> storage;     // size: total
  std::unique_ptr<T*[]> unused;     // size: total
  std::unique_ptr<Point[]> points; // size: total
  std::unique_ptr<int[]> deltas;    // size: numDeltas
  std::unique_ptr<T*[]> lookup;     // size: 1 << (2 * shift)
};

struct World {
  World(double radius) : chunks(radius), mesher(registry) {}

  Block getBlock(int x, int y, int z) {
    if (y < 0) return Block::Bedrock;
    if (y >= kBuildHeight) return Block::Air;

    const auto cx = x >> kChunkBits;
    const auto cz = z >> kChunkBits;
    const auto chunk = chunks.get({cx, cz});
    if (chunk == nullptr) return Block::Unknown;

    return chunk->getBlock(x & kChunkMask, y, z & kChunkMask);
  }

  void setBlock(int x, int y, int z, Block block) {
    if (!(0 <= y && y < kBuildHeight)) return;

    const auto cx = x >> kChunkBits;
    const auto cz = z >> kChunkBits;
    const auto chunk = chunks.get({cx, cz});

    if (chunk) chunk->setBlock(x & kChunkMask, y, z & kChunkMask, block);
  }

  void recenter(Point p) {
    const auto c = Point{p.x >> kChunkBits, p.z >> kChunkBits};
    chunks.recenter(c);

    auto loaded = 0;
    chunks.each([&](Point point) {
      const auto existing = chunks.get(point);
      if (existing != nullptr) return false;
      chunks.set(point, this);
      return (++loaded) == kNumChunksToLoadPerFrame;
    });
  }

  void remesh() {
    auto lit = 0, meshed = 0, total = 0;
    chunks.each([&](Point point) {
      total++;
      const auto canRelight = lit < kNumChunksToLightPerFrame;
      const auto canRemesh = total <= 9 || meshed < kNumChunksToMeshPerFrame;
      if (!(canRelight || canRemesh)) return true;

      const auto chunk = chunks.get(point);
      if (!chunk) return false;

      if (canRemesh && chunk->needsRemesh()) {
        chunk->remeshChunk();
        meshed++;
      } else if (canRelight && chunk->needsRelight()) {
        chunk->relightChunk();
        lit++;
      }
      return false;
    });
  }

  Registry& mutableRegistry() { return registry; };

 private:
  friend struct Chunk;

  Circle<Chunk> chunks;
  Registry registry;
  Mesher mesher;

  DISALLOW_COPY_AND_ASSIGN(World);
};

Mesher& Chunk::getMesher() const {
  return world->mesher;
}

Chunk* Chunk::getNeighbor(Point delta) const {
  return world->chunks.get(point + delta);
}

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels

//////////////////////////////////////////////////////////////////////////////

std::optional<voxels::World> world;

WASM_EXPORT(initializeWorld)
void initializeWorld(double radius) {
  assert(!world);
  world.emplace(radius + 0.5);
}

WASM_EXPORT(recenterWorld)
void recenterWorld(int x, int z) {
  assert(world);
  world->recenter({x, z});
}

WASM_EXPORT(remeshWorld)
void remeshWorld() {
  assert(world);
  world->remesh();
}

WASM_EXPORT(getBlock)
int getBlock(int x, int y, int z) {
  assert(world);
  return static_cast<int>(world->getBlock(x, y, z));
}

WASM_EXPORT(setBlock)
void setBlock(int x, int y, int z, int block) {
  assert(world);
  world->setBlock(x, y, z, static_cast<voxels::Block>(block));
}

WASM_EXPORT(registerBlock)
void registerBlock(int block, bool opaque, bool solid, int light, int face0,
                   int face1, int face2, int face3, int face4, int face5) {
  using voxels::safe_cast;
  const auto material = [](int x) {
    return voxels::MaybeMaterial{safe_cast<uint8_t>(x)};
  };

  assert(world);
  world->mutableRegistry().addBlock(safe_cast<voxels::Block>(block), {
    opaque, solid, safe_cast<int8_t>(light),
    {material(face0), material(face1), material(face2),
     material(face3), material(face4), material(face5)},
  });
}

WASM_EXPORT(registerMaterial)
void registerMaterial(int material, bool liquid, bool alphaTest, int texture,
                      double r, double g, double b, double a) {
  using voxels::safe_cast;

  assert(world);
  world->mutableRegistry().addMaterial({safe_cast<uint8_t>(material)}, {
    liquid, alphaTest, safe_cast<uint8_t>(texture), {r, g, b, a},
  });
}
