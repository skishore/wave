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
#include "renderer.h"
#include "worldgen.h"

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

// The maximum light level: that cast by the sun.
constexpr int kSunlightLevel = 0xf;

// Stage 1 lighting operates on ChunkTensor3 indices. To quickly move from an
// index to a neighbor's index, first mask it and compare against test - if it
// is equal, the neighbor is out of bounds. Otherwise, add diff.
struct LightSpread { int diff; int mask; int test; };
constexpr LightSpread kLightSpread[6] = {
  {-0x0100, 0x0f00, 0x0000},
  {+0x0100, 0x0f00, 0x0f00},
  {-0x1000, 0xf000, 0x0000},
  {+0x1000, 0xf000, 0xf000},
  {-0x0001, 0x00ff, 0x0000},
  {+0x0001, 0x00ff, 0x00ff},
};

struct LightDelta { int location; int value; };

// Lighting cellular automaton buffers.
static NonCopyArray<std::vector<int>, kSunlightLevel - 2> kLightBuffers;
static std::vector<LightDelta> kLightDeltas;
static HashSet<int> kNextDirtyLights;

// If the light at a cell changes from `prev` to `next`, what range
// of lights in neighboring cells may need updating? The bounds are
// inclusive on both sides.
//
// These equations are tricky. We do some casework to derive them:
//
//   - If the light value in a cell drops 8 -> 4, then adjacent cells
//     with lights in {4, 5, 6, 7} may also drop. 8 is too big, since
//     an adjacent cell with the same light has a different source.
//     But 3 is too small: we can cast a light of value 3.
//
//   - If the light value increases from 4 -> 8, then adjacent cells
//     with lights in {3, 4, 5, 6} may increase. 7 is too big, since
//     we can't raise the adjacent light to 8.
//
//   - As a special case, a cell in full sunlight can raise a neighbor
//     (the one right below) to full sunlight, so we include it here.
//     `max - (max < kSunlightLevel ? 1 : 0)` is the max we can cast.
//
// If we allow for blocks that filter more than one light level at a
// time, then the lower bounds fail, but the upper bounds still hold.
//
constexpr int maxUpdatedNeighborLight(int next, int prev) {
  const auto max = std::max(next, prev);
  return max - (max < kSunlightLevel ? 1 : 0) - (next > prev ? 1 : 0);
};
constexpr int minUpdatedNeighborLight(int next, int prev) {
  const auto min = std::min(next, prev);
  return int(min - (next > prev ? 1 : 0));
};

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
constexpr Point kZone[kNumNeighbors + 1] = {
  { 0,  0},
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

//////////////////////////////////////////////////////////////////////////////

struct World;

struct Chunk {
  Chunk() = default;

  void create(Point p, World* w) {
    assert(w != nullptr);
    assert(!solid && !water);

    point = p;
    world = w;
    neighbors = 0;

    instances.clear();
    point_lights.clear();
    stage1_dirty.clear();
    stage1_edges.clear();
    stage2_lights.clear();

    load();
    lightingInit();

    eachNeighbor([&](Chunk* chunk) {
      chunk->notifyNeighborLoaded();
      neighbors++;
    });
    dirty = stage2_dirty = true;
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

  int getLightLevel(int x, int y, int z) {
    assert(0 <= x && x < kChunkWidth);
    assert(0 <= z && z < kChunkWidth);
    assert(0 <= y && y < kWorldHeight);

    const auto index = voxels.index(x, y, z);
    const auto it = stage2_lights.find(index);
    const auto base = it != stage2_lights.end()
        ? it->second : stage1_lights.data[index];

    const auto& data = getRegistry().getBlockUnsafe(voxels.data[index]);
    return std::min(base + (data.mesh ? 1 : 0), kSunlightLevel);
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
    eachNeighbor([](Chunk* chunk) {
      chunk->lightingStage1();
    });
    lightingStage1();
    lightingStage2();
    setLightTexture();
  }

  void remeshChunk() {
    assert(needsRemesh());
    remeshSprites();
    remeshTerrain();
    relightChunk();
    dirty = false;
  }

  void setBlock(int x, int y, int z, Block block) {
    assert(0 <= x && x < kChunkWidth);
    assert(0 <= z && z < kChunkWidth);
    assert(0 <= y && y < kBuildHeight);

    const auto index = voxels.index(x, y, z);
    const auto old_block = voxels.data[index];
    if (old_block == block) return;

    voxels.data[index] = block;
    stage1_dirty.insert(index);
    dirty = stage2_dirty = true;
    updateHeightmap(x, z, y, 1, block, index);
    updateInstance(index, old_block, block);
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

  void setPointLight(int x, int y, int z, int level) {
    assert(0 <= x && x < kChunkWidth);
    assert(0 <= z && z < kChunkWidth);
    assert(0 <= y && y < kWorldHeight);

    const auto index = voxels.index(x, y, z);
    if (level > 0) {
      point_lights[index] = level;
    } else {
      point_lights.erase(index);
    }

    stage1_dirty.insert(index);
    stage2_dirty = true;
  }

 private:
  struct ChunkItem { Block block; uint8_t index; };

  template <typename T> friend struct Circle;

  Mesher& getMesher() const;
  Chunk* getNeighbor(Point delta) const;
  const Registry& getRegistry() const;

  bool checkReady() const {
    return neighbors == kNumNeighbors;
  }

  void dropMeshes() {
    for (auto& [index, instance] : instances) {
      instance.mesh = std::nullopt;
    }
    light = std::nullopt;
    solid = std::nullopt;
    water = std::nullopt;
    dirty = true;
  }

  template <typename Fn>
  void eachNeighbor(Fn fn) {
    for (const auto& delta : kNeighbors) {
      const auto neighbor = getNeighbor(delta);
      if (neighbor) fn(neighbor);
    }
  }

  void lightingInit() {
    // Use for fast bitwise index propagation below.
    static_assert(decltype(stage1_lights)::stride[0] == kLightSpread[1].diff);
    static_assert(decltype(stage1_lights)::stride[1] == kLightSpread[5].diff);
    static_assert(decltype(stage1_lights)::stride[2] == kLightSpread[3].diff);
    static_assert(decltype(heightmap)::stride[0] == kLightSpread[1].diff >> 8);
    static_assert(decltype(heightmap)::stride[1] == kLightSpread[3].diff >> 8);

    const auto& registry = getRegistry();
    stage1_lights.data.fill(kSunlightLevel);

    for (auto x = 0; x < kChunkWidth; x++) {
      for (auto z = 0; z < kChunkWidth; z++) {
        const auto index = (x << 8) | (z << 12);
        const auto height = heightmap.data[index >> 8];

        for (auto i = 0; i < 4; i++) {
          const auto& spread = kLightSpread[i];
          if ((index & spread.mask) == spread.test) continue;

          const auto neighbor_index = index + spread.diff;
          const auto neighbor_height = heightmap.data[neighbor_index >> 8];
          for (auto y = height; y < neighbor_height; y++) {
            stage1_dirty.insert(neighbor_index + y);
          }
        }

        if (height > 0) {
          const auto below = index + height - 1;
          const auto& data = registry.getBlockUnsafe(voxels.data[below]);
          if (!data.opaque) stage1_dirty.insert(below);
          static_assert(sizeof(stage1_lights.data[0]) == 1);
          memset(&stage1_lights.data[index], 0, height);
        }
      }
    }
  }

  void lightingStage1() {
    if (stage1_dirty.empty()) return;

    // Stage 1 lighting operates on "index" values, which are (x, y, z)
    // coordinates represented as indices into our {lights, voxel} Tensor3.
    const auto& registry = getRegistry();
    auto& prev = stage1_dirty;
    auto& next = kNextDirtyLights;
    next.clear();

    // Returns true if the given index is on an x-z edge of the chunk.
    const auto edge = [&](int index) {
      const auto x_edge = (((index >> 8)  + 1) & 0xf) < 2;
      const auto z_edge = (((index >> 12) + 1) & 0xf) < 2;
      return x_edge || z_edge;
    };

    // Returns the updated lighting value at the given index. Note that we
    // can never use the `prev` light value in this computation: it can be
    // arbitrarily out-of-date since the chunk contents can change.
    const auto query = [&](int index) {
      const auto& data = registry.getBlockUnsafe(voxels.data[index]);
      const auto from_block = static_cast<int>(data.light);
      if (from_block < 0) return 0;

      const auto it = point_lights.find(index);
      const auto from_point = it != point_lights.end() ? it->second : 0;
      const auto base = std::max(from_block, from_point);

      const auto height = heightmap.data[index >> 8];
      if ((index & 0xff) >= height) return kSunlightLevel;

      auto max_neighbor = base + 1;
      for (const auto& spread : kLightSpread) {
        if ((index & spread.mask) == spread.test) continue;
        const auto neighbor = stage1_lights.data[index + spread.diff];
        if (neighbor > max_neighbor) max_neighbor = neighbor;
      }
      return max_neighbor - 1;
    };

    // Enqueues new indices that may be affected by the given change.
    const auto enqueue = [&](int index, int hi, int lo) {
      for (const auto& spread : kLightSpread) {
        if ((index & spread.mask) == spread.test) continue;
        const auto neighbor_index = index + spread.diff;
        const auto neighbor = stage1_lights.data[neighbor_index];
        if (lo <= neighbor && neighbor <= hi) next.insert(neighbor_index);
      }
    };

    while (!prev.empty()) {
      for (const auto index : prev) {
        const auto prev_level = stage1_lights.data[index];
        const auto next_level = query(index);
        if (next_level == prev_level) continue;

        stage1_lights.data[index] = static_cast<uint8_t>(next_level);

        if (edge(index)) {
          // The edge lights map only contains cells on the edge that are not
          // at full sunlight, since the heightmap takes care of the rest.
          const auto next_in_map = 1 < next_level && next_level < kSunlightLevel;
          const auto prev_in_map = 1 < prev_level && prev_level < kSunlightLevel;
          if (next_in_map != prev_in_map) {
            if (next_in_map) stage1_edges.insert(index);
            else stage1_edges.erase(index);
          }
        }

        const auto hi = maxUpdatedNeighborLight(next_level, prev_level);
        const auto lo = minUpdatedNeighborLight(next_level, prev_level);
        enqueue(index, hi, lo);
      }
      std::swap(prev, next);
      next.clear();
    }

    assert(stage1_dirty.empty());
    eachNeighbor([](Chunk* chunk) { chunk->stage2_dirty = true; });
  }

  void lightingStage2() {
    if (!(ready && stage2_dirty)) return;

    const auto& registry = getRegistry();
    const auto opaque = [&](Block block) {
      return registry.getBlockUnsafe(block).opaque;
    };

    const auto getIndex = [](Point delta) {
      return (delta.x + 1) | ((delta.z + 1) << 2);
    };
    NonCopyArray<Chunk*, 16> zone;
    for (const auto& delta : kZone) {
      const auto neighbor = getNeighbor(delta);
      zone[getIndex(delta)] = neighbor;
      assert(neighbor != nullptr);
    }

    // Stage 1 lighting tracks nodes by "index", where an index can be used
    // to look up a chunk (x, y, z) coordinate in a Tensor3. Stage 2 lighting
    // deals with multiple chunks, so we deal with "locations". The first 16
    // bits of a location are an index; bits 16:18 are a chunk x coordinate,
    // and bits 18:20 are a chunk z coordinate.
    //
    // To keep the cellular automaton as fast as possible, we update stage 1
    // lighting in place. We must undo these changes at the end of this call,
    // so we track a list of (location, previous value) pairs in `deltas` as
    // we make the updates.
    //
    // Cells at a light level of i appear in buffer[kSunlightLevel - i - 1].
    // Cells at a light level of {0, 1} don't propagate, so we drop them.
    static_assert(kSunlightLevel > 2);
    static_assert(kLightBuffers.size() == kSunlightLevel - 2);
    for (auto& buffer : kLightBuffers) buffer.clear();
    kLightDeltas.clear();

    for (const auto& delta : kZone) {
      const auto chunk = zone[getIndex(delta)];
      const auto& light = chunk->stage1_lights.data;
      const auto& heightmap = chunk->heightmap.data;

      for (auto i = 0; i < 4; i++) {
        const auto& spread = kLightSpread[i];
        assert(spread.mask == 0x0f00 || spread.mask == 0xf000);
        const auto dx = spread.mask == 0x0f00 ? spread.diff >> 8  : 0;
        const auto dz = spread.mask == 0xf000 ? spread.diff >> 12 : 0;
        const auto n = delta + Point{dx, dz};
        if (!(-1 <= n.x && n.x <= 1 && -1 <= n.z && n.z <= 1)) continue;

        const auto ni = getIndex(n);
        const auto neighbor_union = ni << 16;
        const auto neighbor_chunk = zone[ni];
        const auto& neighbor_voxel = neighbor_chunk->voxels.data;
        auto& neighbor_light = neighbor_chunk->stage1_lights.data;

        // Update the neighbor cell's light value to `level`, if it is greater
        // than the neighbor's current light value.
        const auto propagate = [&](int level, int neighbor_index) {
          const auto neighbor_level = neighbor_light[neighbor_index];
          if (level <= neighbor_level) return;
          if (!neighbor_level && opaque(neighbor_voxel[neighbor_index])) return;

          const auto neighbor_location = neighbor_index | neighbor_union;
          neighbor_light[neighbor_index] = static_cast<uint8_t>(level);
          kLightDeltas.push_back({neighbor_location, neighbor_level});
          if (level <= 1) return;

          const auto buffer = kSunlightLevel - level - 1;
          kLightBuffers[buffer].push_back(neighbor_location);
        };

        // Propagate light from the sparse edge map.
        for (const auto index : chunk->stage1_edges) {
          if ((index & spread.mask) != spread.test) continue;
          const auto neighbor_index = index ^ spread.mask;
          propagate(light[index] - 1, neighbor_index);
        }

        auto offset = 0;
        const auto source = spread.test;
        const auto target = source ^ spread.mask;
        const auto stride = spread.mask == 0x0f00 ? 0x1000 : 0x0100;
        const auto& neighbor_heightmap = neighbor_chunk->heightmap.data;

        // Propagate light from fully-lit cells on the edge, using heights.
        for (auto j = 0; j < kChunkWidth; j++, offset += stride) {
          const auto height = heightmap[(source + offset) >> 8];
          const auto neighbor_height = neighbor_heightmap[(target + offset) >> 8];
          for (auto y = height; y < neighbor_height; y++) {
            const auto neighbor_index = target + offset + y;
            propagate(kSunlightLevel - 1, neighbor_index);
          }
        }
      }
    }

    // Returns the taxicab distance from the location to the center chunk.
    const auto distance = [](int location) {
      const auto cx = (location >> 16) & 0x3;
      const auto x  = (location >> 8 ) & 0xf;
      const auto dx = cx == 0 ? 16 - x : cx == 1 ? 0 : x - 31;

      const auto cz = (location >> 18) & 0x3;
      const auto z  = (location >> 12) & 0xf;
      const auto dz = cz == 0 ? 16 - z : cz == 1 ? 0 : z - 31;

      return dx + dz;
    };

    // Returns the given location, shifted by the delta. If the shift is out
    // of bounds any direction, it'll return -1.
    const auto shift = [](int location, const LightSpread& spread) {
      const auto [diff, mask, test] = spread;
      if ((location & mask) != test) return int(location + diff);
      switch (mask) {
        case 0x00ff: return -1;
        case 0x0f00: {
          const auto x = ((location >> 16) & 0x3) + (diff >> 8);
          const auto z = ((location >> 18) & 0x3);
          if (!(0 <= x && x <= 2)) return -1;
          return ((location & 0xffff) ^ mask) | (x << 16) | (z << 18);
        }
        case 0xf000: {
          const auto x = ((location >> 16) & 0x3);
          const auto z = ((location >> 18) & 0x3) + (diff >> 12);
          if (!(0 <= z && z <= 2)) return -1;
          return ((location & 0xffff) ^ mask) | (x << 16) | (z << 18);
        }
        default: assert(false);
      }
      return -1;
    };

    constexpr int max = static_cast<int>(kSunlightLevel - 2);
    for (auto level = max; level > 0; level--) {
      const auto prev = &kLightBuffers[max - level];
      const auto next = level > 1 ? &kLightBuffers[max - level + 1] : nullptr;
      const auto prev_level = level + 1;

      for (const auto location : *prev) {
        if (distance(location) > level) continue;
        const auto chunk = zone[location >> 16];
        const auto current_level = chunk->stage1_lights.data[location & 0xffff];
        if (current_level != prev_level) continue;

        for (const auto& spread : kLightSpread) {
          const auto neighbor_location = shift(location, spread);
          if (neighbor_location < 0) continue;

          const auto neighbor_chunk = zone[neighbor_location >> 16];
          const auto& neighbor_voxel = neighbor_chunk->voxels.data;
          auto& neighbor_light = neighbor_chunk->stage1_lights.data;

          const auto neighbor_index = neighbor_location & 0xffff;
          const auto neighbor_level = neighbor_light[neighbor_index];
          if (level <= neighbor_level) continue;
          if (!neighbor_level && opaque(neighbor_voxel[neighbor_index])) continue;

          neighbor_light[neighbor_index] = static_cast<uint8_t>(level);
          kLightDeltas.push_back({neighbor_location, neighbor_level});
          if (next) next->push_back(neighbor_location);
        }
      }
    }

    const auto test = getIndex({0, 0});
    const auto& input = stage1_lights.data;
    auto& output = stage2_lights;
    output.clear();

    for (const auto& delta : kLightDeltas) {
      if ((delta.location >> 16) != test) continue;
      const auto index = delta.location & 0xffff;
      output[index] = input[index];
    }
    for (auto it = kLightDeltas.rbegin(); it != kLightDeltas.rend(); it++) {
      const auto location = it->location;
      const auto value = static_cast<uint8_t>(it->value);
      zone[location >> 16]->stage1_lights.data[location & 0xffff] = value;
    }
    stage2_dirty = false;
  }

  void setLightTexture() {
    if (!hasMesh()) return;

    kLightDeltas.clear();
    for (const auto& pair : stage2_lights) {
      kLightDeltas.push_back({pair.first, stage1_lights.data[pair.first]});
      stage1_lights.data[pair.first] = static_cast<uint8_t>(pair.second);
    }

    light.emplace(stage1_lights);
    if (solid) solid->setLight(*light);
    if (water) water->setLight(*light);

    for (auto& [index, instance] : instances) {
      if (!instance.mesh) continue;
      const auto base = stage1_lights.data[index];
      instance.mesh->setLight(std::min(base + 1, kSunlightLevel));
    }

    for (const auto& delta : kLightDeltas) {
      stage1_lights.data[delta.location] = static_cast<uint8_t>(delta.value);
    }
  }

  void load() {
    NonCopyArray<int, kWorldHeight> mismatches;
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
          const auto index = voxels.index(x, item.index, z);
          const auto old_block = voxels.data[index];
          setColumn(x, z, item.index, 1, item.block);
          updateInstance(index, old_block, item.block);
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
                        NonCopyArray<int, kWorldHeight>& mismatches) {
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

  void remeshSprites() {
    static_assert(decltype(voxels)::stride[0] == 0x0100);
    static_assert(decltype(voxels)::stride[1] == 0x0001);
    static_assert(decltype(voxels)::stride[2] == 0x1000);

    const auto bx = point.x << kChunkBits;
    const auto bz = point.z << kChunkBits;

    for (auto& [index, instance] : instances) {
      if (instance.mesh) continue;
      const auto x = (index >> 8 ) & 0xf;
      const auto y = (index >> 0 ) & 0xff;
      const auto z = (index >> 12) & 0xf;
      instance.mesh.emplace(instance.block, x + bx, y, z + bz);
    }
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

    const auto mesh = [&](auto& mesh, const auto& quads, int phase) {
      if (quads.empty()) return mesh.reset();
      mesh ? mesh->setGeometry(quads) : void(mesh.emplace(quads, phase));
      mesh->setPosition(point.x << kChunkBits, 0, point.z << kChunkBits);
    };
    mesher.meshChunk();
    mesh(solid, mesher.solid_geo, 0);
    mesh(water, mesher.water_geo, 1);
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

    const auto light = getRegistry().getBlock(block).light;
    if (light > 0) {
      for (auto i = 0; i < count; i++) {
        stage1_dirty.insert(index + i);
      }
    }

    updateHeightmap(x, z, start, count, block, index);
  }

  void updateHeightmap(int x, int z, int start, int count,
                       Block block, int index) {
    const auto end = start + count;
    const auto offset = heightmap.index(x, z);
    const auto height = heightmap.data[offset];
    using T = decltype(height);

    if (block == Block::Air && start < height && height <= end) {
      auto i = 0;
      for (; i < start; i++) {
        if (voxels.data[index - i - 1] != Block::Air) break;
      }
      heightmap.data[offset] = static_cast<T>(start - i);
    } else if (block != Block::Air && height <= end) {
      heightmap.data[offset] = static_cast<T>(end);
    }
  }

  void updateInstance(int index, Block old_block, Block new_block) {
    const auto& registry = getRegistry();
    const auto& old_data = registry.getBlockUnsafe(old_block);
    const auto& new_data = registry.getBlockUnsafe(new_block);

    if (new_data.mesh) {
      auto& instance = instances[index];
      instance.block = new_block;
      instance.mesh = std::nullopt;
    } else if (old_data.mesh) {
      instances.erase(index);
    }
  }

  // Cellular automaton lighting is the most complex and expensive logic here.
  // The main problem here is to get lighting to work across multiple chunks.
  // We use the fact that the max light level is smaller than a chunk's width.
  //
  // When we traverse the voxel graph to propagate lighting values, we always
  // track voxels by their index in a chunk. The index is just a 16-bit int,
  // and we can extract (x, y, z) coordinates from it or compute neighboring
  // indices with simple arithmetic.
  //
  // Stage 1 lighting is chunk-local. It assumes that all neighboring chunks
  // are completely dark and propagates sunlight within this chunk. When we
  // edit blocks in a chunk, we only need to recompute its stage 1 lighting -
  // never any neighbors'. We use an incremental algorithm to compute these
  // values, tracking a list of dirty sources when we edit the chunk. We store
  // stage 1 lights in a dense array.
  //
  // When we update stage 1 lighting, we also keep track of "edges": blocks on
  // the x-z boundary of the chunk that could shine light into neighbors in
  // other chunks. The edge map is sparse: it only includes edge voxels with
  // light values x where 1 < x < kSunlightMax. The vast majority of the edge
  // voxels have light values equal to kSunlightMax, and these are implicit in
  // the heightmap, so we save memory by skipping those.
  //
  // Stage 2 lighting includes neighboring voxels. To compute it for a given
  // chunk, we load the chunk and its neighbors and propagate the neighbors'
  // edge lighting (including the implicit lights implied by the heightmap).
  // We store stage 2 lights sparsely, as a delta on stage 1 lights.

  // Basic chunk metadata.
  bool dirty;
  bool ready;
  bool stage2_dirty;
  Point point;
  World* world;
  int neighbors;

  // JS renderer resources.
  std::optional<LightTexture> light;
  std::optional<VoxelMesh> solid;
  std::optional<VoxelMesh> water;

  struct Instance {
    Block block;
    std::optional<InstancedMesh> mesh;
  };
  HashMap<int, Instance> instances;

  // Lighting; the stage 1 light array comes later.
  HashSet<int> stage1_dirty;
  HashSet<int> stage1_edges;
  HashMap<int, int> stage2_lights;
  HashMap<int, int> point_lights;

  // Large data arrays, in increasing order of size.
  ChunkTensor1<uint8_t> equilevels;
  ChunkTensor2<uint8_t> heightmap;
  ChunkTensor3<uint8_t> stage1_lights;
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

    const auto cx = x >> kChunkBits, xm = x & kChunkMask;
    const auto cz = z >> kChunkBits, zm = z & kChunkMask;
    const auto chunk = chunks.get({cx, cz});

    return chunk ? chunk->getBlock(xm, y, zm) : Block::Unknown;
  }

  int getLightLevel(int x, int y, int z) {
    if (y < 0) return 0;
    if (y >= kWorldHeight) return kSunlightLevel;

    const auto cx = x >> kChunkBits, xm = x & kChunkMask;
    const auto cz = z >> kChunkBits, zm = z & kChunkMask;
    const auto chunk = chunks.get({cx, cz});

    return chunk ? chunk->getLightLevel(xm, y, zm) : kSunlightLevel;
  }

  void setBlock(int x, int y, int z, Block block) {
    if (!(0 <= y && y < kBuildHeight)) return;

    const auto cx = x >> kChunkBits, xm = x & kChunkMask;
    const auto cz = z >> kChunkBits, zm = z & kChunkMask;
    const auto chunk = chunks.get({cx, cz});

    if (chunk) chunk->setBlock(xm, y, zm, block);
  }

  void setPointLight(int x, int y, int z, int level) {
    if (!(0 <= y && y < kWorldHeight)) return;

    const auto cx = x >> kChunkBits, xm = x & kChunkMask;
    const auto cz = z >> kChunkBits, zm = z & kChunkMask;
    const auto chunk = chunks.get({cx, cz});

    // We can't support a block light of kSunlightLevel until we have separate
    // channels for block light and sunlight.
    level = std::min(level, static_cast<int>(kSunlightLevel - 1));
    if (chunk) chunk->setPointLight(xm, y, zm, level);
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

const Registry& Chunk::getRegistry() const {
  return world->registry;
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

WASM_EXPORT(getLightLevel)
int getLightLevel(int x, int y, int z) {
  assert(world);
  return world->getLightLevel(x, y, z);
}

WASM_EXPORT(setBlock)
void setBlock(int x, int y, int z, int block) {
  assert(world);
  world->setBlock(x, y, z, static_cast<voxels::Block>(block));
}

WASM_EXPORT(setPointLight)
void setPointLight(int x, int y, int z, int level) {
  assert(world);
  world->setPointLight(x, y, z, level);
}

WASM_EXPORT(registerBlock)
void registerBlock(
    int block, bool mesh, bool opaque, bool solid, int light,
    int face0, int face1, int face2, int face3, int face4, int face5) {
  using voxels::safe_cast;
  const auto material = [](int x) {
    return voxels::MaybeMaterial{safe_cast<uint8_t>(x)};
  };

  assert(world);
  world->mutableRegistry().addBlock(safe_cast<voxels::Block>(block), {
    mesh, opaque, solid, safe_cast<int8_t>(light),
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
