#include "worldgen.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "base.h"
#include "open-simplex-2d.h"

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

namespace {

//////////////////////////////////////////////////////////////////////////////

constexpr auto kIslandRadius = 1024;
constexpr auto kSeaLevel = kWorldHeight / 4;

constexpr auto kCaveLevels = 3;
constexpr auto kCaveDeltaY = 0;
constexpr auto kCaveHeight = 8;
constexpr auto kCaveRadius = 16;
constexpr auto kCaveCutoff = 0.25;
constexpr auto kCaveWaveHeight = 16;
constexpr auto kCaveWaveRadius = 256;

struct HeightmapResult {
  Block block;
  int height;
  int snow_depth;
};

static ChunkData chunkData;
static std::vector<uint32_t> heightmapData;

static constexpr uint32_t kSeed = 0;
static uint32_t g_noise2DSeed = kSeed;

const std::vector<uint16_t>& getRandomness() {
  static std::vector<uint16_t> result;
  if (result.empty()) {
    constexpr size_t size = 1 << 20;
    result.reserve(size);
    srand(kSeed + 17);
    for (auto i = 0; i < size; i++) {
      result.push_back(static_cast<uint16_t>(rand()));
    }
  }
  return result;
}

const uint16_t hash_point(int x, int z) {
  const auto& randomness = getRandomness();
  constexpr size_t bits = 10;
  x &= (1 << bits) - 1;
  z &= (1 << bits) - 1;
  return randomness[(x << bits) | z];
}

auto MinetestNoise2D(double offset, double scale, double spread,
                     size_t octaves, double persistence, double lacunarity) {
  const auto inverse_spread = 1 / spread;
  auto components = std::make_unique<Noise2D[]>(octaves);
  for (auto i = 0; i < octaves; i++) {
    new (&components[i]) Noise2D(g_noise2DSeed++);
  }

  return [=, components = std::move(components)](double x, double y) {
    auto s = inverse_spread, g = 1.0, result = 0.0;
    for (auto i = 0; i < octaves; i++) {
      result += g * components[i].query(x * s, y * s);
      g *= persistence;
      s *= lacunarity;
    }
    return scale * result + offset;
  };
}

auto RidgeNoise(size_t octaves, double persistence, double spread) {
  const auto inverse_spread = 1 / spread;
  auto components = std::make_unique<Noise2D[]>(octaves);
  for (auto i = 0; i < octaves; i++) {
    new (&components[i]) Noise2D(g_noise2DSeed++);
  }

  return [=, components = std::move(components)](double x, double y) {
    auto s = inverse_spread, g = 1.0, result = 0.0;
    for (auto i = 0; i < octaves; i++) {
      result += (1 - abs(components[i].query(x * s, y * s))) * g;
      g *= persistence;
      s *= 2;
    }
    return result;
  };
}

const auto mgv7_np_cliff_select    = MinetestNoise2D(0, 1,  512, 4, 0.7, 2.0);
const auto mgv7_np_mountain_select = MinetestNoise2D(0, 1,  512, 4, 0.7, 2.0);
const auto mgv7_np_terrain_ground  = MinetestNoise2D(2, 8,  512, 6, 0.6, 2.0);
const auto mgv7_np_terrain_cliff   = MinetestNoise2D(8, 16, 512, 6, 0.6, 2.0);

const auto mgv7_mountain_ridge = RidgeNoise(4, 0.5, 500);

static_assert(kCaveLevels == 3);
const Noise2D cave_noises[2 * kCaveLevels] = {
  Noise2D(g_noise2DSeed++), Noise2D(g_noise2DSeed++),
  Noise2D(g_noise2DSeed++), Noise2D(g_noise2DSeed++),
  Noise2D(g_noise2DSeed++), Noise2D(g_noise2DSeed++),
};

HeightmapResult* heightmap(int x, int z) {
  static HeightmapResult kHeightmapResult;

  const auto base = sqrt(x * x + z * z) / kIslandRadius;
  const auto falloff = 16 * base * base;
  if (falloff >= kSeaLevel) {
    kHeightmapResult.block = Block::Bedrock;
    kHeightmapResult.height = 0;
    kHeightmapResult.snow_depth = 0;
    return &kHeightmapResult;
  }

  const auto cliff_select = mgv7_np_cliff_select(x, z);
  const auto cliff_x = std::clamp(16 * abs(cliff_select) - 4, 0.0, 1.0);

  const auto mountain_select = mgv7_np_mountain_select(x, z);
  const auto mountain_x = sqrt(fmax(8 * mountain_select, 0.0));

  const auto cliff = cliff_x - mountain_x;
  const auto mountain = -cliff;

  const auto height_ground = mgv7_np_terrain_ground(x, z);
  const auto height_cliff = cliff > 0
    ? mgv7_np_terrain_cliff(x, z)
    : height_ground;
  const auto height_mountain = mountain > 0
    ? height_ground + 64 * pow((mgv7_mountain_ridge(x, z) - 1.25), 1.5)
    : height_ground;

  const auto height = [&]{
    if (height_mountain > height_ground) {
      return height_mountain * mountain + height_ground * (1 - mountain);
    } else if (height_cliff > height_ground) {
      return height_cliff * cliff + height_ground * (1 - cliff);
    }
    return height_ground;
  }();

  const auto truncated = static_cast<int>(height - falloff);
  const auto abs_height = truncated + kSeaLevel;
  const auto block = [&]{
    if (truncated < -1) return Block::Dirt;
    if (height_mountain > height_ground) {
      const auto base = height - (72 - 8 * mountain);
      return base > 0 ? Block::Snow : Block::Stone;
    }
    if (height_cliff > height_ground) return Block::Dirt;
    return truncated < 1 ? Block::Sand : Block::Grass;
  }();

  kHeightmapResult.block = block;
  kHeightmapResult.height = abs_height;
  kHeightmapResult.snow_depth = block == Block::Snow
    ? static_cast<int>(height - (72 - 8 * mountain))
    : 0;
  return &kHeightmapResult;
}

int carveCaves(int x, int z, int limit, int height, ChunkData* data) {
  auto max = 0;
  auto min = kWorldHeight;
  const auto start = kSeaLevel - 1.0 * kCaveDeltaY * (kCaveLevels - 1) / 2;

  for (auto i = 0; i < kCaveLevels; i++) {
    const auto& carver_noise = cave_noises[2 * i + 0];
    const auto& height_noise = cave_noises[2 * i + 1];
    const auto carver = carver_noise.query(
        1.0 * x / kCaveRadius, 1.0 * z / kCaveRadius);
    if (carver <= kCaveCutoff) continue;

    const auto dy = start + i * kCaveDeltaY;
    const auto height = height_noise.query(
        1.0 * x / kCaveWaveRadius, 1.0 * z / kCaveWaveRadius);
    const auto offset = static_cast<int>(dy + kCaveWaveHeight * height);
    const auto blocks = static_cast<int>((carver - kCaveCutoff) * kCaveHeight);

    const auto ay = offset - blocks;
    const auto by = std::min(offset + blocks + 3, limit);
    for (auto i = ay; i < by; i++) {
      data->decorate(Block::Air, i);
    }
    max = std::max(max, by);
    min = std::min(min, ay);
  }

  if (max < height && max < limit && (hash_point(x, z) & 63) == 4) {
    data->decorate(Block::Fungi, min);
  }
  return max;
}

void loadChunk(int x, int z, ChunkData* data) {
  static_assert(isPowTwo(kChunkWidth));

  constexpr auto kBuffer = 1;
  constexpr auto kExpandedWidth = kChunkWidth + 2 * kBuffer;
  constexpr auto max = std::numeric_limits<int>::max();
  constexpr int kNeighborOffsets[5] =
      {0, 1, -1, kExpandedWidth, -kExpandedWidth};

  static Point lastChunk = {max, max};
  static NonCopyArray<HeightmapResult, kExpandedWidth * kExpandedWidth> raw;

  const auto cx = (x & ~(kChunkWidth - 1)) / kChunkWidth;
  const auto cz = (z & ~(kChunkWidth - 1)) / kChunkWidth;
  const auto dx = cx * kChunkWidth - kBuffer;
  const auto dz = cz * kChunkWidth - kBuffer;
  const auto chunk = Point{cx, cz};

  if (lastChunk != chunk) {
    lastChunk = chunk;
    for (auto j = 0; j < kExpandedWidth; j++) {
      for (auto i = 0; i < kExpandedWidth; i++) {
        raw[i + j * kExpandedWidth] = *heightmap(i + dx, j + dz);
      }
    }
  }

  const auto index = (x - dx) + (z - dz) * kExpandedWidth;
  const auto& cache = raw[index];
  if (cache.block == Block::Snow) {
    data->push(Block::Stone, cache.height - cache.snow_depth);
  } else if (cache.block != Block::Stone) {
    data->push(Block::Stone, cache.height - 4);
    data->push(Block::Dirt,  cache.height - 1);
  }
  data->push(cache.block, cache.height);
  data->push(Block::Water, kSeaLevel);

  auto limit = kWorldHeight - 1;
  for (const auto offset : kNeighborOffsets) {
    const auto neighbor_height = raw[index + offset].height;
    if (neighbor_height >= kSeaLevel) continue;
    limit = std::min(limit, neighbor_height - 1);
  }
  const auto cave_height = carveCaves(x, z, limit, cache.height, data);

  if (cache.block == Block::Grass && cave_height < cache.height) {
    const auto hash = hash_point(x, z) & 63;
    if (hash < 2) data->decorate(Block::Bush, cache.height);
    else if (hash < 4) data->decorate(Block::Rock, cache.height);
  }
  data->commit();
}

uint32_t packHeightmapData(int x, int z) {
  static_assert(sizeof(Block) == 1);
  static_assert(static_cast<uint32_t>(Block::Air) == 0);

  const auto result = heightmap(x, z);
  const Block solid_block = result->block;
  const uint8_t solid_height =
      static_cast<uint8_t>(std::clamp(result->height, 0x00, 0xff));

  if (solid_height >= kSeaLevel) {
    return (static_cast<uint32_t>(solid_block)  << 0) |
           (static_cast<uint32_t>(solid_height) << 8);
  }
  return (static_cast<uint32_t>(solid_block)  << 0 ) |
         (static_cast<uint32_t>(solid_height) << 8 ) |
         (static_cast<uint32_t>(Block::Water) << 16) |
         (static_cast<uint32_t>(kSeaLevel)    << 24);
}

//////////////////////////////////////////////////////////////////////////////

} // namespace

//////////////////////////////////////////////////////////////////////////////

void ChunkData::commit() {
  push(Block::Air, kWorldHeight);
  serialized.push_back(static_cast<uint8_t>(decorated));
  for (auto i = 0; i < decorated; i++) {
    const auto height = decorations[i].height;
    auto& decoration = decorations[height];
    serialized.push_back(static_cast<uint8_t>(decoration.block));
    serialized.push_back(static_cast<uint8_t>(height));
    clearDecoration(decoration);
  }
  height = 0;
  decorated = 0;
}

void ChunkData::decorate(Block block, int height) {
  if (!(0 <= height && height < kWorldHeight - 1)) return;

  auto& decoration = decorations[height];
  decoration.block = block;
  if (decoration.decorated) return;

  decoration.decorated = true;
  decorations[decorated++].height = static_cast<uint8_t>(height);
}

void ChunkData::push(Block block, int limit) {
  limit = std::min(limit, kWorldHeight - 1);
  if (limit <= height) return;
  serialized.push_back(static_cast<uint8_t>(block));
  serialized.push_back(static_cast<uint8_t>(limit));
  height = limit;
}

void ChunkData::reset() {
  for (auto i = 0; i < decorated; i++) {
    clearDecoration(decorations[decorations[i].height]);
  }
  height = 0;
  decorated = 0;
  serialized.clear();
}

void ChunkData::clearDecoration(Decoration& decoration) {
  decoration.block = Block::Air;
  decoration.decorated = false;
}

//////////////////////////////////////////////////////////////////////////////

int getBaseHeight(int x, int z) {
  return heightmap(x, z)->height;
}

ChunkDataRange loadChunkData(int cx, int cz) {
  chunkData.reset();
  const auto bx = cx << kChunkBits, bz = cz << kChunkBits;
  for (auto j = 0; j < kChunkWidth; j++) {
    for (auto i = 0; i < kChunkWidth; i++) {
      loadChunk(i + bx, j + bz, &chunkData);
    }
  }
  return {chunkData.data(), chunkData.data() + chunkData.size()};
}

HeightmapRange loadHeightmap(int cx, int cz, int level) {
  heightmapData.clear();
  const auto bx = cx << kChunkBits, bz = cz << kChunkBits;
  for (auto j = 0; j < kChunkWidth; j++) {
    for (auto i = 0; i < kChunkWidth; i++) {
      const auto ax = (2 * (i + bx) + 1) << level;
      const auto az = (2 * (j + bz) + 1) << level;
      heightmapData.push_back(packHeightmapData(ax, az));
    }
  }
  return {heightmapData.data(), heightmapData.data() + heightmapData.size()};
}

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
