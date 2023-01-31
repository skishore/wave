#pragma once

#include <memory>

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(int) == 4);

constexpr int kChunkBits  = 4;
constexpr int kChunkWidth = 1 << kChunkBits;
constexpr int kChunkMask  = kChunkWidth - 1;
constexpr int kWorldHeight = 256;

enum class Block : uint8_t {
  Air, Unknown, Bedrock, Bush, Dirt, Fungi, Grass,
  Rock, Sand, Snow, Stone, Trunk, Water };

#define WASM_EXPORT(X) __attribute__((export_name(#X))) extern "C"

#define DISALLOW_COPY_AND_ASSIGN(X) \
  X(X&&) = delete;                  \
  X(const X&) = delete;             \
  X& operator=(X&&) = delete;       \
  X& operator=(const X&) = delete;

constexpr bool isPowTwo(int input) {
  return (input & (input - 1)) == 0;
}

template <typename T, typename U>
constexpr T safe_cast(U u) {
  const auto result = static_cast<T>(u);
  assert(static_cast<U>(result) == u);
  return result;
}

template <typename T>
std::unique_ptr<T[]> allocate_array(size_t size, T value) {
  auto result = std::make_unique<T[]>(size);
  for (size_t i = 0; i < size; i++) result[i] = value;
  return result;
}

struct Point {
  int normSquared() const {
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

  bool operator!=(const Point& o) const {
    return !(*this == o);
  }

  int x;
  int z;
};

template <typename T>
struct ChunkTensor2 {
  T get(int x, int z) const {
    return data[index(x, z)];
  }

  void set(int x, int z, T value) {
    data[index(x, z)] = value;
  }

  static int index(int x, int z) {
    static_assert(isPowTwo(kChunkWidth));
    return x | (z * kChunkWidth);
  }

  std::array<T, kChunkWidth * kChunkWidth> data;
};

template <typename T>
struct ChunkTensor3 {
  T get(int x, int y, int z) const {
    return data[index(x, y, z)];
  }

  void set(int x, int y, int z, T value) {
    data[index(x, y, z)] = value;
  }

  static int index(int x, int y, int z) {
    static_assert(isPowTwo(kChunkWidth));
    static_assert(isPowTwo(kWorldHeight));
    return y | (x * kWorldHeight) | (z * kChunkWidth * kWorldHeight);
  }

  std::array<T, kWorldHeight * kChunkWidth * kChunkWidth> data;
};

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
