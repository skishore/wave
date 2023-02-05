#pragma once

#include <cassert>
#include <memory>

#include "parallel-hashmap/phmap.h"

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

#define WASM_EXPORT(X) __attribute__((export_name(#X))) extern "C"

#define DISALLOW_COPY_AND_ASSIGN(X) \
  X(X&&) = delete;                  \
  X(const X&) = delete;             \
  X& operator=(X&&) = delete;       \
  X& operator=(const X&) = delete;

template <typename T>
using HashSet = phmap::flat_hash_set<T>;

template <typename K, typename V>
using HashMap = phmap::flat_hash_map<K, V>;

//////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(int) == 4);

constexpr int kChunkBits  = 4;
constexpr int kChunkWidth = 1 << kChunkBits;
constexpr int kWorldHeight = 256;

enum class Block : uint8_t {
  Air, Unknown, Bedrock, Bush, Dirt, Fungi, Grass,
  Rock, Sand, Snow, Stone, Trunk, Water };

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

template <typename T, size_t X, size_t Z>
struct Tensor2 {
  T get(int x, int z) const {
    return data[index(x, z)];
  }

  void set(int x, int z, T value) {
    data[index(x, z)] = value;
  }

  static int index(int x, int z) {
    if constexpr (isPowTwo(X)) {
      return x | (z * X);
    }
    return x + (z * X);
  }

  std::array<T, X * Z> data;
  constexpr static size_t shape[2]  = {X, Z};
  constexpr static size_t stride[2] = {1, X};
};

template <typename T, size_t X, size_t Y, size_t Z>
struct Tensor3 {
  T get(int x, int y, int z) const {
    return data[index(x, y, z)];
  }

  void set(int x, int y, int z, T value) {
    data[index(x, y, z)] = value;
  }

  static int index(int x, int y, int z) {
    if constexpr (isPowTwo(X) && isPowTwo(Y)) {
      return y | (x * Y) | (z * X * Y);
    }
    return y + (x * Y) + (z * X * Y);
  }

  std::array<T, X * Y * Z> data;
  constexpr static size_t shape[3]  = {X, Y, Z};
  constexpr static size_t stride[3] = {Y, 1, X * Y};
};

template <typename T> using ChunkTensor1 =
  std::array<T, kWorldHeight>;
template <typename T> using ChunkTensor2 =
  Tensor2<T, kChunkWidth, kChunkWidth>;
template <typename T> using ChunkTensor3 =
  Tensor3<T, kChunkWidth, kWorldHeight, kChunkWidth>;

template <typename T> using MeshTensor1 =
  std::array<T, kWorldHeight + 2>;
template <typename T> using MeshTensor2 =
  Tensor2<T, kChunkWidth + 2, kChunkWidth + 2>;
template <typename T> using MeshTensor3 =
  Tensor3<T, kChunkWidth + 2, kWorldHeight + 2, kChunkWidth + 2>;

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
