#pragma once

#include "base.h"

#include <array>
#include <ranges>
#include <vector>

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

struct ChunkData {
  static_assert(kWorldHeight - 1 == 0xff);
  static_assert(sizeof(Block) == sizeof(uint8_t));

  const uint8_t* data() const { return serialized.data(); }
  size_t size() const { return serialized.size(); }

  void commit();
  void decorate(Block block, int32_t height);
  void push(Block block, int32_t limit);
  void reset();

 private:
  struct Decoration {
    Block block = Block::Air;
    bool decorated = false;
    uint8_t height = 0;
  };

  void clearDecoration(Decoration& decoration);

  int32_t height = 0;
  int32_t decorated = 0;
  std::vector<uint8_t> serialized;
  NonCopyArray<Decoration, kWorldHeight> decorations;
};

struct ChunkDataRange {
  const uint8_t* start;
  const uint8_t* end;
};
ChunkDataRange loadChunkData(int32_t cx, int32_t cz);

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
