#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "base.h"

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

struct VoxelMesh {
  using Quad = std::array<uint32_t, 4>;
  using Quads = std::vector<Quad>;

  VoxelMesh(const Quads& quads, int phase);
  ~VoxelMesh();

  void setGeometry(const Quads& quads);
  void setPosition(int x, int y, int z);

 private:
  int binding;

  DISALLOW_COPY_AND_ASSIGN(VoxelMesh);
};

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
