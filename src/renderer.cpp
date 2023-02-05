#include "renderer.h"

#include "emscripten.h"

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

#define JS(return_type, name, arg_types) \
  EM_JS(return_type, name, arg_types, { throw new Error(); });

JS(int,  js_AddVoxelMesh,  (const uint32_t* data, int size, int phase));
JS(void, js_FreeVoxelMesh, (int handle));
JS(int,  js_SetVoxelMeshGeometry, (int handle, const uint32_t* data, int size));
JS(int,  js_SetVoxelMeshPosition, (int handle, int x, int y, int z));

#undef JS

//////////////////////////////////////////////////////////////////////////////

VoxelMesh::VoxelMesh(const Quads& quads, int phase) {
  const auto data = reinterpret_cast<const uint32_t*>(quads.data());
  const auto size = quads.size() * sizeof(Quad) / sizeof(uint32_t);
  binding = js_AddVoxelMesh(data, size, phase);
}

VoxelMesh::~VoxelMesh() { js_FreeVoxelMesh(binding); }

void VoxelMesh::setGeometry(const Quads& quads) {
  const auto data = reinterpret_cast<const uint32_t*>(quads.data());
  const auto size = quads.size() * sizeof(Quad) / sizeof(uint32_t);
  js_SetVoxelMeshGeometry(binding, data, size);
}

void VoxelMesh::setPosition(int x, int y, int z) {
  js_SetVoxelMeshPosition(binding, x, y, z);
}

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
