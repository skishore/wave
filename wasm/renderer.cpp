#include "renderer.h"

#include "emscripten.h"

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

#define JS(return_type, name, arg_types) \
  EM_JS(return_type, name, arg_types, { throw new Error(); });

JS(int,  js_AddLightTexture,  (const uint8_t* data, int size));
JS(int,  js_FreeLightTexture, (int handle));

JS(int,  js_AddInstancedMesh,      (int block, int x, int y, int z));
JS(void, js_FreeInstancedMesh,     (int handle));
JS(int,  js_SetInstancedMeshLight, (int handle, int level));

JS(int,  js_AddVoxelMesh,  (const uint32_t* data, int size, int phase));
JS(void, js_FreeVoxelMesh, (int handle));
JS(int,  js_AddVoxelMeshGeometry, (int handle, const uint32_t* data, int size));
JS(int,  js_SetVoxelMeshGeometry, (int handle, const uint32_t* data, int size));
JS(int,  js_SetVoxelMeshLight,    (int handle, int texture));
JS(int,  js_SetVoxelMeshMask,     (int handle, int m0, int m1, bool shown));
JS(int,  js_SetVoxelMeshPosition, (int handle, int x, int y, int z));

#undef JS

//////////////////////////////////////////////////////////////////////////////

LightTexture::LightTexture(const LightTensor3<uint8_t>& lights) {
  binding = js_AddLightTexture(lights.data.data(), lights.data.size());
}

LightTexture::~LightTexture() { js_FreeLightTexture(binding); }

InstancedMesh::InstancedMesh(Block block, int x, int y, int z) {
  binding = js_AddInstancedMesh(static_cast<int>(block), x, y, z);
  lightLevel = -1;
}

InstancedMesh::~InstancedMesh() {
  if (binding >= 0) js_FreeInstancedMesh(binding);
}

void InstancedMesh::setLight(int level) {
  if (binding < 0 || level == lightLevel) return;
  js_SetInstancedMeshLight(binding, level);
  lightLevel = level;
}

VoxelMesh::VoxelMesh(const Quads& quads, int phase) {
  const auto data = reinterpret_cast<const uint32_t*>(quads.data());
  const auto size = quads.size() * sizeof(Quad) / sizeof(uint32_t);
  binding = js_AddVoxelMesh(data, size, phase);
}

VoxelMesh::~VoxelMesh() { js_FreeVoxelMesh(binding); }

void VoxelMesh::appendGeometry(const Quads& quads) {
  const auto data = reinterpret_cast<const uint32_t*>(quads.data());
  const auto size = quads.size() * sizeof(Quad) / sizeof(uint32_t);
  js_AddVoxelMeshGeometry(binding, data, size);
}

void VoxelMesh::setGeometry(const Quads& quads) {
  const auto data = reinterpret_cast<const uint32_t*>(quads.data());
  const auto size = quads.size() * sizeof(Quad) / sizeof(uint32_t);
  js_SetVoxelMeshGeometry(binding, data, size);
}

void VoxelMesh::setLight(const LightTexture& texture) {
  js_SetVoxelMeshLight(binding, texture.binding);
}

void VoxelMesh::setMask(int m0, int m1, bool shown) {
  js_SetVoxelMeshMask(binding, m0, m1, shown);
}

void VoxelMesh::setPosition(int x, int y, int z) {
  js_SetVoxelMeshPosition(binding, x, y, z);
}

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
