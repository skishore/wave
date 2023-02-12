#include "mesher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <type_traits>
#include <vector>

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

namespace {

constexpr int pack_indices(const NonCopyArray<int, 6>& indices) {
  auto result = 0;
  for (auto i = 0; i < indices.size(); i++) {
    const auto x = indices[i];
    assert(0 <= x && x < 4);
    result |= x << (i * 2);
  }
  return result;
};

constexpr NonCopyArray<int, 3> kWaveValues = {0b0110, 0b1111, 0b1100};

constexpr NonCopyArray<int, 4> kIndexOffsets = {
  pack_indices({0, 1, 2, 0, 2, 3}),
  pack_indices({1, 2, 3, 0, 1, 3}),
  pack_indices({0, 2, 1, 0, 3, 2}),
  pack_indices({3, 1, 0, 3, 2, 1}),
};

} // namespace

//////////////////////////////////////////////////////////////////////////////

Mesher::Mesher(const Registry& r) : registry(r) {
  equilevels.fill(1);
  heightmap.data.fill(0);
  voxels.data.fill(Block::Air);
  const auto m = safe_cast<int>(voxels.shape[1] - 1);
  for (auto x = 0; x < voxels.shape[0]; x++) {
    for (auto z = 0; z < voxels.shape[2]; z++) {
      voxels.set(x, 0, z, Block::Bedrock);
      voxels.set(x, m, z, Block::Air);
    }
  }
}

void Mesher::meshChunk() {
  solid_geo.clear();
  water_geo.clear();

  auto max_height = 0;
  for (const auto entry : heightmap.data) {
    max_height = std::max(max_height, static_cast<int>(entry) + 1);
  }

  using Voxels = decltype(voxels);
  using Equilevels = decltype(equilevels);
  static_assert(Voxels::stride[1] == 1);
  static_assert(Voxels::shape[1] == Equilevels::Size);

  const auto skip_level = [&](int i) {
    const auto el0 = equilevels[i + 0];
    const auto el1 = equilevels[i + 1];
    if (el0 + el1 != 2) return false;
    const auto block0 = voxels.data[i + 0];
    const auto block1 = voxels.data[i + 1];
    if (block0 == block1) return true;
    return registry.getBlock(block0).opaque &&
           registry.getBlock(block1).opaque;
  };

  const auto limit = equilevels.size() - 1;
  for (auto i = 0; i < limit; i++) {
    if (skip_level(i)) continue;
    auto j = i + 1;
    for (; j < limit; j++) {
      if (skip_level(j)) break;
    }
    auto y_min = i;
    auto y_max = std::min(j, max_height) + 1;
    if (y_min >= y_max) break;
    computeChunkGeometry(y_min, y_max);
    i = j;
  }
}

void Mesher::meshFrontier(const HeightmapEntry* start, int n,
                          Point pos, int scale, int index) {
  solid_geo.clear();
  water_geo.clear();

  assert(n % 2 == 0);
  const auto half = n >> 1;
  const auto stride = Point{2, 2 * n};

  for (auto k = 0; k < 4; k++) {
    const auto x_offset = (k & 1) ? half : 0;
    const auto z_offset = (k & 2) ? half : 0;
    const Point sub = {pos.x + x_offset * scale, pos.z + z_offset * scale};
    const auto mask = 4 * index + k;
    const auto ptr = &start[x_offset + n * z_offset].fields[0];
    computeFrontierGeometry(&solid_geo, ptr + 0, half, sub, stride, scale, mask, 1);
    computeFrontierGeometry(&water_geo, ptr + 1, half, sub, stride, scale, mask, 0);
  }
}

void Mesher::addQuad(
    Quads* quads, const MaterialData& material, int dir, int ao,
    int wave, int d, int w, int h, const Pos& pos) {
  static_assert(std::is_trivially_default_constructible_v<Quad>);
  auto& quad = quads->emplace_back();

  const auto triangleHint = getTriangleHint(ao);
  const auto indices = dir > 0
    ? (triangleHint ? kIndexOffsets[2] : kIndexOffsets[3])
    : (triangleHint ? kIndexOffsets[0] : kIndexOffsets[1]);

  const auto [x, y, z] = pos;
  const auto texture = material.texture;
  const auto dir_bit = dir > 0 ? 1 : 0;

  assert(x == static_cast<int16_t>(x));
  assert(y == static_cast<int16_t>(y));
  assert(z == static_cast<int16_t>(z));
  assert(w == static_cast<int16_t>(w));
  assert(h == static_cast<int16_t>(h));

  const auto packTwoInts = [](int a, int b) {
    return (static_cast<uint32_t>(a & 0xffff) << 0) |
           (static_cast<uint32_t>(b & 0xffff) << 16);
  };

  quad[0] = packTwoInts(x, y);
  quad[1] = packTwoInts(z, indices);
  quad[2] = packTwoInts(w, h);
  quad[3] = (static_cast<uint32_t>(texture) << 8)  |
            (static_cast<uint32_t>(ao)      << 16) |
            (static_cast<uint32_t>(wave)    << 24) |
            (static_cast<uint32_t>(d)       << 28) |
            (static_cast<uint32_t>(dir_bit) << 30);
}

void Mesher::computeChunkGeometry(int y_min, int y_max) {
  Pos pos;
  Pos stride{
    static_cast<int>(voxels.stride[0]),
    static_cast<int>(voxels.stride[1]),
    static_cast<int>(voxels.stride[2])};
  Pos shape{
    static_cast<int>(voxels.shape[0]),
    static_cast<int>(y_max - y_min),
    static_cast<int>(voxels.shape[2])};

  for (auto dx = 0; dx < 3; dx++) {
    const auto d = dx == 2 ? dx : 1 - dx;
    const auto face = 2 * d;
    const auto v = d == 1 ? 0 : 1;
    const auto u = 3 - d - v;
    const auto ld = shape[d] - 1, lu = shape[u] - 2, lv = shape[v] - 2;
    const auto sd = stride[d], su = stride[u], sv = stride[v];
    const auto base = su + sv + y_min * stride[1];

    // d is the dimension that the quad faces. A d of {0, 1, 2} corresponds
    // to a quad with a normal that's a unit vector on the {x, y, z} axis,
    // respectively. u and v are the orthogonal dimensions along which we
    // compute the quad's width and height.
    //
    // The simplest way to handle coordinates here is to let (d, u, v)
    // be consecutive dimensions mod 3. That's how VoxelShader interprets
    // data for a quad facing a given dimension d.
    //
    // However, to optimize greedy meshing, we want to take advantage of
    // the fact that the y-axis is privileged in multiple ways:
    //
    //    1. Our chunks are limited in the x- and z-dimensions, but span
    //       the entire world in the y-dimension, so this axis is longer.
    //
    //    2. The caller may have a heightmap limiting the maximum height
    //       of a voxel by (x, z) coordinate, which we can use to cut the
    //       greedy meshing inner loop short.
    //
    // As a result, we tweak the d = 0 case to use (u, v) = (2, 1) instead
    // of (u, v) = (1, 2). To map back to the standard coordinates used by
    // the shader, we only need to fix up two inputs to addQuad: (w, h) and
    // the bit-packed AO mask. w_fixed, h_fixed, su_fixed, and sv_fixed are
    // the standard-coordinates versions of these values.
    //
    const auto su_fixed = d > 0 ? su : sv;
    const auto sv_fixed = d > 0 ? sv : su;

    const auto area = lu * lv;
    if (mask_data.size() < area) {
      mask_data.resize(area);
    }
    if (mask_union.size() < lu) {
      mask_union.resize(lu);
    }

    for (auto id = 0; id < ld; id++) {
      auto n = 0;
      auto complete_union = 0;
      for (auto iu = 0; iu < lu; iu++) {
        mask_union[iu] = 0;
        auto index = base + id * sd + iu * su;
        for (auto iv = 0; iv < lv; iv++, n++, index += sv) {
          // mask[n] is the face between (id, iu, iv) and (id + 1, iu, iv).
          //
          // It contains packed binary data, including the material's index,
          // the direction of the face, and ambient occlusion for each vertex.
          // We include all these values in the mask to split adjacent voxels
          // that do not match during greedy meshing. The packed layout is:
          //
          //    - bits 0:8:  AO value (4 x 2-bit values)
          //    - bits 8:9:  dir in {0, 1} (0 -> -1, 1 -> +1)
          //    - bits 9:24: material index
          //
          const auto block0 = voxels.data[index];
          const auto block1 = voxels.data[index + sd];
          if (block0 == block1) continue;
          const auto dir = getFaceDir(block0, block1, face);
          if (dir == 0) continue;

          const auto material = dir > 0
            ? registry.getBlockUnsafe(block0).faces[face + 0]
            : registry.getBlockUnsafe(block1).faces[face + 1];
          const auto ao = dir > 0
            ? packAOMask(index + sd, index, su_fixed, sv_fixed)
            : packAOMask(index, index + sd, su_fixed, sv_fixed);
          const auto mask = (material.id << 9) | (dir > 0 ? 1 << 8 : 0) | ao;

          mask_data[n] = mask;
          mask_union[iu] |= mask;
          complete_union |= mask;
        }
      }
      if (complete_union == 0) continue;

      // Our data includes a 1-voxel-wide border all around our chunk in
      // all directions. In the y direction, this border is synthetic, but
      // in the x and z direction, the border cells come from other chunks.
      //
      // To avoid meshing a block face twice, we mesh a face the face faces
      // into our chunk. This check applies in the x and z directions.
      //
      // We should actually mesh the face that faces out of the chunk. An
      // LOD mesh, by necessity, has solid walls facing out on all sides,
      // because it must work next to an arbitrary LOD or chunk mesh. By
      // meshing faces facing into chunk meshes, we cause z-fighting at the
      // boundary between chunk meshes and LOD meshes.
      //
      // But we don't yet have a 1-cell border in our lighting textures,
      // so we'll stick with this approach until we do smooth lighting.
      if (d != 1) {
        if (id == 0) {
          for (auto i = 0; i < area; i++) {
            if ((mask_data[i] & 0x100) == 0) mask_data[i] = 0;
          }
        } else if (id == ld - 1) {
          for (auto i = 0; i < area; i++) {
            if ((mask_data[i] & 0x100) != 0) mask_data[i] = 0;
          }
        }
      }

      n = 0;
      for (auto iu = 0; iu < lu; iu++) {
        if (mask_union[iu] == 0) {
          n += lv;
          continue;
        }

        auto h = 1;
        for (auto iv = 0; iv < lv; iv += h, n += h) {
          const auto mask = mask_data[n];
          if (mask == 0) {
            h = 1;
            continue;
          }

          for (h = 1; h < lv - iv; h++) {
            if (mask != mask_data[n + h]) break;
          }

          auto w = 1;
          auto nw = n + lv;
          for (; w < lu - iu; w++, nw += lv) {
            for (auto x = 0; x < h; x++) {
              if (mask != mask_data[nw + x]) goto done;
            }
          }
          done:

          pos[d] = id;
          pos[u] = iu;
          pos[v] = iv;
          pos[1] += y_min;

          const auto ao  = mask & 0xff;
          const auto dir = mask & 0x100 ? 1 : -1;

          static_assert(sizeof(MaybeMaterial) == 1);
          const auto& material = registry.getMaterialUnsafe(
              assertMaterialUnsafe({static_cast<uint8_t>(mask >> 9)}));
          const auto geo = material.color[3] < 1 ? &water_geo : &solid_geo;

          const auto w_fixed = d > 0 ? w : h;
          const auto h_fixed = d > 0 ? h : w;

          if (material.liquid) {
            if (d == 1) {
              if (dir > 0) {
                const auto wave = kWaveValues[d];
                addQuad(geo, material, dir, ao, wave, d, w, h, pos);
                patchLiquidSurfaceQuads(geo, ao, w, h, pos);
              } else {
                addQuad(geo, material, dir, ao, 0, d, w, h, pos);
              }
            } else {
              const auto wave = kWaveValues[d];
              if (h == lv - iv) {
                addQuad(geo, material, dir, ao, wave, d, w_fixed, h_fixed, pos);
              } else {
                splitLiquidSideQuads(geo, material, dir, ao, wave, d, w, h, pos);
              }
            }
          } else {
            addQuad(geo, material, dir, ao, 0, d, w_fixed, h_fixed, pos);
            if (material.alphaTest) {
              addQuad(geo, material, -dir, ao, 0, d, w_fixed, h_fixed, pos);
            }
          }

          nw = n;
          for (auto wx = 0; wx < w; wx++, nw += lv) {
            for (auto hx = 0; hx < h; hx++) {
              mask_data[nw + hx] = 0;
            }
          }
        }
      }
    }
  }
}

void Mesher::computeFrontierGeometry(
    Quads* quads, const HeightmapField* start, int n, Point pos,
    Point stride, int scale, int mask, bool solid) {

  const auto mask_8bits = static_cast<uint8_t>(mask);
  assert(mask_8bits == mask);

  const auto size = n + 2;
  const auto area = size * size;
  if (height_mask.size() < area) {
    height_mask.resize(area, {Block::Air, 0});
  }

  for (auto z = 0; z < n; z++) {
    auto source = &start[z * stride.z];
    const auto target = (z + 1) * size + 1;
    for (auto x = 0; x < n; x++, source += stride.x) {
      height_mask[target + x] = *source;
    }
  }

  // Use 1D greedy meshing to mesh each of the four horizontal faces.
  for (auto k = solid ? 0 : 4; k < 4; k++) {
    const auto d = k & 2 ? 2 : 0;
    const auto dir = k & 1 ? -1 : 1;

    const auto si = d == 0 ? 1 : size;
    const auto sj = size + 1 - si;

    const auto ao = d == 0 ? 0x82 : 0x0A;
    const auto di = dir > 0 ? si : -si;

    for (auto i = 0; i < n; i++) {
      const auto ii = dir > 0 ? 1 : 0;
      auto offset = (i + 1) * si + sj;
      for (auto j = 0; j < n; j++, offset += sj) {
        const auto [block, height] = height_mask[offset];
        if (block == Block::Air) continue;

        // We could use the material at the side of the block with:
        //  const face = 2 * d + ((1 - dir) >> 1);
        //
        // But doing so muddles grass, etc. textures at a distance.
        const auto id = registry.getBlockUnsafe(block).faces[2];
        if (id == kNoMaterial) continue;

        const auto neighbor_height = height_mask[offset + di].height;
        if (neighbor_height >= height) continue;

        auto w = 1;
        const auto limit = n - j;
        for (auto index = offset + sj; w < limit; w++, index += sj) {
          const auto match = height_mask[index].block == block &&
                             height_mask[index].height == height &&
                             height_mask[index + di].height == neighbor_height;
          if (!match) break;
        }

        const auto px = d == 0 ? (i + ii) * scale : j * scale;
        const auto pz = d == 0 ? j * scale : (i + ii) * scale;
        const auto wi = d == 0 ? height - neighbor_height : w * scale;
        const auto hi = d == 0 ? w * scale : height - neighbor_height;

        std::array<int, 3> tmp{pos.x + px, neighbor_height, pos.z + pz};
        const auto& material = registry.getMaterialUnsafe(assertMaterial(id));
        const auto wave = material.liquid ? 0b1111 : 0;
        addQuad(quads, material, dir, ao, wave, d, wi, hi, tmp);
        quads->back()[3] |= static_cast<uint32_t>(mask_8bits);

        const auto extra = w - 1;
        offset += extra * sj;
        j += extra;
      }
    }
  }

  // Use 2D greedy meshing to mesh the heightmap's top faces. This step is
  // second because we operate destructively on the height mask here.
  for (auto z = 0; z < n; z++) {
    auto prev = &height_mask[(z + 1) * size + 1];
    for (auto x = 0; x < n; x++, prev++) {
      if (prev->block == Block::Air) continue;
      const auto id = registry.getBlockUnsafe(prev->block).faces[2];
      if (id == kNoMaterial) continue;

      const auto match = [&](const HeightmapField* next) {
        return next->block == prev->block && next->height == prev->height;
      };

      auto lx = n - x, lz = n - z, w = 1, h = 1;
      for (auto next = prev + size; w < lz; w++, next += size) {
        if (!match(next)) break;
      }
      for (; h < lx; h++) {
        auto next = prev + h;
        for (auto i = 0; i < w; i++, next += size) {
          if (!match(next)) goto OUTER;
        }
      }
      OUTER:

      std::array<int, 3> tmp{pos.x + x * scale, prev->height, pos.z + z * scale};
      const auto& material = registry.getMaterialUnsafe(assertMaterial(id));
      const auto wave = material.liquid ? 0b1111 : 0;
      addQuad(quads, material, 1, 0, wave, 1, scale * w, scale * h, tmp);
      quads->back()[3] |= static_cast<uint32_t>(mask_8bits);

      for (auto wi = 0; wi < w; wi++) {
        const auto target = &prev[wi * size];
        for (auto hi = 0; hi < h; hi++) target[hi].block = Block::Air;
      }

      const auto extra = h - 1;
      prev += extra;
      x += extra;
    }
  }
}

// We displace a liquid's upper surface downward using the `wave` attribute.
//
// When a liquid is adjacent to a downward surface, such as a rock that ends
// right above the water, we have to add small vertical patches to avoid
// leaving gaps in the liquid's surface.
//
// NOTE: The AO values here are not quite right. For each of the faces we
// consider (-x, +x, -z, +z), we should broadcast a different subset of the
// input AO. But doing that is tricky and AO doesn't matter much here.
void Mesher::patchLiquidSurfaceQuads(
    Quads* quads, int ao, int w, int h, const Pos& pos) {
  const auto base_x = pos[0];
  const auto base_y = pos[1];
  const auto base_z = pos[2];
  const auto water = voxels.get(base_x + 1, base_y, base_z + 1);
  const auto id = registry.getBlockUnsafe(water).faces[0];
  if (id == kNoMaterial) return;

  const auto patch = [&](int x, int z, int face) {
    const auto ax = base_x + x + 1;
    const auto az = base_z + z + 1;

    const auto& below = registry.getBlockUnsafe(voxels.get(ax, base_y + 0, az));
    if (below.opaque || below.faces[face] == kNoMaterial) return false;

    const auto& above = registry.getBlockUnsafe(voxels.get(ax, base_y + 1, az));
    return above.opaque || above.faces[3] != kNoMaterial;
  };

  auto tmp = pos;
  const auto& material = registry.getMaterialUnsafe(assertMaterialUnsafe(id));

  for (auto face = 4; face < 6; face++) {
    const auto dz = face == 4 ? -1 : w;
    const auto wave = kWaveValues[1] - kWaveValues[2];
    for (auto x = 0; x < h; x++) {
      if (!patch(x, dz, face)) continue;
      auto start = x;
      for (x++; x < h; x++) {
        if (!patch(x, dz, face)) break;
      }
      tmp[0] = base_x + start;
      tmp[2] = base_z + std::max(dz, 0);
      addQuad(quads, material, 1, ao, wave, 2, x - start, 0, tmp);
    }
  }

  for (auto face = 0; face < 2; face++) {
    const auto dx = face == 0 ? -1 : h;
    const auto wave = kWaveValues[1] - kWaveValues[0];
    for (auto z = 0; z < w; z++) {
      if (!patch(dx, z, face)) continue;
      auto start = z;
      for (z++; z < w; z++) {
        if (!patch(dx, z, face)) break;
      }
      tmp[0] = base_x + std::max(dx, 0);
      tmp[2] = base_z + start;
      addQuad(quads, material, 1, ao, wave, 0, 0, z - start, tmp);
    }
  }
}

// For vertical liquid surfaces, we need to check the block right above the
// surface to check if the top of this quad should get the wave effect. This
// test may change along the width of the liquid quad, so we may end up
// splitting one quad into multiple quads here.
void Mesher::splitLiquidSideQuads(
    Quads* quads, const MaterialData& material, int dir, int ao,
    int wave, int d, int w, int h, const Pos& pos) {
  const auto base_x = pos[0];
  const auto base_y = pos[1];
  const auto base_z = pos[2];

  const auto ax = base_x + (d == 0 && dir > 0 ? 0 : 1);
  const auto az = base_z + (d == 2 && dir > 0 ? 0 : 1);
  const auto ay = base_y + h + 1;

  auto tmp = pos;

  const auto test = [&](int i) {
    const auto above = d == 0 ? voxels.get(ax, ay, az + i)
                              : voxels.get(ax + i, ay, az);
    const auto& data = registry.getBlockUnsafe(above);
    return data.opaque || data.faces[3] == kNoMaterial;
  };

  auto last = test(0);
  for (auto i = 0; i < w; i++) {
    auto j = i + 1;
    for (; j < w && test(j) == last; j++) {}
    const auto w_fixed = d > 0 ? j - i : h;
    const auto h_fixed = d > 0 ? h : j - i;
    addQuad(quads, material, dir, ao, last ? wave : 0, d, w_fixed, h_fixed, tmp);
    tmp[2 - d] += j - i;
    last = !last;
    i = j - 1;
  }
}

bool Mesher::getTriangleHint(int ao) const {
  const auto a00 = (ao >> 0) & 3;
  const auto a10 = (ao >> 2) & 3;
  const auto a11 = (ao >> 4) & 3;
  const auto a01 = (ao >> 6) & 3;
  if (a00 == a11) return (a10 == a01) ? a10 == 3 : true;
  return (a10 == a01) ? false : (a00 + a11 > a10 + a01);
}

int Mesher::getFaceDir(Block block0, Block block1, int face) const {
  const auto& data0 = registry.getBlockUnsafe(block0);
  const auto& data1 = registry.getBlockUnsafe(block1);
  const auto opaque0 = data0.opaque;
  const auto opaque1 = data1.opaque;
  if (opaque0 && opaque1) return 0;
  if (opaque0) return 1;
  if (opaque1) return -1;

  const auto material0 = data0.faces[face];
  const auto material1 = data1.faces[face];
  if (material0 == material1)  return 0;
  if (material0 == kNoMaterial) return -1;
  if (material1 == kNoMaterial) return 1;
  return 0;
}

int Mesher::packAOMask(int ipos, int ineg, int dj, int dk) const {
  static_assert(sizeof(Block) == 1);

  const auto opaque = [&](Block block) {
    return registry.getBlockUnsafe(block).opaque;
  };

  auto a00 = 0, a01 = 0, a10 = 0, a11 = 0;

  const auto b0 = voxels.data[ipos + dj];
  const auto b1 = voxels.data[ipos - dj];
  const auto b2 = voxels.data[ipos + dk];
  const auto b3 = voxels.data[ipos - dk];

  // Optimize for the special case of completely unoccluded blocks.
  const auto bsum = static_cast<uint8_t>(b0) + static_cast<uint8_t>(b1) +
                    static_cast<uint8_t>(b2) + static_cast<uint8_t>(b3);
  if (bsum == 0) {
    const auto d0 = voxels.data[ipos - dj - dk];
    const auto d1 = voxels.data[ipos - dj + dk];
    const auto d2 = voxels.data[ipos + dj - dk];
    const auto d3 = voxels.data[ipos + dj + dk];

    const auto dsum = static_cast<uint8_t>(d0) + static_cast<uint8_t>(d1) +
                      static_cast<uint8_t>(d2) + static_cast<uint8_t>(d3);
    if (dsum == 0) return 0;

    if (opaque(d0)) a00++;
    if (opaque(d1)) a01++;
    if (opaque(d2)) a10++;
    if (opaque(d3)) a11++;
    return (a01 << 6) | (a11 << 4) | (a10 << 2) | a00;
  }

  if (opaque(b0)) { a10++; a11++; }
  if (opaque(b1)) { a00++; a01++; }
  if (opaque(b2)) { a01++; a11++; }
  if (opaque(b3)) { a00++; a10++; }

  if (a00 == 0 && opaque(voxels.data[ipos - dj - dk])) a00++;
  if (a01 == 0 && opaque(voxels.data[ipos - dj + dk])) a01++;
  if (a10 == 0 && opaque(voxels.data[ipos + dj - dk])) a10++;
  if (a11 == 0 && opaque(voxels.data[ipos + dj + dk])) a11++;

  // Order here matches the order in which we push vertices in addQuad.
  return (a01 << 6) | (a11 << 4) | (a10 << 2) | a00;
}

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels
