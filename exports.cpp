#include <stdint.h>

namespace {

// Voxel data structures.

typedef int Vec3[3];

struct Block {
  int facets[6];
  bool opaque;
  bool solid;
};

struct Facet {
  double color[4];
  int texture;
};

struct Tensor3 {
  Vec3 shape;
  Vec3 stride;
  uint8_t data[1 << 18];
};

// Static memory.

constexpr int kNoMaterial = 0;
constexpr int kRegistrySize = 1 << 16;

Block s_blocks[kRegistrySize];
Facet s_facets[kRegistrySize];
int s_num_blocks;
int s_num_facets;

Tensor3 s_voxels;

// Helper methods.

int getFaceDir(int block0, int block1, int dir) {
  auto const opaque0 = s_blocks[block0].opaque;
  auto const opaque1 = s_blocks[block1].opaque;
  if (opaque0 && opaque1) return 0;
  if (opaque0) return 1;
  if (opaque1) return -1;

  auto const material0 = s_blocks[block0].facets[dir];
  auto const material1 = s_blocks[block1].facets[dir + 1];
  if (material0 == material1) return 0;
  if (material0 == kNoMaterial) return -1;
  if (material1 == kNoMaterial) return 1;
  return 0;
}

int packAOMask(const uint8_t* data, int ipos, int ineg, int dj, int dk) {
  (void)ineg;
  auto a00 = 0, a01 = 0, a10 = 0, a11 = 0;
  if (s_blocks[data[ipos + dj]].solid) { a10++; a11++; }
  if (s_blocks[data[ipos - dj]].solid) { a00++; a01++; }
  if (s_blocks[data[ipos + dk]].solid) { a01++; a11++; }
  if (s_blocks[data[ipos - dk]].solid) { a00++; a10++; }

  if (a00 == 0 && s_blocks[data[ipos - dj - dk]].solid) a00++;
  if (a01 == 0 && s_blocks[data[ipos - dj + dk]].solid) a01++;
  if (a10 == 0 && s_blocks[data[ipos + dj - dk]].solid) a10++;
  if (a11 == 0 && s_blocks[data[ipos + dj + dk]].solid) a11++;

  // Order here matches the order in which we push vertices in addQuad.
  return (a01 << 6) | (a11 << 4) | (a10 << 2) | a00;
}

}

extern "C" void register_block(
    int f0, int f1, int f2, int f3, int f4, int f5, bool opaque, bool solid) {
  auto const block = s_num_blocks++;
  s_blocks[block] = {{f0, f1, f2, f3, f4, f5}, opaque, solid};
}

extern "C" void register_facet(
    double c0, double c1, double c2, double c3, int texture) {
  auto const facet = s_num_facets++;
  s_facets[facet] = {{c0, c1, c2, c3}, texture};
}

extern "C" uint8_t* allocate_voxels(int x, int y, int z) {
  s_voxels.shape[0] = x;
  s_voxels.shape[1] = y;
  s_voxels.shape[2] = z;
  s_voxels.stride[0] = 1;
  s_voxels.stride[1] = x;
  s_voxels.stride[2] = x * y;
  return s_voxels.data;
}

extern "C" int mesh() {
  auto const& data = s_voxels.data;
  auto const& shape = s_voxels.shape;
  auto const& stride = s_voxels.stride;

  if (shape[0] < 2 || shape[1] < 2 || shape[2] < 2) return 0;

  auto result = 0;

  for (auto d = 0; d < 3; d++) {
    auto const dir = d * 2;
    auto const u = (d + 1) % 3;
    auto const v = (d + 2) % 3;
    auto const ld = shape[d] - 2, lu = shape[u] - 2, lv = shape[v] - 2;
    auto const sd = stride[d], su = stride[u], sv = stride[v];
    auto const base = su + sv;

    static int16_t kMaskData[1 << 18];
    for (auto id = 0; id < ld; id++) {
      auto n = 0;
      for (auto iu = 0; iu < lu; iu++) {
        auto index = base + id * sd + iu * su;
        for (auto iv = 0; iv < lv; iv++, index += sv, n += 1) {
          // mask[n] is the face between (id, iu, iv) and (id + 1, iu, iv).
          // Its value is the MaterialId to use, times -1, if it is in the
          // direction opposite `dir`.
          //
          // When we enable ambient occlusion, we shift these masks left by
          // 8 bits and pack AO values for each vertex into the lower byte.
          auto const block0 = data[index];
          auto const block1 = data[index + sd];
          if (block0 == block1) continue;
          auto const facing = getFaceDir(block0, block1, dir);
          if (facing == 0) continue;

          auto const mask = facing > 0
            ?  s_blocks[block0].facets[dir]
            : -s_blocks[block1].facets[dir + 1];
          auto const ao = facing > 0
            ? packAOMask(data, index + sd, index, su, sv)
            : packAOMask(data, index, index + sd, su, sv);
          kMaskData[n] = (mask << 8) | ao;
        }
      }

      n = 0;

      for (auto iu = 0; iu < lu; iu++) {
        auto h = 1;
        for (auto iv = 0; iv < lv; iv += h, n += h) {
          auto const mask = kMaskData[n];
          if (mask == 0) {
            h = 1;
            continue;
          }

          for (h = 1; h < lv - iv; h++) {
            if (mask != kMaskData[n + h]) break;
          }

          auto w = 1, nw = n + lv;
          for (; w < lu - iu; w++, nw += lv) {
            for (auto x = 0; x < h; x++) {
              if (mask != kMaskData[nw + x]) goto DONE;
            }
          }
          DONE:

          result++;

          nw = n;
          for (auto wx = 0; wx < w; wx++, nw += lv) {
            for (auto hx = 0; hx < h; hx++) {
              kMaskData[nw + hx] = 0;
            }
          }
        }
      }
    }
  }

  return result;
}
