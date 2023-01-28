#include "open-simplex-2d.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

//////////////////////////////////////////////////////////////////////////////

namespace voxels {

//////////////////////////////////////////////////////////////////////////////

namespace {

//////////////////////////////////////////////////////////////////////////////

constexpr double kNorm2D    = 1.0 / 47.0;
constexpr double kRootThree = 1.7320508075688772;
constexpr double kSquish2D  = (kRootThree - 1) / 2;
constexpr double kStretch2D = (1 / kRootThree - 1) / 2;

uint32_t shuffleSeed(uint32_t seed) {
  return seed * 1664525 + 1013904223;
}

constexpr int kBase2D[2][3][3] = {
  {{1, 1, 0}, {1, 0, 1}, {0, 0, 0}},
  {{1, 1, 0}, {1, 0, 1}, {2, 1, 1}},
};

constexpr int kLookupPairs2D[12][2] = {
  {0,  1}, {1,  0}, {4,  1}, {17, 0}, {20, 2}, {21, 2},
  {22, 5}, {23, 5}, {26, 4}, {39, 3}, {42, 4}, {43, 3},
};

constexpr int kP2D[6][4] = {
  {0, 0, 1, -1}, {0, 0, -1, 1}, {0, 2, 1, 1},
  {1, 2, 2,  0}, {1, 2,  0, 2}, {1, 0, 0, 0},
};

struct Contribution {
  double dx;
  double dy;
  int xsb;
  int ysb;
};

struct Precomputation {
  std::array<std::array<Contribution, 4>, 6> contributions;
  std::array<int8_t, 128> lookup;
};

constexpr Precomputation kPrecomputation = ([]{
  auto result = Precomputation{};

  const auto set = [&](Contribution& contribution, int multiplier, int xsb, int ysb) {
    contribution.dx = -xsb - multiplier * kSquish2D;
    contribution.dy = -ysb - multiplier * kSquish2D;
    contribution.xsb = xsb;
    contribution.ysb = ysb;
  };

  auto i = 0;
  for (const auto [baseIndex, multiplier, dx, dy] : kP2D) {
    auto j = 0;
    auto& contribution = result.contributions[i++];
    for (const auto [bm, bx, by] : kBase2D[baseIndex]) {
      set(contribution[j++], bm, bx, by);
    }
    set(contribution[j++], multiplier, dx, dy);
  }

  for (auto i = 0; i < result.lookup.size(); i++) {
    result.lookup[i] = -1;
  }
  for (const auto& pair : kLookupPairs2D) {
    result.lookup[pair[0]] = static_cast<int8_t>(pair[1]);
  }

  return result;
})();

//////////////////////////////////////////////////////////////////////////////

} // namespace

//////////////////////////////////////////////////////////////////////////////

Noise2D::Noise2D(uint32_t seed) {
  seed = shuffleSeed(shuffleSeed(shuffleSeed(seed)));

  std::array<uint8_t, 256> source;
  for (auto i = 0; i < source.size(); i++) {
    source[i] = static_cast<uint8_t>(i);
  }

  for (int i = source.size() - 1; i >= 0; i--) {
    seed = shuffleSeed(seed);
    uint32_t r = (seed + 31) % (i + 1);
    perm[i] = source[r];
    source[r] = source[i];
  }
}

double Noise2D::query(double x, double y) const {
  const auto stretchOffset = (x + y) * kStretch2D;

  const auto xs = x + stretchOffset;
  const auto ys = y + stretchOffset;

  const auto xsb = floor(xs);
  const auto ysb = floor(ys);

  const auto squishOffset = (xsb + ysb) * kSquish2D;

  const auto dx0 = x - (xsb + squishOffset);
  const auto dy0 = y - (ysb + squishOffset);

  const auto xins = xs - xsb;
  const auto yins = ys - ysb;

  const auto inSum = xins + yins;
  const auto hash = (static_cast<int>(xins - yins + 1) << 0) |
                    (static_cast<int>(inSum)           << 1) |
                    (static_cast<int>(inSum + yins)    << 2) |
                    (static_cast<int>(inSum + xins)    << 4);

  auto value = 0.0;
  const auto index = kPrecomputation.lookup[hash];

  for (const auto& contribution : kPrecomputation.contributions[index]) {
    const auto dx = dx0 + contribution.dx;
    const auto dy = dy0 + contribution.dy;

    const auto attn = 2 - dx * dx - dy * dy;
    if (attn <= 0) continue;

    const auto px = static_cast<int>(xsb) + contribution.xsb;
    const auto py = static_cast<int>(ysb) + contribution.ysb;

    const auto indexPart = perm[px & 0xff];
    const auto index = perm[(indexPart + py) & 0xff] & 0xe;

    const auto absGradientX = (index & 2) ? 2 : 5;
    const auto absGradientY = 7 - absGradientX;
    const auto gradientX = (index & 4) ? -absGradientX : absGradientX;
    const auto gradientY = (index & 8) ? -absGradientY : absGradientY;

    const auto valuePart = gradientX * dx + gradientY * dy;
    value += attn * attn * attn * attn * valuePart;
  }
  return value * kNorm2D;
}

//////////////////////////////////////////////////////////////////////////////

} // namespace voxels

//////////////////////////////////////////////////////////////////////////////

WASM_EXPORT(createNoise2D)
voxels::Noise2D* createNoise2D(int seed) {
  return new voxels::Noise2D(static_cast<uint32_t>(seed));
}

WASM_EXPORT(queryNoise2D)
double queryNoise2D(const voxels::Noise2D* noise, double x, double y) {
  return noise->query(x, y);
}
