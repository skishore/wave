#include <stdint.h>

namespace {

constexpr double kNorm2D    = 1.0 / 47.0;
constexpr double kRootThree = 1.7320508075688772;
constexpr double kSquish2D  = (kRootThree - 1) / 2;
constexpr double kStretch2D = (1 / kRootThree - 1) / 2;

constexpr int kContributionsStride  = 4;
constexpr int kContributionsPerHash = 4;

double sContributions[4 * 4 * 6] = {};
uint8_t sLookup[256] = {};

uint8_t sPermutations[64][256] = {};

constexpr double gradients2D[16] = {
  5,  2, 2,  5, -5,  2, -2,  5,
  5, -2, 2, -5, -5, -2, -2, -5,
};

}

extern "C" double* getContributions() {
  return sContributions;
}

extern "C" uint8_t* getLookup() {
  return sLookup;
}

extern "C" uint8_t* getPermutations(int i) {
  return sPermutations[i];
}

extern "C" double noise2D(int i, double x, double y) {
  auto const& perm = sPermutations[i];
  auto const stretchOffset = (x + y) * kStretch2D;

  auto const xs = x + stretchOffset;
  auto const ys = y + stretchOffset;

  auto const xsb = __builtin_floor(xs);
  auto const ysb = __builtin_floor(ys);

  auto const squishOffset = (xsb + ysb) * kSquish2D;

  auto const dx0 = x - (xsb + squishOffset);
  auto const dy0 = y - (ysb + squishOffset);

  auto const xins = xs - xsb;
  auto const yins = ys - ysb;

  auto const inSum = xins + yins;
  auto const hash = (static_cast<int>(xins - yins + 1))   |
                    (static_cast<int>(inSum)        << 1) |
                    (static_cast<int>(inSum + yins) << 2) |
                    (static_cast<int>(inSum + xins) << 4);

  auto value = 0.0;
  auto c = sLookup[hash];
  auto const limit = c + kContributionsStride * kContributionsPerHash;

  for (; c < limit; c += kContributionsStride) {
    auto const cdx = sContributions[c + 0];
    auto const cdy = sContributions[c + 1];

    auto const dx = dx0 + cdx;
    auto const dy = dy0 + cdy;

    auto const attn = 2 - dx * dx - dy * dy;
    if (attn > 0) {
      auto const cxsb = sContributions[c + 2];
      auto const cysb = sContributions[c + 3];

      auto const px = static_cast<int>(xsb + cxsb);
      auto const py = static_cast<int>(ysb + cysb);

      auto const indexPart = perm[px & 0xff];
      auto const index = perm[(indexPart + py) & 0xff] & 0x0e;

      auto const valuePart = gradients2D[index] * dx +
                             gradients2D[index + 1] * dy;
      value += attn * attn * attn * attn * valuePart;
    }
  }

  return value * kNorm2D;
}
