#ifndef CIRCOM_ZKEY_PROVING_KEY_H_
#define CIRCOM_ZKEY_PROVING_KEY_H_

#include "absl/types/span.h"

#include "circom/zkey/verifying_key.h"

namespace zkx::circom {

template <typename Curve>
struct ProvingKey {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  VerifyingKey<Curve> verifying_key;
  absl::Span<const G1AffinePoint> ic;
  absl::Span<const G1AffinePoint> a_g1_query;
  absl::Span<const G1AffinePoint> b_g1_query;
  absl::Span<const G2AffinePoint> b_g2_query;
  absl::Span<const G1AffinePoint> c_g1_query;
  absl::Span<const G1AffinePoint> h_g1_query;

  bool operator==(const ProvingKey& other) const {
    return verifying_key == other.verifying_key && ic == other.ic &&
           a_g1_query == other.a_g1_query && b_g1_query == other.b_g1_query &&
           b_g2_query == other.b_g2_query && c_g1_query == other.c_g1_query &&
           h_g1_query == other.h_g1_query;
  }
  bool operator!=(const ProvingKey& other) const { return !operator==(other); }
};

}  // namespace zkx::circom

#endif  // CIRCOM_ZKEY_PROVING_KEY_H_
