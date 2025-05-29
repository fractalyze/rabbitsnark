#ifndef CIRCOM_ZKEY_VERIFYING_KEY_H_
#define CIRCOM_ZKEY_VERIFYING_KEY_H_

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/read_only_buffer.h"

namespace zkx::circom {

template <typename Curve>
struct VerifyingKey {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  G1AffinePoint* alpha_g1 = nullptr;
  G1AffinePoint* beta_g1 = nullptr;
  G2AffinePoint* beta_g2 = nullptr;
  G2AffinePoint* gamma_g2 = nullptr;
  G1AffinePoint* delta_g1 = nullptr;
  G2AffinePoint* delta_g2 = nullptr;

  bool operator==(const VerifyingKey& other) const {
    return *alpha_g1 == *other.alpha_g1 && *beta_g1 == *other.beta_g1 &&
           *beta_g2 == *other.beta_g2 && *gamma_g2 == *other.gamma_g2 &&
           *delta_g1 == *other.delta_g1 && *delta_g2 == *other.delta_g2;
  }
  bool operator!=(const VerifyingKey& other) const {
    return !operator==(other);
  }

  absl::Status Read(const base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&alpha_g1, 1));
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&beta_g1, 1));
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&beta_g2, 1));
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&gamma_g2, 1));
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&delta_g1, 1));
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&delta_g2, 1));
    return absl::OkStatus();
  }
};

}  // namespace zkx::circom

#endif  // CIRCOM_ZKEY_VERIFYING_KEY_H_
