#ifndef GNARK_BIN_PROOF_WRITER_H_
#define GNARK_BIN_PROOF_WRITER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"

#include "zkx/literal.h"

namespace rabbitsnark::gnark {

template <typename Curve>
absl::Status WriteProofToBin(zkx::Literal& proof, const std::string& path) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using Fq = typename G1AffinePoint::BaseField;
  using F = typename G1AffinePoint::ScalarField;

  std::vector<zkx::Literal> decomposed = proof.DecomposeTuple();
  const G1AffinePoint& pi_a = decomposed[0].data<G1AffinePoint>()[0];
  const G2AffinePoint& pi_b = decomposed[1].data<G2AffinePoint>()[0];
  const G1AffinePoint& pi_c = decomposed[2].data<G1AffinePoint>()[0];

  zkx::base::Uint8VectorBuffer write_buf;
  TF_RETURN_IF_ERROR(write_buf.Grow(zkx::base::EstimateSize(
      pi_a, pi_b, pi_c, uint32_t{0}, G1AffinePoint::Zero())));
  write_buf.set_endian(zkx::base::Endian::kBig);
  zkx::base::AutoReset<bool> reset_scalar_field_is_in_montgomery(
      &zkx::base::Serde<F>::s_is_in_montgomery, false);
  zkx::base::AutoReset<bool> reset_base_field_is_in_montgomery(
      &zkx::base::Serde<Fq>::s_is_in_montgomery, false);
  zkx::base::AutoReset<zkx::math::AffinePointSerdeMode> reset_g1_s_mode(
      &zkx::base::Serde<G1AffinePoint>::s_mode,
      zkx::math::AffinePointSerdeMode::kGnarkRaw);
  zkx::base::AutoReset<zkx::math::AffinePointSerdeMode> reset_g2_s_mode(
      &zkx::base::Serde<G2AffinePoint>::s_mode,
      zkx::math::AffinePointSerdeMode::kGnarkRaw);

  TF_RETURN_IF_ERROR(write_buf.WriteMany(pi_a, pi_b, pi_c, uint32_t{0},
                                         G1AffinePoint::Zero()));

  std::string proof_string(reinterpret_cast<const char*>(write_buf.buffer()),
                           write_buf.buffer_offset());
  return tsl::WriteStringToFile(tsl::Env::Default(), path, proof_string);
}

}  // namespace rabbitsnark::gnark

#endif  // GNARK_BIN_PROOF_WRITER_H_
