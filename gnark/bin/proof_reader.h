/* Copyright 2025 The RabbitSNARK Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GNARK_BIN_PROOF_READER_H_
#define GNARK_BIN_PROOF_READER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/endian.h"
#include "zkx/base/buffer/read_only_buffer.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"

namespace rabbitsnark::gnark {

template <typename Curve>
absl::StatusOr<zkx::Literal> ReadProofFromBin(std::string_view path) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using Fq = typename G1AffinePoint::BaseField;
  using F = typename G1AffinePoint::ScalarField;

  std::string content;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), path, &content));

  zkx::base::ReadOnlyBuffer read_buf(content.data(), content.size());

  read_buf.set_endian(zkx::base::Endian::kBig);
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

  G1AffinePoint pi_a;
  G2AffinePoint pi_b;
  G1AffinePoint pi_c;
  uint32_t unused;
  G1AffinePoint unused2;

  TF_RETURN_IF_ERROR(read_buf.ReadMany(&pi_a, &pi_b, &pi_c, &unused, &unused2));

  std::vector<zkx::Literal> proof_elements;
  proof_elements.push_back(zkx::LiteralUtil::CreateR0<G1AffinePoint>(pi_a));
  proof_elements.push_back(zkx::LiteralUtil::CreateR0<G2AffinePoint>(pi_b));
  proof_elements.push_back(zkx::LiteralUtil::CreateR0<G1AffinePoint>(pi_c));
  return zkx::LiteralUtil::MakeTupleOwned(std::move(proof_elements));
}

}  // namespace rabbitsnark::gnark

#endif  // GNARK_BIN_PROOF_READER_H_
