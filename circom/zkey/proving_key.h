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

#ifndef CIRCOM_ZKEY_PROVING_KEY_H_
#define CIRCOM_ZKEY_PROVING_KEY_H_

#include "absl/types/span.h"

#include "circom/zkey/verifying_key.h"

namespace rabbitsnark::circom {

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

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_ZKEY_PROVING_KEY_H_
