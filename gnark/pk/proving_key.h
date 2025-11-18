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

#ifndef GNARK_PK_PROVING_KEY_H_
#define GNARK_PK_PROVING_KEY_H_

#include <sys/mman.h>

#include <memory>
#include <string_view>
#include <vector>

#include "gnark/reader_utils.h"
#include "gnark/serde_mode.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/read_only_buffer.h"

namespace rabbitsnark::gnark {

// See
// https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/fr/fft/domain.go#L22-L53
template <typename F>
struct Domain {
  uint64_t cardinality;
  F cardinality_inv;
  F generator;
  F generator_inv;
  F fr_multiplicative_gen;
  F fr_multiplicative_gen_inv;
  bool with_precompute;

  bool operator==(const Domain& other) const {
    return cardinality == other.cardinality &&
           cardinality_inv == other.cardinality_inv &&
           generator == other.generator &&
           generator_inv == other.generator_inv &&
           fr_multiplicative_gen == other.fr_multiplicative_gen &&
           fr_multiplicative_gen_inv == other.fr_multiplicative_gen_inv &&
           with_precompute == other.with_precompute;
  }
  bool operator!=(const Domain& other) const { return !operator==(other); }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
    return buffer.ReadMany(&cardinality, &cardinality_inv, &generator,
                           &generator_inv, &fr_multiplicative_gen,
                           &fr_multiplicative_gen_inv, &with_precompute);
  }
};

// See
// https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/fr/pedersen/pedersen.go#L18-L22
template <typename G1AffinePoint>
struct PedersenProvingKey {
  std::vector<G1AffinePoint> basis;
  std::vector<G1AffinePoint> basis_exp_sigma;

  bool operator==(const PedersenProvingKey& other) const {
    return basis == other.basis && basis_exp_sigma == other.basis_exp_sigma;
  }
  bool operator!=(const PedersenProvingKey& other) const {
    return !operator==(other);
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &basis));
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &basis_exp_sigma));
    return absl::OkStatus();
  }

  absl::Status ReadDump(const zkx::base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(ReadSliceWithLength(buffer, &basis));
    TF_RETURN_IF_ERROR(ReadSliceWithLength(buffer, &basis_exp_sigma));
    return absl::OkStatus();
  }
};

// See
// https://github.com/Consensys/gnark/blob/a9f014f/backend/groth16/bn254/setup.go#L23-L48
template <typename Curve>
struct ProvingKey {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using ScalarField = typename Curve::G1Curve::ScalarField;

  Domain<ScalarField> domain;

  G1AffinePoint alpha_g1;
  G1AffinePoint beta_g1;
  G1AffinePoint delta_g1;
  std::vector<G1AffinePoint> a_g1_query;
  std::vector<G1AffinePoint> b_g1_query;
  std::vector<G1AffinePoint> z_g1_query;
  std::vector<G1AffinePoint> k_g1_query;

  G2AffinePoint beta_g2;
  G2AffinePoint delta_g2;
  std::vector<G2AffinePoint> b_g2_query;

  std::vector<bool> infinity_a;
  std::vector<bool> infinity_b;
  uint64_t num_infinity_a;
  uint64_t num_infinity_b;

  std::vector<PedersenProvingKey<G1AffinePoint>> commitment_keys;

  bool operator==(const ProvingKey& other) const {
    return alpha_g1 == other.alpha_g1 && beta_g1 == other.beta_g1 &&
           delta_g1 == other.delta_g1 && a_g1_query == other.a_g1_query &&
           b_g1_query == other.b_g1_query && z_g1_query == other.z_g1_query &&
           k_g1_query == other.k_g1_query && beta_g2 == other.beta_g2 &&
           delta_g2 == other.delta_g2 && b_g2_query == other.b_g2_query &&
           infinity_a == other.infinity_a && infinity_b == other.infinity_b &&
           num_infinity_a == other.num_infinity_a &&
           num_infinity_b == other.num_infinity_b &&
           commitment_keys == other.commitment_keys;
  }
  bool operator!=(const ProvingKey& other) const { return !operator==(other); }

  // See
  // https://github.com/Consensys/gnark/blob/a9f014f/backend/groth16/bn254/marshal.go#L315-L373
  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(domain.Read(buffer));
    TF_RETURN_IF_ERROR(buffer.ReadMany(&alpha_g1, &beta_g1, &delta_g1));
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &a_g1_query));
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &b_g1_query));
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &z_g1_query));
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &k_g1_query));
    TF_RETURN_IF_ERROR(buffer.ReadMany(&beta_g2, &delta_g2));
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &b_g2_query));
    uint64_t num_wires;
    TF_RETURN_IF_ERROR(
        buffer.ReadMany(&num_wires, &num_infinity_a, &num_infinity_b));
    infinity_a.resize(num_wires);
    infinity_b.resize(num_wires);
    TF_RETURN_IF_ERROR(ReadElements(buffer, &infinity_a));
    TF_RETURN_IF_ERROR(ReadElements(buffer, &infinity_b));
    uint32_t num_commitment_keys;
    TF_RETURN_IF_ERROR(buffer.Read(&num_commitment_keys));
    commitment_keys.resize(num_commitment_keys);
    for (uint32_t i = 0; i < num_commitment_keys; ++i) {
      TF_RETURN_IF_ERROR(commitment_keys[i].Read(buffer));
    }
    return absl::OkStatus();
  }

  // See
  // https://github.com/Consensys/gnark/blob/a9f014f/backend/groth16/bn254/marshal.go#L447-L539
  absl::Status ReadDump(const zkx::base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(ReadUnsafeMarker(buffer));
    TF_RETURN_IF_ERROR(domain.Read(buffer));
    uint64_t num_wires;
    TF_RETURN_IF_ERROR(buffer.ReadMany(&alpha_g1, &beta_g1, &delta_g1, &beta_g2,
                                       &delta_g2, &num_wires, &num_infinity_a,
                                       &num_infinity_b));
    infinity_a.resize(num_wires);
    infinity_b.resize(num_wires);
    TF_RETURN_IF_ERROR(ReadElements(buffer, &infinity_a));
    TF_RETURN_IF_ERROR(ReadElements(buffer, &infinity_b));

    uint32_t num_commitment_keys;
    TF_RETURN_IF_ERROR(buffer.Read(&num_commitment_keys));

    TF_RETURN_IF_ERROR(ReadSliceWithLength(buffer, &a_g1_query));
    TF_RETURN_IF_ERROR(ReadSliceWithLength(buffer, &b_g1_query));
    TF_RETURN_IF_ERROR(ReadSliceWithLength(buffer, &z_g1_query));
    TF_RETURN_IF_ERROR(ReadSliceWithLength(buffer, &k_g1_query));
    TF_RETURN_IF_ERROR(ReadSliceWithLength(buffer, &b_g2_query));

    commitment_keys.resize(num_commitment_keys);
    for (uint32_t i = 0; i < num_commitment_keys; ++i) {
      TF_RETURN_IF_ERROR(commitment_keys[i].ReadDump(buffer));
    }
    return absl::OkStatus();
  }
};

template <typename Curve>
absl::StatusOr<std::unique_ptr<ProvingKey<Curve>>> ParseProvingKey(
    std::string_view path, SerdeMode mode) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using ScalarField = typename Curve::G1Curve::ScalarField;
  using BaseField = typename Curve::G1Curve::BaseField;

  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(path, &region));
  if (madvise(const_cast<void*>(region->data()), region->length(),
              MADV_SEQUENTIAL) != 0) {
    return absl::InternalError("failed to call madvice()");
  }

  std::unique_ptr<ProvingKey<Curve>> proving_key(new ProvingKey<Curve>());
  zkx::base::ReadOnlyBuffer buffer(region->data(), region->length());
  buffer.set_endian(zkx::base::Endian::kBig);
  zkx::base::AutoReset<bool> reset_scalar_field_is_in_montgomery(
      &zkx::base::Serde<ScalarField>::s_is_in_montgomery, false);
  zkx::base::AutoReset<bool> reset_base_field_is_in_montgomery(
      &zkx::base::Serde<BaseField>::s_is_in_montgomery, false);
  zkx::base::AutoReset<zkx::math::AffinePointSerdeMode> reset_g1_s_mode(
      &zkx::base::Serde<G1AffinePoint>::s_mode, ToAffinePointSerdeMode(mode));
  zkx::base::AutoReset<zkx::math::AffinePointSerdeMode> reset_g2_s_mode(
      &zkx::base::Serde<G2AffinePoint>::s_mode, ToAffinePointSerdeMode(mode));
  if (mode == SerdeMode::kDump) {
    TF_RETURN_IF_ERROR(proving_key->ReadDump(buffer));
  } else {
    TF_RETURN_IF_ERROR(proving_key->Read(buffer));
  }
  return proving_key;
}

}  // namespace rabbitsnark::gnark

#endif  // GNARK_PK_PROVING_KEY_H_
