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

#ifndef CIRCOM_JSON_PROOF_READER_H_
#define CIRCOM_JSON_PROOF_READER_H_

#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"

#include "common/json/json_reader.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"

namespace rabbitsnark::circom {

template <typename AffinePoint>
absl::StatusOr<zkx::Literal> ReadPoint(const rapidjson::Value& value) {
  using BaseField = typename AffinePoint::BaseField;

  if (value.GetType() != rapidjson::Type::kArrayType) {
    return absl::InvalidArgumentError("value is not an array");
  }
  if constexpr (BaseField::ExtensionDegree() == 1) {
    if (value.Size() != 3) {
      return absl::InvalidArgumentError("invalid length of G1 point");
    }
    if (std::string_view(value[2].GetString()) != "1") {
      return absl::InvalidArgumentError("invalid G1 point z value");
    }
    TF_ASSIGN_OR_RETURN(BaseField x,
                        BaseField::FromDecString(value[0].GetString()));
    TF_ASSIGN_OR_RETURN(BaseField y,
                        BaseField::FromDecString(value[1].GetString()));
    return zkx::LiteralUtil::CreateR0<AffinePoint>({x, y});
  } else {
    static_assert(BaseField::ExtensionDegree() == 2);
    using BasePrimeField = typename BaseField::BasePrimeField;
    if (value.Size() != 3) {
      return absl::InvalidArgumentError("invalid length of G2 point");
    }
    if (value[0].Size() != 2) {
      return absl::InvalidArgumentError("invalid length of G2 point x");
    }
    if (value[1].Size() != 2) {
      return absl::InvalidArgumentError("invalid length of G2 point y");
    }
    if (value[2].Size() != 2) {
      return absl::InvalidArgumentError("invalid length of G2 point z");
    }
    if (std::string_view(value[2][0].GetString()) != "1" ||
        std::string_view(value[2][1].GetString()) != "0") {
      return absl::InvalidArgumentError("invalid G2 point z value");
    }
    TF_ASSIGN_OR_RETURN(BasePrimeField x0,
                        BasePrimeField::FromDecString(value[0][0].GetString()));
    TF_ASSIGN_OR_RETURN(BasePrimeField x1,
                        BasePrimeField::FromDecString(value[0][1].GetString()));
    TF_ASSIGN_OR_RETURN(BasePrimeField y0,
                        BasePrimeField::FromDecString(value[1][0].GetString()));
    TF_ASSIGN_OR_RETURN(BasePrimeField y1,
                        BasePrimeField::FromDecString(value[1][1].GetString()));
    return zkx::LiteralUtil::CreateR0<AffinePoint>({{x0, x1}, {y0, y1}});
  }
}

template <typename Curve>
absl::StatusOr<zkx::Literal> ReadProofFromJson(std::string_view path) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  TF_ASSIGN_OR_RETURN(rapidjson::Document document, ReadFromJson(path));

  const rapidjson::Value& object = document.GetObject();
  const rapidjson::Value& protocol = object["protocol"];
  if (std::string_view(protocol.GetString()) != "groth16") {
    return absl::InvalidArgumentError("protocol is not groth16");
  }

  std::vector<zkx::Literal> proof_elements;
  TF_ASSIGN_OR_RETURN(proof_elements.emplace_back(),
                      ReadPoint<G1AffinePoint>(object["pi_a"]));
  TF_ASSIGN_OR_RETURN(proof_elements.emplace_back(),
                      ReadPoint<G2AffinePoint>(object["pi_b"]));
  TF_ASSIGN_OR_RETURN(proof_elements.emplace_back(),
                      ReadPoint<G1AffinePoint>(object["pi_c"]));

  return zkx::LiteralUtil::MakeTupleOwned(std::move(proof_elements));
}

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_JSON_PROOF_READER_H_
