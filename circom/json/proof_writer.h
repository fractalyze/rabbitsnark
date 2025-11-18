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

#ifndef CIRCOM_JSON_PROOF_WRITER_H_
#define CIRCOM_JSON_PROOF_WRITER_H_

#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "rapidjson/document.h"

#include "common/json/json_writer.h"
#include "zkx/literal.h"
#include "zkx/math/geometry/point_declarations.h"

namespace rabbitsnark::circom {

template <typename Curve>
void AddPoint(rapidjson::Document& doc, std::string_view key,
              const zkx::math::AffinePoint<Curve>& point) {
  using BaseField = typename zkx::math::AffinePoint<Curve>::BaseField;

  rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
  const BaseField& x = point.x();
  const BaseField& y = point.y();
  rapidjson::Value value;
  value.SetArray();
  if constexpr (BaseField::ExtensionDegree() == 1) {
    value.PushBack({x.ToString(), allocator}, allocator);
    value.PushBack({y.ToString(), allocator}, allocator);
    value.PushBack("1", allocator);
  } else {
    static_assert(BaseField::ExtensionDegree() == 2);
    {
      rapidjson::Value inner_array;
      inner_array.SetArray();
      inner_array.PushBack({x[0].ToString(), allocator}, allocator);
      inner_array.PushBack({x[1].ToString(), allocator}, allocator);
      value.PushBack(inner_array, allocator);
    }
    {
      rapidjson::Value inner_array;
      inner_array.SetArray();
      inner_array.PushBack({y[0].ToString(), allocator}, allocator);
      inner_array.PushBack({y[1].ToString(), allocator}, allocator);
      value.PushBack(inner_array, allocator);
    }
    {
      rapidjson::Value inner_array;
      inner_array.SetArray();
      inner_array.PushBack("1", allocator);
      inner_array.PushBack("0", allocator);
      value.PushBack(inner_array, allocator);
    }
  }
  doc.AddMember(rapidjson::StringRef(key.data(), key.size()), value, allocator);
}

template <typename Curve>
absl::Status WriteProofToJson(zkx::Literal& proof, std::string_view path) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  rapidjson::Document doc;
  std::vector<zkx::Literal> decomposed = proof.DecomposeTuple();

  doc.SetObject();
  AddPoint(doc, "pi_a", decomposed[0].data<G1AffinePoint>()[0]);
  AddPoint(doc, "pi_b", decomposed[1].data<G2AffinePoint>()[0]);
  AddPoint(doc, "pi_c", decomposed[2].data<G1AffinePoint>()[0]);

  doc.AddMember("protocol", "groth16", doc.GetAllocator());
  return WriteToJson(doc, path);
}

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_JSON_PROOF_WRITER_H_
