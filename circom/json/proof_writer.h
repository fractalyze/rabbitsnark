#ifndef CIRCOM_JSON_PROOF_WRITER_H_
#define CIRCOM_JSON_PROOF_WRITER_H_

#include <string>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "rapidjson/document.h"

#include "circom/json/json_writer.h"
#include "zkx/literal.h"
#include "zkx/math/elliptic_curves/bn/bn254/curve.h"
#include "zkx/math/geometry/point_declarations.h"

namespace zkx::circom {

template <typename Curve>
void AddPoint(rapidjson::Document& doc, std::string_view key,
              const math::AffinePoint<Curve>& point) {
  using BaseField = typename math::AffinePoint<Curve>::BaseField;

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
absl::Status WriteProofToJson(Literal& proof, const std::string& path) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  rapidjson::Document doc;
  std::vector<Literal> decomposed = proof.DecomposeTuple();

  doc.SetObject();
  AddPoint(doc, "pi_a", decomposed[0].data<G1AffinePoint>()[0]);
  AddPoint(doc, "pi_b", decomposed[1].data<G2AffinePoint>()[0]);
  AddPoint(doc, "pi_c", decomposed[2].data<G1AffinePoint>()[0]);

  doc.AddMember("protocol", "groth16", doc.GetAllocator());
  return WriteToJson(doc, path);
}

}  // namespace zkx::circom

#endif  // CIRCOM_JSON_PROOF_WRITER_H_
