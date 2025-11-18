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

#ifndef CIRCOM_JSON_PUBLIC_READER_H_
#define CIRCOM_JSON_PUBLIC_READER_H_

#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "rapidjson/document.h"

#include "common/json/json_reader.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/literal_util.h"

namespace rabbitsnark::circom {

template <typename F>
absl::StatusOr<zkx::Literal> ReadPublicFromJson(std::string_view path) {
  TF_ASSIGN_OR_RETURN(rapidjson::Document document, ReadFromJson(path));

  const rapidjson::Value& array = document.GetArray();
  if (array.GetType() != rapidjson::Type::kArrayType) {
    return absl::InvalidArgumentError("public is not an array");
  }
  std::vector<F> public_values;
  public_values.reserve(array.Size());
  for (auto it = array.Begin(); it != array.End(); ++it) {
    std::string_view string_value = it->GetString();
    TF_ASSIGN_OR_RETURN(F public_value, F::FromDecString(string_value));
    public_values.push_back(public_value);
  }

  return zkx::LiteralUtil::CreateR1<F>(absl::MakeConstSpan(public_values));
}

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_JSON_PUBLIC_READER_H_
