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

#ifndef CIRCOM_JSON_PUBLIC_WRITER_H_
#define CIRCOM_JSON_PUBLIC_WRITER_H_

#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "rapidjson/document.h"

#include "common/json/json_writer.h"

namespace rabbitsnark::circom {

template <typename F>
absl::Status WritePublicToJson(absl::Span<const F> public_values,
                               std::string_view path) {
  rapidjson::Document doc;
  doc.SetArray();

  for (const F& public_value : public_values) {
    rapidjson::Value value;
    value.SetString(F(public_value.value()).ToString(), doc.GetAllocator());
    doc.PushBack(value, doc.GetAllocator());
  }

  return WriteToJson(doc, path);
}

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_JSON_PUBLIC_WRITER_H_
