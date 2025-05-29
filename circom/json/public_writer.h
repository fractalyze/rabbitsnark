#ifndef CIRCOM_JSON_PUBLIC_WRITER_H_
#define CIRCOM_JSON_PUBLIC_WRITER_H_

#include <string>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "rapidjson/document.h"

#include "circom/json/json_writer.h"

namespace zkx::circom {

template <typename F>
absl::Status WritePublicToJson(absl::Span<const F> public_values,
                               const std::string& path) {
  rapidjson::Document doc;
  doc.SetArray();

  for (const F& public_value : public_values) {
    rapidjson::Value value;
    value.SetString(F(public_value.value()).ToString(), doc.GetAllocator());
    doc.PushBack(value, doc.GetAllocator());
  }

  return WriteToJson(doc, path);
}

}  // namespace zkx::circom

#endif  // CIRCOM_JSON_PUBLIC_WRITER_H_
