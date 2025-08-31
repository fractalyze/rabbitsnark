#ifndef CIRCOM_JSON_JSON_WRITER_H_
#define CIRCOM_JSON_JSON_WRITER_H_

#include <string_view>

#include "absl/status/status.h"
#include "rapidjson/document.h"

namespace rabbitsnark::circom {

absl::Status WriteToJson(rapidjson::Document& doc, std::string_view path);

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_JSON_JSON_WRITER_H_
