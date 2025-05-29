#ifndef CIRCOM_JSON_JSON_WRITER_H_
#define CIRCOM_JSON_JSON_WRITER_H_

#include <string>

#include "absl/status/status.h"
#include "rapidjson/document.h"

namespace zkx::circom {

absl::Status WriteToJson(rapidjson::Document& doc, const std::string& path);

}  // namespace zkx::circom

#endif  // CIRCOM_JSON_JSON_WRITER_H_
