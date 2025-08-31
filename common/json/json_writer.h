#ifndef COMMON_JSON_JSON_WRITER_H_
#define COMMON_JSON_JSON_WRITER_H_

#include <string_view>

#include "absl/status/status.h"
#include "rapidjson/document.h"

namespace rabbitsnark {

absl::Status WriteToJson(rapidjson::Document& doc, std::string_view path);

}  // namespace rabbitsnark

#endif  // COMMON_JSON_JSON_WRITER_H_
