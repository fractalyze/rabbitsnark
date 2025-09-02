#ifndef COMMON_JSON_JSON_READER_H_
#define COMMON_JSON_JSON_READER_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "rapidjson/document.h"

namespace rabbitsnark {

absl::StatusOr<rapidjson::Document> ReadFromJson(std::string_view path);

}  // namespace rabbitsnark

#endif  // COMMON_JSON_JSON_READER_H_
