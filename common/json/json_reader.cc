#include "common/json/json_reader.h"

#include <string>
#include <utility>

#include "absl/strings/substitute.h"
#include "rapidjson/error/en.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"

namespace rabbitsnark {

absl::StatusOr<rapidjson::Document> ReadFromJson(std::string_view path) {
  std::string content;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), path, &content));

  rapidjson::Document document;
  document.Parse(content.data(), content.length());
  if (document.HasParseError()) {
    return absl::InvalidArgumentError(
        absl::Substitute("Failed to parse with error \"$0\" at offset $1",
                         rapidjson::GetParseError_En(document.GetParseError()),
                         document.GetErrorOffset()));
  }

  return std::move(document);
}

}  // namespace rabbitsnark
