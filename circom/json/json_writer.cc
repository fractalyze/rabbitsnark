#include "circom/json/json_writer.h"

#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "xla/tsl/platform/env.h"

namespace rabbitsnark::circom {

absl::Status WriteToJson(rapidjson::Document& doc, std::string_view path) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  return tsl::WriteStringToFile(tsl::Env::Default(), path, buffer.GetString());
}

}  // namespace rabbitsnark::circom
