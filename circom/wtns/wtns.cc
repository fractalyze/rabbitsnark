#include "circom/wtns/wtns.h"

namespace rabbitsnark::circom::v2 {

std::string_view WtnsSectionTypeToString(WtnsSectionType type) {
  switch (type) {
    case WtnsSectionType::kHeader:
      return "Header";
    case WtnsSectionType::kData:
      return "Data";
  }
}

}  // namespace rabbitsnark::circom::v2
