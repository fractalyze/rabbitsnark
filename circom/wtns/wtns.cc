#include "circom/wtns/wtns.h"

namespace zkx::circom::v2 {

std::string_view WtnsSectionTypeToString(WtnsSectionType type) {
  switch (type) {
    case WtnsSectionType::kHeader:
      return "Header";
    case WtnsSectionType::kData:
      return "Data";
  }
}

}  // namespace zkx::circom::v2
