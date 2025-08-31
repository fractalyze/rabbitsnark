#include "circom/zkey/zkey.h"

namespace rabbitsnark::circom::v1 {

std::string_view ZKeySectionTypeToString(ZKeySectionType type) {
  switch (type) {
    case ZKeySectionType::kHeader:
      return "Header";
    case ZKeySectionType::kHeaderGroth:
      return "HeaderGroth";
    case ZKeySectionType::kIC:
      return "IC";
    case ZKeySectionType::kCoefficients:
      return "Coefficients";
    case ZKeySectionType::kPointsA1:
      return "PointsA1";
    case ZKeySectionType::kPointsB1:
      return "PointsB1";
    case ZKeySectionType::kPointsB2:
      return "PointsB2";
    case ZKeySectionType::kPointsC1:
      return "PointsC1";
    case ZKeySectionType::kPointsH1:
      return "PointsH1";
    case ZKeySectionType::kContribution:
      return "Contribution";
  }
}

}  // namespace rabbitsnark::circom::v1
