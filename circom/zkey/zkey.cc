/* Copyright 2025 The RabbitSNARK Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
