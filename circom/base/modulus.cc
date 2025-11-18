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

#include "circom/base/modulus.h"

#include <string>

#include "absl/base/optimization.h"

namespace rabbitsnark::circom {

std::string Modulus::ToString() const {
#define MODULUS_TO_STRING(n) ToBigInt<n>().ToString()
  switch (bytes.size() / 8) {
    case 1:
      return MODULUS_TO_STRING(1);
    case 2:
      return MODULUS_TO_STRING(2);
    case 3:
      return MODULUS_TO_STRING(3);
    case 4:
      return MODULUS_TO_STRING(4);
    case 5:
      return MODULUS_TO_STRING(5);
    case 6:
      return MODULUS_TO_STRING(6);
    case 7:
      return MODULUS_TO_STRING(7);
    case 8:
      return MODULUS_TO_STRING(8);
    case 9:
      return MODULUS_TO_STRING(9);
  }
#undef MODULUS_TO_STRING
  ABSL_UNREACHABLE();
  return "";
}

}  // namespace rabbitsnark::circom
