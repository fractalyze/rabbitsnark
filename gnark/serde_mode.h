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

#ifndef GNARK_SERDE_MODE_H_
#define GNARK_SERDE_MODE_H_

#include "zkx/math/elliptic_curves/short_weierstrass/affine_point.h"

namespace rabbitsnark::gnark {

enum class SerdeMode {
  // For data saved with WriteTo(); must be read with ReadFrom().
  kDefault,
  // For data saved with WriteRawTo(); must be read with ReadFrom().
  kRaw,
  // For data saved with Gnark's WriteDump(); must be read with ReadDump().
  kDump,
};

zkx::math::AffinePointSerdeMode ToAffinePointSerdeMode(SerdeMode mode);

}  // namespace rabbitsnark::gnark

#endif  // GNARK_SERDE_MODE_H_
