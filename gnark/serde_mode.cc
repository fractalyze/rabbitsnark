#include "gnark/serde_mode.h"

#include "absl/base/optimization.h"

namespace rabbitsnark::gnark {

namespace math = zkx::math;

math::AffinePointSerdeMode ToAffinePointSerdeMode(SerdeMode mode) {
  switch (mode) {
    case SerdeMode::kDefault:
      return math::AffinePointSerdeMode::kGnarkDefault;
    case SerdeMode::kRaw:
      return math::AffinePointSerdeMode::kGnarkRaw;
    case SerdeMode::kDump:
      return math::AffinePointSerdeMode::kGnarkRaw;
  }
  ABSL_UNREACHABLE();
  return math::AffinePointSerdeMode::kNone;
}

}  // namespace rabbitsnark::gnark
