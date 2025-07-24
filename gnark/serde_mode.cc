#include "gnark/serde_mode.h"

#include "absl/log/check.h"

namespace zkx::gnark {

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

}  // namespace zkx::gnark
