#ifndef GNARK_SERDE_MODE_H_
#define GNARK_SERDE_MODE_H_

#include "zkx/math/elliptic_curves/short_weierstrass/affine_point.h"

namespace zkx::gnark {

enum class SerdeMode {
  // For data saved with WriteTo(); must be read with ReadFrom().
  kDefault,
  // For data saved with WriteRawTo(); must be read with ReadFrom().
  kRaw,
  // For data saved with Gnark's WriteDump(); must be read with ReadDump().
  kDump,
};

math::AffinePointSerdeMode ToAffinePointSerdeMode(SerdeMode mode);

}  // namespace zkx::gnark

#endif  // GNARK_SERDE_MODE_H_
