#include "gnark/reader_utils.h"

#include <vector>

namespace zkx::gnark {

// See
// https://github.com/Consensys/gnark-crypto/blob/43897fd/utils/unsafe/dump_slice.go#L89-L102
absl::Status ReadUnsafeMarker(const base::ReadOnlyBuffer& buffer) {
  base::EndianAutoReset auto_reset(buffer, base::Endian::kNative);
  uint64_t marker;
  TF_RETURN_IF_ERROR(buffer.Read(&marker));
  if (marker != 0xdeadbeef) {
    return absl::NotFoundError("Invalid unsafe marker");
  }
  return absl::OkStatus();
}

absl::Status ReadElements(const base::ReadOnlyBuffer& buffer,
                          std::vector<bool>* result) {
  for (uint32_t i = 0; i < result->size(); ++i) {
    bool value;
    TF_RETURN_IF_ERROR(buffer.Read(&value));
    (*result)[i] = value;
  }
  return absl::OkStatus();
}

}  // namespace zkx::gnark
