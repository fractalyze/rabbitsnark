#ifndef GNARK_READER_UTILS_H_
#define GNARK_READER_UTILS_H_

#include <stdint.h>
#include <string.h>

#include <vector>

#include "absl/status/status.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/endian_auto_reset.h"
#include "zkx/base/buffer/read_only_buffer.h"

namespace rabbitsnark::gnark {

absl::Status ReadUnsafeMarker(const zkx::base::ReadOnlyBuffer& buffer);

template <typename T>
absl::Status ReadElements(const zkx::base::ReadOnlyBuffer& buffer,
                          std::vector<T>* result) {
  for (uint32_t i = 0; i < result->size(); ++i) {
    TF_RETURN_IF_ERROR(buffer.Read(&(*result)[i]));
  }
  return absl::OkStatus();
}

absl::Status ReadElements(const zkx::base::ReadOnlyBuffer& buffer,
                          std::vector<bool>* result);

// See
// https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/fr/vector.go#L133-L159
template <typename T>
absl::Status ReadElementsWithLength(const zkx::base::ReadOnlyBuffer& buffer,
                                    std::vector<T>* result) {
  uint32_t length;
  TF_RETURN_IF_ERROR(buffer.Read(&length));
  result->resize(length);
  return ReadElements(buffer, result);
}

// See
// https://github.com/Consensys/gnark-crypto/blob/43897fd/utils/unsafe/dump_slice.go#L34-L77
template <typename T>
absl::Status ReadSliceWithLength(const zkx::base::ReadOnlyBuffer& buffer,
                                 std::vector<T>* result) {
  uint64_t length;
  {
    zkx::base::EndianAutoReset auto_reset(buffer, zkx::base::Endian::kLittle);
    TF_RETURN_IF_ERROR(buffer.Read(&length));
  }
  if (length != 0) {
    result->resize(length);
    memcpy(result->data(),
           &(reinterpret_cast<const char*>(
               buffer.buffer())[buffer.buffer_offset()]),
           result->size() * sizeof(T));
    buffer.set_buffer_offset(buffer.buffer_offset() +
                             result->size() * sizeof(T));
  }
  return absl::OkStatus();
}

}  // namespace rabbitsnark::gnark

#endif  // GNARK_READER_UTILS_H_
