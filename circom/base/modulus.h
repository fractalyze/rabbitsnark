#ifndef CIRCOM_BASE_MODULUS_H_
#define CIRCOM_BASE_MODULUS_H_

#include <array>
#include <vector>

#include "absl/log/check.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/read_only_buffer.h"
#include "zkx/base/logging.h"
#include "zkx/math/base/big_int.h"

namespace zkx::circom {

struct Modulus {
  std::vector<uint8_t> bytes;

  bool operator==(const Modulus& other) const { return bytes == other.bytes; }
  bool operator!=(const Modulus& other) const { return bytes != other.bytes; }

  template <size_t N>
  math::BigInt<N> ToBigInt() const {
    CHECK_EQ(bytes.size() / 8, N);
    return math::BigInt<N>::FromBytesLE(bytes);
  }

  template <size_t N>
  static Modulus FromBigInt(const math::BigInt<N>& big_int) {
    std::array<uint8_t, N * 8> bytes = big_int.ToBytesLE();
    return {{bytes.begin(), bytes.end()}};
  }

  absl::Status Read(const base::ReadOnlyBuffer& buffer) {
    uint32_t num_bytes;
    TF_RETURN_IF_ERROR(buffer.Read(&num_bytes));
    if (num_bytes % 8 != 0) {
      return absl::InvalidArgumentError("field size is not a multiple of 8");
    }
    bytes.resize(num_bytes);
    return buffer.Read(bytes.data(), bytes.size());
  }

  std::string ToString() const;
};

}  // namespace zkx::circom

#endif  // CIRCOM_BASE_MODULUS_H_
