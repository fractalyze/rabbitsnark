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

#ifndef CIRCOM_BASE_MODULUS_H_
#define CIRCOM_BASE_MODULUS_H_

#include <array>
#include <string>
#include <vector>

#include "absl/log/check.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/read_only_buffer.h"
#include "zkx/math/base/big_int.h"

namespace rabbitsnark::circom {

struct Modulus {
  std::vector<uint8_t> bytes;

  bool operator==(const Modulus& other) const { return bytes == other.bytes; }
  bool operator!=(const Modulus& other) const { return bytes != other.bytes; }

  template <size_t N>
  zkx::math::BigInt<N> ToBigInt() const {
    CHECK_EQ(bytes.size() / 8, N);
    return zkx::math::BigInt<N>::FromBytesLE(bytes);
  }

  template <size_t N>
  static Modulus FromBigInt(const zkx::math::BigInt<N>& big_int) {
    std::array<uint8_t, N * 8> bytes = big_int.ToBytesLE();
    return {{bytes.begin(), bytes.end()}};
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
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

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_BASE_MODULUS_H_
