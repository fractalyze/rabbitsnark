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

#ifndef GNARK_BIN_PUBLIC_WRITER_H_
#define GNARK_BIN_PUBLIC_WRITER_H_

#include <string_view>

#include "absl/status/status.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/endian.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace rabbitsnark::gnark {

template <typename F>
absl::Status WritePublicToBin(absl::Span<const F> public_values,
                              std::string_view path) {
  zkx::base::Uint8VectorBuffer write_buf;
  size_t size = 0;
  if (public_values.size() > 0) {
    size += public_values.size() * zkx::base::EstimateSize(public_values[0]);
  }
  TF_RETURN_IF_ERROR(write_buf.Grow(size + 3 * sizeof(uint32_t)));
  write_buf.set_endian(zkx::base::Endian::kBig);
  zkx::base::AutoReset<bool> reset_scalar_field_is_in_montgomery(
      &zkx::base::Serde<F>::s_is_in_montgomery, false);
  TF_RETURN_IF_ERROR(write_buf.WriteMany(
      static_cast<uint32_t>(public_values.size()), uint32_t{0},
      static_cast<uint32_t>(public_values.size())));
  for (size_t i = 0; i < public_values.size(); i++) {
    TF_RETURN_IF_ERROR(write_buf.Write(public_values[i]));
  }
  std::string_view public_string(
      reinterpret_cast<const char*>(write_buf.buffer()),
      write_buf.buffer_offset());
  return tsl::WriteStringToFile(tsl::Env::Default(), path, public_string);
}

}  // namespace rabbitsnark::gnark

#endif  // GNARK_BIN_PUBLIC_WRITER_H_
