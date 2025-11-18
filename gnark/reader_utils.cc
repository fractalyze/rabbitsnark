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

#include "gnark/reader_utils.h"

#include <vector>

namespace rabbitsnark::gnark {

namespace base = zkx::base;

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

}  // namespace rabbitsnark::gnark
