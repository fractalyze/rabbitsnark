#ifndef GNARK_BIN_PUBLIC_READER_H_
#define GNARK_BIN_PUBLIC_READER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/endian.h"
#include "zkx/base/buffer/read_only_buffer.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"

namespace rabbitsnark::gnark {

template <typename F>
absl::StatusOr<zkx::Literal> ReadPublicFromBin(std::string_view path) {
  std::string content;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), path, &content));

  zkx::base::ReadOnlyBuffer read_buf(content.data(), content.size());

  read_buf.set_endian(zkx::base::Endian::kBig);
  zkx::base::AutoReset<bool> reset_scalar_field_is_in_montgomery(
      &zkx::base::Serde<F>::s_is_in_montgomery, false);

  uint32_t size;
  uint32_t unused;
  uint32_t size2;
  TF_RETURN_IF_ERROR(read_buf.ReadMany(&size, &unused, &size2));
  if (size != size2) {
    return absl::InvalidArgumentError(
        absl::Substitute("size mismatch: $0 != $1", size, size2));
  }
  std::vector<F> public_values;
  public_values.resize(size);
  for (size_t i = 0; i < size; i++) {
    TF_RETURN_IF_ERROR(read_buf.Read(&public_values[i]));
  }
  return zkx::LiteralUtil::CreateR1<F>(absl::MakeConstSpan(public_values));
}

}  // namespace rabbitsnark::gnark

#endif  // GNARK_BIN_PUBLIC_READER_H_
