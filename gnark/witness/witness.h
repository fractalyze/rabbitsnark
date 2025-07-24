#ifndef GNARK_WITNESS_WITNESS_H_
#define GNARK_WITNESS_WITNESS_H_

#include <sys/mman.h>

#include "gnark/reader_utils.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/endian_auto_reset.h"
#include "zkx/base/buffer/read_only_buffer.h"

namespace zkx::gnark {

template <typename F>
struct Witness {
  uint32_t num_publics;
  uint32_t num_secrets;
  std::vector<F> secrets;

  bool operator==(const Witness& other) const {
    return num_publics == other.num_publics && num_secrets == other.num_secrets;
  }
  bool operator!=(const Witness& other) const { return !operator==(other); }

  // See
  // https://github.com/Consensys/gnark/blob/a9f014f/backend/witness/witness.go#L205-L256
  absl::Status Read(const base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(buffer.ReadMany(&num_publics, &num_secrets));
    TF_RETURN_IF_ERROR(ReadElementsWithLength(buffer, &secrets));
    return absl::OkStatus();
  }
};

// The witness must be stored with Gnark's MarshalBinary
template <typename F>
absl::StatusOr<std::unique_ptr<Witness<F>>> ParseWitness(
    const std::string& path) {
  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(path, &region));
  if (madvise(const_cast<void*>(region->data()), region->length(),
              MADV_SEQUENTIAL) != 0) {
    return absl::InternalError("failed to call madvice()");
  }

  std::unique_ptr<Witness<F>> witness(new Witness<F>());
  base::ReadOnlyBuffer buffer(region->data(), region->length());
  buffer.set_endian(base::Endian::kBig);
  base::AutoReset<bool> reset_scalar_field_is_in_montgomery(
      &base::Serde<F>::s_is_in_montgomery, false);
  TF_RETURN_IF_ERROR(witness->Read(buffer));
  return witness;
}

}  // namespace zkx::gnark

#endif  // GNARK_WITNESS_WITNESS_H_
