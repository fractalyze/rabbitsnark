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

#ifndef CIRCOM_WTNS_WTNS_H_
#define CIRCOM_WTNS_WTNS_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"

#include "circom/base/modulus.h"
#include "circom/base/sections.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/buffer/read_only_buffer.h"

namespace rabbitsnark::circom {
namespace v2 {

template <typename F>
struct Wtns;

}  // namespace v2

template <typename F>
struct Wtns {
  explicit Wtns(std::unique_ptr<tsl::ReadOnlyMemoryRegion> region)
      : region(std::move(region)) {}
  virtual ~Wtns() = default;

  virtual uint32_t GetVersion() const = 0;

  virtual v2::Wtns<F>* ToV2() { return nullptr; }

  virtual absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) = 0;

  virtual size_t GetNumWitness() const = 0;

  virtual absl::Span<const F> GetWitnesses() const = 0;

  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
};

constexpr char kWtnsMagic[4] = {'w', 't', 'n', 's'};

template <typename F>
absl::StatusOr<std::unique_ptr<Wtns<F>>> ParseWtns(std::string_view path) {
  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(path, &region));
  if (madvise(const_cast<void*>(region->data()), region->length(),
              MADV_SEQUENTIAL) != 0) {
    return absl::InternalError("failed to call madvice()");
  }

  zkx::base::ReadOnlyBuffer buffer(region->data(), region->length());
  buffer.set_endian(zkx::base::Endian::kLittle);
  char magic[4];
  uint32_t version;
  TF_RETURN_IF_ERROR(buffer.ReadMany(magic, &version));
  if (memcmp(magic, kWtnsMagic, 4) != 0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "magic is invalid. \"$0\" vs \"$1\"", magic, kWtnsMagic));
  }
  std::unique_ptr<Wtns<F>> wtns;
  if (version == 2) {
    wtns.reset(new v2::Wtns<F>(std::move(region)));
    TF_RETURN_IF_ERROR(wtns->ToV2()->Read(buffer));
  } else {
    return absl::InvalidArgumentError(
        absl::Substitute("version is invalid. 2 vs $0", version));
  }
  return wtns;
}

namespace v2 {

enum class WtnsSectionType : uint32_t {
  kHeader = 0x1,
  kData = 0x2,
};

std::string_view WtnsSectionTypeToString(WtnsSectionType type);

struct WtnsHeaderSection {
  Modulus modulus;
  uint32_t num_witness;

  bool operator==(const WtnsHeaderSection& other) const {
    return modulus == other.modulus && num_witness == other.num_witness;
  }
  bool operator!=(const WtnsHeaderSection& other) const {
    return !operator==(other);
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(modulus.Read(buffer));
    return buffer.ReadMany(&num_witness);
  }
};

template <typename F>
struct WtnsDataSection {
  absl::Span<const F> witnesses;

  bool operator==(const WtnsDataSection& other) const {
    return witnesses == other.witnesses;
  }
  bool operator!=(const WtnsDataSection& other) const {
    return witnesses != other.witnesses;
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer,
                    const WtnsHeaderSection& header) {
    F* ptr;
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&ptr, header.num_witness));
    witnesses = {ptr, header.num_witness};
    return absl::OkStatus();
  }
};

template <typename F>
struct Wtns : public circom::Wtns<F> {
  WtnsHeaderSection header;
  WtnsDataSection<F> data;

  explicit Wtns(std::unique_ptr<tsl::ReadOnlyMemoryRegion> region)
      : circom::Wtns<F>(std::move(region)) {}

  // circom::Wtns methods
  uint32_t GetVersion() const override { return 2; }
  Wtns* ToV2() override { return this; }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) override {
    Sections<WtnsSectionType> sections(buffer, &WtnsSectionTypeToString);
    TF_RETURN_IF_ERROR(sections.Read());

    TF_RETURN_IF_ERROR(sections.MoveTo(WtnsSectionType::kHeader));
    TF_RETURN_IF_ERROR(header.Read(buffer));

    TF_RETURN_IF_ERROR(sections.MoveTo(WtnsSectionType::kData));
    TF_RETURN_IF_ERROR(data.Read(buffer, header));
    return absl::OkStatus();
  }

  size_t GetNumWitness() const override { return data.witnesses.size(); }

  absl::Span<const F> GetWitnesses() const override { return data.witnesses; }
};

}  // namespace v2
}  // namespace rabbitsnark::circom

#endif  // CIRCOM_WTNS_WTNS_H_
