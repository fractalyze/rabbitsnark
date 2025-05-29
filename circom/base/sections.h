#ifndef CIRCOM_BASE_SECTIONS_H_
#define CIRCOM_BASE_SECTIONS_H_

#include <stdint.h>

#include <string_view>
#include <vector>

#include "absl/strings/substitute.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/read_only_buffer.h"
#include "zkx/base/logging.h"

namespace zkx::circom {

template <typename T>
struct Section {
  T type;
  uint64_t from;
  uint64_t size;
};

template <typename T>
class Sections {
 public:
  typedef std::string_view (*ErrorFn)(T type);

  Sections(const base::ReadOnlyBuffer& buffer, ErrorFn error_fn)
      : buffer_(buffer), error_fn_(error_fn) {}

  absl::Status Read() {
    uint32_t num_sections;
    TF_RETURN_IF_ERROR(buffer_.Read(&num_sections));

    sections_.reserve(num_sections);
    for (uint32_t i = 0; i < num_sections; ++i) {
      TF_RETURN_IF_ERROR(Add());
    }
    return absl::OkStatus();
  }

  absl::Status MoveTo(T type) const {
    auto it = std::find_if(
        sections_.begin(), sections_.end(),
        [type](const Section<T>& section) { return section.type == type; });
    if (it == sections_.end()) {
      return absl::InvalidArgumentError(
          absl::Substitute("$0 is empty", error_fn_(type)));
    }
    buffer_.set_buffer_offset(it->from);
    return absl::OkStatus();
  }

 private:
  absl::Status Add() {
    T type;
    uint64_t size;
    TF_RETURN_IF_ERROR(buffer_.ReadMany(&type, &size));

    sections_.push_back({type, buffer_.buffer_offset(), size});
    buffer_.set_buffer_offset(buffer_.buffer_offset() + size);
    return absl::OkStatus();
  }

  const base::ReadOnlyBuffer& buffer_;
  ErrorFn error_fn_;
  std::vector<Section<T>> sections_;
};

}  // namespace zkx::circom

#endif  // CIRCOM_BASE_SECTIONS_H_
