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

#ifndef COMMON_COMMAND_RUNNER_INTERFACE_H_
#define COMMON_COMMAND_RUNNER_INTERFACE_H_

#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/file_system.h"
#include "zkx/literal.h"
#include "zkx/primitive_util.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/shape_util.h"

namespace rabbitsnark {

struct Options {
  std::string proving_key_path;
  std::string r1cs_path;
  std::string witness_path;
  std::string proof_path;
  std::string public_path;
  std::string output_dir;
  int32_t h_msm_window_bits;
  int32_t non_h_msm_window_bits;
  bool skip_hlo;
  bool no_zk;
};

class CommandRunnerInterface {
 public:
  CommandRunnerInterface();
  virtual ~CommandRunnerInterface() = default;

  virtual absl::Status Compile(const Options& options) = 0;
  virtual absl::Status Prove(const Options& options) = 0;

 protected:
  template <typename T>
  absl::Status AddScalarParameter(
      const T& value, std::vector<zkx::ScopedShapedBuffer>* buffers) {
    zkx::BorrowingLiteral literal(
        reinterpret_cast<const char*>(&value),
        zkx::ShapeUtil::MakeScalarShape(
            zkx::primitive_util::NativeToPrimitiveType<T>()));
    TF_ASSIGN_OR_RETURN(
        zkx::ScopedShapedBuffer shaped_buffer,
        runner_.TransferLiteralToDevice(literal, buffers->size()));
    buffers->push_back(std::move(shaped_buffer));
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status AddVectorParameter(
      const absl::Span<const T> values,
      std::vector<zkx::ScopedShapedBuffer>* buffers) {
    zkx::BorrowingLiteral literal(
        reinterpret_cast<const char*>(values.data()),
        zkx::ShapeUtil::MakeShape(
            zkx::primitive_util::NativeToPrimitiveType<T>(),
            {static_cast<int64_t>(values.size())}));
    TF_ASSIGN_OR_RETURN(
        zkx::ScopedShapedBuffer shaped_buffer,
        runner_.TransferLiteralToDevice(literal, buffers->size()));
    buffers->push_back(std::move(shaped_buffer));
    return absl::OkStatus();
  }

  absl::Status AddSparseMatrixParameterFromFile(
      const Options& options, std::string_view fname,
      std::vector<zkx::ScopedShapedBuffer>* buffers,
      std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions);

  absl::Status AddVectorParameterFromFile(
      const Options& options, std::string_view fname, const zkx::Shape& shape,
      std::vector<zkx::ScopedShapedBuffer>* buffers,
      std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions);

  zkx::HloRunner runner_;
};

}  // namespace rabbitsnark

#endif  // COMMON_COMMAND_RUNNER_INTERFACE_H_
