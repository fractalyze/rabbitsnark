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

namespace zkx {

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
  absl::Status AddScalarParameter(const T& value,
                                  std::vector<ScopedShapedBuffer>* buffers) {
    BorrowingLiteral literal(
        reinterpret_cast<const char*>(&value),
        ShapeUtil::MakeScalarShape(primitive_util::NativeToPrimitiveType<T>()));
    TF_ASSIGN_OR_RETURN(
        ScopedShapedBuffer shaped_buffer,
        runner_.TransferLiteralToDevice(literal, buffers->size()));
    buffers->push_back(std::move(shaped_buffer));
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status AddVectorParameter(const absl::Span<const T> values,
                                  std::vector<ScopedShapedBuffer>* buffers) {
    BorrowingLiteral literal(
        reinterpret_cast<const char*>(values.data()),
        ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<T>(),
                             {static_cast<int64_t>(values.size())}));
    TF_ASSIGN_OR_RETURN(
        ScopedShapedBuffer shaped_buffer,
        runner_.TransferLiteralToDevice(literal, buffers->size()));
    buffers->push_back(std::move(shaped_buffer));
    return absl::OkStatus();
  }

  absl::Status AddSparseMatrixParameterFromFile(
      const Options& options, std::string_view fname,
      std::vector<ScopedShapedBuffer>* buffers,
      std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions);

  absl::Status AddVectorParameterFromFile(
      const Options& options, std::string_view fname, const Shape& shape,
      std::vector<ScopedShapedBuffer>* buffers,
      std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions);

  HloRunner runner_;
};

}  // namespace zkx

#endif  // COMMON_COMMAND_RUNNER_INTERFACE_H_
