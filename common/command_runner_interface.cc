#include "common/command_runner_interface.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/tsl/platform/path.h"
#include "zkx/service/platform_util.h"

namespace rabbitsnark {

CommandRunnerInterface::CommandRunnerInterface()
    : runner_(zkx::PlatformUtil::GetPlatform("cpu").value()) {}

absl::Status CommandRunnerInterface::AddSparseMatrixParameterFromFile(
    const Options& options, std::string_view fname,
    std::vector<zkx::ScopedShapedBuffer>* buffers,
    std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions) {
  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
      tsl::io::JoinPath(options.output_dir, fname), &region));
  zkx::ShapedBuffer shaped_buffer(
      zkx::ShapeUtil::MakeShape(zkx::U8,
                                {static_cast<int64_t>(region->length())}),
      0);
  shaped_buffer.set_buffer(
      se::DeviceMemoryBase(const_cast<void*>(region->data()), region->length()),
      zkx::ShapeIndex{});
  buffers->push_back(
      zkx::ScopedShapedBuffer(std::move(shaped_buffer), nullptr));
  regions->push_back(std::move(region));
  return absl::OkStatus();
}

absl::Status CommandRunnerInterface::AddVectorParameterFromFile(
    const Options& options, std::string_view fname, const zkx::Shape& shape,
    std::vector<zkx::ScopedShapedBuffer>* buffers,
    std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions) {
  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
      tsl::io::JoinPath(options.output_dir, fname), &region));
  zkx::ShapedBuffer shaped_buffer(shape, 0);
  shaped_buffer.set_buffer(
      se::DeviceMemoryBase(const_cast<void*>(region->data()), region->length()),
      zkx::ShapeIndex{});
  buffers->push_back(
      zkx::ScopedShapedBuffer(std::move(shaped_buffer), nullptr));
  regions->push_back(std::move(region));
  return absl::OkStatus();
}

}  // namespace rabbitsnark
