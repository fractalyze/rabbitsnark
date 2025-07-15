#include "common/command_runner_interface.h"

#include "xla/tsl/platform/path.h"
#include "zkx/service/platform_util.h"

namespace zkx {

CommandRunnerInterface::CommandRunnerInterface()
    : runner_(PlatformUtil::GetPlatform("cpu").value()) {}

absl::Status CommandRunnerInterface::AddSparseMatrixParameterFromFile(
    const Options& options, std::string_view fname,
    std::vector<ScopedShapedBuffer>* buffers,
    std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions) {
  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
      tsl::io::JoinPath(options.output_dir, fname), &region));
  ShapedBuffer shaped_buffer(
      ShapeUtil::MakeShape(U8, {static_cast<int64_t>(region->length())}), 0);
  shaped_buffer.set_buffer(
      se::DeviceMemoryBase(const_cast<void*>(region->data()), region->length()),
      ShapeIndex{});
  buffers->push_back(ScopedShapedBuffer(std::move(shaped_buffer), nullptr));
  regions->push_back(std::move(region));
  return absl::OkStatus();
}

absl::Status CommandRunnerInterface::AddVectorParameterFromFile(
    const Options& options, std::string_view fname, const Shape& shape,
    std::vector<ScopedShapedBuffer>* buffers,
    std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions) {
  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
      tsl::io::JoinPath(options.output_dir, fname), &region));
  ShapedBuffer shaped_buffer(shape, 0);
  shaped_buffer.set_buffer(
      se::DeviceMemoryBase(const_cast<void*>(region->data()), region->length()),
      ShapeIndex{});
  buffers->push_back(ScopedShapedBuffer(std::move(shaped_buffer), nullptr));
  regions->push_back(std::move(region));
  return absl::OkStatus();
}

}  // namespace zkx
