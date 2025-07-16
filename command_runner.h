#ifndef COMMAND_RUNNER_H_
#define COMMAND_RUNNER_H_

#include "absl/status/status.h"

namespace zkx {

class CommandRunner {
 public:
  CommandRunner() = default;
  CommandRunner(const CommandRunner&) = delete;
  CommandRunner& operator=(const CommandRunner&) = delete;
  ~CommandRunner() = default;

  absl::Status Run(int argc, char** argv);
};

}  // namespace zkx

#endif  // COMMAND_RUNNER_H_
