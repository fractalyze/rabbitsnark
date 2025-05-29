#ifndef COMMAND_RUNNER_H_
#define COMMAND_RUNNER_H_

#include "absl/status/status.h"

namespace zkx::circom {

class CommandRunner {
 public:
  CommandRunner() = default;
  CommandRunner(const CommandRunner&) = delete;
  CommandRunner& operator=(const CommandRunner&) = delete;
  ~CommandRunner() = default;

  absl::Status Run(int argc, char** argv);
};

}  // namespace zkx::circom

#endif  // COMMAND_RUNNER_H_
