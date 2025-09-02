#ifndef COMMAND_RUNNER_H_
#define COMMAND_RUNNER_H_

#include "absl/status/status.h"

namespace rabbitsnark {

class CommandRunner {
 public:
  CommandRunner() = default;
  CommandRunner(const CommandRunner&) = delete;
  CommandRunner& operator=(const CommandRunner&) = delete;
  ~CommandRunner() = default;

  absl::Status Run(int argc, char** argv);
};

}  // namespace rabbitsnark

#endif  // COMMAND_RUNNER_H_
