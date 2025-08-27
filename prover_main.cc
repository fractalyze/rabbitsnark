#include <iostream>

#include "command_runner.h"  // NOLINT(build/include_subdir)

int main(int argc, char** argv) {
  zkx::CommandRunner runner;
  absl::Status s = runner.Run(argc, argv);
  if (!s.ok()) {
    std::cerr << s.message() << std::endl;
    return 1;
  }
  return 0;
}
