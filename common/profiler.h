#ifndef COMMON_PROFILER_H_
#define COMMON_PROFILER_H_

#include <iostream>

#include "absl/time/clock.h"

#define RUN_WITH_PROFILE(tag, expr)                                       \
  do {                                                                    \
    std::cout << "┌────────────────────────────────────────────┐"         \
              << std::endl;                                               \
    std::cout << "│ 🚀 Running: " << tag << std::endl;                    \
    std::cout << "└────────────────────────────────────────────┘"         \
              << std::endl;                                               \
    absl::Time start = absl::Now();                                       \
    expr;                                                                 \
    absl::Time end = absl::Now();                                         \
    std::cout << "⏱️ Duration [" << tag                                    \
              << "]: " << absl::FormatDuration(end - start) << std::endl; \
  } while (false)

#endif  // COMMON_PROFILER_H_
