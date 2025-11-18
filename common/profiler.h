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
