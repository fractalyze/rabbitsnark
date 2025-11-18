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

#include "circom/base/sections.h"

#include "gtest/gtest.h"

#include "zkx/base/buffer/serde.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace rabbitsnark::circom {

namespace base = zkx::base;

namespace {

enum class Type {
  kDummy,
  kDummy2,
};

std::string_view TypeToString(Type type) {
  switch (type) {
    case Type::kDummy:
      return "Dummy";
    case Type::kDummy2:
      return "Dummy2";
  }
}

}  // namespace

TEST(SectionsTest, ReadAndGet) {
  {
    base::Uint8VectorBuffer buffer;
    Sections<Type> sections(buffer, &TypeToString);
    // Should return false when it fails to read the number of sections.
    ASSERT_FALSE(sections.Read().ok());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{1}).ok());
    buffer.set_buffer_offset(0);
    Sections<Type> sections(buffer, &TypeToString);
    // Should return false when it fails to read the section type.
    ASSERT_FALSE(sections.Read().ok());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{1}).ok());
    ASSERT_TRUE(buffer.Write(Type::kDummy).ok());
    buffer.set_buffer_offset(0);
    Sections<Type> sections(buffer, &TypeToString);
    // Should return false when it fails to read the section size.
    ASSERT_FALSE(sections.Read().ok());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{1}).ok());
    ASSERT_TRUE(buffer.Write(Type::kDummy).ok());
    ASSERT_TRUE(buffer.Write(uint64_t{32}).ok());
    size_t expected = buffer.buffer_offset();
    buffer.set_buffer_offset(0);
    Sections<Type> sections(buffer, &TypeToString);
    ASSERT_TRUE(sections.Read().ok());

    ASSERT_FALSE(sections.MoveTo(Type::kDummy2).ok());
    ASSERT_TRUE(sections.MoveTo(Type::kDummy).ok());
    EXPECT_EQ(buffer.buffer_offset(), expected);
  }
}

}  // namespace rabbitsnark::circom
