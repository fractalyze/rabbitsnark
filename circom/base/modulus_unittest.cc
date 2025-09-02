#include "circom/base/modulus.h"

#include "gtest/gtest.h"

#include "zkx/base/buffer/serde.h"
#include "zkx/base/buffer/vector_buffer.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

namespace rabbitsnark::circom {

namespace base = zkx::base;
namespace math = zkx::math;

using F = math::bn254::Fr;

TEST(ModulusTest, BigIntConversions) {
  math::BigInt<4> expected = math::BigInt<4>::Random();
  Modulus field = Modulus::FromBigInt(expected);
  EXPECT_EQ(field.ToBigInt<4>(), expected);
}

TEST(ModulusTest, Read) {
  std::array<uint8_t, 8> data;

  {
    base::Uint8VectorBuffer buffer;
    Modulus field;
    // Should return false when it fails to read the field size.
    ASSERT_FALSE(field.Read(buffer).ok());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{3}).ok());
    buffer.set_buffer_offset(0);
    Modulus field;
    // Should return false when the field size is not a multiple of 8.
    ASSERT_FALSE(field.Read(buffer).ok());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{8}).ok());
    buffer.set_buffer_offset(0);
    Modulus field;
    // Should return false when it fails to read the field data.
    ASSERT_FALSE(field.Read(buffer).ok());
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{8}).ok());
    ASSERT_TRUE(buffer.Write(data).ok());
    buffer.set_buffer_offset(0);
    Modulus field;
    ASSERT_TRUE(field.Read(buffer).ok());
  }
}

}  // namespace rabbitsnark::circom
