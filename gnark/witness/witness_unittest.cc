#include "gnark/witness/witness.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

namespace rabbitsnark::gnark {

namespace math = zkx::math;

using F = math::bn254::Fr;

TEST(WitnessTest, Read) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Witness<F>> witness,
                          ParseWitness<F>("gnark/witness/witness.bin"));

  Witness<F> expected_witness = {
      1,
      1,
      {F(35), F(3)},
  };
  EXPECT_EQ(*witness, expected_witness);
}

}  // namespace rabbitsnark::gnark
