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
