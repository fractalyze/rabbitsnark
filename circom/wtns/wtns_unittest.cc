#include "circom/wtns/wtns.h"

#include <vector>

#include "gtest/gtest.h"

#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

namespace rabbitsnark::circom {

namespace math = zkx::math;

using F = math::bn254::Fr;

TEST(WtnsTest, Parse) {
  // Generated with { "in": ["3", "4", "5"] }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Wtns<F>> wtns,
                          ParseWtns<F>("circom/wtns/multiplier_3.wtns"));
  ASSERT_EQ(wtns->GetVersion(), 2);

  std::array<uint8_t, 32> bytes = math::bn254::FrConfig::kModulus.ToBytesLE();
  v2::WtnsHeaderSection expected_header = {
      Modulus{std::vector<uint8_t>(bytes.begin(), bytes.end())},
      6,
  };
  EXPECT_EQ(wtns->ToV2()->header, expected_header);

  std::vector<F> expected_witnesses = {
      F::FromUnchecked(1), F::FromUnchecked(60), F::FromUnchecked(3),
      F::FromUnchecked(4), F::FromUnchecked(5),  F::FromUnchecked(12),
  };
  v2::WtnsDataSection<F> expected_data{absl::MakeConstSpan(expected_witnesses)};
  EXPECT_EQ(wtns->ToV2()->data, expected_data);
}

}  // namespace rabbitsnark::circom
