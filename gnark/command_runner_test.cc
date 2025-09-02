#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "gnark/bin/proof_reader.h"
#include "gnark/bin/public_reader.h"
#include "gnark/command_runner_impl.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/curve.h"

namespace rabbitsnark::gnark {

namespace math = zkx::math;

TEST(CommandRunnerTest, CompileAndProve) {
#if defined(ZKX_HAS_SP1)
  GTEST_SKIP() << "Skipping test for SP1 configuration";
#endif

  Options options{
      .proving_key_path = "gnark/pk/pk.dump.bin",
      .r1cs_path = "gnark/r1cs.bin",
      .witness_path = "gnark/witness/witness.bin",
      .proof_path = "proof.json",
      .public_path = "public.json",
      .output_dir = "output",
      .h_msm_window_bits = 0,
      .non_h_msm_window_bits = 0,
      .skip_hlo = false,
      .no_zk = true,
  };
  auto command_runner =
      std::make_unique<CommandRunnerImpl<math::bn254::Curve>>();
  ASSERT_TRUE(command_runner->Compile(options).ok());
  ASSERT_TRUE(command_runner->Prove(options).ok());

  TF_ASSERT_OK_AND_ASSIGN(
      zkx::Literal proof,
      ReadProofFromBin<math::bn254::Curve>(options.proof_path));

  math::bn254::G1AffinePoint a{
      // clang-format off
      *math::bn254::Fq::FromDecString("8487077006035265844709000494279066366900668751811434029808152991587221331020"),
      *math::bn254::Fq::FromDecString("833025712615174418224759151358095362475530955668744551303332741950985238346"),
      // clang-format on
  };
  math::bn254::G2AffinePoint b{
      // clang-format off
      {
        *math::bn254::Fq::FromDecString("14121243020167288872815516742017761880174098712130318271091567244308107432286"),
        *math::bn254::Fq::FromDecString("19429834152397150257656130660220431867283415626605494414624349336928711852715"),
      },
      {
        *math::bn254::Fq::FromDecString("10457246779336107254922714884596585253531799671875713266995821809369809850270"),
        *math::bn254::Fq::FromDecString("1718082702356866272105142323779144408195643082335293924182829651335905843586"),
      },
      // clang-format on
  };
  math::bn254::G1AffinePoint c{
      // clang-format off
     *math::bn254::Fq::FromDecString("13629451268421506883023176702600727507099526822748134477005282133434151092601"),
     *math::bn254::Fq::FromDecString("17414423152612856072346110917934701189758548974343287215796587046476314547605"),
      // clang-format on
  };

  std::vector<zkx::Literal> expected_proof_elements;
  expected_proof_elements.push_back(
      zkx::LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(a));
  expected_proof_elements.push_back(
      zkx::LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(b));
  expected_proof_elements.push_back(
      zkx::LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(c));

  zkx::Literal expected_proof =
      zkx::LiteralUtil::MakeTupleOwned(std::move(expected_proof_elements));

  EXPECT_EQ(expected_proof, proof);

  TF_ASSERT_OK_AND_ASSIGN(
      zkx::Literal public_input,
      ReadPublicFromBin<math::bn254::Fr>(options.public_path));

  EXPECT_EQ(zkx::LiteralUtil::CreateR1<math::bn254::Fr>({35}), public_input);
}

}  // namespace rabbitsnark::gnark
