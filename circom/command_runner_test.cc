#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "circom/command_runner_impl.h"
#include "circom/json/proof_reader.h"
#include "circom/json/public_reader.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/curve.h"

namespace rabbitsnark::circom {

namespace math = zkx::math;

TEST(CommandRunnerTest, CompileAndProve) {
  Options options{
      .proving_key_path = "circom/zkey/multiplier_3.zkey",
      .witness_path = "circom/wtns/multiplier_3.wtns",
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
      ReadProofFromJson<math::bn254::Curve>(options.proof_path));

  math::bn254::G1AffinePoint a{
      // clang-format off
      *math::bn254::Fq::FromDecString("21247490073426246904925106854454081689674250620397342903403266789270288729017"),
      *math::bn254::Fq::FromDecString("7141632196135129904077222084973425275118737643424218690780868770459075028477"),
      // clang-format on
  };
  math::bn254::G2AffinePoint b{
      // clang-format off
      {
        *math::bn254::Fq::FromDecString("14403118566317031836665339814656991632324332656657522862592150589339004605644"),
        *math::bn254::Fq::FromDecString("11803997679830881358036342460191493718227195076691879294802349973479368498102"),
      },
      {
        *math::bn254::Fq::FromDecString("14537849652604619505808702423754749409977944765027057989761494268256013732658"),
        *math::bn254::Fq::FromDecString("1257131106885001804229513552477099266526058636846047192698807633773843510518"),
      },
      // clang-format on
  };
  math::bn254::G1AffinePoint c{
      // clang-format off
     *math::bn254::Fq::FromDecString("6219160522774409478473402241817214405476703911884670619509939623612797986487"),
     *math::bn254::Fq::FromDecString("18804195883924525886170802694475450807340049480631634128960283159819630780536"),
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
      ReadPublicFromJson<math::bn254::Fr>(options.public_path));

  EXPECT_EQ(zkx::LiteralUtil::CreateR1<math::bn254::Fr>({60}), public_input);
}

}  // namespace rabbitsnark::circom
