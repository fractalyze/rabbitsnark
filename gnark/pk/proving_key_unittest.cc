#include "gnark/pk/proving_key.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/curve.h"

namespace rabbitsnark::gnark {

namespace {

namespace base = zkx::base;
namespace math = zkx::math;

using Curve = math::bn254::Curve;
using G1AffinePoint = math::bn254::G1AffinePoint;
using G2AffinePoint = math::bn254::G2AffinePoint;
using F = math::bn254::Fr;

G1AffinePoint ToG1AffinePoint(std::string_view g1[2]) {
  math::bn254::Fq x = *math::bn254::Fq::FromDecString(g1[0]);
  math::bn254::Fq y = *math::bn254::Fq::FromDecString(g1[1]);
  return {x, y};
}

G2AffinePoint ToG2AffinePoint(std::string_view g2[2][2]) {
  math::bn254::Fq2 x({*math::bn254::Fq::FromDecString(g2[0][0]),
                      *math::bn254::Fq::FromDecString(g2[0][1])});
  math::bn254::Fq2 y({*math::bn254::Fq::FromDecString(g2[1][0]),
                      *math::bn254::Fq::FromDecString(g2[1][1])});
  return {x, y};
}

void RunTest(const std::string& path, SerdeMode mode) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProvingKey<Curve>> proving_key,
                          ParseProvingKey<Curve>(path, mode));

  // clang-format off
  std::string_view generator_str = "21888242871839275217838484774961031246007050428528088939761107053157389710902";
  // clang-format on
  F generator = *F::FromDecString(generator_str);
  F fr_multiplicative_gen = F(5);
  Domain<F> expected_domain = {
      4,
      *F(4).Inverse(),
      generator,
      *generator.Inverse(),
      fr_multiplicative_gen,
      *fr_multiplicative_gen.Inverse(),
      true,
  };
  EXPECT_EQ(proving_key->domain, expected_domain);

  // clang-format off
    std::string_view alpha_g1_str[2] = {
      "13472183060820836120186349319033151220759998775268853376119728413291875060539",
      "12164076311350139092950436896488212436297629742296742179529087974040222607479",
    };
    std::string_view beta_g1_str[2] = {
      "19950615703050501461391390148340494832712581435520827264410271282453431900693",
      "513036010311580189545705789509334501426802096225273230345328397567650625477",
    };
    std::string_view delta_g1_str[2] = {
      "3781571481935863675877416462655006291292309382829331220737258497314372703611",
      "2175227156216158545756833969768422406773036425831499476057077205126233827808",
    };
    std::string_view a_g1_query_str[3][2] = {
      {"8324883888492961397449334820490173984281119765208883931574665812159102318814",
       "8340361088987478363182465103992470916622925417179216037709366069359501048537"},
      {"12907893015510682249398010859296222468796965024514555606724276272289685103440",
       "10176829227333798751289268839365339721592869073579074946416700807745061678740"},
      {"776779542602599091932813340194322952628002021930359841643463724318699203736",
       "16759720519432866316412291731801392875982827027373069146402931489028798775559"},
    };
    std::string_view b_g1_query_str[2][2] = {
      {"8324883888492961397449334820490173984281119765208883931574665812159102318814",
       "8340361088987478363182465103992470916622925417179216037709366069359501048537"},
      {"432102568239586304337873356065116141552743123615431063970553259557204108086",
       "16925068336302232899640683760512094209167664894097238125764803086122282728545"},
    };
    std::string_view z_g1_query_str[3][2] = {
      {"11293465795324669809630581381063806729515215238003649868798068571764792032188",
       "10406130680914782032365512358746982865832482904226570195626565216270282716752"},
      {"20718445704625739746283511560035048278429046402500685944498015444135894805347",
       "13408668924742530191461998759325163685674584279362366130677325532516169851357"},
      {"6768735703945250750947333703307801010402842403313349141943013597294502652158",
       "19903562130080485416027942613999880619147880747053010862816750062295062188037"},
    };
    std::string_view k_g1_query_str[3][2] = {
      {"20891106894778710087550738377057848343648992636195968517700537139472860917910",
       "14499136278091248099481739391056580957155112370342524978224112102605944375035"},
      {"3934560238881716621519202542124981273345124356369433169790035400327124306140",
       "14755320572776837086691733624928060539856449887252272148330195029150074072178"},
      {"11013848781757217685817155394074495209460615905305169996083181600439314721382",
       "2582984757625386491703940567441797266016291475061414047110250074258201300236"},
    };
    std::string_view beta_g2_str[2][2] = {
      {"2823219271202837980648681937708959351012480241364712947166452067675659797814",
       "8085425370901111426285102389426113925798329382711895299380758772289750210201"},
      {"3129686503139256931926270545768094932192459797957919793395456612437503973763",
       "4172255987100985225501073180397806446757437700536162321177003897115580929397"},
    };
    std::string_view delta_g2_str[2][2] = {
      {"11634081626372338095242641112283278855456441796889116422888255372905288865199",
       "1167942084692628941746379436121233400051984080964469605532026571147948833630"},
      {"5359681060217220503189251675128736061735735422017792461783827663836241938117",
       "16191013688097276697284858114069824657809851522714148900759366182453227031838"},
    };
    std::string_view b_g2_query_str[2][2][2] = {
      {
        {"6691977537254291414709727412157616102927988952721163630399986445408393638033",
         "6640771707077301544499137449224400463357751698751005003530609063951341260856"},
        {"14028902812675800514949725995390979029235305705402944703296063830575875172027",
         "8568763855624591427660946066494556884374012678383987768910669843285111199347"},
      },
      {
        {"20119418993248860640919130124067134759871283701659035003144958695247912355117",
         "20563433727742184188458188598977361100298650450890538642710836548912467352679"},
        {"15507056505800068863833498545224101662698688858447337908563097412319686289896",
         "4623320254460292410634198068029570729576377410193409968279938526895029211530"},
      },
    };
    std::vector<bool> infinity_a = {
      false,
      true,
      false,
      false,
      true,
    };
    std::vector<bool> infinity_b = {
      true,
      false,
      false,
      true,
      true,
    };
  // clang-format on

  G1AffinePoint alpha_g1 = ToG1AffinePoint(alpha_g1_str);
  G1AffinePoint beta_g1 = ToG1AffinePoint(beta_g1_str);
  G1AffinePoint delta_g1 = ToG1AffinePoint(delta_g1_str);
  std::vector<G1AffinePoint> a_g1_query =
      base::Map(a_g1_query_str,
                [](std::string_view g1[2]) { return ToG1AffinePoint(g1); });
  std::vector<G1AffinePoint> b_g1_query =
      base::Map(b_g1_query_str,
                [](std::string_view g1[2]) { return ToG1AffinePoint(g1); });
  std::vector<G1AffinePoint> z_g1_query =
      base::Map(z_g1_query_str,
                [](std::string_view g1[2]) { return ToG1AffinePoint(g1); });
  std::vector<G1AffinePoint> k_g1_query =
      base::Map(k_g1_query_str,
                [](std::string_view g1[2]) { return ToG1AffinePoint(g1); });
  G2AffinePoint beta_g2 = ToG2AffinePoint(beta_g2_str);
  G2AffinePoint delta_g2 = ToG2AffinePoint(delta_g2_str);
  std::vector<G2AffinePoint> b_g2_query =
      base::Map(b_g2_query_str,
                [](std::string_view g2[2][2]) { return ToG2AffinePoint(g2); });

  EXPECT_EQ(proving_key->alpha_g1, alpha_g1);
  EXPECT_EQ(proving_key->beta_g1, beta_g1);
  EXPECT_EQ(proving_key->delta_g1, delta_g1);
  EXPECT_EQ(proving_key->a_g1_query, a_g1_query);
  EXPECT_EQ(proving_key->b_g1_query, b_g1_query);
  EXPECT_EQ(proving_key->z_g1_query, z_g1_query);
  EXPECT_EQ(proving_key->k_g1_query, k_g1_query);
  EXPECT_EQ(proving_key->beta_g2, beta_g2);
  EXPECT_EQ(proving_key->delta_g2, delta_g2);
  EXPECT_EQ(proving_key->b_g2_query, b_g2_query);
  EXPECT_EQ(proving_key->infinity_a, infinity_a);
  EXPECT_EQ(proving_key->infinity_b, infinity_b);
  EXPECT_EQ(proving_key->num_infinity_a, 2);
  EXPECT_EQ(proving_key->num_infinity_b, 3);
  EXPECT_TRUE(proving_key->commitment_keys.empty());
}

}  // namespace

TEST(ProvingKeyTest, Read) { RunTest("gnark/pk/pk.bin", SerdeMode::kDefault); }

TEST(ProvingKeyTest, ReadRaw) {
  RunTest("gnark/pk/pk.raw.bin", SerdeMode::kRaw);
}

TEST(ProvingKeyTest, ReadDump) {
  RunTest("gnark/pk/pk.dump.bin", SerdeMode::kDump);
}

}  // namespace rabbitsnark::gnark
