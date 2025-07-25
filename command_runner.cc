#include "command_runner.h"

#include <iostream>

#include "absl/log/globals.h"
#include "absl/strings/substitute.h"

#include "circom/command_runner_impl.h"
#include "gnark/command_runner_impl.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/flag/flag_parser.h"
#include "zkx/base/flag/flag_value_traits.h"
#include "zkx/base/flag/numeric_flags.h"
#include "zkx/math/elliptic_curves/bn/bn254/curve.h"
// clang-format off
#include "version_generated.h"
// clang-format on

namespace zkx {

enum class Curve {
  kBn254,
};

namespace base {

template <>
class FlagValueTraits<Curve> {
 public:
  static absl::Status ParseValue(std::string_view input, Curve* value) {
    if (input == "bn254") {
      *value = Curve::kBn254;
    } else {
      return absl::NotFoundError(absl::Substitute("Unknown curve: $0", input));
    }
    return absl::OkStatus();
  }
};

}  // namespace base

namespace {

void AddCompileCommand(base::SubParser& parser, Options& options) {
  base::SubParser& compile_parser =
      parser.AddSubParser().set_name("compile").set_help(
          "Compile the Groth16 prover");
  compile_parser.AddFlag<base::StringFlag>(&options.proving_key_path)
      .set_name("proving_key")
      .set_help(
          "Path to the Groth16 proving key (`.zkey` for circom, `.bin` for "
          "gnark)");
  compile_parser.AddFlag<base::StringFlag>(&options.output_dir)
      .set_name("output")
      .set_help("Directory to store compiled prover output");
  compile_parser.AddFlag<base::IntFlag>(&options.h_msm_window_bits)
      .set_long_name("--h_msm_window_bits")
      .set_default_value(0)
      .set_help(
          "Window size (in bits) for h MSM (Multi-Scalar Multiplication). "
          "If set to 0, it will be estimated automatically using a cost model "
          "based on Pippenger’s algorithm. See: "
          // clang-format off
            "https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#total-complexity "
          // clang-format on
          "(Default: 0)");
  compile_parser.AddFlag<base::IntFlag>(&options.non_h_msm_window_bits)
      .set_long_name("--non_h_msm_window_bits")
      .set_default_value(0)
      .set_help(
          "Window size (in bits) for non h MSM (Multi-Scalar Multiplication). "
          "If set to 0, it will be estimated automatically using a cost model "
          "based on Pippenger’s algorithm. See: "
          // clang-format off
            "https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm#total-complexity "
          // clang-format on
          "(Default: 0)");
  compile_parser.AddFlag<base::BoolFlag>(&options.skip_hlo)
      .set_long_name("--skip_hlo")
      .set_default_value(false)
      .set_help(
          "Skip HLO generation from zkey (including sparse matrix processing). "
          "Intended for compiler developers to speed up iterative compilation "
          "tests when HLO has already been generated once.");
}

void AddProveCommand(base::SubParser& parser, Options& options) {
  base::SubParser& prove_parser =
      parser.AddSubParser().set_name("prove").set_help(
          "Generate a proof using compiled Groth16 prover");
  if (parser.name() == "gnark") {
    prove_parser.AddFlag<base::StringFlag>(&options.proving_key_path)
        .set_name("proving_key")
        .set_help("Path to the Groth16 proving key (`.bin`)");
    prove_parser.AddFlag<base::StringFlag>(&options.r1cs_path)
        .set_name("r1cs")
        .set_help("Path to the R1CS (`.bin`)");
  }
  prove_parser.AddFlag<base::StringFlag>(&options.witness_path)
      .set_name("witness")
      .set_help("Path to the witness (`.wtns` for circom)");
  prove_parser.AddFlag<base::StringFlag>(&options.proof_path)
      .set_name("proof")
      .set_help("Output path for proof (`.json` for circom, `.bin` for gnark)");
  prove_parser.AddFlag<base::StringFlag>(&options.public_path)
      .set_name("public")
      .set_help(
          "Output path for public inputs (`.json` for circom, `.bin` for "
          "gnark)");
  prove_parser.AddFlag<base::StringFlag>(&options.output_dir)
      .set_name("output")
      .set_help("Directory containing compiled prover files");
  prove_parser.AddFlag<base::BoolFlag>(&options.no_zk)
      .set_long_name("--no_zk")
      .set_default_value(false)
      .set_help(
          "Disable zero-knowledge (for debugging or comparison with "
          "RapidSnark). Disabled by default.");
}

}  // namespace

absl::Status CommandRunner::Run(int argc, char** argv) {
  Options options;
  bool full_version;
  Curve curve;
  int vlog_level;

  base::FlagParser parser;

  base::SubParser& gnark_parser = parser.AddSubParser().set_name("gnark");

  AddCompileCommand(gnark_parser, options);
  AddProveCommand(gnark_parser, options);

  base::SubParser& circom_parser = parser.AddSubParser().set_name("circom");

  AddCompileCommand(circom_parser, options);
  AddProveCommand(circom_parser, options);

  base::SubParser& version_parser =
      parser.AddSubParser().set_name("version").set_help("Print version");
  version_parser.AddFlag<base::BoolFlag>(&full_version)
      .set_long_name("--full")
      .set_default_value(false)
      .set_help("Print full version");

  parser.AddFlag<base::Flag<Curve>>(&curve)
      .set_long_name("--curve")
      .set_default_value(Curve::kBn254)
      .set_help("Elliptic curve to use (available: bn254). Default: bn254");
  parser.AddFlag<base::IntFlag>(&vlog_level, &base::ParsePositiveValue<int>)
      .set_short_name("-v")
      .set_long_name("--vlog_level")
      .set_default_value(0)
      .set_help("Logging verbosity level (default: 0)");

  TF_RETURN_IF_ERROR(parser.Parse(argc, argv));

  if (version_parser.is_set()) {
    std::cout << "RabbitSnark version: "
              << (full_version ? RABBIT_SNARK_VERSION_FULL_STR
                               : RABBIT_SNARK_VERSION_STR)
              << std::endl;
    return absl::OkStatus();
  }

  if (vlog_level > 0) {
    std::cout << "Setting vlog level to " << vlog_level << std::endl;
    absl::SetGlobalVLogLevel(3);
  }

  std::unique_ptr<CommandRunnerInterface> interface;
  auto handle_parser =
      [&](auto& parser,
          std::function<CommandRunnerInterface*()> make_impl) -> absl::Status {
    if (!parser.is_set()) return absl::OkStatus();
    if (curve == Curve::kBn254) {
      interface.reset(make_impl());
    } else {
      return absl::InternalError("Invalid curve");
    }
    for (const auto& flag : *parser.flags()) {
      if (flag->is_set()) {
        if (flag->name() == "compile") {
          return interface->Compile(options);
        } else if (flag->name() == "prove") {
          return interface->Prove(options);
        }
      }
    }
    return absl::InternalError("No subcommand is set");
  };

  absl::Status status = handle_parser(circom_parser, []() {
    return new circom::CommandRunnerImpl<math::bn254::Curve>();
  });
  if (!status.ok()) return status;

  status = handle_parser(gnark_parser, []() {
    return new gnark::CommandRunnerImpl<math::bn254::Curve>();
  });
  if (!status.ok()) return status;

  return absl::OkStatus();
}

}  // namespace zkx
