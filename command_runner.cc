#include "command_runner.h"

#include <iostream>

#include "absl/log/globals.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"

#include "circom/hlo/hlo_generator.h"
#include "circom/json/proof_writer.h"
#include "circom/json/public_writer.h"
#include "circom/wtns/wtns.h"
#include "circom/zkey/zkey.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/flag/flag_parser.h"
#include "zkx/base/flag/flag_value_traits.h"
#include "zkx/base/flag/numeric_flags.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/literal_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/curve.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"

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

namespace zkx {
namespace circom {
namespace {

enum class Curve {
  kBn254,
};

struct Options {
  std::string zkey_path;
  std::string wtns_path;
  std::string proof_path;
  std::string public_path;
  std::string output_dir;
  int32_t h_msm_window_bits;
  int32_t non_h_msm_window_bits;
  bool skip_hlo;
  bool no_zk;
};

class CommandRunnerInterface {
 public:
  CommandRunnerInterface()
      : runner_(PlatformUtil::GetPlatform("cpu").value()) {}
  virtual ~CommandRunnerInterface() = default;

  virtual absl::Status Compile(const Options& options) = 0;
  virtual absl::Status Prove(const Options& options) = 0;

 protected:
  HloRunner runner_;
};

template <typename Curve>
class CommandRunnerImpl : public CommandRunnerInterface {
 public:
  absl::Status Compile(const Options& options) final {
    std::unique_ptr<ZKey<Curve>> zkey;
    RUN_WITH_PROFILE(
        "parsing zkey",
        TF_ASSIGN_OR_RETURN(zkey,
                            ParseZKey<Curve>(options.zkey_path,
                                             /*process_coefficients=*/true)));
    const v1::ZKey<Curve>* v1_zkey = zkey->ToV1();
    const v1::ZKeyHeaderGrothSection<Curve>& header = v1_zkey->header_groth;

    if (options.h_msm_window_bits > header.domain_size) {
      return absl::InvalidArgumentError(absl::Substitute(
          "MSM window size for h ($0) is larger than domain size ($1)",
          options.h_msm_window_bits, header.domain_size));
    }
    if (options.non_h_msm_window_bits > header.num_vars) {
      return absl::InvalidArgumentError(absl::Substitute(
          "MSM window size for non-h ($0) is larger than num vars ($1)",
          options.non_h_msm_window_bits, header.num_vars));
    }

    std::unique_ptr<HloModule> module;
    if (options.skip_hlo) {
      RUN_WITH_PROFILE("loading hlo", {
        std::string hlo_string;
        TF_RETURN_IF_ERROR(tsl::ReadFileToString(
            tsl::Env::Default(),
            tsl::io::JoinPath(options.output_dir, "groth16.hlo"), &hlo_string));
        TF_ASSIGN_OR_RETURN(module, ParseAndReturnUnverifiedModule(hlo_string));
      });
    } else {
      RUN_WITH_PROFILE("generating hlo", {
        TF_ASSIGN_OR_RETURN(
            std::string hlo_string,
            GenerateHLO<Curve>(*zkey.get(), options.h_msm_window_bits,
                               options.non_h_msm_window_bits,
                               options.output_dir));
        TF_ASSIGN_OR_RETURN(module, ParseAndReturnUnverifiedModule(hlo_string));
      });
    }

    module->mutable_config().mutable_debug_options().set_zkx_obj_file_dir(
        options.output_dir);

    std::unique_ptr<OpaqueExecutable> opaque_executable;
    RUN_WITH_PROFILE("compiling", {
      TF_ASSIGN_OR_RETURN(opaque_executable,
                          runner_.CreateExecutable(std::move(module),
                                                   /*run_hlo_passes=*/false));
    });

    RUN_WITH_PROFILE("storing executable", {
      TF_ASSIGN_OR_RETURN(
          Executable * executable,
          runner_.ExecutableFromWrapped(opaque_executable.get()));

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<AotCompilationResult> exported_aot_result,
          runner_.backend().compiler()->Export(executable));

      TF_ASSIGN_OR_RETURN(std::string aot_result_string,
                          exported_aot_result->SerializeAsString());

      TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
          tsl::Env::Default(),
          tsl::io::JoinPath(options.output_dir, "groth16.bin"),
          aot_result_string));
    });

    return absl::OkStatus();
  }

  absl::Status Prove(const Options& options) final {
    using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
    using F = typename G1AffinePoint::ScalarField;

    std::unique_ptr<OpaqueExecutable> opaque_executable;
    RUN_WITH_PROFILE("loading executable", {
      std::string aot_result_string;
      TF_RETURN_IF_ERROR(tsl::ReadFileToString(
          tsl::Env::Default(),
          tsl::io::JoinPath(options.output_dir, "groth16.bin"),
          &aot_result_string));

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<AotCompilationResult> aot_result,
          runner_.backend().compiler()->LoadAotCompilationResult(
              aot_result_string));

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Executable> executable,
          aot_result->LoadExecutable(runner_.backend().compiler(),
                                     /*executor=*/nullptr));

      opaque_executable = runner_.WrapExecutable(std::move(executable));
    });

    std::unique_ptr<ZKey<Curve>> zkey;
    RUN_WITH_PROFILE(
        "parsing zkey",
        TF_ASSIGN_OR_RETURN(zkey,
                            ParseZKey<Curve>(options.zkey_path,
                                             /*process_coefficients=*/false)));
    const ProvingKey<Curve>& pk = zkey->GetProvingKey();
    const v1::ZKey<Curve>* v1_zkey = zkey->ToV1();
    const v1::ZKeyHeaderGrothSection<Curve>& header = v1_zkey->header_groth;
    int64_t l = header.num_public_inputs;
    int64_t m = header.num_vars;
    int64_t n = header.domain_size;

    std::unique_ptr<Wtns<F>> wtns;
    RUN_WITH_PROFILE("parsing witness", {
      TF_ASSIGN_OR_RETURN(wtns, ParseWtns<F>(options.wtns_path));
      CHECK_EQ(wtns->GetVersion(), 2);
    });

    std::vector<ScopedShapedBuffer> buffers;
    std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>> regions;
    RUN_WITH_PROFILE("sending zkey parameters", {
      TF_RETURN_IF_ERROR(
          AddScalarParameter(*pk.verifying_key.alpha_g1, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(*pk.verifying_key.beta_g2, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(*pk.verifying_key.gamma_g2, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(*pk.verifying_key.delta_g2, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(*pk.verifying_key.beta_g1, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(*pk.verifying_key.delta_g1, &buffers));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "a_g1_query.bin", ShapeUtil::MakeShape(BN254_G1_AFFINE, {m}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "b_g1_query.bin", ShapeUtil::MakeShape(BN254_G1_AFFINE, {m}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "b_g2_query.bin", ShapeUtil::MakeShape(BN254_G2_AFFINE, {m}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "l_g1_query.bin",
          ShapeUtil::MakeShape(BN254_G1_AFFINE, {m - l - 1}), &buffers,
          &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "h_g1_query.bin", ShapeUtil::MakeShape(BN254_G1_AFFINE, {n}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddSparseMatrixParameterFromFile(options, "a.bin",
                                                          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddSparseMatrixParameterFromFile(options, "b.bin",
                                                          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "twiddles.bin", ShapeUtil::MakeShape(BN254_SCALAR, {n}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "fft_twiddles.bin", ShapeUtil::MakeShape(BN254_SCALAR, {n}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "ifft_twiddles.bin", ShapeUtil::MakeShape(BN254_SCALAR, {n}),
          &buffers, &regions));
    });
    RUN_WITH_PROFILE("sending witness parameters", {
      TF_RETURN_IF_ERROR(AddVectorParameter(wtns->GetWitnesses(), &buffers));
      if (options.no_zk) {
        TF_RETURN_IF_ERROR(AddScalarParameter(F(0), &buffers));
        TF_RETURN_IF_ERROR(AddScalarParameter(F(0), &buffers));
      } else {
        TF_RETURN_IF_ERROR(AddScalarParameter(F::Random(), &buffers));
        TF_RETURN_IF_ERROR(AddScalarParameter(F::Random(), &buffers));
      }
    });

    Literal proof;
    RUN_WITH_PROFILE("generating proof", {
      TF_ASSIGN_OR_RETURN(
          ExecutionOutput output,
          runner_.ExecuteWithDeviceBuffers(opaque_executable.get(), buffers,
                                           /*profile=*/nullptr));

      TF_ASSIGN_OR_RETURN(proof,
                          runner_.TransferLiteralFromDevice(output.Result()));
    });

    std::cout << proof.ToString() << std::endl;
    TF_RETURN_IF_ERROR(WriteProofToJson<Curve>(proof, options.proof_path));
    std::cout << "Proof is saved to \"" << options.proof_path << "\""
              << std::endl;
    absl::Span<const F> public_values = wtns->GetWitnesses().subspan(1, l);
    TF_RETURN_IF_ERROR(
        WritePublicToJson<F>(public_values, options.public_path));
    std::cout << "Public values are saved to \"" << options.public_path << "\""
              << std::endl;

    return absl::OkStatus();
  }

 private:
  template <typename T>
  absl::Status AddScalarParameter(const T& value,
                                  std::vector<ScopedShapedBuffer>* buffers) {
    BorrowingLiteral literal(
        reinterpret_cast<const char*>(&value),
        ShapeUtil::MakeScalarShape(primitive_util::NativeToPrimitiveType<T>()));
    TF_ASSIGN_OR_RETURN(
        ScopedShapedBuffer shaped_buffer,
        runner_.TransferLiteralToDevice(literal, buffers->size()));
    buffers->push_back(std::move(shaped_buffer));
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status AddVectorParameter(const absl::Span<const T> values,
                                  std::vector<ScopedShapedBuffer>* buffers) {
    BorrowingLiteral literal(
        reinterpret_cast<const char*>(values.data()),
        ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<T>(),
                             {static_cast<int64_t>(values.size())}));
    TF_ASSIGN_OR_RETURN(
        ScopedShapedBuffer shaped_buffer,
        runner_.TransferLiteralToDevice(literal, buffers->size()));
    buffers->push_back(std::move(shaped_buffer));
    return absl::OkStatus();
  }

  absl::Status AddSparseMatrixParameterFromFile(
      const Options& options, std::string_view fname,
      std::vector<ScopedShapedBuffer>* buffers,
      std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions) {
    std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
        tsl::io::JoinPath(options.output_dir, fname), &region));
    ShapedBuffer shaped_buffer(
        ShapeUtil::MakeShape(U8, {static_cast<int64_t>(region->length())}), 0);
    shaped_buffer.set_buffer(
        se::DeviceMemoryBase(const_cast<void*>(region->data()),
                             region->length()),
        ShapeIndex{});
    buffers->push_back(ScopedShapedBuffer(std::move(shaped_buffer), nullptr));
    regions->push_back(std::move(region));
    return absl::OkStatus();
  }

  absl::Status AddVectorParameterFromFile(
      const Options& options, std::string_view fname, const Shape& shape,
      std::vector<ScopedShapedBuffer>* buffers,
      std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>>* regions) {
    std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
        tsl::io::JoinPath(options.output_dir, fname), &region));
    ShapedBuffer shaped_buffer(shape, 0);
    shaped_buffer.set_buffer(
        se::DeviceMemoryBase(const_cast<void*>(region->data()),
                             region->length()),
        ShapeIndex{});
    buffers->push_back(ScopedShapedBuffer(std::move(shaped_buffer), nullptr));
    regions->push_back(std::move(region));
    return absl::OkStatus();
  }
};

}  // namespace

absl::Status CommandRunner::Run(int argc, char** argv) {
  Options options;
  Curve curve;
  int vlog_level;

  base::FlagParser parser;

  base::SubParser& compile_parser =
      parser.AddSubParser().set_name("compile").set_help(
          "Compile the Groth16 prover");
  compile_parser.AddFlag<base::StringFlag>(&options.zkey_path)
      .set_name("zkey")
      .set_help("Path to the .zkey file (Groth16 proving key)");
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

  base::SubParser& prove_parser =
      parser.AddSubParser().set_name("prove").set_help(
          "Generate a proof using compiled Groth16 prover");
  prove_parser.AddFlag<base::StringFlag>(&options.zkey_path)
      .set_name("zkey")
      .set_help("Path to the .zkey file (Groth16 proving key)");
  prove_parser.AddFlag<base::StringFlag>(&options.wtns_path)
      .set_name("wtns")
      .set_help("Path to the witness (.wtns) file");
  prove_parser.AddFlag<base::StringFlag>(&options.proof_path)
      .set_name("proof")
      .set_help("Output path for proof (.json)");
  prove_parser.AddFlag<base::StringFlag>(&options.public_path)
      .set_name("public")
      .set_help("Output path for public inputs (.json)");
  prove_parser.AddFlag<base::StringFlag>(&options.output_dir)
      .set_name("output")
      .set_help("Directory containing compiled prover files");
  prove_parser.AddFlag<base::BoolFlag>(&options.no_zk)
      .set_long_name("--no_zk")
      .set_default_value(false)
      .set_help(
          "Disable zero-knowledge (for debugging or comparison with "
          "RapidSnark). Disabled by default.");

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

  if (vlog_level > 0) {
    std::cout << "Setting vlog level to " << vlog_level << std::endl;
    absl::SetGlobalVLogLevel(3);
  }

  std::unique_ptr<CommandRunnerInterface> interface;
  if (curve == Curve::kBn254) {
    interface.reset(new CommandRunnerImpl<math::bn254::Curve>());
  } else {
    return absl::InternalError("Invalid curve");
  }
  if (compile_parser.is_set()) {
    return interface->Compile(options);
  } else if (prove_parser.is_set()) {
    return interface->Prove(options);
  }

  return absl::InternalError("No subcommand is set");
}

}  // namespace circom

namespace base {

template <>
class FlagValueTraits<circom::Curve> {
 public:
  static absl::Status ParseValue(std::string_view input, circom::Curve* value) {
    if (input == "bn254") {
      *value = circom::Curve::kBn254;
    } else {
      return absl::NotFoundError(absl::Substitute("Unknown curve: $0", input));
    }
    return absl::OkStatus();
  }
};

}  // namespace base
}  // namespace zkx
