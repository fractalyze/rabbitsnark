#ifndef CIRCOM_COMMAND_RUNNER_H_
#define CIRCOM_COMMAND_RUNNER_H_

#include <iostream>
#include <memory>

#include "absl/strings/substitute.h"
#include "common/command_runner_interface.h"
#include "common/profiler.h"

#include "circom/hlo/hlo_generator.h"
#include "circom/json/proof_writer.h"
#include "circom/json/public_writer.h"
#include "circom/wtns/wtns.h"
#include "circom/zkey/zkey.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"

namespace zkx::circom {

template <typename Curve>
class CommandRunnerImpl : public CommandRunnerInterface {
 public:
  absl::Status Compile(const Options& options) final {
    std::unique_ptr<ZKey<Curve>> zkey;
    RUN_WITH_PROFILE(
        "parsing zkey",
        TF_ASSIGN_OR_RETURN(zkey, ParseZKey<Curve>(options.proving_key_path)));
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

    ProvingKeyAdditionalData<Curve> pk_additional_data;
    RUN_WITH_PROFILE("loading pk additional data", {
      TF_ASSIGN_OR_RETURN(
          pk_additional_data,
          ProvingKeyAdditionalData<Curve>::ReadFromFile(options.output_dir));
    });

    int64_t l = pk_additional_data.l;
    int64_t m = pk_additional_data.m;
    int64_t n = pk_additional_data.n;

    std::vector<ScopedShapedBuffer> buffers;
    std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>> regions;
    RUN_WITH_PROFILE("sending zkey parameters", {
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.alpha_g1, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.beta_g2, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.gamma_g2, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.delta_g2, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.beta_g1, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.delta_g1, &buffers));
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

    std::vector<F> public_values;
    {
      std::unique_ptr<Wtns<F>> wtns;
      RUN_WITH_PROFILE("parsing witness", {
        TF_ASSIGN_OR_RETURN(wtns, ParseWtns<F>(options.witness_path));
        CHECK_EQ(wtns->GetVersion(), 2);
        absl::Span<const F> public_values_span =
            wtns->GetWitnesses().subspan(1, l);
        public_values.assign(public_values_span.begin(),
                             public_values_span.end());
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
    }

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
    TF_RETURN_IF_ERROR(
        WritePublicToJson<F>(public_values, options.public_path));
    std::cout << "Public values are saved to \"" << options.public_path << "\""
              << std::endl;

    return absl::OkStatus();
  }
};

}  // namespace zkx::circom

#endif  // CIRCOM_COMMAND_RUNNER_H_
