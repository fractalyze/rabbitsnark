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

#ifndef GNARK_COMMAND_RUNNER_IMPL_H_
#define GNARK_COMMAND_RUNNER_IMPL_H_

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "common/command_runner_interface.h"
#include "common/profiler.h"
#include "gnark/bin/proof_writer.h"
#include "gnark/bin/public_writer.h"
#include "gnark/hlo/hlo_generator.h"
#include "gnark/pk/proving_key.h"
#include "gnark/witness/witness.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"

#if defined(ZKX_HAS_SP1)
#include "gnark/libsp1.h"
#else
#include "gnark_go/gnark.h"
#endif

namespace rabbitsnark::gnark {

template <typename Curve>
class CommandRunnerImpl : public CommandRunnerInterface {
 public:
  absl::Status Compile(const Options& options) final {
    std::unique_ptr<ProvingKey<Curve>> proving_key;
    RUN_WITH_PROFILE("loading proving key", {
      TF_ASSIGN_OR_RETURN(
          proving_key,
          ParseProvingKey<Curve>(options.proving_key_path, SerdeMode::kDump));
    });

    if (options.h_msm_window_bits > proving_key->domain.cardinality) {
      return absl::InvalidArgumentError(absl::Substitute(
          "MSM window size for h ($0) is larger than domain size ($1)",
          options.h_msm_window_bits, proving_key->domain.cardinality));
    }
    if (options.non_h_msm_window_bits > proving_key->infinity_a.size()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "MSM window size for non-h ($0) is larger than num vars ($1)",
          options.non_h_msm_window_bits, proving_key->infinity_a.size()));
    }

    std::unique_ptr<zkx::HloModule> module;
    if (options.skip_hlo) {
      RUN_WITH_PROFILE("loading hlo", {
        std::string hlo_string;
        TF_RETURN_IF_ERROR(tsl::ReadFileToString(
            tsl::Env::Default(),
            tsl::io::JoinPath(options.output_dir, "groth16.hlo"), &hlo_string));
        TF_ASSIGN_OR_RETURN(module,
                            zkx::ParseAndReturnUnverifiedModule(hlo_string));
      });
    } else {
      RUN_WITH_PROFILE("generating hlo", {
        TF_ASSIGN_OR_RETURN(
            std::string hlo_string,
            gnark::GenerateHLO<Curve>(*proving_key, options.h_msm_window_bits,
                                      options.non_h_msm_window_bits,
                                      options.output_dir));
        TF_ASSIGN_OR_RETURN(module,
                            zkx::ParseAndReturnUnverifiedModule(hlo_string));
      });
    }

    std::unique_ptr<zkx::OpaqueExecutable> opaque_executable;
    RUN_WITH_PROFILE("compiling", {
      TF_ASSIGN_OR_RETURN(opaque_executable,
                          runner_.CreateExecutable(std::move(module),
                                                   /*run_hlo_passes=*/false));
    });

    RUN_WITH_PROFILE("storing executable", {
      TF_ASSIGN_OR_RETURN(
          zkx::Executable * executable,
          runner_.ExecutableFromWrapped(opaque_executable.get()));

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<zkx::AotCompilationResult> exported_aot_result,
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

    std::unique_ptr<zkx::OpaqueExecutable> opaque_executable;
    RUN_WITH_PROFILE("loading executable", {
      std::string aot_result_string;
      TF_RETURN_IF_ERROR(tsl::ReadFileToString(
          tsl::Env::Default(),
          tsl::io::JoinPath(options.output_dir, "groth16.bin"),
          &aot_result_string));

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<zkx::AotCompilationResult> aot_result,
          runner_.backend().compiler()->LoadAotCompilationResult(
              aot_result_string));

      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<zkx::Executable> executable,
          aot_result->LoadExecutable(runner_.backend().compiler(),
                                     /*executor=*/nullptr));

      opaque_executable = runner_.WrapExecutable(std::move(executable));
    });

    GnarkProvingKeyAdditionalData<Curve> pk_additional_data;
    RUN_WITH_PROFILE("loading pk additional data", {
      TF_ASSIGN_OR_RETURN(pk_additional_data,
                          GnarkProvingKeyAdditionalData<Curve>::ReadFromFile(
                              options.output_dir));
    });

    int64_t l = pk_additional_data.l;
    int64_t m = pk_additional_data.m;
    int64_t n = pk_additional_data.n;

    std::vector<zkx::ScopedShapedBuffer> buffers;
    std::vector<std::unique_ptr<tsl::ReadOnlyMemoryRegion>> regions;
    RUN_WITH_PROFILE("sending pk parameters", {
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.alpha_g1, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.beta_g1, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.beta_g2, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.delta_g1, &buffers));
      TF_RETURN_IF_ERROR(
          AddScalarParameter(pk_additional_data.delta_g2, &buffers));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "a_g1_query.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_G1_AFFINE,
                                    {pk_additional_data.a_g1_query_size}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "b_g1_query.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_G1_AFFINE,
                                    {pk_additional_data.b_g1_query_size}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "b_g2_query.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_G2_AFFINE,
                                    {pk_additional_data.b_g1_query_size}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "z_g1_query.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_G1_AFFINE, {n - 1}), &buffers,
          &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "k_g1_query.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_G1_AFFINE, {m - l - 1}),
          &buffers, &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "coset_twiddles.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_SCALAR, {n}), &buffers,
          &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "coset_inv_twiddles.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_SCALAR, {n}), &buffers,
          &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "fft_twiddles.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_SCALAR, {n}), &buffers,
          &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "ifft_twiddles.bin",
          zkx::ShapeUtil::MakeShape(zkx::BN254_SCALAR, {n}), &buffers,
          &regions));
      TF_RETURN_IF_ERROR(AddVectorParameterFromFile(
          options, "den.bin", zkx::ShapeUtil::MakeShape(zkx::BN254_SCALAR, {n}),
          &buffers, &regions));
    });
    std::string witness_path;
#if defined(ZKX_HAS_SP1)
    RUN_WITH_PROFILE("converting witness to binary", {
      std::string_view witness_dir = tsl::io::Dirname(options.witness_path);
      witness_path = tsl::io::JoinPath(witness_dir, "witness.bin");
      MakeWitness(options.witness_path.c_str(), witness_path.c_str());
    });
#else
    witness_path = options.witness_path;
#endif
    F* az = nullptr;
    F* bz = nullptr;
    F* cz = nullptr;
    F* witness_values = nullptr;
    MakeSolutions(options.r1cs_path.c_str(), options.proving_key_path.c_str(),
                  witness_path.c_str(), &az, &bz, &cz, &witness_values);

    std::unique_ptr<tsl::ReadOnlyMemoryRegion> a_region;
    std::unique_ptr<tsl::ReadOnlyMemoryRegion> b_region;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
        tsl::io::JoinPath(options.output_dir, "infinity_a.bin"), &a_region));
    TF_RETURN_IF_ERROR(tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(
        tsl::io::JoinPath(options.output_dir, "infinity_b.bin"), &b_region));

    std::vector<F> a_wires(m - pk_additional_data.num_infinity_a);
    std::vector<F> b_wires(m - pk_additional_data.num_infinity_b);
    std::vector<F> wire_values(m - l - 1);
#if defined(ZKX_HAS_OPENMP)
#pragma omp parallel sections
    {
#pragma omp section
      {
#endif
        size_t a_idx = 0;
        for (size_t i = 0; i < m; i++) {
          F val = witness_values[i];
          if (!reinterpret_cast<const bool*>(a_region->data())[i]) {
            a_wires[a_idx] = val;
            a_idx++;
          }
        }
#if defined(ZKX_HAS_OPENMP)
      }
#pragma omp section
      {
#endif
        size_t b_idx = 0;
        for (size_t i = 0; i < m; i++) {
          F val = witness_values[i];
          if (!reinterpret_cast<const bool*>(b_region->data())[i]) {
            b_wires[b_idx] = val;
            b_idx++;
          }
        }
#if defined(ZKX_HAS_OPENMP)
      }
    }
#endif
    wire_values.assign(witness_values + (l + 1), witness_values + m);

    RUN_WITH_PROFILE("sending witness parameters", {
      TF_RETURN_IF_ERROR(
          AddVectorParameter(absl::MakeConstSpan(a_wires), &buffers));
      TF_RETURN_IF_ERROR(
          AddVectorParameter(absl::MakeConstSpan(b_wires), &buffers));
      TF_RETURN_IF_ERROR(
          AddVectorParameter(absl::MakeConstSpan(wire_values), &buffers));
      absl::Span<const F> az_span(az, n);
      TF_RETURN_IF_ERROR(AddVectorParameter(az_span, &buffers));
      absl::Span<const F> bz_span(bz, n);
      TF_RETURN_IF_ERROR(AddVectorParameter(bz_span, &buffers));
      absl::Span<const F> cz_span(cz, n);
      TF_RETURN_IF_ERROR(AddVectorParameter(cz_span, &buffers));
      if (options.no_zk) {
        TF_RETURN_IF_ERROR(AddScalarParameter(F(0), &buffers));
        TF_RETURN_IF_ERROR(AddScalarParameter(F(0), &buffers));
      } else {
        TF_RETURN_IF_ERROR(AddScalarParameter(F::Random(), &buffers));
        TF_RETURN_IF_ERROR(AddScalarParameter(F::Random(), &buffers));
      }
    });

    zkx::Literal proof;
    RUN_WITH_PROFILE("generating proof", {
      TF_ASSIGN_OR_RETURN(
          zkx::ExecutionOutput output,
          runner_.ExecuteWithDeviceBuffers(opaque_executable.get(), buffers,
                                           /*profile=*/nullptr));

      TF_ASSIGN_OR_RETURN(proof,
                          runner_.TransferLiteralFromDevice(output.Result()));
    });

    std::cout << "Proof:" << proof.ToString() << std::endl;

    TF_RETURN_IF_ERROR(WriteProofToBin<Curve>(proof, options.proof_path));
    std::cout << "Proof is saved to \"" << options.proof_path << "\""
              << std::endl;

    // TODO(chokobole): Consider retrieving public values directly from the Go
    // binding if parsing an additional witness becomes a performance
    // bottleneck.
    std::unique_ptr<Witness<F>> witness;
    RUN_WITH_PROFILE("parsing witness", {
      TF_ASSIGN_OR_RETURN(witness, ParseWitness<F>(witness_path.c_str()));
    });
    TF_RETURN_IF_ERROR(WritePublicToBin<F>(
        absl::MakeConstSpan(witness->secrets).subspan(0, witness->num_publics),
        options.public_path));
    std::cout << "Public values are saved to \"" << options.public_path << "\""
              << std::endl;

    return absl::OkStatus();
  }
};

}  // namespace rabbitsnark::gnark

#endif  // GNARK_COMMAND_RUNNER_IMPL_H_
