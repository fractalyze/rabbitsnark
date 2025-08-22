#ifndef GNARK_HLO_HLO_GENERATOR_H_
#define GNARK_HLO_HLO_GENERATOR_H_

#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "common/hlo/hlo_generator_util.h"

#include "gnark/pk/proving_key.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/path.h"
#include "zkx/base/logging.h"
#include "zkx/math/poly/bit_reverse.h"

namespace zkx::gnark {

template <typename Curve>
struct GnarkProvingKeyAdditionalData : public ProvingKeyAdditionalData<Curve> {
  int64_t a_g1_query_size;
  int64_t b_g1_query_size;

  uint64_t num_infinity_a;
  uint64_t num_infinity_b;

  absl::Status WriteToFile(std::string_view output_dir) const {
    return tsl::WriteStringToFile(
        tsl::Env::Default(),
        tsl::io::JoinPath(output_dir, "pk_additional_data.bin"),
        std::string_view(reinterpret_cast<const char*>(this),
                         sizeof(GnarkProvingKeyAdditionalData)));
  }

  static absl::StatusOr<GnarkProvingKeyAdditionalData> ReadFromFile(
      std::string_view output_dir) {
    std::string content;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(output_dir, "pk_additional_data.bin"), &content));
    GnarkProvingKeyAdditionalData additional_data;
    memcpy(&additional_data, content.data(),
           sizeof(GnarkProvingKeyAdditionalData));
    return std::move(additional_data);
  }
};

const std::string_view kHloText = R"(
ENTRY %groth16 () -> (bn254.g1_affine[], bn254.g2_affine[], bn254.g1_affine[])  {
  %pk.alpha_g1 = bn254.g1_affine[] parameter(0)
  %pk.beta_g1 = bn254.g1_affine[] parameter(1)
  %pk.beta_g2 = bn254.g2_affine[] parameter(2)
  %pk.delta_g1 = bn254.g1_affine[] parameter(3)
  %pk.delta_g2 = bn254.g2_affine[] parameter(4)

  %pk.a_g1_query = bn254.g1_affine[$a_g1_query_size] parameter(5)
  %pk.b_g1_query = bn254.g1_affine[$b_g1_query_size] parameter(6)
  %pk.b_g2_query = bn254.g2_affine[$b_g1_query_size] parameter(7)
  // h_g1_query in circom
  %pk.z_g1_query = bn254.g1_affine[$(n - 1)] parameter(8)
  %pk.k_g1_query = bn254.g1_affine[$(m - l - 1)] parameter(9)

  %coset_twiddles = bn254.sf[$n] parameter(10)
  %coset_inv_twiddles = bn254.sf[$n] parameter(11)
  %fft_twiddles = bn254.sf[$n] parameter(12)
  %ifft_twiddles = bn254.sf[$n] parameter(13)

  // TODO(chokobole): Replace with a broadcast operation once fusion support is added.
  %den = bn254.sf[$n] parameter(14)

  %a_wires = bn254.sf[$a_g1_query_size]{0:MONT(true)} parameter(15)
  %b_wires = bn254.sf[$b_g1_query_size]{0:MONT(true)} parameter(16)
  %wire_values = bn254.sf[$(m - l - 1)]{0:MONT(true)} parameter(17)

  %Az = bn254.sf[$n] parameter(18)
  %Bz = bn254.sf[$n] parameter(19)
  %Cz = bn254.sf[$n] parameter(20)

  %r = bn254.sf[] parameter(21)
  %s = bn254.sf[] parameter(22)

  // [A]_1
  %msm_1 = bn254.g1_xyzz[] msm(%a_wires, %pk.a_g1_query), window_bits=$non_h_msm_window_bits
  %proof.A.tmp = bn254.g1_xyzz[] add(%pk.alpha_g1, %msm_1)
  %r_delta_g1 = bn254.g1_xyzz[] multiply(%r, %pk.delta_g1)
  %proof.A_xyzz = bn254.g1_xyzz[] add(%proof.A.tmp, %r_delta_g1)
  %proof.A = bn254.g1_affine[] convert(%proof.A_xyzz)

  // [B]_1
  %msm_2 = bn254.g1_xyzz[] msm(%b_wires, %pk.b_g1_query), window_bits=$non_h_msm_window_bits
  %proof.B.tmp_g1 = bn254.g1_xyzz[] add(%pk.beta_g1, %msm_2)
  %s_delta_g1 = bn254.g1_xyzz[] multiply(%s, %pk.delta_g1)
  %proof.B_g1 = bn254.g1_xyzz[] add(%proof.B.tmp_g1, %s_delta_g1)

  // [B]_2
  %msm_3 = bn254.g2_xyzz[] msm(%b_wires, %pk.b_g2_query), window_bits=$non_h_msm_window_bits
  %s_delta_g2 = bn254.g2_xyzz[] multiply(%s, %pk.delta_g2)
  %proof.B_g2.tmp = bn254.g2_xyzz[] add(%msm_3, %s_delta_g2)
  %proof.B_g2.xyzz = bn254.g2_xyzz[] add(%proof.B_g2.tmp, pk.beta_g2)
  %proof.B_g2 = bn254.g2_affine[] convert(%proof.B_g2.xyzz)

  // [C]_1
  %a.poly = bn254.sf[$n] fft(%Az, %ifft_twiddles), fft_type=IFFT, fft_length=$n, fft_do_bit_reverse=false
  %b.poly = bn254.sf[$n] fft(%Bz, %ifft_twiddles), fft_type=IFFT, fft_length=$n, fft_do_bit_reverse=false
  %c.poly = bn254.sf[$n] fft(%Cz, %ifft_twiddles), fft_type=IFFT, fft_length=$n, fft_do_bit_reverse=false

  %a.poly_x_coset_twiddles = bn254.sf[$n] multiply(%a.poly, %coset_twiddles)
  %b.poly_x_coset_twiddles = bn254.sf[$n] multiply(%b.poly, %coset_twiddles)
  %c.poly_x_coset_twiddles = bn254.sf[$n] multiply(%c.poly, %coset_twiddles)

  %a.evals = bn254.sf[$n] fft(%a.poly_x_coset_twiddles, %fft_twiddles), fft_type=FFT, fft_length=$n, fft_do_bit_reverse=false
  %b.evals = bn254.sf[$n] fft(%b.poly_x_coset_twiddles, %fft_twiddles), fft_type=FFT, fft_length=$n, fft_do_bit_reverse=false
  %c.evals = bn254.sf[$n] fft(%c.poly_x_coset_twiddles, %fft_twiddles), fft_type=FFT, fft_length=$n, fft_do_bit_reverse=false

  %h.evals.tmp_1 = bn254.sf[$n] multiply(%a.evals, %b.evals)
  %h.evals.tmp_2 = bn254.sf[$n] subtract(%h.evals.tmp_1, %c.evals)
  %h.evals = bn254.sf[$n] multiply(%h.evals.tmp_2, %den)

  %h.poly = bn254.sf[$n] fft(%h.evals, %ifft_twiddles), fft_type=IFFT, fft_length=$n, fft_do_bit_reverse=false
  %h.poly_x_coset_inv_twiddles = bn254.sf[$n] multiply(%h.poly, %coset_inv_twiddles)

  %msm_4 = bn254.g1_xyzz[] msm(%wire_values, %pk.k_g1_query), window_bits=$non_h_msm_window_bits
  %msm_5 = bn254.g1_xyzz[] msm(%h.poly_x_coset_inv_twiddles, %pk.z_g1_query), window_bits=$non_h_msm_window_bits

  %proof.C.tmp = bn254.g1_xyzz[] add(%msm_4, %msm_5)
  %s_A_g1 = bn254.g1_xyzz[] multiply(%s, %proof.A)
  %r_B_g1 = bn254.g1_xyzz[] multiply(%r, %proof.B_g1)
  %proof.C.tmp_2 = bn254.g1_xyzz[] add(%proof.C.tmp, %s_A_g1)
  %proof.C.tmp_3 = bn254.g1_xyzz[] add(%proof.C.tmp_2, %r_B_g1)
  %rs = bn254.sf[] multiply(%r, %s)
  %rs_delta_g1 = bn254.g1_xyzz[] multiply(%rs, %pk.delta_g1)
  %proof.C_xyzz = bn254.g1_xyzz[] subtract(%proof.C.tmp_3, %rs_delta_g1)
  %proof.C = bn254.g1_affine[] convert(%proof.C_xyzz)

  ROOT %proof = (bn254.g1_affine[], bn254.g2_affine[], bn254.g1_affine[]) tuple(%proof.A, %proof.B_g2, %proof.C)
}
)";

template <typename T>
absl::Status WriteCosetTwiddlesToFile(T multiplicative_gen, size_t domain_size,
                                      std::string_view output_dir) {
  std::vector<T> coset_twiddles(domain_size);
  T x = 1;
  for (int64_t i = 0; i < domain_size; ++i) {
    coset_twiddles[i] = x;
    x *= multiplicative_gen;
  }

  math::BitReverseShuffleInPlace(coset_twiddles);

  return WriteSpanToFile(absl::MakeConstSpan(coset_twiddles), output_dir,
                         "coset_twiddles");
}

template <typename T>
absl::Status WriteCosetInvTwiddlesToFile(T multiplicative_gen_inv,
                                         size_t domain_size,
                                         std::string_view output_dir) {
  std::vector<T> coset_inv_twiddles(domain_size);
  T x = 1;
  for (int64_t i = 0; i < domain_size; ++i) {
    coset_inv_twiddles[i] = x;
    x *= multiplicative_gen_inv;
  }

  math::BitReverseShuffleInPlace(coset_inv_twiddles);

  return WriteSpanToFile(absl::MakeConstSpan(coset_inv_twiddles), output_dir,
                         "coset_inv_twiddles");
}

template <typename T>
absl::Status WriteDenToFile(T multiplicative_gen, T cardinality,
                            size_t domain_size, std::string_view output_dir) {
  T den = (multiplicative_gen.Pow(cardinality) - T::One()).Inverse().value();
  std::vector<T> den_vec(domain_size);
  for (int64_t i = 0; i < domain_size; ++i) {
    den_vec[i] = den;
  }

  return WriteSpanToFile(absl::MakeConstSpan(den_vec), output_dir, "den");
}

template <typename Curve>
absl::StatusOr<std::string> GenerateHLO(const ProvingKey<Curve>& proving_key,
                                        int32_t h_msm_window_bits,
                                        int32_t non_h_msm_window_bits,
                                        std::string_view output_dir) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using F = typename G1AffinePoint::ScalarField;

  std::map<std::string, std::string> replacements;
  size_t m = proving_key.infinity_a.size();
  size_t l = m - 1 - proving_key.k_g1_query.size();
  size_t n = proving_key.domain.cardinality;

  std::cout << "l: " << l << std::endl;
  std::cout << "m: " << m << std::endl;
  std::cout << "n: " << n << std::endl;

  replacements["$m"] = absl::StrCat(m);
  replacements["$n"] = absl::StrCat(n);
  replacements["$a_g1_query_size"] =
      absl::StrCat(proving_key.a_g1_query.size());
  replacements["$b_g1_query_size"] =
      absl::StrCat(proving_key.b_g1_query.size());
  replacements["$(n - 1)"] = absl::StrCat(n - 1);
  replacements["$(m - l - 1)"] = absl::StrCat(m - l - 1);
  replacements["$h_msm_window_bits"] = absl::StrCat(h_msm_window_bits);
  replacements["$non_h_msm_window_bits"] = absl::StrCat(non_h_msm_window_bits);

  CHECK_LE(proving_key.a_g1_query.size(), n);
  CHECK_LE(proving_key.b_g1_query.size(), n);
  CHECK_LE(proving_key.b_g2_query.size(), n);
  CHECK_EQ(proving_key.z_g1_query.size(), n - 1);
  CHECK_EQ(proving_key.k_g1_query.size(), m - l - 1);

  GnarkProvingKeyAdditionalData<Curve> additional_data;
  additional_data.l = l;
  additional_data.m = m;
  additional_data.n = n;
  additional_data.a_g1_query_size = proving_key.a_g1_query.size();
  additional_data.b_g1_query_size = proving_key.b_g1_query.size();
  additional_data.num_infinity_a = proving_key.num_infinity_a;
  additional_data.num_infinity_b = proving_key.num_infinity_b;

  additional_data.alpha_g1 = proving_key.alpha_g1;
  additional_data.beta_g2 = proving_key.beta_g2;
  additional_data.delta_g2 = proving_key.delta_g2;
  additional_data.beta_g1 = proving_key.beta_g1;
  additional_data.delta_g1 = proving_key.delta_g1;

  absl::Status s = tsl::Env::Default()->CreateDir(output_dir);
  if (!(s.ok() || absl::IsAlreadyExists(s))) {
    return s;
  }

  TF_RETURN_IF_ERROR(additional_data.WriteToFile(output_dir));
  TF_RETURN_IF_ERROR(WriteSpanToFile(
      absl::MakeConstSpan(proving_key.a_g1_query), output_dir, "a_g1_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(
      absl::MakeConstSpan(proving_key.b_g1_query), output_dir, "b_g1_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(
      absl::MakeConstSpan(proving_key.b_g2_query), output_dir, "b_g2_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(
      absl::MakeConstSpan(proving_key.z_g1_query), output_dir, "z_g1_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(
      absl::MakeConstSpan(proving_key.k_g1_query), output_dir, "k_g1_query"));

  std::vector<uint8_t> converted_a(proving_key.infinity_a.begin(),
                                   proving_key.infinity_a.end());
  std::vector<uint8_t> converted_b(proving_key.infinity_b.begin(),
                                   proving_key.infinity_b.end());
  TF_RETURN_IF_ERROR(WriteSpanToFile(absl::MakeConstSpan(converted_a),
                                     output_dir, "infinity_a"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(absl::MakeConstSpan(converted_b),
                                     output_dir, "infinity_b"));

  TF_RETURN_IF_ERROR(WriteCosetTwiddlesToFile<F>(
      proving_key.domain.fr_multiplicative_gen, n, output_dir));
  TF_RETURN_IF_ERROR(WriteCosetInvTwiddlesToFile<F>(
      proving_key.domain.fr_multiplicative_gen_inv, n, output_dir));
  TF_RETURN_IF_ERROR(WriteFFTTwiddlesToFile<F>(n, output_dir));
  TF_RETURN_IF_ERROR(WriteIFFTTwiddlesToFile<F>(n, output_dir));
  TF_RETURN_IF_ERROR(WriteDenToFile<F>(proving_key.domain.fr_multiplicative_gen,
                                       proving_key.domain.cardinality, n,
                                       output_dir));

  std::string hlo_string = absl::StrReplaceAll(kHloText, replacements);

  VLOG(1) << hlo_string;
  std::string output_path = tsl::io::JoinPath(output_dir, "groth16.hlo");
  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), output_path, hlo_string));

  return hlo_string;
}

}  // namespace zkx::gnark

#endif  // GNARK_HLO_HLO_GENERATOR_H_
