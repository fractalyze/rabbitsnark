#ifndef CIRCOM_HLO_HLO_GENERATOR_H_
#define CIRCOM_HLO_HLO_GENERATOR_H_

#include <map>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "common/hlo/hlo_generator_util.h"

#include "circom/zkey/zkey.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/path.h"
#include "zkx/base/logging.h"
#include "zkx/math/poly/bit_reverse.h"

namespace zkx::circom {
namespace {

const std::string_view kHloText = R"(
ENTRY %groth16 () -> (bn254.g1_affine[], bn254.g2_affine[], bn254.g1_affine[]) {
  %vk.alpha_g1 = bn254.g1_affine[] parameter(0)
  %vk.beta_g2 = bn254.g2_affine[] parameter(1)
  %vk.gamma_g2 = bn254.g2_affine[] parameter(2)
  %vk.delta_g2 = bn254.g2_affine[] parameter(3)

  %pk.beta_g1 = bn254.g1_affine[] parameter(4)
  %pk.delta_g1 = bn254.g1_affine[] parameter(5)
  %pk.a_g1_query = bn254.g1_affine[$m] parameter(6)
  %pk.b_g1_query = bn254.g1_affine[$m] parameter(7)
  %pk.b_g2_query = bn254.g2_affine[$m] parameter(8)
  %pk.l_g1_query = bn254.g1_affine[$(m - l - 1)] parameter(9)
  %pk.h_g1_query = bn254.g1_affine[$n] parameter(10)

  %A = bn254.sf[$n, $m]{1, 0:D(D, C)NNZ($a_num_non_zeros)} parameter(11)
  %B = bn254.sf[$n, $m]{1, 0:D(D, C)NNZ($b_num_non_zeros)} parameter(12)

  %twiddles = bn254.sf[$n] parameter(13)
  %fft_twiddles = bn254.sf[$n] parameter(14)
  %ifft_twiddles = bn254.sf[$n] parameter(15)

  %z = bn254.sf[$m]{0:MONT(false)} parameter(16)
  %r = bn254.sf[] parameter(17)
  %s = bn254.sf[] parameter(18)

  %z.in_mont = bn254.sf[$m]{0} convert(%z)

  %Az = bn254.sf[$n] dot(%A, %z.in_mont)
  %Bz = bn254.sf[$n] dot(%B, %z.in_mont)
  %Cz = bn254.sf[$n] multiply(%Az, %Bz)

  %a.poly = bn254.sf[$n] fft(%Az, %ifft_twiddles), fft_type=IFFT, fft_length=$n, fft_do_bit_reverse=false, control-predecessors={%Cz}
  %b.poly = bn254.sf[$n] fft(%Bz, %ifft_twiddles), fft_type=IFFT, fft_length=$n, fft_do_bit_reverse=false, control-predecessors={%Cz}
  %c.poly = bn254.sf[$n] fft(%Cz, %ifft_twiddles), fft_type=IFFT, fft_length=$n, fft_do_bit_reverse=false

  %a.poly_x_twiddles = bn254.sf[$n] multiply(%a.poly, %twiddles)
  %b.poly_x_twiddles = bn254.sf[$n] multiply(%b.poly, %twiddles)
  %c.poly_x_twiddles = bn254.sf[$n] multiply(%c.poly, %twiddles)

  %a.evals = bn254.sf[$n] fft(%a.poly_x_twiddles, %fft_twiddles), fft_type=FFT, fft_length=$n, fft_do_bit_reverse=false
  %b.evals = bn254.sf[$n] fft(%b.poly_x_twiddles, %fft_twiddles), fft_type=FFT, fft_length=$n, fft_do_bit_reverse=false
  %c.evals = bn254.sf[$n] fft(%c.poly_x_twiddles, %fft_twiddles), fft_type=FFT, fft_length=$n, fft_do_bit_reverse=false

  %h.evals.tmp = bn254.sf[$n] multiply(%a.evals, %b.evals)
  %h.evals = bn254.sf[$n] subtract(%h.evals.tmp, %c.evals)

  %h.evals.in_std = bn254.sf[$n]{0:MONT(false)} convert(%h.evals)

  %msm_1 = bn254.g1_xyzz[] msm(%z, %pk.a_g1_query), window_bits=$non_h_msm_window_bits
  %msm_2 = bn254.g1_xyzz[] msm(%z, %pk.b_g1_query), window_bits=$non_h_msm_window_bits
  %msm_3 = bn254.g2_xyzz[] msm(%z, %pk.b_g2_query), window_bits=$non_h_msm_window_bits
  %z.witness = bn254.sf[$(m - l - 1)]{0:MONT(false)} slice(%z), slice={[$(l + 1):$m]}
  %msm_4 = bn254.g1_xyzz[] msm(%z.witness, %pk.l_g1_query), window_bits=$non_h_msm_window_bits
  %msm_5 = bn254.g1_xyzz[] msm(%h.evals.in_std, %pk.h_g1_query), window_bits=$h_msm_window_bits

  %proof.A.tmp = bn254.g1_xyzz[] add(%vk.alpha_g1, %msm_1)
  %r_delta_g1 = bn254.g1_xyzz[] multiply(%r, %pk.delta_g1)
  %proof.A_xyzz = bn254.g1_xyzz[] add(%proof.A.tmp, %r_delta_g1)
  %proof.A = bn254.g1_affine[] convert(%proof.A_xyzz)

  %proof.B.tmp_g1 = bn254.g1_xyzz[] add(%pk.beta_g1, %msm_2)
  %s_delta_g1 = bn254.g1_xyzz[] multiply(%s, %pk.delta_g1)
  %proof.B_g1 = bn254.g1_xyzz[] add(%proof.B.tmp_g1, %s_delta_g1)

  %proof.B.tmp_g2 = bn254.g2_xyzz[] add(%vk.beta_g2, %msm_3)
  %s_delta_g2 = bn254.g2_xyzz[] multiply(%s, %vk.delta_g2)
  %proof.B_g2_xyzz = bn254.g2_xyzz[] add(%proof.B.tmp_g2, %s_delta_g2)
  %proof.B_g2 = bn254.g2_affine[] convert(%proof.B_g2_xyzz)

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
absl::Status WriteABMatricesToFile(
    size_t num_rows, size_t num_cols,
    const std::vector<Coefficient<T>>& coefficients,
    std::string_view output_dir,
    std::map<std::string, std::string>& replacements) {
  math::SparseMatrix<T> a_matrix(num_rows, num_cols);
  math::SparseMatrix<T> b_matrix(num_rows, num_cols);
  Coefficient<T>::ToSparseMatrices(coefficients, a_matrix, b_matrix);

  TF_RETURN_IF_ERROR(
      WriteCSRSparseMatrixToFile(a_matrix, output_dir, "a", replacements));
  TF_RETURN_IF_ERROR(
      WriteCSRSparseMatrixToFile(b_matrix, output_dir, "b", replacements));
  return absl::OkStatus();
}

template <typename T>
absl::Status WriteTwiddlesToFile(
    size_t domain_size, std::string_view output_dir,
    std::map<std::string, std::string>& replacements) {
  TF_ASSIGN_OR_RETURN(T w, math::GetRootOfUnity<T>(2 * domain_size));

  std::vector<T> twiddles(domain_size);
  T twiddle = T::One();
  for (size_t i = 0; i < domain_size; ++i) {
    twiddles[i] = twiddle;
    twiddle *= w;
  }

  math::BitReverseShuffleInPlace(twiddles);

  return WriteSpanToFile(absl::MakeConstSpan(twiddles), output_dir, "twiddles");
}

}  // namespace

template <typename Curve>
absl::StatusOr<std::string> GenerateHLO(const ZKey<Curve>& zkey,
                                        int32_t h_msm_window_bits,
                                        int32_t non_h_msm_window_bits,
                                        std::string_view output_dir) {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using F = typename G1AffinePoint::ScalarField;

  const v1::ZKey<Curve>* v1_zkey = zkey.ToV1();
  const v1::ZKeyHeaderGrothSection<Curve>& header = v1_zkey->header_groth;
  const v1::CoefficientsSection<F>& coefficients = v1_zkey->coefficients;
  const ProvingKey<Curve>& pk = zkey.GetProvingKey();

  std::map<std::string, std::string> replacements;
  size_t l = header.num_public_inputs;
  size_t m = header.num_vars;
  size_t n = header.domain_size;

  std::cout << "l: " << l << std::endl;
  std::cout << "m: " << m << std::endl;
  std::cout << "n: " << n << std::endl;

  replacements["$m"] = absl::StrCat(m);
  replacements["$n"] = absl::StrCat(n);
  replacements["$(l + 1)"] = absl::StrCat(l + 1);
  replacements["$(m - l - 1)"] = absl::StrCat(m - l - 1);
  replacements["$n"] = absl::StrCat(n);
  replacements["$h_msm_window_bits"] = absl::StrCat(h_msm_window_bits);
  replacements["$non_h_msm_window_bits"] = absl::StrCat(non_h_msm_window_bits);

  CHECK_EQ(pk.a_g1_query.size(), m);
  CHECK_EQ(pk.b_g1_query.size(), m);
  CHECK_EQ(pk.b_g2_query.size(), m);
  CHECK_EQ(pk.c_g1_query.size(), m - l - 1);
  CHECK_EQ(pk.h_g1_query.size(), n);

  ProvingKeyAdditionalData<Curve> additional_data;
  additional_data.l = l;
  additional_data.m = m;
  additional_data.n = n;
  additional_data.alpha_g1 = *pk.verifying_key.alpha_g1;
  additional_data.beta_g2 = *pk.verifying_key.beta_g2;
  additional_data.gamma_g2 = *pk.verifying_key.gamma_g2;
  additional_data.delta_g2 = *pk.verifying_key.delta_g2;
  additional_data.beta_g1 = *pk.verifying_key.beta_g1;
  additional_data.delta_g1 = *pk.verifying_key.delta_g1;

  // TODO(chokobole): Use `output_dir` instead of `std::string(output_dir)`
  // after `CreateDir()` can accept `std::string_view`.
  absl::Status s = tsl::Env::Default()->CreateDir(std::string(output_dir));
  if (!(s.ok() || absl::IsAlreadyExists(s))) {
    return s;
  }

  TF_RETURN_IF_ERROR(additional_data.WriteToFile(output_dir));
  TF_RETURN_IF_ERROR(WriteSpanToFile(pk.a_g1_query, output_dir, "a_g1_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(pk.b_g1_query, output_dir, "b_g1_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(pk.b_g2_query, output_dir, "b_g2_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(pk.c_g1_query, output_dir, "l_g1_query"));
  TF_RETURN_IF_ERROR(WriteSpanToFile(pk.h_g1_query, output_dir, "h_g1_query"));

  TF_RETURN_IF_ERROR(WriteABMatricesToFile(n, m, coefficients.coefficients,
                                           output_dir, replacements));
  TF_RETURN_IF_ERROR(WriteTwiddlesToFile<F>(n, output_dir, replacements));
  TF_RETURN_IF_ERROR(WriteFFTTwiddlesToFile<F>(n, output_dir, replacements));
  TF_RETURN_IF_ERROR(WriteIFFTTwiddlesToFile<F>(n, output_dir, replacements));

  std::string hlo_string = absl::StrReplaceAll(kHloText, replacements);

  VLOG(1) << hlo_string;
  std::string output_path = tsl::io::JoinPath(output_dir, "groth16.hlo");
  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), output_path, hlo_string));

  return hlo_string;
}

}  // namespace zkx::circom

#endif  // CIRCOM_HLO_HLO_GENERATOR_H_
