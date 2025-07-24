#ifndef COMMON_HLO_HLO_GENERATOR_UTIL_H_
#define COMMON_HLO_HLO_GENERATOR_UTIL_H_

#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/math/base/batch_inverse.h"
#include "zkx/math/base/sparse_matrix.h"
#include "zkx/math/poly/root_of_unity.h"

namespace zkx {

template <typename Curve>
struct ProvingKeyAdditionalData {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;

  int64_t l;
  int64_t m;
  int64_t n;

  G1AffinePoint alpha_g1;
  G2AffinePoint beta_g2;
  G2AffinePoint gamma_g2;
  G2AffinePoint delta_g2;
  G1AffinePoint beta_g1;
  G1AffinePoint delta_g1;

  absl::Status WriteToFile(std::string_view output_dir) const {
    return tsl::WriteStringToFile(
        tsl::Env::Default(),
        tsl::io::JoinPath(output_dir, "pk_additional_data.bin"),
        std::string_view(reinterpret_cast<const char*>(this),
                         sizeof(ProvingKeyAdditionalData)));
  }

  static absl::StatusOr<ProvingKeyAdditionalData> ReadFromFile(
      std::string_view output_dir) {
    std::string content;
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(output_dir, "pk_additional_data.bin"), &content));
    ProvingKeyAdditionalData additional_data;
    memcpy(&additional_data, content.data(), sizeof(ProvingKeyAdditionalData));
    return std::move(additional_data);
  }
};

template <typename T>
absl::Status WriteSpanToFile(absl::Span<const T> span,
                             std::string_view output_dir,
                             std::string_view name) {
  std::string basename = absl::StrCat(name, ".bin");
  return tsl::WriteStringToFile(
      tsl::Env::Default(), tsl::io::JoinPath(output_dir, basename),
      std::string_view(reinterpret_cast<const char*>(span.data()),
                       span.size() * sizeof(T)));
}

template <typename T>
absl::Status WriteCSRSparseMatrixToFile(
    const math::SparseMatrix<T>& matrix, std::string_view output_dir,
    std::string_view name, std::map<std::string, std::string>& replacements) {
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> buffer,
                      matrix.ToCSRBuffer(/*sort=*/true));
  replacements[absl::StrCat("$", name, "_num_non_zeros")] =
      absl::StrCat(matrix.NumNonZeros());
  return WriteSpanToFile(absl::MakeConstSpan(buffer), output_dir, name);
}

template <typename T>
absl::Status WriteFFTTwiddlesToFile(size_t domain_size,
                                    std::string_view output_dir) {
  TF_ASSIGN_OR_RETURN(T w, math::GetRootOfUnity<T>(domain_size));

  std::vector<T> fft_twiddles(domain_size);
  T x = 1;
  for (int64_t i = 0; i < domain_size; ++i) {
    fft_twiddles[i] = x;
    x *= w;
  }

  return WriteSpanToFile(absl::MakeConstSpan(fft_twiddles), output_dir,
                         "fft_twiddles");
}

template <typename T>
absl::Status WriteIFFTTwiddlesToFile(size_t domain_size,
                                     std::string_view output_dir) {
  TF_ASSIGN_OR_RETURN(T w, math::GetRootOfUnity<T>(domain_size));

  T x = 1;
  std::vector<T> ifft_twiddles(domain_size);
  for (int64_t i = 0; i < domain_size; ++i) {
    ifft_twiddles[i] = x;
    x *= w;
  }
  TF_RETURN_IF_ERROR(math::BatchInverse(ifft_twiddles, &ifft_twiddles));

  return WriteSpanToFile(absl::MakeConstSpan(ifft_twiddles), output_dir,
                         "ifft_twiddles");
}

}  // namespace zkx

#endif  // COMMON_HLO_HLO_GENERATOR_UTIL_H_
