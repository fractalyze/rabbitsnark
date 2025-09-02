#ifndef CIRCOM_ZKEY_COEFFICIENT_H_
#define CIRCOM_ZKEY_COEFFICIENT_H_

#include <stdint.h>

#include <algorithm>
#include <vector>

#include "absl/types/span.h"

#include "zkx/math/base/sparse_matrix.h"

namespace rabbitsnark::circom {

#pragma pack(push, 1)
// R1CS is represented as A * z ∘ B * z = C * z, where ∘ is the Hadamard
// product. Each constraint is composed as follows:
//
// - [aᵢ,₀, ...,  aᵢ,ₘ₋₁] * [z₀, ..., zₘ₋₁]
// - [bᵢ,₀, ...,  bᵢ,ₘ₋₁] * [z₀, ..., zₘ₋₁]
// - [cᵢ,₀, ...,  cᵢ,ₘ₋₁] * [z₀, ..., zₘ₋₁]
//
// where i is the index of the constraints (0 ≤ i < n),
// m is the number of QAP variables, and n is the number of constraints.
//
// The last constraint is computed if we know the first two constraints.
// Therefore, `Coefficient` represents the first two constraints.
template <typename F>
struct Coefficient {
  // A value denoting the matrix this constraint is for. If 0, this constraint
  // is for matrix A. Else, this constraint is for matrix B.
  uint32_t matrix;
  // The index of the constraint, (0 ≤ i < n).
  uint32_t constraint;
  // The index of the QAP variables, (0 ≤ j < m).
  uint32_t signal;
  // The values of the coefficient; if the `matrix` is 0, then this points to
  // the a[i][j]. Otherwise, this points to b[i][j].
  F value;

  bool operator==(const Coefficient& other) const {
    return matrix == other.matrix && constraint == other.constraint &&
           signal == other.signal && value == other.value;
  }
  bool operator!=(const Coefficient& other) const { return !operator==(other); }

  // Helper function to create CSR matrices from coefficients
  static void ToSparseMatrices(absl::Span<const Coefficient> coefficients,
                               zkx::math::SparseMatrix<F>& a_matrix,
                               zkx::math::SparseMatrix<F>& b_matrix) {
    for (const Coefficient& coeff : coefficients) {
      if (coeff.matrix == 0) {
        a_matrix.InsertUnique(coeff.constraint, coeff.signal, coeff.value);
      } else {
        b_matrix.InsertUnique(coeff.constraint, coeff.signal, coeff.value);
      }
    }
  }
};
#pragma pack(pop)

}  // namespace rabbitsnark::circom

#endif  // CIRCOM_ZKEY_COEFFICIENT_H_
