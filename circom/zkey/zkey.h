#ifndef CIRCOM_ZKEY_ZKEY_H_
#define CIRCOM_ZKEY_ZKEY_H_

#include <string.h>
#include <sys/mman.h>

#include <memory>
#include <string_view>
#include <utility>

#include "circom/base/modulus.h"
#include "circom/base/sections.h"
#include "circom/zkey/coefficient.h"
#include "circom/zkey/proving_key.h"
#include "xla/tsl/platform/env.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/read_only_buffer.h"

namespace rabbitsnark::circom {
namespace v1 {

template <typename Curve>
struct ZKey;

}  // namespace v1

template <typename Curve>
struct ZKey {
  using F = typename Curve::G1Curve::ScalarField;

  explicit ZKey(std::unique_ptr<tsl::ReadOnlyMemoryRegion> region)
      : region(std::move(region)) {}
  virtual ~ZKey() = default;

  virtual uint32_t GetVersion() const = 0;

  virtual v1::ZKey<Curve>* ToV1() { return nullptr; }
  virtual const v1::ZKey<Curve>* ToV1() const { return nullptr; }

  virtual absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) = 0;

  virtual ProvingKey<Curve> GetProvingKey() const = 0;
  virtual absl::Span<const Coefficient<F>> GetCoefficients() const = 0;
  virtual size_t GetDomainSize() const = 0;
  virtual size_t GetNumInstanceVariables() const = 0;
  virtual size_t GetNumWitnessVariables() const = 0;

  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
};

constexpr char kZKeyMagic[4] = {'z', 'k', 'e', 'y'};

template <typename Curve>
absl::StatusOr<std::unique_ptr<ZKey<Curve>>> ParseZKey(std::string_view path) {
  std::unique_ptr<tsl::ReadOnlyMemoryRegion> region;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->NewReadOnlyMemoryRegionFromFile(path, &region));
  if (madvise(const_cast<void*>(region->data()), region->length(),
              MADV_SEQUENTIAL) != 0) {
    return absl::InternalError("failed to call madvice()");
  }

  zkx::base::ReadOnlyBuffer buffer(region->data(), region->length());
  buffer.set_endian(zkx::base::Endian::kLittle);
  char magic[4];
  uint32_t version;
  TF_RETURN_IF_ERROR(buffer.ReadMany(magic, &version));
  if (memcmp(magic, kZKeyMagic, 4) != 0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "magic is invalid. \"$0\" vs \"$1\"", magic, kZKeyMagic));
  }
  std::unique_ptr<ZKey<Curve>> zkey;
  if (version == 1) {
    zkey.reset(new v1::ZKey<Curve>(std::move(region)));
    TF_RETURN_IF_ERROR(zkey->ToV1()->Read(buffer));
  } else {
    return absl::InvalidArgumentError(
        absl::Substitute("version is invalid. 1 vs $0", version));
  }
  return zkey;
}

namespace v1 {

enum class ZKeySectionType : uint32_t {
  kHeader = 0x1,
  kHeaderGroth = 0x2,
  kIC = 0x3,
  kCoefficients = 0x4,
  kPointsA1 = 0x5,
  kPointsB1 = 0x6,
  kPointsB2 = 0x7,
  kPointsC1 = 0x8,
  kPointsH1 = 0x9,
  kContribution = 0xa,
};

std::string_view ZKeySectionTypeToString(ZKeySectionType type);

struct ZKeyHeaderSection {
  uint32_t prover_type;

  bool operator==(const ZKeyHeaderSection& other) const {
    return prover_type == other.prover_type;
  }
  bool operator!=(const ZKeyHeaderSection& other) const {
    return prover_type != other.prover_type;
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(buffer.Read(&prover_type));
    if (prover_type != 1) {
      return absl::UnimplementedError(
          absl::Substitute("Not implemented for prover type: $0", prover_type));
    }
    return absl::OkStatus();
  }
};

template <typename Curve>
struct ZKeyHeaderGrothSection {
  Modulus q;
  Modulus r;
  uint32_t num_vars;
  uint32_t num_public_inputs;
  uint32_t domain_size;
  VerifyingKey<Curve> vkey;

  bool operator==(const ZKeyHeaderGrothSection& other) const {
    return q == other.q && r == other.r && num_vars == other.num_vars &&
           num_public_inputs == other.num_public_inputs &&
           domain_size == other.domain_size && vkey == other.vkey;
  }
  bool operator!=(const ZKeyHeaderGrothSection& other) const {
    return !operator==(other);
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
    TF_RETURN_IF_ERROR(q.Read(buffer));
    TF_RETURN_IF_ERROR(r.Read(buffer));
    TF_RETURN_IF_ERROR(
        buffer.ReadMany(&num_vars, &num_public_inputs, &domain_size));
    TF_RETURN_IF_ERROR(vkey.Read(buffer));
    return absl::OkStatus();
  }
};

template <typename T>
struct CommitmentsSection {
  absl::Span<T> commitments;

  bool operator==(const CommitmentsSection& other) const {
    return commitments == other.commitments;
  }
  bool operator!=(const CommitmentsSection& other) const {
    return commitments != other.commitments;
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer,
                    uint32_t num_commitments) {
    T* ptr;
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&ptr, num_commitments));
    commitments = {ptr, num_commitments};
    return absl::OkStatus();
  }
};

template <typename C>
using ICSection = CommitmentsSection<C>;
template <typename C>
using PointsA1Section = CommitmentsSection<C>;
template <typename C>
using PointsB1Section = CommitmentsSection<C>;
template <typename C>
using PointsB2Section = CommitmentsSection<C>;
template <typename C>
using PointsC1Section = CommitmentsSection<C>;
template <typename C>
using PointsH1Section = CommitmentsSection<C>;

template <typename F>
struct CoefficientsSection {
  absl::Span<const Coefficient<F>> coefficients;

  bool operator==(const CoefficientsSection& other) const {
    return coefficients == other.coefficients;
  }
  bool operator!=(const CoefficientsSection& other) const {
    return !operator==(other);
  }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) {
    uint32_t num_coefficients;
    TF_RETURN_IF_ERROR(buffer.Read(&num_coefficients));
    Coefficient<F>* ptr;
    TF_RETURN_IF_ERROR(buffer.ReadPtr(&ptr, num_coefficients));
    coefficients = {ptr, num_coefficients};
    return absl::OkStatus();
  }
};

template <typename Curve>
struct ZKey : public circom::ZKey<Curve> {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using F = typename G1AffinePoint::ScalarField;

  ZKeyHeaderSection header;
  ZKeyHeaderGrothSection<Curve> header_groth;
  ICSection<G1AffinePoint> ic;
  CoefficientsSection<F> coefficients;
  PointsA1Section<G1AffinePoint> points_a1;
  PointsB1Section<G1AffinePoint> points_b1;
  PointsB2Section<G2AffinePoint> points_b2;
  PointsC1Section<G1AffinePoint> points_c1;
  PointsH1Section<G1AffinePoint> points_h1;

  explicit ZKey(std::unique_ptr<tsl::ReadOnlyMemoryRegion> region)
      : circom::ZKey<Curve>(std::move(region)) {}

  // circom::ZKey methods
  uint32_t GetVersion() const override { return 1; }
  ZKey<Curve>* ToV1() override { return this; }
  const v1::ZKey<Curve>* ToV1() const override { return this; }

  absl::Status Read(const zkx::base::ReadOnlyBuffer& buffer) override {
    using BaseField = typename G1AffinePoint::BaseField;

    zkx::base::AutoReset<bool> auto_reset(
        &zkx::base::Serde<F>::s_is_in_montgomery, true);
    zkx::base::AutoReset<bool> auto_reset2(
        &zkx::base::Serde<BaseField>::s_is_in_montgomery, true);

    Sections<ZKeySectionType> sections(buffer, &ZKeySectionTypeToString);
    TF_RETURN_IF_ERROR(sections.Read());

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kHeader));
    TF_RETURN_IF_ERROR(header.Read(buffer));

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kHeaderGroth));
    TF_RETURN_IF_ERROR(header_groth.Read(buffer));
    uint32_t num_vars = header_groth.num_vars;
    uint32_t num_public_inputs = header_groth.num_public_inputs;
    uint32_t domain_size = header_groth.domain_size;

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kIC));
    TF_RETURN_IF_ERROR(ic.Read(buffer, num_public_inputs + 1));

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kCoefficients));
    TF_RETURN_IF_ERROR(coefficients.Read(buffer));

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kPointsA1));
    TF_RETURN_IF_ERROR(points_a1.Read(buffer, num_vars));

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kPointsB1));
    TF_RETURN_IF_ERROR(points_b1.Read(buffer, num_vars));

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kPointsB2));
    TF_RETURN_IF_ERROR(points_b2.Read(buffer, num_vars));

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kPointsC1));
    TF_RETURN_IF_ERROR(
        points_c1.Read(buffer, num_vars - num_public_inputs - 1));

    TF_RETURN_IF_ERROR(sections.MoveTo(ZKeySectionType::kPointsH1));
    TF_RETURN_IF_ERROR(points_h1.Read(buffer, domain_size));
    return absl::OkStatus();
  }

  ProvingKey<Curve> GetProvingKey() const override {
    return {
        header_groth.vkey,     ic.commitments,        points_a1.commitments,
        points_b1.commitments, points_b2.commitments, points_c1.commitments,
        points_h1.commitments,
    };
  }

  absl::Span<const Coefficient<F>> GetCoefficients() const override {
    return coefficients.coefficients;
  }

  size_t GetDomainSize() const override { return header_groth.domain_size; }

  size_t GetNumInstanceVariables() const override {
    return header_groth.num_public_inputs + 1;
  }

  size_t GetNumWitnessVariables() const override {
    return header_groth.num_vars - header_groth.num_public_inputs - 1;
  }
};

}  // namespace v1
}  // namespace rabbitsnark::circom

#endif  // CIRCOM_ZKEY_ZKEY_H_
