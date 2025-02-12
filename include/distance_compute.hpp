#include <cstdint>
#include <immintrin.h>

#include <immintrin.h>

inline uint32_t my_mm512_reduce_add_epi32(__m512i v) {
  __m256i v256 = _mm256_add_epi32(_mm512_extracti64x4_epi64(v, 0),
                                  _mm512_extracti64x4_epi64(v, 1));
  __m128i v128 = _mm_add_epi32(_mm256_extracti128_si256(v256, 0),
                               _mm256_extracti128_si256(v256, 1));
  v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
  v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(v128);
}

template <typename DistDataType = uint8_t, typename DistanceType = uint32_t>
inline auto l2_sqr_sq4_avx512(DistDataType *x, DistDataType *y,
                              size_t dim) -> DistanceType {
  uint32_t sum = 0;
  size_t i = 0;

  __m512i sum_vec = _mm512_setzero_si512();
  const __m512i mask = _mm512_set1_epi8(0x0F);

  for (; i + 64 <= dim; i += 64) {
    __m512i xx = _mm512_loadu_si512(x + i);
    __m512i yy = _mm512_loadu_si512(y + i);

    __m512i xx_lo = _mm512_and_si512(xx, mask);
    __m512i xx_hi = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);
    __m512i yy_lo = _mm512_and_si512(yy, mask);
    __m512i yy_hi = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);

    __m512i diff_lo = _mm512_abs_epi8(_mm512_sub_epi8(xx_lo, yy_lo));
    __m512i diff_hi = _mm512_abs_epi8(_mm512_sub_epi8(xx_hi, yy_hi));

    __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
    __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

    __m256i sq_lo_low = _mm512_castsi512_si256(sq_lo);
    __m256i sq_lo_high = _mm512_extracti64x4_epi64(sq_lo, 1);
    sum_vec = _mm512_add_epi32(sum_vec, _mm512_cvtepu16_epi32(sq_lo_low));
    sum_vec = _mm512_add_epi32(sum_vec, _mm512_cvtepu16_epi32(sq_lo_high));

    __m256i sq_hi_low = _mm512_castsi512_si256(sq_hi);
    __m256i sq_hi_high = _mm512_extracti64x4_epi64(sq_hi, 1);
    sum_vec = _mm512_add_epi32(sum_vec, _mm512_cvtepu16_epi32(sq_hi_low));
    sum_vec = _mm512_add_epi32(sum_vec, _mm512_cvtepu16_epi32(sq_hi_high));
  }

  sum = my_mm512_reduce_add_epi32(sum_vec);

  for (; i < dim; ++i) {
    uint8_t xi = x[i], yi = y[i];
    int diff_lo = (xi & 0x0F) - (yi & 0x0F);
    int diff_hi = (xi >> 4) - (yi >> 4);
    sum += diff_lo * diff_lo + diff_hi * diff_hi;
  }

  return sum;
}

template <typename DistDataType = uint8_t, typename DistanceType = uint32_t>
inline auto l2_sqr_sq4_scalar(DistDataType *x, DistDataType *y,
                              size_t dim) -> DistanceType {
  uint32_t sum = 0;

  for (size_t i = 0; i < dim; ++i) {
    uint32_t xx = x[i] & 15;
    uint32_t yy = y[i] & 15;
    sum += (xx - yy) * (xx - yy);

    xx = x[i] >> 4 & 15;
    yy = y[i] >> 4 & 15;
    sum += (xx - yy) * (xx - yy);
  }

  return sum;
}
