#include "distance_compute.hpp"
#include <cstdint>
#include <gtest/gtest.h>

TEST(L2SqrSQ4Test, AVX512_Correctness) {
  uint8_t x[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
  uint8_t y[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
  size_t dim = 8;

  uint32_t expected = l2_sqr_sq4_scalar(x, y, dim);

  uint32_t result = l2_sqr_sq4_avx512(x, y, dim);

  EXPECT_EQ(result, expected);
}

TEST(L2SqrSQ4Test, AVX512_NonZeroResult) {
  uint8_t x[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
  uint8_t y[] = {0x22, 0x44, 0x66, 0x88, 0xAA, 0xCC, 0xEE, 0x00};
  size_t dim = 8;

  uint32_t expected = l2_sqr_sq4_scalar(x, y, dim);

  uint32_t result = l2_sqr_sq4_avx512(x, y, dim);

  EXPECT_EQ(result, expected);
}

TEST(L2SqrSQ4Test, LargeDimensionAVX512) {
  size_t dim = 1024;
  uint8_t x[dim], y[dim];

  for (size_t i = 0; i < dim; ++i) {
    x[i] = static_cast<uint8_t>(i);
    y[i] = static_cast<uint8_t>(dim - i);
  }

  uint32_t expected = l2_sqr_sq4_scalar(x, y, dim);

  uint32_t result = l2_sqr_sq4_avx512(x, y, dim);

  EXPECT_EQ(result, expected);
}
