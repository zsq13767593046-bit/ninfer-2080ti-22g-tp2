#include "ops/common/mma.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 32;

void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

// Pack represented signed codes using the PTX m16n8k32 operand-fragment mapping.
// A: a0/a2 are the low/high K halves of rows 0..7, and a1/a3 of rows 8..15.
// B: b0/b1 are the low/high K halves for the lane's output column.
__global__ void run_mma(const std::int8_t* a, const std::int8_t* b, std::int32_t* output) {
    const int lane = threadIdx.x;
    const int row = lane >> 2;
    const int group_lane = lane & 3;
    unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
    for (int i = 0; i < 4; ++i) {
        const int low_k = group_lane * 4 + i;
        const unsigned shift = static_cast<unsigned>(i * 8);
        a0 |= static_cast<unsigned>(static_cast<std::uint8_t>(a[row * kK + low_k])) << shift;
        a1 |= static_cast<unsigned>(static_cast<std::uint8_t>(a[(row + 8) * kK + low_k])) << shift;
        a2 |= static_cast<unsigned>(static_cast<std::uint8_t>(a[row * kK + low_k + 16])) << shift;
        a3 |= static_cast<unsigned>(static_cast<std::uint8_t>(a[(row + 8) * kK + low_k + 16])) << shift;
        b0 |= static_cast<unsigned>(static_cast<std::uint8_t>(b[low_k * kN + row])) << shift;
        b1 |= static_cast<unsigned>(static_cast<std::uint8_t>(b[(low_k + 16) * kN + row])) << shift;
    }

    int c0 = -7, c1 = 11, c2 = 19, c3 = -23;
    ninfer::ops::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    output[lane * 4] = c0;
    output[lane * 4 + 1] = c1;
    output[lane * 4 + 2] = c2;
    output[lane * 4 + 3] = c3;
}

} // namespace

namespace {

constexpr int kF16M = 16;
constexpr int kF16N = 8;
constexpr int kF16K = 16;

__device__ unsigned pack_halves(float lo, float hi) {
    const __half2 pair = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&pair);
}

// Fragment packing follows the PTX m16n8k16 f16 A/B layout, which is also what
// ldmatrix.x4 with this codebase's lane addressing produces:
// a0/a1 = rows 0-7/8-15 of k 0-7, a2/a3 = rows 0-7/8-15 of k 8-15,
// b0/b1 = B rows 0-7/8-15 (k) for the lane's output column.
__global__ void run_mma_f16(const __half* a, const __half* b, float* output) {
    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const unsigned a0 = pack_halves(__half2float(a[g * kF16K + 2 * t]),
                                    __half2float(a[g * kF16K + 2 * t + 1]));
    const unsigned a1 = pack_halves(__half2float(a[(g + 8) * kF16K + 2 * t]),
                                    __half2float(a[(g + 8) * kF16K + 2 * t + 1]));
    const unsigned a2 = pack_halves(__half2float(a[g * kF16K + 8 + 2 * t]),
                                    __half2float(a[g * kF16K + 8 + 2 * t + 1]));
    const unsigned a3 = pack_halves(__half2float(a[(g + 8) * kF16K + 8 + 2 * t]),
                                    __half2float(a[(g + 8) * kF16K + 8 + 2 * t + 1]));
    const unsigned b0 = pack_halves(__half2float(b[(2 * t) * kF16N + g]),
                                    __half2float(b[(2 * t + 1) * kF16N + g]));
    const unsigned b1 = pack_halves(__half2float(b[(8 + 2 * t) * kF16N + g]),
                                    __half2float(b[(8 + 2 * t + 1) * kF16N + g]));

    float c0 = -3.5f, c1 = 2.25f, c2 = -1.75f, c3 = 4.5f;
    ninfer::ops::mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    output[lane * 4]     = c0;
    output[lane * 4 + 1] = c1;
    output[lane * 4 + 2] = c2;
    output[lane * 4 + 3] = c3;
}

int run_f16_case() {
    std::array<__half, kF16M * kF16K> a{};
    std::array<__half, kF16K * kF16N> b{};
    for (int m = 0; m < kF16M; ++m) {
        for (int k = 0; k < kF16K; ++k) {
            a[m * kF16K + k] = __float2half_rn(static_cast<float>(((m * 13 + k * 7) % 7) - 3));
        }
    }
    for (int k = 0; k < kF16K; ++k) {
        for (int n = 0; n < kF16N; ++n) {
            b[k * kF16N + n] = __float2half_rn(static_cast<float>(((k * 11 + n * 5) % 9) - 4));
        }
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_c  = nullptr;
    check(cudaMalloc(&device_a, a.size() * sizeof(__half)));
    check(cudaMalloc(&device_b, b.size() * sizeof(__half)));
    check(cudaMalloc(&device_c, 32 * 4 * sizeof(float)));
    check(cudaMemcpy(device_a, a.data(), a.size() * sizeof(__half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice));
    run_mma_f16<<<1, 32>>>(device_a, device_b, device_c);
    check(cudaGetLastError());
    std::array<float, 32 * 4> actual{};
    check(cudaMemcpy(actual.data(), device_c, actual.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    check(cudaFree(device_c));
    check(cudaFree(device_b));
    check(cudaFree(device_a));

    constexpr std::array<float, 4> initial{-3.5f, 2.25f, -1.75f, 4.5f};
    int failures = 0;
    for (int m = 0; m < kF16M; ++m) {
        for (int n = 0; n < kF16N; ++n) {
            const int lane           = (m % 8) * 4 + n / 2;
            const int register_index = (m / 8) * 2 + n % 2;
            float expected           = initial[register_index];
            for (int k = 0; k < kF16K; ++k) {
                expected += __half2float(a[m * kF16K + k]) * __half2float(b[k * kF16N + n]);
            }
            const float got = actual[lane * 4 + register_index];
            if (got != expected) {
                if (++failures <= 8) {
                    std::cerr << "f16 row=" << m << " col=" << n << " actual=" << got
                              << " expected=" << expected << '\n';
                }
            }
        }
    }
    if (failures) { std::cerr << "mma_f16 errors=" << failures << '\n'; }
    return failures;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }

    int failures = run_f16_case();
    std::array<std::int8_t, kM * kK> a{};
    std::array<std::int8_t, kK * kN> b{};
    for (int row = 0; row < kM; ++row) {
        for (int k = 0; k < kK; ++k) {
            a[row * kK + k] = static_cast<std::int8_t>(((row * 17 + k * 13) % 25) - 12);
        }
    }
    for (int k = 0; k < kK; ++k) {
        for (int col = 0; col < kN; ++col) {
            b[k * kN + col] = static_cast<std::int8_t>(((k * 19 + col * 11) % 23) - 11);
        }
    }

    std::int8_t *device_a = nullptr, *device_b = nullptr;
    std::int32_t* device_c = nullptr;
    check(cudaMalloc(&device_a, a.size()));
    check(cudaMalloc(&device_b, b.size()));
    check(cudaMalloc(&device_c, 32 * 4 * sizeof(std::int32_t)));
    check(cudaMemcpy(device_a, a.data(), a.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b.data(), b.size(), cudaMemcpyHostToDevice));
    run_mma<<<1, 32>>>(device_a, device_b, device_c);
    check(cudaGetLastError());
    std::array<std::int32_t, 32 * 4> actual{};
    check(cudaMemcpy(actual.data(), device_c, actual.size() * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost));
    check(cudaFree(device_c));
    check(cudaFree(device_b));
    check(cudaFree(device_a));

    int s8_failures = 0;
    constexpr std::array<int, 4> initial{-7, 11, 19, -23};
    for (int row = 0; row < kM; ++row) {
        for (int col = 0; col < kN; ++col) {
            const int lane = (row % 8) * 4 + col / 2;
            const int register_index = (row / 8) * 2 + col % 2;
            int expected = initial[register_index];
            for (int k = 0; k < kK; ++k) {
                expected += static_cast<int>(a[row * kK + k]) *
                            static_cast<int>(b[k * kN + col]);
            }
            const int got = actual[lane * 4 + register_index];
            if (got != expected) {
                if (++s8_failures <= 8) {
                    std::cerr << "row=" << row << " col=" << col << " actual=" << got
                              << " expected=" << expected << '\n';
                }
            }
        }
    }
    if (s8_failures) { std::cerr << "mma_s8 errors=" << s8_failures << '\n'; }
    failures += s8_failures;
    return failures == 0 ? 0 : 1;
}
