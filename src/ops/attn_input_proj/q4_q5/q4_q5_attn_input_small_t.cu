#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The launchers below are compile-time-exact to the fused Q|K|Gate|V row layout they serve. The
// tp1 parent carries ParentRows=7168 with the Q/K split at SplitRow=6144; the tp2 column shard
// carries the head-local half, ParentRows=3584 with SplitRow=3072. One template, two
// instantiations -- the kernel bodies are unchanged between them.
template <std::int32_t ParentRows, std::int32_t SplitRow>
struct AttnShape {
    static constexpr std::int32_t kParentRows = ParentRows;
    static constexpr std::int32_t kSplitRow   = SplitRow;
    static constexpr std::int32_t kHidden     = 5120;
};

using AttnParentShape = AttnShape<7168, 6144>;
using AttnShardShape  = AttnShape<3584, 3072>;

using Q4AttnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4AttnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <class Shape>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(Shape::kParentRows, Schedule::kRowsPerCta)), 1u,
                    1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, Shape::kSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), Shape::kParentRows, Shape::kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Shape, class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(Shape::kParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, Shape::kSplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(key.data), q.ne[0], key.ne[0], Shape::kParentRows,
            Shape::kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Shape, class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    const bool full = (Shape::kParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((Shape::kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) ==
                          0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Shape, Schedule, true>(x, weight, q, key, stream);
    } else {
        launch_q4_simt<Shape, Schedule, false>(x, weight, q, key, stream);
    }
}

template <class Shape>
void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
               cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv<Shape>(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
        launch_q4_simt_route<Shape, Q4AttnSimtR8C4Schedule>(x, weight, q, key, stream);
        return;
    case 8:
    case 16:
        launch_q4_simt_route<Shape, Q4AttnSimtR8C8Schedule>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 split-output requires T in [1,16]");
    }
}

template <class Shape>
void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = Shape::kParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<Shape::kParentRows, Shape::kHidden, kRowsPerBlock, 2, true, false,
                            true, Shape::kSplitRow>
        <<<kGrid, kBlockThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              static_cast<const std::uint8_t*>(weight.qdata),
                                              static_cast<const std::uint8_t*>(weight.qhigh),
                                              static_cast<const std::uint8_t*>(weight.scales),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Shape, int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(Shape::kParentRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, Shape::kHidden, true,
                                        Shape::kSplitRow>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(gate.data),
                                        static_cast<__nv_bfloat16*>(value.data), Shape::kParentRows,
                                        gate.ne[0], Shape::kHidden, Cols, weight.padded_shape[1],
                                        5);
    CUDA_CHECK(cudaGetLastError());
}

template <class Shape>
void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<Shape, 2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<Shape, 3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<Shape, 4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<Shape, 5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<Shape, 6>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,6]");
    }
}

template <class Shape, int ColsPerTile>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(Shape::kParentRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages, true,
                                 Shape::kSplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), Shape::kParentRows, gate.ne[0], Shape::kHidden,
        cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

template <class Shape>
void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv<Shape>(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact<Shape>(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q5_simt<Shape, 4>(x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,16]");
}

template <class Shape>
void small_t_launch_for(const Tensor& x, const Weight& query_key_weight,
                        const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                        Tensor& v, cudaStream_t stream) {
    launch_q4<Shape>(x, query_key_weight, q, k, stream);
    launch_q5<Shape>(x, gate_value_weight, gate, v, stream);
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    small_t_launch_for<AttnParentShape>(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                         stream);
}

// The tp2 column-shard sibling (ParentRows=3584, SplitRow=3072): the same launcher family
// instantiated at the halved head-local extents.
void q4_q5_attn_input_small_t_shard_launch(const Tensor& x, const Weight& query_key_weight,
                                           const Weight& gate_value_weight, Tensor& q,
                                           Tensor& gate, Tensor& k, Tensor& v,
                                           cudaStream_t stream) {
    small_t_launch_for<AttnShardShape>(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
}

} // namespace ninfer::ops::detail
