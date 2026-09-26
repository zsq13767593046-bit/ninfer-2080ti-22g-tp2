// ninfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
#include "ops/launcher/gqa_attention.h"

#include "ops/launcher/gqa_attention_decode_launch.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/launcher/gqa_geometry_dispatch.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

PagedKVBatchLayerView single_row_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
    };
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 || (cache_dtype != DType::BF16 && cache_dtype != DType::I8) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("gqa_attention split capacity: invalid profile");
    }
    return dispatch_gqa_geometry(q_heads, [&]<typename Geometry>() {
        return gqa_small_t_launch_capacity<Geometry>(envelope, tokens, cache_dtype);
    });
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    const GqaAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                               static_cast<const __nv_bfloat16*>(v.data)};
    const GqaSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    dispatch_gqa_geometry(q.ne[1], [&]<typename Geometry>() {
        gqa_attention_small_t_launch_for<Geometry>(q, input, pos, scale, cache, invocation,
                                                   envelope, partial_acc, partial_m, partial_l, out,
                                                   stream);
    });
}

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream) {
    const GqaCachedInput input{};
    const GqaSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    dispatch_gqa_geometry(q.ne[1], [&]<typename Geometry>() {
        gqa_attention_small_t_launch_for<Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                   envelope, partial_acc, partial_m, partial_l, out,
                                                   stream);
    });
}

// Whole-model geometries live in this translation unit; the TP2 head-local geometry's launcher
// routes are instantiated in gqa_attention_decode_tp2.cu to keep per-TU ptxas modules bounded.
NINFER_GQA_DECODE_INSTANTIATE(Gqa27Geometry)
NINFER_GQA_DECODE_INSTANTIATE(Gqa35Geometry)

} // namespace ninfer::ops::detail
