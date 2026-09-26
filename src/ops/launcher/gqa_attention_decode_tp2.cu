// ninfer::ops - split-KV GQA small-T launcher: TP2 head-local geometry instantiation.
//
// Gqa27Tp2Geometry (12 Q / 2 KV heads per device) is instantiated in its own translation unit so
// this TU's ptxas module stays bounded on sm_75; see gqa_attention_decode_launch.cuh.
#include "ops/launcher/gqa_attention_decode_launch.cuh"

namespace ninfer::ops::detail {

NINFER_GQA_DECODE_INSTANTIATE(Gqa27Tp2Geometry)

} // namespace ninfer::ops::detail
