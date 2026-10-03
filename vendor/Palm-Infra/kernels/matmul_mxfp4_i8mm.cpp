#include "kernels/matmul_internal.h"

#include <cstddef>
#include <cstdint>

#if HAS_NEON && defined(__ARM_FEATURE_MATMUL_INT8)

namespace {

inline int32x4_t mxfp4_i8mm_dot32_dual_pair(
    const uint8_t* packed0, const uint8_t* packed1,
    const int8_t* q_even, const int8_t* q_odd,
    const int8_t* residual_even, const int8_t* residual_odd) {
    const int8x16_t coefficient_table = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
    };
    const uint8x16_t bytes0 = vld1q_u8(packed0);
    const uint8x16_t bytes1 = vld1q_u8(packed1);
    const int8x16_t weight_even0 = vqtbl1q_s8(
        coefficient_table, vandq_u8(bytes0, vdupq_n_u8(0x0f)));
    const int8x16_t weight_even1 = vqtbl1q_s8(
        coefficient_table, vandq_u8(bytes1, vdupq_n_u8(0x0f)));
    const int8x16_t weight_odd0 = vqtbl1q_s8(
        coefficient_table, vshrq_n_u8(bytes0, 4));
    const int8x16_t weight_odd1 = vqtbl1q_s8(
        coefficient_table, vshrq_n_u8(bytes1, 4));
    int32x4_t dots = vdupq_n_s32(0);
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_even), vld1_s8(residual_even)),
        vcombine_s8(vget_low_s8(weight_even0),
                    vget_low_s8(weight_even1)));
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_even + 8),
                    vld1_s8(residual_even + 8)),
        vcombine_s8(vget_high_s8(weight_even0),
                    vget_high_s8(weight_even1)));
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_odd), vld1_s8(residual_odd)),
        vcombine_s8(vget_low_s8(weight_odd0),
                    vget_low_s8(weight_odd1)));
    dots = vmmlaq_s32(
        dots,
        vcombine_s8(vld1_s8(q_odd + 8),
                    vld1_s8(residual_odd + 8)),
        vcombine_s8(vget_high_s8(weight_odd0),
                    vget_high_s8(weight_odd1)));
    // Native SMMLA result layout is [primary0, primary1,
    // correction0, correction1]. Keep it in registers; the four-row caller
    // combines two pairs without scalar lane extraction or stack traffic.
    return dots;
}

inline float32x4_t decode_e8m0x4(uint32x4_t codes) {
    uint32x4_t bits = vshlq_n_u32(codes, 23);
    bits = vbslq_u32(
        vceqq_u32(codes, vdupq_n_u32(0)),
        vdupq_n_u32(uint32_t{1} << 22), bits);
    bits = vbslq_u32(
        vceqq_u32(codes, vdupq_n_u32(0xff)),
        vdupq_n_u32(0x7fc00000u), bits);
    return vreinterpretq_f32_u32(bits);
}

}  // namespace

void matmul_mxfp4_q8_gemv_i8mm_range(
    const int8_t* q_even, const int8_t* q_odd, const float* a_scales,
    const int8_t* residual_even, const int8_t* residual_odd,
    const float* residual_scales, const uint8_t* weights,
    const uint8_t* scales, float* output, int N, int K,
    int n_begin, int n_end) {
    (void)N;
    constexpr int kBlock = 32;
    const int groups = K / kBlock;
    const int bytes_per_row = K / 2;
    for (int n = n_begin; n + 3 < n_end; n += 4) {
        const uint8_t* weight_rows[4] = {
            weights + static_cast<size_t>(n) * bytes_per_row,
            weights + static_cast<size_t>(n + 1) * bytes_per_row,
            weights + static_cast<size_t>(n + 2) * bytes_per_row,
            weights + static_cast<size_t>(n + 3) * bytes_per_row,
        };
        const uint8_t* scale_rows[4] = {
            scales + static_cast<size_t>(n) * groups,
            scales + static_cast<size_t>(n + 1) * groups,
            scales + static_cast<size_t>(n + 2) * groups,
            scales + static_cast<size_t>(n + 3) * groups,
        };
        float32x4_t sums = vdupq_n_f32(0.0f);
        for (int group = 0; group < groups; ++group) {
            const size_t offset = static_cast<size_t>(group) * 16;
            const int32x4_t pair01 = mxfp4_i8mm_dot32_dual_pair(
                weight_rows[0] + offset, weight_rows[1] + offset,
                q_even + offset, q_odd + offset,
                residual_even + offset, residual_odd + offset);
            const int32x4_t pair23 = mxfp4_i8mm_dot32_dual_pair(
                weight_rows[2] + offset, weight_rows[3] + offset,
                q_even + offset, q_odd + offset,
                residual_even + offset, residual_odd + offset);
            const int32x4_t primary = vcombine_s32(
                vget_low_s32(pair01), vget_low_s32(pair23));
            const int32x4_t correction = vcombine_s32(
                vget_high_s32(pair01), vget_high_s32(pair23));
            const uint32x4_t scale_codes = {
                scale_rows[0][group], scale_rows[1][group],
                scale_rows[2][group], scale_rows[3][group],
            };
            const float32x4_t combined = vmlaq_n_f32(
                vmulq_n_f32(
                    vcvtq_f32_s32(primary),
                    a_scales[group]),
                vcvtq_f32_s32(correction),
                residual_scales[group]);
            sums = vfmaq_f32(
                sums, combined,
                vmulq_n_f32(decode_e8m0x4(scale_codes), 0.5f));
        }
        vst1q_f32(output + n, sums);
    }
}

#endif
