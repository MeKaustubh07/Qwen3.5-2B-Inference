#include <metal_stdlib>
using namespace metal;

// Decode-time linear layer: y[N] = W[N, K] (bf16) . x[K] (fp32) (+ bias) (+ residual). fp32 accumulation.
// One 32-thread SIMD group per output row: lanes read consecutive weights (coalesced), each lane
// accumulates a strided slice of the dot product, and simd_sum combines the 32 partial sums.
// Weights are read 4 at a time as a packed bfloat4 when K is a multiple of 128.
kernel void matvec_bf16(device float* y          [[buffer(0)]],
                        device const bfloat* W   [[buffer(1)]],
                        device const float* x    [[buffer(2)]],
                        device const bfloat* b   [[buffer(3)]],
                        device const float* res  [[buffer(4)]],
                        constant uint& K         [[buffer(5)]],
                        constant uint& N         [[buffer(6)]],
                        constant uint& has_bias  [[buffer(7)]],
                        constant uint& has_res   [[buffer(8)]],
                        uint gid  [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= N) return;
    device const bfloat* wr = W + (ulong)row * K;
    float acc = 0.0f;
    if ((K & 127) == 0) {
        device const bfloat4* w4 = (device const bfloat4*)wr;
        device const float4* x4 = (device const float4*)x;
        for (uint j = lane; j < K / 4; j += 32) acc += dot(float4(w4[j]), x4[j]);
    } else {
        for (uint j = lane; j < K; j += 32) acc += float(wr[j]) * x[j];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc + (has_bias ? float(b[row]) : 0.0f) + (has_res ? res[row] : 0.0f);
}

constant uint MAX_BATCH = 8;                          // also used by qmatvec.metal (all files compile as one source)

// Decode matvec for M <= 8 activation rows, R = 2 output rows per SIMD group. For each step a lane dequantizes
// its 8 weights of all R rows once, then loads each x value once and uses it for all R rows: x is read R times
// less often than with one row per SIMD group (with M = 8, x traffic, not weight traffic, was the bottleneck).
// Loops over M and R are fully unrolled with guards, so accumulators stay in registers. Used for every M >= 2
// (measured on Qwen3.5-2B shapes: 2 rows beat 4 rows and 1 row per SIMD group from M = 2 to 8). K % 4 == 0.
constant uint RR = 2;                           // output rows per SIMD group

#define ROWS_EPILOGUE                                                                             \
    for (uint r = 0; r < RR; r++) {                                                               \
        uint row = row0 + r;                                                                      \
        for (uint m = 0; m < MAX_BATCH; m++) {                                                           \
            if (m < M) {                                                                          \
                float a = simd_sum(acc[r][m]);                                                    \
                if (lane == 0 && row < N)                                                         \
                    y[m * N + row] = a + (has_bias ? float(b[row]) : 0.0f) + (has_res ? res[m * N + row] : 0.0f); \
            }                                                                                     \
        }                                                                                         \
    }

kernel void matvec_bf16_rows(device float* y [[buffer(0)]], device const bfloat4* W [[buffer(1)]],
                             device const float4* x [[buffer(2)]], device const bfloat* b [[buffer(3)]],
                             device const float* res [[buffer(4)]], constant uint& K [[buffer(5)]],
                             constant uint& N [[buffer(6)]], constant uint& has_bias [[buffer(7)]],
                             constant uint& has_res [[buffer(8)]], constant uint& M [[buffer(9)]],
                             uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * RR;
    if (row0 >= N) return;
    float acc[RR][MAX_BATCH];
    for (uint r = 0; r < RR; r++) for (uint m = 0; m < MAX_BATCH; m++) acc[r][m] = 0.0f;
    for (uint j = lane; j < K / 4; j += 32) {
        float4 w[RR];
        for (uint r = 0; r < RR; r++) w[r] = float4(W[(ulong)min(row0 + r, N - 1) * (K / 4) + j]);
        for (uint m = 0; m < MAX_BATCH; m++) {
            if (m < M) {
                float4 xv = x[m * (K / 4) + j];
                for (uint r = 0; r < RR; r++) acc[r][m] += dot(w[r], xv);
            }
        }
    }
    ROWS_EPILOGUE
}

// SiLU, in the operation order of torch's CPU kernel so the fp32 results track the reference. Defined here because
// the native runtime compiles this file alone; elementwise.metal declares it (all files compile as one source).
float silu(float g) { return g / (1.0f + precise::exp(-g)); }

// Fused GLU for decode: Wgu = [gate; up] stacked, shape [2F, K]. Output row i reads gate row i and up row F+i in
// the same pass and writes act(gate) * up directly (one dispatch instead of three). K % 4 == 0.
inline float2 glu_dots(device const bfloat* Wgu, device const float* x, uint K, uint F, uint row, uint lane) {
    device const bfloat4* g4 = (device const bfloat4*)(Wgu + (ulong)row * K);
    device const bfloat4* u4 = (device const bfloat4*)(Wgu + (ulong)(row + F) * K);
    device const float4* x4 = (device const float4*)x;
    float g = 0.0f, u = 0.0f;
    for (uint j = lane; j < K / 4; j += 32) { float4 xv = x4[j]; g += dot(float4(g4[j]), xv); u += dot(float4(u4[j]), xv); }
    return float2(simd_sum(g), simd_sum(u));
}

kernel void matvec_swiglu_bf16(device float* out        [[buffer(0)]],
                               device const bfloat* Wgu [[buffer(1)]],
                               device const float* x    [[buffer(2)]],
                               constant uint& K         [[buffer(3)]],
                               constant uint& F         [[buffer(4)]],
                               uint gid  [[thread_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= F) return;
    float2 gu = glu_dots(Wgu, x, K, F, row, lane);
    if (lane == 0) out[row] = silu(gu.x) * gu.y;
}
