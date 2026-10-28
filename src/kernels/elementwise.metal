#include <metal_stdlib>
using namespace metal;

float silu(float g);                                  // defined in matvec.metal

// SwiGLU core over the fused projection, rows of 2F = [gate | up]: out[r, c] = silu(gate_up[r, c]) * gate_up[r, F + c].
// One thread per output; output i is row i / F, so its gate sits at i + (i / F) * F.
kernel void silu_mul(device float* out           [[buffer(0)]],
                     device const float* gate_up [[buffer(1)]],
                     constant uint& F            [[buffer(2)]],
                     constant uint& n            [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint g = i + i / F * F;
    out[i] = silu(gate_up[g]) * gate_up[g + F];
}
