#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>
#include <mach/mach_time.h>
#include <vector>

namespace {
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLLibrary> g_library = nil;
id<MTLComputePipelineState> g_matvec = nil;
std::vector<id<MTLBuffer>> g_buffers;           // handle = index into this table

void set_error(char* err, int errlen, NSString* msg) {
    if (err && errlen > 0) std::strncpy(err, msg.UTF8String, errlen - 1), err[errlen - 1] = '\0';
}
}  // namespace

extern "C" {

// Create the device and queue, compile the MSL source, and build the matvec pipeline. Returns 0 on success.
int mr_init(const char* msl_source, const char* kernel_name, char* err, int errlen) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) { set_error(err, errlen, @"no Metal device"); return 1; }
        g_queue = [g_device newCommandQueue];
        NSError* e = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        g_library = [g_device newLibraryWithSource:[NSString stringWithUTF8String:msl_source] options:opts error:&e];
        if (!g_library) { set_error(err, errlen, e.localizedDescription); return 2; }
        id<MTLFunction> fn = [g_library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        if (!fn) { set_error(err, errlen, @"kernel not found in library"); return 3; }
        g_matvec = [g_device newComputePipelineStateWithFunction:fn error:&e];
        if (!g_matvec) { set_error(err, errlen, e.localizedDescription); return 4; }
        return 0;
    }
}

const char* mr_device_name() { return g_device ? g_device.name.UTF8String : "none"; }

// Copy host data into a new GPU-visible buffer (shared storage: CPU and GPU see the same unified memory).
int mr_upload(const void* data, uint64_t nbytes) {
    id<MTLBuffer> buf = [g_device newBufferWithBytes:data length:nbytes options:MTLResourceStorageModeShared];
    g_buffers.push_back(buf);
    return (int)g_buffers.size() - 1;
}

int mr_alloc(uint64_t nbytes) {
    id<MTLBuffer> buf = [g_device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
    g_buffers.push_back(buf);
    return (int)g_buffers.size() - 1;
}

void* mr_contents(int handle) { return g_buffers[handle].contents; }

// y = W . x for W [N, K] bf16, x/y fp32 buffers. Encodes `iters` dispatches in ONE command buffer (commit once,
// wait once), cycling through the `n_w` weight buffers in `ws` so that, with enough distinct copies, every read
// comes from DRAM rather than the on-chip cache (decode reads each weight once per token). Returns GPU seconds
// per dispatch (GPU timestamps: excludes CPU encode/commit time).
double mr_matvec_bf16_rotate(const int* ws, int n_w, int x, int y, uint32_t N, uint32_t K, int iters) {
    @autoreleasepool {
        uint32_t has_bias = 0, has_res = 0;
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_matvec];
        for (int i = 0; i < iters; ++i) {
            int w = ws[i % n_w];
            [enc setBuffer:g_buffers[y] offset:0 atIndex:0];
            [enc setBuffer:g_buffers[w] offset:0 atIndex:1];
            [enc setBuffer:g_buffers[x] offset:0 atIndex:2];
            [enc setBuffer:g_buffers[w] offset:0 atIndex:3];   // bias unused (has_bias = 0)
            [enc setBuffer:g_buffers[x] offset:0 atIndex:4];   // residual unused (has_res = 0)
            [enc setBytes:&K length:4 atIndex:5];
            [enc setBytes:&N length:4 atIndex:6];
            [enc setBytes:&has_bias length:4 atIndex:7];
            [enc setBytes:&has_res length:4 atIndex:8];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)N * 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) / iters;
    }
}

double mr_matvec_bf16(int w, int x, int y, uint32_t N, uint32_t K, int iters) {
    return mr_matvec_bf16_rotate(&w, 1, x, y, N, K, iters);
}

void mr_shutdown() { g_buffers.clear(); g_matvec = nil; g_library = nil; g_queue = nil; g_device = nil; }

}  // extern "C"
