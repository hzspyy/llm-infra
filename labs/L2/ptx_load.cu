// L2.2-C · 用 driver API 加载 PTX/cubin，打印真实错误并计时（第一次 vs 之后）。
// 编译：nvcc -O2 -std=c++17 -arch=sm_120 ptx_load.cu -o ptx_load -L/usr/lib/x86_64-linux-gnu -l:libcuda.so.1
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("用法: ptx_load <file.ptx|file.cubin>\n"); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("打不开 %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> buf(n + 1, 0);
    if (fread(buf.data(), 1, n, f) != (size_t)n) { printf("读取失败\n"); return 2; }
    fclose(f);

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { printf("cuInit: %d\n", (int)r); return 1; }
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx = nullptr;
    cuDevicePrimaryCtxRetain(&ctx, dev);      // 避免 cuCtxCreate 的版本化签名差异
    cuCtxSetCurrent(ctx);

    auto t0 = std::chrono::steady_clock::now();
    CUmodule mod = nullptr;
    r = cuModuleLoadData(&mod, buf.data());
    double load_ms = ms_since(t0);
    const char* name = "?"; const char* str = "?";
    cuGetErrorName(r, &name); cuGetErrorString(r, &str);
    printf("file=%s  load=%.3f ms  result=%s\n", argv[1], load_ms, name);
    if (r != CUDA_SUCCESS) { printf("        detail: %s\n", str); return 1; }

    CUfunction fn = nullptr;
    r = cuModuleGetFunction(&fn, mod, "saxpy_kernel");
    if (r != CUDA_SUCCESS) { printf("  cuModuleGetFunction(saxpy_kernel): %s\n", name); return 1; }

    int N = 1 << 20;
    CUdeviceptr dx, dy;
    cuMemAlloc(&dx, N * 4); cuMemAlloc(&dy, N * 4);
    cuMemsetD32(dx, 0x3f800000u, N);      // dx 全 1.0f
    float a = 2.0f; int n_elems = N;
    void* args[] = {&dx, &dy, &n_elems, &a};
    auto t1 = std::chrono::steady_clock::now();
    r = cuLaunchKernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
    cuCtxSynchronize();
    double first_launch = ms_since(t1);
    if (r != CUDA_SUCCESS) { printf("  首次 launch 失败: %d\n", (int)r); return 1; }
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) cuLaunchKernel(fn, (N + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
    cuCtxSynchronize();
    float got = -1.0f;
    cuMemcpyDtoH(&got, dy, 4);
    printf("  首次 launch=%.3f ms  之后 100 次平均=%.4f ms  y[0]=%.1f（期望 3.0）\n",
           first_launch, ms_since(t2) / 100.0, got);

    cuModuleUnload(mod);
    return 0;
}
