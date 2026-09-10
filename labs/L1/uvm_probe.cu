// L1.4 lab · 统一内存（Unified Memory）的缺页代价。
//
// cudaMallocManaged 分配的内存 CPU 和 GPU 都能直接访问，
// 底层靠**按页迁移**实现：谁访问，页就迁到谁那边。迁移由缺页中断触发。
//
// 这很方便（可以超额订阅显存），但每次迁移都要走驱动的缺页处理路径。
// 本实验量化这个代价，并与显式 cudaMemcpy 对照。
//
//   nvcc -O3 -arch=sm_120 -o uvm_probe uvm_probe.cu && ./uvm_probe

#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("  [CUDA 错误] %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); } } while(0)

__global__ void touch(float* p, size_t n, float* sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0;
    for (; i < n; i += stride) acc += p[i];
    if (acc == 1234.5f) *sink = acc;
}

static float time_kernel(float* p, size_t n, float* sink, int sms) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    touch<<<sms * 4, 256>>>(p, n, sink);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const size_t MB = 256, nbytes = MB << 20, n = nbytes / sizeof(float);
    float* sink; CK(cudaMalloc(&sink, sizeof(float)));
    printf("=== %s   缓冲 %zu MB\n\n", prop.name, MB);

    // ---------- 1. 托管内存：host first-touch 之后让 GPU 访问 ----------
    float* um = nullptr;
    CK(cudaMallocManaged(&um, nbytes));
    for (size_t i = 0; i < n; ++i) um[i] = 1.0f;      // CPU 写，页落在 host
    CK(cudaDeviceSynchronize());

    float t_first = time_kernel(um, n, sink, prop.multiProcessorCount);
    float t_second = time_kernel(um, n, sink, prop.multiProcessorCount);
    printf("[1] 托管内存，CPU 先写过\n");
    printf("    GPU 首次访问（含缺页迁移）  %8.3f ms  -> %7.1f GB/s\n",
           t_first, nbytes / (t_first * 1e-3) / 1e9);
    printf("    GPU 再次访问（页已在显存）  %8.3f ms  -> %7.1f GB/s\n",
           t_second, nbytes / (t_second * 1e-3) / 1e9);
    printf("    缺页迁移让首次访问慢了 %.1f 倍\n\n", t_first / t_second);

    // ---------- 2. 显式 prefetch ----------
    for (size_t i = 0; i < n; ++i) um[i] = 2.0f;      // 再次把页拉回 host
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    // CUDA 13 改了签名：设备参数从 int 变成 cudaMemLocation 结构体。
    // 这正是本实验第一版用 ctypes 调它时静默失败的原因——
    // 参数类型不匹配，返回错误码被忽略，测出 9923 GB/s 的荒谬数字。
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = 0;
    cudaError_t pf = cudaMemPrefetchAsync(um, nbytes, loc, 0, 0);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float t_pf = 0; cudaEventElapsedTime(&t_pf, a, b);
    printf("[2] cudaMemPrefetchAsync 显式预取\n");
    if (pf != cudaSuccess) {
        printf("    返回 %s —— 本平台不支持或签名已变\n", cudaGetErrorString(pf));
    } else {
        printf("    预取 %zu MB  %8.3f ms  -> %7.1f GB/s\n",
               MB, t_pf, nbytes / (t_pf * 1e-3) / 1e9);
        float t_after = time_kernel(um, n, sink, prop.multiProcessorCount);
        printf("    预取后 GPU 访问          %8.3f ms  （已无缺页）\n", t_after);
    }

    // ---------- 3. 对照：普通显存 + 显式 H2D ----------
    float* dev; CK(cudaMalloc(&dev, nbytes));
    float* host; CK(cudaMallocHost(&host, nbytes));           // pinned
    for (size_t i = 0; i < n; ++i) host[i] = 3.0f;
    cudaEventRecord(a);
    CK(cudaMemcpy(dev, host, nbytes, cudaMemcpyHostToDevice));
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float t_cp = 0; cudaEventElapsedTime(&t_cp, a, b);
    float t_dev = time_kernel(dev, n, sink, prop.multiProcessorCount);
    printf("\n[3] 对照：cudaMalloc + pinned H2D\n");
    printf("    显式 H2D 拷贝            %8.3f ms  -> %7.1f GB/s\n",
           t_cp, nbytes / (t_cp * 1e-3) / 1e9);
    printf("    之后 GPU 访问            %8.3f ms  -> %7.1f GB/s\n",
           t_dev, nbytes / (t_dev * 1e-3) / 1e9);

    printf("\n结论：UVM 的价值是**编程方便与显存超额订阅**，不是速度。\n");
    printf("      热路径上要可预测的性能，就用显式拷贝 + pinned 缓冲。\n");
    cudaFree(um); cudaFree(dev); cudaFreeHost(host); cudaFree(sink);
    return 0;
}
