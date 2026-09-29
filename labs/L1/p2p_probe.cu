// L1.3 lab · P2P 直连到底通没通：驱动层探针
//
// 为什么需要它：在 torch 里做跨卡 copy_ 时，本机观察到「同一个卡对，
// 有时 2.97 ms/64MB（≈22 GB/s，明显是 PCIe 直连），有时 98.8 ms 且与
// 数据量无关」这两种结果会互相翻转。要判断这是驱动/P2P 的问题还是
// 上层框架的问题，必须在驱动层做一次受控测量：
//
//   1. cudaDeviceCanAccessPeer 的实际返回值
//   2. cudaDeviceEnablePeerAccess 的返回码（不是「能不能」而是「开没开」）
//   3. 显式开启 P2P 之后，cudaMemcpyPeerAsync 的分尺寸吞吐
//   4. 不开 P2P 时的对照（同一 API 走主机中转），以及逐次耗时
//
// 编译：
//   nvcc -O2 -o p2p_probe p2p_probe.cu -lcuda -lcudart
//
// 用法：
//   ./p2p_probe 0 1            # 只测 0->1
//   ./p2p_probe --matrix       # 全部卡对
//   ./p2p_probe 0 3 --no-enable

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("  [cuda error] %s:%d %s -> %s\n", __FILE__, __LINE__, #x, \
           cudaGetErrorName(_e)); } } while (0)

static double median(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void bench(int src, int dst, size_t bytes, bool enable, int iters = 20) {
    cudaSetDevice(src);
    void *a = nullptr, *b = nullptr;
    CK(cudaMalloc(&a, bytes));
    cudaSetDevice(dst);
    CK(cudaMalloc(&b, bytes));
    cudaSetDevice(src);
    CK(cudaMemset(a, 1, bytes));
    CK(cudaDeviceSynchronize());

    int can = 0;
    CK(cudaDeviceCanAccessPeer(&can, src, dst));
    const char *enable_res = "not-requested";
    if (enable && can) {
        cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
        enable_res = cudaGetErrorName(e);      // cudaSuccess / cudaErrorPeerAccessAlreadyEnabled
    }

    std::vector<double> ts;
    for (int i = 0; i < iters; i++) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, 0);
        cudaMemcpyPeerAsync(b, dst, a, src, bytes, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ts.push_back(ms);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    double med = median(ts);
    printf("  %d->%d  %6zu KiB  canPeer=%d enable=%s  中位 %8.3f ms  (%6.2f GB/s)"
           "  首次 %8.3f ms  最小 %8.3f ms\n",
           src, dst, bytes / 1024, can, enable_res, med,
           bytes / (med * 1e-3) / 1e9, ts.front(), *std::min_element(ts.begin(), ts.end()));
    cudaFree(a);
    cudaSetDevice(dst);
    cudaFree(b);
}

int main(int argc, char **argv) {
    int ngpu = 0;
    cudaGetDeviceCount(&ngpu);
    printf("=== %d 张 GPU，P2P 驱动层探针\n", ngpu);
    for (int i = 0; i < ngpu; i++) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, i);
        printf("    GPU%d %s  asyncEngineCount=%d  unifiedAddressing=%d\n",
               i, p.name, p.asyncEngineCount, p.unifiedAddressing);
    }

    std::vector<std::pair<int, int>> pairs;
    if (argc >= 3) {
        pairs.push_back({atoi(argv[1]), atoi(argv[2])});
    } else {
        for (int i = 0; i < ngpu; i++)
            for (int j = 0; j < ngpu; j++)
                if (i != j) pairs.push_back({i, j});
    }
    bool enable = true;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--no-enable")) enable = false;

    printf("\n[显式开启 P2P 后的分尺寸吞吐]%s\n", enable ? "" : "（--no-enable：对照）");
    const size_t sizes_kb[] = {4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144};
    for (auto &pr : pairs) {
        for (size_t kb : sizes_kb) {
            bench(pr.first, pr.second, kb * 1024, enable);
        }
        printf("\n");
    }
    return 0;
}
