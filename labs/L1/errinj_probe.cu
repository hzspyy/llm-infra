// L1.4 lab · 错误注入探针：非法访问、错误 launch、版本/架构不兼容
//
// 四个用例，每个都在**独立子进程**里跑（由 error_injection.py 调度）：
//   oob        越界写显存：launch 本身返回成功，错误在同步时才暴露（sticky）
//   badlaunch  block dim 超上限：launch 立刻返回错误，context 仍然可用
//   ptx        用指定 .version 的 PTX 让驱动 JIT，观察驱动的 PTX 版本天花板
//   cubin      加载为别的架构（sm_89）编译的 cubin
//
// 编译（crater 上 nvcc 来自 serve venv 的 wheel 工具链）：
//     nvcc -O2 -arch=sm_120 -o errinj_probe errinj_probe.cu -lcuda -lcudart
//
// 用法：
//     ./errinj_probe oob --sync
//     ./errinj_probe oob --nosync
//     ./errinj_probe badlaunch
//     ./errinj_probe ptx /path/to/mod.ptx
//     ./errinj_probe cubin /path/to/wrong_arch.cubin
//
// 退出码约定：0 = 这一步没有报错；1 = 捕获到错误但进程能正常退出；
//             2 = 错误已经污染 context，后续调用全部失败。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const char *cuda_err(cudaError_t e) { return cudaGetErrorName(e); }

static const char *drv_err(CUresult r) {
    const char *s = "?";
    cuGetErrorName(r, &s);
    return s;
}

// 越界写：idx 故意超出分配范围。
// 偏移必须**足够大**：第一版只用 +4M 元素（16 MB），结果没有报错——
// 因为驱动给这块 1 MB 分配的附近还有大量已映射的设备内存（VMM 池），
// 写到那里只是踩到了别人的内存，并不会触发非法地址。要让它越界到
// 未映射的地址空间，偏移得远大于显存容量。这里取 2^33 个 float = 32 GB。
__global__ void oob_kernel(float *p, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i + (1LL << 33)] = 1.0f;   // 越界 32 GB
    }
}

__global__ void benign_kernel(float *p) { p[threadIdx.x] = 2.0f; }

static int case_oob(bool do_sync) {
    float *d = nullptr;
    cudaError_t e = cudaMalloc(&d, 1 << 20);          // 1 MB
    printf("cudaMalloc: %s\n", cuda_err(e));
    oob_kernel<<<4, 256>>>(d, 1024);
    e = cudaGetLastError();
    printf("launch 之后 cudaGetLastError: %s%s\n", cuda_err(e),
           (e == cudaSuccess) ? "   <- 越界在 launch 时看不出来" : "");
    if (do_sync) {
        e = cudaDeviceSynchronize();
        printf("cudaDeviceSynchronize: %s%s\n", cuda_err(e),
               (e != cudaSuccess) ? "   <- 错误在这里才暴露" : "");
    } else {
        printf("（不做同步，直接进下一步）\n");
    }
    // 再跑一个正常 kernel：如果 context 已被污染，这里也会失败
    float *t = nullptr;
    e = cudaMalloc(&t, 4096);
    printf("后续 cudaMalloc: %s\n", cuda_err(e));
    benign_kernel<<<1, 32>>>(d ? d : t);
    e = cudaGetLastError();
    printf("后续正常 kernel launch: %s\n", cuda_err(e));
    e = cudaDeviceSynchronize();
    printf("后续 cudaDeviceSynchronize: %s\n", cuda_err(e));
    cudaError_t r = cudaDeviceReset();
    printf("cudaDeviceReset: %s\n", cuda_err(r));
    bool poisoned = (r != cudaSuccess);
    return poisoned ? 2 : 1;
}

static int case_badlaunch() {
    float *d = nullptr;
    cudaMalloc(&d, 4096);
    // block dim 4096 超过 sm_120 的 1024 上限：这是**配置错误**，不是访问错误
    benign_kernel<<<1, 4096>>>(d);
    cudaError_t e = cudaGetLastError();
    printf("非法 block dim(4096) 的 launch: %s\n", cuda_err(e));
    bool caught = (e != cudaSuccess);
    // context 应该还活着：跑一个合法的 kernel 验证「可恢复」
    benign_kernel<<<1, 32>>>(d);
    cudaError_t e2 = cudaGetLastError();
    cudaError_t e3 = cudaDeviceSynchronize();
    printf("随后的合法 launch: %s / sync: %s%s\n", cuda_err(e2), cuda_err(e3),
           (e2 == cudaSuccess && e3 == cudaSuccess) ? "   <- context 仍可用" : "");
    bool ok = (e2 == cudaSuccess && e3 == cudaSuccess);
    cudaFree(d);
    return (caught && ok) ? 1 : 2;
}

static int case_ptx(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("打不开 %s\n", path); return 3; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(n + 1);
    size_t got = fread(buf, 1, n, f);
    buf[got] = 0;
    fclose(f);

    if (cuInit(0) != CUDA_SUCCESS) { printf("cuInit 失败\n"); return 2; }
    // 用 runtime 建 context，driver API 直接用当前 context
    float *d = nullptr;
    cudaMalloc(&d, 4096);
    CUmodule mod = nullptr;
    CUresult r = cuModuleLoadDataEx(&mod, buf, 0, nullptr, nullptr);
    printf("cuModuleLoadDataEx(%s): %s\n", path, drv_err(r));
    if (mod) cuModuleUnload(mod);
    cudaFree(d);
    cudaDeviceReset();
    return (r == CUDA_SUCCESS) ? 0 : 1;
}

static int case_cubin(const char *path) {
    if (cuInit(0) != CUDA_SUCCESS) { printf("cuInit 失败\n"); return 2; }
    float *d = nullptr;
    cudaMalloc(&d, 4096);
    CUmodule mod = nullptr;
    CUresult r = cuModuleLoad(&mod, path);
    printf("cuModuleLoad(%s): %s\n", path, drv_err(r));
    if (mod) cuModuleUnload(mod);
    cudaFree(d);
    cudaDeviceReset();
    return (r == CUDA_SUCCESS) ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("用法: %s <oob|badlaunch|ptx|cubin> [args]\n", argv[0]); return 3; }
    const char *mode = argv[1];
    printf("PID %d  case=%s\n", (int)getpid(), mode);

    if (!strcmp(mode, "oob")) {
        bool sync = (argc > 2 && !strcmp(argv[2], "--sync"));
        return case_oob(sync);
    }
    if (!strcmp(mode, "badlaunch")) return case_badlaunch();
    if (!strcmp(mode, "ptx")) return case_ptx(argc > 2 ? argv[2] : "mod.ptx");
    if (!strcmp(mode, "cubin")) return case_cubin(argc > 2 ? argv[2] : "wrong.cubin");
    printf("未知 case\n");
    return 3;
}
