// L2.1 lab · CUDA Graph 的捕获与 replay：哪些变化合法。
//
// 捕获一次，replay 多次。逐个改变输入值、指针、shape、分配，看哪些还能用：
//   C1 捕获/实例化/首次/热 replay 的成本与显存
//   C2 改输入的值（同一块内存）      —— 合法
//   C3 换输入指针                    —— 不合法，图里烧的是旧地址
//   C4 改元素数（shape）             —— 不合法；给两条合法修法
//   C5 释放并重新分配                —— 不合法（`./graph_capture c5` 跑该路径）
//   C6 把 n 放进设备内存             —— 同一张图跑不同 n（动态尺寸的正解）
//   C7 十个 kernel 的链：逐个 launch 与一次 replay 的 CPU/GPU 时间
//
// 编译：
//   nvcc -O2 -std=c++17 -arch=sm_120 -lineinfo -o graph_capture graph_capture.cu
// 运行：
//   ./graph_capture          # 正常路径（C5 只打印地址对比）
//   ./graph_capture c5       # 执行 C5 的 free + replay，配合 compute-sanitizer 用

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); } } while(0)

static const int BLOCK = 128;
static const int MAXN = 1 << 20;
static const int CHAIN = 10;

__global__ void saxpy_kernel(const float* __restrict__ x, float* __restrict__ y,
                             int n, float a) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = a * x[i] + 1.0f;
}

// n 放在设备内存里：grid 固定开满，实际处理多少元素由设备端读到的值决定
__global__ void dyn_kernel(const float* __restrict__ x, float* __restrict__ y,
                           const int* __restrict__ n_dev) {
    int n = *n_dev;
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i] + 1.0f;
}

static double ms_between(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms;
}

static void fill_host(float* p, int n, float base) {
    for (int i = 0; i < n; ++i) p[i] = base + i;
}

static int count_bad(const float* got, const float* want, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i) if (got[i] != want[i]) ++bad;
    return bad;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "";
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s  sm_%d%d  SM=%d ===\n", p.name, p.major, p.minor,
           p.multiProcessorCount);

    float *d_x, *d_y, *d_x2, *d_y2;
    CK(cudaMalloc(&d_x, MAXN * sizeof(float)));
    CK(cudaMalloc(&d_y, MAXN * sizeof(float)));
    CK(cudaMalloc(&d_x2, MAXN * sizeof(float)));
    CK(cudaMalloc(&d_y2, MAXN * sizeof(float)));
    float* h = (float*)malloc(MAXN * sizeof(float));
    float* h_out = (float*)malloc(MAXN * sizeof(float));
    float* h_ref = (float*)malloc(MAXN * sizeof(float));

    const int N0 = 100000;                     // 捕获时固定下来的元素数
    fill_host(h, MAXN, 0.0f);
    CK(cudaMemcpy(d_x, h, MAXN * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x2, h, MAXN * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb);

    // ---------------------------------------------------------------- C1
    printf("\n[C1] 捕获 -> 实例化 -> 首次 -> 热 replay\n");
    size_t free_before = 0, total = 0;
    CK(cudaMemGetInfo(&free_before, &total));

    cudaGraph_t graph; cudaGraphExec_t exec;
    auto t0 = std::chrono::steady_clock::now();
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    saxpy_kernel<<<(N0 + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(d_x, d_y, N0, 2.0f);
    CK(cudaStreamEndCapture(s, &graph));
    auto t1 = std::chrono::steady_clock::now();
    CK(cudaGraphInstantiate(&exec, graph, 0));
    auto t2 = std::chrono::steady_clock::now();
    double cap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double inst_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    size_t n_nodes = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &n_nodes));

    cudaEventRecord(ea);
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    cudaEventRecord(eb); CK(cudaEventSynchronize(eb));
    float first_ms = ms_between(ea, eb);

    cudaEventRecord(ea);
    for (int i = 0; i < 100; ++i) CK(cudaGraphLaunch(exec, s));
    cudaEventRecord(eb); CK(cudaEventSynchronize(eb));
    double hot_ms = ms_between(ea, eb) / 100.0;

    size_t free_after = 0, total2 = 0;
    CK(cudaMemGetInfo(&free_after, &total2));
    printf("    捕获 %.3f ms   实例化 %.3f ms   节点数 %zu\n", cap_ms, inst_ms, n_nodes);
    printf("    首次 launch %.4f ms   热 replay %.4f ms（100 次平均）\n",
           first_ms, hot_ms);
    printf("    图中每步处理 n=%d，grid=%d\n", N0, (N0 + BLOCK - 1) / BLOCK);
    printf("    显存 free 变化 %+.1f MB\n",
           (double)((long long)free_after - (long long)free_before) / (1 << 20));

    // ---------------------------------------------------------------- C2
    printf("\n[C2] 改输入的值（同一块 d_x）—— 合法\n");
    fill_host(h, N0, 7.0f);
    CK(cudaMemcpy(d_x, h, N0 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_y, 0, N0 * sizeof(float)));
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h_out, d_y, N0 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N0; ++i) h_ref[i] = 2.0f * h[i] + 1.0f;
    printf("    与新输入的不一致元素 = %d（图只固定地址，不固定内容）\n",
           count_bad(h_out, h_ref, N0));

    // ---------------------------------------------------------------- C3
    printf("\n[C3] 换输入指针（改用 d_x2 / d_y2）—— 不合法\n");
    fill_host(h, N0, 100.0f);
    CK(cudaMemcpy(d_x2, h, N0 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_y, 0, N0 * sizeof(float)));
    CK(cudaMemset(d_y2, 0, N0 * sizeof(float)));
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h_out, d_y, N0 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N0; ++i) h_ref[i] = 2.0f * h[i] + 1.0f;
    int bad_new = count_bad(h_out, h_ref, N0);          // 与新输入（d_x2）比
    for (int i = 0; i < N0; ++i) h_ref[i] = 2.0f * (0.0f + i) + 1.0f;   // d_x 里的旧内容
    int bad_old = count_bad(h_out, h_ref, N0);          // 与旧输入（d_x）比
    float y2_first = -1.0f;
    CK(cudaMemcpy(&y2_first, d_y2, sizeof(float), cudaMemcpyDeviceToHost));
    printf("    写到 d_y（旧地址）的结果：与旧输入一致 %d/%d，与新输入一致 %d/%d\n",
           N0 - bad_old, N0, N0 - bad_new, N0);
    printf("    d_y2[0] = %.1f（新地址没有任何一个字节被写）\n", y2_first);
    printf("    => 执行用的是捕获时的地址；host 侧换个变量不会有任何效果。\n");

    // ---------------------------------------------------------------- C4
    printf("\n[C4] 改元素数（shape）—— 不合法\n");
    const int N1 = 200000;
    fill_host(h, N1, 0.0f);
    CK(cudaMemcpy(d_x, h, N1 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_y, 0, N1 * sizeof(float)));
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h_out, d_y, N1 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N1; ++i) h_ref[i] = 2.0f * h[i] + 1.0f;
    printf("    replay 后 [0,%d) 内不匹配 = %d\n", N0, count_bad(h_out, h_ref, N0));
    int tail_zero = 1;
    for (int i = N0; i < N1; ++i) if (h_out[i] != 0.0f) tail_zero = 0;
    printf("    [%d, %d) 仍为 0：%s（grid 还是捕获时的 %d 个 block）\n",
           N0, N1, tail_zero ? "是" : "否", (N0 + BLOCK - 1) / BLOCK);

    printf("    合法修法一：按新 n 重新捕获 -> ");
    cudaGraph_t g2; cudaGraphExec_t e2;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    saxpy_kernel<<<(N1 + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(d_x, d_y, N1, 2.0f);
    CK(cudaStreamEndCapture(s, &g2));
    CK(cudaGraphInstantiate(&e2, g2, 0));
    CK(cudaMemset(d_y, 0, N1 * sizeof(float)));
    CK(cudaGraphLaunch(e2, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h_out, d_y, N1 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("不匹配 = %d\n", count_bad(h_out, h_ref, N1));

    printf("    合法修法二：cudaGraphExecKernelNodeSetParams 改节点参数 -> ");
    size_t nn = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &nn));
    std::vector<cudaGraphNode_t> nodes(nn);
    CK(cudaGraphGetNodes(graph, nodes.data(), &nn));
    cudaGraphNode_t knode = nullptr;
    for (auto nd : nodes) {
        cudaGraphNodeType ty;
        CK(cudaGraphNodeGetType(nd, &ty));
        if (ty == cudaGraphNodeTypeKernel) { knode = nd; break; }
    }
    float alpha = 2.0f;
    void* kargs[] = {&d_x, &d_y, (void*)&N1, &alpha};
    cudaKernelNodeParams np = {};
    np.func = (void*)saxpy_kernel;
    np.gridDim = dim3((N1 + BLOCK - 1) / BLOCK);
    np.blockDim = dim3(BLOCK);
    np.sharedMemBytes = 0;
    np.kernelParams = kargs;
    cudaError_t perr = knode ? cudaGraphExecKernelNodeSetParams(exec, knode, &np)
                             : cudaErrorInvalidValue;
    CK(cudaMemset(d_y, 0, N1 * sizeof(float)));
    CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(h_out, d_y, N1 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("%s，不匹配 = %d\n",
           perr == cudaSuccess ? "成功" : cudaGetErrorString(perr),
           count_bad(h_out, h_ref, N1));

    // ---------------------------------------------------------------- C5
    printf("\n[C5] 释放并重新分配 —— 不合法\n");
    float* d_x3; float* d_y3;
    CK(cudaMalloc(&d_x3, N0 * sizeof(float)));
    CK(cudaMalloc(&d_y3, N0 * sizeof(float)));
    printf("    捕获时地址 d_x=%p d_y=%p；后来分配 d_x3=%p d_y3=%p\n",
           (void*)d_x, (void*)d_y, (void*)d_x3, (void*)d_y3);
    if (!strcmp(mode, "c5")) {
        cudaGraph_t g5; cudaGraphExec_t e5;
        float* dx_tmp = d_x3; float* dy_tmp = d_y3;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        saxpy_kernel<<<(N0 + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(dx_tmp, dy_tmp, N0, 2.0f);
        CK(cudaStreamEndCapture(s, &g5));
        CK(cudaGraphInstantiate(&e5, g5, 0));
        CK(cudaStreamSynchronize(s));
        CK(cudaFree(d_x3)); CK(cudaFree(d_y3));           // 图里的地址被释放
        printf("    已释放 d_x3/d_y3，现在 replay 这张图（sanitizer 应报非法访问）\n");
        CK(cudaGraphLaunch(e5, s)); CK(cudaStreamSynchronize(s));
        printf("    replay 返回（同步错误由上面的 CK 打印）\n");
        cudaGraphDestroy(g5); cudaGraphExecDestroy(e5);
    } else {
        printf("    图仍指旧地址：驱动没回收时看起来能跑，回收后是非法访问。\n");
        printf("    用 `./graph_capture c5` 跑这条路径，配合 compute-sanitizer 观察。\n");
    }

    // ---------------------------------------------------------------- C6
    printf("\n[C6] 把 n 放进设备内存 —— 同一张图跑不同 n（动态尺寸的正解）\n");
    int* d_n; CK(cudaMalloc(&d_n, sizeof(int)));
    cudaGraph_t g3; cudaGraphExec_t e3;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    dyn_kernel<<<(MAXN + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(d_x, d_y, d_n);
    CK(cudaStreamEndCapture(s, &g3));
    CK(cudaGraphInstantiate(&e3, g3, 0));
    printf("    %-10s %-10s %s\n", "实际 n", "不匹配", "说明");
    for (int n : {1000, N0, N1, MAXN}) {
        CK(cudaMemcpyAsync(d_n, &n, sizeof(int), cudaMemcpyHostToDevice, s));
        CK(cudaMemsetAsync(d_y, 0, MAXN * sizeof(float), s));
        CK(cudaGraphLaunch(e3, s));
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(h_out, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) h_ref[i] = 2.0f * h[i] + 1.0f;
        printf("    %-10d %-10d grid 固定 %d 个 block，逻辑尺寸由设备端决定\n",
               n, count_bad(h_out, h_ref, n), (MAXN + BLOCK - 1) / BLOCK);
    }

    // ---------------------------------------------------------------- C7
    printf("\n[C7] %d 个 kernel 的链：逐个 launch vs 一次 replay\n", CHAIN);
    {
        const int n = N0, blocks = (n + BLOCK - 1) / BLOCK;
        cudaGraph_t g4; cudaGraphExec_t e4;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int k = 0; k < CHAIN; ++k)
            saxpy_kernel<<<blocks, BLOCK, 0, s>>>(d_x, d_y, n, 1.0f + 0.1f * k);
        CK(cudaStreamEndCapture(s, &g4));
        CK(cudaGraphInstantiate(&e4, g4, 0));

        CK(cudaStreamSynchronize(s));
        auto c0 = std::chrono::steady_clock::now();
        for (int k = 0; k < CHAIN; ++k)
            saxpy_kernel<<<blocks, BLOCK, 0, s>>>(d_x, d_y, n, 1.0f + 0.1f * k);
        auto c1 = std::chrono::steady_clock::now();
        CK(cudaStreamSynchronize(s));
        double cpu_launch = std::chrono::duration<double, std::milli>(c1 - c0).count();

        CK(cudaStreamSynchronize(s));
        c0 = std::chrono::steady_clock::now();
        CK(cudaGraphLaunch(e4, s));
        c1 = std::chrono::steady_clock::now();
        CK(cudaStreamSynchronize(s));
        double cpu_graph = std::chrono::duration<double, std::milli>(c1 - c0).count();

        cudaEventRecord(ea);
        for (int r = 0; r < 20; ++r) {
            for (int k = 0; k < CHAIN; ++k)
                saxpy_kernel<<<blocks, BLOCK, 0, s>>>(d_x, d_y, n, 1.0f + 0.1f * k);
        }
        cudaEventRecord(eb); CK(cudaEventSynchronize(eb));
        double gpu_launch = ms_between(ea, eb) / 20.0;

        cudaEventRecord(ea);
        for (int r = 0; r < 20; ++r) CK(cudaGraphLaunch(e4, s));
        cudaEventRecord(eb); CK(cudaEventSynchronize(eb));
        double gpu_graph = ms_between(ea, eb) / 20.0;

        printf("    CPU 侧提交：逐个 launch %.3f ms   一次 replay %.3f ms   (%.2f x)\n",
               cpu_launch, cpu_graph, cpu_launch / cpu_graph);
        printf("    GPU 侧执行：逐个 launch %.3f ms   一次 replay %.3f ms   (%.2f x)\n",
               gpu_launch, gpu_graph, gpu_launch / gpu_graph);
        printf("    => replay 省的是 CPU 提交；GPU 上该跑的 kernel 一个不少。\n");
        cudaGraphDestroy(g4); cudaGraphExecDestroy(e4);
    }

    cudaGraphDestroy(graph); cudaGraphExecDestroy(exec);
    cudaGraphDestroy(g2); cudaGraphExecDestroy(e2);
    cudaGraphDestroy(g3); cudaGraphExecDestroy(e3);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_x2); cudaFree(d_y2);
    if (strcmp(mode, "c5")) { cudaFree(d_x3); cudaFree(d_y3); }
    cudaFree(d_n);
    cudaStreamDestroy(s);
    free(h); free(h_out); free(h_ref);
    return 0;
}
