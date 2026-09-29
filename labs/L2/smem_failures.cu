// L2.1 失败小例 · 共享内存 producer/consumer、同步掩码与跨流依赖。
//
// 每个小例都有「错」与「对」两个版本，跑同一份输入，打印不一致元素数。
// 用法：
//   nvcc -O2 -std=c++17 -arch=sm_120 -lineinfo -o smem_failures smem_failures.cu
//   ./smem_failures all
//   ./smem_failures race        # 只跑跨 warp 竞态
//   compute-sanitizer --tool racecheck ./smem_failures race
//
// 四个小例：
//   oob   去掉边界判断后的越界写（memcheck 能直接报）
//   race  缺 __syncthreads 的跨 warp 读写竞态
//   mask  发散的 __syncwarp / __shfl_down_sync 用错掩码
//   stream 两条流之间缺事件依赖

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("[CUDA] %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); } } while(0)

// ---------------------------------------------------------------- oob
// 故意不判断 i < n：越界写。正确版本加回判断。
__global__ void oob_kernel(const float* __restrict__ in, float* __restrict__ out,
                           int n, int guard) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (guard && i >= n) return;
    out[i] = in[i] * 2.0f + 1.0f;          // guard=0 时最后一块越界
}

// ---------------------------------------------------------------- race
// 写 smem[t] 之后读 smem[(t+1) % bt]：跨 warp 的 producer/consumer。
// 缺 __syncthreads 时，后面的 warp 可能在前面 warp 写完之前就读。
__global__ void race_kernel(const float* __restrict__ in, float* __restrict__ out,
                            int n, int sync, int delay) {
    extern __shared__ float s[];
    int bt = blockDim.x, t = threadIdx.x, base = blockIdx.x * bt;
    if (delay && t < 32) {
        // 只让第 0 个 warp 晚写：把竞态窗口放大，便于观察（不是错误来源）
        float x = t * 0.001f;
        for (int k = 0; k < 200000; ++k) x = fmaf(x, 1.0001f, 0.5f);
        if (x == 1234.5f) s[0] = -1.0f;
    }
    if (base + t < n) s[t] = in[base + t];
    if (sync) __syncthreads();
    int j = (t + 32) % bt;                  // 读下一个 warp 的同一 lane
    float v = (base + j < n) ? s[j] : 0.0f;
    if (base + t < n) out[base + t] = v + 1.0f;
}

// ---------------------------------------------------------------- mask
// 只有低 16 个 lane 参与 shuffle，却传 0xffffffff。
// 正确版本用 __activemask() 取当前活跃线程集合。
__global__ void mask_kernel(const float* __restrict__ in, float* __restrict__ out,
                            int n, int correct) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    int lane = threadIdx.x % 32;
    if (correct) {
        // 正确写法：整个 warp 都执行 shuffle，只用低 16 个 lane 的结果
        float t = __shfl_down_sync(0xffffffffu, v, 8);
        if (lane < 16) v += t;
    } else {
        // 错误写法：掩码声明 32 个 lane 参与，实际只有 16 个 lane 执行
        if (lane < 16) v += __shfl_down_sync(0xffffffffu, v, 8);
    }
    out[i] = v;
}

// ---------------------------------------------------------------- barrier
// 只有一半线程到达 __syncthreads：块级屏障要求块内所有线程都执行到同一条屏障。
__global__ void barrier_kernel(int* out, int n, int correct) {
    __shared__ int s[64];
    int t = threadIdx.x;
    if (t < 64) s[t] = t;
    if (correct) {
        __syncthreads();
    } else {
        if (t < 32) __syncthreads();      // 发散屏障
    }
    if (t == 0 && n > 0) out[0] = s[0] + s[63];
}

// ---------------------------------------------------------------- stream
__global__ void producer_kernel(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (float)i + 1.0f;
}

__global__ void consumer_kernel(const float* buf, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = buf[i] * 2.0f;
}

// ---------------------------------------------------------------- helpers
static void fill(float* p, int n) { for (int i = 0; i < n; ++i) p[i] = (float)i; }

static int diff_count(const float* a, const float* b, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i) if (a[i] != b[i]) ++bad;
    return bad;
}

static void cpu_reference(const float* in, float* ref, int n, int kind, int block) {
    for (int i = 0; i < n; ++i) {
        if (kind == 0) ref[i] = in[i] * 2.0f + 1.0f;                 // oob
        else if (kind == 1) {                                        // race
            int base = (i / block) * block, t = i % block;
            ref[i] = in[base + (t + 32) % block] + 1.0f;
        } else ref[i] = in[i];                                       // mask
    }
}

static void run_oob(int n, int guard) {
    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, n * sizeof(float)));
    CK(cudaMalloc(&d_out, n * sizeof(float)));
    float* h_in = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc(n * sizeof(float));
    float* h_ref = (float*)malloc(n * sizeof(float));
    fill(h_in, n); cpu_reference(h_in, h_ref, n, 0, 128);
    CK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
    int grid = (n + 127) / 128;                 // 不整除，最后一块只有 1 个有效元素
    oob_kernel<<<grid, 128>>>(d_in, d_out, n, guard);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  [oob] guard=%d  n=%d grid=%d  越界线程 = %d  不一致 = %d\n",
           guard, n, grid, grid * 128 - n, diff_count(h_out, h_ref, n));
    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_out); free(h_ref);
}

static void run_race(int n, int sync, int delay, int block, int runs) {
    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, n * sizeof(float)));
    CK(cudaMalloc(&d_out, n * sizeof(float)));
    float* h_in = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc(n * sizeof(float));
    float* h_ref = (float*)malloc(n * sizeof(float));
    fill(h_in, n); cpu_reference(h_in, h_ref, n, 1, block);
    CK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
    int grid = (n + block - 1) / block;
    int bad_runs = 0, worst = 0;
    for (int r = 0; r < runs; ++r) {
        CK(cudaMemset(d_out, 0, n * sizeof(float)));
        race_kernel<<<grid, block, block * sizeof(float)>>>(d_in, d_out, n, sync, delay);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = diff_count(h_out, h_ref, n);
        if (bad) { ++bad_runs; if (bad > worst) worst = bad; }
    }
    printf("  [race] sync=%d delay=%d block=%d grid=%d  %d 次里 %d 次不一致，最多 %d 个元素\n",
           sync, delay, block, grid, runs, bad_runs, worst);
    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_out); free(h_ref);
}

static void run_mask(int n, int correct, int runs) {
    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, n * sizeof(float)));
    CK(cudaMalloc(&d_out, n * sizeof(float)));
    float* h_in = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc(n * sizeof(float));
    fill(h_in, n);
    CK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
    int bad_runs = 0;
    for (int r = 0; r < runs; ++r) {
        CK(cudaMemset(d_out, 0, n * sizeof(float)));
        mask_kernel<<<(n + 63) / 64, 64>>>(d_in, d_out, n, correct);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        // 正确结果：低 16 lane 加上 lane+8 的值，高 16 lane 不变
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            int lane = i % 32;
            float want = h_in[i];
            if (lane < 16) want += h_in[(i / 32) * 32 + lane + 8];
            if (h_out[i] != want) ++bad;
        }
        if (bad) ++bad_runs;
    }
    printf("  [mask] correct=%d  %d 次里 %d 次结果与掩码语义不符\n",
           correct, runs, bad_runs);
    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_out);
}

static void run_stream(int n, int use_event, int runs) {
    float *d_buf, *d_out;
    CK(cudaMalloc(&d_buf, n * sizeof(float)));
    CK(cudaMalloc(&d_out, n * sizeof(float)));
    float* h_out = (float*)malloc(n * sizeof(float));
    float* h_ref = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) h_ref[i] = ((float)i + 1.0f) * 2.0f;

    cudaStream_t s1, s2;
    CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2));
    cudaEvent_t ev; CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    int bad_runs = 0;
    for (int r = 0; r < runs; ++r) {
        CK(cudaMemsetAsync(d_buf, 0, n * sizeof(float), s1));
        // 消费者先启动：没有依赖时它会读到清零后的缓冲
        producer_kernel<<<(n + 127) / 128, 128, 0, s1>>>(d_buf, n);
        if (use_event) CK(cudaEventRecord(ev, s1));
        if (use_event) CK(cudaStreamWaitEvent(s2, ev, 0));
        consumer_kernel<<<(n + 127) / 128, 128, 0, s2>>>(d_buf, d_out, n);
        CK(cudaStreamSynchronize(s1)); CK(cudaStreamSynchronize(s2));
        CK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; ++i) if (h_out[i] != h_ref[i]) ++bad;
        if (bad) ++bad_runs;
    }
    printf("  [stream] use_event=%d  %d 次里 %d 次读到未完成的数据\n",
           use_event, runs, bad_runs);
    cudaFree(d_buf); cudaFree(d_out); free(h_out); free(h_ref);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2); cudaEventDestroy(ev);
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "all";
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== %s  sm_%d%d  SM=%d ===\n", p.name, p.major, p.minor,
           p.multiProcessorCount);

    if (!strcmp(mode, "oob") || !strcmp(mode, "all")) {
        printf("\n[1] 去掉 i < n 判断后的越界写（每 SM 只有很少线程时量小，便于 sanitizer）\n");
        run_oob(1025, 0);
        run_oob(1025, 1);
    }
    if (!strcmp(mode, "race") || !strcmp(mode, "all")) {
        printf("\n[2] 缺 __syncthreads 的跨 warp 读写（delay=1 只放大窗口，不改变对错）\n");
        run_race(128 * 64, 0, 0, 128, 20);
        run_race(128 * 64, 0, 1, 128, 20);
        run_race(128 * 64, 1, 1, 128, 20);
    }
    if (!strcmp(mode, "mask") || !strcmp(mode, "all")) {
        printf("\n[3] 发散的 shuffle 用错掩码\n");
        run_mask(64 * 32, 0, 20);
        run_mask(64 * 32, 1, 20);
    }
    if (!strcmp(mode, "racefixed")) {
        run_race(128 * 64, 1, 1, 128, 20);          // 加了 __syncthreads 的修复版
    }
    if (!strcmp(mode, "barrier") || !strcmp(mode, "all")) {
        printf("\n[3b] 发散的 __syncthreads\n");
        int* d_bo; CK(cudaMalloc(&d_bo, sizeof(int)));
        for (int correct : {0, 1}) {
            CK(cudaMemset(d_bo, 0, sizeof(int)));
            barrier_kernel<<<2, 64>>>(d_bo, 1, correct);
            cudaError_t e = cudaDeviceSynchronize();
            int v = -1; CK(cudaMemcpy(&v, d_bo, sizeof(int), cudaMemcpyDeviceToHost));
            printf("  [barrier] correct=%d  同步返回=%s  写出值=%d\n",
                   correct, e == cudaSuccess ? "成功" : cudaGetErrorString(e), v);
        }
        cudaFree(d_bo);
    }
    if (!strcmp(mode, "stream") || !strcmp(mode, "all")) {
        printf("\n[4] 两条流之间缺事件依赖\n");
        run_stream(1 << 20, 0, 20);
        run_stream(1 << 20, 1, 20);
    }
    printf("\n用法：./smem_failures [all|oob|race|racefixed|mask|barrier|stream]\n");
    return 0;
}
