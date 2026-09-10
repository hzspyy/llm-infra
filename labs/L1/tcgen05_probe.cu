// L1.2 lab · 故意去撞那堵墙：在 sm_120 上编译 tcgen05 指令。
//
// 「Blackwell」是一个营销名，底下是两套**不兼容**的 tensor core 编程模型：
//   sm_100/103（B200 等数据中心卡）：tcgen05.mma + Tensor Memory (TMEM)
//   sm_120/121（RTX 50 系消费卡）：没有 tcgen05，走 Ampere 血统的 mma.sync
//
// 与其记住这句话，不如亲手编一次，看编译器怎么骂你。
//
//   nvcc -arch=sm_120a -cubin -o /dev/null tcgen05_probe.cu   # 应该失败
//   nvcc -arch=sm_100a -cubin -o /dev/null tcgen05_probe.cu   # 语法本身是合法的

#include <cstdint>

__global__ void tcgen05_smoke(uint32_t* tmem_addr) {
    // tcgen05 的第一步：从 Tensor Memory 里分配一块。
    // sm_120 连这条指令都不认识——它根本没有 TMEM 这个存储空间。
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n"
                 :: "r"((uint32_t)(uintptr_t)tmem_addr));
}
