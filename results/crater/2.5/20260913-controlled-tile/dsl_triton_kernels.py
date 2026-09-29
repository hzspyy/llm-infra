
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, C, bias, M, N, K,
                sam, sak, sbk, sbn, scm, scn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                GROUP: tl.constexpr, B_TRANS: tl.constexpr, FUSE: tl.constexpr):
    pid = tl.program_id(0)
    n_m, n_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    per_group = GROUP * n_n
    gid = pid // per_group
    first_m = gid * GROUP
    group_m = min(n_m - first_m, GROUP)
    pid_m = first_m + ((pid % per_group) % group_m)
    pid_n = (pid % per_group) // group_m

    offs_m = (pid_m * BM + tl.arange(0, BM)) % M
    offs_n = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)
    a_ptr = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    if B_TRANS:
        b_ptr = B + offs_n[None, :] * sbn + offs_k[:, None] * sbk
    else:
        b_ptr = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        am = tl.load(a_ptr, mask=offs_k[None, :] < K - k * BK, other=0.0)
        bm = tl.load(b_ptr, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(am, bm, acc)
        a_ptr += BK * sak
        b_ptr += BK * sbk

    if FUSE:
        bv = tl.load(bias + offs_n, mask=offs_n < N, other=0.0)
        acc = tl.maximum(acc + bv[None, :], 0.0)

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptr = C + offs_cm[:, None] * scm + offs_cn[None, :] * scn
    tl.store(c_ptr, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
