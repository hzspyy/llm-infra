**vllm** `0.29.0` · `/scratch/learn/envs/serve/lib/python3.12/site-packages/vllm`
**sglang** `0.5.19` · `/scratch/learn/envs/sgl/lib/python3.12/site-packages/sglang/srt`

| # | 阶段 | vllm | sglang |
|---|---|---|---|
| 1 | **HTTP 路由** | `entrypoints/openai/chat_completion/api_router.py:54`<br><small>API 进程 · asyncio</small> | `entrypoints/http_server.py:1734`<br><small>HTTP 进程 · asyncio</small> |
| 2 | **协议层** | `entrypoints/openai/chat_completion/serving.py:242`<br><small>API 进程 · asyncio</small> | `entrypoints/openai/serving_chat.py:244`<br><small>HTTP 进程 · asyncio</small> |
| 3 | **模板渲染/分词** | `renderers/hf.py:936`<br><small>API 进程 · asyncio</small> | `managers/tokenizer_manager.py:394`<br><small>HTTP 进程 · asyncio</small> |
| 4 | **前端入口** | `v1/engine/async_llm.py:625`<br><small>API 进程 · asyncio</small> | `managers/tokenizer_manager.py:770`<br><small>HTTP 进程 · asyncio</small> |
| 5 | **输入处理** | `v1/engine/input_processor.py:281`<br><small>API 进程 · asyncio</small> | `managers/io_struct.py:174`<br><small>HTTP 进程 · asyncio</small> |
| 6 | **跨进程 IPC** | `v1/engine/core_client.py:240`<br><small>API 进程 → 引擎进程</small> | `managers/tokenizer_manager.py:1814`<br><small>Tokenizer 进程 → Scheduler 进程</small> |
| 7 | **引擎主循环** | `v1/engine/core.py:1411`<br><small>引擎进程 · busy loop</small> | `managers/scheduler.py:1808`<br><small>Scheduler 进程 · busy loop</small> |
| 8 | **一步 = 一次前向** | `v1/engine/core.py:597`<br><small>引擎进程</small> | `managers/scheduler.py:4013`<br><small>Scheduler 进程</small> |
| 9 | **调度器** | `v1/core/sched/scheduler.py:501`<br><small>引擎进程</small> | `managers/scheduler.py:3335`<br><small>Scheduler 进程</small> |
| 10 | **前缀复用查询** | `v1/core/kv_cache_manager.py:228`<br><small>引擎进程</small> | `mem_cache/radix_cache.py:377`<br><small>Scheduler 进程</small> |
| 11 | **显存分配** | `v1/core/block_pool.py:647`<br><small>引擎进程</small> | `mem_cache/radix_cache.py:593`<br><small>Scheduler 进程</small> |
| 12 | **执行/模型前向** | `v1/worker/gpu_model_runner.py:4249`<br><small>worker · GPU</small> | `model_executor/model_runner.py:1568`<br><small>TP worker · GPU</small> |
| 13 | **采样** | `v1/sample/sampler.py:73`<br><small>worker · GPU</small> | `layers/sampler.py:114`<br><small>TP worker · GPU</small> |
| 14 | **结果回传与流式输出** | `v1/engine/output_processor.py:621`<br><small>API 进程 · asyncio</small> | `managers/detokenizer_manager.py:177`<br><small>Detokenizer 进程</small> |

### 每一层各自在做什么

**1. HTTP 路由**
- `vllm`：FastAPI 路由。到这里为止都还是普通 web 服务：反序列化 JSON、校验 pydantic 模型。
- `sglang`：同样是 FastAPI，但 SGLang 把路由集中在**一个** http_server.py 里，而不是像 vLLM 那样每个能力一个 api_router 包。

**2. 协议层**
- `vllm`：OpenAI 协议语义：解析 messages、工具定义、采样参数，决定流式还是一次性返回。
- `sglang`：OpenAI 协议适配。注意 SGLang 把 thinking mode（推理模型的 <think> 段）做成了协议层的一等公民（同文件 ThinkingMode）。

**3. 模板渲染/分词**
- `vllm`：套 chat template → 字符串 → token id；多模态输入拆成占位符 + 图像/视频数据。0.29 把 renderer 提成了顶层可插拔包（renderers/registry.py），因为 DeepSeek / Kimi / Mistral 的模板语义塞不进一个通用函数。
- `sglang`：**这是 SGLang 与 vLLM 最大的结构差异**：分词不是一个函数，而是一个独立的 TokenizerManager 进程/组件，同时负责分词、请求登记（ReqState）、以及把结果路由回对应的 HTTP 协程。

**4. 前端入口**
- `vllm`：AsyncLLM.generate：给请求建一个输出队列（RequestOutputCollector）然后 await。HTTP 协程从此挂起，直到有 token 被推回来。
- `sglang`：前端入口。与 vLLM 的 AsyncLLM.generate 对应：登记 ReqState，await 事件。

**5. 输入处理**
- `vllm`：构造 EngineCoreRequest：token id、采样参数、多模态数据的哈希与缓存键、LoRA 请求。
- `sglang`：SGLang 的请求结构体集中在 io_struct.py。对照 vLLM 的 EngineCoreRequest 看两边各自认为「一个请求」必须携带哪些字段，很能看出设计取向。

**6. 跨进程 IPC**
- `vllm`：序列化后经 ZeroMQ 送到独立的 EngineCore 进程。拆进程是为了绕开 GIL：HTTP / 分词 / detokenize 不能和 GPU 调度抢同一个解释器。
- `sglang`：SGLang 是**三进程**结构：TokenizerManager → Scheduler → DetokenizerManager，三者用 ZeroMQ 串起来。vLLM 是两进程（API + EngineCore），detokenize 在 API 进程。多一次进程边界换来 detokenize 不阻塞调度——这是一个真实的取舍。

**7. 引擎主循环**
- `vllm`：EngineCore 的 while True：收新请求 → step() → 把输出塞回 IPC。整个系统的心跳。
- `sglang`：Scheduler 的主循环。同文件还有 event_loop_overlap 变体：把 CPU 侧调度与 GPU 执行重叠起来（vLLM 的对应物是异步调度）。

**8. 一步 = 一次前向**
- `vllm`：step()：调度 → 执行 → 收结果。continuous batching 的「iteration-level」就是这个 step 的粒度：每一步都能换一批请求，不必等某个请求生成完。
- `sglang`：跑一个 batch。与 vLLM 的 step() 对应。

**9. 调度器**
- `vllm`：本步跑哪些请求、每个跑几个 token。chunked prefill、抢占、优先级、token 预算都在这里。输出 SchedulerOutput——一份给 GPU 的施工图。
- `sglang`：选批策略。SGLang 把策略单独放在 managers/schedule_policy.py，支持 LPM（最长前缀优先）等对 radix cache 友好的排序——这是它和 vLLM 的关键差异之一。

**10. 前缀复用查询**
- `vllm`：按**块哈希**查前缀有多少已算过。vLLM 用哈希表（O(1) 查一个块），命中的块直接引用，对应 token 不再进入 prefill。
- `sglang`：**RadixAttention**：前缀不是哈希表里的独立块，而是一棵基数树（radix tree）。match_prefix 沿树走最长公共前缀，天然支持共享分支与子树级淘汰。对照 vLLM 的哈希表方案：树能表达前缀间的包含关系，哈希表不能。

**11. 显存分配**
- `vllm`：物理块分配器：free list + 块哈希表。PagedAttention 的「页」就是这里的 KVCacheBlock。
- `sglang`：按 LRU 从叶子往上淘汰整棵子树。同文件的 cache_finished_req / cache_unfinished_req 决定一个请求的 KV 何时进入共享树。

**12. 执行/模型前向**
- `vllm`：组装输入张量（_prepare_inputs，同文件）→ 选 CUDA Graph 或 eager → 模型 forward。
- `sglang`：ModelRunner.forward。输入批信息在 model_executor/forward_batch_info.py 的 ForwardBatch 里（对照 vLLM 的 _prepare_inputs）。

**13. 采样**
- `vllm`：logits → 惩罚 → 温度 → top-k/top-p → 采样。全在 GPU 上做，避免把 logits 拷回 CPU。
- `sglang`：采样。SGLang 默认走 FlashInfer 的采样 kernel。

**14. 结果回传与流式输出**
- `vllm`：增量 detokenize、处理 stop string、推进每个请求的输出队列；再由 serving.py 的 chat_completion_stream_generator 包成 SSE 写回 socket。
- `sglang`：**独立进程**做增量 detokenize，再经 ZeroMQ 回到 TokenizerManager，由它唤醒对应的 HTTP 协程写 SSE。vLLM 把这一步放在 API 进程内。

