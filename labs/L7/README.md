# L7 实验入口

从项目根目录运行。Python、GPU、缓存与临时目录配置见 [ENVIRONMENTS.md](../../ENVIRONMENTS.md)。`RUN_DIR` 是学习盘上为本次实验新建的父目录；`TMPDIR` 也必须在学习盘上。本机使用项目指定 venv。带 `--outdir`/`--out-dir` 的输出叶目录必须不存在，避免覆盖原始证据。

| 入口 | 执行范围 |
|---|---|
| `mini_autograd.py` | NumPy 小引擎及重复 backward 与 PyTorch 对拍 |
| `_tinylm.py` | 7.0b 共用的 FP64 最小因果 LM 与二维 flow 小网络，不单独运行 |
| `supervision_contract.py` | shift、attention/loss mask、有效元素分母与连续目标对照 |
| `one_update_reference.py` | global clip 与手写 AdamW 同 PyTorch 逐参数对拍 |
| `accumulation_denominator.py` | 三条累积路径，含 dropout/clip/scheduler 次序反例 |
| `start_modes_and_resume.py` | 五种启动方式与严格恢复的逐项反例，checkpoint 只落内存 |
| `recipe_state_ledger.py` | 只读解析 SmolLM3 配方与发布 config，算参数量、batch、LR 与状态字节 |
| `training_step_smollm.py` | SmolLM2-360M 一次完整更新与三条累积路径（GPU） |
| `sharded_update_contract.py` | 两 rank：single/DDP/FSDP2 的状态字节、collective 计数与分母契约（GPU） |
| `pipeline_schedules.py` | GPipe/1F1B/interleaved/ZB-V 的事件模拟、空泡与权重版本 |
| `parallel_failure_modes.py` | 多 rank 失败面（gloo）与并行配置整除检查 |
| `recipe_inspector.py` | 安全解析真实 YAML/JSON/TOML，需要 `--config` |
| `checkpoint_state_schema.py` | 训练状态清单、按类型的必存项、阶段产物与保存频率预算 |
| `dcp_checkpoint_contract.py` | 真实 DCP 的同步/异步、四种失败注入、latest 选择与 2→4 重新分片 |
| `resumable_training.py` | 数据游标与严格恢复的逐项反例，含三种 scheduler 的敏感度 |
| `sft_adapter_contract.py` | 合成 SFT mask、一层 LoRA 更新/保存/加载/合并 |
| `preference_loss_reference.py` | DPO/APO 目标与解析梯度 |
| `rl_objective_reference.py` | PPO clipping、GRPO 风格梯度与两组 GAE 轨迹 |
| `distillation_objectives.py` | masked CE/KL/reverse KL/JSD、温度与教师 detach |
| `draft_objectives.py` | DFlash 块 mask、EAGLE3/DFlash 两种分母、位置衰减、acceptance/LK、compact teacher |
| `teacher_feature_store.py` | 真实落盘特征库：shard、ACK、崩溃重放、过期拦截、三种生产模式与字节账 |
| `teacher_store_contract.py` | 早期元数据检查；真实存储与 I/O 见 `teacher_feature_store.py` |
| `training_data_contract.py` | 自制文本去重与污染排查、padding/packing 与 loss 边界 |
| `sample_schema_survey.py` | 从快照源码抽字段建统一 schema，拆分泄漏与规则变更影响 |
| `data_mixture_sampling.py` | 固定 RNG 的真实采样序列，四种混合权重单位与 sampler 尾部 |
| `data_supply_paths.py` | Parquet/tar/mmap 的结构代价、真实 DataLoader 重叠、坏样本与游标 |
| `optimizer_state_reference.py` | SGD/momentum/Adam/AdamW 的 FP64 对拍、参数组、调度时钟与 EMA |
| `training_precision_ledger.py` | 五种配置的实际 dtype/字节、保存值账、五种训练方法的常驻状态 |
| `precision_error_budget.py` | autocast 分类、间距与下溢、长归约、前向/dX/dW/更新四处误差 |
| `loss_scaling_reference.py` | 官方 GradScaler 六次尝试的状态机与三个次序反例 |
| `low_precision_formats.py` | FP8/MXFP8/NVFP4 的格式、缩放粒度、三类 GEMM、随机舍入与 Hadamard |
| `numerical_diagnostics.py` | 五种注入的逐层记录与首个坏张量，另查 optimizer 常数在低精度下的失效 |
| `fp8_gemm_probe.py` | GPU：TF32 与 scaled_mm 的数值、约束报错与计时（需 sm_89 以上） |
| `multimodal_train_contract.py` | 三目标教学模型、冻结与解冻，共两次更新 |
| `training_job_simulator.py` | 真实本机子进程、ACK、超时、清理与提交重放 |
| `training_scheduling_budget.py` | `--section admission\|budget\|public-flows`：准入模拟、SmolLM3 预算账、三条公开流程字段账 |
| `training_trace_analysis.py` | 显式 `--input` 事件或 `--synthetic-demo` 预算 |
| `training_step_timeline.py` | GPU：SmolLM2-360M 的阶段时间线、重算对照、精度边界与编译代价 |
| `loop_lifecycles.py` | 四种训练循环的状态账、冻结段建图条件与特征缓存键 |
| `distributed_collectives.py` | `--exp` 选择通信计时/拓扑，traffic 公式不代表算法观测 |
| `teaching_data.py` | 7.8-I 教学语料：过滤、两层去重、按文档划分、tokenizer 分片与预算账 |
| `teaching_pretrain.py` | 7.1-F 预算内预训练：复用 MiniMind 模型，记录 grad/update/吞吐/峰值并支持恢复 |
| `teaching_sft.py` | 7.5-I SFT：assistant 段监督、回答 token 加权 loss、验证与导出 |
| `teaching_lora.py` | 7.5-J LoRA 领域适配：只训练低秩旁路、held-out 子技能曲线、合并导出 |
| `teaching_dpo.py` | 7.5-J DPO：与上游 `train_dpo.py` 损失对拍、held-out margin 曲线、导出 |
| `teaching_preference_eval.py` | 后训练分支评测：偏好 margin、末轮生成与工具调用切点两套判据、合并对拍 |
| `teaching_distill.py` | 7.7-J 蒸馏学生：在线 KD / 纯 CE 对照 / top-k 教师缓存三种 arm |
| `teaching_student_cost.py` | 学生与教师的参数、prefill、解码与峰值显存对照 |
| `teaching_grpo.py` | 7.6-J 可验证算术任务的 GRPO：规则奖励、组内基线、组内零方差与评测 |
| `teaching_vlm.py` | 7.10-J 视觉适配：冻结 CLIP + projector 两阶段训练、换图/空图对照 |
| `teaching_framework_path.py` | 7.3-G 同一训练契约的单卡与 FSDP2 路径：损失轨迹对拍、集合通信计数与 DCP 往返 |
| `framework_interface_map.py` | 7.3-G 教学 loop 九阶段到 FSDP2/DeepSpeed/Megatron/TorchTitan 的 `file:line` 映射 |
| `rl_runtime_contract.py` | 7.6-D/E/F 三框架角色归属定位 + 异步队列（lag/慢 reward/重复回包/背压）模拟 |
| `distill_generative_contracts.py` | 7.7-G/H 一维解析场的 LCM/DMD2 目标与梯度检查，加四份公开实现的阶段定位 |
| `multimodal_stage_matrix.py` | 7.10-A/D/E/F/G/H 六基座阶段矩阵、分桶浪费与缓存身份检查 |
| `multimodal_failure_localization.py` | 7.10-I 跨阶段失败定位：mask/codec/flow/normalizer 四类注入 |
| `teaching_eval.py` | 留出评测：随机初始化/预训练/SFT/后训练权重的 CE 与贪心生成 exact/F1 |
| `teaching_export.py` | 阶段权重导出为上游可加载的 `.pth`（权重与训练态分开） |
| `teaching_compare_runs.py` | 两条运行的逐步 loss/grad 对齐与参数逐张量最大差 |
| `teaching_cost_report.py` | 7.11-I 成本账：墙钟、有效 token、GPU 小时、峰值与 checkpoint 字节 |
| `run_teaching_lm.sh` | 教学主线入口：`data/pilot/pretrain/eval-pretrain/sft/eval-sft/export/cost` |
| `run_teaching_posttrain.sh` | 后训练分支入口：`data/lora-pilot/lora/lora-full-ft/dpo-pilot/dpo/eval`（7.5-J） |
| `run_teaching_distill.sh` | 蒸馏入口：`kd-pilot/kd/ce/cache-build/cache-online/cache-use/eval/cost`（7.7-J） |
| `run_teaching_rl.sh` | RL 入口：`sweep/data/base/train/train-kl/eval`（7.6-J） |
| `run_teaching_vlm.sh` | 视觉适配入口：`data/base/align/sft/sft-long/eval`（7.10-J） |
| `run_teaching_checks.sh` | 机制对拍入口：`resume`（7.4-H）与 `precision-window`（7.9-I） |

```bash
python labs/L7/sft_adapter_contract.py --outdir "$RUN_DIR/sft-lora"
python labs/L7/recipe_inspector.py \
  --config results/local/7.3/20260914-review-fixes/source/smollm/text/pretraining/smollm3/stage1_8T.yaml \
  --outdir "$RUN_DIR/recipe"
torchrun --standalone --nproc_per_node=2 \
  labs/L7/sharded_update_contract.py --mode fsdp2 --outdir "$RUN_DIR/fsdp2" --profile
```

`run_7.0b.sh` 只执行 CPU 机制与短恢复。GPU 入口（`training_step_smollm.py`、`training_step_timeline.py`、`fp8_gemm_probe.py`）启动前检查显存占用，结束后确认本次进程已退出。

`training_step_smollm.py` 的模型是 SmolLM2-360M，不是 SmolLM3；它只加载权重、不写 checkpoint，保存与恢复的实测在 7.4。三条累积路径分别对应"全局分母""传 `num_items_in_batch`""每份自己的均值再除以 M"，比较判据是梯度而不是一次更新后的参数。`mini_rl_iteration.py` 和 `training_job_manifest.py` 的早期材料只按各自实际模拟范围解释，不作为生产运行或整章验收。

原始工件位于 `results/<machine>/<chapter>/<run>/`。小数值、源码核对、真实进程、GPU 机制和训练质量分别验收；具体进度只记录在 [STATUS.md](../../STATUS.md)。
