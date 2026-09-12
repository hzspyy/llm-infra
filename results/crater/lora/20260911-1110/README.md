# L5.10 evidence index

Synthetic seeded LoRA adapters are systems fixtures, not trained models.
initial/: first instrumentation run, failed to serialize a numpy int32.
completed/: corrected logging; full serving experiment followed by intentional rank-limit failure.
The invalid rank was caught, but the next valid generate call failed again; exit code is 1.
trace/: separate successful run, instrumenting only real requests after model initialization; exit 0.
completed/sglang-kernel.log: actual segmented Triton A/B kernels vs dense FP32 reference.
Adapter config, safetensors headers and SHA-256 are retained. Synthetic weight files remain remote and can be regenerated from the recorded seeds.
Derived medians are in derived.json. No base model weights were duplicated.
source_manifest_complete.json includes all copied source files.
