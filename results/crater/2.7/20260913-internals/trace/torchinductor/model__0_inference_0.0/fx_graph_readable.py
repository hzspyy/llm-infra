class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 128]", arg1_1: "f32[128, 128]", arg2_1: "f32[64, 128]"):
        # File: /scratch/learn/work/labs/compiler_internals.py:135 in f, code: h = F.linear(x, w)
        permute: "f32[128, 128]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "f32[64, 128]" = torch.ops.aten.mm.default(arg0_1, permute);  arg0_1 = permute = None

        # File: /scratch/learn/work/labs/compiler_internals.py:136 in f, code: h = F.relu(h)
        relu: "f32[64, 128]" = torch.ops.aten.relu.default(mm);  mm = None

        # File: /scratch/learn/work/labs/compiler_internals.py:137 in f, code: h = h + residual
        add: "f32[64, 128]" = torch.ops.aten.add.Tensor(relu, arg2_1);  relu = None

        # File: /scratch/learn/work/labs/compiler_internals.py:138 in f, code: residual.add_(h)          # 原地写：会被函数式化改写
        add_1: "f32[64, 128]" = torch.ops.aten.add.Tensor(arg2_1, add)

        # File: /scratch/learn/work/labs/compiler_internals.py:139 in f, code: return h.sum()
        sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None

        # File: /scratch/learn/work/labs/compiler_internals.py:138 in f, code: residual.add_(h)          # 原地写：会被函数式化改写
        copy_: "f32[64, 128]" = torch.ops.aten.copy_.default(arg2_1, add_1);  arg2_1 = add_1 = copy_ = None
        return (sum_1,)
