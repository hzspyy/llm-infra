
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root/tmpovukz4xc'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_root/tmpovukz4xc/triton'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.debug_dir = 'out/2.7/20260913-internals/trace'
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.13.0+cu130
# torch cuda version: 13.0
# torch git version: cf30153c4c131c8164ee7798e5022d810682e2cb


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA GeForce RTX 5090 D : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1):
        permute = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm = torch.ops.aten.mm.default(arg0_1, permute);  arg0_1 = permute = None
        relu = torch.ops.aten.relu.default(mm);  mm = None
        add = torch.ops.aten.add.Tensor(relu, arg2_1);  relu = None
        add_1 = torch.ops.aten.add.Tensor(arg2_1, add)
        sum_1 = torch.ops.aten.sum.default(add);  add = None
        copy_ = torch.ops.aten.copy_.default(arg2_1, add_1);  arg2_1 = add_1 = copy_ = None
        return (sum_1,)

def load_args(reader):
    buf0 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 128), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 128), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64, 128), is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)