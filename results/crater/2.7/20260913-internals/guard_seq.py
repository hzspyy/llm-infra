import torch
torch.manual_seed(0)
D = 128
def f(x, scale):
    return (x * scale).relu().sum()
fn = torch.compile(f, dynamic=False)
x = torch.randn(8, D)
seq = [("首次调用", lambda: fn(x, 2.0)),
       ("同样输入", lambda: fn(x, 2.0)),
       ("标量换值 3.0", lambda: fn(x, 3.0)),
       ("标量换成 int 2", lambda: fn(x, 2)),
       ("batch 8→9", lambda: fn(torch.randn(9, D), 2.0)),
       ("rank 2D→3D", lambda: fn(torch.randn(2, 4, D), 2.0)),
       ("dtype fp32→fp64", lambda: fn(x.double(), 2.0)),
       ("回到最初", lambda: fn(x, 2.0))]
for name, call in seq:
    try:
        call()
        print(f"OK   {name}")
    except Exception as e:
        print(f"ERR  {name}: {type(e).__name__}: {str(e)[:80]}")
