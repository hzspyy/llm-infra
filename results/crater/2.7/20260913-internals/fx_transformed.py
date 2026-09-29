


def forward(self, x):
    mul = x * 2.0;  x = None
    clamp_min = torch.clamp_min(mul, 0.0);  mul = None
    add = clamp_min + 1.0;  clamp_min = None
    relu_1 = add.relu();  add = None
    return relu_1
    