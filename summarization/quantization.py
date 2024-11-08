from precompiled import *

class QTrainConfig:
    def __init__(self):
        self.batch_size = 32  # 4
        self.gradient_accumulation_steps = 4
        self.epochs = 5 * self.gradient_accumulation_steps

q_train_config = QTrainConfig()

q_type = torch.int8   # fp16/fp32 -> int8
q_info = torch.iinfo(q_type)

def affine(matrix: Tensor) -> (Tensor, Tensor):
    assert matrix.ndim >= 2, "sanity"
    # default: row-based quantization
    q_max, q_min = q_info.max, q_info.min
    scale = torch.amax(torch.abs(matrix), list(range(1, matrix.ndim)), True) / q_info.max
    q_matrix = torch.round(matrix / scale).clamp(q_min, q_max).to(q_type)
    return scale, q_matrix

def de_affine(q_matrix: Tensor, scale: Tensor) -> Tensor:
    res = q_matrix * scale   # broadcast "scale", also auto promote q_matrix to "scale"'s type
    assert res.dtype == scale.dtype, f"type mismatch: {res.dtype} {scale.dtype}"
    return res

def find_module_last_parent(root: nn.Module, module_name: str) -> (nn.Module, str):
    names = module_name.split('.')
    parents = names[:-1]   # remove our target's name
    for parent in parents:
        root = getattr(root, parent)
        assert root, "sanity"
    return root, names[-1]  # return: parent, child's name

def get_ptq_model(model: nn.Module) -> BartForConditionalGeneration:
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent, name = find_module_last_parent(model, full_name)
            q_linear = QLinear.from_linear(module)
            setattr(parent, name, q_linear)
    return model

def freeze(q_model: BartForConditionalGeneration) -> dict:
    q_dict = dict()
    for full_name, module in q_model.named_modules():
        if isinstance(module, QLinear):
            module.freeze()
            q_dict[full_name] = {
                "q_weights": module.q_weight.shape,
                "scale": module.scale.shape,
            }
    return q_dict

def thaw(q_model: BartForConditionalGeneration):
    for full_name, module in q_model.named_modules():
        if isinstance(module, QLinear):
            module.thaw()

class QAutoGrad(Function):   # QAT

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input = args[0]
        weight = args[1]
        bias = args[2]

        scale, q_weight = affine(weight)
        q_weight = q_weight.float()  # just for convenience so we can avoid many .float() calls
        output = torch.matmul(input, q_weight.t())
        output = de_affine(output, scale.t())
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(input, q_weight, scale, bias)

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        input, q_weight, scale, bias = ctx.saved_tensors  # type: (Tensor, Tensor, Tensor)
        output_dim, input_dim = q_weight.shape

        # STE: we only calculate matmul & bias.

        grad_output: Tensor = grad_outputs[0]
        # input: (bs, seq_len, input_dim)
        # weight: (output_dim, input_dim)
        # grad_output: (bs, seq_len, output_dim), which is the same as the $output in forward().
        # print(grad_output.shape, input.shape, q_weight.shape, bias.shape if bias is not None else None)

        # (1) grad on input:
        # g_o @ d(input @ q_weight.T)/d(input) = g_o @ q_weight
        # => (bs, seq_len, output_dim) @ (output_dim, input_dim)
        # => (bs, seq_len, input_dim). same as "input".
        a = torch.matmul(grad_output, q_weight * scale)
        # (2) grad on q_weight:
        # g_o @ d(input @ q_weight.T)/d(q_weight) = g_o.T @ input
        # => (output_dim, bs * seq_len) @ (bs * seq_len, input_dim)
        # => (output_dim, input_dim). same as "q_weight".
        b = torch.matmul(grad_output.reshape(-1, output_dim).t().to(input.dtype),  # input might be: Half16
                         input.reshape(-1, input_dim))
        # (3) grad on bias:
        #
        c = None
        if ctx.needs_input_grad[2]:
            assert grad_output.ndim == 3, f"sanity: {grad_output.shape}"
            c = grad_output.sum(dim=(0, 1))

        return a, b, c


class QLinear(nn.Linear):
    def __init__(self,
                 module: nn.Linear,
                 weight: Optional[Parameter] = None,
                 scale: Optional[Parameter] = None,
                 q_weight: Optional[Parameter] = None,
                 ):
        # Hack: See source code of nn.Linear for more info
        super().__init__(module.in_features, module.out_features, bias=True, device=module.weight.device, dtype=module.weight.dtype)
        self.weight = weight
        self.bias = module.bias
        self.scale = scale
        self.q_weight = q_weight

    @classmethod
    def from_linear(cls, module: nn.Linear):
        return cls(module, weight=Parameter(module.weight, True))

    @classmethod
    def from_fake(cls, module: nn.Linear, scale_shape: Size, q_weight_shape: Size):
        fake_scale = torch.zeros(scale_shape)
        fake_q_weights = torch.zeros(q_weight_shape)
        return cls(module,
                   scale=Parameter(fake_scale, False),
                   q_weight=Parameter(fake_q_weights, False))

    # call just before saving the module
    def freeze(self):
        scale, q_weight = affine(self.weight)
        self.scale = Parameter(scale, False)
        self.q_weight = Parameter(q_weight, False)
        del self.weight

    # call just after loading the module
    def thaw(self):
        assert not hasattr(self, 'weight') or self.weight is None, "initializing"
        self.weight = Parameter(de_affine(self.q_weight, self.scale))
        del self.scale
        del self.q_weight

    def forward(self, input: Tensor) -> Tensor:
        # q_weight: int8, (output_dim, input_dim)
        # input: int8, (bs, input_dim)
        # bias: float32, (output_dim, )
        if self.bias is not None:
            assert self.bias.device == input.device, f"device mismatch: {self.bias.device} {input.device}"

        if self.training:
            output = QAutoGrad.apply(input, self.weight, self.bias)
        else:
            output = F.linear(input, self.weight, self.bias)
        return output


