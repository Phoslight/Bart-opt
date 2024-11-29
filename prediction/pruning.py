from data import *

class PTrainConfig:
    def __init__(self):
        self.batch_size = 32  # 4
        self.gradient_accumulation_steps = 4
        self.epochs = 1
        self.total_steps = len(train_dataset) // self.batch_size // self.gradient_accumulation_steps * self.epochs

# Note: This will NOT be serialized by checkpoints.
class ThresholdScheduler:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.progress = 0
        self.init_threshold = 0
        self.ffn_final_threshold = 0.6  # we will finally prune {final_threshold}% amount of total.
        self.attn_final_threshold = 0.3

        self.ffn_threshold = self.init_threshold
        self.attn_threshold = self.init_threshold

    def update(self, cur_step: int):
        fine_tune_steps = self.total_steps * 0.85

        self.progress = min((cur_step / fine_tune_steps), 1.0)
        self.ffn_threshold = self.ffn_final_threshold - (self.ffn_final_threshold - self.init_threshold) * (1 - self.progress) ** 3
        self.attn_threshold = self.attn_final_threshold - (self.attn_final_threshold - self.init_threshold) * (1 - self.progress) ** 3

p_train_config = PTrainConfig()
scheduler = ThresholdScheduler(p_train_config.total_steps)

def find_module_last_parent(root: nn.Module, module_name: str) -> (nn.Module, str):
    names = module_name.split('.')
    parents = names[:-1]   # remove our target's name
    for parent in parents:
        root = getattr(root, parent)
        assert root, "sanity"
    return root, names[-1]  # return: parent, child's name


class PrunedFFNAutoGradMask(Function):  # Neuron Movement Pruning (Masking)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        neuron_scores: Tensor = args[0]
        k = int(scheduler.ffn_threshold * neuron_scores.numel())
        _, topk_indices = torch.topk(neuron_scores, k=k, largest=False)
        mask = torch.ones_like(neuron_scores, dtype=torch.float32)
        mask[topk_indices] = 0
        return mask

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0]

class FFNGroup(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        # shared neuron scores btn fc1 & fc2 in this layer.
        self.neuron_scores = nn.Parameter(torch.empty((output_dim,)))
        torch.nn.init.uniform(self.neuron_scores)

    def gen_neuron_mask(self):
        return PrunedFFNAutoGradMask.apply(self.neuron_scores)

class PrunedFFNConv1D(ABCConv1D):   # Neuron Movement Pruning
    def __init__(self, module: Conv1D, grp: FFNGroup, is_fc1: bool):  # fc1/fc2
        super().__init__(module.weight.shape[1], module.weight.shape[0])
        self.weight = module.weight
        self.bias = module.bias
        self.grp = grp
        self.is_fc1 = is_fc1

    def gen_masked_weight(self):
        mask = self.grp.gen_neuron_mask()
        if self.is_fc1:
            return mask.view(1, -1) * self.weight
        else:
            return mask.view(-1, 1) * self.weight

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            output = self.linear(input, self.gen_masked_weight())
        else:
            with torch.no_grad():
                output = self.linear(input, self.weight)
        return output

    def freeze(self):
        with torch.no_grad():
            self.weight = Parameter(self.gen_masked_weight())


class PrunedAttnAutoGradMask(Function):  # Attn Movement Pruning (Masking)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        head_scores: Tensor = args[0]
        k = int(scheduler.attn_threshold * head_scores.numel())
        _, topk_indices = torch.topk(head_scores, k=k, largest=False)
        mask = torch.ones_like(head_scores, dtype=torch.float32)
        mask[topk_indices] = 0
        # special: gpt2 has unified qkv
        mask = torch.cat([mask, mask, mask])
        return mask

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0][:grad_outputs[0].shape[0] // 3]

class AttnGroup(nn.Module):
    def __init__(self, n_head: int, head_size: int):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size

        self.head_scores = nn.Parameter(torch.empty((n_head,)))
        torch.nn.init.uniform(self.head_scores)

    def gen_head_mask(self) -> Tensor:
        return PrunedAttnAutoGradMask.apply(self.head_scores)

class PrunedAttnConv1D(ABCConv1D):   # Attn Movement Pruning
    def __init__(self, module: Conv1D, grp: AttnGroup):
        super().__init__(module.weight.shape[1], module.weight.shape[0])
        self.weight = module.weight
        self.bias = module.bias
        self.grp = grp
        assert self.weight.shape[0] % self.grp.head_size == 0, "sanity"

    def expand(self):
        # rows: [1, 0, 1, 0] -> [1 * head_sz, 0 * head_sz, 1 * head_sz, 0 * head_sz]
        head_mask = self.grp.gen_head_mask()
        expanded_head_mask = head_mask.repeat_interleave(self.grp.head_size)
        # logging.info(f"{head_mask}, {expanded_head_mask}")
        return expanded_head_mask

    def gen_masked_weight(self):
        return self.expand().view(1, -1) * self.weight

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            output = self.linear(input, self.gen_masked_weight())
        else:
            with torch.no_grad():
                output = self.linear(input, self.weight)
        return output


FC1_NAME, FC2_NAME, SPECIAL_NAME = "c_fc", "c_proj", "mlp"
KQV_NAME, DENSE_NAME = "c_attn", "c_proj"
N_HEAD_NAME, HEAD_SIZE_NAME, TOTAL_NAME, SPLIT_NAME = "num_heads", "head_dim", "embed_dim", "split_size"

def convert_model(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    for full_name, module in model.named_modules():
        if hasattr(module, N_HEAD_NAME):   # this is a BartAttention layer
            n_head = getattr(module, N_HEAD_NAME)
            head_size = getattr(module, HEAD_SIZE_NAME)

            kqv_proj: Conv1D = getattr(module, KQV_NAME)

            grp = AttnGroup(n_head, head_size)

            kqv_linear = PrunedAttnConv1D(kqv_proj, grp)

            setattr(module, KQV_NAME, kqv_linear)

        elif hasattr(module, FC1_NAME):

            fc1: Conv1D = getattr(module, FC1_NAME)
            fc2: Conv1D = getattr(module, FC2_NAME)

            grp = FFNGroup(fc1.weight.shape[1])

            new_fc1 = PrunedFFNConv1D(fc1, grp, True)
            new_fc2 = PrunedFFNConv1D(fc2, grp, False)

            setattr(module, FC1_NAME, new_fc1)
            setattr(module, FC2_NAME, new_fc2)

    return model

class SchedulerUpdateCallback(TrainerCallback):
    def __init__(self):
        self.output_area = widgets.Output()
        display(self.output_area)

    def on_step_begin(self, args, state, control, **kwargs):
        cur_step = state.global_step
        scheduler.update(cur_step)

    def on_log(self, args, state, control, **kwargs):
        assert len(state.log_history) >= 1, "sanity"

        state.log_history[-1].update({"ffn_threshold": scheduler.ffn_threshold,
                                      "attn_threshold": scheduler.attn_threshold,
                                      "progress": scheduler.progress})
        df = pd.DataFrame(state.log_history).drop(columns=["grad_norm", "learning_rate", "epoch"])

        with self.output_area:
            clear_output(wait=True)
            # display(df.to_html(max_rows=None))
            with pd.option_context('display.max_rows', None):
                display(df)


def FFN_freeze(model: nn.Module):
    for full_name, module in model.named_modules():
        if isinstance(module, PrunedFFNConv1D):
            module.freeze()

            new_conv1d = Conv1D(module.weight.shape[1], module.weight.shape[0])
            new_conv1d.weight = Parameter(module.weight)
            new_conv1d.bias = Parameter(module.bias) if module.bias is not None else None
            parent, name = find_module_last_parent(model, full_name)
            setattr(parent, name, new_conv1d)

            del module

def print_linear_density(full_name: str, linear: nn.Module):
    with torch.no_grad():
        for p_name, p in linear.named_parameters():
            if p_name == "weight":
                assert p.ndim == 2, f"Not (output_dim * input_dim): {p.shape}"
                # row density:
                rows = (p != 0).sum(dim=1)  # unify all cols
                useful_rows = (rows != 0).sum(dim=0)  # unify the vector
                # col density:
                cols = (p != 0).sum(dim=0)  # unify all rows
                useful_cols = (cols != 0).sum(dim=0)  # unify the vector
                # print
                print(f"{full_name}: row density: {(useful_rows / p.shape[0]):.1f}; col density: {(useful_cols / p.shape[1]):.1f}")

def print_linear_density_all(model: nn.Module):
    for full_name, module in model.named_modules():
        if isinstance(module, Conv1D):
            print_linear_density(full_name, module)

def FFN_prune_zeros(model: nn.Module):
    for full_name, module in model.named_modules():
        if isinstance(module, Conv1D):
            with torch.no_grad():
                if FC1_NAME in full_name:
                    cols = (module.weight != 0).sum(dim=0)
                    useful_cols = torch.where(cols != 0)[0].tolist()
                    new_weight = module.weight[:, useful_cols]
                    # print(module.weight.shape, new_weight.shape)
                    print(f"fc1: weight: {module.weight.shape} -> new_weight: {new_weight.shape}")
                    new_conv1d = Conv1D(new_weight.shape[1], new_weight.shape[0])
                    new_conv1d.weight = Parameter(new_weight.contiguous())
                    new_conv1d.bias = Parameter(module.bias[useful_cols].contiguous()) if module.bias is not None else None
                elif FC2_NAME in full_name and SPECIAL_NAME in full_name:   # there are two "c_proj"s: from attn and mlp. We only need the one from mlp.
                    rows = (module.weight != 0).sum(dim=1)
                    useful_rows = torch.where(rows != 0)[0].tolist()
                    new_weight = module.weight[useful_rows, :]
                    # print(module.weight.shape, new_weight.shape)
                    print(f"fc2: weight: {module.weight.shape} -> new_weight: {new_weight.shape}")
                    new_conv1d = Conv1D(new_weight.shape[1], new_weight.shape[0])
                    new_conv1d.weight = Parameter(new_weight.contiguous())
                    new_conv1d.bias = Parameter(module.bias)
                else:
                    continue

                parent, name = find_module_last_parent(model, full_name)
                setattr(parent, name, new_conv1d)

def prune_kqv_heads(kqv: Conv1D, useful_indices: List[int]) -> Conv1D:
    new_weight = Parameter(kqv.weight[:, useful_indices].clone().detach().contiguous())
    new_bias = None
    if kqv.bias is not None:
        new_bias = Parameter(kqv.bias[useful_indices].clone().detach().contiguous())
    new_layer = Conv1D(new_weight.shape[1], new_weight.shape[0])
    new_layer.weight = new_weight
    new_layer.bias = new_bias
    return new_layer

def prune_dense_heads(dense: Conv1D, useful_indices: List[int]) -> Conv1D:
    # nearly the same as above. too lazy to refactor. :)
    new_weight = Parameter(dense.weight[useful_indices].clone().detach().contiguous())
    new_bias = None
    if dense.bias is not None:
        new_bias = Parameter(dense.bias.clone().detach().contiguous())  # Note: output_dim doesn't change.
    new_layer = Conv1D(new_weight.shape[1], new_weight.shape[0])
    new_layer.weight = new_weight
    new_layer.bias = new_bias
    return new_layer

def ATTN_prune_zeros(model: nn.Module):
    with torch.no_grad():
        for full_name, module in model.named_modules():
            if hasattr(module, N_HEAD_NAME):   # this is an Attn layer
                n_head = getattr(module, N_HEAD_NAME)
                head_size = getattr(module, HEAD_SIZE_NAME)

                kqv_proj: PrunedAttnConv1D = getattr(module, KQV_NAME)

                # print(n_head, head_size)

                zero_mask = kqv_proj.grp.gen_head_mask() == 0
                # print(zero_mask)

                if zero_mask.any():
                    useless_heads = torch.nonzero(zero_mask, as_tuple=True)[0].tolist()
                    print(f"{full_name}: useless_heads: {useless_heads}")
                    assert len(useless_heads) % 3 == 0, f"bundled k,q,v {len(useless_heads)}"
                    # head (e.g.: 3) -> indices (e.g.: [3*head_size, 4*head_size) )
                    useless_indices = torch.cat([torch.arange(h * head_size, (h + 1) * head_size) for h in useless_heads])
                    # print(useless_heads, useless_indices)
                    useful_indices = list(set(range(n_head * head_size * 3)) - set(useless_indices.tolist()))
                    # print(useful_indices)

                    new_kqv_proj = prune_kqv_heads(kqv_proj, useful_indices)
                    setattr(module, KQV_NAME, new_kqv_proj)

                    dense_proj = getattr(module, DENSE_NAME)
                    new_dense_proj = prune_dense_heads(dense_proj, useful_indices[:len(useful_indices)//3])
                    setattr(module, DENSE_NAME, new_dense_proj)

                    # Update metadata
                    new_n_head = (n_head * 3 - len(useless_heads)) // 3
                    setattr(module, N_HEAD_NAME, new_n_head)
                    setattr(module, TOTAL_NAME, new_n_head * head_size)
                    setattr(module, SPLIT_NAME, new_n_head * head_size)

def print_pruning_density(model: nn.Module):
    sample_ffn  = [(full_name, module) for full_name, module in model.named_modules() if isinstance(module, PrunedFFNConv1D)]
    sample_ffn = random.sample(sample_ffn, min(5, len(sample_ffn)))

    sample_attn = [(full_name, module) for full_name, module in model.named_modules() if isinstance(module, PrunedAttnConv1D)]
    sample_attn = random.sample(sample_attn, min(5, len(sample_attn)))

    with torch.no_grad():
        for full_name, module in (sample_ffn + sample_attn):
            # print(module.weight.shape, module.bias.shape)
            w = module.gen_masked_weight()
            plt.imshow((w != 0).detach().cpu())
            plt.title(full_name)
            plt.show()

