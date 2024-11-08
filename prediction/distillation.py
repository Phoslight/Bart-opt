from precompiled import *

selected_layers: List[int] = [0,2,5,7,9,11]  # hard-coded. This config is both for encoder and decoder.

def create_student(teacher: GPT2LMHeadModel):
    # create student config from teacher's, with the differences being encoder/decoder layers.
    cfg = GPT2Config.from_pretrained(model_name)
    cfg.n_layer = 6
    # create a smaller student.
    student = GPT2LMHeadModel(cfg)

    # copy the teacher model to the student model.
    def copy_model_layers():
        # cover fully on all parameters.
        student.load_state_dict(teacher.state_dict(), strict=False)
        # deepcopy the target encoder/decoder layers we want to copy.
        teacher_dec_to_copy = nn.ModuleList([teacher.transformer.h[i] for i in selected_layers])
        student.transformer.h.load_state_dict(teacher_dec_to_copy.state_dict(), strict=True)

    copy_model_layers()

    # print(student)
    return student

# hyperparameters
T = 2    # temperature
w_label = 0     # weight of label loss
w_hidden = 3    # weight of hidden loss   # https://arxiv.org/pdf/2010.13002, equation (5)
w_kl = 0.8      # weight of kl loss

class Distiller(nn.Module):

    def __init__(self, teacher: GPT2LMHeadModel, student: GPT2LMHeadModel):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")

    def forward(self,
                input_ids: torch.LongTensor,        # (bs, (enc)seq_len)
                attention_mask: torch.FloatTensor,  # (bs, (enc)seq_len)
                labels: torch.LongTensor,           # (bs, (dec)seq_len)  # summaries are like input_ids
                ):
        self.teacher.eval()
        assert not self.teacher.training, "must be in eval() mode"

        # forward
        t_res = self.teacher.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        s_res = self.student.forward(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # losses
        label_loss = s_res.loss

        hidden_loss = self.calculate_hidden_loss(t_res.hidden_states,
                                                 s_res.hidden_states,
                                                 attention_mask)

        kl_loss = self.calculate_kl_divergence(t_res.logits,
                                               s_res.logits,
                                               attention_mask)

        total_loss = label_loss * w_label + hidden_loss * w_hidden + kl_loss * w_kl

        # return total_loss
        return {"loss": total_loss}


    # Use MSE to calculate hidden layer loss between teacher & student
    def calculate_hidden_loss(self,
                              t_hidden_states,  # [(bs, (enc/dec)seq_len, n_dim) * 12]
                              s_hidden_states,  # [(bs, (enc/dec)seq_len, n_dim) * 6]
                              mask,             # (bs, (enc/dec)seq_len)
                              ):
        t_hidden_states = [t_hidden_states[0]] + [t_hidden_states[i+1] for i in selected_layers]
        s_hidden_states = [s_hidden_states[0]] + [s_hidden_states[i+1] for i in range(len(selected_layers))]

        t_hidden_states = torch.stack(t_hidden_states, dim=0)  # (6, bs, (enc/dec)seq_len, n_dim)
        s_hidden_states = torch.stack(s_hidden_states, dim=0)  # (6, bs, (enc/dec)seq_len, n_dim)

        # compute MSE between hidden states in corresponding layers of the teacher and student.
        t_hidden_states = F.layer_norm(t_hidden_states, t_hidden_states.shape[1:])
        s_hidden_states = F.layer_norm(s_hidden_states, s_hidden_states.shape[1:])

        mse_loss = F.mse_loss(s_hidden_states, t_hidden_states, reduction="none")

        mse_loss = mse_loss * mask.unsqueeze(-1).unsqueeze(0)

        # the final process of reduction="mean".
        n_dim = t_hidden_states.shape[-1]
        return mse_loss.sum() / (mask.sum() * n_dim)

    def calculate_kl_divergence(self,
                                t_logits,   # (bs, (dec)seq_len, vocab_size)
                                s_logits,   # (bs, (dec)seq_len, vocab_size)
                                attn_mask,  # (bs, (dec)seq_len)
                                ):
        attn_mask = attn_mask.to(bool)

        vocab_size = t_logits.shape[-1]

        # (bs, seq_len, vocab_size) -> a flattened vector -> ( <= bs*seq_len, vocab_size)
        t_logits = torch.masked_select(t_logits, attn_mask.unsqueeze(-1)).view(-1, vocab_size)
        s_logits = torch.masked_select(s_logits, attn_mask.unsqueeze(-1)).view(-1, vocab_size)

        loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * T ** 2

        return loss
