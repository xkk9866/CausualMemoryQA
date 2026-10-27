"""Response-Mediated Strategy Network (RMN).

A strategy is a property of the response the supporter is about to give. RMN makes that
mediator explicit: the context attends over a memory of real supporter responses, the
attention is trained to find the responses that fit this context (InfoNCE against the gold
response), and the strategy is read from the attended responses.

    alpha(j | H) = softmax_j( <q(H), k(r_j)> / tau )          over the response memory
    p_med(s | H) = sum_j alpha(j | H) y_j                     strategies of attended responses
    v(H)         = sum_j alpha(j | H) W_v r_j                 the anticipated response
    p_par(s | H) = softmax( MLP([h; v]) )
    p(s | H)     = g(H) p_par + (1 - g(H)) p_med
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class RMN(nn.Module):
    def __init__(self, backbone, bank, bank_label, bank_dialog, k, d=256, decoder=False,
                 lora=None, smooth=0.1, use_med=True, use_value=True, dtype=torch.float32):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone, torch_dtype=dtype)
        if lora is not None:
            from peft import get_peft_model
            self.enc.gradient_checkpointing_enable()
            self.enc.enable_input_require_grads()
            self.enc = get_peft_model(self.enc, lora)
        else:
            self.enc.gradient_checkpointing_enable()
        self.decoder = decoder
        hd = self.enc.config.hidden_size
        rd = bank.shape[1]
        self.register_buffer("bank", bank.float())
        self.register_buffer("bank_dialog", bank_dialog.long())
        y = F.one_hot(bank_label.long(), k).float()
        self.register_buffer("bank_y", y * (1 - smooth) + smooth / k)
        self.k, self.use_med, self.use_value = k, use_med, use_value
        self.q = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, d))
        self.key = nn.Linear(rd, d)
        self.val = nn.Linear(rd, d)
        self.log_tau = nn.Parameter(torch.tensor(-3.0))  # tau ~ 0.05 on unit vectors
        self.h = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, d))
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, k))
        self.gate = nn.Linear(2 * d, 1)

    def pool(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.decoder:  # left padding: the last position is the last real token
            return out[:, -1].float()
        return out[:, 0].float()

    def forward(self, input_ids, attention_mask, dialog=None, self_index=None):
        h = self.pool(input_ids, attention_mask)
        q = F.normalize(self.q(h), dim=-1)
        keys = F.normalize(self.key(self.bank), dim=-1)
        score = q @ keys.t() / self.log_tau.exp().clamp(min=1e-3)
        out = {"score": score}
        if dialog is not None:  # training: a dialogue may not read its own responses
            med_score = score.masked_fill(dialog[:, None] == self.bank_dialog[None, :], float("-inf"))
        else:
            med_score = score
        alpha = med_score.softmax(-1)
        hp = self.h(h)
        v = alpha @ self.val(self.bank) if self.use_value else torch.zeros_like(hp)
        z = torch.cat([hp, v], -1)
        p_par = self.head(z).softmax(-1)
        if self.use_med:
            p_med = alpha @ self.bank_y
            g = torch.sigmoid(self.gate(z))
            p = g * p_par + (1 - g) * p_med
            out.update(p_med=p_med, gate=g)
        else:
            p = p_par
        out["logp"] = torch.log(p.clamp(min=1e-8))
        return out
