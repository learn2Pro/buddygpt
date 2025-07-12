import torch
import torch.nn.functional as F
import torch.nn as nn

from dataclasses import dataclass
from transformers import PretrainedConfig, AutoConfig, AutoModelForCausalLM
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
from transformers.generation import GenerationConfig
from transformers.generation import utils
from dataclasses import dataclass


# class GPTConfig(PretrainedConfig):
#     model_type = "buddygpt"

#     def __init__(
#         self,
#         n_block=1024,
#         n_layer=16,
#         n_head=16,
#         n_kv_head=8,
#         n_embed=1536,
#         n_vocab=151669,
#         pad_token_id=151643,
#         eos_token_id=151645,
#         tie_word_embeddings=True,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.n_block = n_block
#         self.n_layer = n_layer
#         self.n_head = n_head
#         self.n_embed = n_embed
#         self.n_vocab = n_vocab
#         self.n_kv_head = n_kv_head
#         self.pad_token_id = pad_token_id
#         self.eos_token_id = eos_token_id
#         self.tie_word_embeddings = tie_word_embeddings
        
@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "buddygpt"
    
    n_block:int = 1024
    n_layer:int = 16
    n_head:int = 16
    n_kv_head:int = 8
    n_embed:int = 1536
    n_vocab:int = 151669
    pad_token_id:int = 151643
    eos_token_id:int = 151645
    tie_word_embeddings = True

    # mla
    q_lora_rank: int = 16
    qk_rope_head_dim: int = 4
    kv_lora_rank: int = 16
    v_head_dim: int = 16
    qk_nope_head_dim: int = 12


device = "cuda" if torch.cuda.is_available() else "cpu"
FLASH = 0

import torch.nn as nn
import torch


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """ 旋转位置编码
            - dim (int): 旋转嵌入的维度大小。
            - max_position_embeddings (int): 预计算的最大位置嵌入数，默认为2048。
            - base (int): 用于计算逆频率的基本频率，默认为10000。
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算逆频率值，并将其注册为模型的缓冲区
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了支持`torch.jit.trace`功能，立即计算预存储的余弦和正弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """ 预计算的余弦和正弦缓存
        """
        self.max_seq_len_cached = seq_len
        # 创建一个从0到最大序列长度-1的整数张量，与 inv_freq 具有相同的设备和数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算每个位置与每个维度的频率，形成频谱矩阵
        freqs = torch.outer(t, self.inv_freq)
        
        # 不同于论文中的实现，这里采用了不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """ 旋转输入一半的 hidden dim
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """ 在 qk 应用旋转位置编码

    Args:
        q (`torch.Tensor`): q
        k (`torch.Tensor`): k
        cos (`torch.Tensor`): 旋转位置嵌入的余弦部分
        sin (`torch.Tensor`): 旋转位置嵌入的正弦部分
        position_ids (`torch.Tensor`): 与q和k对应位置的标记索引。例如，在处理KV缓存时，可以使用偏移过的位置ID。
        unsqueeze_dim (`int`, *optional*, defaults to 1): 'unsqueeze_dim' 参数指定了沿哪个维度对 cos[position_ids] 
            和 sin[position_ids] 进行扩展，以便它们能够适当地广播到 q 和 k 的维度上。
            例如，注意 cos[position_ids] 和 sin[position_ids] 具有形状 [batch_size, seq_len, head_dim]。
            那么，如果 q 和 k 的形状分别为 [batch_size, heads, seq_len, head_dim]，
            则设置 unsqueeze_dim=1 可使 cos[position_ids] 和 sin[position_ids] 可以广播到 q 和 k 的形状上。
            同样地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则应将 unsqueeze_dim 设置为 2
    Returns:
        包含使用旋转位置嵌入变换后的q和k张量的 `tuple(torch.Tensor)`。
    """
    # print("ori cos: ", cos.shape)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # print("q: ", q.shape)
    # print("cos: ", cos.shape)
    # print("sin: ", sin.shape)
    # print("rotate_half: ", rotate_half(q).shape)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # split last dim into two
        return F.silu(x1) * x2  # silu == swish


class GQA(nn.Module):
    def __init__(self, config, rope=None):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embed = config.n_embed
        self.head_dim = self.n_embed // self.n_head
        self.kv_head_dim = self.head_dim * self.n_kv_head
        self.repeat_factor = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(self.n_embed, self.n_embed)
        self.k_proj = nn.Linear(self.n_embed, self.kv_head_dim)
        self.v_proj = nn.Linear(self.n_embed, self.kv_head_dim)
        self.out_proj = nn.Linear(self.n_embed, self.n_embed)
        self.rope = RotaryEmbedding(self.head_dim) if rope is None else rope
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.n_block, config.n_block)).view(
                1, 1, config.n_block, config.n_block
            ),
        )

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, -1)  # B, T, n_head, n_embed
        k = self.k_proj(x).view(B, T, self.n_kv_head, -1)  # B, T, n_kv_head, n_embed
        v = self.v_proj(x).view(B, T, self.n_kv_head, -1)  # B, T, n_kv_head, n_embed

        xq = xq.transpose(1, 2)  # B, n_head, T, n_embed
        xk = xk.transpose(1, 2)  # B, n_kv_head, T, n_embed
        xv = v.transpose(1, 2)  # B, n_kv_head, T, n_embed

        cos, sin = self.rope(xq, T)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if self.repeat_factor > 1:
            xk = xk.repeat_interleave(self.repeat_factor, dim=1)  # B, n_head, T, n_embed
            xv = xv.repeat_interleave(self.repeat_factor, dim=1)  # B, n_head, T, n_embed

        if FLASH:
            # print('this way')
            o_attn = F.scaled_dot_product_attention(xq.contiguous(), xk.contiguous(), xv.contiguous(), is_causal=True)
        else:
            # print('this way2')
            qk = torch.matmul(xq, xk.transpose(-2, -1))
            qk = qk.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            qk = F.softmax(qk, dim=-1) * (self.n_embed**-0.5)
            o_attn = qk @ xv  # B, n_head, T, n_embed
        o_attn = o_attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(o_attn)


class SelfCausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.n_block = config.n_block
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.rope = RotaryEmbedding(dim=config.n_embed)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.n_block, config.n_block)).view(
                1, 1, config.n_block, config.n_block
            ),
        )

    def forward(self, x):
        B, T, _ = x.size()
        attn = self.c_attn(x)
        q, k, v = attn.split(self.n_embed, dim=-1)  # B,n_block,n_embed
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)  # B, n_head, n_block, n_embed//n_head
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)  # B, n_head, n_block, n_embed//n_head
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)  # B, n_head, n_block, n_embed//n_head
        cos, sin = self.rope(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if FLASH:
            o_attn = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(), is_causal=True
            )
        else:
            qk = q @ k.transpose(-2, -1)
            qk = qk.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            o_attn = (F.softmax(qk, dim=-1) * (self.n_embed**-0.5)) @ v
        o_attn = o_attn.transpose(1, 2).view(B, T, -1).contiguous()
        y = self.c_proj(o_attn)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.ln1 = nn.Linear(config.n_embed, 2 * config.n_embed)
        self.silu = SwiGLU()
        self.ln2 = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = self.silu(x)
        x = self.ln2(x)
        return x


class Layer(nn.Module):
    def __init__(self, config, rope=None):
        super().__init__()
        self.mha = GQA(config, rope)
        self.mlp = MLP(config)
        self.pre_norm = nn.RMSNorm(config.n_embed)
        self.post_norm = nn.RMSNorm(config.n_embed)

    def forward(self, x):
        x = x + self.mha(self.pre_norm(x))
        x = x + self.mlp(self.post_norm(x))
        return x


class BuddyGPT(PreTrainedModel):
    config_class = GPTConfig
    # 定义了模型内部子模块命名的基础前缀，当加载或保存模型时，这个前缀将用于识别模型主体部分。
    base_model_prefix = "model"
    # 表明该模型支持梯度检查点技术，这是一种内存优化策略，可减少模型训练时所需的显存
    supports_gradient_checkpointing = True
    # Scaled Dot Product Attention (SDPA)
    _supports_sdpa = True
    # 表示模型支持缓存机制，这在自回归模型（如Transformer解码器）中很常见，
    # 用于存储先前计算的结果以加快后续时间步长的计算速度。
    _supports_cache_class = True

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.n_vocab = config.n_vocab
        rope = RotaryEmbedding(config.n_embed // config.n_head)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.n_vocab, config.n_embed, config.pad_token_id),
                layers=nn.ModuleList(
                    [Layer(config, rope) for _ in range(config.n_layer)]
                ),
                ln_norm=nn.RMSNorm(config.n_embed),
                # rope = rope,
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)
        self.eos_token_id = config.eos_token_id
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight # https://paperswithcode.com/method/weight-tying
        #  =
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / (2 * self.config.n_layer) ** 0.5
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=std, generator=self.init_rng
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02, generator=self.init_rng
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, labels=None, **kwargs):
        input_ids = input_ids.to(device)
        B, T = input_ids.size()
        # pos = torch.arange(0, T, dtype=torch.long, device=device)
        token_embed = self.transformer.wte(input_ids)
        # pos_embed = self.transformer.wpe(pos)
        x = token_embed
        for layer in self.transformer.layers:
            x = layer(x)
        x = self.transformer.ln_norm(x)

        if labels is not None:
            labels = labels.to(device)
            logits = self.lm_head(x)
            shape_logits = logits[:, :-1, :].contiguous().view(-1, self.n_vocab)
            targets = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shape_logits, targets, ignore_index=-100)
        else:
            logits = self.lm_head(x)  # B, 1, n_vocab
            loss = None
        return (
            CausalLMOutputWithPast(loss=loss, logits=logits)
            if loss
            else CausalLMOutputWithPast(logits=logits)
        )

    @torch.no_grad()
    def generate(
        self, input_ids, max_new_tokens=512, temperature=1.0, top_k=None, **kwargs
    ):
        x = input_ids
        for _ in range(max_new_tokens):
            idx_cond = (
                x if x.size(1) <= self.config.n_block else x[:, -self.config.n_block :]
            )
            logits = self(idx_cond).logits
            if temperature == 0.0:
                # 当temperature为0时，选取概率最高的单一索引
                _, predict = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits[:, -1, :] / temperature  # last token
                # 如果指定了top_k参数，保留top_k个概率最高的选项
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)  # B, n_vocab
                predict = torch.multinomial(probs, num_samples=1)  # B, 1

            x = torch.cat([x, predict], dim=-1)
            if self.eos_token_id and self.eos_token_id == predict[-1][-1].item():
                break
        return x

    # def generate(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     generation_config: Optional[GenerationConfig] = None,
    #     streamer=None,
    #     **kwargs,
    # ):
    #     return super().generate(
    #         inputs=input_ids,
    #         generation_config=generation_config,
    #         streamer=streamer,
    #         **kwargs,
    #     )


AutoConfig.register("buddygpt", GPTConfig)
AutoModelForCausalLM.register(GPTConfig, BuddyGPT)
AutoModelForCausalLM = BuddyGPT

