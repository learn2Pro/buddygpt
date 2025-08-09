from transformers.configuration_utils import PretrainedConfig
from loguru import  logger as logging


class BuddyGPTConfig(PretrainedConfig):
    """ TinyLLM 配置文件
    """
    
    model_type = "buddygpt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self, 
        vocab_size=151669,
        hidden_size=4096,
        intermediate_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        num_seq_len=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        rope_theta=100000.0,
        attention_dropout=0.0,
        attn_impl="mla",
        _attn_implementation="sdpa",
        q_lora_rank: int = 16,
        qk_rope_head_dim: int = 4,
        qk_nope_head_dim: int = 12,
        kv_lora_rank: int = 16,
        v_head_dim: int = 16,
        n_expert=None,
        n_expert_per_token=2,
        n_group=2,
        n_topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=0.2,
        scoring_func='sigmoid',
        topk_method='noaux_tc',
        n_shared_experts=2,
        moe_intermediate_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.num_seq_len = num_seq_len
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
            
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout   
        self.attn_impl = attn_impl

        # mla
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim

        # moe
        self.n_expert = n_expert
        self.n_expert_per_token = n_expert_per_token
        self.n_group = n_group
        self.n_topk_group = n_topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor=routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = n_shared_experts
            
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )