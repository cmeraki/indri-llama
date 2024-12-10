import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
import inspect 

class LlamaConfig:
    def __init__(self, 
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        vocab_size: int = -1,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 500000,
        use_scaled_rope: bool = False,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.use_scaled_rope = use_scaled_rope
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.dropout = dropout

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_scaling(freqs: torch.Tensor):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_kv_heads
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.cache_k = torch.zeros(
            (config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if start_pos >= 0:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys, values = xk, xv

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=4 * config.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = torch.nn.ModuleList()
        
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(config, layer_id))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index=-100):
        _bsz, seqlen = inputs.shape
        h = self.tok_embeddings(inputs)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.type_as(h)
        start_pos = -1 
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        loss = F.cross_entropy(
            input=logits.transpose(1, 2), target=targets, ignore_index=ignore_index)
        return loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Separate weight decay for 2D params
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Print parameter information
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @classmethod
    def from_pretrained(cls, model_type, weights=None):
        """
        Load pretrained Llama model weights
        Note: Implementation will depend on specific pretrained checkpoints
        """
        from transformers import LlamaForCausalLM

        # Define model configurations (example, adjust as needed)
        config_args = {
            'llama-7b': dict(n_layers=32, dim=4096, n_heads=32),
            'llama-13b': dict(n_layers=40, dim=5120, n_heads=40),
            'llama-33b': dict(n_layers=60, dim=6656, n_heads=52),
            'llama-1b': dict(n_layers=12, dim=1024, n_heads=16),
            # Add more model types as needed
        }

        assert model_type in config_args, f"Unsupported model type: {model_type}"

        # Create configuration
        config_args[model_type]['vocab_size'] = 32000  # Typical Llama vocab size
        config_args[model_type]['max_seq_len'] = 2048  # Typical sequence length
        config = LlamaConfig(**config_args[model_type])

        # Create model
        model = cls(config)

        # Load HuggingFace model
        model_hf = LlamaForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        # Mapping and copying weights (simplified, may need adjustments)
        for k, v in sd_hf.items():
            if k in sd:
                sd[k].copy_(v)

        return model

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """
        Generate sequence with similar logic to GPT generation method
        Adapted for Llama's specific forward methods
        """
        start_pos = 0
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]           
            logits = self.forward_inference(idx_cond, start_pos)            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)

            if stop_token is not None and idx_next == stop_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            start_pos += 1

        return idx

    def expand_vocab(self, new_vocab_size):
        print(f"Updating embeddings {self.vocab_size, self.config.dim} => {new_vocab_size, self.config.dim}")
        
        old_embeddings = self.tok_embeddings.weight
        new_embeddings = torch.Tensor(new_vocab_size, self.config.dim).to(old_embeddings.device)

        mu = torch.mean(old_embeddings, dim=0)
        sigma = ((old_embeddings - mu).T @ (old_embeddings - mu)) / self.vocab_size
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5 * sigma)

        new_embeddings[:self.vocab_size] = self.tok_embeddings.weight
        new_embeddings[self.vocab_size:] = torch.stack(
            tuple((dist.sample() for _ in range(new_vocab_size - self.vocab_size))), 
            dim=0
        )

        self.tok_embeddings = nn.Embedding(_weight=new_embeddings, 
                                           num_embeddings=new_vocab_size, 
                                           embedding_dim=self.config.dim)
        self.output = nn.Linear(self.config.dim, new_vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight

        self.config.vocab_size = new_vocab_size
        self.vocab_size = new_vocab_size


def get_model(
        model_type='llama-7b',
        vocab_size=32000,
        dropout=0.0,
        max_seq_len=2048,
        bias=False,
        device='cpu',
        compile=True,
        path=None
    ):

    config_args = {
        'llama-7b': dict(
            n_layers=32, 
            dim=4096, 
            n_heads=32, 
            rope_theta=10000.0, 
            norm_eps=1e-5,
            use_scaled_rope=True
        ),
        'llama-13b': dict(
            n_layers=40, 
            dim=5120, 
            n_heads=40, 
            rope_theta=10000.0, 
            norm_eps=1e-5,
            use_scaled_rope=True
        ),
        'llama-33b': dict(
            n_layers=60, 
            dim=6656, 
            n_heads=52, 
            rope_theta=10000.0, 
            norm_eps=1e-5,
            use_scaled_rope=True
        ),
        'llama-1b': dict(
            n_layers=12,
            dim=1024,  
            n_heads=16,   
            rope_theta=10000.0,
            norm_eps=1e-5,
            use_scaled_rope=True
        )
    }[model_type]

    model_args = dict(
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        rope_theta=config_args.get('rope_theta', 10000.0),
        norm_eps=config_args.get('norm_eps', 1e-5),
        use_scaled_rope=config_args.get('use_scaled_rope', True)
    )
    model_args.update(config_args)

    if path:
        checkpoint = torch.load(path, map_location=device)
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                model_args.update(config)
                llamaconf = LlamaConfig(**model_args)
            else:
                llamaconf = config
        else:
            llamaconf = LlamaConfig(**model_args)
    else:
        llamaconf = LlamaConfig(**model_args)

    print("MODEL CONFIG: ", llamaconf)

    model = Llama(llamaconf)

    # Load weights if path is provided
    if path:
        state_dict = checkpoint['model']
        
        # Remove any unwanted prefixes (common in distributed training)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
    
    model.to(device)

    if compile:
        print("compiling the model... (takes a ~minute)")
        torch.compile(model)

    return model