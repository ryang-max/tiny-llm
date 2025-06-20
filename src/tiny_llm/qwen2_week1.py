import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(
            self.dim,
            self.max_seq_len,
            self.theta
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        #print(f"x shape: {x.shape}, wq shape: {self.wq.shape}, wk shape: {self.wk.shape}, wv shape: {self.wv.shape}")
        q = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, -1)
        k = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, -1)
        v = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, -1)
        #print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        q = self.rope(q, slice(offset, offset + L))
        k = self.rope(k, slice(offset, offset + L))
        #print(f"q shape after rope: {q.shape}, k shape after rope: {k.shape}")
        attn = scaled_dot_product_attention_grouped(
            q.transpose(0, 2, 1, 3).astype(mx.float32),
            k.transpose(0, 2, 1, 3).astype(mx.float32),
            v.transpose(0, 2, 1, 3).astype(mx.float32),
            scale=self.scale,
            mask=mask.astype(mx.float32) if mask is mx.array else None if mask is None else mask,
        ).transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size).astype(x.dtype)
        out = linear(attn, self.wo)
        return out



class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        out = mx.multiply(gate, up)
        out = linear(out, self.w_down)
        return out



class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.attn = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            max_seq_len,
            theta,
        )
        self.input_layernorm = RMSNorm(
            hidden_size,
            w_input_layernorm,
            rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            rms_norm_eps,
        )
        self.mlp = Qwen2MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        out = self.input_layernorm(x)
        out = self.attn(out, offset, mask)
        out1 = x + out
        out = self.post_attention_layernorm(out1)
        out = self.mlp(out)
        out2 = out1 + out
        return out2



class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        print(mlx_model.args)
        #print(mlx_model.model)
        self.embedding = Embedding(
            vocab_size=mlx_model.args.vocab_size,
            embedding_dim=mlx_model.args.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
        )
        self.layers = [
            Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj).astype(mx.float16),
                wk=dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj).astype(mx.float16),
                wv=dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj).astype(mx.float16),
                wo=dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj).astype(mx.float16),
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(mx.float16),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(mx.float16),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(mx.float16),
                w_gate=dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj).astype(mx.float16),
                w_up=dequantize_linear(mlx_model.model.layers[i].mlp.up_proj).astype(mx.float16),
                w_down=dequantize_linear(mlx_model.model.layers[i].mlp.down_proj).astype(mx.float16),
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight.astype(mx.float16),
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight.astype(mx.float16),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            for i in range(mlx_model.args.num_hidden_layers)
        ]
        self.rms_norm = RMSNorm(
            mlx_model.args.hidden_size,
            mlx_model.model.norm.weight.astype(mx.float16),
            mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        x = self.embedding(inputs)
        for layer in self.layers:
            x = layer(x, offset, mask="causal" if x.shape[1] > 1 else None)
        x = self.rms_norm(x)
        if self.w_lm_head is not None:
            return linear(x, self.w_lm_head)
        else:
            return self.embedding.as_linear(x)
