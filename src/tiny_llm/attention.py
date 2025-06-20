import mlx.core as mx
from .basics import softmax, linear
import einops


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = mx.rsqrt(query.shape[-1])
    if mask is None:
        mask = mx.zeros_like(mx.matmul(query, mx.einsum("... L D -> ... D L", key)))
    return mx.matmul(
        softmax(
            mx.matmul(query, mx.einsum("... L D -> ... D L", key)) * scale + mask,
            axis=-1,
        ),
        value,
    )


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.dim)
            .transpose(0, 2, 1, 3)
        )
        attn = scaled_dot_product_attention_simple(q, k, v, mask=mask)
        return linear(
            attn.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size), self.wo
        )


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.full([L, S], -mx.inf, dtype=dtype)
    #print(f"mask shape: {mask.shape}, dtype: {mask.dtype}")
    return mx.triu(mask, k=1 + max(0, S - L))


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    assert H_q % H == 0, "Query heads must be a multiple of key heads"
    n_repeats = H_q // H
    query = query.reshape(*query.shape[:-3], H, n_repeats, L, D)
    key = key.reshape(*key.shape[:-3], H, 1, S, D)
    value = value.reshape(*value.shape[:-3], H, 1, S, D)
    if isinstance(mask, mx.array):
        mask = mask.reshape(*mask.shape[:-3], H, n_repeats, L, S)
    elif isinstance(mask, str) and mask == "causal":
        mask = causal_mask(L, S, query.dtype)
        mask = mask.reshape(*((1,) * (len(query.shape) - 2)), L, S)
        #print(f"mask final shape: {mask.shape}, dtype: {mask.dtype}")
    attn = scaled_dot_product_attention_simple(
        query, key, value, scale=scale, mask=mask
    )
    return attn.flatten(-4, -3)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
