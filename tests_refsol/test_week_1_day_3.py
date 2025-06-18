import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *
from mlx_lm import load
from .tiny_llm_base import qwen2_week1
from .tiny_llm_base import dequantize_linear


def grouped_attention_helper(
    stream: mx.Stream,
    precision: mx.Dtype,
    batch_dimension: int,
    scale: float | None,
    is_causal_mask: bool,
):
    with mx.stream(stream):
        H_q = 18
        H = 6
        L = 3
        D = 5
        S = 7
        BATCH = 10
        BATCH_2 = 2
        if batch_dimension == 0:
            q_shape = (H_q, L, D)
            kv_shape = (H, S, D)
            mask_shape = (H_q, L, S)
        elif batch_dimension == 1:
            q_shape = (BATCH, H_q, L, D)
            kv_shape = (BATCH, H, S, D)
            mask_shape = (BATCH, H_q, L, S)
        elif batch_dimension == 2:
            q_shape = (BATCH_2, BATCH, H_q, L, D)
            kv_shape = (BATCH_2, BATCH, H, S, D)
            mask_shape = (BATCH_2, BATCH, H_q, L, S)
        for _ in range(100):
            query = mx.random.uniform(shape=q_shape, dtype=precision)
            key = mx.random.uniform(shape=kv_shape, dtype=precision)
            value = mx.random.uniform(shape=kv_shape, dtype=precision)
            mask = mx.random.uniform(shape=mask_shape, dtype=precision)

            reference_output = mx.fast.scaled_dot_product_attention(
                q=query.reshape(-1, H_q, L, D),
                k=key.reshape(-1, H, S, D),
                v=value.reshape(-1, H, S, D),
                scale=scale if scale is not None else (1.0 / (D**0.5)),
                mask=mask.reshape(-1, H_q, L, S) if not is_causal_mask else "causal",
            )
            # Reshape reference output back to original shape
            reference_output = reference_output.reshape(query.shape)
            user_output = scaled_dot_product_attention_grouped(
                query,
                key,
                value,
                scale=scale,
                mask=mask if not is_causal_mask else "causal",
            )

            assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_1_grouped_attention(
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, False)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_mask_only_same_dim(
    stream: mx.Stream,
):
    with mx.stream(stream):
        L = 3
        S = 3
        user_output = causal_mask(
            L,
            S,
            mx.float32,
        )
        assert_allclose(
            user_output,
            mx.array(
                [
                    [0, -mx.inf, -mx.inf],
                    [0, 0, -mx.inf],
                    [0, 0, 0],
                ]
            ),
            precision=mx.float32,
        )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_task_2_mask_only_different_dim(
    stream: mx.Stream,
):
    with mx.stream(stream):
        L = 3
        S = 5
        user_output = causal_mask(
            L,
            S,
            mx.float32,
        )
        assert_allclose(
            user_output,
            mx.array(
                [
                    [0, 0, 0, -mx.inf, -mx.inf],
                    [0, 0, 0, 0, -mx.inf],
                    [0, 0, 0, 0, 0],
                ]
            ),
            precision=mx.float32,
        )


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_2_grouped_attention_causal_mask(
    stream: mx.Stream, precision: mx.Dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(stream, precision, batch_dimension, scale, True)

@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_3_qwen2_grouped_query_attention():
    mlx_model, _ = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    print(mlx_model.args)
    #print(mlx_model.model)
    wq = dequantize_linear(mlx_model.model.layers[0].self_attn.q_proj).astype(mx.float16)
    wk = dequantize_linear(mlx_model.model.layers[0].self_attn.k_proj).astype(mx.float16)
    wv = dequantize_linear(mlx_model.model.layers[0].self_attn.v_proj).astype(mx.float16)
    wo = dequantize_linear(mlx_model.model.layers[0].self_attn.o_proj).astype(mx.float16)
    bq=mlx_model.model.layers[0].self_attn.q_proj.bias.astype(mx.float16)
    bk=mlx_model.model.layers[0].self_attn.k_proj.bias.astype(mx.float16)
    bv=mlx_model.model.layers[0].self_attn.v_proj.bias.astype(mx.float16)
    attention = qwen2_week1.Qwen2MultiHeadAttention(
        mlx_model.args.hidden_size,
        mlx_model.args.num_attention_heads,
        mlx_model.args.num_key_value_heads,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        bq=bq,
        bk=bk,
        bv=bv,
        max_seq_len=mlx_model.args.max_position_embeddings,
        theta=mlx_model.args.rope_theta
    )
    input = mx.random.uniform(shape=(1, 100, mlx_model.args.hidden_size), dtype=mx.float16)
    res_impl = attention(input, 0)
    res_ref = mlx_model.model.layers[0].self_attn(input)
    assert_allclose(res_impl, res_ref, precision=mx.float16)

    # raise NotImplementedError(
    #     "This test is not implemented yet. Please implement the test for Qwen2-0.5B-Instruct-MLX."
    # )
