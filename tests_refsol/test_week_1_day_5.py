import pytest
from .utils import *
from .tiny_llm_base import Qwen2ModelWeek1, Embedding, dequantize_linear, qwen2_week1
from mlx_lm import load

# TODO: task 1 tests


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_utils_qwen_2_05b():
    pass


@pytest.mark.skipif(
    not qwen_2_7b_model_exists(), reason="Qwen2-7B-Instruct-MLX model not found"
)
def test_utils_qwen_2_7b():
    pass


@pytest.mark.skipif(
    not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct-MLX model not found"
)
def test_utils_qwen_2_15b():
    pass


def helper_test_task_3(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    model = Qwen2ModelWeek1(mlx_model)
    for _ in range(iters):
        input = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=(1, 10))
        user_output = model(input, 0)
        user_output = user_output - mx.logsumexp(user_output, keepdims=True)
        ref_output = mlx_model(input)
        ref_output = ref_output - mx.logsumexp(ref_output, keepdims=True)
        assert_allclose(user_output, ref_output, precision=mx.float16, rtol=1e-1)

def helper_test_task1(model_name: str, iters: int = 10):
    mlx_model, tokenizer = load(model_name)
    transformer = qwen2_week1.Qwen2TransformerBlock(
        num_attention_heads=mlx_model.args.num_attention_heads,
        num_kv_heads=mlx_model.args.num_key_value_heads,
        hidden_size=mlx_model.args.hidden_size,
        intermediate_size=mlx_model.args.intermediate_size,
        rms_norm_eps=mlx_model.args.rms_norm_eps,
        wq=dequantize_linear(mlx_model.model.layers[0].self_attn.q_proj).astype(mx.float16),
        wk=dequantize_linear(mlx_model.model.layers[0].self_attn.k_proj).astype(mx.float16),
        wv=dequantize_linear(mlx_model.model.layers[0].self_attn.v_proj).astype(mx.float16),
        wo=dequantize_linear(mlx_model.model.layers[0].self_attn.o_proj).astype(mx.float16),
        bq=mlx_model.model.layers[0].self_attn.q_proj.bias.astype(mx.float16),
        bk=mlx_model.model.layers[0].self_attn.k_proj.bias.astype(mx.float16),
        bv=mlx_model.model.layers[0].self_attn.v_proj.bias.astype(mx.float16),
        w_gate=dequantize_linear(mlx_model.model.layers[0].mlp.gate_proj).astype(mx.float16),
        w_up=dequantize_linear(mlx_model.model.layers[0].mlp.up_proj).astype(mx.float16),
        w_down=dequantize_linear(mlx_model.model.layers[0].mlp.down_proj).astype(mx.float16),
        w_input_layernorm=mlx_model.model.layers[0].input_layernorm.weight.astype(mx.float16),
        w_post_attention_layernorm=mlx_model.model.layers[0].post_attention_layernorm.weight.astype(mx.float16),
    )
    for _ in range(iters):
        input = mx.random.randint(low=0, high=mlx_model.args.hidden_size, shape=(1, 100, mlx_model.args.hidden_size))
        user_output = transformer(input, 0)
        #user_output = user_output - mx.logsumexp(user_output, keepdims=True)
        ref_output = mlx_model.model.layers[0](input)
        #ref_output = ref_output - mx.logsumexp(ref_output, keepdims=True)
        assert_allclose(user_output, ref_output, precision=mx.float16, rtol=1e-1)

@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_1_qwen_2_05b():
    helper_test_task1("Qwen/Qwen2-0.5B-Instruct-MLX", 3)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_2_embedding_call():
    mlx_model, _ = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
    )
    for _ in range(50):
        input = mx.random.randint(low=0, high=mlx_model.args.vocab_size, shape=(1, 10))
        user_output = embedding(input)
        ref_output = mlx_model.model.embed_tokens(input)
        assert_allclose(user_output, ref_output, precision=mx.float16)

@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_2_embedding_as_linear():
    mlx_model, _ = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    embedding = Embedding(
        mlx_model.args.vocab_size,
        mlx_model.args.hidden_size,
        dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
    )
    for _ in range(50):
        input = mx.random.uniform(shape=(1, 10, mlx_model.args.hidden_size))
        user_output = embedding.as_linear(input)
        ref_output = mlx_model.model.embed_tokens.as_linear(input)
        assert_allclose(user_output, ref_output, precision=mx.float16, atol=1e-1)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_task_3_qwen_2_05b():
    helper_test_task_3("Qwen/Qwen2-0.5B-Instruct-MLX", 5)


@pytest.mark.skipif(
    not qwen_2_7b_model_exists(), reason="Qwen2-7B-Instruct-MLX model not found"
)
def test_task_3_qwen_2_7b():
    helper_test_task_3("Qwen/Qwen2-7B-Instruct-MLX", 1)


@pytest.mark.skipif(
    not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct-MLX model not found"
)
def test_task_3_qwen_2_15b():
    helper_test_task_3("Qwen/Qwen2-1.5B-Instruct-MLX", 3)
