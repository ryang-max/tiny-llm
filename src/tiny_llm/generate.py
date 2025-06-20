import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def _step(
    y: mx.array,
    offset: int,
    model: Qwen2ModelWeek1 | Qwen2ModelWeek2,
):
    logits = model(y, offset)
    logits = logits[:, -1, :]
    return mx.argmax(logits, axis=-1)


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    input = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    #print(f"input shape: {input.shape}")
    token = _step(input, 0, model).reshape(1, -1)
    output = [token]
    while token.item() != tokenizer.eos_token_id:
        #print(f"token shape: {token.shape}")
        input = mx.concat((input, token), axis=1)
        token = _step(input, input.shape[1], model).reshape(1, -1)
        print(tokenizer.decode(token.item()), end="", flush=True)
    #print([token.item() for token in output])
    #print("".join(tokenizer.decode(token.item()) for token in output))



def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    pass


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    pass
