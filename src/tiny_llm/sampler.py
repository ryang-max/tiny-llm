import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if top_k is not None:
            mask_index = mx.argpartition(
                -logprobs, kth=top_k, axis=-1
            )[:, top_k:]
            logprobs[:, mask_index] = -mx.inf
        if top_p is not None:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(sorted_logprobs, axis=-1)
            logprobs[:, sorted_idx] = mx.where(cumsum < top_p, sorted_logprobs, -mx.inf)
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        else:
            return mx.random.categorical(
                logprobs / temp, axis=-1
            )

    return sample
