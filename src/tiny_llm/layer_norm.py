import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight.astype(mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        summation = mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + self.eps)
        return x * summation * self.weight
