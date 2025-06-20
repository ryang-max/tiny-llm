import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.sine_freqs = mx.outer(
            mx.arange(seq_len),
            mx.power(base, -(mx.arange(0, dims, 2, dtype=mx.float32) / dims)),
        ).sin()
        self.cosine_freqs = mx.outer(
            mx.arange(seq_len),
            mx.power(base, -(mx.arange(0, dims, 2, dtype=mx.float32) / dims)),
        ).cos()

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        cosine_basis = mx.reshape(
            self.cosine_freqs[:L, :]
            if offset is None
            else self.cosine_freqs[offset, :],
            (-1, L, 1, D // 2),
        )
        sine_basis = mx.reshape(
            self.sine_freqs[:L, :] if offset is None else self.sine_freqs[offset, :],
            (-1, L, 1, D // 2),
        )
        if self.traditional:
            x = x.reshape(N, L, H, D // 2, 2)
            x_0 = x[..., 0]
            x_1 = x[..., 1]
            output_0 = mx.multiply(x_0, cosine_basis) - mx.multiply(x_1, sine_basis)
            output_1 = mx.multiply(x_1, cosine_basis) + mx.multiply(x_0, sine_basis)
        else:
            x_0 = x[..., : D // 2]
            x_1 = x[..., D // 2 :]
            output_0 = mx.multiply(x_0, cosine_basis) - mx.multiply(x_1, sine_basis)
            output_1 = mx.multiply(x_1, cosine_basis) + mx.multiply(x_0, sine_basis)
        #print(f"output_0 shape: {output_0.shape}, output_1 shape: {output_1.shape}")
        if self.traditional:
            output = mx.stack((output_0, output_1), axis=-1)
        else:
            output = mx.concat((output_0, output_1), axis=-1)
        output = output.reshape(N, L, H, D)
        return output
