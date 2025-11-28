# jax_models.py
# Optimized for TPU with bfloat16 support
import math
from typing import Sequence, Tuple, Optional, Any

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
except ImportError as e:
    raise ImportError(f"Failed to import JAX/Flax dependencies: {e}. Make sure JAX and Flax are installed.") from e

# Default dtype - will be overridden for TPU
Dtype = Any


# ---------- Basic Blocks ----------

class ConvBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.features, 
            self.kernel_size,
            strides=self.strides, 
            padding="SAME",
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(x)
        x = nn.gelu(x)
        return x


class ResBlock(nn.Module):
    features: int
    strides: Tuple[int, int] = (1, 1)
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(
            self.features, (3, 3), 
            strides=self.strides,
            padding="SAME", 
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(x)
        y = nn.gelu(y)
        y = nn.Conv(
            self.features, (3, 3), 
            strides=(1, 1),
            padding="SAME", 
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(y)

        if residual.shape[-1] != self.features or self.strides != (1, 1):
            residual = nn.Conv(
                self.features, (1, 1),
                strides=self.strides,
                padding="SAME", 
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.dtype,
            )(residual)

        return nn.gelu(residual + y)


class CNNEncoder(nn.Module):
    """
    Simple CNN encoder with configurable dtype for TPU optimization.
    """
    widths: Sequence[int] = (32, 64, 128, 256)
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Ensure input is correct dtype
        x = x.astype(self.dtype)
        
        for i, w in enumerate(self.widths):
            x = ConvBlock(w, (3, 3), strides=(2, 2), dtype=self.dtype)(x)
            x = ResBlock(w, dtype=self.dtype)(x)
        return x


# ---------- GeM + Global Branch ----------

class GeM(nn.Module):
    p_init: float = 3.0
    eps: float = 1e-6
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None):
        """
        x: [B,H,W,C], mask: [B,H,W,1] or None
        """
        p = self.param("p", lambda rng, shape: jnp.ones(shape, dtype=jnp.float32) * self.p_init, (1,))
        
        # Use float32 for GeM computation (numerical stability)
        x_f32 = x.astype(jnp.float32)
        p_f32 = p.astype(jnp.float32)
        
        x_clamp = jnp.clip(x_f32, a_min=self.eps)
        if mask is not None:
            mask_f32 = mask.astype(jnp.float32)
            x_pow = (x_clamp ** p_f32) * mask_f32
            sum_pool = jnp.sum(x_pow, axis=(1, 2))
            denom = jnp.sum(mask_f32, axis=(1, 2)) + self.eps
            pool = sum_pool / denom
        else:
            pool = jnp.mean(x_clamp ** p_f32, axis=(1, 2))
        
        result = pool ** (1.0 / p_f32)
        return result.astype(self.dtype)


class GlobalBranch(nn.Module):
    in_dim: int
    out_dim: int = 512
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None):
        """
        x: [B,H,W,C], mask: [B,H,W,1]
        """
        g = GeM(dtype=self.dtype)(x, mask)
        g = nn.Dense(self.out_dim, dtype=self.dtype, param_dtype=self.dtype)(g)
        g = nn.gelu(g)
        g = nn.Dense(self.out_dim, dtype=self.dtype, param_dtype=self.dtype)(g)
        
        # Normalize in float32 for stability
        g_f32 = g.astype(jnp.float32)
        g_norm = g_f32 / (jnp.linalg.norm(g_f32, axis=-1, keepdims=True) + 1e-6)
        return g_norm.astype(self.dtype)


# ---------- Transformer with cross-attn ----------

class MLPBlock(nn.Module):
    hidden_dim: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        y = nn.Dense(self.hidden_dim, dtype=self.dtype, param_dtype=self.dtype)(x)
        y = nn.gelu(y)
        y = nn.Dense(d, dtype=self.dtype, param_dtype=self.dtype)(y)
        return y


class SelfAttentionBlock(nn.Module):
    num_heads: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # x: [B,N,C]
        y = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            deterministic=True,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(y)
        x = x + y
        x = x + MLPBlock(hidden_dim=4 * x.shape[-1], dtype=self.dtype)(
            nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)
        )
        return x


class CrossAttentionBlock(nn.Module):
    num_heads: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x_q, x_kv, mask_k=None):
        # x_q: [B,Nq,C], x_kv: [B,Nk,C]
        y_q = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x_q)
        y_kv = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x_kv)

        mask = None
        if mask_k is not None:
            mask_k = mask_k.astype(jnp.float32)
            mask = mask_k[:, None, None, :] > 0.5

        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=True,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )(y_q, y_kv, mask=mask)

        x_q = x_q + y
        x_q = x_q + MLPBlock(hidden_dim=4 * x_q.shape[-1], dtype=self.dtype)(
            nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x_q)
        )
        return x_q

class PairTransformer(nn.Module):
    embed_dim: int
    num_heads: int = 8
    num_layers: int = 6
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, f0, f1, mask0=None, mask1=None):
        """
        f0, f1: [B,H,W,C]
        mask0, mask1: [B,H,W,1] in {0,1}
        Return: logits [B,1]
        """
        B, H, W, C = f0.shape
        x0 = jnp.reshape(f0, (B, H * W, C))
        x1 = jnp.reshape(f1, (B, H * W, C))

        mask0_flat = None
        mask1_flat = None
        if mask0 is not None:
            mask0_flat = jnp.reshape(mask0, (B, H * W))
        if mask1 is not None:
            mask1_flat = jnp.reshape(mask1, (B, H * W))

        for _ in range(self.num_layers):
            if mask0_flat is not None:
                x0 = x0 * mask0_flat[:, :, None]
            if mask1_flat is not None:
                x1 = x1 * mask1_flat[:, :, None]

            x0 = SelfAttentionBlock(self.num_heads, dtype=self.dtype)(x0)
            x1 = SelfAttentionBlock(self.num_heads, dtype=self.dtype)(x1)

            if mask0_flat is not None:
                x0 = x0 * mask0_flat[:, :, None]
            if mask1_flat is not None:
                x1 = x1 * mask1_flat[:, :, None]

            x0 = CrossAttentionBlock(self.num_heads, dtype=self.dtype)(x0, x1, mask_k=mask1_flat)
            x1 = CrossAttentionBlock(self.num_heads, dtype=self.dtype)(x1, x0, mask_k=mask0_flat)

        def masked_mean(x, m):
            m_exp = m[:, :, None]
            sum_val = jnp.sum(x * m_exp, axis=1)
            denom = jnp.sum(m_exp, axis=1) + 1e-6
            return sum_val / denom

        if mask0_flat is not None and mask1_flat is not None:
            x0_mean = masked_mean(x0, mask0_flat)
            x1_mean = masked_mean(x1, mask1_flat)
        else:
            x0_mean = jnp.mean(x0, axis=1)
            x1_mean = jnp.mean(x1, axis=1)

        h = jnp.concatenate(
            [x0_mean, x1_mean, jnp.abs(x0_mean - x1_mean)],
            axis=-1
        )
        h = nn.Dense(256, dtype=self.dtype, param_dtype=self.dtype)(h)
        h = nn.gelu(h)
        logits = nn.Dense(1, dtype=self.dtype, param_dtype=self.dtype)(h)
        return logits


# ---------- JIPNetFull (Flax) with TPU optimization ----------

class JIPNetFullFlax(nn.Module):
    input_size: int = 320
    img_channel: int = 1
    global_hidden_dim: int = 512
    transformer_layers: int = 6
    transformer_heads: int = 8
    dtype: Dtype = jnp.float32  # Use bfloat16 for TPU

    @nn.compact
    def __call__(self,
                 img1: jnp.ndarray,
                 img2: jnp.ndarray,
                 mask1: Optional[jnp.ndarray],
                 mask2: Optional[jnp.ndarray],
                 fusion_alpha: float):
        """
        img1, img2: [B,H,W,1] in [0,1]
        mask1, mask2: [B,H,W,1] in {0,1}
        fusion_alpha: scalar in [0,1]
        """
        # Cast inputs to model dtype
        img1 = img1.astype(self.dtype)
        img2 = img2.astype(self.dtype)
        
        encoder = CNNEncoder(widths=(32, 64, 128, 256), dtype=self.dtype)
        f1 = encoder(img1)
        f2 = encoder(img2)

        if mask1 is not None and mask2 is not None:
            Hf, Wf = f1.shape[1], f1.shape[2]
            mask1_res = jax.image.resize(mask1,
                                         (mask1.shape[0], Hf, Wf, 1),
                                         method="nearest")
            mask2_res = jax.image.resize(mask2,
                                         (mask2.shape[0], Hf, Wf, 1),
                                         method="nearest")
        else:
            mask1_res = mask2_res = None

        gbranch = GlobalBranch(in_dim=f1.shape[-1],
                               out_dim=self.global_hidden_dim,
                               dtype=self.dtype)
        g1 = gbranch(f1, mask1_res)
        g2 = gbranch(f2, mask2_res)
        
        # Compute cosine similarity in float32 for stability
        g1_f32 = g1.astype(jnp.float32)
        g2_f32 = g2.astype(jnp.float32)
        score_global = jnp.sum(g1_f32 * g2_f32, axis=-1, keepdims=True)
        score_global = (score_global + 1.0) / 2.0

        transformer = PairTransformer(embed_dim=f1.shape[-1],
                                      num_heads=self.transformer_heads,
                                      num_layers=self.transformer_layers,
                                      dtype=self.dtype)
        logits_ca = transformer(f1, f2, mask0=mask1_res, mask1=mask2_res)
        
        # Sigmoid in float32 for numerical stability
        score_ca = jax.nn.sigmoid(logits_ca.astype(jnp.float32))

        score_fused = fusion_alpha * score_global + \
                      (1.0 - fusion_alpha) * score_ca
        return score_fused.astype(jnp.float32)  # Always return float32 scores
