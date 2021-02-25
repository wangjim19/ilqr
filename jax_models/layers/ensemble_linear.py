import haiku as hk
import jax
import jax.numpy as jnp
from typing import Any, Callable, Iterable, Optional, Type
import numpy as np

class EnsembleLinear(hk.Module):
  """Linear module."""

  def __init__(
      self,
      ensemble_size: int,
      output_size: int,
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    """Constructs the Linear module.
    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.ensemble_size = ensemble_size
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Computes a linear transform of the input."""
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    ensemble_size = self.ensemble_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [ensemble_size, input_size, output_size], dtype, init=w_init)

    reshaped = inputs if len(inputs.shape) == 3 else inputs[:, jnp.newaxis, :]
    out = jnp.einsum('ebi,eio->ebo', reshaped, w)

    if self.with_bias:
      b = hk.get_parameter("b", [self.ensemble_size, self.output_size], dtype, init=self.b_init)
      #b = jnp.tile(b, (reshaped.shape[1], 1, 1)).transpose((1, 0, 2))
      out = out + b[:, jnp.newaxis, :]

    if len(inputs.shape) == 2:
        out = out.squeeze(axis=1)
    return out
