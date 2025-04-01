"""Zero Operator."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import Self

from mrmax.operators.LinearOperator import LinearOperator


class ZeroOp(LinearOperator, eqx.Module):
    """A constant zero operator.

    This operator always returns zero when applied to a tensor.
    It is the neutral element of the addition of operators.
    """

    keep_shape: bool

    def __init__(self, keep_shape: bool = False) -> None:
        """Initialize the Zero Operator.

        Returns a constant zero, either as a scalar or as a tensor of the same shape as the input,
        depending on the value of keep_shape.
        Returning a scalar can save memory and computation time in some cases.

        Parameters
        ----------
        keep_shape
            If True, the shape of the input is kept.
            If False, the output is, regardless of the input shape, an integer scalar 0,
            which can broadcast to the input shape and dtype.
        """
        super().__init__()
        self.keep_shape = keep_shape

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the operator to the input.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        zeros_like(x) or scalar 0
        """
        if self.keep_shape:
            return (jnp.zeros_like(x),)
        else:
            return (jnp.array(0),)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint of the operator to the input.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        zeros_like(x)
        """
        if self.keep_shape:
            return (jnp.zeros_like(x),)
        else:
            return (jnp.array(0),)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Zero operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of ZeroOp."""
        return f'{type(self).__name__}(keep_shape={self.keep_shape})'
