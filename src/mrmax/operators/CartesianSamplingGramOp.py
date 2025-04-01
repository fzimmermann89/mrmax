"""Cartesian Sampling Gram Operator."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import Self

from mrmax.operators.CartesianSamplingOp import CartesianSamplingOp
from mrmax.operators.LinearOperator import LinearOperator


class CartesianSamplingGramOp(LinearOperator, eqx.Module):
    """Gram operator for the Cartesian Sampling Operator.

    The Gram operator is the composition CartesianSamplingOp.H @ CartesianSamplingOp.
    """

    _sampling_op: CartesianSamplingOp
    _mask: Array | None

    def __init__(self, sampling_op: CartesianSamplingOp) -> None:
        """Initialize Cartesian Sampling Gram Operator.

        This should not be used directly, but rather through the `gram` method of a
        `CartesianSamplingOp` object.

        Parameters
        ----------
        sampling_op
            The Cartesian Sampling Operator for which to create the Gram operator.
        """
        super().__init__()
        self._sampling_op = sampling_op

        # Create a mask for the gram operator
        if sampling_op._needs_indexing and sampling_op._fft_idx is not None:
            # Create a mask for the gram operator
            self._mask = jnp.zeros(
                sampling_op._matrix_size,
                dtype=jnp.bool_,
            )
            self._mask = self._mask.at[
                sampling_op._fft_idx[..., 0], sampling_op._fft_idx[..., 1], sampling_op._fft_idx[..., 2]
            ].set(True)
        else:
            self._mask = None

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the Gram operator.

        Parameters
        ----------
        x
            Input data, shape `(..., coils, k2, k1, k0)`

        Returns
        -------
            Output data, shape `(..., coils, k2, k1, k0)`
        """
        if self._mask is not None:
            x = x * self._mask
        return (x,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint of the Gram operator.

        Parameters
        ----------
        x
            Input data, shape `(..., coils, k2, k1, k0)`

        Returns
        -------
            Output data, shape `(..., coils, k2, k1, k0)`
        """
        return self.forward(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Gram operator."""
        return self

    def __repr__(self) -> str:
        """Representation method for Cartesian Sampling Gram operator."""
        return f'{type(self).__name__}'
