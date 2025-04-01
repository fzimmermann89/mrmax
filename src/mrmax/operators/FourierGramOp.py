"""Fourier Gram Operator."""

from __future__ import annotations

import jax.numpy as jnp
from typing_extensions import Self

from mrmax.operators.FourierOp import FourierOp
from mrmax.operators.LinearOperator import LinearOperator


class FourierGramOp(LinearOperator):
    """Gram operator for the Fourier Operator.

    The Gram operator is the composition FourierOp.H @ FourierOp.
    """

    def __init__(self, fourier_op: FourierOp) -> None:
        """Initialize Fourier Gram Operator.

        This should not be used directly, but rather through the `gram` method of a
        `FourierOp` object.

        Parameters
        ----------
        fourier_op
            The Fourier Operator for which to create the Gram operator.
        """
        super().__init__()
        self._fourier_op = fourier_op

        # Create a mask for the gram operator
        ones = jnp.ones((*fourier_op._traj.broadcast_shape[:-3], *fourier_op._recon_matrix))
        (mask,) = fourier_op.adjoint(*fourier_op.forward(ones))
        self._mask = mask

    def forward(self, x: jnp.ndarray) -> tuple[jnp.ndarray]:
        """Apply the Gram operator.

        Parameters
        ----------
        x
            Input data, shape `(..., coils, z, y, x)`

        Returns
        -------
            Output data, shape `(..., coils, z, y, x)`
        """
        return (x * self._mask,)

    def adjoint(self, x: jnp.ndarray) -> tuple[jnp.ndarray]:
        """Apply the adjoint of the Gram operator.

        Parameters
        ----------
        x
            Input data, shape `(..., coils, z, y, x)`

        Returns
        -------
            Output data, shape `(..., coils, z, y, x)`
        """
        return self.forward(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Gram operator."""
        return self

    def __repr__(self) -> str:
        """Representation method for Fourier Gram operator."""
        return f'{type(self).__name__}'
