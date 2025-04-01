"""Matrix Multiplication Operator."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import Self

from mrmax.operators.LinearOperator import LinearOperator


class MatrixMultiplicationOp(LinearOperator, eqx.Module):
    """Matrix Multiplication Operator.

    This operator performs matrix multiplication between a fixed matrix and input tensors.
    It supports both forward and adjoint operations, and can handle sparse matrices efficiently.
    """

    matrix: Array | None
    matrix_adjoint: Array | None
    _range_shape: tuple[int, ...]
    _domain_shape: tuple[int, ...]

    def __init__(
        self,
        matrix: Array,
        optimize_for: str = 'both',
    ) -> None:
        """Initialize the Matrix Multiplication Operator.

        Parameters
        ----------
        matrix
            The matrix to use for multiplication operations.
        optimize_for
            Whether to optimize for 'forward', 'adjoint', or 'both' operations.
            Optimizing for both takes more memory but is faster for both operations.
        """
        super().__init__()
        if optimize_for not in ('forward', 'adjoint', 'both'):
            raise ValueError("optimize_for must be one of 'forward', 'adjoint', 'both'")

        # Store matrices based on optimization preference
        if optimize_for == 'forward':
            self.matrix = matrix
            self.matrix_adjoint = None
        elif optimize_for == 'adjoint':
            self.matrix_adjoint = matrix.H
            self.matrix = None
        else:  # 'both'
            self.matrix = matrix
            self.matrix_adjoint = matrix.H

        # Store shapes for validation
        self._range_shape = matrix.shape[:-1]  # Remove last dimension
        self._domain_shape = matrix.shape[1:]  # Remove first dimension

    def forward(self, x: Array) -> tuple[Array]:
        """Apply forward matrix multiplication.

        Parameters
        ----------
        x
            Input tensor to multiply with the matrix.

        Returns
        -------
        tuple[Array]
            Result of matrix multiplication.
        """
        match (self.matrix, self.matrix_adjoint):
            case (None, None):
                raise RuntimeError('Either matrix or matrix adjoint must be set')
            case (matrix, None) if matrix is not None:
                matrix_adjoint = matrix.H
            case (None, matrix_adjoint) if matrix_adjoint is not None:
                matrix = matrix_adjoint.H
            case (matrix, matrix_adjoint):
                ...

        # Reshape input if needed
        x_flat = jnp.reshape(x, (-1, *self._domain_shape))

        # Perform matrix multiplication
        y = jnp.matmul(matrix, x_flat)

        # Reshape output to match expected shape
        y = jnp.reshape(y, (*x.shape[: -len(self._domain_shape)], *self._range_shape))

        return (y,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply adjoint matrix multiplication.

        Parameters
        ----------
        x
            Input tensor to multiply with the adjoint matrix.

        Returns
        -------
        tuple[Array]
            Result of adjoint matrix multiplication.
        """
        match (self.matrix, self.matrix_adjoint):
            case (None, None):
                raise RuntimeError('Either matrix or matrix adjoint must be set')
            case (matrix, None) if matrix is not None:
                matrix_adjoint = matrix.H
            case (None, matrix_adjoint) if matrix_adjoint is not None:
                matrix = matrix_adjoint.H
            case (matrix, matrix_adjoint):
                ...

        # Reshape input if needed
        x_flat = jnp.reshape(x, (-1, *self._range_shape))

        # Perform adjoint matrix multiplication
        y = jnp.matmul(matrix_adjoint, x_flat)

        # Reshape output to match expected shape
        y = jnp.reshape(y, (*x.shape[: -len(self._range_shape)], *self._domain_shape))

        return (y,)

    @property
    def H(self) -> Self:  # noqa: N802
        """Return the adjoint operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of MatrixMultiplicationOp."""
        return f'{type(self).__name__}(range_shape={self._range_shape}, domain_shape={self._domain_shape})'
