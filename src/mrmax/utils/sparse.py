"""Sparse matrix utilities."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class SparseMatrix:
    """Sparse matrix in COO format."""

    def __init__(
        self,
        indices: Int[Array, '2 *'],
        values: Float[Array, '*'],
        shape: tuple[int, ...],
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        """Initialize sparse matrix.

        Parameters
        ----------
        indices
            2D array of indices, shape (2, nnz)
        values
            1D array of values, shape (nnz,)
        shape
            Shape of the matrix
        dtype
            Data type of the matrix
        """
        self.indices = indices
        self.values = values
        self.shape = tuple(map(int, shape))
        self.dtype = dtype

    def coalesce(self) -> SparseMatrix:
        """Combine duplicate indices by summing their values."""
        # Sort indices lexicographically
        sorted_indices = jnp.lexsort(self.indices[::-1])
        indices = self.indices[:, sorted_indices]
        values = self.values[sorted_indices]

        # Find unique indices and sum their values
        unique_indices, inverse_indices = jnp.unique(indices, axis=1, return_inverse=True)
        unique_values = jnp.zeros(unique_indices.shape[1], dtype=self.dtype)
        unique_values = unique_values.at[inverse_indices].add(values)

        return SparseMatrix(unique_indices, unique_values, self.shape, self.dtype)

    def sum(self, axis: int) -> Float[Array, ...]:
        """Sum along an axis."""
        if axis == 0:
            return jnp.bincount(self.indices[1], weights=self.values, length=self.shape[1])
        elif axis == 1:
            return jnp.bincount(self.indices[0], weights=self.values, length=self.shape[0])
        else:
            raise ValueError('axis must be 0 or 1')

    def to_dense(self) -> Float[Array, ...]:
        """Convert to dense matrix."""
        dense = jnp.zeros(self.shape, dtype=self.dtype)
        return dense.at[tuple(self.indices)].add(self.values)

    def __mul__(self, other: Float[Array, ...]) -> SparseMatrix:
        """Element-wise multiplication with a dense array."""
        if isinstance(other, Array):
            if other.ndim == 1:
                # Broadcast along rows
                new_values = self.values * other[self.indices[0]]
            elif other.ndim == 2:
                # Element-wise multiplication
                new_values = self.values * other[tuple(self.indices)]
            else:
                raise ValueError('other must be 1D or 2D')
            return SparseMatrix(self.indices, new_values, self.shape, self.dtype)
        else:
            return NotImplemented

    def __rmul__(self, other: Float[Array, ...]) -> SparseMatrix:
        """Element-wise multiplication with a dense array."""
        return self.__mul__(other)

    def __matmul__(self, other: Float[Array, ...]) -> Float[Array, ...]:
        """Matrix multiplication with a dense array."""
        if isinstance(other, Array):
            if other.ndim == 1:
                # Matrix-vector multiplication
                if other.shape[0] != self.shape[1]:
                    raise ValueError('Shape mismatch in matrix-vector multiplication')
                result = jnp.zeros(self.shape[0], dtype=self.dtype)
                return result.at[self.indices[0]].add(self.values * other[self.indices[1]])
            elif other.ndim == 2:
                # Matrix-matrix multiplication
                if other.shape[0] != self.shape[1]:
                    raise ValueError('Shape mismatch in matrix-matrix multiplication')
                result = jnp.zeros((self.shape[0], other.shape[1]), dtype=self.dtype)
                return result.at[self.indices[0], :].add(self.values[:, None] * other[self.indices[1], :])
            else:
                raise ValueError('other must be 1D or 2D')
        else:
            return NotImplemented

    def __rmatmul__(self, other: Float[Array, ...]) -> Float[Array, ...]:
        """Matrix multiplication with a dense array."""
        if isinstance(other, Array):
            if other.ndim == 1:
                # Vector-matrix multiplication
                if other.shape[0] != self.shape[0]:
                    raise ValueError('Shape mismatch in vector-matrix multiplication')
                result = jnp.zeros(self.shape[1], dtype=self.dtype)
                return result.at[self.indices[1]].add(self.values * other[self.indices[0]])
            elif other.ndim == 2:
                # Matrix-matrix multiplication
                if other.shape[1] != self.shape[0]:
                    raise ValueError('Shape mismatch in matrix-matrix multiplication')
                result = jnp.zeros((other.shape[0], self.shape[1]), dtype=self.dtype)
                return result.at[:, self.indices[1]].add(other[:, self.indices[0]] * self.values)
            else:
                raise ValueError('other must be 1D or 2D')
        else:
            return NotImplemented

    @property
    def H(self) -> SparseMatrix:  # noqa: N802
        """Return the conjugate transpose."""
        return SparseMatrix(
            jnp.stack([self.indices[1], self.indices[0]]),
            jnp.conj(self.values),
            (self.shape[1], self.shape[0]),
            self.dtype,
        )


def sparse_coo_matrix(
    indices: Int[Array, '2 *'],
    values: Float[Array, '*'],
    shape: tuple[int, ...],
    dtype: jnp.dtype = jnp.float32,
) -> SparseMatrix:
    """Create a sparse COO matrix.

    Parameters
    ----------
    indices
        2D array of indices, shape (2, nnz)
    values
        1D array of values, shape (nnz,)
    shape
        Shape of the matrix
    dtype
        Data type of the matrix

    Returns
    -------
    SparseMatrix
        Sparse COO matrix
    """
    return SparseMatrix(indices, values, tuple(map(int, shape)), dtype)


def ravel_multi_index(indices: tuple[Int[Array, ...], ...], shape: tuple[int, ...]) -> Int[Array, ...]:
    """Convert a tuple of coordinate arrays to a flattened index array.

    Parameters
    ----------
    indices
        Tuple of coordinate arrays
    shape
        Shape of the array

    Returns
    -------
    Array
        Flattened index array
    """
    return jnp.ravel_multi_index(indices, tuple(map(int, shape)))
