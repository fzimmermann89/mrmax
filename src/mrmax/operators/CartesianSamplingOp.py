"""Cartesian Sampling Operator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import Self

from mrmax.data.KTrajectory import KTrajectory
from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.operators.LinearOperator import LinearOperator


class CartesianSamplingOp(LinearOperator, eqx.Module):
    """Cartesian Sampling Operator class.

    This operator samples a Cartesian grid at non-uniform positions. It is used for Cartesian data on a regular grid.
    """

    _encoding_matrix: SpatialDimension | Sequence[int]
    _traj: KTrajectory
    _fft_idx: Array | None
    _needs_indexing: bool
    _matrix_size: tuple[int, int, int]

    def __init__(
        self,
        encoding_matrix: SpatialDimension | Sequence[int],
        traj: KTrajectory,
    ) -> None:
        """Initialize Cartesian Sampling Operator.

        Parameters
        ----------
        encoding_matrix
            Dimension of the encoded k-space. If this is `~mrmax.data.SpatialDimension` only values of directions
            will be used. Otherwise, it should be a `Sequence` of the same length as direction.
        traj
            The k-space trajectories where the frequencies are sampled.
        """
        super().__init__()
        self._encoding_matrix = encoding_matrix
        self._traj = traj

        # Get the matrix size
        if isinstance(encoding_matrix, SpatialDimension):
            self._matrix_size = (
                int(encoding_matrix.z),
                int(encoding_matrix.y),
                int(encoding_matrix.x),
            )
        else:
            matrix_size = tuple(encoding_matrix)
            if len(matrix_size) != 3:
                raise ValueError('encoding_matrix must have exactly 3 dimensions')
            self._matrix_size = cast(tuple[int, int, int], matrix_size)

        # Get the indices for sampling
        self._fft_idx = self._get_fft_indices()
        self._needs_indexing = self._fft_idx is not None

    def _get_fft_indices(self) -> Array | None:
        """Get the indices for sampling.

        Returns
        -------
            indices for sampling, or None if no indexing is needed
        """
        if not self._traj.type_along_k210[0] == 'uniform':
            return None

        # Get the indices for sampling
        kz = self._traj.kz
        ky = self._traj.ky
        kx = self._traj.kx

        # Get the indices for sampling
        kz_idx = jnp.round(kz).astype(jnp.int32)
        ky_idx = jnp.round(ky).astype(jnp.int32)
        kx_idx = jnp.round(kx).astype(jnp.int32)

        # Check if indices are within bounds
        kz_in_bounds = jnp.all((kz_idx >= 0) & (kz_idx < self._matrix_size[0]))
        ky_in_bounds = jnp.all((ky_idx >= 0) & (ky_idx < self._matrix_size[1]))
        kx_in_bounds = jnp.all((kx_idx >= 0) & (kx_idx < self._matrix_size[2]))

        if not (kz_in_bounds and ky_in_bounds and kx_in_bounds):
            return None

        return jnp.stack([kz_idx, ky_idx, kx_idx], axis=-1)

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, z, y, x)`

        Returns
        -------
            output tensor, shape `(..., coils, k2, k1, k0)`
        """
        if self._needs_indexing and self._fft_idx is not None:
            # Use the indices to sample the input tensor
            x = x[..., self._fft_idx[..., 0], self._fft_idx[..., 1], self._fft_idx[..., 2]]
        return (x,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, k2, k1, k0)`

        Returns
        -------
            output tensor, shape `(..., coils, z, y, x)`
        """
        if self._needs_indexing and self._fft_idx is not None:
            # Create a zero tensor with the correct shape
            output = jnp.zeros(
                (*x.shape[:-3], *self._matrix_size),
                dtype=x.dtype,
            )
            # Use the indices to scatter the input tensor
            output = output.at[..., self._fft_idx[..., 0], self._fft_idx[..., 1], self._fft_idx[..., 2]].set(x)
            return (output,)
        return (x,)

    @property
    def gram(self) -> LinearOperator:
        """Return the gram operator."""
        return CartesianSamplingGramOp(self)

    def __repr__(self) -> str:
        """Return string representation of CartesianSamplingOp."""
        return f'{type(self).__name__}(encoding_matrix={self._encoding_matrix}, traj={self._traj})'


class CartesianSamplingGramOp(LinearOperator, eqx.Module):
    """Gram operator for `CartesianSamplingOp`.

    Implements the adjoint of the forward operator of the Cartesian sampling operator, i.e. the gram operator
    `S.H@S`.

    Uses a multiplication with a binary mask in Fourier space to calculate the gram operator.

    This should not be used directly, but rather through the `~CartesianSamplingOp.gram` method of a
    `CartesianSamplingOp` object.
    """

    _mask: Array | None

    def __init__(self, sampling_op: CartesianSamplingOp) -> None:
        """Initialize the gram operator.

        Parameters
        ----------
        sampling_op
            The py:class:`CartesianSamplingOp` to calculate the gram operator for.
        """
        super().__init__()
        self._mask = None

        if sampling_op._needs_indexing and sampling_op._fft_idx is not None:
            # Create a mask for the gram operator
            self._mask = jnp.zeros(
                sampling_op._matrix_size,
                dtype=jnp.bool_,
            )
            self._mask = self._mask.at[
                sampling_op._fft_idx[..., 0], sampling_op._fft_idx[..., 1], sampling_op._fft_idx[..., 2]
            ].set(True)

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, z, y, x)`
        """
        if self._mask is not None:
            x = x * self._mask
        return (x,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, k2, k1, k0)`
        """
        return self.forward(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the gram operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of CartesianSamplingGramOp."""
        return f'{type(self).__name__}'
