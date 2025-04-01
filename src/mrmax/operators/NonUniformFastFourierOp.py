"""Non-Uniform Fast Fourier Operator."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jax_finufft import finufft_type1, finufft_type2
from jaxtyping import Array
from typing_extensions import Self

from mrmax.data.KTrajectory import KTrajectory
from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.operators.FastFourierOp import FastFourierOp
from mrmax.operators.LinearOperator import LinearOperator


class NonUniformFastFourierOp(LinearOperator, eqx.Module):
    """Non-Uniform Fast Fourier Operator class."""

    _direction: tuple[Literal['x', 'y', 'z', -3, -2, -1], ...]
    _recon_matrix: SpatialDimension | Sequence[int]
    _encoding_matrix: SpatialDimension | Sequence[int]
    _traj: KTrajectory
    _direction_zyx: tuple[int, ...]
    _dimension_210: tuple[int, ...]
    _im_size: tuple[int, ...]
    _omega: Array
    _traj_broadcast_shape: tuple[int, ...]
    scale: float
    oversampling: float

    def __init__(
        self,
        direction: Sequence[Literal['x', 'y', 'z', -3, -2, -1]],
        recon_matrix: SpatialDimension | Sequence[int],
        encoding_matrix: SpatialDimension | Sequence[int],
        traj: KTrajectory,
        oversampling: float = 2.0,
    ) -> None:
        """Initialize Non-Uniform Fast Fourier Operator.

        ```{note}
        Consider using `~mrmax.operators.FourierOp` instead of this operator. It automatically detects if a non-uniform
        or regular fast Fourier transformation is required and can also be constructed automatically from
        a `mrmax.data.KData` object.
        ````

        ```{note}
        The NUFFT is scaled such that it matches 'orthonormal' FFT scaling for cartesian trajectories.
        This is different from other packages, which apply scaling based on the size of the oversampled grid.
        ````

        Parameters
        ----------
        direction
            direction along which non-uniform FFT is applied
        recon_matrix
            Dimension of the reconstructed image. If this is `~mrmax.data.SpatialDimension` only values of directions
            will be used. Otherwise, it should be a `Sequence` of the same length as direction.
        encoding_matrix
            Dimension of the encoded k-space. If this is `~mrmax.data.SpatialDimension` only values of directions will
            be used. Otherwise, it should be a `Sequence` of the same length as direction.
        traj
            The k-space trajectories where the frequencies are sampled.
        oversampling
            Oversampling used for interpolation in non-uniform FFTs.
            On GPU, 2.0 uses an optimized kernel, any value > 1.0 will work.
            On CPU, there are kernels for 2.0 and 1.25. The latter saves memory. Set to 0.0 for automatic selection.
        """
        super().__init__()
        self._direction = tuple(direction)
        self._recon_matrix = recon_matrix
        self._encoding_matrix = encoding_matrix
        self._traj = traj
        self.oversampling = oversampling

        # Get the matrix size
        if isinstance(recon_matrix, SpatialDimension):
            self._im_size = (int(recon_matrix.z), int(recon_matrix.y), int(recon_matrix.x))
        else:
            self._im_size = tuple(recon_matrix)

        # Get the direction indices
        self._direction_zyx = tuple({'z': -3, 'y': -2, 'x': -1, -3: -3, -2: -2, -1: -1}[d] for d in direction)
        self._dimension_210 = tuple({'z': 0, 'y': 1, 'x': 2, -3: 0, -2: 1, -1: 2}[d] for d in direction)

        # Get the trajectory broadcast shape
        self._traj_broadcast_shape = self._traj.broadcasted_shape

        # Get the omega values for the NUFFT
        self._omega = jnp.stack(
            [
                self._traj.kz,
                self._traj.ky,
                self._traj.kx,
            ],
            axis=-1,
        )

        # Calculate the scale factor
        self.scale = 1.0 / jnp.sqrt(jnp.prod(jnp.array(self._im_size)))

    def _separate_joint_dimensions(self, ndim: int) -> tuple[list[int], list[int], list[int], list[int]]:
        """Separate dimensions into joint and separate dimensions.

        Parameters
        ----------
        ndim
            number of dimensions

        Returns
        -------
            tuple of lists of indices for:
            - joint dimensions
            - zyx dimensions
            - separate dimensions
            - 210 dimensions
        """
        # Get the indices for the zyx dimensions
        zyx_indices = list(range(ndim))
        for d in self._direction_zyx:
            zyx_indices.remove(d)

        # Get the indices for the 210 dimensions
        indices_210 = list(range(ndim))
        for d in self._dimension_210:
            indices_210.remove(d)

        return zyx_indices, self._direction_zyx, indices_210, self._dimension_210

    def forward(self, x: Array) -> tuple[Array]:
        """NUFFT from image space to k-space.

        Parameters
        ----------
        x
            coil image data with shape `(... coils z y x)`

        Returns
        -------
            coil k-space data with shape `(... coils k2 k1 k0)`
        """
        if len(self._direction_zyx):
            # We rearrange x into (sep_dims, joint_dims, nufft_directions)
            _, permute_zyx, sep_dims_210, permute_210 = self._separate_joint_dimensions(x.ndim)
            unpermute_zyx = jnp.array(permute_zyx).argsort().tolist()

            x = x.transpose(*permute_210)
            unflatten_other_shape = x.shape[: -len(self._dimension_210) - 1]  # -1 for coil
            # combine sep_dims
            x = x.reshape(-1, *x.shape[len(sep_dims_210) :]) if len(sep_dims_210) else x[None, :]
            # combine joint_dims and nufft_dims
            x = x.reshape(*unflatten_other_shape, -1, *x.shape[-len(self._direction_zyx) :])
            x = x.transpose(*unpermute_zyx)

            # Apply NUFFT Type 2 (image to k-space)
            x = vmap(
                partial(finufft_type2, upsampfac=self.oversampling, modeord=0, isign=-1, output_shape=self._im_size)
            )(self._omega, x)
            x = x * self.scale

            # Reshape back to original shape
            x = x.reshape(*unflatten_other_shape, -1, *x.shape[-len(self._direction_zyx) :])
            x = x.transpose(*unpermute_zyx)
        return (x,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """NUFFT from k-space to image space.

        Parameters
        ----------
        x
            coil k-space data with shape `(... coils k2 k1 k0)`

        Returns
        -------
            coil image data with shape `(... coils z y x)`
        """
        if len(self._direction_zyx):
            # We rearrange x into (sep_dims, joint_dims, nufft_directions)
            _, permute_zyx, sep_dims_210, permute_210 = self._separate_joint_dimensions(x.ndim)
            unpermute_zyx = jnp.array(permute_zyx).argsort().tolist()

            x = x.transpose(*permute_210)
            unflatten_other_shape = x.shape[: -len(self._dimension_210) - 1]  # -1 for coil
            # combine sep_dims
            x = x.reshape(-1, *x.shape[len(sep_dims_210) :]) if len(sep_dims_210) else x[None, :]
            # combine joint_dims and nufft_dims
            x = x.reshape(*unflatten_other_shape, -1, *x.shape[-len(self._direction_zyx) :])
            x = x.transpose(*unpermute_zyx)

            # Apply NUFFT Type 1 (k-space to image)
            x = vmap(
                partial(finufft_type1, upsampfac=self.oversampling, modeord=0, isign=1, output_shape=self._im_size)
            )(self._omega, x)
            x = x * self.scale

            # Reshape back to original shape
            x = x.reshape(*unflatten_other_shape, -1, *x.shape[-len(self._direction_zyx) :])
            x = x.transpose(*unpermute_zyx)
        return (x,)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the non-uniform Fast Fourier operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of NonUniformFastFourierOp."""
        return f'{type(self).__name__}(direction={self._direction})'


class NonUniformFastFourierOpGramOp(LinearOperator, eqx.Module):
    """Gram operator for `NonUniformFastFourierOp`.

    Implements the adjoint of the forward operator of the non-uniform Fast Fourier operator, i.e. the gram operator
    `NUFFT.H@NUFFT`.

    Uses a convolution, implemented as multiplication in Fourier space, to calculate the gram operator
    for the Toeplitz NUFFT operator.

    This should not be used directly, but rather through the `~NonUniformFastFourierOp.gram` method of a
    `NonUniformFastFourierOp` object.
    """

    _kernel: Array | None
    nufft_gram: LinearOperator | None

    def __init__(self, nufft_op: NonUniformFastFourierOp) -> None:
        """Initialize the gram operator.

        Parameters
        ----------
        nufft_op
            The py:class:`NonUniformFastFourierOp` to calculate the gram operator for.

        """
        super().__init__()
        self.nufft_gram = None

        if not nufft_op._dimension_210:
            return

        weight = jnp.ones(
            [*nufft_op._traj_broadcast_shape[:-4], 1, *nufft_op._traj_broadcast_shape[-3:]],
        )

        # We rearrange weight into (sep_dims, joint_dims, nufft_dims)
        _, permute_zyx, sep_dims_210, permute_210 = nufft_op._separate_joint_dimensions(weight.ndim)
        unpermute_zyx = jnp.array(permute_zyx).argsort().tolist()

        weight = weight.transpose(*permute_210)
        unflatten_other_shape = weight.shape[: -len(nufft_op._dimension_210) - 1]  # -1 for coil
        # combine sep_dims
        weight = weight.reshape(-1, *weight.shape[len(sep_dims_210) :]) if len(sep_dims_210) else weight[None, :]
        # combine joint_dims and nufft_dims
        weight = weight.reshape(*unflatten_other_shape, -1, *weight.shape[-len(nufft_op._direction_zyx) :])
        weight = weight.transpose(*unpermute_zyx)
        weight = weight * (nufft_op.scale) ** 2

        fft = FastFourierOp(
            dim=nufft_op._direction_zyx,
            encoding_matrix=[2 * s for s in nufft_op._im_size],
            recon_matrix=nufft_op._im_size,
        )
        self.nufft_gram = fft.H * weight @ fft

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, z, y, x)`
        """
        if self.nufft_gram is not None:
            (x,) = self.nufft_gram(x)

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
        """Return string representation of NonUniformFastFourierOpGramOp."""
        return f'{type(self).__name__}'
