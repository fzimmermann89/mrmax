"""Class for Fast Fourier Operator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import Self

from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.operators.LinearOperator import LinearOperator
from mrmax.operators.ZeroPadOp import ZeroPadOp


class FastFourierOp(LinearOperator, eqx.Module):
    """Fast Fourier operator class.

    Applies a Fast Fourier Transformation along selected dimensions with cropping/zero-padding
    along these selected dimensions

    The transformation is done with 'ortho' normalization, i.e. the normalization constant is split between
    forward and adjoint [FFT]_.

    Remark regarding the fftshift/ifftshift:

    fftshift shifts the zero-frequency point to the center of the data, ifftshift undoes this operation.
    The input to both `~FastFourierOp.forward` and `~FastFourierOp.adjoint`
    are assumed to have the zero-frequency in the center of the data. `jnp.fft.fftn`
    and `jnp.fft.ifftn` expect the zero-frequency to be the first entry in the tensor.
    Therefore in `~FastFourierOp.forward` and `~FastFourierOp.adjoint`,
    first `jnp.fft.ifftshift`, then `jnp.fft.fftn` or `jnp.fft.ifftn`,
    finally `jnp.fft.ifftshift` are applied.

    .. note::
       See also `~mrmax.operators.FourierOp` for a Fourier operator that handles
       automatic sorting of the k-space data based on a trajectory.


    References
    ----------
    .. [FFT] https://numpy.org/doc/stable/reference/routines.fft.html
    """

    _dim: tuple[int, ...]
    _pad_op: ZeroPadOp

    def __init__(
        self,
        dim: Sequence[int],
        recon_matrix: SpatialDimension | Sequence[int],
        encoding_matrix: SpatialDimension | Sequence[int],
    ) -> None:
        """Initialize Fast Fourier Operator.

        Parameters
        ----------
        dim
            dimensions along which FFT is applied
        recon_matrix
            dimension of the reconstructed image
        encoding_matrix
            dimension of the encoded k-space
        """
        super().__init__()
        self._dim = tuple(dim)
        if isinstance(recon_matrix, SpatialDimension):
            recon_matrix = astuple(recon_matrix)
        if isinstance(encoding_matrix, SpatialDimension):
            encoding_matrix = astuple(encoding_matrix)
        self._pad_op = ZeroPadOp(dim=dim, recon_matrix=recon_matrix, encoding_matrix=encoding_matrix)

    def forward(self, x: Array) -> tuple[Array]:
        """FFT from image space to k-space.

        Parameters
        ----------
        x
            image data on Cartesian grid

        Returns
        -------
            FFT of `x`
        """
        y = jnp.fft.fftshift(
            jnp.fft.fftn(jnp.fft.ifftshift(*self._pad_op(x), axes=self._dim), axes=self._dim, norm='ortho'),
            axes=self._dim,
        )
        return (y,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """IFFT from k-space to image space.

        Parameters
        ----------
        x
            k-space data on Cartesian grid

        Returns
        -------
            IFFT of `x`
        """
        # FFT
        return self._pad_op.adjoint(
            jnp.fft.fftshift(
                jnp.fft.ifftn(jnp.fft.ifftshift(x, axes=self._dim), axes=self._dim, norm='ortho'),
                axes=self._dim,
            ),
        )

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Fast Fourier operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of FastFourierOp."""
        return f'{type(self).__name__}(dim={self._dim})'
