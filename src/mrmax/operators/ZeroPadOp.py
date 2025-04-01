"""Class for Zero Pad Operator."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
from jaxtyping import Array
from typing_extensions import Self

from mrmax.operators.LinearOperator import LinearOperator
from mrmax.utils import zero_pad_or_crop


class ZeroPadOp(LinearOperator, eqx.Module):
    """Zero Pad operator class."""

    dim: tuple[int, ...]
    original_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]

    def __init__(self, dim: Sequence[int], original_shape: Sequence[int], padded_shape: Sequence[int]) -> None:
        """Zero Pad Operator class.

        The operator carries out zero-padding if the `padded_shape` is larger than `orig_shape` and cropping if the
        `padded_shape` is smaller.

        Parameters
        ----------
        dim
            dimensions along which padding should be applied
        original_shape
            shape of original data along dim, same length as `dim`
        padded_shape
            shape of padded data along dim, same length as `dim`
        """
        if len(dim) != len(original_shape) or len(dim) != len(padded_shape):
            raise ValueError('Dim, orig_shape and padded_shape have to be of same length')

        super().__init__()
        self.dim = tuple(dim)
        self.original_shape = tuple(original_shape)
        self.padded_shape = tuple(padded_shape)

    def forward(self, x: Array) -> tuple[Array]:
        """Pad or crop data.

        Parameters
        ----------
        x
            data with shape orig_shape

        Returns
        -------
            data with shape padded_shape
        """
        return (zero_pad_or_crop(x, self.padded_shape, self.dim),)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Crop or pad data.

        Parameters
        ----------
        x
            data with shape padded_shape

        Returns
        -------
            data with shape orig_shape
        """
        return (zero_pad_or_crop(x, self.original_shape, self.dim),)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Zero Pad operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of ZeroPadOp."""
        return f'{type(self).__name__}(dim={self.dim}, original_shape={self.original_shape}, padded_shape={self.padded_shape})'
