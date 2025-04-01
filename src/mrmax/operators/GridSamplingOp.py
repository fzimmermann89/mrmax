"""Grid Sampling Operator."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import Self

from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.operators.LinearOperator import LinearOperator


class GridSamplingOp(LinearOperator, eqx.Module):
    """Grid Sampling Operator.

    Given an "input" tensor and a "grid", computes the output by taking the input values at the locations
    determined by grid with interpolation. Thus, the output size will be determined by the grid size.
    For the adjoint to be defined, the grid and the shape of the "input" has to be known.
    """

    grid: Array
    input_shape: SpatialDimension
    interpolation_mode: Literal['bilinear', 'nearest', 'bicubic']
    padding_mode: Literal['zeros', 'border', 'reflection']
    align_corners: bool

    def __init__(
        self,
        grid: Array,
        input_shape: SpatialDimension,
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = False,
    ) -> None:
        r"""Initialize Sampling Operator.

        Parameters
        ----------
        grid
            sampling grid. Shape `*batchdim, z,y,x,3` / `*batchdim, y,x,2`.
            Values should be in ``[-1, 1.]``.
        input_shape
            Used in the adjoint. The z, y, x shape of the domain of the operator.
            If grid has 2 as the last dimension, only y and x will be used.
        interpolation_mode
            mode used for interpolation. bilinear is trilinear in 3D, bicubic is only supported in 2D.
        padding_mode
            how the input of the forward is padded.
        align_corners
            if True, the corner pixels of the input and output tensors are aligned,
            and thus preserve the values at those pixels
        """
        super().__init__()
        self.grid = grid
        self.input_shape = input_shape
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, z, y, x)`

        Returns
        -------
            output tensor, shape `(..., coils, *grid.shape[:-1])`
        """
        # Reshape grid to match input shape
        grid = self.grid.reshape(*self.grid.shape[:-1], -1, self.grid.shape[-1])
        grid = jnp.broadcast_to(grid, (*x.shape[:-3], *grid.shape[-2:]))

        # Apply grid sampling
        if self.interpolation_mode == 'bilinear':
            y = jnp.interp(
                grid,
                jnp.linspace(-1.0, 1.0, num=int(x.shape[-1])),
                x,
                left=0,
                right=0,
            )
        elif self.interpolation_mode == 'nearest':
            indices = jnp.round((grid + 1) * (x.shape[-1] - 1) / 2).astype(jnp.int32)
            indices = jnp.clip(indices, 0, x.shape[-1] - 1)
            y = x[..., indices]
        else:
            raise ValueError(f'Unsupported interpolation mode: {self.interpolation_mode}')

        return (y,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, *grid.shape[:-1])`

        Returns
        -------
            output tensor, shape `(..., coils, z, y, x)`
        """
        # Reshape grid to match input shape
        grid = self.grid.reshape(*self.grid.shape[:-1], -1, self.grid.shape[-1])
        grid = jnp.broadcast_to(grid, (*x.shape[:-3], *grid.shape[-2:]))

        # Apply adjoint grid sampling
        if self.interpolation_mode == 'bilinear':
            y = jnp.interp(
                grid,
                jnp.linspace(-1.0, 1.0, num=int(self.input_shape.x)),
                x,
                left=0,
                right=0,
            )
        elif self.interpolation_mode == 'nearest':
            indices = jnp.round((grid + 1) * (self.input_shape.x - 1) / 2).astype(jnp.int32)
            indices = jnp.clip(indices, 0, self.input_shape.x - 1)
            y = x[..., indices]
        else:
            raise ValueError(f'Unsupported interpolation mode: {self.interpolation_mode}')

        return (y,)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Grid Sampling operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of GridSamplingOp."""
        return f'{type(self).__name__}(grid_shape={self.grid.shape}, input_shape={self.input_shape})'
