"""Spatial dimension class."""

import dataclasses
from collections.abc import Callable
from typing import Any, Union

import jax.numpy as jnp


@dataclasses.dataclass(slots=True, frozen=True)
class SpatialDimension:
    """Spatial dimension class."""

    z: float | int | jnp.ndarray
    """Z dimension."""

    y: float | int | jnp.ndarray
    """Y dimension."""

    x: float | int | jnp.ndarray
    """X dimension."""

    @classmethod
    def from_xyz(
        cls, xyz: tuple[float | int | jnp.ndarray, float | int | jnp.ndarray, float | int | jnp.ndarray]
    ) -> 'SpatialDimension':
        """Create SpatialDimension object from xyz tuple.

        Parameters
        ----------
        xyz
            Tuple of (x, y, z) values.

        Returns
        -------
        SpatialDimension
            Spatial dimension object.
        """
        return cls(z=xyz[2], y=xyz[1], x=xyz[0])

    def apply(self, func: Callable[[float | int | jnp.ndarray], Any]) -> 'SpatialDimension':
        """Apply a function to each dimension.

        Parameters
        ----------
        func
            Function to apply to each dimension.

        Returns
        -------
        SpatialDimension
            Spatial dimension object with function applied to each dimension.
        """
        return type(self)(z=func(self.z), y=func(self.y), x=func(self.x))

    def __add__(self, other: Union[float, int, jnp.ndarray, 'SpatialDimension']) -> 'SpatialDimension':
        """Add two spatial dimensions or a scalar.

        Parameters
        ----------
        other
            Spatial dimension or scalar to add.

        Returns
        -------
        SpatialDimension
            Sum of spatial dimensions.
        """
        if isinstance(other, SpatialDimension):
            return type(self)(z=self.z + other.z, y=self.y + other.y, x=self.x + other.x)
        else:
            return type(self)(z=self.z + other, y=self.y + other, x=self.x + other)

    def __sub__(self, other: Union[float, int, jnp.ndarray, 'SpatialDimension']) -> 'SpatialDimension':
        """Subtract two spatial dimensions or a scalar.

        Parameters
        ----------
        other
            Spatial dimension or scalar to subtract.

        Returns
        -------
        SpatialDimension
            Difference of spatial dimensions.
        """
        if isinstance(other, SpatialDimension):
            return type(self)(z=self.z - other.z, y=self.y - other.y, x=self.x - other.x)
        else:
            return type(self)(z=self.z - other, y=self.y - other, x=self.x - other)

    def __mul__(self, other: Union[float, int, jnp.ndarray, 'SpatialDimension']) -> 'SpatialDimension':
        """Multiply two spatial dimensions or a scalar.

        Parameters
        ----------
        other
            Spatial dimension or scalar to multiply.

        Returns
        -------
        SpatialDimension
            Product of spatial dimensions.
        """
        if isinstance(other, SpatialDimension):
            return type(self)(z=self.z * other.z, y=self.y * other.y, x=self.x * other.x)
        else:
            return type(self)(z=self.z * other, y=self.y * other, x=self.x * other)

    def __truediv__(self, other: Union[float, int, jnp.ndarray, 'SpatialDimension']) -> 'SpatialDimension':
        """Divide two spatial dimensions or a scalar.

        Parameters
        ----------
        other
            Spatial dimension or scalar to divide by.

        Returns
        -------
        SpatialDimension
            Quotient of spatial dimensions.
        """
        if isinstance(other, SpatialDimension):
            return type(self)(z=self.z / other.z, y=self.y / other.y, x=self.x / other.x)
        else:
            return type(self)(z=self.z / other, y=self.y / other, x=self.x / other)

    def __repr__(self):
        """Representation method for SpatialDimension class."""
        return f'{type(self).__name__}(z={self.z!s}, y={self.y!s}, x={self.x!s})'
