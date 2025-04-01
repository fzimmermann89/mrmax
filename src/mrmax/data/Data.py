"""Base class for data objects."""

from __future__ import annotations

import dataclasses
from abc import ABC
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing_extensions import Self

T = TypeVar('T', bound='Data')


@register_pytree_node_class
@dataclasses.dataclass(slots=True, frozen=True)
class Data(ABC):
    """A general data class with field data and header."""

    data: jnp.ndarray
    """Data. Shape `(...other coils k2 k1 k0)`"""

    header: Any
    """Header information for data."""

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], dict[str, Any]]:
        """Flatten the tree structure for JAX.

        Returns
        -------
        tuple[tuple[jnp.ndarray, ...], dict[str, Any]]
            A tuple containing the children and auxiliary data.
        """
        children = (self.data,)
        aux_data = {'header': self.header}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[jnp.ndarray, ...]) -> Self:
        """Unflatten the tree structure for JAX.

        Parameters
        ----------
        aux_data : dict[str, Any]
            The auxiliary data containing non-array attributes.
        children : tuple[jnp.ndarray, ...]
            The array attributes of the class.

        Returns
        -------
        Self
            A new instance of the class.
        """
        (data,) = children
        return cls(data=data, header=aux_data['header'])

    def to(self, device: str | jax.Device | None = None, dtype: jnp.dtype | None = None) -> Self:
        """Move data to device and convert dtype if necessary.

        Parameters
        ----------
        device : str | jax.Device | None
            The destination device.
        dtype : jnp.dtype | None
            The destination dtype.

        Returns
        -------
        Self
            A new instance with moved data.
        """
        data = jax.device_put(self.data, device) if device is not None else self.data
        data = data.astype(dtype) if dtype is not None else data
        return self.__class__(data=data, header=self.header)

    def __repr__(self) -> str:
        """Get string representation of Data.

        Returns
        -------
        str
            String representation of Data.
        """
        try:
            device = str(self.data.device())
        except RuntimeError:
            device = 'mixed'
        out = (
            f'{type(self).__name__} with shape: {list(self.data.shape)!s} and dtype {self.data.dtype}\nDevice: {device}'
        )
        return out
