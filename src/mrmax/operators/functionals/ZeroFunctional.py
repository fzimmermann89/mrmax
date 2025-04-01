"""Zero functional class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeVar, TypeVarTuple, Unpack

import jax.numpy as jnp

from mrmax.operators.Functional import ElementaryProximableFunctional

# Type variables for input and output types
Tin: TypeAlias = TypeVarTuple('Tin')
Tout: TypeAlias = TypeVar('Tout')


class ZeroFunctional(ElementaryProximableFunctional[Unpack[Tin]]):
    """Zero functional class.

    A functional that always returns zero.
    """

    def __init__(
        self,
        weight: jnp.ndarray | float = 1.0,
        target: jnp.ndarray | float = 0.0,
        dim: Sequence[int] | None = None,
        keepdim: bool = False,
        divide_by_n: bool = False,
    ) -> None:
        """Initialize the zero functional.

        Parameters
        ----------
        weight
            Weight to apply to the input
        target
            Target value to subtract from the input
        dim
            Dimensions to reduce over
        keepdim
            Whether to keep the reduced dimensions
        divide_by_n
            Whether to divide by the number of elements
        """
        super().__init__(weight=weight, target=target, dim=dim, keepdim=keepdim, divide_by_n=divide_by_n)

    def forward(self, *args: Unpack[Tin]) -> tuple[jnp.ndarray]:
        """Forward method.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Zero array with the same shape as the input
        """
        x = args[0]
        return (jnp.zeros_like(x),)

    def prox(self, x: jnp.ndarray, sigma: jnp.ndarray | float = 1.0) -> tuple[jnp.ndarray]:
        """Proximal Mapping.

        Parameters
        ----------
        x
            Input array
        sigma
            Scaling factor

        Returns
        -------
            Proximal mapping applied to the input array
        """
        if not isinstance(sigma, jnp.ndarray):
            sigma = jnp.array(1.0 * sigma)
        self._throw_if_negative_or_complex(sigma)
        sigma = jnp.clip(sigma, a_min=1e-8)
        return (x,)

    def prox_convex_conj(self, x: jnp.ndarray, sigma: jnp.ndarray | float = 1.0) -> tuple[jnp.ndarray]:
        """Proximal Mapping of the convex conjugate.

        Parameters
        ----------
        x
            Input array
        sigma
            Scaling factor

        Returns
        -------
            Proximal mapping of the convex conjugate applied to the input array
        """
        if not isinstance(sigma, jnp.ndarray):
            sigma = jnp.array(1.0 * sigma)
        self._throw_if_negative_or_complex(sigma)
        sigma = jnp.clip(sigma, a_min=1e-8)
        return (jnp.zeros_like(x),)
