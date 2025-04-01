"""ProximableFunctionalSeparableSum class."""

from __future__ import annotations

from typing import TypeAlias, TypeVar, TypeVarTuple, Unpack

import jax.numpy as jnp

from mrmax.operators.Functional import ProximableFunctional

# Type variables for input and output types
Tin: TypeAlias = TypeVarTuple('Tin')
Tout: TypeAlias = TypeVar('Tout')


class ProximableFunctionalSeparableSum(ProximableFunctional[Unpack[Tin]]):
    """ProximableFunctionalSeparableSum class.

    A proximable functional that is the sum of two proximable functionals.
    The proximal operator of the sum is computed using the proximal operator of each functional.
    """

    def __init__(
        self, functional1: ProximableFunctional[Unpack[Tin]], functional2: ProximableFunctional[Unpack[Tin]]
    ) -> None:
        """Initialize the separable sum functional.

        Parameters
        ----------
        functional1
            First functional
        functional2
            Second functional
        """
        super().__init__()
        self.functional1 = functional1
        self.functional2 = functional2

    def forward(self, *args: Unpack[Tin]) -> tuple[jnp.ndarray]:
        """Forward method.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Sum of the two functionals evaluated at the input arguments
        """
        return (self.functional1(*args)[0] + self.functional2(*args)[0],)

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
        return (self.functional1.prox(x, sigma)[0],)

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
        return (self.functional1.prox_convex_conj(x, sigma)[0],)
