"""Base class for functionals."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, TypeVar, TypeVarTuple, Unpack

import jax.numpy as jnp

from mrmax.operators.Operator import Operator

# Type variables for input and output types
Tin: TypeAlias = TypeVarTuple('Tin')
Tout: TypeAlias = TypeVar('Tout')

# Type alias for functional type
FunctionalType: TypeAlias = Callable[..., tuple[Any, ...]]


class Functional(Operator[Unpack[Tin], tuple[jnp.ndarray]]):
    """Base class for functionals.

    A functional is an operator that maps from a space to the real numbers.
    The input and output types are specified by type variables.
    """

    @abstractmethod
    def forward(self, *args: Unpack[Tin]) -> tuple[jnp.ndarray]:
        """Apply forward operator.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Output of the operator
        """
        ...

    def __call__(self, *args: Unpack[Tin]) -> tuple[jnp.ndarray]:
        """Call the operator.

        This is a convenience method that calls the forward method.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Output of the operator
        """
        return self.forward(*args)


class ElementaryFunctional(Functional[Unpack[Tin]]):
    r"""Elementary functional base class.

    An elementary functional is a functional that can be written as
    :math:`f(x) = \phi ( \mathrm{weight} ( x - \mathrm{target}))`, returning a real value.
    It does not require another functional for initialization.
    """

    def __init__(
        self,
        weight: jnp.ndarray | float = 1.0,
        target: jnp.ndarray | float = 0.0,
        dim: Sequence[int] | None = None,
        keepdim: bool = False,
        divide_by_n: bool = False,
    ) -> None:
        """Initialize the elementary functional.

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
        super().__init__()
        self.weight = jnp.asarray(weight)
        self.target = jnp.asarray(target)
        self.dim = dim
        self.keepdim = keepdim
        self.divide_by_n = divide_by_n

    def _throw_if_negative_or_complex(self, value: jnp.ndarray | float, message: str | None = None) -> None:
        """Throw an error if the value is negative or complex.

        Parameters
        ----------
        value
            Value to check
        message
            Error message to use
        """
        if message is None:
            message = 'Value must be real and non-negative'
        if not jnp.all(jnp.real(value) >= 0):
            raise ValueError(message)
        if not jnp.all(jnp.imag(value) == 0):
            raise ValueError(message)

    def _divide_by_n(self, value: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Divide by the number of elements if divide_by_n is True.

        Parameters
        ----------
        value
            Value to divide
        shape
            Shape to use for division

        Returns
        -------
            Divided value
        """
        if self.divide_by_n:
            n = jnp.prod(jnp.array(shape))
            return value / n
        return value


class ProximableFunctional(Functional[Unpack[Tin]]):
    r"""ProximableFunctional Base Class.

    A proximable functional is a functional :math:`f(x)` that has a prox implementation,
    i.e. a function that yields :math:`\mathrm{argmin}_x \sigma f(x) + 1/2 ||x - y||_2^2`
    and a prox_convex_conjugate, yielding the prox of the convex conjugate.
    """

    @abstractmethod
    def prox(self, x: jnp.ndarray, sigma: jnp.ndarray | float = 1.0) -> tuple[jnp.ndarray]:
        r"""Apply proximal operator.

        Yields :math:`\mathrm{prox}_{\sigma f}(x) = \mathrm{argmin}_{p} (\sigma f(p) + 1/2 \|x-p\|_2^2` given :math:`x`
        and :math:`\sigma`.

        Parameters
        ----------
        x
            input array
        sigma
            scaling factor, must be positive

        Returns
        -------
            Proximal operator applied to the input array
        """
        ...

    def prox_convex_conj(self, x: jnp.ndarray, sigma: jnp.ndarray | float = 1.0) -> tuple[jnp.ndarray]:
        r"""Apply proximal operator of convex conjugate of functional.

        Yields :math:`\mathrm{prox}_{\sigma f^*}(x) = \mathrm{argmin}_{p} (\sigma f^*(p) + 1/2 \|x-p\|_2^2`,
        where :math:`f^*` denotes the convex conjugate of :math:`f`, given :math:`x` and :math:`\sigma`.

        Parameters
        ----------
        x
            input array
        sigma
            scaling factor, must be positive

        Returns
        -------
            Proximal operator  of the convex conjugate applied to the input array
        """
        if not isinstance(sigma, jnp.ndarray):
            sigma = jnp.array(1.0 * sigma)
        self._throw_if_negative_or_complex(sigma)
        sigma = jnp.clip(sigma, a_min=1e-8)
        return (x - sigma * self.prox(x / sigma, 1 / sigma)[0],)

    def __rmul__(self, scalar: jnp.ndarray | complex) -> ProximableFunctional:
        """Multiply functional with scalar."""
        if not isinstance(scalar, int | float | jnp.ndarray):
            return NotImplemented
        return ScaledProximableFunctional(self, scalar)

    def __or__(self, other: ProximableFunctional) -> mrmax.operators.ProximableFunctionalSeparableSum:
        """Create a ProximableFunctionalSeparableSum object from two proximable functionals.

        Parameters
        ----------
        other
            second functional to be summed

        Returns
        -------
            ProximableFunctionalSeparableSum object
        """
        if isinstance(other, ProximableFunctional):
            return mrmax.operators.ProximableFunctionalSeparableSum(self, other)
        return NotImplemented  # type: ignore[unreachable]


class ElementaryProximableFunctional(ElementaryFunctional[Unpack[Tin]], ProximableFunctional[Unpack[Tin]]):
    r"""Elementary proximable functional base class.

    Here, an 'elementary' functional is a functional that can be written as
    :math:`f(x) = \phi ( \mathrm{weight} ( x - \mathrm{target}))`, returning a real value.
    It does not require another functional for initialization.

    A proximable functional is a functional :math:`f(x)` that has a prox implementation,
    i.e. a function that yields :math:`\mathrm{argmin}_x \sigma f(x) + 1/2 \|x - y\|^2`.
    """


class ScaledFunctional(Functional):
    """Functional scaled by a scalar."""

    def __init__(self, functional: Functional, scale: jnp.ndarray | float) -> None:
        r"""Initialize a scaled functional.

        A scaled functional is a functional that is scaled by a scalar factor :math:`\alpha`,
        i.e. :math:`f(x) = \alpha g(x)`.

        Parameters
        ----------
        functional
            functional to be scaled
        scale
            scaling factor, must be real and positive
        """
        super().__init__()
        self.functional = functional
        self.scale = jnp.asarray(scale)

    def forward(self, x: jnp.ndarray) -> tuple[jnp.ndarray]:
        """Forward method.

        Parameters
        ----------
        x
            input array

        Returns
        -------
            scaled output of the functional
        """
        return (self.scale * self.functional(x)[0],)


class ScaledProximableFunctional(ProximableFunctional):
    """Proximable Functional scaled by a scalar."""

    def __init__(self, functional: ProximableFunctional, scale: jnp.ndarray | float) -> None:
        r"""Initialize a scaled proximable functional.

        A scaled functional is a functional that is scaled by a scalar factor :math:`\alpha`,
        i.e. :math:`f(x) = \alpha g(x)`.

        Parameters
        ----------
        functional
            proximable functional to be scaled
        scale
            scaling factor, must be real and positive
        """
        super().__init__()
        self.functional = functional
        self.scale = jnp.asarray(scale)

    def forward(self, x: jnp.ndarray) -> tuple[jnp.ndarray]:
        """Forward method.

        Parameters
        ----------
        x
            input array

        Returns
        -------
            scaled output of the functional
        """
        return (self.scale * self.functional(x)[0],)

    def prox(self, x: jnp.ndarray, sigma: jnp.ndarray | float = 1.0) -> tuple[jnp.ndarray]:
        """Proximal Mapping.

        Parameters
        ----------
        x
            input array
        sigma
            scaling factor

        Returns
        -------
            Proximal mapping applied to the input array
        """
        self._throw_if_negative_or_complex(
            self.scale, 'For prox to be defined, the scaling factor must be real and non-negative'
        )
        return (self.functional.prox(x, sigma * self.scale)[0],)

    def prox_convex_conj(self, x: jnp.ndarray, sigma: jnp.ndarray | float = 1.0) -> tuple[jnp.ndarray]:
        """Proximal Mapping of the convex conjugate.

        Parameters
        ----------
        x
            input array
        sigma
            scaling factor

        Returns
        -------
            Proximal mapping of the convex conjugate applied to the input array
        """
        self._throw_if_negative_or_complex(
            self.scale, 'For prox_convex_conj to be defined, the scaling factor must be real and non-negative'
        )
        return (self.scale * self.functional.prox_convex_conj(x / self.scale, sigma / self.scale)[0],)
