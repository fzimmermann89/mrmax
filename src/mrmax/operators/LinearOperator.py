"""General Linear Operator."""

from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import cast

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import Self

from mrmax.operators.Operator import Operator, OperatorSum
from mrmax.operators.ZeroOp import ZeroOp


class LinearOperator(Operator[Array, tuple[Array]]):
    """General Linear Operator.

    LinearOperators have exactly one input tensors and one output tensor,
    and fulfill :math:`f(a*x + b*y) = a*f(x) + b*f(y)`
    with :math:`a`, :math:`b` scalars and :math:`x`, :math:`y` tensors.

    LinearOperators can be composed, added, multiplied, applied to tensors.
    LinearOperators have an `~LinearOperator.H` property that returns the adjoint operator,
    and a `~LinearOperator.gram` property that returns the Gram operator.

    Subclasses must implement the forward and adjoint methods.
    When subclassing, the `adjoint_as_backward` class attribute can be set to `True`::

            class MyOperator(LinearOperator, adjoint_as_backward=True):
                ...

    This will make JAX use the adjoint method as the backward method of the forward,
    and the forward method as the backward method of the adjoint, avoiding the need to
    have differentiable forward and adjoint methods.
    """

    @abstractmethod
    def forward(self, x: Array) -> tuple[Array]:
        """Apply forward operator."""

    @abstractmethod
    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply adjoint operator."""

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator."""
        return LinearOperatorAdjoint(self)

    @property
    def gram(self) -> LinearOperator:
        """Gram operator."""
        return LinearOperatorComposition(self.H, self)

    def operator_norm(
        self,
        initial_value: Array,
        dim: Sequence[int] | None,
        max_iterations: int = 20,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-5,
        callback: Callable[[Array], None] | None = None,
    ) -> Array:
        """Power iteration for computing the operator norm of the operator.

        Parameters
        ----------
        initial_value
            Initial value for the power iteration.
        dim
            Dimensions to calculate the operator norm over. Other dimensions are assumed to be
            batch dimensions. None means all dimensions.
        max_iterations
            Maximum number of iterations used in the power iteration.
        relative_tolerance
            Relative tolerance for convergence.
        absolute_tolerance
            Absolute tolerance for convergence.
        callback
            Callback function to be called with the current estimate of the operator norm.

        Returns
        -------
        Estimated operator norm.
        """
        # initialize vector
        vector = initial_value
        op_norm_old = jnp.array(0.0)

        # power iteration
        for _ in range(max_iterations):
            # apply the operator to the vector
            (vector_new,) = self.gram(vector)

            # compute estimate of the operator norm
            product = vector.real * vector_new.real
            if jnp.iscomplexobj(vector) and jnp.iscomplexobj(vector_new):
                product = product + vector.imag * vector_new.imag
            op_norm = jnp.sqrt(jnp.sum(product, axis=dim, keepdims=True))

            # check if stopping criterion is fulfilled; if not continue the iteration
            if (absolute_tolerance > 0 or relative_tolerance > 0) and jnp.allclose(
                op_norm, op_norm_old, atol=absolute_tolerance, rtol=relative_tolerance
            ):
                break

            # normalize vector
            vector = vector_new / jnp.linalg.norm(vector_new, axis=dim, keepdims=True)
            op_norm_old = op_norm

            if callback is not None:
                callback(op_norm)

        return op_norm

    def __matmul__(self, other: LinearOperator) -> LinearOperator:
        """Operator composition.

        Returns ``lambda x: self(other(x))``
        """
        if isinstance(other, LinearOperator):
            return LinearOperatorComposition(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __add__(
        self, other: Operator[Array, tuple[Array]] | LinearOperator | Array
    ) -> Operator[Array, tuple[Array]] | LinearOperator:
        """Operator addition.

        Returns ``lambda x: self(x) + other(x)`` if other is a operator,
        ``lambda x: self(x) + other`` if other is a tensor
        """
        if isinstance(other, Array):
            # tensor addition
            return LinearOperatorSum(self, ZeroOp() * other)
        elif isinstance(self, ZeroOp):
            # neutral element of addition
            return other
        elif isinstance(other, ZeroOp):
            # neutral element of addition
            return self
        elif isinstance(other, LinearOperator):
            # sum of LinearOperators is linear
            return LinearOperatorSum(self, other)
        elif isinstance(other, Operator):
            # for general operators
            return OperatorSum(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __mul__(self, other: Array | complex) -> LinearOperator:
        """Operator elementwise left multiplication with tensor/scalar.

        Returns ``lambda x: self(x*other)``
        """
        if isinstance(other, (complex, float, int)):
            if other == 0:
                return ZeroOp()
            if other == 1:
                return self
            else:
                return LinearOperatorElementwiseProductLeft(self, other)
        elif isinstance(other, Array):
            return LinearOperatorElementwiseProductLeft(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __rmul__(self, other: Array | complex) -> LinearOperator:
        """Operator elementwise right multiplication with tensor/scalar.

        Returns ``lambda x: other*self(x)``
        """
        if isinstance(other, (complex, float, int)):
            if other == 0:
                return ZeroOp()
            if other == 1:
                return self
            else:
                return LinearOperatorElementwiseProductRight(self, other)
        elif isinstance(other, Array):
            return LinearOperatorElementwiseProductRight(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __and__(self, other: LinearOperator) -> LinearOperatorMatrix:
        """Vertical stacking of two LinearOperators.

        ``A&B`` is a `~mrmax.operators.LinearOperatorMatrix` with two rows,
        with ``(A&B)(x) == (A(x), B(x))``.
        See `mrmax.operators.LinearOperatorMatrix` for more information.
        """
        if not isinstance(other, LinearOperator):
            return NotImplemented  # type: ignore[unreachable]
        operators = [[self], [other]]
        return LinearOperatorMatrix(operators)

    def __or__(self, other: LinearOperator) -> LinearOperatorMatrix:
        """Horizontal stacking of two LinearOperators.

        ``A|B`` is a `~mrmax.operators.LinearOperatorMatrix` with two columns,
        with ``(A|B)(x1,x2) == A(x1)+B(x2)``.
        See `mrmax.operators.LinearOperatorMatrix` for more information.
        """
        if not isinstance(other, LinearOperator):
            return NotImplemented  # type: ignore[unreachable]
        operators = [[self, other]]
        return LinearOperatorMatrix(operators)


class LinearOperatorAdjoint(LinearOperator):
    """Adjoint of a linear operator."""

    def __init__(self, operator: LinearOperator) -> None:
        """Initialize the adjoint operator.

        Parameters
        ----------
        operator
            Operator to take the adjoint of.
        """
        super().__init__()
        self._operator = operator

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the adjoint operator.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Output tensor.
        """
        return self._operator.adjoint(x)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the operator.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Output tensor.
        """
        return self._operator.forward(x)


class LinearOperatorComposition(LinearOperator):
    """Linear operator composition.

    Performs operator1(operator2(x))
    """

    def __init__(self, operator1: LinearOperator, operator2: LinearOperator) -> None:
        """Linear operator composition initialization.

        Returns ``lambda x: operator1(operator2(x))``

        Parameters
        ----------
        operator1
            outer operator
        operator2
            inner operator
        """
        super().__init__()
        self._operator1 = operator1
        self._operator2 = operator2

    def forward(self, x: Array) -> tuple[Array]:
        """Linear operator composition."""
        return self._operator1(*self._operator2(x))

    def adjoint(self, x: Array) -> tuple[Array]:
        """Adjoint of the linear operator composition."""
        # (AB)^H = B^H A^H
        return self._operator2.adjoint(*self._operator1.adjoint(x))

    @property
    def gram(self) -> LinearOperator:
        """Gram operator."""
        # (AB)^H(AB) = B^H (A^H A) B
        return self._operator2.H @ self._operator1.gram @ self._operator2


class LinearOperatorSum(LinearOperator):
    """Linear operator addition."""

    _operators: list[LinearOperator]

    def __init__(self, operator1: LinearOperator, /, *other_operators: LinearOperator):
        """Linear operator addition initialization."""
        super().__init__()
        ops: list[LinearOperator] = []
        for op in (operator1, *other_operators):
            if isinstance(op, LinearOperatorSum):
                ops.extend(op._operators)
            else:
                ops.append(op)
        self._operators = cast(list[LinearOperator], eqx.ModuleList(ops))

    def forward(self, x: Array) -> tuple[Array]:
        """Linear operator addition."""
        return (functools.reduce(operator.add, (op(x)[0] for op in self._operators)),)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Adjoint of the linear operator addition."""
        # (A+B)^H = A^H + B^H
        return (functools.reduce(operator.add, (op.adjoint(x)[0] for op in self._operators)),)


class LinearOperatorElementwiseProductLeft(LinearOperator):
    """Left elementwise product of a linear operator with a tensor."""

    def __init__(self, operator: LinearOperator, tensor: Array) -> None:
        """Initialize the elementwise product.

        Parameters
        ----------
        operator
            Linear operator.
        tensor
            Tensor to multiply with.
        """
        super().__init__()
        self._operator = operator
        self._tensor = tensor

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the elementwise product.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Output tensor.
        """
        return (self._tensor * self._operator(x)[0],)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint of the elementwise product.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Output tensor.
        """
        return (self._operator.adjoint(self._tensor.conj() * x)[0],)


class LinearOperatorElementwiseProductRight(LinearOperator):
    """Right elementwise product of a linear operator with a tensor."""

    def __init__(self, operator: LinearOperator, tensor: Array) -> None:
        """Initialize the elementwise product.

        Parameters
        ----------
        operator
            Linear operator.
        tensor
            Tensor to multiply with.
        """
        super().__init__()
        self._operator = operator
        self._tensor = tensor

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the elementwise product.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Output tensor.
        """
        return (self._operator(x * self._tensor)[0],)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint of the elementwise product.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Output tensor.
        """
        return (self._operator.adjoint(x)[0] * self._tensor.conj(),)
