"""Base class for operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import equinox as eqx


class Operator(eqx.Module, ABC):
    """Base class for operators.

    An operator is a function that maps from one space to another.
    """

    @abstractmethod
    def forward(self, *args: Any) -> Any:
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

    def __call__(self, *args: Any) -> Any:
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

    def __matmul__(self, other: Operator) -> Operator:
        """Compose operators.

        Parameters
        ----------
        other
            Operator to compose with

        Returns
        -------
            Composed operator
        """
        return OperatorComposition(self, other)

    def __rmatmul__(self, other: Operator) -> Operator:
        """Compose operators.

        Parameters
        ----------
        other
            Operator to compose with

        Returns
        -------
            Composed operator
        """
        return OperatorComposition(other, self)


class OperatorComposition(Operator):
    """Composition of two operators."""

    def __init__(self, first: Operator, second: Operator) -> None:
        """Initialize the composition.

        Parameters
        ----------
        first
            First operator to apply
        second
            Second operator to apply
        """
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *args: Any) -> Any:
        """Apply the composed operators.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Output of the composed operators
        """
        return self.second(*self.first(*args))


class OperatorSum(Operator):
    """Sum of two operators."""

    def __init__(self, first: Operator, second: Operator) -> None:
        """Initialize the sum.

        Parameters
        ----------
        first
            First operator to apply
        second
            Second operator to apply
        """
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *args: Any) -> Any:
        """Apply the sum of operators.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Output of the sum of operators
        """
        return self.first(*args) + self.second(*args)


class LinearOperator(Operator):
    """Linear operator base class.

    A linear operator is an operator that satisfies:
    f(a*x + b*y) = a*f(x) + b*f(y)
    for all scalars a, b and arrays x, y.
    """

    @abstractmethod
    def forward(self, *args: Any) -> Any:
        """Apply forward operator."""
        ...

    @abstractmethod
    def adjoint(self, *args: Any) -> tuple[Any]:
        """Apply adjoint operator."""
        ...

    def gram(self) -> LinearOperator:
        """Return the Gram operator.

        The Gram operator is the composition of the adjoint with the forward operator.
        """
        return LinearOperatorComposition(self.adjoint, self)


class LinearOperatorComposition(LinearOperator):
    """Composition of two linear operators."""

    def __init__(self, first: LinearOperator, second: LinearOperator) -> None:
        """Initialize the composition.

        Parameters
        ----------
        first
            First operator to apply
        second
            Second operator to apply
        """
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *args: Any) -> Any:
        """Apply the composed operators.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Output of the composed operators
        """
        return self.second(*self.first(*args))

    def adjoint(self, *args: Any) -> tuple[Any]:
        """Apply the adjoint of the composed operators.

        Parameters
        ----------
        args
            Input arguments

        Returns
        -------
            Output of the adjoint of the composed operators
        """
        return self.first.adjoint(*self.second.adjoint(*args))
