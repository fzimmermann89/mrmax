"""Rearrange Operator."""

from __future__ import annotations

import re

import equinox as eqx
from einops import rearrange
from jaxtyping import Array
from typing_extensions import Self

from mrmax.operators.LinearOperator import LinearOperator


class RearrangeOp(LinearOperator, eqx.Module):
    """A Linear Operator that implements rearranging of axes.

    Wraps the `einops.rearrange` function to rearrange the axes of a tensor.
    """

    _forward_pattern: str
    _adjoint_pattern: str
    additional_info: dict[str, int]

    def __init__(self, pattern: str, additional_info: dict[str, int] | None = None) -> None:
        """Initialize Einsum Operator.

        Parameters
        ----------
        pattern
            Pattern describing the forward of the operator.
            Also see `einops.rearrange` for more information.
            Example: "... h w -> ... (w h)"
        additional_info
            Additional information passed to the rearrange function,
            describing the size of certain dimensions.
            Might be required for the adjoint rule.
            Example: {'h': 2, 'w': 2}
        """
        super().__init__()
        if (match := re.match('(.+)->(.+)', pattern)) is None:
            raise ValueError(f'pattern should match (.+)->(.+) but got {pattern}.')
        input_pattern, output_pattern = match.groups()
        # swapping the input and output gets the adjoint rule
        self._adjoint_pattern = f'{output_pattern}->{input_pattern}'
        self._forward_pattern = pattern
        self.additional_info = {} if additional_info is None else additional_info

    def forward(self, x: Array) -> tuple[Array]:
        """Rearrange input.

        The rule used to perform the rearranging is set at initialization.

        Parameters
        ----------
        x
            input tensor to be rearranged

        Returns
        -------
            rearranged tensor
        """
        y = rearrange(x, self._forward_pattern, **self.additional_info)
        return (y,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Rearrange input with the adjoint rule.

        Parameters
        ----------
        x
            tensor to be rearranged

        Returns
        -------
            rearranged tensor
        """
        y = rearrange(x, self._adjoint_pattern, **self.additional_info)
        return (y,)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Rearrange operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of RearrangeOp."""
        return f'{type(self).__name__}(pattern={self._forward_pattern})'
