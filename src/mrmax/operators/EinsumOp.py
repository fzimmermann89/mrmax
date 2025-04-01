"""Generalized Sum Multiplication Operator."""

from __future__ import annotations

import re

import equinox as eqx
from einops import einsum
from jaxtyping import Array
from typing_extensions import Self

from mrmax.operators.LinearOperator import LinearOperator


class EinsumOp(LinearOperator, eqx.Module):
    r"""A Linear Operator that implements sum products in Einstein notation.

    Implements :math:`A_{\mathrm{indices}_A}*x^{\mathrm{indices}_x} = y_{\mathrm{indices}_y}`
    with Einstein summation rules over the :math:`indices`, see `torch.einsum` or `einops.einsum`
    for more information. Note, that the indices must be space separated (einops convention).


    It can be used to implement tensor contractions, such as for example, different versions of
    matrix-vector or matrix-matrix products of the form `A @ x`, depending on the chosen einsum rules and
    shapes of `A` and `x`.

    Examples are:

    - matrix-vector multiplication of :math:`A` and the batched vector :math:`x = [x1, ..., xN]` consisting
      of :math:`N` vectors :math:`x1, x2, ..., xN`. Then, the operation defined by
      :math:`A @ x := \mathrm{diag}(A, A, ..., A) * [x1, x2, ..., xN]^T` = :math:`[A*x1, A*x2, ..., A*xN]^T`
      can be implemented by the einsum rule ``"i j, ... j -> ... i"``.

    - matrix-vector multiplication of a matrix :math:`A` consisting of :math:`N` different matrices
      :math:`A1, A2, ... AN` with one vector :math:`x`. Then, the operation defined by
      :math:`A @ x: = \mathrm{diag}(A1, A2,..., AN) * [x, x, ..., x]^T`
      can be implemented by the einsum rule ``"... i j, j -> ... i"``.

    - matrix-vector multiplication of a matrix :math:`A` consisting of :math:`N` different matrices
      :math:`A1, A2, ... AN` with a vector :math:`x = [x1,...,xN]` consisting
      of :math:`N` vectors :math:`x1, x2, ..., xN`. Then, the operation defined by
      :math:`A @ x: = \mathrm{diag}(A1, A2,..., AN) * [x1, x2, ..., xN]^T`
      can be implemented by the einsum rule ``"... i j, ... j -> ... i"``.
      This is the default behavior of the operator.
    """

    matrix: Array
    _forward_pattern: str
    _adjoint_pattern: str

    def __init__(self, matrix: Array, einsum_rule: str = '... i j, ... j -> ... i') -> None:
        """Initialize Einsum Operator.

        Parameters
        ----------
        matrix
            'Matrix' :math:`A` to be used as first factor in the sum product :math:`A*x`

        einsum_rule
            Einstein summation rule describing the forward of the operator.
            Also see torch.einsum for more information.
        """
        super().__init__()
        if (match := re.match('(.+),(.+)->(.+)', einsum_rule)) is None:
            raise ValueError(f'Einsum pattern should match (.+),(.+)->(.+) but got {einsum_rule}.')
        indices_matrix, indices_input, indices_output = match.groups()
        # swapping the input and output indices gets the adjoint rule
        self._adjoint_pattern = f'{indices_matrix},{indices_output}->{indices_input}'
        self._forward_pattern = einsum_rule
        self.matrix = matrix

    def forward(self, x: Array) -> tuple[Array]:
        """Sum-Multiplication of input :math:`x` with :math:`A`.

        :math:`A` and the rule used to perform the sum-product is set at initialization.

        Parameters
        ----------
        x
            input tensor to be multiplied with the 'matrix' :math:`A`.

        Returns
        -------
            result of matrix-vector multiplication
        """
        y = einsum(self.matrix, x, self._forward_pattern)
        return (y,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Multiplication of input with the adjoint of :math:`A`.

        Parameters
        ----------
        x
            tensor to be multiplied with hermitian/adjoint 'matrix' :math:`A`

        Returns
        -------
            result of adjoint sum product
        """
        y = einsum(self.matrix.conj(), x, self._adjoint_pattern)
        return (y,)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Einsum operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of EinsumOp."""
        return f'{type(self).__name__}(pattern={self._forward_pattern})'
