"""Averaging operator."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias
from warnings import warn

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from mrmax.operators.LinearOperator import LinearOperator

# Type alias for index types
IndexType: TypeAlias = Sequence[int] | Array | slice
IndexSequence: TypeAlias = list[IndexType]


@dataclass(frozen=True)
class Indices:
    """Immutable container for indices."""

    values: IndexSequence


class AveragingOp(LinearOperator, eqx.Module):
    """Averaging operator.

    This operator averages the input tensor along a specified dimension.
    The averaging is performed over groups of elements defined by the `idx` parameter.
    The output tensor will have the same shape as the input tensor, except for the `dim` dimension,
    which will have a size equal to the number of groups specified in `idx`. For each group,
    the average of the elements in that group is computed.

    For example, this operator can be used to simulate the effect of a sliding window average
    on a signal model.
    """

    domain_size: int | None
    _last_domain_size: int | None
    idx: Indices
    dim: int

    def __init__(
        self,
        dim: int,
        idx: IndexType | IndexSequence = slice(None),  # noqa: B008
        domain_size: int | None = None,
    ) -> None:
        """Initialize the averaging operator.

        Parameters
        ----------
        dim
            The dimension along which to average.
        idx
            The indices of the input tensor to average over. Each element of the sequence will result in a
            separate entry in the `dim` dimension of the output tensor.
            The entries can be either a sequence of integers or an integer tensor, a slice object, or a boolean tensor.
        domain_size
            The size of the input along `dim`. It is only used in the `adjoint` method.
            If not set, the size will be guessed from the input tensor during the forward pass.
        """
        super().__init__()
        self.domain_size = domain_size
        self._last_domain_size = domain_size
        if isinstance(idx, (Array, slice)):
            indices = [idx]
        elif isinstance(idx, Sequence):
            if all(isinstance(x, int) for x in idx):
                indices = [idx]
            else:
                indices = [x for x in idx]
        else:
            indices = [idx]
        self.idx = Indices(indices)
        self.dim = dim

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the averaging operator to the input tensor."""
        if self.domain_size and self.domain_size != x.shape[self.dim]:
            raise ValueError(f'Expected domain size {self.domain_size}, got {x.shape[self.dim]}')
        self._last_domain_size = x.shape[self.dim]

        placeholder = (slice(None),) * (self.dim % x.ndim)
        averaged = jnp.stack([jnp.mean(x[(*placeholder, i)], axis=self.dim) for i in self.idx.values], axis=self.dim)
        return (averaged,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint of the averaging operator to the input tensor."""
        if self.domain_size is None:
            if self._last_domain_size is None:
                raise ValueError('Domain size is not set. Please set it explicitly or run forward first.')
            warn(
                'Domain size is not set. Guessing the last used input size of the forward pass. '
                'Consider setting the domain size explicitly.',
                stacklevel=2,
            )
            self.domain_size = self._last_domain_size

        adjoint = jnp.zeros((*x.shape[: self.dim], self.domain_size, *x.shape[self.dim + 1 :]), dtype=x.dtype)
        placeholder = (slice(None),) * (self.dim % x.ndim)
        for i, group in enumerate(self.idx.values):
            if isinstance(group, slice):
                n = len(range(*group.indices(self.domain_size)))
            elif isinstance(group, Array) and group.dtype == jnp.bool_:
                n = int(jnp.sum(group))
            else:
                n = len(group)

            adjoint = adjoint.at[(*placeholder, group)].add(
                jnp.expand_dims(x[(*placeholder, i)], axis=self.dim).repeat(n, axis=self.dim) / n
            )

        return (adjoint,)
