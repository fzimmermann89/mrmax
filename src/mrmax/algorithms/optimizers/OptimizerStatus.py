"""Optimizer Status base class."""

import jax.numpy as jnp
from typing_extensions import TypedDict


class OptimizerStatus(TypedDict):
    """Base class for OptimizerStatus."""

    solution: tuple[jnp.ndarray, ...]
    """Current estimate(s) of the solution. """

    iteration_number: int
    """Current iteration of the (iterative) algorithm."""
