"""Density compensation data (DcfData) class."""

from __future__ import annotations

import dataclasses
from functools import reduce
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing_extensions import Self

from mrmax.algorithms.dcf.dcf_voronoi import dcf_1d, dcf_2d3d_voronoi
from mrmax.data.Data import Data
from mrmax.data.KTrajectory import KTrajectory
from mrmax.utils import smap

if TYPE_CHECKING:
    from mrmax.operators.DensityCompensationOp import DensityCompensationOp


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DcfData(Data):
    """Density compensation data (DcfData) class."""

    data: jnp.ndarray
    """Density compensation values. Shape `(... other, coils, k2, k1, k0)`"""

    @classmethod
    def from_traj_voronoi(cls, traj: KTrajectory) -> Self:
        """Calculate dcf using voronoi approach for 2D or 3D trajectories.

        Parameters
        ----------
        traj
            Trajectory to calculate the density compensation for. Can be broadcasted or dense.

        Returns
        -------
        Self
            A new DcfData instance.
        """
        dcfs = []

        ks = [traj.kz, traj.ky, traj.kx]
        spatial_dims = (-3, -2, -1)
        ks_needing_voronoi = set()
        for dim in spatial_dims:
            non_singleton_ks = [ax for ax in ks if ax.shape[dim] != 1]
            if len(non_singleton_ks) == 1:
                # Found a dimension with only one non-singleton axes in ks
                # --> Can handle this as a 1D trajectory
                dcfs.append(smap(dcf_1d, non_singleton_ks.pop(), (dim,)))
            elif len(non_singleton_ks) > 0:
                # More than one of the ks is non-singleton
                # --> A full dimension needing voronoi
                ks_needing_voronoi |= set(non_singleton_ks)
            else:
                # A dimension in which each of ks is singleton
                # --> Don't need to do anything
                pass

        if ks_needing_voronoi:
            # Handle full dimensions needing voronoi
            stacked = jnp.stack(jax.numpy.broadcast_arrays(*ks_needing_voronoi), -4)
            dcfs.append(smap(dcf_2d3d_voronoi, stacked, 4))

        if dcfs:
            # Multiply all dcfs together
            dcf = reduce(jnp.multiply, dcfs)
        else:
            # Edgecase: Only singleton spatial dimensions
            dcf = jnp.ones((*traj.broadcasted_shape[-3:], 1, 1, 1))

        return cls(data=dcf, header=None)

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

    def as_operator(self) -> DensityCompensationOp:
        """Create a density compensation operator using a copy of the DCF.

        Returns
        -------
        DensityCompensationOp
            A new density compensation operator instance.
        """
        from mrmax.operators.DensityCompensationOp import DensityCompensationOp

        return DensityCompensationOp(jax.tree_util.tree_map(lambda x: x, self.data))

    def __repr__(self) -> str:
        """Return string representation of DcfData object.

        Returns
        -------
        str
            String representation of DcfData object.
        """
        try:
            device = str(self.data.device())
        except RuntimeError:
            device = 'mixed'
        out = (
            f'{type(self).__name__} with shape: {list(self.data.shape)!s} and dtype {self.data.dtype}\nDevice: {device}'
        )
        return out
