"""Class for coil sensitivity maps (csm)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing_extensions import Self

from mrmax.data.IData import IData
from mrmax.data.QData import QData
from mrmax.data.SpatialDimension import SpatialDimension

if TYPE_CHECKING:
    from mrmax.operators.SensitivityOp import SensitivityOp


@register_pytree_node_class
class CsmData(QData):
    """Coil sensitivity map class."""

    @classmethod
    def from_idata_walsh(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
    ) -> Self:
        """Create csm object from image data using iterative Walsh method.

        See also `~mrmax.algorithms.csm.walsh`.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            width of smoothing filter.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is `None`, which means that all elements are processed at once.

        Returns
        -------
        Self
            A new CsmData instance.
        """
        from mrmax.algorithms.csm.walsh import walsh

        # convert smoothing_width to SpatialDimension if int
        if isinstance(smoothing_width, int):
            smoothing_width = SpatialDimension(smoothing_width, smoothing_width, smoothing_width)

        # Flatten the data for batch processing
        shape = idata.data.shape
        flattened = jnp.reshape(idata.data, (-1, *shape[-4:]))

        # Process in batches using vmap
        csm_fun = jax.vmap(lambda img: walsh(img, smoothing_width), in_axes=0, out_axes=0)
        if chunk_size_otherdim is not None:
            # Split into chunks and process
            n_chunks = (flattened.shape[0] + chunk_size_otherdim - 1) // chunk_size_otherdim
            chunks = jnp.array_split(flattened, n_chunks)
            csm_chunks = [csm_fun(chunk) for chunk in chunks]
            csm_tensor = jnp.concatenate(csm_chunks, axis=0)
        else:
            csm_tensor = csm_fun(flattened)

        # Reshape back to original dimensions
        csm_tensor = jnp.reshape(csm_tensor, shape)
        return cls(header=idata.header, data=csm_tensor)

    @classmethod
    def from_idata_inati(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
    ) -> Self:
        """Create csm object from image data using Inati method.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            Size of the smoothing kernel.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is None, which means that all elements are processed at once.

        Returns
        -------
        Self
            A new CsmData instance.
        """
        from mrmax.algorithms.csm.inati import inati

        # Process in batches using vmap
        csm_fun = jax.vmap(lambda img: inati(img, smoothing_width), in_axes=0, out_axes=0)
        if chunk_size_otherdim is not None:
            # Split into chunks and process
            shape = idata.data.shape
            flattened = jnp.reshape(idata.data, (-1, *shape[-4:]))
            n_chunks = (flattened.shape[0] + chunk_size_otherdim - 1) // chunk_size_otherdim
            chunks = jnp.array_split(flattened, n_chunks)
            csm_chunks = [csm_fun(chunk) for chunk in chunks]
            csm_tensor = jnp.concatenate(csm_chunks, axis=0)
            csm_tensor = jnp.reshape(csm_tensor, shape)
        else:
            csm_tensor = csm_fun(idata.data)

        return cls(header=idata.header, data=csm_tensor)

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

    def as_operator(self) -> SensitivityOp:
        """Create SensitivityOp using a copy of the CSMs.

        Returns
        -------
        SensitivityOp
            A new sensitivity operator instance.
        """
        from mrmax.operators.SensitivityOp import SensitivityOp

        return SensitivityOp(self)
