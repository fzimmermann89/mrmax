"""MR image data (IData) class."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from einops import repeat
from jax.tree_util import register_pytree_node_class
from pydicom import dcmread
from typing_extensions import Self

from mrmax.data.Data import Data
from mrmax.data.IHeader import IHeader
from mrmax.data.KHeader import KHeader


@register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class IData(Data):
    """MR image data (IData) class.

    Attributes
    ----------
    data : jnp.ndarray
        Image data tensor with dimensions `(other, coils, z, y, x)`.
    header : IHeader
        Header describing image data.
    """

    data: jnp.ndarray
    """Image data tensor with dimensions `(other, coils, z, y, x)`."""
    header: IHeader
    """Header describing image data."""

    @classmethod
    def create(cls, data: jnp.ndarray, header: KHeader | IHeader) -> Self:
        """Create IData object from a tensor and an arbitrary mrmax header.

        Parameters
        ----------
        data : jnp.ndarray
            Image data tensor with dimensions `(other, coils, z, y, x)`.
        header : KHeader | IHeader
            mrmax header containing required meta data for the IHeader.

        Returns
        -------
        Self
            A new IData instance.

        Raises
        ------
        ValueError
            If the header type is not supported.
        """
        if isinstance(header, KHeader):
            iheader = IHeader.from_kheader(header)
        elif isinstance(header, IHeader):
            iheader = header
        else:
            raise ValueError(f'Invalid header type: {type(header)}')

        return cls(data=data, header=iheader)

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

    @classmethod
    def from_single_dicom(cls, filename: str | Path) -> Self:
        """Read single DICOM file and return IData object.

        Parameters
        ----------
        filename : str | Path
            Path to DICOM file.

        Returns
        -------
        Self
            IData object containing the DICOM data.
        """
        dataset = dcmread(filename)
        # Image data is 2D np.array of Uint16, which cannot directly be converted to tensor
        idata = jnp.asarray(dataset.pixel_array.astype(jnp.complex64))
        idata = repeat(idata, 'y x -> other coils z y x', other=1, coils=1, z=1)
        header = IHeader.from_dicom(dataset)
        return cls(data=idata, header=header)

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

    def root_sum_of_squares(self) -> jnp.ndarray:
        """Calculate root sum of squares over coil dimension.

        Computes the root sum of squares (RSS) over the coil dimension of the image data.
        This is a common method for combining multi-coil MRI data.

        Returns
        -------
        jnp.ndarray
            Root sum of squares image with dimensions `(other, 1, z, y, x)`.
            The coil dimension is reduced to 1 by taking the RSS.
        """
        return jnp.sqrt(jnp.sum(jnp.abs(self.data) ** 2, axis=1, keepdims=True))

    def __repr__(self) -> str:
        """Return string representation of IData object.

        Returns
        -------
        str
            String representation of IData object.
        """
        try:
            device = str(self.data.device())
        except RuntimeError:
            device = 'mixed'
        out = (
            f'{type(self).__name__} with shape: {list(self.data.shape)!s} and dtype {self.data.dtype}\n'
            f'Device: {device}\nResolution [m]: {self.header.resolution!s}.'
        )
        return out
