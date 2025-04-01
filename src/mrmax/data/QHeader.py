"""MR quantitative data header (QHeader) class."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array
from pydicom.dataset import Dataset
from typing_extensions import Self

from mrmax.data.IHeader import IHeader
from mrmax.data.KHeader import KHeader
from mrmax.data.SpatialDimension import SpatialDimension


class QHeader:
    """MR quantitative data header (QHeader) class.

    Attributes
    ----------
    resolution : SpatialDimension
        Resolution in meters.
    position : SpatialDimension
        Position in meters.
    orientation : Array
        Orientation matrix.
    index : int
        Index for the quantitative data.
    """

    resolution: SpatialDimension
    position: SpatialDimension
    orientation: Array
    index: int

    def __init__(
        self,
        resolution: SpatialDimension,
        position: SpatialDimension,
        orientation: Array,
        index: int,
    ):
        """Initialize QHeader.

        Parameters
        ----------
        resolution : SpatialDimension
            Resolution in meters.
        position : SpatialDimension
            Position in meters.
        orientation : Array
            Orientation matrix.
        index : int
            Index for the quantitative data.
        """
        self.resolution = resolution
        self.position = position
        self.orientation = orientation
        self.index = index

    def tree_flatten(self) -> tuple[tuple[Array], dict[str, Any]]:
        """Flatten the tree structure for JAX.

        Returns
        -------
        tuple[tuple[Array], dict[str, Any]]
            A tuple containing the children and auxiliary data.
        """
        children = (self.orientation,)
        aux_data = {
            'resolution': self.resolution,
            'position': self.position,
            'index': self.index,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Array]) -> Self:
        """Unflatten the tree structure for JAX.

        Parameters
        ----------
        aux_data : dict[str, Any]
            The auxiliary data containing non-array attributes.
        children : tuple[Array]
            The array attributes of the class.

        Returns
        -------
        Self
            A new instance of the class.
        """
        (orientation,) = children
        return cls(
            resolution=aux_data['resolution'],
            position=aux_data['position'],
            orientation=orientation,
            index=aux_data['index'],
        )

    @classmethod
    def from_iheader(cls, header: IHeader) -> Self:
        """Create QHeader object from an IHeader object.

        Parameters
        ----------
        header : IHeader
            MR image data header containing required meta data for the quantitative header.

        Returns
        -------
        Self
            Quantitative header object.
        """
        return cls(
            resolution=header.resolution,
            position=header.position,
            orientation=header.orientation,
            index=0,  # Default to 0 for single quantitative data
        )

    @classmethod
    def from_kheader(cls, header: KHeader) -> Self:
        """Create QHeader object from a KHeader object.

        Parameters
        ----------
        header : KHeader
            MR raw data header containing required meta data for the quantitative header.

        Returns
        -------
        Self
            Quantitative header object.
        """
        # Calculate resolution from FOV and matrix size
        resolution = header.recon_fov / header.recon_matrix

        # Get position and orientation from acquisition info
        position = header.acq_info.position
        orientation = header.acq_info.orientation.as_matrix()

        return cls(
            resolution=resolution,
            position=position,
            orientation=orientation,
            index=0,  # Default to 0 for single quantitative data
        )

    @classmethod
    def from_dicom(cls, dataset: Dataset) -> Self:
        """Create QHeader object from a DICOM dataset.

        Parameters
        ----------
        dataset : Dataset
            DICOM dataset containing required meta data for the quantitative header.

        Returns
        -------
        Self
            Quantitative header object.
        """
        # Pixel spacing in mm
        pixel_spacing = dataset.PixelSpacing
        # Slice thickness in mm
        slice_thickness = dataset.SliceThickness
        # Image position in mm
        image_position = dataset.ImagePositionPatient
        # Image orientation in mm
        image_orientation = dataset.ImageOrientationPatient

        # Convert to meters using JAX operations
        resolution = SpatialDimension(
            z=jnp.asarray(slice_thickness, dtype=jnp.float32) / 1000.0,
            y=jnp.asarray(pixel_spacing[1], dtype=jnp.float32) / 1000.0,
            x=jnp.asarray(pixel_spacing[0], dtype=jnp.float32) / 1000.0,
        )
        position = SpatialDimension(
            z=jnp.asarray(image_position[2], dtype=jnp.float32) / 1000.0,
            y=jnp.asarray(image_position[1], dtype=jnp.float32) / 1000.0,
            x=jnp.asarray(image_position[0], dtype=jnp.float32) / 1000.0,
        )
        orientation = jnp.asarray(image_orientation, dtype=jnp.float32).reshape(2, 3)

        # Get index
        index = int(dataset.InstanceNumber)

        return cls(
            resolution=resolution,
            position=position,
            orientation=orientation,
            index=index,
        )

    def __repr__(self) -> str:
        """Return string representation of QHeader object.

        Returns
        -------
        str
            String representation of QHeader object.
        """
        out = (
            f'{type(self).__name__} with:\n'
            f'Resolution [m]: {self.resolution!s}\n'
            f'Position [m]: {self.position!s}\n'
            f'Orientation: {self.orientation!s}\n'
            f'Index: {self.index!s}'
        )
        return out


# Register QHeader as a pytree
jax.tree_util.register_pytree_node_class(QHeader)
