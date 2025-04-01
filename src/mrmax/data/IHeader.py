"""MR image data header (IHeader) class."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
from pydicom.dataset import Dataset
from typing_extensions import Self

from mrmax.data.KHeader import KHeader
from mrmax.data.SpatialDimension import SpatialDimension


class IHeader(eqx.Module):
    """MR image data header (IHeader) class.

    Attributes
    ----------
    resolution : SpatialDimension
        Resolution in meters.
    position : SpatialDimension
        Position in meters.
    orientation : Array
        Orientation matrix with shape (2, 3).
    echo_time : Float[Array, ""]
        Echo time in seconds.
    inversion_time : Float[Array, ""]
        Inversion time in seconds.
    flip_angle : Float[Array, ""]
        Flip angle in degrees.
    repetition_time : Float[Array, ""]
        Repetition time in seconds.
    slice_idx : int
        Slice index.
    phase_idx : int
        Phase index.
    repetition_idx : int
        Repetition index.
    """

    resolution: SpatialDimension
    position: SpatialDimension
    orientation: Array
    echo_time: Float[Array, '']
    inversion_time: Float[Array, '']
    flip_angle: Float[Array, '']
    repetition_time: Float[Array, '']
    slice_idx: int
    phase_idx: int
    repetition_idx: int

    def __init__(
        self,
        resolution: SpatialDimension,
        position: SpatialDimension,
        orientation: Array,
        echo_time: float | Array,
        inversion_time: float | Array,
        flip_angle: float | Array,
        repetition_time: float | Array,
        slice_idx: int,
        phase_idx: int,
        repetition_idx: int,
    ):
        """Initialize IHeader.

        Parameters
        ----------
        resolution : SpatialDimension
            Resolution in meters.
        position : SpatialDimension
            Position in meters.
        orientation : Array
            Orientation matrix with shape (2, 3).
        echo_time : float | Array
            Echo time in seconds.
        inversion_time : float | Array
            Inversion time in seconds.
        flip_angle : float | Array
            Flip angle in degrees.
        repetition_time : float | Array
            Repetition time in seconds.
        slice_idx : int
            Slice index.
        phase_idx : int
            Phase index.
        repetition_idx : int
            Repetition index.
        """
        self.resolution = resolution
        self.position = position
        self.orientation = orientation
        self.echo_time = jnp.asarray(echo_time)
        self.inversion_time = jnp.asarray(inversion_time)
        self.flip_angle = jnp.asarray(flip_angle)
        self.repetition_time = jnp.asarray(repetition_time)
        self.slice_idx = slice_idx
        self.phase_idx = phase_idx
        self.repetition_idx = repetition_idx

    @classmethod
    def from_kheader(cls, header: KHeader) -> Self:
        """Create IHeader object from a KHeader object.

        Parameters
        ----------
        header : KHeader
            MR raw data header containing required meta data for the image header.

        Returns
        -------
        Self
            Image header object.
        """
        # Calculate resolution from FOV and matrix size
        resolution = header.recon_fov / header.recon_matrix

        # Get position and orientation from acquisition info
        position = header.acq_info.position
        orientation = header.acq_info.orientation.as_matrix()

        # Get timing parameters using JAX's type-safe operations
        def get_mean_value(arr: Array | list | None, default: float = 0.0) -> Array:
            if arr is None:
                return jnp.array(default)
            if isinstance(arr, Array):
                return jnp.mean(arr)
            return jnp.array(arr[0]) if arr else jnp.array(default)

        echo_time = get_mean_value(header.te)
        inversion_time = get_mean_value(header.ti)
        flip_angle = jnp.rad2deg(get_mean_value(header.fa))
        repetition_time = get_mean_value(header.tr)

        # Get indices from acquisition info using JAX's type-safe operations
        def get_mean_index(arr: Array | list | None, default: int = 0) -> int:
            if arr is None:
                return default
            if isinstance(arr, Array):
                return int(jnp.mean(arr))
            return int(arr[0]) if arr else default

        slice_idx = get_mean_index(header.acq_info.idx.slice)
        phase_idx = get_mean_index(header.acq_info.idx.phase)
        repetition_idx = get_mean_index(header.acq_info.idx.repetition)

        return cls(
            resolution=resolution,
            position=position,
            orientation=orientation,
            echo_time=echo_time,
            inversion_time=inversion_time,
            flip_angle=flip_angle,
            repetition_time=repetition_time,
            slice_idx=slice_idx,
            phase_idx=phase_idx,
            repetition_idx=repetition_idx,
        )

    @classmethod
    def from_dicom(cls, dataset: Dataset) -> Self:
        """Create IHeader object from a DICOM dataset.

        Parameters
        ----------
        dataset : Dataset
            DICOM dataset containing required meta data for the image header.

        Returns
        -------
        Self
            Image header object.
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

        # Get timing parameters and convert to seconds
        echo_time = jnp.asarray(dataset.EchoTime, dtype=jnp.float32) / 1000.0
        inversion_time = jnp.asarray(dataset.InversionTime, dtype=jnp.float32) / 1000.0
        flip_angle = jnp.asarray(dataset.FlipAngle, dtype=jnp.float32)
        repetition_time = jnp.asarray(dataset.RepetitionTime, dtype=jnp.float32) / 1000.0

        # Get indices
        slice_idx = int(dataset.InstanceNumber)
        phase_idx = int(dataset.PhaseNumber)
        repetition_idx = int(dataset.RepetitionNumber)

        return cls(
            resolution=resolution,
            position=position,
            orientation=orientation,
            echo_time=echo_time,
            inversion_time=inversion_time,
            flip_angle=flip_angle,
            repetition_time=repetition_time,
            slice_idx=slice_idx,
            phase_idx=phase_idx,
            repetition_idx=repetition_idx,
        )

    def __repr__(self) -> str:
        """Return string representation of IHeader object.

        Returns
        -------
        str
            String representation of IHeader object.
        """
        out = (
            f'{type(self).__name__} with:\n'
            f'Resolution [m]: {self.resolution!s}\n'
            f'Position [m]: {self.position!s}\n'
            f'Orientation: {self.orientation!s}\n'
            f'Echo time [s]: {self.echo_time!s}\n'
            f'Inversion time [s]: {self.inversion_time!s}\n'
            f'Flip angle [deg]: {self.flip_angle!s}\n'
            f'Repetition time [s]: {self.repetition_time!s}\n'
            f'Slice index: {self.slice_idx!s}\n'
            f'Phase index: {self.phase_idx!s}\n'
            f'Repetition index: {self.repetition_idx!s}'
        )
        return out
