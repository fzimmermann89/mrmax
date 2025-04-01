"""MR raw data / k-space data class."""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import EllipsisType
from typing import Any, Literal, TypeVar

import h5py
import ismrmrd
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jax.tree_util import register_pytree_node_class
from typing_extensions import Self

from mrmax.data.acq_filters import has_n_coils, is_image_acquisition
from mrmax.data.AcqInfo import AcqFlags, AcqInfo, convert_time_stamp_osi2, convert_time_stamp_siemens
from mrmax.data.EncodingLimits import EncodingLimits
from mrmax.data.KHeader import KHeader
from mrmax.data.KTrajectory import KTrajectory, KTrajectoryCalculator, KTrajectoryIsmrmrd
from mrmax.data.Rotation import Rotation
from mrmax.data.traj_calculators import KTrajectoryCalculator, KTrajectoryIsmrmrd
from mrmax.utils.typing import FileOrPath

RotationOrArray = TypeVar('RotationOrArray', bound=jnp.ndarray | Rotation)

KDIM_SORT_LABELS = (
    'k1',
    'k2',
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
    'user0',
    'user1',
    'user2',
    'user3',
    'user4',
    'user7',
)

OTHER_LABELS = (
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
    'user0',
    'user1',
    'user2',
    'user3',
    'user4',
    'user7',
)

T = TypeVar('T', bound='KData')


@register_pytree_node_class
@dataclass(frozen=True)
class KData:
    """MR raw data / k-space data class."""

    header: KHeader
    """Header information for k-space data"""
    data: jnp.ndarray
    """K-space data. Shape (*other coils k2 k1 k0)"""
    traj: KTrajectory
    """K-space trajectory along kz, ky and kx. Shape (*other k2 k1 k0)"""

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, KTrajectory], dict[str, Any]]:
        """Flatten the tree structure for JAX.

        Returns
        -------
        tuple[tuple[jnp.ndarray, KTrajectory], dict[str, Any]]
            A tuple containing the children and auxiliary data.
        """
        children = (self.data, self.traj)
        aux_data = {'header': self.header}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[jnp.ndarray, KTrajectory]) -> Self:
        """Unflatten the tree structure for JAX.

        Parameters
        ----------
        aux_data : dict[str, Any]
            The auxiliary data containing non-array attributes.
        children : tuple[jnp.ndarray, KTrajectory]
            The array attributes of the class.

        Returns
        -------
        Self
            A new instance of the class.
        """
        data, traj = children
        if not isinstance(traj, KTrajectory):
            raise TypeError(f'Expected KTrajectory, got {type(traj)}')
        return cls(header=aux_data['header'], data=data, traj=traj)

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
        traj = self.traj.to(device=device, dtype=dtype) if hasattr(self.traj, 'to') else self.traj
        return self.__class__(header=self.header, data=data, traj=traj)

    @classmethod
    def from_file(
        cls,
        filename: FileOrPath,
        trajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd,
        header_overwrites: dict[str, Any] | None = None,
        dataset_idx: int = -1,
        acquisition_filter_criterion: Callable = is_image_acquisition,
    ) -> Self:
        """Load k-space data from an ISMRMRD file.

        Parameters
        ----------
        filename
            path to the ISMRMRD file or file-like object
        trajectory
            KTrajectoryCalculator to calculate the k-space trajectory or an already calculated KTrajectory
            If a KTrajectory is given, the shape should be `(acquisisions 1 1 k0)` in the same order as the acquisitions
            in the ISMRMRD file.
        header_overwrites
            dictionary of key-value pairs to overwrite the header
        dataset_idx
            index of the ISMRMRD dataset to load (converter creates dataset, dataset_1, ...)
        acquisition_filter_criterion
            function which returns True if an acquisition should be included in KData
        """
        # Can raise FileNotFoundError
        with ismrmrd.File(filename, 'r') as file:
            dataset = file[list(file.keys())[dataset_idx]]
            ismrmrd_header = dataset.header
            acquisitions = dataset.acquisitions[:]
            try:
                mtime: int = h5py.h5g.get_objinfo(dataset['data']._contents.id).mtime
            except AttributeError:
                mtime = 0
            modification_time = datetime.datetime.fromtimestamp(mtime)

        acquisitions = [acq for acq in acquisitions if acquisition_filter_criterion(acq)]

        # we need the same number of receiver coils for all acquisitions
        n_coils_available = {acq.data.shape[0] for acq in acquisitions}
        if len(n_coils_available) > 1:
            if (
                ismrmrd_header.acquisitionSystemInformation is not None
                and ismrmrd_header.acquisitionSystemInformation.receiverChannels is not None
            ):
                n_coils = int(ismrmrd_header.acquisitionSystemInformation.receiverChannels)
            else:
                # most likely, highest number of elements are the coils used for imaging
                n_coils = int(max(n_coils_available))

            warnings.warn(
                f'Acquisitions with different number {n_coils_available} of receiver coil elements detected. '
                f'Data with {n_coils} receiver coil elements will be used.',
                stacklevel=1,
            )
            acquisitions = [acq for acq in acquisitions if has_n_coils(n_coils, acq)]

        if not acquisitions:
            raise ValueError('No acquisitions meeting the given filter criteria were found.')

        if ismrmrd_header.acquisitionSystemInformation is not None and isinstance(
            ismrmrd_header.acquisitionSystemInformation.systemVendor, str
        ):
            match ismrmrd_header.acquisitionSystemInformation.systemVendor.lower():
                case 'siemens':
                    convert_time_stamp = convert_time_stamp_siemens  # 2.5ms time steps
                case 'osi2':
                    convert_time_stamp = convert_time_stamp_osi2  # 1ms time steps
                case str(vendor):
                    warnings.warn(
                        f'Unknown vendor {vendor}. '
                        'Assuming Siemens time stamp format. If this is wrong, consider opening an Issue.',
                        stacklevel=1,
                    )
                    convert_time_stamp = convert_time_stamp_siemens  # 2.5ms time steps
        else:
            warnings.warn('No vendor information found. Assuming Siemens time stamp format.', stacklevel=1)
            convert_time_stamp = convert_time_stamp_siemens

        acq_info, (k0_center, n_k0_array, discard_pre, discard_post) = AcqInfo.from_ismrmrd_acquisitions(
            acquisitions,
            additional_fields=('center_sample', 'number_of_samples', 'discard_pre', 'discard_post'),
            convert_time_stamp=convert_time_stamp,
        )

        if len(jnp.unique(acq_info.idx.user5)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 5 indices for storing the kspace center line number.\n'
                'User 5 indices will be ignored',
                stacklevel=1,
            )

        if len(jnp.unique(acq_info.idx.user6)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 6 indices for storing the kspace center partition number.\n'
                'User 6 indices will be ignored',
                stacklevel=1,
            )

        shapes = jnp.unique(jnp.array([acq.data.shape[-1] for acq in acquisitions]) - discard_pre - discard_post)
        if len(shapes) > 1:
            warnings.warn(
                f'Acquisitions have different shape. Got {list(shapes)}. '
                f'Keeping only acquisistions with {shapes[-1]} data samples. Note: discard_pre and discard_post '
                f'{"have been applied. " if discard_pre.any() or discard_post.any() else "were empty. "}'
                'Please open an issue of you need to handle this kind of data.',
                stacklevel=1,
            )
        data = jnp.stack(
            [
                jnp.array(acq.data[..., pre : acq.data.shape[-1] - post], dtype=jnp.complex64)
                for acq, pre, post in zip(acquisitions, discard_pre, discard_post, strict=True)
                if acq.data.shape[-1] - pre - post == shapes[-1]
            ]
        )
        data = rearrange(data, 'acquisitions coils k0 -> acquisitions coils 1 1 k0')

        # Raises ValueError if required fields are missing in the header
        header = KHeader.from_ismrmrd(
            ismrmrd_header,
            acq_info,
            defaults={
                'datetime': modification_time,  # use the modification time of the dataset as fallback
                'trajectory': trajectory,
            },
            overwrite=header_overwrites,
        )
        # Calculate trajectory and check if it matches the kdata shape
        match trajectory:
            case KTrajectoryIsmrmrd():
                trajectory_ = trajectory(acquisitions, encoding_matrix=header.encoding_matrix)
            case KTrajectoryCalculator():
                reversed_readout_mask = (header.acq_info.flags[..., 0] & AcqFlags.ACQ_IS_REVERSE.value).astype(bool)
                n_k0_unique = jnp.unique(n_k0_array)
                if len(n_k0_unique) > 1:
                    raise ValueError(
                        'Trajectory can only be calculated for constant number of readout samples.\n'
                        f'Got unique values {list(n_k0_unique)}'
                    )
                encoding_limits = EncodingLimits.from_ismrmrd_header(ismrmrd_header)
                trajectory_ = trajectory(
                    n_k0=int(n_k0_unique[0]),
                    k0_center=k0_center,
                    k1_idx=header.acq_info.idx.k1,
                    k1_center=encoding_limits.k1.center,
                    k2_idx=header.acq_info.idx.k2,
                    k2_center=encoding_limits.k2.center,
                    reversed_readout_mask=reversed_readout_mask,
                    encoding_matrix=header.encoding_matrix,
                )
            case KTrajectory():
                try:
                    jax.eval_shape(lambda x, y: None, trajectory.broadcasted_shape, (data.shape[0], *data.shape[-3:]))
                except ValueError:
                    raise ValueError(
                        f'Trajectory shape {trajectory.broadcasted_shape} does not match data shape {data.shape}.'
                    ) from None
                trajectory_ = trajectory
            case _:
                raise TypeError(
                    'ktrajectory must be KTrajectoryIsmrmrd, KTrajectory or KTrajectoryCalculator'
                    f'not {type(trajectory)}',
                )

        kdata = cls(header=header, data=data, traj=trajectory_)
        kdata = kdata.reshape_by_idx()
        return kdata

    def reshape_by_idx(self) -> Self:
        """Reshape data and trajectory according to the indices in the header.

        Returns
        -------
        Self
            A new KData object with reshaped data and trajectory
        """
        # Get the shape of the data
        idx = self.header.acq_info.idx
        shape_dict = {
            'average': len(jnp.unique(idx.average)),
            'slice': len(jnp.unique(idx.slice)),
            'contrast': len(jnp.unique(idx.contrast)),
            'phase': len(jnp.unique(idx.phase)),
            'repetition': len(jnp.unique(idx.repetition)),
            'set': len(jnp.unique(idx.set)),
            'user0': len(jnp.unique(idx.user0)),
            'user1': len(jnp.unique(idx.user1)),
            'user2': len(jnp.unique(idx.user2)),
            'user3': len(jnp.unique(idx.user3)),
            'user4': len(jnp.unique(idx.user4)),
            'user7': len(jnp.unique(idx.user7)),
            'k2': len(jnp.unique(idx.k2)),
            'k1': len(jnp.unique(idx.k1)),
        }

        # Create the new shape
        new_shape = []
        for label in KDIM_SORT_LABELS:
            if shape_dict[label] > 1:
                new_shape.append(shape_dict[label])

        # Reshape data and trajectory
        if new_shape:
            data = jnp.reshape(self.data, (*new_shape, *self.data.shape[1:]))
            traj = KTrajectory(
                kz=jnp.reshape(self.traj.kz, (*new_shape, *self.traj.kz.shape[1:])),
                ky=jnp.reshape(self.traj.ky, (*new_shape, *self.traj.ky.shape[1:])),
                kx=jnp.reshape(self.traj.kx, (*new_shape, *self.traj.kx.shape[1:])),
            )
        else:
            data = self.data
            traj = self.traj

        return self.__class__(header=self.header, data=data, traj=traj)

    def __repr__(self) -> str:
        """Get string representation of KData.

        Returns
        -------
        str
            String representation of KData.
        """
        try:
            device = str(self.data.device())
        except RuntimeError:
            device = 'mixed'
        out = (
            f'{type(self).__name__} with shape: {list(self.data.shape)!s} and dtype {self.data.dtype}\nDevice: {device}'
        )
        return out

    def compress_coils(
        self,
        n_compressed_coils: int,
        batch_dims: None | Sequence[int] = None,
        joint_dims: Sequence[int] | EllipsisType = ...,
    ) -> Self:
        """Compress coils using SVD.

        Parameters
        ----------
        n_compressed_coils
            Number of coils after compression
        batch_dims
            Dimensions to batch over. If None, all dimensions except coils and k-space dimensions are batched.
        joint_dims
            Dimensions to jointly compress. If ..., k-space dimensions are jointly compressed.

        Returns
        -------
        Self
            A new KData object with compressed coils
        """
        # Get the shape of the data
        shape = self.data.shape
        n_dims = len(shape)

        # Default batch_dims: all dimensions except coils and k-space
        if batch_dims is None:
            batch_dims = list(range(n_dims - 4))

        # Default joint_dims: k-space dimensions
        if joint_dims is ...:
            joint_dims = list(range(n_dims - 3, n_dims))

        # Prepare dimensions for reshape
        batch_size = int(jnp.prod(jnp.array([shape[i] for i in batch_dims]))) if batch_dims else 1
        joint_size = int(jnp.prod(jnp.array([shape[i] for i in joint_dims])))

        # Reshape data for SVD
        data_reshaped = jnp.reshape(self.data, (batch_size, shape[-4], joint_size))

        # Perform SVD for each batch
        def compress_batch(x):
            u, s, _ = jnp.linalg.svd(x, full_matrices=False)
            return u[..., :n_compressed_coils] * s[..., :n_compressed_coils, None]

        compressed_data = jax.vmap(compress_batch)(data_reshaped)

        # Reshape back to original dimensions
        new_shape = list(shape)
        new_shape[-4] = n_compressed_coils
        compressed_data = jnp.reshape(compressed_data, new_shape)

        return self.__class__(header=self.header, data=compressed_data, traj=self.traj)

    def remove_readout_os(self) -> Self:
        """Remove readout oversampling.

        Returns
        -------
        Self
            A new KData object with removed readout oversampling
        """
        # Get the shape of the data
        shape = self.data.shape

        # Define the crop function
        def crop_readout(data_to_crop: jnp.ndarray) -> jnp.ndarray:
            """Crop the readout dimension of the data."""
            # Get the center and size
            n_k0 = data_to_crop.shape[-1]
            n_k0_cropped = n_k0 // 2

            # Calculate start and end indices
            start_idx = (n_k0 - n_k0_cropped) // 2
            end_idx = start_idx + n_k0_cropped

            # Crop the data
            return data_to_crop[..., start_idx:end_idx]

        # Crop data and trajectory
        cropped_data = crop_readout(self.data)
        cropped_traj = KTrajectory(
            kz=crop_readout(self.traj.kz),
            ky=crop_readout(self.traj.ky),
            kx=crop_readout(self.traj.kx),
        )

        return self.__class__(header=self.header, data=cropped_data, traj=cropped_traj)

    def select_other_subset(
        self,
        subset_idx: jnp.ndarray,
        subset_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Select a subset of the data along a dimension.

        Parameters
        ----------
        subset_idx
            Indices to select
        subset_label
            Label of the dimension to select from

        Returns
        -------
        Self
            A new KData object with the selected subset
        """
        # Get the shape of the data
        shape = self.data.shape

        # Find the dimension to select from
        try:
            dim_idx = KDIM_SORT_LABELS.index(subset_label)
        except ValueError:
            raise ValueError(f'Unknown label {subset_label}') from None

        # Select the subset
        selected_data = jnp.take(self.data, subset_idx, axis=dim_idx)
        selected_traj = KTrajectory(
            kz=jnp.take(self.traj.kz, subset_idx, axis=dim_idx),
            ky=jnp.take(self.traj.ky, subset_idx, axis=dim_idx),
            kx=jnp.take(self.traj.kx, subset_idx, axis=dim_idx),
        )

        return self.__class__(header=self.header, data=selected_data, traj=selected_traj)

    def split_k1_into_other(
        self,
        split_idx: jnp.ndarray,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Split k1 dimension into other dimension.

        Parameters
        ----------
        split_idx
            Indices to split at
        other_label
            Label of the dimension to split into

        Returns
        -------
        Self
            A new KData object with the split dimension
        """
        # Get the shape of the data
        shape = self.data.shape

        # Find the dimensions
        k1_idx = KDIM_SORT_LABELS.index('k1')
        other_idx = KDIM_SORT_LABELS.index(other_label)

        # Define the split function
        def split(data: RotationOrArray) -> RotationOrArray:
            """Split the k1 dimension."""
            # Broadcast k1 dimension
            data = repeat(data, f'... k1 k0 -> ... ({len(split_idx)} k1) k0')
            return data

        # Split data and trajectory
        split_data = split(self.data)
        split_traj = KTrajectory(
            kz=split(self.traj.kz),
            ky=split(self.traj.ky),
            kx=split(self.traj.kx),
        )

        return self.__class__(header=self.header, data=split_data, traj=split_traj)
