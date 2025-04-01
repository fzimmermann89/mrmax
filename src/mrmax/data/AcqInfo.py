"""Acquisition information class."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias, overload

import ismrmrd
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from typing_extensions import Self

from mrmax.data.Rotation import Rotation
from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.utils.reshape import unsqueeze_at, unsqueeze_right

_convert_time_stamp_type: TypeAlias = Callable[
    [
        Float[Array, '*other coils k2 k1 k0'],
        Literal[
            'acquisition_time_stamp', 'physiology_time_stamp_1', 'physiology_time_stamp_2', 'physiology_time_stamp_3'
        ],
    ],
    Float[Array, '*other coils k2 k1 k0'],
]


def convert_time_stamp_siemens(
    timestamp: Float[Array, '*other coils k2 k1 k0'],
    _: str,
) -> Float[Array, '*other coils k2 k1 k0']:
    """Convert Siemens time stamp to seconds."""
    return timestamp.astype(jnp.float64) * 2.5e-3


def convert_time_stamp_osi2(
    timestamp: Float[Array, '*other coils k2 k1 k0'],
    _: str,
) -> Float[Array, '*other coils k2 k1 k0']:
    """Convert OSI2 time stamp to seconds."""
    return timestamp.astype(jnp.float64) * 1e-3


def _int_factory() -> Int[Array, '1 1 1 1 1']:
    return jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.int64)


def _float_factory() -> Float[Array, '1 1 1 1 1']:
    return jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)


def _position_factory() -> SpatialDimension[Float[Array, '1 1 1 1 1']]:
    return SpatialDimension(
        jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32),
    )


class AcqIdx:
    """Acquisition index for each readout."""

    k1: Int[Array, '*other coils k2 k1 k0']
    k2: Int[Array, '*other coils k2 k1 k0']
    average: Int[Array, '*other coils k2 k1 k0']
    slice: Int[Array, '*other coils k2 k1 k0']
    contrast: Int[Array, '*other coils k2 k1 k0']
    phase: Int[Array, '*other coils k2 k1 k0']
    repetition: Int[Array, '*other coils k2 k1 k0']
    set: Int[Array, '*other coils k2 k1 k0']
    segment: Int[Array, '*other coils k2 k1 k0']
    user0: Int[Array, '*other coils k2 k1 k0']
    user1: Int[Array, '*other coils k2 k1 k0']
    user2: Int[Array, '*other coils k2 k1 k0']
    user3: Int[Array, '*other coils k2 k1 k0']
    user4: Int[Array, '*other coils k2 k1 k0']
    user5: Int[Array, '*other coils k2 k1 k0']
    user6: Int[Array, '*other coils k2 k1 k0']
    user7: Int[Array, '*other coils k2 k1 k0']

    def __init__(
        self,
        k1: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        k2: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        average: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        slice: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        contrast: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        phase: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        repetition: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        set: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        segment: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user0: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user1: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user2: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user3: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user4: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user5: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user6: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        user7: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
    ):
        """Initialize AcqIdx.

        Parameters
        ----------
        k1 : Int[Array, '*other coils k2 k1 k0']
            First phase encoding.
        k2 : Int[Array, '*other coils k2 k1 k0']
            Second phase encoding.
        average : Int[Array, '*other coils k2 k1 k0']
            Signal average.
        slice : Int[Array, '*other coils k2 k1 k0']
            Slice number (multi-slice 2D).
        contrast : Int[Array, '*other coils k2 k1 k0']
            Echo number in multi-echo.
        phase : Int[Array, '*other coils k2 k1 k0']
            Cardiac phase.
        repetition : Int[Array, '*other coils k2 k1 k0']
            Counter in repeated/dynamic acquisitions.
        set : Int[Array, '*other coils k2 k1 k0']
            Sets of different preparation, e.g. flow encoding, diffusion weighting.
        segment : Int[Array, '*other coils k2 k1 k0']
            Counter for segmented acquisitions.
        user0 : Int[Array, '*other coils k2 k1 k0']
            User index 0.
        user1 : Int[Array, '*other coils k2 k1 k0']
            User index 1.
        user2 : Int[Array, '*other coils k2 k1 k0']
            User index 2.
        user3 : Int[Array, '*other coils k2 k1 k0']
            User index 3.
        user4 : Int[Array, '*other coils k2 k1 k0']
            User index 4.
        user5 : Int[Array, '*other coils k2 k1 k0']
            User index 5.
        user6 : Int[Array, '*other coils k2 k1 k0']
            User index 6.
        user7 : Int[Array, '*other coils k2 k1 k0']
            User index 7.
        """
        self.k1 = k1
        self.k2 = k2
        self.average = average
        self.slice = slice
        self.contrast = contrast
        self.phase = phase
        self.repetition = repetition
        self.set = set
        self.segment = segment
        self.user0 = user0
        self.user1 = user1
        self.user2 = user2
        self.user3 = user3
        self.user4 = user4
        self.user5 = user5
        self.user6 = user6
        self.user7 = user7

        # Ensure that all indices are broadcastable
        try:
            jax.lax.broadcast_shapes(*[getattr(self, attr).shape for attr in self.__dict__])
        except ValueError:
            raise ValueError('The acquisition index dimensions must be broadcastable.') from None
        if any(getattr(self, attr).ndim < 5 for attr in self.__dict__):
            raise ValueError('The acquisition index tensors should each have at least 5 dimensions.')

    def tree_flatten(self) -> tuple[tuple[Array, ...], dict[str, Any]]:
        """Flatten the tree structure for JAX."""
        children = (
            self.k1,
            self.k2,
            self.average,
            self.slice,
            self.contrast,
            self.phase,
            self.repetition,
            self.set,
            self.segment,
            self.user0,
            self.user1,
            self.user2,
            self.user3,
            self.user4,
            self.user5,
            self.user6,
            self.user7,
        )
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Array, ...]) -> Self:
        """Unflatten the tree structure for JAX."""
        (
            k1,
            k2,
            average,
            slice,
            contrast,
            phase,
            repetition,
            set,
            segment,
            user0,
            user1,
            user2,
            user3,
            user4,
            user5,
            user6,
            user7,
        ) = children
        return cls(
            k1=k1,
            k2=k2,
            average=average,
            slice=slice,
            contrast=contrast,
            phase=phase,
            repetition=repetition,
            set=set,
            segment=segment,
            user0=user0,
            user1=user1,
            user2=user2,
            user3=user3,
            user4=user4,
            user5=user5,
            user6=user6,
            user7=user7,
        )


class UserValues:
    """User Values used in AcqInfo."""

    float0: Float[Array, '*other coils k2 k1 k0']
    float1: Float[Array, '*other coils k2 k1 k0']
    float2: Float[Array, '*other coils k2 k1 k0']
    float3: Float[Array, '*other coils k2 k1 k0']
    float4: Float[Array, '*other coils k2 k1 k0']
    float5: Float[Array, '*other coils k2 k1 k0']
    float6: Float[Array, '*other coils k2 k1 k0']
    float7: Float[Array, '*other coils k2 k1 k0']
    int0: Int[Array, '*other coils k2 k1 k0']
    int1: Int[Array, '*other coils k2 k1 k0']
    int2: Int[Array, '*other coils k2 k1 k0']
    int3: Int[Array, '*other coils k2 k1 k0']
    int4: Int[Array, '*other coils k2 k1 k0']
    int5: Int[Array, '*other coils k2 k1 k0']
    int6: Int[Array, '*other coils k2 k1 k0']
    int7: Int[Array, '*other coils k2 k1 k0']

    def __init__(
        self,
        float0: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        float1: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        float2: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        float3: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        float4: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        float5: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        float6: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        float7: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        int0: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        int1: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        int2: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        int3: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        int4: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        int5: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        int6: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        int7: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
    ):
        """Initialize UserValues."""
        self.float0 = float0
        self.float1 = float1
        self.float2 = float2
        self.float3 = float3
        self.float4 = float4
        self.float5 = float5
        self.float6 = float6
        self.float7 = float7
        self.int0 = int0
        self.int1 = int1
        self.int2 = int2
        self.int3 = int3
        self.int4 = int4
        self.int5 = int5
        self.int6 = int6
        self.int7 = int7

    def tree_flatten(self) -> tuple[tuple[Array, ...], dict[str, Any]]:
        """Flatten the tree structure for JAX."""
        children = (
            self.float0,
            self.float1,
            self.float2,
            self.float3,
            self.float4,
            self.float5,
            self.float6,
            self.float7,
            self.int0,
            self.int1,
            self.int2,
            self.int3,
            self.int4,
            self.int5,
            self.int6,
            self.int7,
        )
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Array, ...]) -> Self:
        """Unflatten the tree structure for JAX."""
        (
            float0,
            float1,
            float2,
            float3,
            float4,
            float5,
            float6,
            float7,
            int0,
            int1,
            int2,
            int3,
            int4,
            int5,
            int6,
            int7,
        ) = children
        return cls(
            float0=float0,
            float1=float1,
            float2=float2,
            float3=float3,
            float4=float4,
            float5=float5,
            float6=float6,
            float7=float7,
            int0=int0,
            int1=int1,
            int2=int2,
            int3=int3,
            int4=int4,
            int5=int5,
            int6=int6,
            int7=int7,
        )


class PhysiologyTimestamps:
    """Time stamps relative to physiological triggering, e.g. ECG, in seconds."""

    timestamp0: Float[Array, '*other coils k2 k1 k0']
    timestamp1: Float[Array, '*other coils k2 k1 k0']
    timestamp2: Float[Array, '*other coils k2 k1 k0']

    def __init__(
        self,
        timestamp0: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        timestamp1: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        timestamp2: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
    ):
        """Initialize PhysiologyTimestamps."""
        self.timestamp0 = timestamp0
        self.timestamp1 = timestamp1
        self.timestamp2 = timestamp2

    def tree_flatten(self) -> tuple[tuple[Array, ...], dict[str, Any]]:
        """Flatten the tree structure for JAX."""
        children = (self.timestamp0, self.timestamp1, self.timestamp2)
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Array, ...]) -> Self:
        """Unflatten the tree structure for JAX."""
        timestamp0, timestamp1, timestamp2 = children
        return cls(timestamp0=timestamp0, timestamp1=timestamp1, timestamp2=timestamp2)


class AcqInfo:
    """Acquisition information for each readout."""

    idx: AcqIdx
    acquisition_time_stamp: Float[Array, '*other coils k2 k1 k0']
    flags: Int[Array, '*other coils k2 k1 k0']
    orientation: Rotation
    patient_table_position: SpatialDimension[Float[Array, '*other coils k2 k1 k0']]
    physiology_time_stamps: PhysiologyTimestamps
    position: SpatialDimension[Float[Array, '*other coils k2 k1 k0']]
    sample_time_us: Float[Array, '*other coils k2 k1 k0']
    user: UserValues

    def __init__(
        self,
        idx: AcqIdx = AcqIdx(),
        acquisition_time_stamp: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        flags: Int[Array, '*other coils k2 k1 k0'] = _int_factory(),
        orientation: Rotation = Rotation.identity((1, 1, 1, 1, 1)),
        patient_table_position: SpatialDimension[Float[Array, '*other coils k2 k1 k0']] = _position_factory(),
        physiology_time_stamps: PhysiologyTimestamps = PhysiologyTimestamps(),
        position: SpatialDimension[Float[Array, '*other coils k2 k1 k0']] = _position_factory(),
        sample_time_us: Float[Array, '*other coils k2 k1 k0'] = _float_factory(),
        user: UserValues = UserValues(),
    ):
        """Initialize AcqInfo."""
        self.idx = idx
        self.acquisition_time_stamp = acquisition_time_stamp
        self.flags = flags
        self.orientation = orientation
        self.patient_table_position = patient_table_position
        self.physiology_time_stamps = physiology_time_stamps
        self.position = position
        self.sample_time_us = sample_time_us
        self.user = user

    def tree_flatten(self) -> tuple[tuple[Array, ...], dict[str, Any]]:
        """Flatten the tree structure for JAX."""
        children = (
            self.idx,
            self.acquisition_time_stamp,
            self.flags,
            self.orientation,
            self.patient_table_position,
            self.physiology_time_stamps,
            self.position,
            self.sample_time_us,
            self.user,
        )
        return children, {}

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Array, ...]) -> Self:
        """Unflatten the tree structure for JAX."""
        (
            idx,
            acquisition_time_stamp,
            flags,
            orientation,
            patient_table_position,
            physiology_time_stamps,
            position,
            sample_time_us,
            user,
        ) = children
        return cls(
            idx=idx,
            acquisition_time_stamp=acquisition_time_stamp,
            flags=flags,
            orientation=orientation,
            patient_table_position=patient_table_position,
            physiology_time_stamps=physiology_time_stamps,
            position=position,
            sample_time_us=sample_time_us,
            user=user,
        )

    @overload
    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: Sequence[ismrmrd.acquisition.Acquisition],
        *,
        additional_fields: None,
        convert_time_stamp: _convert_time_stamp_type = convert_time_stamp_siemens,
    ) -> Self: ...

    @overload
    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: Sequence[ismrmrd.acquisition.Acquisition],
        *,
        additional_fields: Sequence[str],
        convert_time_stamp: _convert_time_stamp_type = convert_time_stamp_siemens,
    ) -> tuple[Self, tuple[Float[Array, '*other coils k2 k1 k0'], ...]]: ...

    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: Sequence[ismrmrd.acquisition.Acquisition],
        *,
        additional_fields: Sequence[str] | None = None,
        convert_time_stamp: _convert_time_stamp_type = convert_time_stamp_siemens,
    ) -> Self | tuple[Self, tuple[Float[Array, '*other coils k2 k1 k0'], ...]]:
        """Read the header of a list of acquisition and store information.

        Parameters
        ----------
        acquisitions : Sequence[ismrmrd.acquisition.Acquisition]
            list of ismrmrd acquisistions to read from. Needs at least one acquisition.
        additional_fields : Sequence[str] | None
            if supplied, additional information from fields with these names will be extracted from the
            ismrmrd acquisitions and returned as tensors.
        convert_time_stamp : _convert_time_stamp_type
            function used to convert the raw time stamps to seconds.

        Returns
        -------
        Self | tuple[Self, tuple[Float[Array, '*other coils k2 k1 k0'], ...]]
            AcqInfo instance and optionally additional tensors.
        """
        if len(acquisitions) == 0:
            raise ValueError('Acquisition list must not be empty.')

        # Creating the dtype first and casting to bytes
        # is a workaround for a bug in cpython causing a warning
        # if np.array(AcquisitionHeader) is called directly.
        # also, this needs to check the dtype only once.
        acquisition_head_dtype = np.dtype(ismrmrd.AcquisitionHeader)
        headers = np.frombuffer(
            np.array([memoryview(a._head).cast('B') for a in acquisitions]),
            dtype=acquisition_head_dtype,
        )

        idx_data = headers['idx']

        def tensor(data: np.ndarray) -> Float[Array, '*other coils k2 k1 k0']:
            # we have to convert first as jax cant create arrays from np.uint16 arrays
            # we use int32 for uint16 and int64 for uint32 to fit largest values.
            match data.dtype:
                case np.uint16:
                    return jnp.array(data, dtype=jnp.int32)
                case np.uint32:
                    return jnp.array(data, dtype=jnp.int64)
                case _:
                    return jnp.array(data)

        def tensor_5d(data: np.ndarray) -> Float[Array, '*other coils k2 k1 k0']:
            # Convert tensor to jax dtypes and ensure it is 5D
            return unsqueeze_right(unsqueeze_at(tensor(data), 0, 4), 0, 4)

        def spatialdimension_5d(data: np.ndarray) -> SpatialDimension[Float[Array, '*other coils k2 k1 k0']]:
            # Convert tensor to jax dtypes and ensure it is 5D
            return SpatialDimension(
                tensor_5d(data['x']),
                tensor_5d(data['y']),
                tensor_5d(data['z']),
            )

        # Extract indices
        idx = AcqIdx(
            k1=tensor_5d(idx_data['kspace_encode_step_1']),
            k2=tensor_5d(idx_data['kspace_encode_step_2']),
            average=tensor_5d(idx_data['average']),
            slice=tensor_5d(idx_data['slice']),
            contrast=tensor_5d(idx_data['contrast']),
            phase=tensor_5d(idx_data['phase']),
            repetition=tensor_5d(idx_data['repetition']),
            set=tensor_5d(idx_data['set']),
            segment=tensor_5d(idx_data['segment']),
            user0=tensor_5d(idx_data['user'][:, 0]),
            user1=tensor_5d(idx_data['user'][:, 1]),
            user2=tensor_5d(idx_data['user'][:, 2]),
            user3=tensor_5d(idx_data['user'][:, 3]),
            user4=tensor_5d(idx_data['user'][:, 4]),
            user5=tensor_5d(idx_data['user'][:, 5]),
            user6=tensor_5d(idx_data['user'][:, 6]),
            user7=tensor_5d(idx_data['user'][:, 7]),
        )

        # Extract time stamps
        acquisition_time_stamp = tensor_5d(headers['acquisition_time_stamp'])
        acquisition_time_stamp = convert_time_stamp(acquisition_time_stamp, 'acquisition_time_stamp')

        # Extract flags
        flags = tensor_5d(headers['flags'])

        # Extract orientation
        orientation = Rotation.from_ismrmrd(headers['read_dir'], headers['phase_dir'], headers['slice_dir'])

        # Extract patient table position
        patient_table_position = spatialdimension_5d(headers['patient_table_position'])

        # Extract physiology time stamps
        physiology_time_stamps = PhysiologyTimestamps(
            timestamp0=tensor_5d(headers['physiology_time_stamp'][:, 0]),
            timestamp1=tensor_5d(headers['physiology_time_stamp'][:, 1]),
            timestamp2=tensor_5d(headers['physiology_time_stamp'][:, 2]),
        )

        # Extract position
        position = spatialdimension_5d(headers['position'])

        # Extract sample time
        sample_time_us = tensor_5d(headers['sample_time_us'])

        # Extract user values
        user = UserValues(
            float0=tensor_5d(headers['user_float'][:, 0]),
            float1=tensor_5d(headers['user_float'][:, 1]),
            float2=tensor_5d(headers['user_float'][:, 2]),
            float3=tensor_5d(headers['user_float'][:, 3]),
            float4=tensor_5d(headers['user_float'][:, 4]),
            float5=tensor_5d(headers['user_float'][:, 5]),
            float6=tensor_5d(headers['user_float'][:, 6]),
            float7=tensor_5d(headers['user_float'][:, 7]),
            int0=tensor_5d(headers['user_int'][:, 0]),
            int1=tensor_5d(headers['user_int'][:, 1]),
            int2=tensor_5d(headers['user_int'][:, 2]),
            int3=tensor_5d(headers['user_int'][:, 3]),
            int4=tensor_5d(headers['user_int'][:, 4]),
            int5=tensor_5d(headers['user_int'][:, 5]),
            int6=tensor_5d(headers['user_int'][:, 6]),
            int7=tensor_5d(headers['user_int'][:, 7]),
        )

        # Create AcqInfo instance
        acq_info = cls(
            idx=idx,
            acquisition_time_stamp=acquisition_time_stamp,
            flags=flags,
            orientation=orientation,
            patient_table_position=patient_table_position,
            physiology_time_stamps=physiology_time_stamps,
            position=position,
            sample_time_us=sample_time_us,
            user=user,
        )

        # Extract additional fields if requested
        if additional_fields is not None:
            additional_tensors = []
            for field_name in additional_fields:
                if field_name not in headers.dtype.names:
                    raise ValueError(f'Field {field_name} not found in acquisition header.')
                additional_tensors.append(tensor_5d(headers[field_name]))
            return acq_info, tuple(additional_tensors)

            return acq_info


# Register classes as pytrees
jax.tree_util.register_pytree_node_class(AcqIdx)
jax.tree_util.register_pytree_node_class(UserValues)
jax.tree_util.register_pytree_node_class(PhysiologyTimestamps)
jax.tree_util.register_pytree_node_class(AcqInfo)
