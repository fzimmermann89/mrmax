"""KTrajectory class for handling k-space trajectories using JAX."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import ismrmrd
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int
from typing_extensions import Self

from mrmax.data.enums import TrajType
from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.utils.reduce_repeat import reduce_repeat
from mrmax.utils.reshape import unsqueeze_at
from mrmax.utils.typing import FileOrPath


@register_pytree_node_class
@dataclass(frozen=True)
class KTrajectory:
    """K-space trajectory.

    Contains the trajectory in k-space along the three dimensions `kz`, `ky`, `kx`,
    i.e. describes where in k-space each data point was acquired.

    The shape of each of `kx`, `ky`, `kz` is `(*other, coils=1, k2, k1, k0)`,
    where `other` can span multiple dimensions.

    Example for 2D-Cartesian trajectories:

        - `kx` changes along `k0` and is frequency encoding,
        - `ky` changes along `k1` and is phase encoding
        - `kz` is zero with shape `(1, 1, 1, 1, 1)`
    """

    kz: jnp.ndarray
    """Trajectory in z direction / phase encoding direction k2 if Cartesian."""
    ky: jnp.ndarray
    """Trajectory in y direction / phase encoding direction k1 if Cartesian."""
    kx: jnp.ndarray
    """Trajectory in x direction / phase encoding direction k0 if Cartesian."""
    grid_detection_tolerance: float = 1e-3
    """Tolerance of how close trajectory positions have to be to integer grid points."""
    repeat_detection_tolerance: float | None = 1e-3
    """Tolerance for repeat detection. Set to None to disable."""

    def __post_init__(self) -> None:
        """Reduce repeated dimensions to singletons."""

        def as_any_float(array: jnp.ndarray) -> jnp.ndarray:
            return array.astype(jnp.float32) if not jnp.issubdtype(array.dtype, jnp.floating) else array

        if self.repeat_detection_tolerance is not None:
            kz, ky, kx = (
                as_any_float(reduce_repeat(array, self.repeat_detection_tolerance))
                for array in (self.kz, self.ky, self.kx)
            )
            # Use object.__setattr__ since we're using a frozen dataclass
            object.__setattr__(self, 'kz', kz)
            object.__setattr__(self, 'ky', ky)
            object.__setattr__(self, 'kx', kx)

        try:
            shape = self.broadcasted_shape
        except ValueError:
            raise ValueError('The k-space trajectory dimensions must be broadcastable.') from None

        if len(shape) < 5:
            raise ValueError('The k-space trajectory arrays should each have at least 5 dimensions.')

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], dict[str, Any]]:
        """Flatten the tree structure for JAX."""
        children = (self.kz, self.ky, self.kx)
        aux_data = {
            'grid_detection_tolerance': self.grid_detection_tolerance,
            'repeat_detection_tolerance': self.repeat_detection_tolerance,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[jnp.ndarray, ...]) -> Self:
        """Unflatten the tree structure for JAX."""
        kz, ky, kx = children
        return cls(
            kz=kz,
            ky=ky,
            kx=kx,
            grid_detection_tolerance=aux_data['grid_detection_tolerance'],
            repeat_detection_tolerance=aux_data['repeat_detection_tolerance'],
        )

    @classmethod
    def from_array(
        cls,
        array: Float[Array, '*other 1 k2 k1 k0 3'],
        stack_dim: int = 0,
        axes_order: Literal['zxy', 'zyx', 'yxz', 'yzx', 'xyz', 'xzy'] = 'zyx',
        repeat_detection_tolerance: float | None = 1e-6,
        grid_detection_tolerance: float = 1e-3,
        scaling_matrix: SpatialDimension | None = None,
    ) -> Self:
        """Create a KTrajectory from an array representation of the trajectory.

        Reduces repeated dimensions to singletons if repeat_detection_tolerance is not set to `None`.

        Parameters
        ----------
        array
            The array representation of the trajectory.
            This should be a 5-dim array, with (`kz`, `ky`, `kx`) stacked in this order along `stack_dim`.
        stack_dim
            The dimension in the array along which the directions are stacked.
        axes_order
            The order of the axes in the array. The mrmax convention is 'zyx'.
        repeat_detection_tolerance
            Tolerance for detecting repeated dimensions (broadcasting).
            If trajectory points differ by less than this value, they are considered identical.
            Set to None to disable this feature.
        grid_detection_tolerance
            Tolerance for detecting whether trajectory points align with integer grid positions.
            This tolerance is applied after rescaling if `scaling_matrix` is provided.
        scaling_matrix
            If a scaling matrix is provided, the trajectory is rescaled to fit within
            the dimensions of the matrix. If not provided, the trajectory remains unchanged.
        """
        ks = jnp.split(array, 3, axis=stack_dim)
        kz, ky, kx = (ks[axes_order.index(axis)] for axis in 'zyx')

        def rescale(k: Float[Array, '*'], size: float) -> Float[Array, '*']:
            max_abs_range = 2 * jnp.abs(k).max()
            return jnp.where(
                jnp.logical_or(size < 2, max_abs_range < 1e-6),
                jnp.zeros_like(k),  # a single encoding point should be at zero
                k * (size / max_abs_range),
            )

        if scaling_matrix is not None:
            kz = rescale(kz, scaling_matrix.z)
            ky = rescale(ky, scaling_matrix.y)
            kx = rescale(kx, scaling_matrix.x)

        return cls(
            kz=kz,
            ky=ky,
            kx=kx,
            repeat_detection_tolerance=repeat_detection_tolerance,
            grid_detection_tolerance=grid_detection_tolerance,
        )

    @classmethod
    def from_ismrmrd(
        cls,
        filename: FileOrPath,
        dataset_idx: int = -1,
        acquisition_filter_criterion: Callable[[ismrmrd.Acquisition], bool] | None = None,
        normalize: bool = False,
    ) -> Self:
        """Read a k-space trajectory from an ISMRMRD file.

        The trajectory is extracted from an ISMRMRD file. If the encoding matrix is set in the header,
        the trajectory is rescaled to fit within the dimensions of the matrix.

        Parameters
        ----------
        filename
            The path to the ISMRMRD file.
        dataset_idx
            The index of the dataset in the file.
        acquisition_filter_criterion
            A function that takes an ISMRMRD acquisition and returns a boolean.
            If provided, only acquisitions for which the function returns `True` are included in the trajectory.
            `None` includes all acquisitions.
        normalize
            Normalize the trajectory to the encoding matrix. Consider enabling for non-cartesian trajectories or
            non-normalized ISMRMRD trajectories.
        """
        with ismrmrd.File(filename, 'r') as file:
            datasets = list(file.keys())
            if not datasets:
                raise ValueError('No datasets found in the ISMRMRD file.')
            if not -len(datasets) <= dataset_idx < len(datasets):
                raise ValueError(f'Dataset index {dataset_idx} out of range, available datasets: {datasets}')
            dataset = file[datasets[dataset_idx]]

            acquisitions = [
                acq
                for acq in dataset.acquisitions
                if acquisition_filter_criterion is None or acquisition_filter_criterion(acq)
            ]
            if not acquisitions:
                raise ValueError('No matching acquisitions found in the ISMRMRD file.')
            traj = jnp.stack([jnp.array(acq.traj, dtype=jnp.float32) for acq in acquisitions], axis=0)
            if not normalize:
                scaling_matrix = None
            elif (
                dataset.header.encoding
                and dataset.header.encoding[0].encodedSpace
                and dataset.header.encoding[0].encodedSpace.matrixSize
            ):
                scaling_matrix = SpatialDimension.from_xyz(dataset.header.encoding[0].encodedSpace.matrixSize)
            else:
                raise ValueError(
                    'Requested normalization, but the ISMRMRD file does not contain an encoding matrix size. '
                    'Consider adding it to the header.'
                )

        if traj.shape[-1] != 3:  # enforce 3D trajectory
            zero = jnp.zeros_like(traj[..., :1])
            traj = jnp.concatenate([traj, *([zero] * (3 - traj.shape[-1]))], axis=-1)
        traj = unsqueeze_at(traj, dim=-3, n=5 - traj.ndim + 1)  # +1 due to stack_dim

        return cls.from_array(traj, stack_dim=-1, axes_order='xyz', scaling_matrix=scaling_matrix)

    @property
    def broadcasted_shape(self) -> tuple[int, ...]:
        """The broadcasted shape of the trajectory.

        Returns
        -------
            broadcasted shape of trajectory
        """
        return jax.eval_shape(lambda x, y, z: jnp.broadcast_arrays(x, y, z)[0], self.kx, self.ky, self.kz).shape

    @property
    def type_along_kzyx(self) -> tuple[TrajType, TrajType, TrajType]:
        """Type of trajectory along kz-ky-kx."""
        return self._traj_types(self.grid_detection_tolerance)[0]

    @property
    def type_along_k210(self) -> tuple[TrajType, TrajType, TrajType]:
        """Type of trajectory along k2-k1-k0."""
        return self._traj_types(self.grid_detection_tolerance)[1]

    def _traj_types(
        self,
        tolerance: float,
    ) -> tuple[tuple[TrajType, TrajType, TrajType], tuple[TrajType, TrajType, TrajType]]:
        """Calculate the trajectory type along kzkykx and k2k1k0.

        Checks if the entries of the trajectory along certain dimensions
            - are of shape 1 -> `TrajType.SINGLEVALUE`
            - lie on a Cartesian grid -> `TrajType.ONGRID`

        Parameters
        ----------
        tolerance:
            absolute tolerance in checking if points are on integer grid positions

        Returns
        -------
            (`(types along kz,ky,kx)`,`(types along k2,k1,k0)`)
        """
        # Matrix describing trajectory-type [(kz, ky, kx), (k2, k1, k0)]
        # Start with everything not on a grid (arbitrary k-space locations).
        # We use the value of the enum-type to make it easier to do array operations.
        traj_type_matrix: Int[Array, '3 3'] = jnp.zeros((3, 3), dtype=jnp.int32)
        for ind, ks in enumerate((self.kz, self.ky, self.kx)):
            values_on_grid = not jnp.issubdtype(ks.dtype, jnp.floating) or jnp.all(
                jnp.abs(ks - jnp.round(ks)) <= tolerance
            )
            for dim in (-3, -2, -1):
                if ks.shape[dim] == 1:
                    traj_type_matrix = traj_type_matrix.at[ind, dim].set(
                        traj_type_matrix[ind, dim] | TrajType.SINGLEVALUE.value | TrajType.ONGRID.value
                    )
                if values_on_grid:
                    traj_type_matrix = traj_type_matrix.at[ind, dim].set(
                        traj_type_matrix[ind, dim] | TrajType.ONGRID.value
                    )

        # Explicitly create tuples with exactly 3 elements
        types_zyx: list[TrajType] = [TrajType(traj_type_matrix[i, -3]) for i in range(3)]
        types_210: list[TrajType] = [TrajType(traj_type_matrix[i, -1]) for i in range(3)]

        return ((types_zyx[0], types_zyx[1], types_zyx[2]), (types_210[0], types_210[1], types_210[2]))

    def as_array(self, stack_dim: int = 0) -> Float[Array, '*other 1 k2 k1 k0 3']:
        """Stack the trajectory components into a single array.

        Parameters
        ----------
        stack_dim
            The dimension along which to stack the components.

        Returns
        -------
            Array with shape `(*other, coils=1, k2, k1, k0, 3)` if `stack_dim=-1`
        """
        return jnp.stack((self.kz, self.ky, self.kx), axis=stack_dim)

    def __repr__(self) -> str:
        """Return string representation of KTrajectory."""
        return (
            f'{self.__class__.__name__}(\n'
            f'  kz shape: {self.kz.shape}\n'
            f'  ky shape: {self.ky.shape}\n'
            f'  kx shape: {self.kx.shape}\n'
            f'  grid_detection_tolerance: {self.grid_detection_tolerance}\n'
            f'  repeat_detection_tolerance: {self.repeat_detection_tolerance}\n'
            ')'
        )

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
        kz = jax.device_put(self.kz, device) if device is not None else self.kz
        ky = jax.device_put(self.ky, device) if device is not None else self.ky
        kx = jax.device_put(self.kx, device) if device is not None else self.kx

        kz = kz.astype(dtype) if dtype is not None else kz
        ky = ky.astype(dtype) if dtype is not None else ky
        kx = kx.astype(dtype) if dtype is not None else kx

        return self.__class__(
            kz=kz,
            ky=ky,
            kx=kx,
            grid_detection_tolerance=self.grid_detection_tolerance,
            repeat_detection_tolerance=self.repeat_detection_tolerance,
        )
