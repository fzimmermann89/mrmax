"""Slice Projection Operator."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float

from mrmax.data import Rotation, SpatialDimension
from mrmax.operators.LinearOperator import LinearOperator
from mrmax.utils.sparse import SparseMatrix, ravel_multi_index, sparse_coo_matrix

TensorFunction: TypeAlias = Callable[[Array], Array]


class SliceProjectionOp(LinearOperator, eqx.Module):
    """Slice Projection Operator.

    This operation samples from a 3D Volume a slice with a given rotation and shift
    (relative to the center of the volume) according to the `slice_profile`.
    It can, for example, be used to describe the slice selection of a 2D MRI sequence
    from the 3D Volume.

    The projection will be done by sparse matrix multiplication.

    `slice_rotation`, `slice_shift`, and `slice_profile` can have (multiple) batch dimensions. These dimensions will
    be broadcasted to a common shape and added to the front of the volume.
    Different settings for different volume batches are NOT supported, consider creating multiple
    operators if required.
    """

    _slice_profile: Float[Array, ...]
    _slice_rotation: Float[Array, '3 3']
    _slice_shift: Float[Array, 3]
    _range_shape: tuple[int, ...]
    _domain_shape: tuple[int, ...]
    _matrix_op: SparseMatrix | None = None

    def __init__(
        self,
        input_shape: tuple[int, ...],
        slice_rotation: Float[Array, '3 3'],
        slice_shift: Float[Array, 3],
        slice_profile: Float[Array, ...] | None = None,
    ) -> None:
        """Initialize the slice projection operator.

        Parameters
        ----------
        input_shape
            Shape of the input volume (Z, Y, X)
        slice_rotation
            Rotation matrix for the slice (3, 3)
        slice_shift
            Translation vector for the slice (3,)
        slice_profile
            Profile of the slice (optional)
        """
        super().__init__()

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError('Input shape must be 3D')

        # Store slice parameters
        self._slice_rotation = slice_rotation
        self._slice_shift = slice_shift
        self._domain_shape = tuple(map(int, input_shape))

        # Initialize slice profile
        if slice_profile is None:
            self._slice_profile = jnp.ones((1,), dtype=jnp.float32)
        else:
            self._slice_profile = slice_profile

        # Calculate range shape
        self._range_shape = self._calculate_range_shape()

    def _calculate_range_shape(self) -> tuple[int, ...]:
        """Calculate the shape of the output slice."""
        # Get dimensions
        nz, ny, nx = self._domain_shape

        # Calculate slice dimensions based on input volume
        slice_height = int(jnp.sqrt(ny * ny + nz * nz))
        slice_width = int(jnp.sqrt(nx * nx + nz * nz))

        return (slice_height, slice_width)

    def _build_projection_matrix(self) -> SparseMatrix:
        """Build the projection matrix for the slice."""
        # Get dimensions
        nz, ny, nx = self._domain_shape
        nh, nw = self._range_shape

        # Create meshgrid for slice coordinates
        y, x = jnp.meshgrid(
            jnp.linspace(-1, 1, nh),
            jnp.linspace(-1, 1, nw),
            indexing='ij',
        )

        # Apply rotation and translation
        coords = jnp.stack([jnp.zeros_like(y), y, x])
        coords = jnp.einsum('ij,jkl->ikl', self._slice_rotation, coords)
        coords = coords + self._slice_shift[:, None, None]

        # Convert to volume indices
        coords = (coords + 1) * jnp.array([nz - 1, ny - 1, nx - 1])[:, None, None] / 2

        # Round to nearest integer and clip to valid range
        coords = jnp.clip(
            jnp.round(coords),
            jnp.zeros((3,), dtype=jnp.int32),
            jnp.array([nz - 1, ny - 1, nx - 1], dtype=jnp.int32)[:, None, None],
        ).astype(jnp.int32)

        # Create sparse matrix indices and values
        slice_indices = jnp.stack(
            [
                ravel_multi_index(
                    (
                        coords[0],
                        coords[1],
                        coords[2],
                    ),
                    (nz, ny, nx),
                ),
                ravel_multi_index(
                    (
                        jnp.arange(nh, dtype=jnp.int32)[:, None],
                        jnp.arange(nw, dtype=jnp.int32)[None, :],
                    ),
                    (nh, nw),
                ),
            ]
        )

        # Create sparse matrix
        return sparse_coo_matrix(
            slice_indices,
            jnp.ones(slice_indices.shape[1], dtype=jnp.float32),
            (nz * ny * nx, nh * nw),
        )

    def forward(self, x: Float[Array, ...]) -> tuple[Float[Array, ...]]:
        """Apply the forward operator.

        Parameters
        ----------
        x
            Input volume (Z, Y, X)

        Returns
        -------
        tuple[Array]
            Output slice (H, W)
        """
        # Build projection matrix if needed
        if self._matrix_op is None:
            self._matrix_op = self._build_projection_matrix()

        # Reshape input
        x_flat = x.reshape(-1)

        # Apply projection
        y_flat = self._matrix_op @ x_flat

        # Reshape output
        return (y_flat.reshape(self._range_shape),)

    def adjoint(self, x: Float[Array, ...]) -> tuple[Float[Array, ...]]:
        """Apply the adjoint operator.

        Parameters
        ----------
        x
            Input slice (H, W)

        Returns
        -------
        tuple[Array]
            Output volume (Z, Y, X)
        """
        # Build projection matrix if needed
        if self._matrix_op is None:
            self._matrix_op = self._build_projection_matrix()

        # Reshape input
        x_flat = x.reshape(-1)

        # Apply adjoint projection
        y_flat = self._matrix_op.H @ x_flat

        # Reshape output
        return (y_flat.reshape(self._domain_shape),)

    @property
    def H(self) -> SliceProjectionOp:  # noqa: N802
        """Return the adjoint operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        return f'SliceProjectionOp(domain_shape={self._domain_shape}, range_shape={self._range_shape})'

    @staticmethod
    def join_matrices(matrices: Sequence[Array]) -> Array:
        """Join multiple sparse matrices into a block diagonal matrix.

        Parameters
        ----------
        matrices
            List of sparse matrices to join by stacking them as a block diagonal matrix
        """
        if not matrices:
            raise ValueError('Cannot join empty sequence of matrices')

        first_matrix = matrices[0]
        values = []
        target = []
        source = []
        for i, m in enumerate(matrices):
            if not m.shape == first_matrix.shape:
                raise ValueError('all matrices should have the same shape')
            c = m.coalesce()  # we want unique indices
            (ctarget, csource) = c.indices()
            values.append(c.values())
            source.append(csource)
            ctarget = ctarget + i * m.shape[0]
            target.append(ctarget)

        with warnings.catch_warnings():
            # beta status in pytorch causes a warning to be printed
            warnings.filterwarnings('ignore', category=UserWarning, message='Sparse')
            matrix = sparse_coo_matrix(
                indices=jnp.stack([jnp.concatenate(target), jnp.concatenate(source)]),
                values=jnp.concatenate(values),
                shape=(len(matrices) * first_matrix.shape[0], first_matrix.shape[1]),
                dtype=jnp.float32,
            )
        return matrix

    @staticmethod
    def projection_matrix(
        input_shape: SpatialDimension,
        output_shape: SpatialDimension,
        rotation: Rotation,
        offset: Array,
        w: int,
        slice_function: TensorFunction,
        rotation_center: Array | None = None,
    ) -> Array:
        """Create a sparse matrix that represents the projection of a volume onto a plane.

        Outside the volume values are approximately zero padded

        Parameters
        ----------
        input_shape
            Shape of the volume to sample from
        output_shape
            Shape of the resulting plane, 2D. Only the x and y values are used.
        rotation
            Rotation that describes the orientation of the plane
        offset: Array
            Shift of the plane from the center of the volume in the rotated coordinate system
            in units of the 3D volume, order `z, y, x`
        w: int
            Factor that determines the number of pixels that are considered in the projection along
            the slice profile direction.
        slice_function
            Function that describes the slice profile. See `mrmax.utils.slice_profiles` for examples.
        rotation_center
            Center of rotation, if None the center of the volume is used,
            i.e. for 4 pixels 0 1 2 3 it is between 0 and 1

        Returns
        -------
        jnp.sparse_coo_matrix
            Sparse matrix that represents the projection of the volume onto the plane
        """
        x, y = output_shape.x, output_shape.y

        start_x, start_y = (
            (input_shape.x - x) // 2,
            (input_shape.y - y) // 2,
        )
        pixel_coord_y_x_zyx = jnp.stack(
            [
                (input_shape.z / 2 - 0.5) * jnp.ones((y, x)),  # z coordinates
                *jnp.meshgrid(
                    jnp.arange(start_y, start_y + y),  # y coordinates
                    jnp.arange(start_x, start_x + x),  # x coordinates
                    indexing='ij',
                ),
            ],
            axis=-1,
        )  # coordinates of the 2d output pixels in the coordinate system of the input volume, so shape (y,x,3)
        if offset is not None:
            pixel_coord_y_x_zyx = pixel_coord_y_x_zyx + offset
        if rotation_center is None:
            # default rotation center is the center of the volume, i.e. for 4 pixels
            # 0 1 2 3 it is between 0 and 1
            rotation_center = jnp.array([input_shape.z / 2 - 0.5, input_shape.y / 2 - 0.5, input_shape.x / 2 - 0.5])
        pixel_rotated_y_x_zyx = rotation(pixel_coord_y_x_zyx - rotation_center) + rotation_center

        # We cast a ray from the pixel normal to the plane in both directions
        # points in the original volume further away then w will not be considered
        ray = rotation(
            jnp.stack(
                [
                    jnp.arange(-w, w + 1),  # z
                    jnp.zeros(2 * w + 1),  # y
                    jnp.zeros(2 * w + 1),  # x
                ],
                axis=-1,
            )
        )
        # In all possible directions for each point along the line we consider the eight neighboring points
        # by adding all possible combinations of 0 and 1 to the point and flooring
        offsets = jnp.array(list(itertools.product([0, 1], repeat=3)))
        # all points that influence a pixel
        # x,y,8-neighbors,(2*w+1)-raylength,3-dimensions input_shape.xinput_shape.yinput_shape.z)
        points_influencing_pixel = (
            rearrange(pixel_rotated_y_x_zyx, '   y x zyxdim -> y x 1          1   zyxdim')
            + rearrange(ray, '                   ray zyxdim -> 1 1 1          ray zyxdim')
            + rearrange(offsets, '        neighbors zyxdim -> 1 1 neighbors 1   zyxdim')
        ).floor()  # y x neighbors ray zyx
        # directional distance in source volume coordinate system
        distance = pixel_rotated_y_x_zyx[:, :, None, None, :] - points_influencing_pixel
        # Inverse rotation projects this back to the original coordinate system, i.e
        # Distance in z is distance along the line, i.e. the slice profile weighted direction
        # Distance in x and y is the distance of a pixel to the ray and linear interpolation
        # is used to weight the distance
        distance_z, distance_y, distance_x = rotation(distance, inverse=True).unbind(-1)
        weight_yx = (1 - jnp.abs(distance_y)).clip(0) * (1 - jnp.abs(distance_x)).clip(0)
        weight_z = slice_function(distance_z)
        weight = (weight_yx * weight_z).reshape(y * x, -1)

        source = rearrange(
            points_influencing_pixel,
            'y x neighbors raylength zyxdim -> (y x) (neighbors raylength) zyxdim',
        ).astype(jnp.int32)

        # mask of only potential source points inside the source volume
        mask = (
            (source[..., 0] < input_shape.z)
            & (source[..., 0] >= 0)
            & (source[..., 1] < input_shape.y)
            & (source[..., 1] >= 0)
            & (source[..., 2] < input_shape.x)
            & (source[..., 2] >= 0)
        )

        # We need this at the edge of the volume to approximate zero padding
        fraction_in_view = (mask * (weight > 0)).sum(-1) / (weight > 0).sum(-1)

        source_index = ravel_multi_index(source[mask].unbind(-1), (input_shape.z, input_shape.y, input_shape.x))

        target_index = jnp.repeat(jnp.arange(y * x), mask.sum(-1))

        with warnings.catch_warnings():
            # beta status in pytorch causes a warning to be printed
            warnings.filterwarnings('ignore', category=UserWarning, message='Sparse')

            matrix = sparse_coo_matrix(
                indices=jnp.stack((target_index, source_index)),
                values=weight.reshape(y * x, -1)[mask],
                shape=(y * x, input_shape.z * input_shape.y * input_shape.x),
                dtype=jnp.float32,
            ).coalesce()

            # To avoid giving more weight to points that are duplicated in our ray
            # logic and got summed in the coalesce operation, we normalize by the number
            # of duplicates. This is equivalent to the sum of the weights of the duplicates.
            # Count duplicates...

            ones = jnp.ones_like(source_index).astype(jnp.float32)
            ones = sparse_coo_matrix(
                indices=jnp.stack((target_index, source_index)),
                values=ones,
                shape=(y * x, input_shape.z * input_shape.y * input_shape.x),
                dtype=jnp.float32,
            )
            # Coalesce sums the values of duplicate indices
            ones = ones.coalesce()

        # .. and normalize by the number of duplicates
        matrix.values()[:] /= ones.values()

        # Normalize for slice profile, so that the sum of the weights is 1
        # independent of the number of points that are considered.
        # Within the volume, the column sum should be 1,
        # at the edge of the volume, the column sum should be the fraction of the slice
        # that is in view to approximate zero padding
        norm = fraction_in_view / (matrix.sum(1).to_dense() + 1e-6)
        matrix *= norm[:, None]
        return matrix
