"""Fourier Operator."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
from jaxtyping import Array
from typing_extensions import Self

from mrmax.data.KData import KData
from mrmax.data.KTrajectory import KTrajectory
from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.operators.FastFourierOp import FastFourierOp
from mrmax.operators.LinearOperator import LinearOperator
from mrmax.operators.NonUniformFastFourierOp import NonUniformFastFourierOp


class FourierOp(LinearOperator, eqx.Module):
    """Fourier Operator class.

    This is the recommended operator for Fourier transformations. It automatically detects if a non-uniform or
    regular fast Fourier transformation is required and can also be constructed automatically from a
    `mrmax.data.KData` object.

    ```{note}
    The operator is scaled such that it matches 'orthonormal' FFT scaling for cartesian trajectories.
    This is different from other packages, which apply scaling based on the size of the oversampled grid.
    ```
    """

    _recon_matrix: SpatialDimension | Sequence[int]
    _encoding_matrix: SpatialDimension | Sequence[int]
    _traj: KTrajectory | None
    _op: FastFourierOp | NonUniformFastFourierOp
    oversampling: float

    def __init__(
        self,
        recon_matrix: SpatialDimension | Sequence[int],
        encoding_matrix: SpatialDimension | Sequence[int],
        traj: KTrajectory | None = None,
        oversampling: float = 2.0,
    ) -> None:
        """Initialize Fourier Operator.

        Parameters
        ----------
        recon_matrix
            Dimension of the reconstructed image. If this is `~mrmax.data.SpatialDimension` only values of directions
            will be used. Otherwise, it should be a `Sequence` of the same length as direction.
        encoding_matrix
            Dimension of the encoded k-space. If this is `~mrmax.data.SpatialDimension` only values of directions will
            be used. Otherwise, it should be a `Sequence` of the same length as direction.
        traj
            The k-space trajectories where the frequencies are sampled. If None, a regular FFT is used.
        oversampling
            Oversampling used for interpolation in non-uniform FFTs.
            On GPU, 2.0 uses an optimized kernel, any value > 1.0 will work.
            On CPU, there are kernels for 2.0 and 1.25. The latter saves memory. Set to 0.0 for automatic selection.
        """
        super().__init__()
        self._recon_matrix = recon_matrix
        self._encoding_matrix = encoding_matrix
        self._traj = traj
        self.oversampling = oversampling

        if traj is None:
            self._op = FastFourierOp(
                dim=(-3, -2, -1),
                encoding_matrix=encoding_matrix,
                recon_matrix=recon_matrix,
            )
        else:
            self._op = NonUniformFastFourierOp(
                direction=('z', 'y', 'x'),
                recon_matrix=recon_matrix,
                encoding_matrix=encoding_matrix,
                traj=traj,
                oversampling=oversampling,
            )

    @classmethod
    def from_kdata(cls, kdata: KData, oversampling: float = 2.0) -> Self:
        """Create a Fourier Operator from a KData object.

        Parameters
        ----------
        kdata
            The KData object to create the operator from.
        oversampling
            Oversampling used for interpolation in non-uniform FFTs.
            On GPU, 2.0 uses an optimized kernel, any value > 1.0 will work.
            On CPU, there are kernels for 2.0 and 1.25. The latter saves memory. Set to 0.0 for automatic selection.

        Returns
        -------
            A Fourier Operator.
        """
        return cls(
            recon_matrix=kdata.recon_matrix,
            encoding_matrix=kdata.encoding_matrix,
            traj=kdata.traj,
            oversampling=oversampling,
        )

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, z, y, x)`

        Returns
        -------
            output tensor, shape `(..., coils, k2, k1, k0)`
        """
        return self._op(x)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape `(..., coils, k2, k1, k0)`

        Returns
        -------
            output tensor, shape `(..., coils, z, y, x)`
        """
        return self._op.H(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the Fourier operator."""
        return self

    def __repr__(self) -> str:
        """Representation method for Fourier operator."""
        if self._traj is None:
            return f'{type(self).__name__} (FFT)'
        return f'{type(self).__name__} (NUFFT)'


class FourierGramOp(LinearOperator, eqx.Module):
    """Gram operator for the Fourier operator.

    Implements the adjoint of the forward operator of the Fourier operator, i.e. the gram operator
    `F.H@F`.

    Uses a convolution, implemented as multiplication in Fourier space, to calculate the gram operator
    for the toeplitz NUFFT operator.

    Uses a multiplication with a binary mask in Fourier space to calculate the gram operator for
    the Cartesian FFT operator

    This Operator is only used internally and should not be used directly.
    Instead, consider using the py:func:`~FourierOp.gram` property of py:class:`FourierOp`.
    """

    _kernel: Array | None
    nufft_gram: LinearOperator | None
    fast_fourier_gram: LinearOperator | None

    def __init__(self, fourier_op: FourierOp) -> None:
        """Initialize the gram operator.

        If density compensation weights are provided, they the operator
        F.H@dcf@F is calculated.

        Parameters
        ----------
        fourier_op
            the Fourier operator to calculate the gram operator for

        """
        super().__init__()
        if isinstance(fourier_op._op, NonUniformFastFourierOp):
            self.nufft_gram = fourier_op._op.gram
        else:
            self.nufft_gram = None

        if isinstance(fourier_op._op, FastFourierOp):
            self.fast_fourier_gram = fourier_op._op.gram
        else:
            self.fast_fourier_gram = None

    def forward(self, x: Array) -> tuple[Array]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape: `(..., coils, z, y, x)`
        """
        if self.nufft_gram is not None:
            (x,) = self.nufft_gram(x)

        if self.fast_fourier_gram is not None:
            (x,) = self.fast_fourier_gram(x)
        return (x,)

    def adjoint(self, x: Array) -> tuple[Array]:
        """Apply the adjoint operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape: `(..., coils, z, y, x)`
        """
        return self.forward(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the gram operator."""
        return self

    def __repr__(self) -> str:
        """Return string representation of FourierGramOp."""
        return f'{type(self).__name__}'
