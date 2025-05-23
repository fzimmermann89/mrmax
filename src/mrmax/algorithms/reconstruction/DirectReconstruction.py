"""Direct Reconstruction by Adjoint Fourier Transform."""

from collections.abc import Callable

from mrmax.algorithms.reconstruction.Reconstruction import Reconstruction
from mrmax.data.CsmData import CsmData
from mrmax.data.DcfData import DcfData
from mrmax.data.IData import IData
from mrmax.data.KData import KData
from mrmax.data.KNoise import KNoise
from mrmax.operators.FourierOp import FourierOp
from mrmax.operators.LinearOperator import LinearOperator


class DirectReconstruction(Reconstruction):
    """Direct Reconstruction by Adjoint Fourier Transform."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable[[IData], CsmData] | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
    ):
        """Initialize DirectReconstruction.

        A direct reconstruction uses the adjoint of the acquisition operator and a
        density compensation to obtain the complex valued images from k-space data.

        If csm is not set to `None`, a single coil combined image will reconstructed.
        The method for estimating sensitivity maps can be adjusted using the `csm` argument.

        Parameters
        ----------
        kdata
            If `kdata` is provided and `fourier_op` or `dcf` are `None`, then `fourier_op` and `dcf` are estimated
            based on `kdata`. Otherwise `fourier_op` and `dcf` are used as provided.
        fourier_op
            Instance of the `~mrmax.operators.FourierOp` used for reconstruction.
            If `None`, set up based on `kdata`.
        csm
            Sensitivity maps for coil combination. If `None`, no coil combination is carried out, i.e. images for each
            coil are returned. If a `Callable` is provided, coil images are reconstructed using the adjoint of the
            `~mrmax.operators.FourierOp` (including density compensation) and then sensitivity maps are calculated
            using the callable. For this, `kdata` needs also to be provided.
            For examples have a look at the `~mrmax.data.CsmData` class e.g. `~mrmax.data.CsmData.from_idata_walsh`
            or `~mrmax.data.CsmData.from_idata_inati`.
        noise
            Noise used for prewhitening. If `None`, no prewhitening is performed
        dcf
            K-space sampling density compensation. If `None`, set up based on `kdata`.

        Raises
        ------
        `ValueError`
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is None.
        """
        super().__init__()
        if fourier_op is None:
            if kdata is None:
                raise ValueError('Either kdata or fourier_op needs to be defined.')
            self.fourier_op = FourierOp.from_kdata(kdata)
        else:
            self.fourier_op = fourier_op

        if kdata is not None and dcf is None:
            self.dcf = DcfData.from_traj_voronoi(kdata.traj)
        else:
            self.dcf = dcf

        self.noise = noise

        if csm is None or isinstance(csm, CsmData):
            self.csm = csm
        else:
            if kdata is None:
                raise ValueError('kdata needs to be defined to calculate the sensitivity maps.')
            self.recalculate_csm(kdata, csm)

    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction.

        Parameters
        ----------
        kdata
            k-space data to reconstruct.

        Returns
        -------
            the reconstruced image.
        """
        return self.direct_reconstruction(kdata)
