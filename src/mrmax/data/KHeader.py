"""MR raw data / k-space data header dataclass."""

from __future__ import annotations

import datetime
from typing import Any

import equinox as eqx
import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema
import jax.numpy as jnp
from jaxtyping import Array, Int
from typing_extensions import Self

from mrmax.data import enums
from mrmax.data.AcqInfo import AcqInfo
from mrmax.data.SpatialDimension import SpatialDimension
from mrmax.utils.summarize_tensorvalues import summarize_tensorvalues
from mrmax.utils.unit_conversion import deg_to_rad, ms_to_s

UNKNOWN = 'unknown'


class KHeader(eqx.Module):
    """MR raw data header.

    All information that is not covered by the dataclass is stored in
    the misc dict. Our code shall not rely on this information, and it is
    not guaranteed to be present. Also, the information in the misc dict
    is not guaranteed to be correct or tested.

    Attributes
    ----------
    recon_matrix : SpatialDimension
        Dimensions of the reconstruction matrix.
    encoding_matrix : SpatialDimension
        Dimensions of the encoded k-space matrix.
    recon_fov : SpatialDimension
        Field-of-view of the reconstructed image [m].
    encoding_fov : SpatialDimension
        Field of view of the image encoded by the k-space trajectory [m].
    acq_info : AcqInfo
        Information of the acquisitions (i.e. readout lines).
    trajectory : Optional[KTrajectoryCalculator]
        Function to calculate the k-space trajectory.
    lamor_frequency_proton : Optional[float]
        Lamor frequency of hydrogen nuclei [Hz].
    datetime : Optional[datetime.datetime]
        Date and time of acquisition.
    te : Array
        Echo time [s].
    ti : Array
        Inversion time [s].
    fa : Array
        Flip angle [rad].
    tr : Array
        Repetition time [s].
    echo_spacing : Array
        Echo spacing [s].
    echo_train_length : int
        Number of echoes in a multi-echo acquisition.
    sequence_type : str
        Type of sequence.
    model : str
        Scanner model.
    vendor : str
        Scanner vendor.
    protocol_name : str
        Name of the acquisition protocol.
    trajectory_type : enums.TrajectoryType
        Type of trajectory.
    measurement_id : str
        Measurement ID.
    patient_name : str
        Name of the patient.
    _misc : Dict[str, Any]
        Dictionary with miscellaneous parameters.
    """

    recon_matrix: SpatialDimension
    encoding_matrix: SpatialDimension
    recon_fov: SpatialDimension
    encoding_fov: SpatialDimension
    acq_info: AcqInfo
    trajectory: Any | None
    lamor_frequency_proton: float | None
    datetime: datetime.datetime | None
    te: Array
    ti: Array
    fa: Array
    tr: Array
    echo_spacing: Array
    echo_train_length: Int[Array, '']
    sequence_type: str
    model: str
    vendor: str
    protocol_name: str
    trajectory_type: enums.TrajectoryType
    measurement_id: str
    patient_name: str
    _misc: dict[str, Any]

    def __init__(
        self,
        recon_matrix: SpatialDimension,
        encoding_matrix: SpatialDimension,
        recon_fov: SpatialDimension,
        encoding_fov: SpatialDimension,
        acq_info: AcqInfo | None = None,
        trajectory: Any | None = None,
        lamor_frequency_proton: float | None = None,
        datetime: datetime.datetime | None = None,
        te: Array | None = None,
        ti: Array | None = None,
        fa: Array | None = None,
        tr: Array | None = None,
        echo_spacing: Array | None = None,
        echo_train_length: int = 1,
        sequence_type: str = UNKNOWN,
        model: str = UNKNOWN,
        vendor: str = UNKNOWN,
        protocol_name: str = UNKNOWN,
        trajectory_type: enums.TrajectoryType = enums.TrajectoryType.OTHER,
        measurement_id: str = UNKNOWN,
        patient_name: str = UNKNOWN,
        _misc: dict[str, Any] | None = None,
    ):
        """Initialize KHeader.

        Parameters
        ----------
        recon_matrix : SpatialDimension
            Dimensions of the reconstruction matrix.
        encoding_matrix : SpatialDimension
            Dimensions of the encoded k-space matrix.
        recon_fov : SpatialDimension
            Field-of-view of the reconstructed image [m].
        encoding_fov : SpatialDimension
            Field of view of the image encoded by the k-space trajectory [m].
        acq_info : AcqInfo, optional
            Information of the acquisitions (i.e. readout lines).
        trajectory : Optional[KTrajectoryCalculator], optional
            Function to calculate the k-space trajectory.
        lamor_frequency_proton : Optional[float], optional
            Lamor frequency of hydrogen nuclei [Hz].
        datetime : Optional[datetime.datetime], optional
            Date and time of acquisition.
        te : Array, optional
            Echo time [s].
        ti : Array, optional
            Inversion time [s].
        fa : Array, optional
            Flip angle [rad].
        tr : Array, optional
            Repetition time [s].
        echo_spacing : Array, optional
            Echo spacing [s].
        echo_train_length : int, optional
            Number of echoes in a multi-echo acquisition.
        sequence_type : str, optional
            Type of sequence.
        model : str, optional
            Scanner model.
        vendor : str, optional
            Scanner vendor.
        protocol_name : str, optional
            Name of the acquisition protocol.
        trajectory_type : enums.TrajectoryType, optional
            Type of trajectory.
        measurement_id : str, optional
            Measurement ID.
        patient_name : str, optional
            Name of the patient.
        _misc : Dict[str, Any], optional
            Dictionary with miscellaneous parameters.
        """
        self.recon_matrix = recon_matrix
        self.encoding_matrix = encoding_matrix
        self.recon_fov = recon_fov
        self.encoding_fov = encoding_fov
        self.acq_info = acq_info or AcqInfo()
        self.trajectory = trajectory
        self.lamor_frequency_proton = lamor_frequency_proton
        self.datetime = datetime
        self.te = jnp.array(te) if te is not None else jnp.array([])
        self.ti = jnp.array(ti) if ti is not None else jnp.array([])
        self.fa = jnp.array(fa) if fa is not None else jnp.array([])
        self.tr = jnp.array(tr) if tr is not None else jnp.array([])
        self.echo_spacing = jnp.array(echo_spacing) if echo_spacing is not None else jnp.array([])
        self.echo_train_length = jnp.array(echo_train_length)
        self.sequence_type = sequence_type
        self.model = model
        self.vendor = vendor
        self.protocol_name = protocol_name
        self.trajectory_type = trajectory_type
        self.measurement_id = measurement_id
        self.patient_name = patient_name
        self._misc = _misc or {}

    @property
    def fa_degree(self) -> Array:
        """Get flip angle in degrees.

        Returns
        -------
        Array
            The flip angle in degrees.
        """
        return jnp.rad2deg(self.fa)

    @property
    def resolution(self) -> SpatialDimension:
        """Get resolution in [m].

        Returns
        -------
        SpatialDimension
            Resolution in [m].
        """
        return SpatialDimension(
            self.recon_fov.z / self.recon_matrix.z,
            self.recon_fov.y / self.recon_matrix.y,
            self.recon_fov.x / self.recon_matrix.x,
        )

    @property
    def k_resolution(self) -> SpatialDimension:
        """Get k-space resolution in [1/m].

        Returns
        -------
        SpatialDimension
            k-space resolution in [1/m].
        """
        return SpatialDimension(
            1 / self.encoding_fov.z,
            1 / self.encoding_fov.y,
            1 / self.encoding_fov.x,
        )

    @property
    def misc(self) -> dict[str, Any]:
        """Get miscellaneous parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary with miscellaneous parameters.
        """
        return self._misc

    def __repr__(self) -> str:
        """Return string representation of KHeader.

        Returns
        -------
        str
            String representation of KHeader.
        """
        out = (
            f'{type(self).__name__} with:\n'
            f'Resolution [m]: {self.resolution!s}\n'
            f'k-space resolution [1/m]: {self.k_resolution!s}\n'
            f'Matrix: {self.recon_matrix!s}\n'
            f'FOV [m]: {self.recon_fov!s}\n'
            f'Acquisition info: {self.acq_info!s}\n'
            f'Trajectory type: {self.trajectory_type!s}\n'
            f'Sequence type: {self.sequence_type!s}\n'
            f'Protocol name: {self.protocol_name!s}\n'
            f'Patient name: {self.patient_name!s}\n'
            f'Measurement ID: {self.measurement_id!s}\n'
            f'Model: {self.model!s}\n'
            f'Vendor: {self.vendor!s}\n'
            f'Larmor frequency [Hz]: {self.lamor_frequency_proton!s}\n'
            f'Date and time: {self.datetime!s}\n'
            f'TE [s]: {summarize_tensorvalues(self.te)}\n'
            f'TI [s]: {summarize_tensorvalues(self.ti)}\n'
            f'FA [rad]: {summarize_tensorvalues(self.fa)}\n'
            f'TR [s]: {summarize_tensorvalues(self.tr)}\n'
            f'Echo spacing [s]: {summarize_tensorvalues(self.echo_spacing)}\n'
            f'Echo train length: {self.echo_train_length!s}'
        )
        return out

    @classmethod
    def from_ismrmrd(cls, header: ismrmrdschema.ismrmrdHeader) -> Self:
        """Create KHeader from ISMRMRD header.

        Parameters
        ----------
        header : ismrmrdschema.ismrmrdHeader
            ISMRMRD header.

        Returns
        -------
        Self
            A new KHeader instance.
        """
        # Get encoding limits
        encoding_limits = header.encoding[0].encodedSpace
        recon_limits = header.encoding[0].reconSpace

        # Get matrix dimensions
        encoding_matrix = SpatialDimension(
            encoding_limits.matrixSize.z,
            encoding_limits.matrixSize.y,
            encoding_limits.matrixSize.x,
        )
        recon_matrix = SpatialDimension(
            recon_limits.matrixSize.z,
            recon_limits.matrixSize.y,
            recon_limits.matrixSize.x,
        )

        # Get field of view
        encoding_fov = SpatialDimension(
            encoding_limits.fieldOfView_mm.z / 1000,
            encoding_limits.fieldOfView_mm.y / 1000,
            encoding_limits.fieldOfView_mm.x / 1000,
        )
        recon_fov = SpatialDimension(
            recon_limits.fieldOfView_mm.z / 1000,
            recon_limits.fieldOfView_mm.y / 1000,
            recon_limits.fieldOfView_mm.x / 1000,
        )

        # Get acquisition info
        acq_info = AcqInfo.from_ismrmrd(header)

        # Get trajectory info
        trajectory_type = enums.TrajectoryType.from_ismrmrd(header)

        # Get sequence info
        sequence_type = header.sequenceParameters.sequence_type if header.sequenceParameters else UNKNOWN

        # Get scanner info
        model = header.acquisitionSystemInformation.systemModel if header.acquisitionSystemInformation else UNKNOWN
        vendor = header.acquisitionSystemInformation.systemVendor if header.acquisitionSystemInformation else UNKNOWN

        # Get protocol info
        protocol_name = header.measurementInformation.protocolName if header.measurementInformation else UNKNOWN
        measurement_id = header.measurementInformation.measurementID if header.measurementInformation else UNKNOWN
        patient_name = header.subjectInformation.patientName if header.subjectInformation else UNKNOWN

        # Get timing info
        te = header.sequenceParameters.TE if header.sequenceParameters else []
        ti = header.sequenceParameters.TI if header.sequenceParameters else []
        fa = header.sequenceParameters.flipAngle_deg if header.sequenceParameters else []
        tr = header.sequenceParameters.TR if header.sequenceParameters else []
        echo_spacing = header.sequenceParameters.echo_spacing if header.sequenceParameters else []
        echo_train_length = header.encoding[0].echoTrainLength if header.encoding[0].echoTrainLength else 1

        # Convert units
        te = [ms_to_s(t) for t in te]
        ti = [ms_to_s(t) for t in ti]
        fa = [deg_to_rad(f) for f in fa]
        tr = [ms_to_s(t) for t in tr]
        echo_spacing = [ms_to_s(t) for t in echo_spacing]

        # Get Larmor frequency
        lamor_frequency_proton = (
            header.measurementInformation.measurementFrequency if header.measurementInformation else None
        )

        # Get date and time
        datetime_str = header.measurementInformation.measurementTime if header.measurementInformation else None
        datetime_obj = datetime.datetime.fromisoformat(datetime_str) if datetime_str else None

        return cls(
            recon_matrix=recon_matrix,
            encoding_matrix=encoding_matrix,
            recon_fov=recon_fov,
            encoding_fov=encoding_fov,
            acq_info=acq_info,
            trajectory_type=trajectory_type,
            sequence_type=sequence_type,
            model=model,
            vendor=vendor,
            protocol_name=protocol_name,
            measurement_id=measurement_id,
            patient_name=patient_name,
            te=te,
            ti=ti,
            fa=fa,
            tr=tr,
            echo_spacing=echo_spacing,
            echo_train_length=echo_train_length,
            lamor_frequency_proton=lamor_frequency_proton,
            datetime=datetime_obj,
        )

    @classmethod
    def from_dicom(cls, dataset: Any) -> Self:
        """Create KHeader from DICOM dataset.

        Parameters
        ----------
        dataset : Any
            DICOM dataset.

        Returns
        -------
        Self
            A new KHeader instance.
        """
        # Get matrix dimensions
        recon_matrix = SpatialDimension(
            dataset.Rows,
            dataset.Columns,
            1,
        )
        encoding_matrix = recon_matrix

        # Get field of view
        recon_fov = SpatialDimension(
            dataset.PixelSpacing[0] * dataset.Rows / 1000,
            dataset.PixelSpacing[1] * dataset.Columns / 1000,
            1,
        )
        encoding_fov = recon_fov

        # Get acquisition info
        acq_info = AcqInfo.from_dicom(dataset)

        # Get trajectory info
        trajectory_type = enums.TrajectoryType.from_dicom(dataset)

        # Get sequence info
        sequence_type = dataset.SequenceName if hasattr(dataset, 'SequenceName') else UNKNOWN

        # Get scanner info
        model = dataset.ManufacturerModelName if hasattr(dataset, 'ManufacturerModelName') else UNKNOWN
        vendor = dataset.Manufacturer if hasattr(dataset, 'Manufacturer') else UNKNOWN

        # Get protocol info
        protocol_name = dataset.ProtocolName if hasattr(dataset, 'ProtocolName') else UNKNOWN
        measurement_id = dataset.SOPInstanceUID if hasattr(dataset, 'SOPInstanceUID') else UNKNOWN
        patient_name = dataset.PatientName if hasattr(dataset, 'PatientName') else UNKNOWN

        # Get timing info
        te = [dataset.EchoTime / 1000] if hasattr(dataset, 'EchoTime') else []
        ti = [dataset.InversionTime / 1000] if hasattr(dataset, 'InversionTime') else []
        fa = [deg_to_rad(dataset.FlipAngle)] if hasattr(dataset, 'FlipAngle') else []
        tr = [dataset.RepetitionTime / 1000] if hasattr(dataset, 'RepetitionTime') else []
        echo_spacing = [dataset.EchoSpacing / 1000] if hasattr(dataset, 'EchoSpacing') else []
        echo_train_length = dataset.EchoTrainLength if hasattr(dataset, 'EchoTrainLength') else 1

        # Get Larmor frequency
        lamor_frequency_proton = dataset.ImagingFrequency * 1e6 if hasattr(dataset, 'ImagingFrequency') else None

        # Get date and time
        datetime_str = dataset.AcquisitionDateTime if hasattr(dataset, 'AcquisitionDateTime') else None
        datetime_obj = datetime.datetime.fromisoformat(datetime_str) if datetime_str else None

        return cls(
            recon_matrix=recon_matrix,
            encoding_matrix=encoding_matrix,
            recon_fov=recon_fov,
            encoding_fov=encoding_fov,
            acq_info=acq_info,
            trajectory_type=trajectory_type,
            sequence_type=sequence_type,
            model=model,
            vendor=vendor,
            protocol_name=protocol_name,
            measurement_id=measurement_id,
            patient_name=patient_name,
            te=te,
            ti=ti,
            fa=fa,
            tr=tr,
            echo_spacing=echo_spacing,
            echo_train_length=echo_train_length,
            lamor_frequency_proton=lamor_frequency_proton,
            datetime=datetime_obj,
        )
