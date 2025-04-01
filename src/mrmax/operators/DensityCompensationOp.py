"""Class for Density Compensation Operator."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from mrmax.data.DcfData import DcfData
from mrmax.operators.EinsumOp import EinsumOp


class DensityCompensationOp(EinsumOp, eqx.Module):
    """Density Compensation Operator."""

    def __init__(self, dcf: DcfData | Array) -> None:
        """Initialize a Density Compensation Operator.

        Parameters
        ----------
        dcf
           Density Compensation Data
        """
        if isinstance(dcf, DcfData):
            # only tensors can currently be used as buffers
            # thus, einsumop is initialized with the tensor data
            # TODO: change if einsumop can handle dataclasses
            dcf_tensor = dcf.data
        else:
            dcf_tensor = dcf
        super().__init__(dcf_tensor, '...,... -> ...')
