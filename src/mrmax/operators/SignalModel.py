"""Signal Model Operators."""

import torch
from mrmax.operators.Operator import Operator
from typing_extensions import TypeVarTuple, Unpack

Tin = TypeVarTuple('Tin')


# SignalModel has multiple inputs and one output
class SignalModel(Operator[Unpack[Tin], tuple[torch.Tensor,]]):
    """Signal Model Operator."""
