"""Tests for Wavelet Operator."""

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from mrmax.operators import WaveletOp
from mrmax.operators.WaveletOp import WaveletType
from ptwt.conv_transform import wavedec
from ptwt.conv_transform_2 import wavedec2
from ptwt.conv_transform_3 import wavedec3

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
    linear_operator_unitary_test,
    operator_isometry_test,
)


def create_wavelet_op_and_domain_range(
    img_shape: Sequence[int],
    domain_shape: Sequence[int],
    dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    wavelet_name: WaveletType,
) -> tuple[WaveletOp, torch.Tensor, torch.Tensor]:
    """Create a wavelet operator and an element from domain and range."""
    random_generator = RandomGenerator(seed=0)

    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name)

    # calculate 1D length of wavelet coefficients
    wavelet_stack_length = torch.sum(torch.as_tensor([(np.prod(shape)) for shape in wavelet_op.coefficients_shape]))

    # sorted and normed dimensions needed to correctly calculate range
    dim_sorted = sorted([d % len(img_shape) for d in dim], reverse=True)
    range_shape = list(img_shape)
    range_shape[dim_sorted[-1]] = int(wavelet_stack_length)
    [range_shape.pop(d) for d in dim_sorted[:-1]]

    u = random_generator.complex64_tensor(size=img_shape)
    v = random_generator.complex64_tensor(size=range_shape)
    return wavelet_op, u, v


@pytest.mark.parametrize(
    ('img_shape', 'domain_shape', 'dim'),
    [
        ((5, 16, 16, 16), (16,), (-1,)),
        ((5, 16, 16, 16), (16, 16), (-2, -1)),
        ((5, 16, 16, 16), (16, 16, 16), (-3, -2, -1)),
    ],
)
def test_wavelet_op_coefficient_transform(
    img_shape: Sequence[int], domain_shape: Sequence[int], dim: tuple[int] | tuple[int, int] | tuple[int, int, int]
) -> None:
    """Test transform between ptwt and mrmax coefficient format."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.float32_tensor(size=img_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim)
    if len(dim) == 1:
        coeff_ptwt = wavedec(img, 'haar', level=2, mode='reflect')
        coeff_mrmax = wavelet_op._format_coeffs_1d(coeff_ptwt)
        coeff_ptwt_transformed_1d = wavelet_op._undo_format_coeffs_1d(coeff_mrmax)

        # all entries are single tensors
        for i in range(len(coeff_ptwt_transformed_1d)):
            assert torch.allclose(coeff_ptwt[i], coeff_ptwt_transformed_1d[i])

    elif len(dim) == 2:
        coeff_ptwt = wavedec2(img, 'haar', level=2, mode='reflect')
        coeff_mrmax = wavelet_op._format_coeffs_2d(coeff_ptwt)
        coeff_ptwt_transformed_2d = wavelet_op._undo_format_coeffs_2d(coeff_mrmax)

        # first entry is tensor, the rest is list of tensors
        assert torch.allclose(coeff_ptwt[0], coeff_ptwt_transformed_2d[0])  # type: ignore[arg-type]
        for i in range(1, len(coeff_ptwt_transformed_2d)):
            assert all(
                torch.allclose(coeff_ptwt[i][j], coeff_ptwt_transformed_2d[i][j]) for j in range(len(coeff_ptwt[i]))
            )

    elif len(dim) == 3:
        coeff_ptwt = wavedec3(img, 'haar', level=2, mode='reflect')
        coeff_mrmax = wavelet_op._format_coeffs_3d(coeff_ptwt)
        coeff_ptwt_transformed_3d = wavelet_op._undo_format_coeffs_3d(coeff_mrmax)

        # first entry is tensor, the rest is dict
        assert torch.allclose(coeff_ptwt[0], coeff_ptwt_transformed_3d[0])  # type: ignore[arg-type]
        for i in range(1, len(coeff_ptwt_transformed_3d)):
            assert all(torch.allclose(coeff_ptwt[i][key], coeff_ptwt_transformed_3d[i][key]) for key in coeff_ptwt[i])


def test_wavelet_op_wrong_dim() -> None:
    """Wavelet only works for 1D, 2D and 3D data."""
    with pytest.raises(ValueError, match='Only 1D, 2D and 3D wavelet'):
        WaveletOp(dim=(0, 1, 2, 3))  # type: ignore[arg-type]


def test_wavelet_op_mismatch_dim_domain_shape() -> None:
    """Dimensions and shapes need to be of same length."""
    with pytest.raises(ValueError, match='Number of dimensions along which'):
        WaveletOp(domain_shape=(10, 20), dim=(-2,))


def test_wavelet_op_error_for_odd_domain_shape() -> None:
    with pytest.raises(NotImplementedError, match='ptwt only supports wavelet transforms for tensors with even'):
        WaveletOp(domain_shape=(11, 20), dim=(-2, -1))


def test_wavelet_op_complex_real_shape() -> None:
    """Test that the shape of the output tensors is the same for complex/real data."""
    img_shape = (6, 10, 20, 30)
    domain_shape = (20, 30, 6)
    dim = (-2, -1, -4)
    random_generator = RandomGenerator(seed=0)
    img_complex = random_generator.complex64_tensor(size=img_shape)
    img_real = random_generator.float32_tensor(size=img_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name='db4', level=None)
    (coeff_complex,) = wavelet_op(img_complex)
    (coeff_real,) = wavelet_op(img_real)
    assert coeff_complex.shape == coeff_real.shape
    assert wavelet_op.adjoint(coeff_complex)[0].shape == wavelet_op.adjoint(coeff_real)[0].shape


SHAPE_PARAMETERS = pytest.mark.parametrize(
    ('img_shape', 'domain_shape', 'dim'),
    [
        ((1, 5, 20, 30), (30,), (-1,)),
        ((5, 1, 10, 20, 30), (10,), (-3,)),
        ((1, 5, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (20, 30), (-2, -1)),
        ((4, 10, 20, 30), (10, 30), (-3, -1)),
        ((5, 10, 20, 30), (10, 20, 30), (-3, -2, -1)),
        ((6, 10, 20, 30), (6, 20, 30), (-4, -2, -1)),
        ((6, 10, 20, 30), (20, 30, 6), (-2, -1, -4)),
        ((6, 10, 20, 30), (20, 30, 6), (2, 3, 0)),
    ],
)


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@SHAPE_PARAMETERS
def test_wavelet_op_isometry(
    img_shape: Sequence[int],
    domain_shape: Sequence[int],
    dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    wavelet_name: WaveletType,
) -> None:
    """Test that the wavelet operator is a linear isometry."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.complex64_tensor(size=img_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name, level=None)
    operator_isometry_test(wavelet_op, img)


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@SHAPE_PARAMETERS
def test_wavelet_op_adjointness(
    img_shape: Sequence[int],
    domain_shape: Sequence[int],
    dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    wavelet_name: WaveletType,
) -> None:
    """Test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v."""
    dotproduct_adjointness_test(*create_wavelet_op_and_domain_range(img_shape, domain_shape, dim, wavelet_name))


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@SHAPE_PARAMETERS
def test_wavelet_op_unitary(
    img_shape: Sequence[int],
    domain_shape: Sequence[int],
    dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    wavelet_name: WaveletType,
) -> None:
    """Test if wavelet operator is unitary."""
    random_generator = RandomGenerator(seed=0)
    img = random_generator.complex64_tensor(size=img_shape)
    wavelet_op = WaveletOp(domain_shape=domain_shape, dim=dim, wavelet_name=wavelet_name)
    linear_operator_unitary_test(wavelet_op, img)


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@SHAPE_PARAMETERS
def test_wavelet_op_grad(
    img_shape: Sequence[int],
    domain_shape: Sequence[int],
    dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    wavelet_name: WaveletType,
) -> None:
    """Test gradient of wavelet operator."""
    gradient_of_linear_operator_test(*create_wavelet_op_and_domain_range(img_shape, domain_shape, dim, wavelet_name))


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4'])
@SHAPE_PARAMETERS
def test_wavelet_op_forward_mode_autodiff(
    img_shape: Sequence[int],
    domain_shape: Sequence[int],
    dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    wavelet_name: WaveletType,
) -> None:
    """Test forward-mode autodiff of wavelet operator."""
    forward_mode_autodiff_of_linear_operator_test(
        *create_wavelet_op_and_domain_range(img_shape, domain_shape, dim, wavelet_name)
    )
