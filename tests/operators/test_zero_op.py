import torch
from mrmax.operators import IdentityOp, LinearOperator, MagnitudeOp, Operator, ZeroOp
from mrmax.operators.LinearOperator import LinearOperatorSum
from typing_extensions import assert_type

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def test_zero_op_keepshape() -> None:
    """Test that the zero operator returns zeros."""
    generator = RandomGenerator(seed=0)
    x = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp(keep_shape=True)
    (actual,) = operator(x)
    expected = torch.zeros_like(x)
    torch.testing.assert_close(actual, expected)


def test_zero_op_scalar() -> None:
    """Test that the zero operator returns single zero."""
    generator = RandomGenerator(seed=0)
    x = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp(keep_shape=False)
    (actual,) = operator(x)
    expected = torch.tensor(0)
    torch.testing.assert_close(actual, expected)


def test_zero_op_neutral_linop() -> None:
    """Test that the zero operator is neutral for addition."""
    op = IdentityOp()
    zero = ZeroOp()

    rsum = op + zero
    lsum = zero + op
    assert rsum is op
    assert lsum is op
    assert_type(lsum, LinearOperator)
    assert_type(rsum, LinearOperator)

    assert zero + zero is zero

    assert 1 * zero is zero
    assert zero * 1 is zero


def test_zero_op_neutral_op() -> None:
    """Test that the zero operator is neutral for addition."""
    op = MagnitudeOp() @ IdentityOp()
    zero = ZeroOp()
    rsum = op + zero
    lsum = zero + op
    assert rsum is op
    assert lsum is op
    assert_type(rsum, Operator[torch.Tensor, tuple[torch.Tensor]])
    assert_type(lsum, Operator[torch.Tensor, tuple[torch.Tensor]])


def test_zero_op_adjoint_keepshape() -> None:
    """Test that the adjoint of the zero operator is the zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp(keep_shape=True)
    dotproduct_adjointness_test(operator, u, v)


def test_zero_op_grad_keepshape() -> None:
    """Test gradient of zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp(keep_shape=True)
    gradient_of_linear_operator_test(operator, u, v)


def test_zero_op_forward_mode_autodiff_keepshape() -> None:
    """Test forward-mode autodiff of zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    operator = ZeroOp(keep_shape=True)
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)


def test_zero_op_adjoint_scalar() -> None:
    """Test that the adjoint of the zero operator is the zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    # We can't test the operator directly, because the adjointness is only after
    # broadcasting the scalar to the shape of the input and expading the dtype.
    # We achieve this by instead testing ZeroOp + IdentityOp
    # We can't use '+' because it would short-circuit to IdentityOp,
    # as ZeroOp is the neutral element of the addition.
    operator = LinearOperatorSum(ZeroOp(keep_shape=False), IdentityOp())
    dotproduct_adjointness_test(operator, u, v)


def test_zero_op_grad_scalar() -> None:
    """Test gradient of zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    operator = LinearOperatorSum(ZeroOp(keep_shape=False), IdentityOp())
    gradient_of_linear_operator_test(operator, u, v)


def test_zero_op_forward_mode_autodiff_scalar() -> None:
    """Test forward-mode autodiff of zero operator."""
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(2, 3, 4)
    v = generator.complex64_tensor(2, 3, 4)
    operator = LinearOperatorSum(ZeroOp(keep_shape=False), IdentityOp())
    forward_mode_autodiff_of_linear_operator_test(operator, u, v)
