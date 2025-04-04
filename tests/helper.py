"""Helper/Utilities for test functions."""

import torch
from mrmax.operators import LinearOperator, Operator
from typing_extensions import TypeVarTuple, Unpack


def relative_image_difference(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate mean absolute relative difference between two images.

    Parameters
    ----------
    img1
        first image
    img2
        second image

    Returns
    -------
        mean absolute relative difference between images
    """
    image_difference = torch.mean(torch.abs(img1 - img2))
    image_mean = 0.5 * torch.mean(torch.abs(img1) + torch.abs(img2))
    if image_mean == 0:
        raise ValueError('average of images should be larger than 0')
    return image_difference / image_mean


def dotproduct_adjointness_test(
    operator: LinearOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> None:
    """Test the adjointness of linear operator and operator.H.

    Test if
         <Operator(u),v> == <u, Operator^H(v)>
         for one u ∈ domain and one v ∈ range of Operator.
    and if the shapes match.

    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.

    Parameters
    ----------
    operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    `AssertionError`
        if the adjointness property does not hold
    `AssertionError`
        if the shape of operator(u) and v does not match
        if the shape of u and operator.H(v) does not match

    """
    (forward_u,) = operator(u)
    (adjoint_v,) = operator.adjoint(v)

    # explicitly check the shapes, as flatten makes the dot product insensitive to wrong shapes
    assert forward_u.shape == v.shape
    assert adjoint_v.shape == u.shape

    dotproduct_range = torch.vdot(forward_u.flatten(), v.flatten())
    dotproduct_domain = torch.vdot(u.flatten().flatten(), adjoint_v.flatten())
    torch.testing.assert_close(dotproduct_range, dotproduct_domain, rtol=relative_tolerance, atol=absolute_tolerance)


def operator_isometry_test(
    operator: Operator[torch.Tensor, tuple[torch.Tensor]],
    u: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> None:
    """Test the isometry of a operator.

    Test if
         ||Operator(u)|| == ||u||
         for u ∈ domain of Operator.

    Parameters
    ----------
    operator
        operator
    u
        element of the domain of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the adjointness property does not hold
    """
    torch.testing.assert_close(
        torch.norm(u), torch.norm(operator(u)[0]), rtol=relative_tolerance, atol=absolute_tolerance
    )


def linear_operator_unitary_test(
    operator: LinearOperator, u: torch.Tensor, relative_tolerance: float = 1e-3, absolute_tolerance=1e-5
) -> None:
    """Test if a linear operator is unitary.

    Test if
         Operator.adjoint(Operator(u)) == u
         for u ∈ domain of Operator.

    Parameters
    ----------
    operator
        linear operator
    u
        element of the domain of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the adjointness property does not hold
    """
    torch.testing.assert_close(u, operator.adjoint(operator(u)[0])[0], rtol=relative_tolerance, atol=absolute_tolerance)


Tin = TypeVarTuple('Tin')


def autodiff_test(
    operator: Operator[Unpack[Tin], tuple[torch.Tensor, ...]],
    *u: Unpack[Tin],
) -> None:
    """Test if autodiff of an operator is working.
    This test does not check that the gradient is correct but simply that it can be calculated using both torch.func.jvp
    and torch.func.vjp.

    Parameters
    ----------
    operator
        operator
    u
        element(s) of the domain of the operator

    Raises
    ------
    AssertionError
        if autodiff fails or the result is not finite
    """
    # Forward-mode autodiff using jvp
    with torch.autograd.detect_anomaly():
        v_range, jvp = torch.func.jvp(operator.forward, u, u)
    assert all(x.isfinite().all() for x in v_range), 'result is not finite'
    assert all(x.isfinite().all() for x in jvp), 'JVP is not finite'

    # Reverse-mode autodiff using vjp
    with torch.autograd.detect_anomaly():
        (output, vjpfunc) = torch.func.vjp(operator.forward, *u)
        vjp = vjpfunc(v_range)
    assert all(x.isfinite().all() for x in output), 'result is not finite'
    assert all(x.isfinite().all() for x in vjp), 'VJP is not finite'


def gradient_of_linear_operator_test(
    operator: LinearOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> None:
    """Test the gradient of a linear operator is the adjoint.
    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.

    Parameters
    ----------
    operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the gradient is not the adjoint
    """
    # Gradient of the forward via vjp
    (_, vjpfunc) = torch.func.vjp(operator.forward, u)
    assert torch.allclose(vjpfunc((v,))[0], operator.adjoint(v)[0], rtol=relative_tolerance, atol=absolute_tolerance)

    # Gradient of the adjoint via vjp
    (_, vjpfunc) = torch.func.vjp(operator.adjoint, v)
    assert torch.allclose(vjpfunc((u,))[0], operator.forward(u)[0], rtol=relative_tolerance, atol=absolute_tolerance)


def forward_mode_autodiff_of_linear_operator_test(
    operator: LinearOperator,
    u: torch.Tensor,
    v: torch.Tensor,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> None:
    """Test the forward-mode autodiff calculation.
    Verifies that the Jacobian-vector product (jvp) is equivalent to applying the operator.
    Note: This property should hold for all u and v.
    Commonly, this function is called with two random vectors u and v.

    Parameters
    ----------
    operator
        linear operator
    u
        element of the domain of the operator
    v
        element of the range of the operator
    relative_tolerance
        default is pytorch's default for float16
    absolute_tolerance
        default is pytorch's default for float16

    Raises
    ------
    AssertionError
        if the jvp yields different results than applying the operator
    """
    # jvp of the forward
    assert torch.allclose(
        torch.func.jvp(operator.forward, (u,), (u,))[0][0],
        operator.forward(u)[0],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    # jvp of the adjoint
    assert torch.allclose(
        torch.func.jvp(operator.adjoint, (v,), (v,))[0][0],
        operator.adjoint(v)[0],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
