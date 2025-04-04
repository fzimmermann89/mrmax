"""ADAM for solving non-linear minimization problems."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import optax

from mrmax.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrmax.operators.Operator import OperatorType


def adam(
    f: OperatorType,
    initial_parameters: Sequence[jnp.ndarray],
    n_iterations: int,
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0,
    amsgrad: bool = False,
    decoupled_weight_decay: bool = False,
    callback: Callable[[OptimizerStatus], bool | None] | None = None,
) -> tuple[jnp.ndarray, ...]:
    r"""Adam for non-linear minimization problems.

    Adam [KING2015]_ (Adaptive Moment Estimation) is a first-order optimization algorithm that adapts learning rates
    for each parameter using estimates of the first and second moments of the gradients.

    The parameter update rule is:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
        \theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t

    where
    :math:`g_t` is the gradient at step :math:`t`,
    :math:`m_t` and :math:`v_t` are biased estimates of the first and second moments,
    :math:`\hat{m}_t` and :math:`\hat{v}_t` are bias-corrected estimates,
    :math:`\eta` is the learning rate,
    :math:`\epsilon` is a small constant for numerical stability,
    :math:`\beta_1` and :math:`\beta_2` are decay rates for the moment estimates.

    Steps of the Adam algorithm:

    1. Initialize parameters and moment estimates (:math:`m_0`, :math:`v_0`).
    2. Compute the gradient of the objective function.
    3. Compute bias-corrected estimates of the moments :math:`\hat{m}_t` and :math:`\hat{v}_t`.
    4. Update parameters using the adaptive step size.

    This implementation uses Optax's Adam optimizer, supporting both standard Adam and decoupled weight decay regularization (AdamW) [LOS2019]_

    References
    ----------
    .. [KING2015] Kingma DP, Ba J (2015) Adam: A Method for Stochastic Optimization. ICLR.
       https://doi.org/10.48550/arXiv.1412.6980
    .. [LOS2019] Loshchilov I, Hutter F (2019) Decoupled Weight Decay Regularization. ICLR.
       https://doi.org/10.48550/arXiv.1711.05101
    .. [REDDI2019] Sashank J. Reddi, Satyen Kale, Sanjiv Kumar (2019) On the Convergence of Adam and Beyond. ICLR.
       https://doi.org/10.48550/arXiv.1904.09237

    Parameters
    ----------
    f
        scalar-valued function to be optimized
    initial_parameters
        Sequence (for example list) of parameters to be optimized.
        Note that these parameters will not be changed. Instead, we create a copy and
        leave the initial values untouched.
    n_iterations
        number of iterations
    learning_rate
        learning rate
    betas
        coefficients used for computing running averages of gradient and its square
    eps
        term added to the denominator to improve numerical stability
    weight_decay
        weight decay (L2 penalty if `decoupled_weight_decay` is `False`)
    amsgrad
        whether to use the AMSGrad variant [REDDI2019]_
    decoupled_weight_decay
        whether to use Adam (default) or AdamW (if set to `True`) [LOS2019]_
    callback
        function to be called after each iteration. This can be used to monitor the progress of the algorithm.
        If it returns `False`, the algorithm stops at that iteration.

    Returns
    -------
        list of optimized parameters
    """
    # Create a copy of initial parameters
    parameters = tuple(jnp.array(p) for p in initial_parameters)

    # Create optimizer
    if decoupled_weight_decay:
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=betas[0],
            b2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optax.adam(
            learning_rate=learning_rate,
            b1=betas[0],
            b2=betas[1],
            eps=eps,
        )

    # Initialize optimizer state
    opt_state = optimizer.init(parameters)

    # Define the update step
    @jax.jit
    def update_step(params, state):
        # Compute gradients
        grad_fn = jax.grad(lambda *args: f(*args)[0])
        grads = grad_fn(*params)

        # Update parameters
        updates, new_state = optimizer.update(grads, state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_state

    # Run optimization
    for iteration in range(n_iterations):
        parameters, opt_state = update_step(parameters, opt_state)

        if callback is not None:
            continue_iterations = callback({'solution': parameters, 'iteration_number': iteration})
            if continue_iterations is False:
                break

    return parameters
