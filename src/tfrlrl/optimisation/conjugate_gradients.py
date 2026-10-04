from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class ConjugateGradientResult:
    """
    The solution of the conjugate-gradient algorithm, together with the resulting quadratic form, x^T A x.

    Args:
        x: The (approximate) solution of Ax = b.
        quadratic_form: The quadratic form, x^T A x.

    """

    x: np.ndarray
    quadratic_form: float


def calculate_conjugate_gradient(
    mat_v_mult_fn: Callable[[np.ndarray], np.ndarray], b: np.ndarray, n_iters: Optional[int] = None, tol: float = 1e-10
) -> ConjugateGradientResult:
    """
    Perform the conjugate-gradient algorithm to find the (approximate) solution, x, of Ax = b.

    Args:
        mat_v_mult_fn: A callable that takes an input vector, x, and returns the matrix multiplication of A
        with x. The purpose of making this a callable is to allow the user to perform conjugate--gradients without
        ever having to materialise the matrix, A.
        b: The target solution vector of the matrix-vector multiplication in the conjugate-gradient algorithm.
        n_iters: The maximum number of iterations to perform in conjugate-gradients.
        tol: The toleration below which the conjugate-gradient algorithm will terminate.

    Returns:
        A ConjugateGradientResult containing the approximate solution, x, of Ax = b, together with the quadratic
        form x^T A x. The quadratic form is accumulated during the algorithm itself (using the matrix-vector
        products already calculated for the conjugate-gradient recursion), so no additional evaluation of
        mat_v_mult_fn is required to obtain it.

    Raises:
        If the b not a vector, i.e. either a one-dimensional NumPy array or a two dimensional array with a single
        element in the last dimension, then a ValueError will be thrown.

    """
    if b.ndim > 2 or (b.ndim == 2 and b.shape[1] > 1):
        raise ValueError('Input vector to conjugate gradients needs to be a 1-dimensional array: %s', b.shape)

    if n_iters is None:
        n_iters = b.flatten().shape[0]

    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    ax = np.zeros_like(b)
    rdotr = np.inner(r, r)

    if rdotr >= tol:
        for _ in range(n_iters):
            z = mat_v_mult_fn(p)

            v = rdotr / p.T.dot(z)
            x += v * p
            ax += v * z
            r -= v * z

            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr
            if rdotr < tol:
                break

    return ConjugateGradientResult(x=x, quadratic_form=float(x.dot(ax)))
