import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tfrlrl.optimisation.conjugate_gradients import calculate_conjugate_gradient


def _make_spd_matrix(n: int) -> np.ndarray:
    """Construct a random symmetric positive definite matrix of size (n, n)."""
    A = np.random.randn(n, n)
    return np.matmul(A.T, A) + n * np.eye(n)


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_exact_solution_with_n_iters_none(seed: int):
    """
    Test that calculate_conjugate_gradient returns the exact solution when n_iters is None.

    When n_iters is None the algorithm defaults to n iterations (the dimension of b), which is sufficient
    for exact convergence on a symmetric positive definite system.

    Args:
        seed: Random seed for generating the matrix and target vector.

    """
    np.random.seed(seed)
    n = 5
    A = _make_spd_matrix(n)
    b = np.random.randn(n)

    def mat_v_mult_fn(v):
        return np.matmul(A, v)

    x = calculate_conjugate_gradient(mat_v_mult_fn, b)

    np.testing.assert_allclose(np.matmul(A, x), b, rtol=1e-5, atol=1e-5)


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_exact_solution_with_n_iters_equal_to_dimension(seed: int):
    """
    Test that calculate_conjugate_gradient returns the exact solution when n_iters equals the dimension.

    This verifies that an explicit n_iters equal to the problem dimension is equivalent to n_iters=None.

    Args:
        seed: Random seed for generating the matrix and target vector.

    """
    np.random.seed(seed)
    n = 5
    A = _make_spd_matrix(n)
    b = np.random.randn(n)

    def mat_v_mult_fn(v):
        return np.matmul(A, v)

    x = calculate_conjugate_gradient(mat_v_mult_fn, b, n_iters=n)

    np.testing.assert_allclose(np.matmul(A, x), b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('n_iters', [1, 2, 3])
@given(seed=st.integers(min_value=0, max_value=10000))
@settings(deadline=None)
def test_approximate_solution_with_n_iters_less_than_dimension(n_iters: int, seed: int):
    """
    Test that calculate_conjugate_gradient reduces the A-norm error when n_iters is less than the dimension.

    With fewer iterations than the problem dimension, the algorithm returns an approximate solution. This
    test verifies that the output has the correct shape and that the A-norm error is reduced relative to
    the initial A-norm error (starting from x=0), which is guaranteed by the monotone convergence property
    of conjugate gradients on symmetric positive definite systems.

    Args:
        n_iters: The number of conjugate gradient iterations to perform (less than the problem dimension).
        seed: Random seed for generating the matrix and target vector.

    """
    np.random.seed(seed)
    n = 5
    A = _make_spd_matrix(n)
    b = np.random.randn(n)

    def mat_v_mult_fn(v):
        return np.matmul(A, v)

    x = calculate_conjugate_gradient(mat_v_mult_fn, b, n_iters=n_iters)
    x_exact = np.linalg.solve(A, b)

    assert x.shape == b.shape

    e_approx = x - x_exact
    e_init = -x_exact
    assert np.dot(e_approx, np.matmul(A, e_approx)) <= np.dot(e_init, np.matmul(A, e_init))


def test_zero_b_returns_zero_vector_without_nans():
    """
    Test that calculate_conjugate_gradient returns a zero vector when b is exactly zero.

    When b is the zero vector, x=0 already satisfies Ax=b, so the algorithm should return immediately
    rather than dividing rdotr (which is 0) by p.T.dot(z) (also 0), which would produce NaNs.

    """
    n = 5
    A = _make_spd_matrix(n)
    b = np.zeros(n)

    def mat_v_mult_fn(v):
        return np.matmul(A, v)

    x = calculate_conjugate_gradient(mat_v_mult_fn, b)

    np.testing.assert_array_equal(x, np.zeros(n))


@pytest.mark.parametrize(
    'b_shape',
    [
        (5, 2),
        (5, 3),
        (5, 1, 1),
        (2, 3, 5),
    ],
)
def test_raises_value_error_for_invalid_b_shape(b_shape: tuple):
    """
    Test that calculate_conjugate_gradient raises a ValueError for invalid b shapes.

    The function requires b to be a 1-D array or a 2-D column vector. Any other shape — including 2-D
    arrays with more than one column and arrays with more than two dimensions — should raise a ValueError.

    Args:
        b_shape: The invalid shape for the input vector b.

    """
    b = np.random.randn(*b_shape)

    def mat_v_mult_fn(v):
        return v

    with pytest.raises(ValueError):
        calculate_conjugate_gradient(mat_v_mult_fn, b)
