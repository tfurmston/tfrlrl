import os

from pytest import fixture


@fixture(scope='session')
def test_ray_cluster(num_cpus: int = 2):
    """Pytest fixture for construct a test Ray cluster."""
    os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'

    import ray

    ray.init(num_cpus=num_cpus)
    yield
    ray.shutdown()
    del os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO']
