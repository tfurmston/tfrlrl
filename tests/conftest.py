import ray
from pytest import fixture


@fixture(scope='session')
def test_ray_cluster(num_cpus: int = 4, local_mode: bool = True):
    """Pytest fixture for construct a test Ray cluster."""
    ray.init(num_cpus=num_cpus, local_mode=local_mode)
    yield
    ray.shutdown()
