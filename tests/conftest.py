import ray
from pytest import fixture


@fixture(scope='session')
def test_ray_cluster(num_cpus: int = 4):
    ray.init(num_cpus=num_cpus)
    yield
    ray.shutdown()
