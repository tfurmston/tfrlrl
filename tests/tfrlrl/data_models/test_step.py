import numpy as np
import pytest

from tfrlrl.data_models.step import construct_step_dataclasses


@pytest.mark.parametrize('env_id, time_step, observation, action, next_observation, reward, done, info',
                         [
                             (
                                 'CartPole-v1',
                                 0,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 1,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 False,
                                 {},
                             ),
                             (
                                 'CartPole-v1',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 True,
                                 {},
                             ),
                             (
                                 'Ant-v4',
                                 2,
                                 np.random.uniform(0.0, 1.0, (27,)),
                                 np.random.uniform(0.0, 1.0, (8,)),
                                 np.random.uniform(0.0, 1.0, (27,)),
                                 0.1,
                                 True,
                                 {},
                             ),
                         ])
def test_step_valid_example(env_id, time_step, observation, action, next_observation, reward, done, info):
    """Test that the Step class properly formats observation and next_obaservation."""
    step_cls, steps_cls = construct_step_dataclasses(env_id)

    step = step_cls(
        env_id=env_id,
        time_step=time_step,
        observation=observation,
        action=action,
        next_observation=next_observation,
        reward=reward,
        done=done,
        info=info,
    )
    assert step.env_id == env_id
    assert step.time_step == time_step
    assert step.observation.shape == observation.shape + (1, )
    np.testing.assert_allclose(step.observation[:, 0], observation)
    if isinstance(action, int):
        assert step.action == action
    else:
        np.testing.assert_allclose(step.action[:, 0], action)
        assert step.action.shape == action.shape + (1, )
    np.testing.assert_allclose(step.next_observation[:, 0], next_observation)
    assert step.next_observation.shape == next_observation.shape + (1, )
    assert step.reward == reward
    assert step.done is done


@pytest.mark.parametrize('env_id, time_step, observation, action, next_observation, reward, done, info',
                         [
                             (
                                 'CartPole-v1',
                                 2.5,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 True,
                                 {},
                             ),
                             (
                                 'CartPole-v1',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0.1,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 True,
                                 {},
                             ),
                             (
                                 'CartPole-v1',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 1,
                                 True,
                                 {},
                             ),
                             (
                                 'CartPole-v1',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 'true',
                                 {},
                             ),
                             (
                                 'Ant-v4',
                                 2,
                                 np.random.uniform(0.0, 1.0, (27,)),
                                 1,
                                 np.random.uniform(0.0, 1.0, (27,)),
                                 0.1,
                                 True,
                                 {},
                             ),
                         ])
def test_step_invalid_example(env_id, time_step, observation, action, next_observation, reward, done, info):
    """Test that the Step class throws error on invalid data inputs."""
    step_cls, steps_cls = construct_step_dataclasses(env_id)

    with pytest.raises(TypeError):
        step_cls(
            env_id=env_id,
            time_step=time_step,
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done,
            info=info,
        )


@pytest.mark.parametrize('env_id, time_step, observation, action, next_observation, reward, done, info',
                         [
                             (
                                 'CartPole-v1',
                                 0,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 1,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 False,
                                 {}
                             ),
                             (
                                 'Ant-v4',
                                 2,
                                 np.random.uniform(0.0, 1.0, (27,)),
                                 np.random.uniform(0.0, 1.0, (8,)),
                                 np.random.uniform(0.0, 1.0, (27,)),
                                 0.1,
                                 True,
                                 {},
                             ),
                         ])
def test_steps_valid_example(env_id, time_step, observation, action, next_observation, reward, done, info):
    """Test that the Steps class properly formats data from list of steps."""
    step_cls, steps_cls = construct_step_dataclasses(env_id)

    n_steps = 10
    steps = steps_cls(
        sample_steps=[
            step_cls(
                env_id=env_id,
                time_step=i,
                observation=observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )
            for i in range(n_steps)]
    )

    assert steps.n_steps == n_steps
    assert isinstance(steps.env_ids, list)
    assert len(steps.env_ids) == n_steps
    assert isinstance(steps.time_steps, np.ndarray)
    assert isinstance(steps.observations, np.ndarray)
    assert isinstance(steps.actions, np.ndarray)
    assert isinstance(steps.next_observations, np.ndarray)
    assert isinstance(steps.rewards, np.ndarray)
    assert isinstance(steps.dones, np.ndarray)

    assert steps.time_steps.shape == (n_steps, )
    assert steps.observations.shape == observation.shape + (n_steps, )
    assert steps.next_observations.shape == next_observation.shape + (n_steps, )
    assert steps.rewards.shape == (n_steps, )
    assert steps.dones.shape == (n_steps, )
