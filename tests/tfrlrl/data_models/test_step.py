import numpy as np
import pytest

from tfrlrl.data_models.step import Step, Steps


@pytest.mark.parametrize('env_id, time_step, observation, action, next_observation, reward, done, info',
                         [
                             (
                                 'CartPole-v1-1234',
                                 0,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 1,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 False,
                                 {}
                             ),
                             (
                                 'CartPole-v1-1234',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 True,
                                 {}
                             ),
                         ])
def test_step_valid_example(env_id, time_step, observation, action, next_observation, reward, done, info):
    """Test that the Step class properly formats observation and next_obaservation."""
    step = Step(
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
    assert step.observation.shape == (4, 1)
    assert step.action == action
    assert step.next_observation.shape == (4, 1)
    assert step.reward == reward
    assert step.done is done


@pytest.mark.parametrize('env_id, time_step, observation, action, next_observation, reward, done, info',
                         [
                             (
                                 1,
                                 0,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 1,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 False,
                                 {}
                             ),
                             (
                                 'CartPole-v1-1234',
                                 2.5,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 True,
                                 {}
                             ),
                             (
                                 'CartPole-v1-1234',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0.1,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 True,
                                 {}
                             ),
                             (
                                 'CartPole-v1-1234',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 1,
                                 True,
                                 {}
                             ),
                             (
                                 'CartPole-v1-1234',
                                 2,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 0,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 'true',
                                 {}
                             ),
                         ])
def test_step_invalid_example(env_id, time_step, observation, action, next_observation, reward, done, info):
    """Test that the Step class throws error on invalid data inputs."""
    with pytest.raises(TypeError):
        Step(
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
                                 'CartPole-v1-1234',
                                 0,
                                 np.array([-0.0464053, -0.04271065, 0.03379071, -0.04416595]),
                                 1,
                                 np.array([-0.04725951, -0.23830044, 0.03290739, 0.25898385]),
                                 0.1,
                                 False,
                                 {}
                             ),
                         ])
def test_steps_valid_example(env_id, time_step, observation, action, next_observation, reward, done, info):
    """Test that the Steps class properly formats data from list of steps."""
    n_steps = 10
    steps = Steps(
        sample_steps=[
            Step(
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
    assert steps.observations.shape == (4, n_steps)
    assert steps.next_observations.shape == (4, n_steps)
    assert steps.rewards.shape == (n_steps, )
    assert steps.dones.shape == (n_steps, )
