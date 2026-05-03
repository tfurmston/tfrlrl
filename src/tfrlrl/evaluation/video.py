import logging
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from tfrlrl.policies.base import BasePyTorchPolicy

logger = logging.getLogger(__name__)


def construct_video_evaluation(
    env_id: str, policy: BasePyTorchPolicy, n_episodes: int, video_folder: str, name_prefix: str
):
    """
    Construct video evaluations of the policy on the given environment.

    This function constructs video evaluations of the policy on the given environment. One video is constructed
    for each evaluation episode.

    Args:
        env_id: The Gym environment ID to be used in the evaluation.
        policy: The policy which is to be evaluated on the given environment.
        n_episodes: The number of episodes to evaluate the policy on.
        video_folder: The directory in which the video evaluations will be saved.
        name_prefix: The prefix of the video file names.

    """
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(env_id, render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: True,
    )

    for episode_num in range(n_episodes):
        obs, _ = env.reset()
        episode_reward: float = 0.0
        step_count = 0

        episode_over = False
        while not episode_over:
            action = policy.generate_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            step_count += 1
            episode_over = terminated or truncated

        logger.info('Episode %s: %s steps, reward = %s', episode_num + 1, step_count, episode_reward)

    env.close()
