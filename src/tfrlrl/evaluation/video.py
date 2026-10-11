import logging

from tfrlrl.policies.base import BasePolicy
from tfrlrl.sampling.episodic_sampler import EpisodicSampler
from tfrlrl.sampling.sampler import VideoRecordingConfig
from tfrlrl.sampling.statistics_collection import EpisodeEvaluationStatisticsCollector

logger = logging.getLogger(__name__)


def construct_video_evaluation(env_id: str, policy: BasePolicy, n_episodes: int, video_folder: str, name_prefix: str):
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
    statistics_collector = EpisodeEvaluationStatisticsCollector(env_id=env_id)
    sampler = EpisodicSampler(
        env_id,
        statistics_collector,
        n_episodes=n_episodes,
        policy=policy,
        video_config=VideoRecordingConfig(video_folder=video_folder, name_prefix=name_prefix),
    )

    episode_statistics = []
    for episode_num, statistics in enumerate(sampler):
        logger.info(
            'Episode %s: %s steps, reward = %s',
            episode_num + 1,
            int(statistics.n_steps),
            float(statistics.total_reward),
        )
        episode_statistics.append(statistics)

    sampler.close()

    merged_statistics = statistics_collector.merge_statistics(episode_statistics)
    logger.info(
        'Evaluation summary: mean reward = %s, std reward = %s',
        float(merged_statistics.total_reward.mean()),
        float(merged_statistics.total_reward.std()),
    )
