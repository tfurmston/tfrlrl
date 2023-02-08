from tfrlrl.replay_buffer.replay_buffer import ReplayBuffer


def test_add_sample_n_samples_less_buffer_size():
    n_samples = 20
    buffer = ReplayBuffer(
        d_obs=2,
        d_action=1,
        buffer_size=100,
    )

    # Take n_samples
    # how to take the samples......
    #     use numpy

    # Store the samples in dict by time index

    # iterate through samples and check replay buffer against samples