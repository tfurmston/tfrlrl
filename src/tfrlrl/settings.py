import gymnasium as gym
from dynaconf import Dynaconf, Validator


def is_valid_environment(value):
    """
    Dynaconf validation condition function used to check that environment I.D. is valid.

    :param value: The given value of ENV provided to the Dynaconf settings class.
    :return: A Boolean indicating whether the environment is valis.
    """
    try:
        gym.make(value)
    except gym.error.NameNotFound:
        return False
    return True


settings = Dynaconf(
    envvar_prefix='TFRLRL',
    settings_files=['settings/settings.toml', 'settings/settings.local.toml', 'settings/.secrets.toml'],
    environments=True,
    validators=[
        Validator('LOG_LEVEL', must_exist=True, is_in=['DEBUG', 'INFO', 'WARN', 'ERROR']),
        Validator('ENV', must_exist=True, condition=is_valid_environment),
    ]
)
