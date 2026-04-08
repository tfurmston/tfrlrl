# tfrlrl - 0.0.0

A Python reinforcement learning library providing core RL infrastructure including environment sampling, replay buffers, and data models for working with Gymnasium environments.

## Features

- **Environment Sampling**: Single and parallel environment sampling using Ray
- **Replay Buffers**: Efficient circular buffer implementation for experience replay
- **Data Models**: Type-safe dataclasses that match environment specifications
- **Policies**: Support for PyTorch policies 
- **CLI Tools**: Command-line interface for sampling and data collection
- **Configuration Management**: Centralized settings via Dynaconf

### Environment Sampling

This repository provides some helper classes for sampling from environments. This includes the ability to sample whole episodes simply, and the ability to distribute the sampling through the use of Ray. The distributed samplers have the same API as the standard samplers, so it should be possible to swap in distributed sampling as required with no code changes. 

### Data Models

This repository includes a range of dataclasses that are used to designed the samples steps from environments. These dataclasses detect the specification of the environment, e.g. a discrete or continuous action space, and set the appropriate types for the corresponding fields in the data model. All actions, observations and rewards will be stored in NumPy arrays, with any required conversion automatically managed by these data classes. For example, when sampling from a toy environment with a discrete space, the integer state observations returned by Gym will automatically be stored in a NumPy array.

These classes automatically manage an additional dimension to the samples that allows aggregation of samples across multiple steps. This dimension is always the last dimension of the field. For example, `N` discrete actions will be store in a `(1, N)` NumPy array, while `N` observations, each of size `(o_1, o_2)`, will be stored in a `(o_1, 0_2, N)` array. 

### Policies

This repository provides flexible support for different policies. The only requirement is policies are implemented in PyTorch and inherit from the `BasePyTorchPolicy` base class. See the `DensePolicyNetwork` and `LinearSoftMaxNetwork` for examples. 

## Installation

### Production Installation

```bash
poetry install
```

### Development Installation

```bash
poetry install --with dev
```

## CLI Tools

### tfrlrl-sample

Sample steps from Gymnasium environments with support for parallel execution.

**Basic Usage:**

```bash
# Sample 100 steps from a single environment
poetry run tfrlrl-sample --env-id CartPole-v1 --n-steps 100

# Control log level via environment variable
TFRLRL_LOG_LEVEL=DEBUG poetry run tfrlrl-sample --env-id CartPole-v1 --n-steps 100
```

**Options:**

- `--env-id`: Gymnasium environment ID (e.g., CartPole-v1, MountainCar-v0)
- `--n-steps`: Total number of steps to sample

### tfrlrl-sgd

Perform basic stochastic gradient ascent to optimise the policy. This is intended solely for validating the code base on the toy-example. The CLI currently assumes that the environment will have a discrete state and action spaces. The policy is a linear soft-max policy and a one-hot encoding is used for the policy features.

**Basic Usage:**

```bash
# Perform stochastic gradient ascent on the given environment
poetry run tfrlrl-sgd --env-id FrozenLake-v1 --policy-class linear --n-iterations 100 

# With environment-specific configuration
poetry run tfrlrl-sgd --env-id FrozenLake-v1 --policy-class linear --n-iterations 100 --env-kwargs '{"is_slippery": false}'

# With custom hyperparameters
poetry run tfrlrl-sgd --env-id FrozenLake-v1 --policy-class linear --n-iterations 50 --n-episodes 200 --alpha 10.0
```

**Options:**

- `--env-id`: Gymnasium environment ID (e.g., FrozenLake-v1)
- `--n-iterations`: Total number of policy updates to perform (default: 100)
- `--n-episodes`: Total number of episodes to sample during each policy update (default: 100)
- `--alpha`: The initial step size in stochastic gradient ascent. Step sizes are linearly decreased w.r.t. the iteration of stochastic gradients (default: 100.0)
- `--n-samplers`: The number of samplers to use during sampling (default: 1)
- `--env-kwargs`: Environment-specific keyword arguments as a JSON string (default: `{}`). For example, `'{"is_slippery": false}'` for FrozenLake-v1
- `--n-samplers`: The number of samplers to use during sampling (default: 1)
- `--policy-class`: The class of policy to use in the environment. Allowed values are `linear` and `dense`.
- `--n-hidden`: The number of hidden dimensions to use in the case of a dense policy.

## Configuration

The library uses Dynaconf for configuration management. Settings can be controlled via:

- **Settings files**: `settings/settings.toml`, `settings/settings.local.toml`
- **Environment variables**: Prefix with `TFRLRL_` (e.g., `TFRLRL_LOG_LEVEL=DEBUG`)
- **Environments**: Supports default/development/production configurations

**Available Settings:**

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARN, ERROR)
- `ENV`: Default Gymnasium environment ID

## Development Guidelines

This project is configured through [Poetry](https://python-poetry.org/). To install Poetry follow the instructions [here](https://python-poetry.org/docs/#installation).

### Running Tests

```bash
# Run all tests
make test

# Run fast tests, e.g. for local development.
make test-local


# Run with coverage report (requires 94% coverage)
make test-coverage

# Run a specific test file
poetry run pytest tests/tfrlrl/sampling/test_sampler.py
```

### Code Quality

```bash
# Run linting
make check-style

# Run type checking
make check-typing

# Auto-format codebase
make format
```

### Version Management

```bash
make bump_major  # 0.0.0 -> 1.0.0
make bump_minor  # 0.0.0 -> 0.1.0
make bump_patch  # 0.0.0 -> 0.0.1
```

## License

MIT

