# tfrlrl - 0.0.0

A Python reinforcement learning library providing core RL infrastructure including environment sampling, replay buffers, and data models for working with Gymnasium environments.

## Features

- **Environment Sampling**: Single and parallel environment sampling using Ray
- **Replay Buffers**: Efficient circular buffer implementation for experience replay
- **Dynamic Data Models**: Type-safe dataclasses that adapt to environment specifications
- **CLI Tools**: Command-line interface for sampling and data collection
- **Configuration Management**: Centralized settings via Dynaconf

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

# Sample with parallel environments
poetry run tfrlrl-sample --env-id CartPole-v1 --n-steps 1000 --n-envs 4

# Control log level via environment variable
TFRLRL_LOG_LEVEL=DEBUG poetry run tfrlrl-sample --env-id CartPole-v1 --n-steps 100
```

**Options:**

- `--env-id`: Gymnasium environment ID (e.g., CartPole-v1, MountainCar-v0)
- `--n-steps`: Total number of steps to sample
- `--n-envs`: Number of parallel environments (default: 1)

### tfrlrl-sgd

Perform basic stochastic gradient ascent to optimise the policy. This is intended solely for validating the code base on the toy-example. The CLI currently assumes that the environment will have a discrete state and action spaces. The policy is a linear soft-max policy and a one-hot encoding is used for the policy features. 

**Basic Usage:**

```bash
# Perform stochastic gradient ascent on the given environment
poetry run tfrlrl-sgd --env-id FrozenLake-v1 --n-iterations 100

**Options:**

- `--env-id`: Gymnasium environment ID (e.g., FrozenLake-v1)
- `--n-iterations`: Total number of policy updates to perform.
- `--n-episodes`: Total number of episodes to sample during each policy update.
- `--alpha`: The initial step size in stochastic gradient ascent. Step sizes are linearly decreased w.r.t. the iteration of stochastic gradients.

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

