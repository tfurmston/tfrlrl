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

**Output:**

The CLI provides a summary with:
- Total steps collected
- Number of completed episodes
- Mean reward ± standard deviation
- Reward range (min/max)

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

# Run with coverage report (requires 94% coverage)
make test-coverage

# Run a specific test file
poetry run pytest tests/tfrlrl/sampling/test_sampler.py
```

### Code Quality

```bash
# Run linting
make check-style

# Auto-format imports
make isort
```

### Version Management

```bash
make bump_major  # 0.0.0 -> 1.0.0
make bump_minor  # 0.0.0 -> 0.1.0
make bump_patch  # 0.0.0 -> 0.0.1
```

## License

MIT

