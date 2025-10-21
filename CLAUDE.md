# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tfrlrl is a Python reinforcement learning library that provides core RL infrastructure including environment sampling, replay buffers, and data models for working with Gymnasium environments.

## Development Commands

### Setup
```bash
# Install production dependencies
make install

# Install development dependencies (required for linting/testing)
make install-dev
```

### Testing
```bash
# Run all tests with random ordering
make test

# Run tests with coverage report (requires 95% coverage to pass)
make test-coverage

# Run a single test file
poetry run pytest tests/tfrlrl/sampling/test_sampler.py

# Run a specific test
poetry run pytest tests/tfrlrl/sampling/test_sampler.py::test_name
```

### Code Quality
```bash
# Run flake8 linting
make check-style

# Auto-format imports with isort
make isort
```

### Version Management
```bash
# Bump version numbers
make bump_major  # 0.0.0 -> 1.0.0
make bump_minor  # 0.0.0 -> 0.1.0
make bump_patch  # 0.0.0 -> 0.0.1
```

### CLI Tools
```bash
# Sample steps from a Gymnasium environment (after install-dev)
poetry run tfrlrl-sample --env-id CartPole-v1 --n-steps 100

# Sample with parallel environments
poetry run tfrlrl-sample --env-id CartPole-v1 --n-steps 1000 --n-envs 4

# Control log level via environment variable
TFRLRL_LOG_LEVEL=DEBUG poetry run tfrlrl-sample --env-id CartPole-v1 --n-steps 100
```

## Architecture

### Data Models (src/tfrlrl/data_models/)

The library uses a **descriptor-based validation system** for type safety:
- `base.py`: Defines `Validator` abstract base class and concrete descriptors (StringDescriptor, IntDescriptor, FloatDescriptor, NumpyArrayDescriptor, etc.)
- Descriptors validate and preprocess values when set on dataclass fields

**Dynamic dataclass generation** based on environment:
- `step.py`: `construct_step_dataclass(env_id)` creates Step dataclasses dynamically
- `construct_steps_dataclass(env_id)` creates Steps (batch) dataclasses
- Action field type varies based on environment action space (Discrete vs Box)
- Steps dataclass has custom `__post_init__` that aggregates individual steps differently based on action space type

### Sampling (src/tfrlrl/sampling/)

Two-level sampling architecture:
- `Sampler`: Ray remote actor for single-environment sampling
  - Implements iterator protocol
  - Auto-resets environment on done/truncated
  - Each episode gets unique UUID for tracking
  - Takes random actions (policy should be injected separately)
- `RaySampler`: Orchestrates multiple Sampler actors for parallel sampling
  - Returns batched Steps objects via `ray.get()`
  - Uses environment-specific Steps dataclass

### Replay Buffer (src/tfrlrl/replay_buffer/)

Circular buffer implementation:
- Pre-allocated numpy arrays sized by buffer_size
- Stores: observations, next_observations, actions, rewards, dones
- Action array shape adapts to environment (Discrete: (1, N), Box: (action_shape, N))
- Sampling is random without replacement from valid buffer indices
- Returns `ReplayBufferSample` dataclass

### Configuration (settings.py)

Uses Dynaconf for multi-environment configuration:
- Settings files: `settings/settings.toml`, `settings/settings.local.toml`, `settings/.secrets.toml`
- Environment prefix: `TFRLRL_`
- Validators ensure required fields and valid Gymnasium environment IDs
- Supports default/development/production environments
- Logging configuration: Centralized via `LOG_LEVEL` setting (DEBUG, INFO, WARN, ERROR)
  - Set via environment variable: `TFRLRL_LOG_LEVEL=DEBUG`
  - Set via settings files: `LOG_LEVEL = 'DEBUG'`
  - Applied globally in `src/tfrlrl/__init__.py`

### CLI (src/tfrlrl/cli/)

Command-line interface tools for common tasks:
- `sample.py`: CLI for sampling steps from Gymnasium environments
  - Uses argparse for argument parsing (no external dependencies)
  - Supports single or parallel (Ray) environment sampling
  - Prints summary statistics (total steps, episodes, mean/std/min/max rewards)
  - Inherits logging configuration from centralized settings (no --verbose flag)
- Entry point registered as `tfrlrl-sample` console script in pyproject.toml

## Code Style Requirements

- Maximum line length: 120 characters
- Maximum complexity: 8 (flake8)
- isort configuration: line_length=120, multi_line_output=3, include_trailing_comma=true
- Docstrings required for classes and functions (excluding D100, D104)
- Double quotes for strings (enforced by flake8-quotes)

## Testing Requirements

- Minimum 90% code coverage (enforced in CI)
- Tests run in random order (pytest-random-order)
- Branch coverage enabled
- Tests excluded from coverage reports

## Key Dependencies

- **Ray**: Distributed sampling across environments
- **Gymnasium**: RL environment interface (includes Atari, MuJoCo extras)
- **NumPy**: Array operations
- **Dynaconf**: Configuration management

## Pre-commit Hooks

Only isort is configured as a pre-commit hook. Run `make isort` or let pre-commit auto-format imports.
