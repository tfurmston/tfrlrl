# A multi-stage docker build for the Python project, which takes much of its format from
# https://github.com/michaeloliverx/python-poetry-docker-example

FROM python:3.10-slim as python-base

ENV PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=ON \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYSETUP_PATH="/opt/pysetup"

ENV PATH="$POETRY_HOME/bin:$PATH"

# builder-base is used to build dependencies
FROM python-base as builder-base

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    build-essential

# Install Poetry - Respects $POETRY_VERSION & $POERTRY_POETRY_HOME
ENV POETRY_VERSION=1.3.2
RUN curl -sSL https://install.python-poetry.org | python

WORKDIR /app
COPY ./poetry.lock ./pyproject.toml ./
RUN poetry install --without dev

FROM builder-base as development

RUN poetry install --no-root
COPY . .
RUN poetry install

FROM builder-base as production

COPY . .
RUN poetry install
