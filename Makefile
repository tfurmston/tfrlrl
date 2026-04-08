DOCKER_NAME ?= tfrlrl
DOCKER_DEVELOPMENT_NAME ?= tfrlrl_dev
DOCKER_TAG ?= latest

install:
	python -m pip install --upgrade pip
	poetry install

install-dev:
	python -m pip install --upgrade pip
	poetry install --with dev

check-style:
	poetry run ruff check

check-typing:
	poetry run mypy src/tfrlrl

format:
	poetry run ruff format

test:
	poetry run pytest --random-order tests/

test-local:
	poetry run pytest --random-order -m "not slow and not flaky" tests/

test-coverage:
	poetry run pytest --random-order --cov=tfrlrl --cov-config=setup.cfg tests/

docker-build:
	docker build --tag $(DOCKER_NAME):$(DOCKER_TAG) --target production --file docker/Dockerfile .

docker-build-dev:
	docker build --tag $(DOCKER_DEVELOPMENT_NAME):$(DOCKER_TAG) --target development --file docker/Dockerfile .

docker-check-style: docker-build-dev
	docker run $(DOCKER_DEVELOPMENT_NAME):$(DOCKER_TAG) /bin/bash -c 'make check-style'

docker-check-typing: docker-build-dev
	docker run $(DOCKER_DEVELOPMENT_NAME):$(DOCKER_TAG) /bin/bash -c 'make check-typing'

docker-test: docker-build-dev
	docker run $(DOCKER_DEVELOPMENT_NAME):$(DOCKER_TAG) /bin/bash -c 'make test'

docker-test-coverage: docker-build-dev
	docker run $(DOCKER_DEVELOPMENT_NAME):$(DOCKER_TAG) /bin/bash -c 'make test-coverage'

bump_major:
	poetry run bumpversion major

bump_minor:
	poetry run bumpversion minor

bump_patch:
	poetry run bumpversion patch
