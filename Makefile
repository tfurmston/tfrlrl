DOCKER_NAME ?= tfrlrl
DOCKER_DEVELOPMENT_NAME ?= tfrlrl_dev
DOCKER_TAG ?= latest

install:
	python -m pip install --upgrade pip poetry
	poetry install --no-dev

install-dev:
	python -m pip install --upgrade pip poetry
	poetry install

check-style:
	poetry run flake8 src/tfrlrl tests

isort:
	poetry run isort .

test:
	poetry run pytest --random-order tests/

test-coverage:
	poetry run pytest --random-order --cov=tfrlrl --cov-config=setup.cfg tests/

docker-build:
	docker build --tag $(DOCKER_NAME):$(DOCKER_TAG) --target production --file docker/Dockerfile .

docker-build-dev:
	docker build --tag $(DOCKER_DEVELOPMENT_NAME):$(DOCKER_TAG) --target development --file docker/Dockerfile .

docker-check-style: docker-build-dev
	docker run $(DOCKER_DEVELOPMENT_NAME):$(DOCKER_TAG) /bin/bash -c 'make check-style'

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
