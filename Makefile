

install:
	python -m pip install --upgrade pip poetry
	poetry install --no-dev

install-dev:
	python -m pip install --upgrade pip poetry
	poetry install
