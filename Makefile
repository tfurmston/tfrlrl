
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

bump_major:
	poetry run bumpversion major

bump_minor:
	poetry run bumpversion minor

bump_patch:
	poetry run bumpversion patch
