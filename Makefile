
install:
	python -m pip install --upgrade pip poetry
	poetry install --no-dev

install-dev:
	python -m pip install --upgrade pip poetry
	poetry install

check-style:
	poetry run flake8 src/tfrlrl tests

test:
	poetry run pytest --random-order tests/

test-coverage:
	poetry run pytest --random-order --cov=tfrlrl --cov-config=setup.cfg tests/
