.PHONY: test baselines docker-build docker-test docker-baselines docker-up clean

# Local commands (using Python 3.11 environment)
test:
	python -m unittest discover -s tests

baselines:
	python -m oceanembed.train.train_baselines

validate:
	python -m oceanembed.data.validate_data

# Docker commands (smooth, isolated execution)
docker-build:
	docker compose build

docker-test:
	docker compose run --rm oceanembed python -m unittest discover -s tests

docker-baselines:
	docker compose run --rm train

docker-up:
	docker compose up api

clean:
	rm -rf __pycache__ .pytest_cache data/raw/*/.npz
