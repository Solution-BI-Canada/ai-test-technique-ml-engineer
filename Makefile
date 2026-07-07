.PHONY: install train test lint run docker-build docker-run

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	python -m src.training

test:
	pytest tests -v

lint:
	ruff check src tests

run:
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t stockout-api:test .

docker-run:
	docker run --rm -p 8000:8000 stockout-api:test