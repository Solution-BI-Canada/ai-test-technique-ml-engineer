test:
	python -m pytest

train:
	python src/training.py

api:
	uvicorn src.api:app --reload

docker-build:
	docker build -t stockout-api:test .

docker-run:
	docker run -p 8000:8000 stockout-api:test