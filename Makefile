install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv test_make_embeddings.py

format:
	black *.py


lint:
	pylint --disable=R,C make_embeddings.py

all: install lint test