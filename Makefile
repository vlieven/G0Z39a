# Note: Make is a unix tool, this file is likely useless on Windows

requirements:
	pip-compile requirements.in
	pip-sync requirements.txt
	pip install -e .

quality:
	black .
	isort .
	mypy .
	pytest -v

build:
	docker build -t nepal .

run:
	docker run -p 8080:80 nepal
