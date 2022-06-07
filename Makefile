# Note: Make is a unix tool, this file is likely useless on Windows

requirements:
	pip-compile requirements-app.in
	pip-compile requirements-app-only.in
	pip-compile requirements.in
	pip-sync requirements-app.txt requirements-app-only.txt requirements.txt
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

deploy:
	heroku container:push web -a mda-nepal
	heroku container:release web -a mda-nepal
