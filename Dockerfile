FROM python:3.9-slim

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    libgomp1

# install dependencies
COPY requirements-app.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY requirements-app-only.txt ./requirements-extra.txt
RUN pip install -r requirements-extra.txt

COPY datasets/reduced ./datasets/reduced

COPY src ./src
COPY setup.cfg ./setup.cfg
COPY pyproject.toml ./pyproject.toml
RUN pip install -e .

# add and run as non-root user
RUN useradd -r app
USER app

CMD gunicorn src.nepal.app.app:server \
    --bind 0.0.0.0:$PORT \
    --timeout 60
