FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    libgomp1

COPY requirements-app.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY requirements-app-only.txt ./requirements-extra.txt
RUN pip install -r requirements-extra.txt

COPY datasets ./datasets

COPY src ./src
COPY setup.cfg ./setup.cfg
COPY pyproject.toml ./pyproject.toml
RUN pip install -e .

CMD gunicorn src.nepal.app.app:server \
    -b 0.0.0.0:80 \
    --timeout 240
