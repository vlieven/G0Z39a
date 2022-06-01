FROM python:3.9-slim

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY datasets ./datasets

COPY src ./src
COPY setup.cfg ./setup.cfg
COPY pyproject.toml ./pyproject.toml
RUN pip install -e .

CMD gunicorn -b 0.0.0.0:80 src.nepal.app.app:server
