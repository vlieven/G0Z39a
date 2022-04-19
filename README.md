# G0Z39a - Modern Data Analytics

This repository represents the project for the 2022 Modern Data Analytics
course at KU Leuven. The subject of this project is 'Covid 19 in the USA'.

## Development setup
Requirements are managed via [pip-tools](https://pip-tools.readthedocs.io/en/latest/),
which generates the `requirements.txt` file from `requirements.in`.

Using a [virtual environment](https://docs.python.org/3/library/venv.html) 
to isolate this project from others on your system is recommended.

To install the packages listed in the `requirements.txt`,
you can run `python -m pip install -r requirements.txt`.

To install the custom package ('mda_nepal') into your local environment,
you can run `python -m pip install -e .`.

### Python version
This project was developed using Python 3.9.10,
although different minor versions are likely to work.

### Code style
This project uses [black](https://github.com/psf/black)
and [isort](https://pycqa.github.io/isort/) as tools to help with consistent formatting.
