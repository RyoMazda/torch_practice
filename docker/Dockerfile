FROM python:3.9.1-slim-buster as base
WORKDIR /work
RUN apt-get update && apt-get install -y \
    curl unzip \
  && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
    | python - --version 1.1.4
ENV PATH="/root/.poetry/bin:${PATH}"
RUN poetry config virtualenvs.create false
# For speed up of updating the later layers
RUN pip install 'torch>=1.7.1'


FROM base as test
COPY pyproject.toml poetry.lock ./
COPY pigimaru ./pigimaru
RUN poetry install
COPY tests ./tests
CMD poetry run pytest -svx --cov pigimaru


FROM base as train
COPY pyproject.toml poetry.lock ./
COPY pigimaru ./pigimaru
RUN poetry install -E train --no-dev
# cache the model
RUN poetry run python -c "import transformers; name = 'cl-tohoku/bert-base-japanese-whole-word-masking'; transformers.BertModel.from_pretrained(name); transformers.BertJapaneseTokenizer.from_pretrained(name);"


FROM base as jupyter
# # For jupyterlab extension
# # https://github.com/nodesource/distributions/blob/master/README.md
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get update && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*
# For matplotlib Japanese font
RUN apt-get update && apt-get install -y \
    fonts-ipafont
COPY docker/matplotlibrc /root/.config/matplotlib/matplotlibrc
# Install dependencies
COPY pyproject.toml poetry.lock ./
COPY pigimaru ./pigimaru
RUN poetry install -E jupyter
# https://github.com/jupyterlab/jupyterlab-toc
# Anyone using JupyterLab < 3.0 should use this extension, but anyone using JupyterLab >= 3.0 will already have this extension installed with the core JupyterLab extensions.
RUN jupyter labextension install @jupyterlab/toc
# jupyter settings
COPY docker/jupyter_lab_config.py docker/overrides.json /root/.jupyter/
EXPOSE 8888
CMD jupyter lab --watch
