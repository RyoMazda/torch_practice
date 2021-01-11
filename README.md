# torch practice

## Installation
```sh
pip install git+https://github.com/ryomazda/torch_practice.git
```


## Running Unit Tests
```sh
poetry run pytest -svx --cov pigimaru
# or
docker-compose up --build test
```

Using GPU
```sh
docker build -f docker/Dockerfile.gpu -t gpu-test --target test .
docker run gpu-test
```


## Training
Local (just for test)
```sh
poetry install -E train --no-dev
poetry run train \
  --debug \
  path/to/train.tsv \
  path/to/valid.tsv \
  path/to/test.tsv
```

CPU docker (just for test)
```sh
# Only when you wanna update the dependency
docker-compose build train
# Do this every time
docker-compose run --rm train \
  poetry run train \
    --debug \
    path/to/train.tsv \
    path/to/valid.tsv \
    path/to/test.tsv
```

Using GPU
```sh
docker build -f docker/Dockerfile.gpu -t gpu-train --target train .
docker run -it -v $PWD:/work gpu-train poetry run train --debug \
  path/to/train.tsv path/to/valid.tsv path/to/test.tsv
```


## Running jupyter server
For ad hoc analysis & development.
[notebooks](notebooks/) are supposed to be executed through this jupyter server.

```sh
poetry install -E jupyter
poetry run jupyter lab
# or
docker-compose up --build jupyter
# Open `http://localhot:8888` and put "password" as the password.
```
