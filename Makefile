PYTHON = python3
PIP = pip3

build-cuda: 
	docker build -t ravihammond/cuda -f dockerfiles/Dockerfile.cuda .

build-conda: 
	docker build -t ravihammond/conda -f dockerfiles/Dockerfile.conda .

build-project: 
	docker build -t ravihammond/obl-project -f dockerfiles/Dockerfile.project .

build-all: build-cuda build-conda build-project

run:
	bash scripts/run_docker.bash

jupyter:
	bash scripts/jupyter.bash
