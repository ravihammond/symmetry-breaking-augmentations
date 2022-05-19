PYTHON = python3
PIP = pip3

build-cuda: 
	docker build -t ravihammond/cuda -f dockerfiles/Dockerfile.cuda .

build-conda: 
	docker build -t ravihammond/conda -f dockerfiles/Dockerfile.conda .

build-project: 
	docker build -t ravihammond/obl-project-temp -f dockerfiles/Dockerfile.project .

build-cudaconda: 
	docker build -t ravihammond/cudaconda -f dockerfiles/Dockerfile.cudaconda .

build-dev: 
	docker build -t ravihammond/hanabi-project:dev -f dockerfiles/Dockerfile.projectnew --target dev .

build-prod:
	docker build -t ravihammond/hanabi-project:prod -f dockerfiles/Dockerfile.projectnew --target prod .

build-all: build-cuda build-conda build-project

push-prod:
	docker push ravihammond/hanabi-project:prod

run:
	bash scripts/run_docker.bash

run-dev:
	bash scripts/run_docker_dev.bash

run-prod:
	bash scripts/run_docker_prod.bash

run-plot:
	bash scripts/run_docker_plot.bash

jupyter:
	bash scripts/jupyter.bash

build-cuda-temp: 
	docker build --rm=false -t ravihammond/cuda-temp -f dockerfiles/Dockerfile.cuda .

build-conda-temp: 
	docker build --rm=false -t ravihammond/conda-temp -f dockerfiles/Dockerfile.conda .

build-project-temp: 
	docker build --rm=false -t ravihammond/obl-project-temp -f dockerfiles/Dockerfile.project .

run-temp:
	bash scripts/run_docker_temp.bash

