PYTHON = python3
PIP = pip3

default:
	@make -s build-cpp

build-cpp:
	sh scripts/build_cpp.sh

build-cudaconda: 
	docker build $(FLAGS) -t ravihammond/cuda-conda -f dockerfiles/Dockerfile.cudaconda .

build-dev: 
	docker build $(FLAGS) -t ravihammond/hanabi-project:dev -f dockerfiles/Dockerfile.projectnew --target dev .

build-prod:
	docker build $(FLAGS) -t ravihammond/hanabi-project:prod -f dockerfiles/Dockerfile.projectnew --target prod .

push-prod:
	docker push ravihammond/hanabi-project:prod

run-old:
	bash scripts/run_docker.bash

run-dev:
	bash scripts/run_docker_dev.bash

run-prod:
	bash scripts/run_docker_prod.bash $(SCRIPT)

run-plot:
	bash scripts/run_docker_plot.bash

buildrun-prod:
	make -s build-prod
	make -s run-prod

jupyter:
	bash scripts/jupyter.bash
