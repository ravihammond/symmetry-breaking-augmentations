PYTHON = python3
PIP = pip3

build-cudaconda: 
	docker build -t ravihammond/cudaconda -f dockerfiles/Dockerfile.cudaconda .

build-dev: 
	docker build -t ravihammond/hanabi-project:dev -f dockerfiles/Dockerfile.projectnew --target dev .

build-prod:
	docker build -t ravihammond/hanabi-project:prod -f dockerfiles/Dockerfile.projectnew --target prod .

push-prod:
	docker push ravihammond/hanabi-project:prod

run-dev:
	bash scripts/run_docker_dev.bash

run-prod:
	bash scripts/run_docker_prod.bash

run-plot:
	bash scripts/run_docker_plot.bash

jupyter:
	bash scripts/jupyter.bash
