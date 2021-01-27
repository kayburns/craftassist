REMOTE=050625352140.dkr.ecr.us-east-2.amazonaws.com
COMMIT_SHA:=$(shell git show --name-only origin/orphan | head -1 | cut -d' ' -f2)
ARGS=${DOCKER_BUILD_ARGS}

all: build

build: craftassist

craftassist:
	docker build ${ARGS} ../../../ -f Dockerfile -t craftassist
	docker tag $@ $@:${COMMIT_SHA}
	docker tag $@ ${REMOTE}/$@
	docker tag $@ ${REMOTE}/$@:${COMMIT_SHA}

push-versioned:
	docker push ${REMOTE}/craftassist:${COMMIT_SHA}

pull:
	docker pull ${REMOTE}/craftassist
