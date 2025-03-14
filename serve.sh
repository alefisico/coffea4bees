#!/usr/bin/env bash

action() {
    local port="${1:-8000}"
    local host="${2:-127.0.0.1}"
    local image="${3:-ghcr.io/cms-cat/mkdocs-material}"

    docker run \
        --rm -it \
        -p ${host}:${port}:8000 \
        -v "${PWD}":/docs \
        -e GIT_PYTHON_REFRESH=quiet \
        ${image}
}
action "$@"
