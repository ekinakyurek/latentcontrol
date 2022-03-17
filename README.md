# Multi LM

## Installation
```SHELL
    DOCKER_BUILDKIT=1 docker compose up -d --build full
    DOCKER_BUILDKIT=1 make all-full $(tr '\n' ' ' < .env)
```