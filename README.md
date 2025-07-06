# ReelRaider
"Swipe, Stream, Plunder."

Fully personalized movie recommendations, powered by badly trained AI.

There's nothing intelligent about it.

It's just crappy machine learning but AI is a buzzword we have to use nowadays.

## Structure
Vibecoded excuse of a codebase. I'm ashamed.
- `src/static/` — static content (html, css, js)
- `src/server/` — fastify, nodejs
- `flake.nix`, `Dockerfile`, `docker-compose.yml` - config stuff

## Usage
Use docker.

```bash
docker-compose up --build
```

App will be available at http://localhost:8000 or whatever

## Datasets
See the full [datasets/README.md](datasets/README.md) for details on data sources, legal notices, usage instructions, and requirements.

## LICENSE
Credits:
- copilot, sonnet 4
- me
- omdbapi
See the [LICENSE](./LICENSE) at the root of this repository.