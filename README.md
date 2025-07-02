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

Download these into `/datasets/raw`:
- Run `/datasets/fetch_raw.sh`, from https://datasets.imdbws.com/
  - `name.basics.tsv.gz`
  - `title.basics.tsv.gz`
  - `title.principals.tsv.gz`
  - `title.ratings.tsv.gz`
- From TMDB
  - Overviews, API scraped with `/datasets/add_overviews.py`
  