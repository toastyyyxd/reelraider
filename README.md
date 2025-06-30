# CineVerse Static Server

## Structure

- `src/static/` — All static content (HTML, CSS, JS)
- `src/server/` — Fastify Node.js server
- `flake.nix`, `Dockerfile`, `docker-compose.yml` — Top-level configuration

## Usage

### Local (with Docker)

```bash
docker-compose up --build
```

App will be available at http://localhost:8000

### Development
- Edit static files in `src/static/`
- Edit server code in `src/server/`
