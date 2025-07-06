# Copilot Instructions for ReelRaider

## Core Commands
- **Build (TypeScript):**
  - `npm run build` — Build both server and static assets.
  - `npm run build:server` — Build server TypeScript (Fastify backend).
  - `npm run build:static` — Build static TypeScript (frontend).
  - `npm run postbuild:static` — Copy static assets to `dist/static/` and remove `.ts` files.
- **Start:**
  - `npm start` — Run the server from built output.
- **Docker:**
  - `docker-compose up --build` — Build and run the full stack (recommended for local use).

## High-Level Architecture
- **Frontend:** Static HTML/CSS/TS in `src/static/`, bundled to `dist/static/`.
- **Backend:** Fastify server (`src/server/index.ts`) serves static files and API endpoints.
- **Datasets:** Python scripts in `datasets/` for data processing, embedding, and OMDb API integration.
  - Uses OMDb API for movie metadata (see `datasets/omdb.py`).
  - IMDb data is used for filtering and ID mapping only.
- **Data Storage:**
  - Processed data and checkpoints are output to `dist/` (see `datasets/README.md`).
- **Config:**
  - Nix (`flake.nix`), Docker (`Dockerfile`, `docker-compose.yml`), TypeScript configs (`tsconfig.*.json`).

## Style & Coding Rules
- **TypeScript:**
  - Strict mode enabled (`strict: true` in `tsconfig.*.json`).
  - Use ES2022/ESNext modules, ES module imports.
  - Prefer explicit types and interfaces.
  - No lint/format scripts detected; follow idiomatic TypeScript/Node.js style.
- **Python:**
  - Use Polars, not Pandas, for data processing (see `datasets/README.md`).
  - Use python 3.12 or later.
  - All imports at the top of each script, no local imports.
  - Type hints used throughout (e.g., `pl.LazyFrame | pl.DataFrame`).
  - Logging via `logger` (see `datasets/utils.py`), not print.
  - Use assertions and exceptions for error handling.
  - Data processing is modular: each script has a focused responsibility.
  - Run scripts from the root directory as a module (e.g., `python -m datasets.ratings --help`).
- **General:**
  - No custom naming or import rules detected.
  - No linting or formatting tools configured; follow standard conventions for each language.

## Agent/AI Rules
- No `.cursor`, `.cursorrules`, `AGENTS.md`, or similar agent rule files found.
- No repo-specific Copilot or Claude rules detected.
- If adding agent instructions, keep them concise and repo-specific.

## Docs & References
- See `README.md` (root) for project overview and usage.
- See `datasets/README.md` for data pipeline and legal notes.
- License: See `LICENSE` at repo root.

---
For more, see [https://aka.ms/vscode-instructions-docs](https://aka.ms/vscode-instructions-docs).
