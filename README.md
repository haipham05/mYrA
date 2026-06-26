# mYrA — My Research Assistant

mYrA is an early development scaffold for a research assistant. The current application contains a FastAPI health endpoint and a Next.js page that reports whether the API is reachable. Product design and milestone plans are maintained as local, untracked documents.

## Prerequisites

- Docker with the Compose plugin for the two-service setup.
- For host development: Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js 22 with npm.

## Run with Docker

From the repository root:

```bash
docker compose up --build
```

Open <http://localhost:3000>. The page should show **API: Connected**. The API health endpoint is <http://localhost:8000/health> and returns `{"status":"ok","service":"myra-api"}`. Stop the foreground Compose process with Ctrl+C.

## Enable GraphRAG on an existing installation

With configured Supabase/GCS/DeepSeek, the current schema, existing images, and
cached BGE weights, reuse the API, worker and local Neo4j:

```bash
docker compose -f docker-compose.yml -f docker-compose.graph.yml up -d --no-build
```

Open a project's Graph Explorer from the workspace. Indexing calls DeepSeek and
stores only source-verified facts; unsupported candidates are rejected. This
profile reads at most eight child chunks per paper, so an empty result does not
prove the paper has no relationships. It adds no services or public ports and
does not edit `.env`. Use both Compose files for subsequent starts/recreation;
running only the base file restores its configured feature setting. Models must
already be cached because the profile loads them offline.

## Run on the host

In one terminal, start the API:

```bash
cd apps/api
uv sync --all-groups
uv run uvicorn app.main:app --reload
```

In another terminal, start the web app:

```bash
cd apps/web
npm ci
npm run dev
```

The frontend uses `http://localhost:8000` by default. To change Compose settings, copy `.env.example` to `.env`; Compose loads it for the API and passes `NEXT_PUBLIC_API_URL` to the web container. For host development, export API settings in the API terminal and put browser-visible settings in `apps/web/.env.local`. Keep credentials out of Git. The cloud variables are placeholders until database and storage integration is implemented.

## Validate changes

Run these commands from `apps/api/`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

When API dependencies change, update `uv.lock` and regenerate the image's pinned requirements from `apps/api/`:

```bash
uv lock
uv export --frozen --no-dev --no-emit-project --format requirements.txt --output-file requirements.txt
```

Run these commands from `apps/web/`:

```bash
npm run format:check
npm run lint
npm run typecheck
npm test
npm run build
```

## Cloud resources

The Milestone 0 hybrid development profile uses a Supabase PostgreSQL project with the `vector` extension and a private Google Cloud Storage bucket. Their identifiers belong in local configuration; see `.env.example` for the required variable names. No database schema or object-storage adapter is part of this scaffold.

No license has been selected yet; do not treat this repository as open source until one is added.
