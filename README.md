# mYrA — My Research Assistant

mYrA is a single-user research workspace for organizing scientific papers, asking source-grounded questions, comparing literature, and keeping research notes. The web app and API run locally; answer generation uses the configured DeepSeek API, so local deployment still needs network access. No account login is provided: keep the app on loopback or behind a trusted private boundary.

## What works today

- PDF library, Docling ingestion, local BGE-M3 embeddings/reranking, hybrid retrieval, and page/quote citations.
- Conversational QA, paper reading, comparison, claim verification, discovery proposals, notes, and saved research artifacts.
- SQL-backed ingestion and assistant runs with progress, cancellation, approvals, and HTTP APIs.
- Optional GraphRAG, translation, and Langfuse observability overlays.

Translation and visual analysis still have known provider/document coverage limits; see `DEVELOPMENT_STATUS.md` (owner-supplied local file) and the feature plans when available. A checked-in directory or endpoint alone does not mean every planned workflow is complete.

## Existing installation

Use the already configured Docker images and local state. From the repository root:

```bash
docker compose ps --all
docker compose start
```

If containers need creation and their images are already available, use `docker compose up -d --no-build`. Open <http://127.0.0.1:3000>; API health is at <http://127.0.0.1:8000/health> and OpenAPI at <http://127.0.0.1:8000/docs>. Do not run a rebuild, install, or model download as routine startup. Rebuild only the service whose source image or dependency lock actually changed.

The base Compose file runs the API, web app, worker, and optional Neo4j service. Add `-f docker-compose.local.yml` to use the local PostgreSQL/pgvector and filesystem profile. The opt-in `cloud-data` profile keeps application processes local while using the configured Supabase/GCS adapters. Choose one data profile explicitly; never copy or migrate owner data automatically.

Optional overlays reuse the existing application services:

```bash
docker compose -f docker-compose.yml -f docker-compose.graph.yml up -d --no-build
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d --no-build
```

Translation has a separate worker and overlay (`docker-compose.translation.yml`). Follow `infrastructure/observability/README.md` for Langfuse initialization; translation availability depends on its isolated runtime and configured service. Do not expose these no-login services to the public internet. Langfuse is available at <http://127.0.0.1:3001> when its overlay is initialized and running.

## Development and checks

The repository uses Python 3.12/FastAPI with `uv`, and Next.js/TypeScript with npm. Read `AGENTS.md` and `apps/web/AGENTS.md` before making changes; local planning documents may be ignored and owner-supplied.

API checks from `apps/api/`:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Web checks from `apps/web/`:

```bash
npm run format:check
npm run lint
npm run typecheck
npm test
npm run test:e2e
npm run build
```

Use isolated databases and mocked providers for ordinary tests. Live cloud/model tests are opt-in and must stay within the documented approved scope. Never print or commit `.env` values, credentials, local databases, papers, or generated observability keys.

## Design and project status

`docs/SYSTEM_DESIGN.md` describes the current single-user, local-first architecture and distinguishes implemented behavior from optional or planned capabilities. `docs/ASSISTANT_HTTP_API.md` shows how to submit and inspect an assistant run without the browser. No license has been selected; do not treat this project as open source under an assumed license.
