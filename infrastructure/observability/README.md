# Local observability stack

This opt-in overlay adds a self-hosted Langfuse 4.48.0 instance and a separate
Redis cache for mYrA. It does not change the base Compose stack or `.env`.
Langfuse's PostgreSQL, ClickHouse, MinIO, and Redis use their own persistent
volumes and a private Docker network. The application cache is isolated on a
different internal network, capped at 256 MiB, configured for `allkeys-lru`,
and has persistence disabled. The only published observability port is the
loopback-only console at <http://127.0.0.1:3001>.

## Initialize private files

From the repository root, run:

```sh
python3 infrastructure/observability/setup.py
```

The script creates `.local/observability/` with mode `0700` and private
credential/environment files with mode `0600`. It generates credentials once;
subsequent runs preserve them and do not print secret values. The directory is
local-only and must remain ignored by Git. Do not copy these files into `.env`,
logs, issue reports, or chat. If the credentials are lost, keep existing data
volumes intact and follow the documented credential-rotation procedure before
recreating them; deleting volumes is not a setup step.

The generated `credentials.json` holds the console login and personal project
keys. Retrieve them locally when opening the console; do not paste them into
chat. `app.env` wires the personal project keys and app Redis URL to API/worker.
The application stays on the personal project by default.

## Start and inspect

The setup files are required by Compose; startup fails if they are missing.
After setup, validate and start the overlay without rebuilding application
images:

```sh
docker compose -f docker-compose.yml -f docker-compose.observability.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d --no-build langfuse-postgres langfuse-clickhouse langfuse-minio langfuse-redis langfuse-web langfuse-worker myra-cache langfuse-retention
docker compose -f docker-compose.yml -f docker-compose.observability.yml ps --all
```

The explicit service list avoids starting or rebuilding the existing mYrA app,
web, worker, and Neo4j. The retention service waits for a healthy Langfuse console,
runs a bounded cleanup pass hourly after success, retries failures every minute,
and persists only progress metadata in `.local/observability/retention/`; it uses
only the personal project credentials. Cleanup runs independently of tracing:
missing, stale, or failed cleanup does not suppress selected inputs or outputs.
Sanitization and the 128 KiB observation limit still apply. Cleanup failures can
extend trace retention beyond three days; inspect its status separately.
Existing empty observations cannot be backfilled; new requests capture the selected
payloads. The scheduler reads credentials from its private mount and writes status
to the separate directory. Use `docker compose ... logs --tail=100 langfuse-web
langfuse-worker langfuse-retention` for metadata-only diagnostics; never publish
environment output. The overlay reuses the repository's base default network
only for app-to-console access. Langfuse's database/object-store/queue network
and the mYrA cache network are internal-only. No app cache data is persistent
or a source of truth.

The generated `LANGFUSE_INIT_*` variables create one personal Langfuse project
on first startup. Normal use and explicitly enabled live tests both send traces
to that project; mark validation traces with `test_run=true` so they can be
filtered in the console. Do not create or configure a synthetic Langfuse
project, and do not put captured research material in a persistent evaluation
dataset.

This Compose deployment is for local development, not high availability. Keep
the host-bound URL private and do not expose port 3001 or the unauthenticated
mYrA app publicly.

## Pinned images

Manifest digests were resolved from registries on 2026-10-01 without pulling
images. Compose pins these immutable multi-platform manifest digests:

| Service | Version/tag | Manifest digest |
| --- | --- | --- |
| Langfuse web | `4.48.0` | `sha256:8c1b80ed7735be587974d603af0f6e0247b33d3b56c7d5ab9d2337aec313efa0` |
| Langfuse worker | `4.48.0` | `sha256:9349b003a453326b3d2e2033eb94fe5ca9ee12ee237779c10984b9d3dfe0f8ed` |
| PostgreSQL | `17.6-alpine` | `sha256:ef257d85f76e48da1c64832459b59fcaba1a4dac97bf5d7450c77753542eee94` |
| ClickHouse | `25.12.3.21-alpine` | `sha256:74da41cd61db84f652c6364fd30d59e19b7276d34f7c82515f5f0e70d6f325da` |
| Redis | `7.4.7-alpine` | `sha256:02f2cc4882f8bf87c79a220ac958f58c700bdec0dfb9b9ea61b62fb0e8f1bfcf` |
| Chainguard MinIO | immutable manifest | `sha256:4692462f35d97d7e82c30371d82f057703c5d9489bcae726010594c812f2d285` |

The upstream self-hosted Compose guide cautions that Compose is not a highly
available deployment. Langfuse's own initialization docs define a single
`LANGFUSE_INIT_PROJECT_*` project, while its management API docs mark
multi-project programmatic administration as Enterprise. See the [Compose
guide](https://langfuse.com/self-hosting/deployment/docker-compose),
[headless initialization](https://langfuse.com/self-hosting/administration/headless-initialization),
and [instance-management API](https://langfuse.com/self-hosting/administration/instance-management-api).
