# Headless Assistant API

The same assistant run and approval services used by the web UI are available over HTTP. These examples use a local instance; replace the IDs with a project, conversation, and READY paper that already exist in your installation. No browser session or account login is involved. Keep this unauthenticated service bound to loopback or a trusted private network.

Set a local API URL and IDs in your shell:

```sh
API=http://127.0.0.1:8000
PROJECT_ID=<project-uuid>
CONVERSATION_ID=<conversation-uuid>
PAPER_ID=<ready-paper-uuid>
```

## Submit and poll a scoped question

Submit a request with a unique idempotency key. `202 Accepted` means the run is queued, not finished. Use the returned run ID to poll until a terminal state.

```sh
curl -sS -X POST "$API/api/v1/conversations/$CONVERSATION_ID/runs" \
  -H 'Content-Type: application/json' \
  -d "{\"message\":\"What limitation does this paper report?\",\"project_id\":\"$PROJECT_ID\",\"conversation_id\":\"$CONVERSATION_ID\",\"scope\":\"paper\",\"selected_paper_ids\":[\"$PAPER_ID\"],\"intent_override\":\"qa\",\"idempotency_key\":\"cli-$(date +%s)-$RANDOM\"}"

curl -sS "$API/api/v1/runs/<run-id>"
```

Poll `GET /api/v1/runs/{run_id}` while status is `QUEUED` or `RUNNING`. Inspect `result.display_text`, `result.citations`, `result.structured_payload`, `warnings`, and `usage`. Citation objects include paper ID, page, quote, document hash, and any verified text anchors. Unknown provider usage stays empty/unknown rather than being reported as zero.

Other useful states are `NEEDS_INPUT`, `AWAITING_APPROVAL`, `SUCCEEDED`, `FAILED`, and `CANCELLED`. `NEEDS_INPUT` can be resumed with `POST /api/v1/runs/{run_id}/resume` and body `{"additional_input":"..."}`.

## Review an approval

For a run awaiting a persistent action, read its proposals, then explicitly approve or reject the exact proposal:

```sh
curl -sS "$API/api/v1/runs/<run-id>/actions"
curl -sS -X POST "$API/api/v1/actions/<action-id>/approve"
# or: POST /api/v1/actions/<action-id>/reject
```

Approval requeues eligible work. Poll the same run ID; completed work is recovered instead of blindly repeated. Discovery import proposals use `POST /api/v1/runs/{run_id}/discovery-import-proposals` with the selected catalog candidate, then the same approve/reject routes.

## Cancel and save/export research

```sh
curl -sS -X POST "$API/api/v1/runs/<run-id>/cancel"
curl -sS -X POST "$API/api/v1/runs/<completed-run-id>/artifact" \
  -H 'Content-Type: application/json' -d '{"title":"Method review"}'
curl -sS "$API/api/v1/projects/$PROJECT_ID/artifacts"
curl -sS "$API/api/v1/projects/$PROJECT_ID/artifacts/<artifact-id>"
curl -L "$API/api/v1/projects/$PROJECT_ID/artifacts/<artifact-id>/export?format=markdown" \
  -o research-notes.md
```

Artifact export formats are `markdown`, `csv` (comparison artifacts), and `bibtex`. A saved artifact is a versioned, source-linked record; a draft is not saved automatically. The OpenAPI reference at `/docs` lists request schemas, response fields, and status codes.

## Failure and privacy notes

Use the returned safe error/status to decide whether to clarify, retry, resume, or stop. A client timeout does not prove the worker failed; poll the same run. Do not expose this single-user API publicly without a separately designed trusted access boundary. Never put credentials in request bodies or share real research text in public issue reports.
