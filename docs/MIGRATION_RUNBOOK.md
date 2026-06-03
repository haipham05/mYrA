# Migration Runbook & Reversibility Strategy — mYrA

> Operational runbook for database migrations and schema lifecycle management in mYrA.
> Aligned with Milestone 3 requirements (3.2, 3.A1, 3.A2, 3.A3).

---

## 1. Core Principles

1. **Forward-Only Additive Migrations**:
   Schema evolution is strictly forward-only and additive. Never introduce breaking changes (dropping tables, dropping columns, renaming active columns in lockstep) on shared or live storage holding user research papers and citation provenance.

2. **Decoupled Code Rollback (No Shared Database Downgrades)**:
   New columns and tables must be nullable or have safe defaults so older application revisions can safely read from or ignore them. Rolling back application code to a prior container image must leave migrated data fully readable and intact without requiring a destructive database downgrade.

3. **Startup Authority & Gatekeeping**:
   Neither the API nor the worker executes `Base.metadata.create_all()` on deployment boot. Instead, `check_schema_compatibility(engine)` runs during startup lifespan to verify that the connected database matches the checked-in Alembic `head` revision. Incompatible or unmigrated databases fail fast with non-secret diagnostics before accepting traffic or claiming jobs.

---

## 2. Ordered Migration Inventory

| Revision ID | Description | Additive Changes | Rollback Safety |
| :--- | :--- | :--- | :--- |
| `a6a79bdff6de` | Initial Schema | `projects`, `papers`, `paper_pages`, `paper_elements`, `paper_chunks`, `chunk_elements`, `jobs`, `conversations`, `messages` | Baseline schema |
| `b7f1e92d8301` | Document SHA-256 | Adds `document_sha256` (String 64) to `papers` for tamper detection and idempotency | Nullable; older code reads/ignores cleanly |
| `c1a2e34d5678` | Vector & Full-Text Search | Adds `embedding_vec` (`halfvec(1024)`) and `tsv_content` (`tsvector`) to `paper_chunks` | Nullable/generated; legacy JSON embedding preserved |
| `d4e5f6a7b8c9` | Page Raw Text | Adds `raw_text` (`Text`) to `paper_pages` for independent DOM anchor validation | Nullable; fallback to element aggregation |
| `a1b2c3d4e5f6` | Conversation Summary | Adds `summary` (`Text`) to `conversations` | Nullable; optional metadata |
| `e6f7a8b9c0d1` | Single-User Persistence | Adds `is_archived` to `conversations`, `model_name` and `token_count` to `messages` | Safe defaults (`is_archived=False`, nullable metadata) |

---

## 3. Migration Rehearsal & Deployment Procedure

### Step 1: Pre-Migration Backup
Before applying any migration to the production Supabase database:
```bash
# Capture full schema and data dump
pg_dump "$DATABASE_URL" --format=custom --file=myra_backup_$(date +%Y%m%d_%H%M%S).dump
```

### Step 2: Disposable Database Rehearsal
Always test the migration against an isolated, disposable database before touching live infrastructure:
1. Initialize an empty disposable database.
2. Seed it with fixture projects, papers, elements, chunks, and conversations.
3. Run Alembic upgrade:
   ```bash
   alembic upgrade head
   ```
4. Verify data integrity:
   - Existing rows and relations remain unchanged.
   - New columns default properly.
   - Exact text citations and element bounding boxes remain intact.

### Step 3: Production Upgrade
Apply the migration using Alembic CLI:
```bash
alembic upgrade head
```

### Step 4: Verification
Confirm database readiness via API health probe:
```bash
curl http://127.0.0.1:8000/api/v1/health/ready
# Expected: {"status": "ready", "database": "connected", "storage": "available", "schema": "compatible", ...}
```

---

## 4. Rollback and Disaster Recovery Strategy

If an application release must be rolled back:

1. **Application Code Rollback**:
   Revert the application container or process to the prior stable release image. Because all migrations are additive and backward-compatible, the older application code will function normally against the newer database schema.

2. **Failed Backfill Recovery**:
   If an asynchronous backfill job encounters errors, it can be canceled or retried via `POST /api/v1/jobs/{job_id}/retry` or the resource reconciler `POST /api/v1/system/reconcile`. Existing user data is never corrupted or dropped by an incomplete backfill.

3. **Catastrophic Failure Recovery**:
   If live database corruption occurs, restore from the pre-migration snapshot taken in Step 1:
   ```bash
   pg_restore --clean --if-exists -d "$DATABASE_URL" myra_backup_<timestamp>.dump
   ```
