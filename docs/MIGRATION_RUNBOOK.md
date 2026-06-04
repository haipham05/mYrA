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
| `e6f7a8b9c0d1` | Single-User Persistence | Adds `is_archived` to `conversations`, `model_name` and `token_count` to `messages` | Safe defaults (`is_archived=False`, nullable metadata) |
| `a1b2c3d4e5f6` | Conversation Summary | Adds `summary` (`Text`) to `conversations` | Nullable; optional metadata |
| `f1a2b3c4d5e6` | Long-Term Memory | Adds `memories`, `memory_sources`, `memory_audits` tables and indexes | Additive tables with CASCADE delete; legacy queries unaffected |

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
2. Seed it with fixture projects, papers, elements, chunks, and conversations at a predecessor revision (e.g. `e6f7a8b9c0d1`).
3. Run intermediate upgrade (`alembic upgrade a1b2c3d4e5f6`) and verify new columns default properly.
4. Run upgrade to head (`alembic upgrade head`) and verify new memory tables are functional.
5. Verify data integrity:
   - Existing rows and relations remain unchanged across every revision transition.
   - Exact text citations and element bounding boxes remain intact.
   - Automated check: `pytest tests/test_migration_rehearsal.py` passes.

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

1. **Application Code Rollback & Schema Compatibility Policy**:
   The application startup check (`check_schema_compatibility` in `apps/api/app/db/compatibility.py`) requires that the database's current Alembic revision is present in the application's expected heads (`current_rev in expected_heads`).
   - If rolling back to an older container image that lacks newer Alembic scripts, the startup gate will fail fast with `IncompatibleSchemaError`.
   - **Tested Rollback Procedure**: To roll back application code while preserving additive database columns/tables:
     - Ensure the rollback image includes the updated Alembic version scripts (decoupling schema metadata from code logic), OR
     - If deploying an older binary where migration files cannot be refreshed, temporarily set `CHECK_MIGRATION_COMPATIBILITY=false` in the container environment, verifying that all database changes are strictly additive (nullable columns or safe defaults).
   - **Database Downgrade Policy**: Live production databases must NEVER be downgraded via `alembic downgrade` unless emergency restoration is explicitly authorized, because downgrading drops new user data (e.g. long-term memories or conversation summaries).

2. **Failed Backfill Recovery**:
   If an asynchronous backfill job encounters errors, it can be canceled or retried via `POST /api/v1/jobs/{job_id}/retry` or the resource reconciler `POST /api/v1/system/reconcile`. Existing user data is never corrupted or dropped by an incomplete backfill.

3. **Catastrophic Failure Recovery**:
   If live database corruption occurs, restore from the pre-migration snapshot taken in Step 1:
   ```bash
   pg_restore --clean --if-exists -d "$DATABASE_URL" myra_backup_<timestamp>.dump
   ```
