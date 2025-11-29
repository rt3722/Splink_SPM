### **Infrastructure Plan: Content-Only Hybrid Entity Resolution POC**

---

#### **1. Objectives**

- Build an end-to-end pipeline that keeps all raw + processed profiles inside BigQuery (`acs-is-dsia-lod.SPM_TEST_DATA.SPM_SAMPLE_DATA`) while generating embeddings via `ML.GENERATE_EMBEDDING`.
- Generate Gemini embeddings in batch, persist normalized persona data, anchors, and vectors directly in BigQuery tables/views for serving.
- Deploy a Cloud Run application that lets an analyst pick any talent ID, treat it as an incoming candidate, and visualize duplicate detection (bounded candidate set + graph + merge panel).
- Keep everything inside BigQuery for the POC (no Cloud SQL), and keep networking/security/observability lightweight while flagging richer controls for v2.

---

#### **2. Target Architecture Overview**

| Layer | GCP Service / Tooling | Purpose |
| :--- | :--- | :--- |
| **Data Lake** | BigQuery (existing table) | Source of truth for raw SPM talent records. |
| **Staging / Export** | Cloud Storage (Parquet in regional bucket) | Intermediate snapshot for batch processing and replay. |
| **Processing & Embedding** | BigQuery remote model via `ML.GENERATE_EMBEDDING` + Vertex AI `text-embedding-005` | Generates embeddings and normalized text directly inside BigQuery SQL jobs. |
| **Operational Store** | BigQuery canonical + embedding tables | Holds normalized persona, anchors, and embedding vectors that the app reads at runtime. |
| **Secrets / Config** | POC: `.env` + Cloud Run env vars, single service account key in Secret Manager. (V2 adds Config Controller + fine-grained secrets.) |
| **Serving API & UI** | Cloud Run (containerized FastAPI backend + React/Next.js frontend) | Provides duplicate detection endpoint, candidate list, D3 graph, and merge panel UI. |
| **Networking / Security** | POC: Cloud Run with default egress + BigQuery access over Google backbone. (V2 adds VPC connectors, private service connect, IAM hardening.) |
| **Observability** | POC: default Cloud Logging/Monitoring only. (V2 adds SLO-based alerting + tracing.) |

---

#### **3. One-Time Embedding Backfill (BigQuery-Native)**

1. **Create the remote embedding model**
   - In the same region as the dataset (`US` multi-region works), run:
     ```sql
     CREATE OR REPLACE MODEL `spm_models.text_embedding`
     REMOTE WITH CONNECTION `us-vertex-conn`
     OPTIONS(endpoint = 'text-embedding-005');
     ```

2. **Materialize normalized text + anchors**
   - Use a BigQuery script to populate `SPM_SAMPLE_CANON` with normalized persona strings and anchor arrays (mirrors ADR Phase 0/1 logic implemented in SQL UDFs).

3. **Generate embeddings in 1k-row batches**
   - For each slice (e.g., `batch_id`, `mod` of hash), execute:
     ```sql
     INSERT INTO `acs-is-dsia-lod.SPM_TEST_DATA.SPM_SAMPLE_EMBEDS`
     SELECT
       source_id,
       embedding.embedding_values AS embedding_vec,
       CURRENT_TIMESTAMP() AS embedded_at
     FROM ML.GENERATE_EMBEDDING(
       MODEL `spm_models.text_embedding`,
       (
         SELECT source_id, persona_blob AS content
         FROM `acs-is-dsia-lod.SPM_TEST_DATA.SPM_SAMPLE_CANON`
         WHERE batch_token = 'slice_05'
       ),
       STRUCT(TRUE AS flatten_json_output)
     );
     ```
   - Repeat for all slices until every row is embedded.

4. **Verification + publish**
   - Run `SELECT COUNT(*)` on both `SPM_SAMPLE_CANON` and `SPM_SAMPLE_EMBEDS` to ensure parity.
   - Create a view `SPM_SAMPLE_CANON_WITH_EMBEDS` joining persona/anchors with the embedding vector; this becomes the serving table for Cloud Run.

---

#### **4. Embedding & Normalization (BigQuery Implementation)**

- **Normalization UDFs**: Implement SQL UDFs for `NormalizeName`, `NormalizeEmail`, `NormalizePhone`, etc., and apply them while writing to `SPM_SAMPLE_CANON`. Keeps the same logic as ADR Phase 0/1 without leaving BigQuery.
- **Embedding model**: Use the BigQuery remote model `spm_models.text_embedding` that proxies Gemini `text-embedding-005`. This keeps inference in-region and managed. *(Reference: BigQuery `ML.GENERATE_EMBEDDING` syntax docs.)*
- **Batching strategy**: Drive batches via BigQuery scripting—each script iteration processes ~1k rows (based on partition tokens) to stay within quotas and simplify retry.
- **Failure handling**: If `RESOURCE_EXHAUSTED` errors bubble up, rerun the batch slice; BigQuery already retries transient failures automatically.
- **Audit fields**: Capture `embedded_at`, `model_endpoint`, and `embedding_dim` columns in `SPM_SAMPLE_EMBEDS` for traceability and future re-embedding.

---

#### **5. BigQuery Serving Tables**

1. `SPM_SAMPLE_CANON`
   - Columns: `source_id`, `account_id`, normalized persona fields, `persona_blob` (free-form text used for embeddings), `emails ARRAY<STRING>`, `phones ARRAY<STRING>`, `batch_token`, `normalized_at`.
   - Partition: `DATE(normalized_at)` to keep maintenance easy.

2. `SPM_SAMPLE_EMBEDS`
   - Columns: `source_id`, `embedding_vec ARRAY<FLOAT64>`, `embedded_at`, `model_endpoint`, `embedding_dim`.
   - Cluster by `source_id` for fast joins.

3. `SPM_SAMPLE_CANON_WITH_EMBEDS` (view)
   - Joins the above tables; exposes one row per talent with persona, anchors, embedding, and payload IDs.
   - Adds computed columns for quick filtering (e.g., `primary_email`, `city_state`).

4. `SPM_SAMPLE_ANCHOR_INDEX` (optional table)
   - One row per anchor value → `source_id`, enabling fast exact-match lookups from Cloud Run without scanning arrays.

These tables live in `acs-is-dsia-lod.SPM_TEST_DATA` and act as the operational store for the duplicate-detection UI. No Postgres is required for the POC.

---

#### **6. Cloud Run Application Architecture**

**Backend (FastAPI or Node/Express)**

- Endpoints:
  - `GET /talent/{id}/candidates?limit=200`: runs a BigQuery SQL that unions anchor-exact matches with top vector neighbors using `ML.DISTANCE` (cosine). Limits to ~200 rows and returns similarity + shared anchors.
  - `GET /talent/{id}/graph`: pulls those candidates plus their anchors/vectors and builds the fully connected graph in memory (per ADR Phase 3).
  - `POST /talent/{id}/decision`: executes the Phase 4 policy using the graph edges and returns MERGE/REVIEW/INSERT result.
- Data access:
  - Uses the BigQuery Storage API client for low-latency reads.
  - Caches the requested talent row (embedding + anchors) in-memory for the duration of the request—no need to recompute embeddings.
- Optional caching: enable a small in-process LRU cache or Cloud Memorystore if analysts repeatedly query the same IDs.

**Frontend (Next.js/React)**

- **Left panel**: paginated grid (Material UI DataGrid) showing candidate list, similarity score, shared anchors, and quick filters.
- **Right panel**: D3.js or Cytoscape.js visualization of the fully connected graph (node size by weight, edge color by type).
- **Bottom panel**: decision summary with textual explanation, plus buttons to trigger follow-up workflows (manual merge, mark as distinct).

**Deployment & Scaling**

- Build/push docker image via Cloud Build; simple `gcloud run deploy` for POC.
- Cloud Run scales between 0–3 instances; allow unauthenticated invocations but protect with optional basic auth header.
- Environment variables store BigQuery dataset/table names and the remote model ID for transparency.

---

#### **7. Security & Secrets (POC Baseline)**

- **Service account**: reuse the existing project-level SA for both BigQuery SQL jobs and Cloud Run, granting BigQuery Data Editor + BigQuery Job User + Vertex AI Invoker.
- **Secrets**: one Secret Manager entry holds the service account key and any API tokens; load into Cloud Run env vars at deploy time.
- **Networking**: rely on default Google backbone connectivity (no VPC connector). Document that V2 will introduce PSC/private service access.
- **User access**: Cloud Run remains publicly reachable but protected via Basic Auth header; V2 migrates to IAP + per-user IAM.

---

#### **8. Monitoring & Reliability (POC Minimal)**

- Depend on built-in Cloud Logging/Monitoring; no custom dashboards or alert policies.
- After each embedding script execution, inspect the BigQuery job history to confirm row counts and catch `RESOURCE_EXHAUSTED` retries.
- Publish a simple `/healthz` endpoint in Cloud Run that executes a lightweight BigQuery query to ensure permissions/connectivity before use.

---

#### **9. Future-Proofing**

- **Incremental Loads**: Move the BigQuery scripts into scheduled queries or Dataform so new records are normalized + embedded automatically.
- **Low-latency Store**: If analyst traffic grows, replicate the canonical table into Cloud SQL or AlloyDB with pgvector for sub-100 ms lookups, fed by BigQuery change streams.
- **Human-in-the-loop Workflow**: Integrate bottom-panel actions with a Case Management system (Firestore + Workflows) to track manual reviews and merge approvals.
- **Model Iteration**: Version the remote model ID and keep `model_version` columns to support backfills with improved embeddings.
- **Cost Controls**: Use slot reservations / autoscaling for BigQuery, Cloud Run concurrency tuning, and budget alerts to monitor Vertex inference spend.

---

This infrastructure plan keeps the solution grounded in managed GCP services, ensures the same data semantics as the ADR, and sets up a clear path from raw data to analyst-facing tooling without placeholders or mock layers.

