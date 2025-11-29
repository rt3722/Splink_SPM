### **Architecture Decision Record v2: Content-Only Hybrid Entity Resolution (Weighted Graph + Policy Layer)**

#### **1. Context (Same Business Problem, Stronger Operational Guarantees)**

We keep the same core challenge:

- **Random Synthetic IDs:** `ID` and `ACCOUNTID` are arbitrary and cannot be used for linkage.
- **Goal:** Decide whether two random IDs actually refer to the **same human**, using only observable content.
- **Constraints:** We continue to rely only on the existing data columns from the original plan.

The v1 design correctly separates:

- **Similarity (Vector Persona)** vs
- **Identity (Graph over hard anchors)**.

However, v1 leaves some edge cases under-specified:

- How to **weight** conflicting signals (e.g., same vector, different phones).
- How to **scale** beyond in-memory graphs.
- How to translate graph structure into a **deterministic merge / review / insert policy**.

ADR v2 keeps the original intuition but adds:

- A **scored, typed-edge graph model**.
- An explicit **decision policy layer** on top of the graph.
- Clear guidance on **precision-first** vs **recall-first** behavior, using only existing columns.

-----

#### **2. Data Strategy (Same Columns, Clearer Roles & Weighting)**

We use the **same columns** as in the original plan, but refine how they are interpreted and weighted.

| Role | Purpose | Columns Used |
| :--- | :--- | :--- |
| **A. The Persona (Vector)** | Describe *who* the person is and provide fuzzy matching across typos, abbreviations, and near-duplicates. Used for semantic similarity and as a soft check when anchors are missing. | `FIRSTNAME`, `LASTNAME`, `MIDDLENAME`, `TITLE`, `CURRENT_EMPLOYER__C`, `MAILINGCITY`, `MAILINGSTATE`, `MAILINGPOSTALCODE` |
| **B. The Anchors (Graph)** | Hard evidence that two records are strongly connected. Shared anchors produce **high-weight edges** in the identity graph. Different anchors can be treated as negative evidence. | `EMAIL`, `OTHER_EMAIL__C`, `PHONE`, `MOBILEPHONE`, `HOMEPHONE`, `OTHERPHONE` |
| **C. The Payload (Output)** | Ignored for matching; used only for downstream actions once a cluster is identified. | `ID`, `ACCOUNTID` |

**POC implementation note:** All persona, anchor, and embedding fields are persisted in BigQuery (`SPM_SAMPLE_CANON`, `SPM_SAMPLE_EMBEDS`, and the joined view `SPM_SAMPLE_CANON_WITH_EMBEDS`). Embeddings are produced with BigQuery’s `ML.GENERATE_EMBEDDING` remote model that proxies Gemini `text-embedding-005`, so every downstream retrieval step operates directly against these BigQuery tables.

**Why this is an improvement**

- **No new data columns introduced** – the design remains fully compatible with the original schema.
- **Anchors get explicit weight and negative evidence handling**, which v1 did not formalize.
- **Persona fields are consistently incorporated** (including `MAILINGPOSTALCODE` in the narrative), improving location-disambiguation without new fields.

-----

#### **3. Updated Decision**

We will implement a **Weighted Graph Resolution Engine with an Explicit Policy Layer**, still content-only and blind to IDs during matching:

- **Step 1 – Content Extraction:** Normalize persona text, emails, and phones from the incoming record and existing records.
- **Step 2 – Candidate Retrieval:** Use a **hybrid retrieval strategy** (anchors-first, then vector fallback) over the same columns.
- **Step 3 – Graph Construction:** Build a small, in-memory **typed-edge weighted graph** over the candidate set.
- **Step 4 – Scoring & Policy:** Compute a **cluster score** and apply explicit rules to decide: `MERGE`, `HUMAN_REVIEW`, or `INSERT_NEW`.

This preserves the original design’s strengths while making the behavior:

- **More predictable** (formal scoring and thresholds).
- **More tunable** (weights per anchor type).
- **More explainable** (each decision is backed by specific signals and scores).

-----

#### **4. Pipeline Blueprint v2 (Phases)**

##### **Phase 0: Normalization & Standardization**

*Goal: Ensure consistent representation of persona and anchor values before any matching.*

```text
FUNCTION NormalizeInput(input_json):

    // Name normalization (case, whitespace, common nicknames handled in model or mapping)
    first = NormalizeName(input_json.FIRSTNAME)
    middle = NormalizeName(input_json.MIDDLENAME)
    last = NormalizeName(input_json.LASTNAME)

    // Title normalization (case, punctuation)
    title = NormalizeTitle(input_json.TITLE)

    // Employer and location normalization
    employer = NormalizeText(input_json.CURRENT_EMPLOYER__C)
    city = NormalizeText(input_json.MAILINGCITY)
    state = NormalizeRegion(input_json.MAILINGSTATE)
    postal = NormalizePostal(input_json.MAILINGPOSTALCODE)

    // Email normalization (trim, lowercase)
    emails = UniqueNonNull([
        NormalizeEmail(input_json.EMAIL),
        NormalizeEmail(input_json.OTHER_EMAIL__C)
    ])

    // Phone normalization (E.164 or consistent local format)
    phones = UniqueNonNull([
        NormalizePhone(input_json.PHONE),
        NormalizePhone(input_json.MOBILEPHONE),
        NormalizePhone(input_json.HOMEPHONE),
        NormalizePhone(input_json.OTHERPHONE)
    ])

    RETURN {
        "first": first, "middle": middle, "last": last,
        "title": title, "employer": employer,
        "city": city, "state": state, "postal": postal,
        "emails": emails, "phones": phones
    }
```

**Improvement vs v1**

- Makes matching **robust to trivial formatting differences** (e.g. `555-0100` vs `(555) 0100`).
- Ensures the **same columns** yield higher-quality signals without adding new attributes.

-----

##### **Phase 1: Persona Embedding & Anchor Extraction**

*Goal: Turn the normalized persona into an embedding and extract lists of anchors, using only existing fields.*

```text
FUNCTION BuildFeatures(normalized):

    // 1. Construct Narrative Blob (Persona)
    blob_string = f"""
        {normalized.first} {normalized.middle} {normalized.last}
        works as {normalized.title}
        at {normalized.employer}.
        Located in {normalized.city}, {normalized.state} {normalized.postal}.
    """

    // 2. Generate Vector Embedding (BigQuery ML.GENERATE_EMBEDDING in practice)
    vector_embedding = GenerateEmbedding(blob_string)

    // 3. Anchors: already normalized lists
    email_list = normalized.emails
    phone_list = normalized.phones

    RETURN {
        "vector": vector_embedding,
        "emails": email_list,
        "phones": phone_list
    }
```

**Improvement vs v1**

- Persona narrative now includes **postal code** explicitly, sharpening location sensitivity with the same columns.
- Normalization + deduplication of anchors reduces noise and accidental mismatches.

-----

##### **Phase 2: Hybrid Candidate Retrieval (Anchors-First, Vector-Backfill)**

*Goal: Retrieve a small, high-quality candidate set using anchors and vector search, without scanning the entire table.*

```text
FUNCTION FindCandidates(features):

    // All queries run against the BigQuery view SPM_SAMPLE_CANON_WITH_EMBEDS

    // A. Anchor-based high-precision retrieval
    candidates_anchor = SELECT * FROM talent_table
        WHERE
            EMAIL IN features.emails
         OR OTHER_EMAIL__C IN features.emails
         OR PHONE IN features.phones
         OR MOBILEPHONE IN features.phones
         OR HOMEPHONE IN features.phones
         OR OTHERPHONE IN features.phones
        LIMIT 200;

    // B. Vector-based semantic retrieval (only if needed)
    candidates_vector = SELECT * FROM talent_table
        WHERE embedding <=> features.vector < 0.20
        ORDER BY embedding <=> features.vector ASC
        LIMIT 200;

    // C. Union + de-duplicate by ID
    candidates = DeduplicateByID(candidates_anchor + candidates_vector)

    RETURN candidates
```

**Improvement vs v1**

- Makes the retrieval strategy **explicitly anchors-first** (precision), then vector (recall).
- Keeps the candidate set bounded, which is important for both **latency** and **graph complexity**.

-----

##### **Phase 3: Typed, Weighted Full Graph Construction**

*Goal: Represent relationships between the incoming record and candidates as a **full graph** (not just star), so transitive links like `A ↔ B ↔ C` imply `A` and `C` are in the same identity cluster.*

```text
FUNCTION BuildResolutionGraph(new_features, candidates):

    Graph G = new Graph()
    G.add_node("INCOMING", vector=new_features.vector,
                         emails=new_features.emails,
                         phones=new_features.phones)

    // 1. Add candidate nodes with features
    candidate_feats = {}

    FOR candidate IN candidates:
        norm_c = NormalizeInput(candidate)
        feat_c = BuildFeatures(norm_c)

        candidate_feats[candidate.ID] = feat_c

        G.add_node(candidate.ID,
                   vector=feat_c.vector,
                   emails=feat_c.emails,
                   phones=feat_c.phones,
                   account_id=candidate.ACCOUNTID)

        // Direct edges between INCOMING and candidate
        sim_in = CosineSimilarity(new_features.vector, feat_c.vector)
        IF sim_in > 0.90:
            G.add_edge("INCOMING", candidate.ID,
                       type="SIMILAR_PROFILE",
                       weight=0.5,
                       score=sim_in)

        shared_emails_in = Intersect(new_features.emails, feat_c.emails)
        IF shared_emails_in:
            G.add_edge("INCOMING", candidate.ID,
                       type="SHARED_EMAIL",
                       weight=1.0,
                       anchors=shared_emails_in)

        shared_phones_in = Intersect(new_features.phones, feat_c.phones)
        IF shared_phones_in:
            G.add_edge("INCOMING", candidate.ID,
                       type="SHARED_PHONE",
                       weight=1.0,
                       anchors=shared_phones_in)

    // 2. Add candidate-to-candidate edges (full graph over candidate set)
    candidate_ids = Keys(candidate_feats)

    FOR each pair (id1, id2) in AllUnorderedPairs(candidate_ids):
        feat1 = candidate_feats[id1]
        feat2 = candidate_feats[id2]

        // Edge A: Semantic similarity (Persona) between candidates
        sim_cc = CosineSimilarity(feat1.vector, feat2.vector)
        IF sim_cc > 0.92:
            G.add_edge(id1, id2,
                       type="SIMILAR_PROFILE",
                       weight=0.4,
                       score=sim_cc)

        // Edge B: Shared Email (Strong Anchor)
        shared_emails_cc = Intersect(feat1.emails, feat2.emails)
        IF shared_emails_cc:
            G.add_edge(id1, id2,
                       type="SHARED_EMAIL",
                       weight=1.0,
                       anchors=shared_emails_cc)

        // Edge C: Shared Phone (Strong Anchor)
        shared_phones_cc = Intersect(feat1.phones, feat2.phones)
        IF shared_phones_cc:
            G.add_edge(id1, id2,
                       type="SHARED_PHONE",
                       weight=1.0,
                       anchors=shared_phones_cc)

    RETURN G
```

**Improvement vs v1**

- Uses a **full graph over all candidates**, so if `INCOMING` is linked to `A`, and `A` shares a strong anchor with `B`, then `B` is also pulled into the same cluster via graph connectivity.
- Still bounded in size and fast because the candidate set is capped (e.g., ≤ 200), so the full graph is small and tractable in memory.
- Keeps using only existing columns (emails, phones, persona fields) while capturing richer transitive identity structure than the original star-graph design.

-----

##### **Phase 4: Cluster Scoring & Decision Policy**

*Goal: Convert the graph into a deterministic decision using a clear scoring and policy framework.*

```text
FUNCTION DecideAction(G):

    // 1. Get connected component around INCOMING
    cluster_nodes = G.get_connected_component("INCOMING") - {"INCOMING"}

    IF cluster_nodes is empty:
        RETURN "INSERT_NEW", [], []

    decisions = []

    FOR node_id IN cluster_nodes:
        edges = G.get_edges("INCOMING", node_id)

        // Aggregate weights (anchors > persona)
        total_weight = SUM(edge.weight for edge in edges)

        has_email_edge = EXISTS(edge for edge in edges if edge.type == "SHARED_EMAIL")
        has_phone_edge = EXISTS(edge for edge in edges if edge.type == "SHARED_PHONE")

        // 2. Scoring policy
        IF has_email_edge OR has_phone_edge:
            // Strong anchor present
            IF total_weight >= 1.5:
                decisions.append((node_id, "MERGE_STRONG"))
            ELSE:
                decisions.append((node_id, "REVIEW_ANCHOR"))
        ELSE:
            // No strong anchors; rely on persona only
            IF total_weight >= 0.8:
                decisions.append((node_id, "REVIEW_PERSONA"))
            ELSE:
                decisions.append((node_id, "IGNORE"))

    // 3. Aggregate decision to record-level outcome
    merge_ids = [id for (id, label) in decisions if label == "MERGE_STRONG"]
    review_ids = [id for (id, label) in decisions if label in ["REVIEW_ANCHOR", "REVIEW_PERSONA"]]

    IF merge_ids:
        RETURN "MERGE", merge_ids, GetAccountsFor(merge_ids)

    IF review_ids:
        RETURN "HUMAN_REVIEW", review_ids, GetAccountsFor(review_ids)

    RETURN "INSERT_NEW", [], []
```

**Improvement vs v1**

- Introduces **three-way outcome** (`MERGE`, `HUMAN_REVIEW`, `INSERT_NEW`) instead of binary merge/insert.
- Makes anchor-based matches **dominant** over pure vector similarity for safety.
- Ensures each decision can be **explained** by a set of edges (e.g., “shared phone + high similarity”).

-----

#### **5. Why ADR v2 is a Clear Improvement (Within the Same Data Columns)**

- **Stronger use of anchors:** Emails and phones are promoted to **first-class scoring primitives** with explicit weighting; this addresses the “Doppelgänger” failure more reliably.
- **Structured conflict handling:** The design can treat conflicting signals as “review, not auto-merge” without needing new attributes.
- **Explainable decisions:** Each merge or review can be justified via the set of edges and scores between `INCOMING` and candidates.
- **Operational scalability:** Candidate retrieval is limited and controlled; graph size is small and predictable for each incoming record.
- **No schema changes:** The improved behavior is achieved **entirely within the existing columns**: persona (`FIRSTNAME`, `LASTNAME`, `MIDDLENAME`, `TITLE`, `CURRENT_EMPLOYER__C`, `MAILINGCITY`, `MAILINGSTATE`, `MAILINGPOSTALCODE`), anchors (`EMAIL`, `OTHER_EMAIL__C`, `PHONE`, `MOBILEPHONE`, `HOMEPHONE`, `OTHERPHONE`), and payload (`ID`, `ACCOUNTID`).


