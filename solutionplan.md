# Solution Plan: Talent Person Matching (SPM) Using Splink + BlockingPy

## Executive Summary

This document outlines a production-ready solution for **de-duplicating and linking talent records** using **Splink** (probabilistic record linkage) combined with **BlockingPy** (Approximate Nearest Neighbour blocking). This approach addresses the core challenge: *"How do we know if two resume records represent the same person when the system-assigned IDs are meaningless?"*

---

## 1. The Problem We're Solving

### 1.1 Current Pain Points

Our talent data arrives with **synthetic IDs** that have no real-world meaning. The same person can appear multiple times with different IDs, and we face two critical errors:

| Error Type | Description | Business Impact |
|------------|-------------|-----------------|
| **Fragmented Identities** | Same person exists as multiple records | Duplicate outreach, inflated candidate counts, poor analytics |
| **False Merges** | Different people incorrectly linked | Wrong resumes sent to clients, compliance issues, reputation damage |

### 1.2 Why Traditional Methods Fail

| Approach | Why It Fails for Our Data |
|----------|---------------------------|
| **Exact ID matching** | IDs are random/synthetic - meaningless for identity |
| **Exact name matching** | `James Smith` ≠ `Jim Smith` ≠ `J. Smith` (same person, different formats) |
| **Exact email matching** | `john.doe@gmail.com` = `johndoe@gmail.com` (Gmail ignores dots) |
| **Simple rules** | People share names + employers; true duplicates have typos |

### 1.3 Our Data Profile

Based on `spm_data.tsv`, each record contains:

| Field Category | Fields Available | Matching Challenge |
|----------------|------------------|-------------------|
| **Names** | NAME_1, NAME_2, NAME_3 | Variations, nicknames, typos |
| **Contact** | PHONE_1/2/3, EMAIL_1/2/3 | Format variations, multiple contacts per person |
| **Employment** | EMPLOYER_1-4, TITLE_1_1-4_2 | Job title variations, company name differences |
| **Location** | COUNTRY, REGION, MUNICIPALITY | Abbreviations, different granularity |
| **Education** | DEGREE_1_1 to DEGREE_3_2 | Abbreviations (BS, B.S., Bachelor of Science) |

**Data Volume:** ~149,000 records → ~11 billion possible comparisons if done naively

---

## 2. Why Splink + BlockingPy?

### 2.1 The Combined Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPM MATCHING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│   │   RAW       │         │  BLOCKINGPY │         │   SPLINK    │          │
│   │   DATA      │   ──►   │   BLOCKING  │   ──►   │   SCORING   │          │
│   │  149K rows  │         │   (ANN)     │         │(Probabilistic)│         │
│   └─────────────┘         └─────────────┘         └─────────────┘          │
│                                 │                       │                   │
│                                 ▼                       ▼                   │
│                         ┌─────────────┐         ┌─────────────┐            │
│                         │  ~250K      │         │   FINAL     │            │
│                         │  candidates │         │   MATCHES   │            │
│                         │  (99.6%↓)   │         │   + SCORES  │            │
│                         └─────────────┘         └─────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why This Combination Works

| Component | Role | What It Provides |
|-----------|------|------------------|
| **BlockingPy** | Candidate Generation | Finds similar records using text embeddings + ANN search; handles typos, missing data, variations |
| **Splink** | Probabilistic Scoring | Assigns match probability using Fellegi-Sunter model; quantifies uncertainty; explainable decisions |

### 2.3 Decision Matrix: Why Not Other Approaches?

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Exact Matching** | Fast, simple | Misses variations, typos | ❌ Too rigid |
| **Pure Embedding Similarity** | Handles semantics | No interpretability, can't weight fields | ❌ Black box |
| **Splink Only** | Probabilistic, explainable | Deterministic blocking misses fuzzy matches | ⚠️ Incomplete |
| **BlockingPy Only** | Great recall | No probabilistic scoring | ⚠️ Incomplete |
| **Splink + BlockingPy** | Best of both: high recall + interpretable scores | Requires both libraries | ✅ **Recommended** |

---

## 3. Solution Architecture

### 3.1 High-Level Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW ARCHITECTURE                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │ Raw TSV  │ -> │ PHASE 1:     │ -> │ PHASE 2:     │ -> │ PHASE 3:     │             │
│  │ Data     │    │ Clean &      │    │ BlockingPy   │    │ Splink       │             │
│  │          │    │ Normalize    │    │ Blocking     │    │ Matching     │             │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘             │
│                         │                   │                   │                      │
│                         ▼                   ▼                   ▼                      │
│                  ┌────────────┐      ┌────────────┐      ┌────────────┐               │
│                  │ Normalized │      │ Block IDs  │      │ Match      │               │
│                  │ Fields     │      │ Added      │      │ Probability│               │
│                  └────────────┘      └────────────┘      └────────────┘               │
│                                                                 │                      │
│                                                                 ▼                      │
│                                                          ┌────────────┐               │
│                                                          │ PHASE 4:   │               │
│                                                          │ Cluster &  │               │
│                                                          │ Decision   │               │
│                                                          └────────────┘               │
│                                                                 │                      │
│                                                                 ▼                      │
│                                             ┌────────────────────────────────┐         │
│                                             │  AUTO-MERGE | REVIEW | NEW     │         │
│                                             └────────────────────────────────┘         │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase Details

#### Phase 1: Data Cleaning & Normalization

```python
# Fields to normalize
FIELDS = {
    'names': ['NAME_1', 'NAME_2', 'NAME_3'],
    'phones': ['PHONE_1', 'PHONE_2', 'PHONE_3'],
    'emails': ['EMAIL_1', 'EMAIL_2', 'EMAIL_3'],
    'employers': ['EMPLOYER_1', 'EMPLOYER_2', 'EMPLOYER_3', 'EMPLOYER_4'],
    'titles': ['TITLE_1_1', 'TITLE_2_1', 'TITLE_3_1', 'TITLE_4_1'],
    'locations': ['REGION_1_1', 'MUNICIPALITY_1_1', ...],
}

# Normalization Rules
- Names: lowercase, remove titles (Mr., Dr.), standardize spacing
- Phones: E.164 format (+1XXXXXXXXXX), remove non-digits
- Emails: lowercase, normalize Gmail (remove dots), domain aliases
- Employers: lowercase, remove Inc/LLC/Corp suffixes
```

#### Phase 2: BlockingPy ANN Blocking

```python
from blockingpy import Blocker

# Create combined text field for semantic similarity
df['txt'] = (
    df['NAME_1'].fillna('') + ' ' +
    df['EMPLOYER_1'].fillna('') + ' ' +
    df['TITLE_1_1'].fillna('') + ' ' +
    df['REGION_1_1'].fillna('') + ' ' +
    df['EMAIL_1'].fillna('')
)

# Run BlockingPy
blocker = Blocker()
result = blocker.block(
    x=df['txt'],
    ann='hnsw',              # Hierarchical Navigable Small World graph
    k=10,                    # Find 10 nearest neighbors
    random_seed=42
)

# Add block column
df = result.add_block_column(df)
```

**Expected Outcome:**
- **Reduction ratio:** ~99.6% (from 11B comparisons to ~250K candidates)
- **Recall:** 80-90% of true matches captured

#### Phase 3: Splink Probabilistic Matching

```python
from splink import SettingsCreator, Linker, block_on, DuckDBAPI
import splink.comparison_library as cl

settings = SettingsCreator(
    link_type="dedupe_only",
    
    # BLOCKING: Use BlockingPy + deterministic fallbacks
    blocking_rules_to_generate_predictions=[
        block_on("block"),              # BlockingPy's ANN blocks
        block_on("EMAIL_1"),            # Exact email matches
        block_on("PHONE_1"),            # Exact phone matches
        block_on("NAME_1", "EMPLOYER_1"),  # Deterministic fallback
    ],
    
    # COMPARISONS: How to score similarity
    comparisons=[
        cl.NameComparison("NAME_1"),
        cl.EmailComparison("EMAIL_1"),
        cl.LevenshteinAtThresholds("PHONE_1", [1, 2]),
        cl.NameComparison("EMPLOYER_1"),
        cl.JaroWinklerAtThresholds("TITLE_1_1", [0.9, 0.8]),
        cl.ExactMatch("REGION_1_1"),
    ],
)

linker = Linker(df, settings, db_api=DuckDBAPI())

# Train model (using deterministic rules, NOT BlockingPy blocks)
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
linker.training.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[
        "l.EMAIL_1 = r.EMAIL_1",
        "l.PHONE_1 = r.PHONE_1",
    ],
    recall=0.6
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("NAME_1", "EMPLOYER_1")
)

# Generate predictions
predictions = linker.inference.predict(threshold_match_probability=0.7)
```

#### Phase 4: Clustering & Decision

```python
# Cluster connected records into person entities
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    predictions,
    threshold_match_probability=0.9
)

# Decision Rules
THRESHOLDS = {
    'auto_merge': 0.95,    # High confidence → automatic merge
    'human_review': 0.70,  # Medium confidence → queue for review
    'reject': 0.70,        # Below threshold → different people
}
```

---

## 4. Benefits of This Approach

### 4.1 Quantitative Benefits

| Metric | Without Solution | With Splink + BlockingPy |
|--------|------------------|--------------------------|
| **Comparisons needed** | 11 billion | ~250K (99.6% reduction) |
| **Processing time** | Hours/days | Minutes |
| **Duplicate detection rate** | ~50-60% (rules) | ~85-95% (probabilistic) |
| **False positive rate** | High (rigid rules) | Controlled (threshold-based) |

### 4.2 Qualitative Benefits

| Benefit | Description |
|---------|-------------|
| **🎯 Higher Accuracy** | Probabilistic scoring handles typos, variations, missing data |
| **📊 Explainability** | Match weight charts show WHY records matched (audit trail) |
| **⚡ Scalability** | Handles millions of records with proper blocking |
| **🔧 Tunability** | Adjust thresholds based on business tolerance for FP/FN |
| **📈 Iterative Improvement** | Labeling tool helps improve model over time |
| **💰 Cost Effective** | Open source (Splink, BlockingPy) - no licensing fees |

### 4.3 Technical Benefits

| Feature | Splink Provides | BlockingPy Provides |
|---------|-----------------|---------------------|
| **Blocking** | Deterministic (equality-based) | ANN-based (fuzzy similarity) |
| **Matching** | Fellegi-Sunter probabilistic model | - |
| **Training** | EM algorithm, u/m estimation | - |
| **Visualization** | Waterfall charts, cluster viewer | - |
| **Fuzzy matching** | Levenshtein, Jaro-Winkler comparisons | Handles via embeddings |
| **Semantic similarity** | Limited | Text embeddings (model2vec) |

---

## 5. Implementation Roadmap

### Phase 1: POC (Weeks 1-2)
- [ ] Set up Python environment (Splink, BlockingPy, DuckDB)
- [ ] Load sample data (1000 records)
- [ ] Implement basic pipeline
- [ ] Validate on known duplicates

### Phase 2: Tuning (Weeks 3-4)
- [ ] Optimize BlockingPy parameters (algorithm, k, distance metric)
- [ ] Train Splink model on full dataset
- [ ] Adjust thresholds based on precision/recall tradeoff
- [ ] Document edge cases

### Phase 3: Integration (Weeks 5-6)
- [ ] Integrate with existing data pipeline
- [ ] Build review queue for medium-confidence matches
- [ ] Create dashboard for match analytics
- [ ] Performance testing at scale

### Phase 4: Production (Weeks 7-8)
- [ ] Deploy to production environment
- [ ] Implement incremental matching for new records
- [ ] Set up monitoring and alerting
- [ ] Train operations team

---

## 6. Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| **Low recall** | Add multiple blocking rules (BlockingPy + deterministic) |
| **High false positives** | Increase match threshold, add comparison columns |
| **Performance at scale** | Use DuckDB backend, optimize blocking |
| **Model drift** | Regular retraining, monitor precision metrics |
| **Training bias** | Train on deterministic rules, not BlockingPy blocks |

---

## 7. Success Metrics

| KPI | Target | Measurement Method |
|-----|--------|-------------------|
| **Precision** | > 95% | Manual review of sample matches |
| **Recall** | > 85% | Against labeled ground truth |
| **F1 Score** | > 90% | Harmonic mean of P/R |
| **Processing time** | < 10 min for 150K records | Benchmark runs |
| **Reduction ratio** | > 99% | Blocking output vs. full cartesian |

---

## Appendix A: Industry Best Methods (Footnotes)

This section provides context on what other solutions exist in the market and how our approach compares.

### A.1 Commercial Entity Resolution Solutions

| Solution | Type | Best For | Pricing Model |
|----------|------|----------|---------------|
| **AWS Entity Resolution** | Cloud Service | AWS-native architectures | Pay-per-use ($0.25/1K records matched) |
| **Informatica MDM** | Enterprise Platform | Large enterprises, complex hierarchies | Enterprise licensing |
| **Reltio** | Cloud MDM | Healthcare, Life Sciences | Subscription |
| **Tamr** | ML-based MDM | Data unification at scale | Enterprise |
| **Senzing** | API-based | Real-time matching | Usage-based |

**Our Position:** We use open-source tools (Splink + BlockingPy) for cost efficiency and full control.

### A.2 Open Source Alternatives

| Tool | Language | Approach | Status |
|------|----------|----------|--------|
| **Splink** (our choice) | Python | Fellegi-Sunter probabilistic | ✅ Active, UK Gov backed |
| **Dedupe.io** | Python | Active learning | ⚠️ Limited updates |
| **Zingg** | Scala/Spark | ML-based, Spark-native | ✅ Active |
| **RecordLinkage (R)** | R | Statistical | ⚠️ R ecosystem only |
| **FEBRL** | Python | Traditional probabilistic | ❌ Legacy |
| **BlockingPy** (our choice) | Python | ANN blocking | ✅ Active, academic-backed |

**Reference:** [Splink vs alternatives comparison](https://moj-analytical-services.github.io/splink/)

### A.3 Academic Foundations

The solution is built on established research:

| Paper/Method | Year | Contribution |
|--------------|------|--------------|
| **Fellegi-Sunter Model** | 1969 | Mathematical framework for probabilistic matching |
| **EM Algorithm for Record Linkage** | 1980s | Unsupervised parameter estimation |
| **Blocking Techniques Survey** | 2012 | Comprehensive review of blocking methods |
| **Deep Learning for ER** | 2018+ | Neural approaches to entity matching |
| **BlockingPy Paper** | 2024 | ANN-based blocking with graph clustering |

**Key Reference:** [BlockingPy: Approximate Nearest Neighbours for Blocking](https://arxiv.org/html/2504.04266v3)

### A.4 Industry Best Practices Summary

Based on NCBI, Census Bureau, and academic literature:

1. **Data Quality First**
   - Standardize formats before matching
   - Handle missing values explicitly
   - Normalize identifiers (emails, phones)

2. **Multi-Stage Approach**
   - Blocking → Comparison → Classification → Evaluation
   - Each stage has different optimization criteria

3. **Probabilistic over Deterministic**
   - Deterministic rules miss variations
   - Probabilistic models quantify uncertainty
   - Threshold tuning controls precision/recall tradeoff

4. **Privacy-Preserving Techniques**
   - Hash-based blocking for sensitive data
   - Differential privacy for aggregate statistics
   - Secure multi-party computation for cross-org matching

5. **Incremental Matching**
   - Don't re-process entire dataset for new records
   - Index-based lookup for real-time matching
   - Batch processing for periodic reconciliation

6. **Human-in-the-Loop**
   - Active learning improves model over time
   - Manual review for edge cases
   - Feedback loop for continuous improvement

### A.5 Useful Resources

- [Splink Documentation](https://moj-analytical-services.github.io/splink/)
- [BlockingPy Documentation](https://blockingpy.readthedocs.io/)
- [Census Bureau: Optimal Probabilistic Record Linkage](https://www.census.gov/library/working-papers/2019/adrm/ces-wp-19-08.html)
- [NCBI: Record Linkage Best Practices](https://www.ncbi.nlm.nih.gov/books/NBK253312/)
- [Google Research: Entity Matching Benchmark](https://github.com/anhaidgroup/deepmatcher)

---

## Document Information

| Property | Value |
|----------|-------|
| **Created** | December 2024 |
| **Author** | SPM Team |
| **Status** | Draft |
| **Next Review** | After POC completion |

---

*This solution plan combines industry best practices with our specific data characteristics to provide a robust, scalable, and explainable approach to talent record deduplication.*

