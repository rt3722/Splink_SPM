## Deep research summary: fuzzy matching and email similarity

### 1. Library performance and recommendations

#### RapidFuzz (recommended)

- Performance: C++ backend, typically 10–100x faster than FuzzyWuzzy
- Algorithms: Levenshtein, Jaro-Winkler, ratio, partial_ratio, token_sort_ratio, token_set_ratio
- Use cases: Names, emails, general string matching
- Installation: `pip install rapidfuzz`
- License: MIT

#### Jellyfish

- Phonetic algorithms: Soundex, Metaphone, NYSIIS, Match Rating Codex
- String distance: Levenshtein, Jaro-Winkler, Hamming, Damerau-Levenshtein
- Use cases: Names with spelling variations (e.g., "Smith" vs "Smyth", "John" vs "Jon")
- Installation: `pip install jellyfish`

#### FuzzyWuzzy (legacy)

- Status: Slower than RapidFuzz; RapidFuzz is a drop-in replacement
- Recommendation: Use RapidFuzz instead

### 2. Email similarity: vector embeddings vs fuzzy matching

#### Vector embeddings are not reliable for emails

Why:

- Embeddings capture semantic meaning, not character-level structure
- Emails like `rajeshthumu21@gmail.com` vs `rajeshthumu@gmail.com` are structurally similar but semantically identical (both are identifiers)
- Tokenization treats emails as single tokens or splits them unpredictably
- Embeddings may place `rajeshthumu21@gmail.com` closer to `rajeshthumu22@gmail.com` than to `rajeshthumu@gmail.com` due to numeric similarity

What embeddings are good for:

- Persona text (names, titles, employers) where semantic similarity matters
- Example: "Software Engineer" ≈ "Dev" ≈ "Developer" (semantically close)

What embeddings are not good for:

- Email addresses (character-level differences matter)
- Phone numbers (exact or near-exact matching needed)
- Structured identifiers

### 3. Email normalization best practices

#### Domain normalization

```python
# Common domain aliases
DOMAIN_ALIASES = {
    'googlemail.com': 'gmail.com',
    'gmail.co.uk': 'gmail.com',
    'yahoo.co.uk': 'yahoo.com',
    # Add more as needed
}

def normalize_domain(domain):
    domain = domain.lower().strip()
    return DOMAIN_ALIASES.get(domain, domain)
```

#### Gmail-specific handling

- Gmail ignores dots in local part: `rajesh.thumu@gmail.com` = `rajeshthumu@gmail.com`
- Gmail ignores plus addressing: `rajeshthumu+test@gmail.com` = `rajeshthumu@gmail.com`
- Case insensitive: `RajeshThumu@gmail.com` = `rajeshthumu@gmail.com`

#### Local part comparison

- Extract local part (before `@`)
- Normalize: lowercase, remove dots (for Gmail), strip plus addressing
- Use Jaro-Winkler for local part similarity (better for short strings with typos)

### 4. Recommended matching strategy

#### For emails (anchors)

```python
from rapidfuzz import fuzz
from jellyfish import jaro_winkler_similarity

def normalize_email_local_part(email, domain):
    """Normalize email local part based on domain rules"""
    local = email.lower().split('@')[0]

    # Gmail-specific: remove dots and plus addressing
    if domain == 'gmail.com':
        local = local.replace('.', '').split('+')[0]

    return local

def email_similarity(email1, email2):
    """Calculate email similarity score (0.0 to 1.0)"""
    if not email1 or not email2:
        return 0.0

    # Normalize domains
    domain1 = normalize_domain(email1.split('@')[1])
    domain2 = normalize_domain(email2.split('@')[1])

    # Different domains = different people (usually)
    if domain1 != domain2:
        return 0.0

    # Normalize local parts
    local1 = normalize_email_local_part(email1, domain1)
    local2 = normalize_email_local_part(email2, domain2)

    # Exact match after normalization
    if local1 == local2:
        return 1.0

    # Fuzzy match on local part
    # Use Jaro-Winkler (better for short strings with typos)
    similarity = jaro_winkler_similarity(local1, local2)

    # Threshold: 0.85+ for potential match
    return similarity if similarity >= 0.85 else 0.0
```

#### For names (persona)

```python
from rapidfuzz import fuzz
from jellyfish import soundex, metaphone

def name_similarity(name1, name2):
    """Hybrid name matching: phonetic + character-level"""
    if not name1 or not name2:
        return 0.0

    # Normalize
    n1 = name1.strip().lower()
    n2 = name2.strip().lower()

    # Exact match
    if n1 == n2:
        return 1.0

    # Phonetic match (handles "John" vs "Jon")
    if soundex(n1) == soundex(n2) or metaphone(n1) == metaphone(n2):
        return 0.95

    # Character-level similarity (handles typos)
    ratio = fuzz.ratio(n1, n2) / 100.0

    # Threshold: 0.80+ for potential match
    return ratio if ratio >= 0.80 else 0.0
```

### 5. Threshold recommendations (production)

Based on entity resolution best practices:

| Match Type | Algorithm | Threshold | Use Case |
|------------|-----------|-----------|----------|
| Email (exact) | Normalized equality | 1.0 | Strong anchor |
| Email (fuzzy) | Jaro-Winkler | ≥ 0.85 | Potential match |
| Phone (exact) | Normalized equality | 1.0 | Strong anchor |
| Phone (fuzzy) | Levenshtein | ≥ 0.90 | Potential match |
| Name (phonetic) | Soundex/Metaphone | Match | Common variations |
| Name (character) | Levenshtein/Jaro-Winkler | ≥ 0.80 | Typos/variations |
| Persona (vector) | Cosine similarity | ≥ 0.90 | Semantic similarity |

### 6. BigQuery implementation considerations

#### Option 1: Python preprocessing (recommended for POC)

- Normalize emails/names in Python before loading to BigQuery
- Store normalized values in `SPM_SAMPLE_CANON`
- Use exact matching in BigQuery SQL

#### Option 2: BigQuery JavaScript UDFs

- Implement Levenshtein/Jaro-Winkler in JavaScript UDFs
- Slower than Python but keeps logic in SQL
- Example structure:

```sql
CREATE OR REPLACE FUNCTION `spm_udfs.email_similarity`(email1 STRING, email2 STRING)
RETURNS FLOAT64
LANGUAGE js AS """
  // JavaScript implementation of Jaro-Winkler
  // ... implementation ...
""";
```

#### Option 3: Hybrid approach

- Exact matching in BigQuery (fast)
- Fuzzy matching in Cloud Run Python code (flexible)
- Best of both worlds

### 7. Updated architecture recommendation

#### Phase 0: Enhanced normalization

```python
# Use RapidFuzz + Jellyfish for normalization
from rapidfuzz import fuzz
from jellyfish import soundex, metaphone, jaro_winkler_similarity

def normalize_for_matching(record):
    # Names: phonetic + character normalization
    # Emails: domain + local part normalization
    # Phones: E.164 format normalization
    pass
```

#### Phase 2: Hybrid candidate retrieval

```sql
-- Anchor-based: Exact + fuzzy (in Python/Cloud Run)
-- Vector-based: Semantic similarity (in BigQuery)
-- Combine both signals
```

#### Phase 3: Graph construction

- Use exact matching for strong edges (shared email/phone)
- Use fuzzy matching for weak edges (similar emails, threshold ≥ 0.85)
- Use vector similarity for persona edges (threshold ≥ 0.90)

### 8. Final recommendations

1. Use RapidFuzz for general fuzzy matching (fast, production-ready)
2. Use Jellyfish for phonetic name matching (Soundex/Metaphone)
3. Do not rely on vector embeddings for emails — use character-level fuzzy matching
4. Use vector embeddings for persona text (names, titles, employers)
5. Implement email normalization (domain aliases, Gmail rules)
6. Use hybrid thresholds: exact match = 1.0, fuzzy email ≥ 0.85, fuzzy name ≥ 0.80
7. Prefer Python preprocessing for POC; consider BigQuery UDFs for v2

