# BlockingPy Q&A Document

This document contains questions and answers about the BlockingPy library.

---

## Q1: How is the Graph Construction & Connected Components done?

### Short Answer
BlockingPy uses the **igraph** library to construct an undirected graph from ANN search results, then finds **connected components** to group records into blocks. Records that are "neighbours" according to the ANN search end up in the same block.

### Detailed Explanation

The graph construction and connected components logic is implemented in `blocker.py` (lines 217-233). Here's how it works step by step:

#### Step 1: ANN Search Results
After the ANN algorithm finds nearest neighbours, the results are stored in a DataFrame (`x_df`) with columns:
- `x` - Index of the reference record
- `y` - Index of the query record  
- `dist` - Distance between them

#### Step 2: Create Node Labels
Each record gets a unique node label for the graph:

```python
x_df["query_g"] = "q" + x_df["y"].astype(str)      # e.g., "q0", "q1", "q2"...
x_df["index_g"] = np.where(
    deduplication,
    "q" + x_df["x"].astype(str),   # For dedup: same prefix (q) for both
    "i" + x_df["x"].astype(str),   # For linkage: different prefix (i vs q)
)
```

- **Deduplication**: Both records get "q" prefix (e.g., `q0`, `q1`) since they're from the same dataset
- **Record Linkage**: Query records get "q" prefix, index records get "i" prefix (e.g., `q0`, `i5`)

#### Step 3: Build Edge List
Create edges from the neighbour pairs:

```python
edges = list(zip(x_df["query_g"].to_numpy(), x_df["index_g"].to_numpy(), strict=False))
```

If ANN found that record 5 is similar to record 12, we get edge: `("q5", "q12")` (dedup) or `("q5", "i12")` (linkage)

#### Step 4: Construct Graph Using igraph

```python
from igraph import Graph

g = Graph.TupleList(edges=edges, directed=False, vertex_name_attr="name")
```

This creates an **undirected graph** where:
- **Nodes** = records (labeled like "q0", "q1", "i0", "i1")
- **Edges** = similarity relationships from ANN search

#### Step 5: Find Connected Components

```python
comp = g.components(mode="weak")
membership = np.asarray(comp.membership, dtype=np.int64)
names = g.vs["name"]
```

The `components(mode="weak")` method finds **weakly connected components** - groups of nodes where there's a path between any two nodes.

#### Step 6: Assign Block IDs

```python
node_to_comp = dict(zip(names, membership, strict=False))
x_df["block"] = x_df["query_g"].map(node_to_comp).astype("int64")
```

Each connected component becomes a **block**. Records in the same component get the same block ID.

### Visual Example

```
ANN Search Results:        Graph:                    Connected Components:
                                                     
Record 0 → Record 1        q0 ─── q1                 Block 0: {q0, q1, q2}
Record 1 → Record 2             ╲                    Block 1: {q3, q4}
                                 q2                  Block 2: {q5}
Record 3 → Record 4        q3 ─── q4
                                 
Record 5 → (no match)      q5 (isolated)
```

### Why Connected Components?

Using connected components has key advantages:

1. **Transitivity**: If A is similar to B, and B is similar to C, then A, B, and C should all be compared together (same block)

2. **Handles chains**: ANN might find A→B and B→C, but not A→C directly. Connected components still groups them correctly.

3. **Efficient**: igraph's component detection is O(V + E) - linear in the number of vertices and edges.

### Code Reference

From `blocker.py`:

```python
217:        logger.info("===== creating graph =====")
218:        x_df["query_g"] = "q" + x_df["y"].astype(str)
219:        x_df["index_g"] = np.where(
220:            deduplication,
221:            "q" + x_df["x"].astype(str),
222:            "i" + x_df["x"].astype(str),
223:        )
224:
225:        edges = list(zip(x_df["query_g"].to_numpy(), x_df["index_g"].to_numpy(), strict=False))
226:        g = Graph.TupleList(edges=edges, directed=False, vertex_name_attr="name")
227:
228:        comp = g.components(mode="weak")
229:        membership = np.asarray(comp.membership, dtype=np.int64)
230:        names = g.vs["name"]
```

---

## Q2: Does combining Splink and BlockingPy yield better results?

### Short Answer
**Yes!** According to the BlockingPy documentation and examples, combining both approaches **significantly improves performance** by capturing comparison pairs that would otherwise be missed by deterministic blocking alone.

### Evidence from the Code/Documentation

The `splink_integration.md` documentation provides a direct comparison of three blocking strategies on the `fake_1000` dataset:

1. **BlockingPy only**
2. **Splink only** (deterministic rules)
3. **Splink + BlockingPy combined**

#### Configuration Comparison

```python
# 1. BlockingPy only
blocking_rules_to_generate_predictions=[
    block_on("block"),  # Using BlockingPy's ANN blocks
]

# 2. Splink only (traditional deterministic)
blocking_rules_to_generate_predictions=[
    block_on("first_name"),
    block_on("surname"),
    block_on("dob"),
    block_on("email"),
]

# 3. Combined approach (BEST)
blocking_rules_to_generate_predictions=[
    block_on("block"),        # BlockingPy's ANN blocks
    block_on("first_name"),   # Plus Splink's deterministic rules
    block_on("surname"),
    block_on("dob"),
    block_on("email"),
]
```

### Results

From the Splink integration documentation:

> "The comparison between traditional methods, BlockingPy and the combination of both shows that **when using both approaches we were able to significantly improve the performance metrics by capturing comparison pairs that would otherwise be missed**."

The documentation includes ROC curve visualizations showing:
- Combined approach has **better precision-recall tradeoff**
- Captures matches that deterministic blocking misses
- No significant increase in false positives

### Why the Combined Approach Works Better

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Splink only** | Fast, precise on exact matches | Misses typos, variations, missing data |
| **BlockingPy only** | Handles fuzzy matching | May miss exact matches outside k-neighbours |
| **Combined** | Gets both exact AND fuzzy matches | Slightly more candidates (but worth it) |

### Practical Example

Consider two records:
- Record A: `"John Smith, 1990-01-15, john@email.com"`
- Record B: `"Jon Smyth, 1990-01-15, jon@email.com"` (typos in name)

| Blocking Method | Will Block Together? |
|-----------------|---------------------|
| `block_on("first_name")` | ❌ No (John ≠ Jon) |
| `block_on("surname")` | ❌ No (Smith ≠ Smyth) |
| `block_on("dob")` | ✅ Yes |
| BlockingPy ANN | ✅ Yes (similar text vectors) |
| **Combined** | ✅ Yes (caught by both dob AND ANN) |

Now consider:
- Record C: `"Jane Doe, 1985-03-20, jane@doe.com"`
- Record D: `"Jane Doe, 1985-03-20, janedoe@other.com"` (different email domain)

| Blocking Method | Will Block Together? |
|-----------------|---------------------|
| `block_on("first_name")` | ✅ Yes |
| `block_on("email")` | ❌ No |
| BlockingPy ANN | ✅ Yes (similar overall) |
| **Combined** | ✅ Yes (caught by first_name AND ANN) |

### Code Example: Full Integration

From `splink_integration.md`:

```python
from splink import splink_datasets, SettingsCreator, Linker, block_on, DuckDBAPI
import splink.comparison_library as cl
from blockingpy import Blocker

# Load data
df = splink_datasets.fake_1000

# Create text field for BlockingPy
df['txt'] = df['first_name'].fillna('') + ' ' + \
            df['surname'].fillna('') + \
            df['dob'].fillna('') + ' ' + \
            df['city'].fillna('') + ' ' + \
            df['email'].fillna('')

# Run BlockingPy
blocker = Blocker()
res = blocker.block(x=df['txt'], ann='hnsw', random_seed=42)

# Add block column to dataframe
df = res.add_block_column(df)

# Configure Splink with BOTH blocking approaches
settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("block"),       # BlockingPy blocks (fuzzy)
        block_on("first_name"),  # Deterministic rules
        block_on("surname"),
        block_on("dob"),
        block_on("email"),
    ],
    comparisons=[
        cl.ForenameSurnameComparison("first_name", "surname"),
        cl.DateOfBirthComparison("dob", input_is_string=True),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
)

# Create linker and continue with Splink workflow
linker = Linker(df, settings, db_api=DuckDBAPI())
```

### Performance Comparison with blocklib

From `blocklib_comp.md`, BlockingPy consistently outperforms alternatives:

| Algorithm | Recall | Candidate Pairs (150k dataset) |
|-----------|--------|-------------------------------|
| BlockingPy (faiss_hnsw) | **0.832** | 0.375 million |
| BlockingPy (voyager) | **0.715** | 0.675 million |
| blocklib P-Sig | 0.609 | 40.2 million |
| blocklib λ-fold LSH | 0.450 | 95.1 million |

**BlockingPy achieves ~1.4-1.8x higher recall with ~100-250x fewer candidate pairs!**

### Conclusion

**Yes, combining Splink and BlockingPy yields significantly better results:**

1. ✅ **Higher recall** - Catches matches that deterministic rules miss
2. ✅ **Maintained precision** - Splink's probabilistic scoring handles false positives
3. ✅ **Complementary strengths** - Deterministic for exact, ANN for fuzzy
4. ✅ **Production-proven** - Used by Statistics Poland for official statistics

The integration is straightforward: run BlockingPy first, add its block column to your data, then include `block_on("block")` in your Splink blocking rules alongside your traditional rules.

---

## Q3: Does BlockingPy cause column correlation issues in Splink? How to avoid them?

### Short Answer

**The "block" column from BlockingPy does NOT violate Splink's conditional independence assumption for comparisons**, because it's used for **blocking** (candidate pair generation), not as a **comparison column**. However, you **must follow best practices** to avoid subtle bias issues during model training.

### Understanding Splink's Conditional Independence Assumption

#### What Splink Says About Correlated Columns

From the Splink documentation:

> "Splink performs best with input data containing **multiple columns that are not highly correlated**. High correlation occurs when the value of a column is highly constrained (predictable) from the value of another column. For example, a 'city' field is almost perfectly correlated with 'postcode'. Gender is highly correlated with 'first name'."

#### Why Correlation is Problematic

The Fellegi-Sunter model assumes **conditional independence** of comparisons given match status:

```
P(comparison₁, comparison₂ | Match) = P(comparison₁ | Match) × P(comparison₂ | Match)
```

**If violated** (columns are correlated), the model:
- **Overestimates** match weights (double-counting evidence)
- Produces **biased m and u probability estimates**
- May give inflated confidence in matches

**Example of violation:**
- If `first_name` and `gender` are both comparison columns
- And they're highly correlated (most "John"s are male)
- An exact match on both provides less evidence than the model thinks

### Why BlockingPy's "block" Column is Safe

The key insight is that the "block" column is used for **blocking** (generating candidate pairs), NOT as a **comparison column**:

```python
settings = SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("block"),      # ✅ Used for BLOCKING only
    ],
    comparisons=[               # ← Comparison columns (where correlation matters)
        cl.NameComparison("first_name"),
        cl.NameComparison("surname"),
        cl.DateOfBirthComparison("dob"),
        # "block" is NOT here - no violation!
    ],
)
```

| Where Used | Role | Conditional Independence Applies? |
|------------|------|-----------------------------------|
| `blocking_rules_to_generate_predictions` | Generate candidate pairs | **No** - blocking is pre-processing |
| `comparisons` | Calculate match weights | **Yes** - this is where correlation matters |

**The "block" column never becomes a comparison column**, so it doesn't contribute to the match weight calculation.

### Potential Issues to Watch For

Although blocking itself doesn't violate conditional independence, there are **subtle bias issues** that can arise:

#### Issue 1: Training Bias from Restrictive Blocking

**Problem:** If you use `block_on("block")` for EM training, the training sample may not be representative.

**Why:** BlockingPy blocks are designed to have high recall for true matches. Training data from these blocks will have:
- Higher proportion of true matches than random sampling
- Biased distribution of comparison patterns

**Evidence from Splink issues:**
> "Using EM with sampled blocking will give essentially the same result (less precision, but no bias)... under the conditional independence assumption."
>
> "Due to violations of conditional independence, it often gives an answer that's wildly inaccurate."

#### Issue 2: Information Leakage

**Problem:** The "block" column is derived from the same fields used in comparisons.

```python
# BlockingPy input (concatenated fields)
df['txt'] = df['first_name'] + df['surname'] + df['dob'] + df['city'] + df['email']

# These SAME fields appear in comparisons
comparisons=[
    cl.NameComparison("first_name"),    # Also in txt
    cl.NameComparison("surname"),        # Also in txt
    cl.DateOfBirthComparison("dob"),    # Also in txt
]
```

This is generally fine because blocking is separate from comparison scoring, but be aware that records in the same block already have some similarity.

### Best Practices to Avoid Issues

#### ✅ DO: Use BlockingPy Blocks ONLY for Prediction

```python
settings = SettingsCreator(
    blocking_rules_to_generate_predictions=[
        block_on("block"),        # ✅ For prediction only
        block_on("first_name"),   # ✅ Deterministic fallbacks
        block_on("surname"),
    ],
    comparisons=[...],
)
```

#### ✅ DO: Use Different Blocking Rules for EM Training

```python
# Train WITHOUT BlockingPy blocks
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob"),  # ✅ Simple deterministic rule
    estimate_without_term_frequencies=True
)

linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname"),  # ✅ Different rule
    estimate_without_term_frequencies=True
)
```

#### ✅ DO: Use Round-Robin Training

Train on different blocking rules to avoid bias from any single rule:

```python
# Round-robin: each training pass uses different columns
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob")  # Trains first_name, surname
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")  # Trains dob
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("email")  # Cross-validates
)
```

#### ❌ DON'T: Use BlockingPy Blocks for Training

```python
# ❌ BAD: Don't train on BlockingPy blocks
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("block")  # ❌ Too restrictive, biased sample
)
```

#### ❌ DON'T: Add "block" as a Comparison Column

```python
# ❌ BAD: Never compare on the block column
comparisons=[
    cl.ExactMatch("block"),  # ❌ Creates correlation with everything!
    cl.NameComparison("first_name"),
]
```

#### ❌ DON'T: Rely Solely on BlockingPy for Blocking

```python
# ❌ RISKY: Only BlockingPy blocks
blocking_rules_to_generate_predictions=[
    block_on("block"),  # What if ANN misses some exact matches?
]

# ✅ BETTER: Combined approach
blocking_rules_to_generate_predictions=[
    block_on("block"),        # ANN catches fuzzy matches
    block_on("first_name"),   # Deterministic catches exact matches
    block_on("dob"),
]
```

### Recommended Integration Pattern

Here's the full recommended pattern from the Splink integration example:

```python
from splink import SettingsCreator, Linker, block_on, DuckDBAPI
import splink.comparison_library as cl
from blockingpy import Blocker

# 1. Create BlockingPy blocks
df['txt'] = df['first_name'].fillna('') + ' ' + \
            df['surname'].fillna('') + ' ' + \
            df['dob'].fillna('') + ' ' + \
            df['city'].fillna('') + ' ' + \
            df['email'].fillna('')

blocker = Blocker()
res = blocker.block(x=df['txt'], ann='hnsw', random_seed=42)
df = res.add_block_column(df)

# 2. Configure Splink - BlockingPy for PREDICTION only
settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("block"),        # BlockingPy (fuzzy)
        block_on("first_name"),   # Deterministic fallbacks
        block_on("surname"),
        block_on("dob"),
        block_on("email"),
    ],
    comparisons=[
        cl.ForenameSurnameComparison("first_name", "surname"),
        cl.DateOfBirthComparison("dob", input_is_string=True),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
        # ❌ NO: cl.ExactMatch("block") - never add this!
    ],
)

linker = Linker(df, settings, db_api=DuckDBAPI())

# 3. Train using DETERMINISTIC rules (NOT BlockingPy blocks)
linker.training.estimate_probability_two_random_records_match(
    deterministic_rules=[
        "l.first_name = r.first_name and levenshtein(r.dob, l.dob) <= 1",
        "l.surname = r.surname and levenshtein(r.dob, l.dob) <= 1",
        "l.email = r.email",
    ],
    recall=0.7
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=5)

# ✅ Train on simple deterministic rules, NOT block_on("block")
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob"), estimate_without_term_frequencies=True
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("email"), estimate_without_term_frequencies=True
)
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname"), estimate_without_term_frequencies=True
)

# 4. Predict - now BlockingPy blocks are used
predictions = linker.inference.predict(threshold_match_probability=0.9)
```

### Summary Table

| Concern | Risk Level | Mitigation |
|---------|------------|------------|
| **Using "block" for comparisons** | 🔴 HIGH | Never add "block" to comparisons |
| **Training EM on BlockingPy blocks** | 🟡 MEDIUM | Use deterministic rules for EM training |
| **Only using BlockingPy for blocking** | 🟡 MEDIUM | Add deterministic rules as fallback |
| **Using "block" for prediction blocking** | 🟢 LOW | This is the intended use case |
| **Information leakage** | 🟢 LOW | Inherent to all blocking methods |

### Conclusion

BlockingPy **does NOT inherently cause the correlation problems** that Splink warns about, because:

1. The "block" column is used for **blocking** (candidate pair generation), not **comparison scoring**
2. Splink's conditional independence assumption applies to **comparison columns only**
3. The match weight calculation never sees the "block" column

**However**, you must:
- ✅ Use BlockingPy blocks **only for prediction**, not training
- ✅ Train EM using **simple deterministic rules**
- ✅ Include **deterministic blocking rules** as fallbacks
- ❌ Never add "block" as a comparison column

---

## References

- [BlockingPy GitHub](https://github.com/ncn-foreigners/BlockingPy)
- [BlockingPy Splink Integration Example](https://blockingpy.readthedocs.io/en/stable/examples/splink_integration.html)
- [BlockingPy vs blocklib Comparison](https://blockingpy.readthedocs.io/en/stable/blocklib_comp.html)
- [Academic Paper: arXiv:2504.04266v3](https://arxiv.org/html/2504.04266v3)
- [Splink Documentation on Correlated Columns](https://moj-analytical-services.github.io/splink/)

