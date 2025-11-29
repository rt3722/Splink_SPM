# Splink Documentation Links

This document contains all available links from the Splink Getting Started page.

**Source:** https://moj-analytical-services.github.io/splink/getting_started.html

**Last Updated:** November 26, 2025

---

## Main Navigation Links

- [Getting Started](https://moj-analytical-services.github.io/splink/getting_started.html)
- [Tutorial](https://moj-analytical-services.github.io/splink/demos/tutorials/00_Tutorial_Introduction.html)
- [Examples](https://moj-analytical-services.github.io/splink/demos/examples/examples_index.html)
- [API Docs](https://moj-analytical-services.github.io/splink/api_docs/api_docs_index.html)
- [User Guide](https://moj-analytical-services.github.io/splink/topic_guides/topic_guides_index.html)
- [Contributing](https://moj-analytical-services.github.io/splink/dev_guides/index.html)
- [Blog](https://moj-analytical-services.github.io/splink/blog/index.html)

---

## External Links

- [Splink GitHub Repository](https://github.com/moj-analytical-services/splink)
- [Splink Legacy Docs (v3)](https://moj-analytical-services.github.io/splink3_legacy_docs/index.html)
- [Discussion Forum](https://github.com/moj-analytical-services/splink/discussions)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

---

## Tutorials

- [Introduction](https://moj-analytical-services.github.io/splink/demos/tutorials/00_Tutorial_Introduction.html)
- [1. Data prep prerequisites](https://moj-analytical-services.github.io/splink/demos/tutorials/01_Prerequisites.html)
- [2. Exploratory analysis](https://moj-analytical-services.github.io/splink/demos/tutorials/02_Exploratory_analysis.html)
- [3. Blocking](https://moj-analytical-services.github.io/splink/demos/tutorials/03_Blocking.html)
- [4. Estimating model parameters](https://moj-analytical-services.github.io/splink/demos/tutorials/04_Estimating_model_parameters.html)
- [5. Predicting results](https://moj-analytical-services.github.io/splink/demos/tutorials/05_Predicting_results.html)
- [6. Visualising predictions](https://moj-analytical-services.github.io/splink/demos/tutorials/06_Visualising_predictions.html)
- [7. Evaluation](https://moj-analytical-services.github.io/splink/demos/tutorials/07_Evaluation.html)
- [8. Tips for building your own model](https://moj-analytical-services.github.io/splink/demos/tutorials/08_building_your_own_model.html)

---

## Examples

### DuckDB Examples

- [Deduplicate 50k rows historical persons](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/deduplicate_50k_synthetic.html)
- [Linking financial transactions](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/transactions.html)
- [Linking businesses](https://moj-analytical-services.github.io/splink/demos/examples/duckdb_no_test/business_rates_match.html)
- [Linking two tables of persons](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/link_only.html)
- [Real time record linkage](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/real_time_record_linkage.html)
- [Evaluation from ground truth column](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/accuracy_analysis_from_labels_column.html)
- [Estimating m probabilities from labels](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/pairwise_labels.html)
- [Quick and dirty persons model](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/quick_and_dirty_persons.html)
- [Deterministic dedupe](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/deterministic_dedupe.html)
- [Febrl3 Dedupe](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/febrl3.html)
- [Febrl4 link-only](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/febrl4.html)
- [Cookbook](https://moj-analytical-services.github.io/splink/demos/examples/duckdb_no_test/cookbook.html)
- [Investigating Bias in a Splink Model](https://moj-analytical-services.github.io/splink/demos/examples/duckdb_no_test/bias_eval.html)
- [Comparison playground](https://moj-analytical-services.github.io/splink/demos/examples/duckdb_no_test/comparison_playground.html)
- [Pseudopeople Census to ACS link](https://moj-analytical-services.github.io/splink/demos/examples/duckdb_no_test/pseudopeople-acs.html)

### PySpark Examples

- [Deduplication using Pyspark](https://moj-analytical-services.github.io/splink/demos/examples/spark/deduplicate_1k_synthetic.html)

### Athena Examples

- [Deduplicate 50k rows historical persons](https://moj-analytical-services.github.io/splink/demos/examples/athena/deduplicate_50k_synthetic.html)

### SQLite Examples

- [Deduplicate 50k rows historical persons](https://moj-analytical-services.github.io/splink/demos/examples/sqlite/deduplicate_50k_synthetic.html)

---

## API Documentation

### Linker API

- [Training](https://moj-analytical-services.github.io/splink/api_docs/training.html)
- [Visualisations](https://moj-analytical-services.github.io/splink/api_docs/visualisations.html)
- [Inference](https://moj-analytical-services.github.io/splink/api_docs/inference.html)
- [Clustering](https://moj-analytical-services.github.io/splink/api_docs/linker_clustering.html)
- [Evaluation](https://moj-analytical-services.github.io/splink/api_docs/evaluation.html)
- [Table Management](https://moj-analytical-services.github.io/splink/api_docs/table_management.html)
- [Miscellaneous functions](https://moj-analytical-services.github.io/splink/api_docs/misc.html)

### Comparisons Library

- [Comparison Library](https://moj-analytical-services.github.io/splink/api_docs/comparison_library.html)
- [Comparison Level Library](https://moj-analytical-services.github.io/splink/api_docs/comparison_level_library.html)

### Other APIs

- [Exploratory](https://moj-analytical-services.github.io/splink/api_docs/exploratory.html)
- [Blocking analysis](https://moj-analytical-services.github.io/splink/api_docs/blocking_analysis.html)
- [Blocking](https://moj-analytical-services.github.io/splink/api_docs/blocking.html)
- [Clustering](https://moj-analytical-services.github.io/splink/api_docs/clustering.html)
- [SplinkDataFrame](https://moj-analytical-services.github.io/splink/api_docs/splink_dataframe.html)
- [EM Training Session API](https://moj-analytical-services.github.io/splink/api_docs/em_training_session.html)
- [Column Expressions](https://moj-analytical-services.github.io/splink/api_docs/column_expression.html)

### In-build Datasets

- [SplinkDatasets](https://moj-analytical-services.github.io/splink/api_docs/datasets.html)

### Splink Settings

- [Settings Dict](https://moj-analytical-services.github.io/splink/api_docs/settings_dict_guide.html)

---

## User Guide

### Record Linkage Theory

- [Why do we need record linkage?](https://moj-analytical-services.github.io/splink/topic_guides/theory/record_linkage.html)
- [Probabilistic vs Deterministic linkage](https://moj-analytical-services.github.io/splink/topic_guides/theory/probabilistic_vs_deterministic.html)
- [The Fellegi-Sunter Model](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html)
- [Linked Data as Graphs](https://moj-analytical-services.github.io/splink/topic_guides/theory/linked_data_as_graphs.html)

### Linkage Models in Splink

- [Defining Splink models](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/settings.html)
- [Retrieving and querying Splink results](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/querying_splink_results.html)
- [Link type - linking vs deduping](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/link_type.html)

#### Splink's SQL backends

- [Backends overview](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/backends/backends.html)
- [PostgreSQL](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/backends/postgres.html)

### Data Preparation

- [Feature Engineering](https://moj-analytical-services.github.io/splink/topic_guides/data_preparation/feature_engineering.html)

### Blocking

- [What are Blocking Rules?](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html)
- [Computational Performance](https://moj-analytical-services.github.io/splink/topic_guides/blocking/performance.html)
- [Model Training Blocking Rules](https://moj-analytical-services.github.io/splink/topic_guides/blocking/model_training.html)

### Comparing Records

- [Comparisons and comparison levels](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/comparisons_and_comparison_levels.html)
- [Defining and customising comparisons](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/customising_comparisons.html)
- [Out-of-the-box comparisons](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/out_of_the_box_comparisons.html)
- [Term frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)

#### Comparing strings

- [String comparators](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/comparators.html)
- [Choosing string comparators](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/choosing_comparators.html)
- [Phonetic algorithms](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/phonetic.html)
- [Regular expressions](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/regular_expressions.html)

### Training

- [Training rationale](https://moj-analytical-services.github.io/splink/topic_guides/training/training_rationale.html)

### Evaluation

- [Overview](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/overview.html)
- [Model](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/model.html)

#### Edges (Links)

- [Overview](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/edge_overview.html)
- [Edge Metrics](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/edge_metrics.html)
- [Clerical Labelling](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/labelling.html)

#### Clusters

- [Overview](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/overview.html)
- [Graph metrics](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/graph_metrics.html)
- [How to compute graph metrics](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/how_to_compute_metrics.html)

### Performance

- [Run times, performance and linking large data](https://moj-analytical-services.github.io/splink/topic_guides/performance/drivers_of_performance.html)
- [Performance of comparison functions](https://moj-analytical-services.github.io/splink/topic_guides/performance/performance_of_comparison_functions.html)

#### Spark Performance

- [Optimising Spark performance](https://moj-analytical-services.github.io/splink/topic_guides/performance/optimising_spark.html)
- [Salting blocking rules](https://moj-analytical-services.github.io/splink/topic_guides/performance/salting.html)

#### DuckDB Performance

- [Optimising DuckDB performance](https://moj-analytical-services.github.io/splink/topic_guides/performance/optimising_duckdb.html)

### Charts Gallery

- [Charts Gallery](https://moj-analytical-services.github.io/splink/charts/index.html)

#### Exploratory Analysis

- [completeness chart](https://moj-analytical-services.github.io/splink/charts/completeness_chart.html)
- [profile columns](https://moj-analytical-services.github.io/splink/charts/profile_columns.html)

#### Blocking

- [cumulative num comparisons from blocking rules chart](https://moj-analytical-services.github.io/splink/charts/cumulative_comparisons_to_be_scored_from_blocking_rules_chart.html)

#### Similarity analysis

- [Comparator score chart](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/choosing_comparators.html#comparing-string-similarity-and-distance-scores)
- [Comparator score threshold chart](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/choosing_comparators.html#choosing-thresholds)
- [Phonetic match chart](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/choosing_comparators.html#phonetic-matching)

#### Model Training

- [comparison viewer dashboard](https://moj-analytical-services.github.io/splink/charts/comparison_viewer_dashboard.html)
- [match weights chart](https://moj-analytical-services.github.io/splink/charts/match_weights_chart.html)
- [m u parameters chart](https://moj-analytical-services.github.io/splink/charts/m_u_parameters_chart.html)
- [parameter estimate comparisons chart](https://moj-analytical-services.github.io/splink/charts/parameter_estimate_comparisons_chart.html)
- [tf adjustment chart](https://moj-analytical-services.github.io/splink/charts/tf_adjustment_chart.html)
- [unlinkables chart](https://moj-analytical-services.github.io/splink/charts/unlinkables_chart.html)
- [waterfall chart](https://moj-analytical-services.github.io/splink/charts/waterfall_chart.html)

#### Clustering

- [cluster studio dashboard](https://moj-analytical-services.github.io/splink/charts/cluster_studio_dashboard.html)

#### Model Evaluation

- [accuracy chart from labels table](https://moj-analytical-services.github.io/splink/charts/accuracy_analysis_from_labels_table.html)
- [threshold selection tool](https://moj-analytical-services.github.io/splink/charts/threshold_selection_tool_from_labels_table.html)

### LLM Integration

- [LLM prompts](https://moj-analytical-services.github.io/splink/topic_guides/llms/prompting_llms.html)

---

## Contributing

### Contributing to Splink

- [Contributor Guide](https://moj-analytical-services.github.io/splink/dev_guides/CONTRIBUTING.html)
- [Development Quickstart](https://moj-analytical-services.github.io/splink/dev_guides/changing_splink/development_quickstart.html)
- [Linting and Formatting](https://moj-analytical-services.github.io/splink/dev_guides/changing_splink/lint_and_format.html)
- [Testing](https://moj-analytical-services.github.io/splink/dev_guides/changing_splink/testing.html)
- [Contributing to Documentation](https://moj-analytical-services.github.io/splink/dev_guides/changing_splink/contributing_to_docs.html)
- [Managing Environment and Dependencies](https://moj-analytical-services.github.io/splink/dev_guides/changing_splink/managing_dependencies_with_uv.html)
- [Releasing a Package Version](https://moj-analytical-services.github.io/splink/dev_guides/changing_splink/releases.html)
- [Contributing to the Splink Blog](https://moj-analytical-services.github.io/splink/dev_guides/changing_splink/blog_posts.html)

### How Splink works

- [Understanding and debugging Splink](https://moj-analytical-services.github.io/splink/dev_guides/debug_modes.html)
- [Transpilation using sqlglot](https://moj-analytical-services.github.io/splink/dev_guides/transpilation.html)

#### Performance and caching

- [Caching and pipelining](https://moj-analytical-services.github.io/splink/dev_guides/caching.html)
- [Spark caching](https://moj-analytical-services.github.io/splink/dev_guides/spark_pipelining_and_caching.html)

#### Charts

- [Understanding and editing charts](https://moj-analytical-services.github.io/splink/dev_guides/charts/understanding_and_editing_charts.html)
- [Building new charts](https://moj-analytical-services.github.io/splink/dev_guides/charts/building_charts.html)

- [User-Defined Functions](https://moj-analytical-services.github.io/splink/dev_guides/udfs.html)
- [Dependency Compatibility Policy](https://moj-analytical-services.github.io/splink/dev_guides/dependency_compatibility_policy.html)

---

## Blog Categories

- [Bias](https://moj-analytical-services.github.io/splink/blog/category/bias.html)
- [Ethics](https://moj-analytical-services.github.io/splink/blog/category/ethics.html)
- [Feature Updates](https://moj-analytical-services.github.io/splink/blog/category/feature-updates.html)

---

## Summary

Total number of unique links: 180+

This document provides a comprehensive reference to all documentation links available from the Splink Getting Started page, organized by category for easy navigation.

