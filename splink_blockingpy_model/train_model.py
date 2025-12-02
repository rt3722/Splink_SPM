"""
SPM Training Script for Splink + BlockingPy Model

This script:
1. Loads the cleaned data from data_preparation.py output
2. Applies BlockingPy ANN-based blocking
3. Configures and trains a Splink model
4. Generates predictions and clusters
5. Saves outputs and visualizations

Author: SPM Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime

# BlockingPy for ANN-based blocking
from blockingpy import Blocker

# Splink for probabilistic record linkage
from splink import Linker, SettingsCreator, block_on, DuckDBAPI
import splink.comparison_library as cl

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "output"
OUTPUT_DIR = BASE_DIR / "model_output"

# Input files (from data_preparation.py)
TRAIN_DATA = DATA_DIR / "train_cleaned.parquet"
TRAIN_LOOKUP = DATA_DIR / "train_lookup.parquet"

# Model parameters
RANDOM_SEED = 42
MATCH_PROBABILITY_THRESHOLD = 0.9
CLUSTER_THRESHOLD = 0.95

# BlockingPy settings
BLOCKINGPY_ANN = 'hnsw'  # Options: 'faiss', 'hnsw', 'annoy', 'voyager'
BLOCKINGPY_K = 10        # Number of nearest neighbors

# BlockingPy text encoding settings - using model2vec for dense vector embeddings
# Using local path to avoid SSL issues with corporate proxy
MODEL2VEC_LOCAL_PATH = str(BASE_DIR / "models" / "potion-base-32M")

BLOCKINGPY_CONTROL_TXT = {
    "encoder": "embedding",  # Use dense vector embeddings instead of n-gram shingles
    "embedding": {
        "model": MODEL2VEC_LOCAL_PATH,  # Local path to pre-downloaded model2vec model
        "normalize": True,               # Normalize embeddings for cosine similarity
        "max_length": 512,               # Max tokens per text
        "emb_batch_size": 1024,          # Batch size for encoding
    }
}

# =============================================================================
# BLOCKING RULES
# =============================================================================

def get_blocking_rules():
    """
    Define blocking rules for Splink.
    These determine which record pairs are compared.
    """
    return [
        block_on("block"),                                    # BlockingPy ANN blocks
        block_on("firstname_cleaned", "lastname_cleaned"),    # Exact name match
        block_on("firstname_soundex", "lastname_soundex"),    # Phonetic blocking (Smith/Smyth)
        block_on("firstname_cleaned"),                        # First name only
        block_on("lastname_cleaned"),                         # Last name only
    ]


# =============================================================================
# COMPARISONS
# =============================================================================

def get_comparisons():
    """
    Define comparison functions for Splink.
    These determine how record pairs are scored.
    """
    return [
        # =====================================================================
        # NAME COMPARISONS
        # =====================================================================
        # Fuzzy name matching with Jaro-Winkler
        cl.NameComparison("firstname_cleaned"),
        cl.NameComparison("lastname_cleaned"),
        
        # Phonetic comparisons - same soundex = similar sounding names
        cl.ExactMatch("firstname_soundex"),
        cl.ExactMatch("lastname_soundex"),
        
        # Name array intersection for all name variations
        # [2, 1] = Level 1: ≥2 matches (strong), Level 2: ≥1 match (weaker)
        cl.ArrayIntersectAtSizes("names_array", [2, 1]),
        
        # =====================================================================
        # CONTACT COMPARISONS - at least 1 match is significant
        # =====================================================================
        cl.ArrayIntersectAtSizes("phones_array", [1]),     # ≥1 phone match
        cl.ArrayIntersectAtSizes("emails_array", [1]),     # ≥1 email match
        cl.ExactMatch("linkedin_cleaned"),
        
        # =====================================================================
        # EMPLOYMENT COMPARISONS - require at least 2 matches (more strict)
        # =====================================================================
        cl.ArrayIntersectAtSizes("employers_array", [2]),  # ≥2 employer matches
        cl.ArrayIntersectAtSizes("titles_array", [2]),     # ≥2 title matches
        
        # =====================================================================
        # LOCATION COMPARISONS - at least 1 match
        # =====================================================================
        cl.ArrayIntersectAtSizes("countries_array", [1]),
        cl.ArrayIntersectAtSizes("regions_array", [1]),
        cl.ArrayIntersectAtSizes("municipalities_array", [1]),
        
        # =====================================================================
        # EDUCATION - require at least 2 matches (more strict)
        # =====================================================================
        cl.ArrayIntersectAtSizes("degrees_array", [2]),    # ≥2 degree matches
        
        # NOTE: Numeric comparisons (months_experience, avg_months_per_employer)
        # are not supported in Splink's comparison_library. The key matching
        # signals are names, phones, emails, and linkedin which are already covered.
    ]


# =============================================================================
# DETERMINISTIC RULES (for estimating probability two random records match)
# =============================================================================

def get_deterministic_rules():
    """
    High-precision rules that almost certainly indicate a match.
    Used to estimate the prior probability of a match.
    
    Note: These rules should find obvious matches to help estimate
    the probability that two random records match.
    
    DuckDB array syntax:
    - list_intersect(arr1, arr2) returns common elements
    - array_length() or len() returns the count
    """
    return [
        # Exact match on first and last name
        block_on("firstname_cleaned", "lastname_cleaned"),
        
        # Share at least one phone number (very strong signal)
        "array_length(list_intersect(l.phones_array, r.phones_array)) >= 2",
        
        # Share at least one email (very strong signal)
        "array_length(list_intersect(l.emails_array, r.emails_array)) >= 2",
        
        # Compound rule: LinkedIn + experience + employer + title (all must match)
        # This is a very strong signal - same LinkedIn, similar experience, shared employer and title
        """
        l.linkedin_cleaned = r.linkedin_cleaned 
        AND l.linkedin_cleaned IS NOT NULL
        AND ABS(COALESCE(l.months_experience, 0) - COALESCE(r.months_experience, 0)) <= 1
        AND ABS(COALESCE(l.avg_months_per_employer, 0) - COALESCE(r.avg_months_per_employer, 0)) <= 1
        AND array_length(list_intersect(l.employers_array, r.employers_array)) >= 2
        AND array_length(list_intersect(l.titles_array, r.titles_array)) >= 2
        """,
    ]


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """Load cleaned data from parquet file."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def apply_blockingpy(df: pd.DataFrame, text_column: str = 'text_blob') -> pd.DataFrame:
    """
    Apply BlockingPy ANN-based blocking to the dataframe.
    Adds a 'block' column for use in Splink blocking rules.
    
    Uses model2vec for dense vector embeddings instead of n-gram shingles,
    providing more semantically meaningful blocking.
    """
    print("\n" + "=" * 60)
    print("STEP 1: BlockingPy Blocking (with model2vec embeddings)")
    print("=" * 60)
    
    # Filter out empty text blobs
    valid_mask = df[text_column].notna() & (df[text_column] != '')
    print(f"Records with valid text_blob: {valid_mask.sum()} / {len(df)}")
    
    # Initialize blocker
    blocker = Blocker()
    
    # Run blocking with model2vec dense vector encoding
    print(f"Running BlockingPy with {BLOCKINGPY_ANN} algorithm...")
    print(f"Using model2vec encoder (local): {BLOCKINGPY_CONTROL_TXT['embedding']['model']}")
    res = blocker.block(
        x=df[text_column],
        ann=BLOCKINGPY_ANN,
        random_seed=RANDOM_SEED,
        control_txt=BLOCKINGPY_CONTROL_TXT,  # Use model2vec dense vector embeddings
        verbose=1,
    )
    
    # Print blocking results
    print(res)
    
    # Add block column to dataframe
    df = res.add_block_column(df)
    
    print(f"Block column added. Unique blocks: {df['block'].nunique()}")
    
    return df


def create_linker(df: pd.DataFrame) -> Linker:
    """
    Create and configure the Splink Linker.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Creating Splink Linker")
    print("=" * 60)
    
    # Create settings
    settings = SettingsCreator(
        unique_id_column_name="unique_id",
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=get_blocking_rules(),
        comparisons=get_comparisons(),
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )
    
    # Create linker
    db_api = DuckDBAPI()
    linker = Linker(df, settings, db_api=db_api)
    
    print("Linker created successfully")
    
    return linker


def train_model(linker: Linker) -> None:
    """
    Train the Splink model using Expectation Maximization.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Training Model")
    print("=" * 60)
    
    # Step 3a: Estimate u probabilities from random sampling
    print("\n--- Estimating u probabilities (random sampling) ---")
    linker.training.estimate_u_using_random_sampling(max_pairs=1e8, seed=RANDOM_SEED)
    
    # Step 3b: Estimate probability two random records match
    print("\n--- Estimating probability two random records match ---")
    linker.training.estimate_probability_two_random_records_match(
        deterministic_matching_rules=get_deterministic_rules(),
        recall=0.6  # Estimated recall of the deterministic rules
    )
    
    # Step 3c: Estimate m probabilities using EM algorithm
    print("\n--- Estimating m probabilities (EM algorithm) ---")
    
    # Train on different blocking rules to get good estimates
    print("  Training on firstname_cleaned blocks...")
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("firstname_cleaned"),
        estimate_without_term_frequencies=True
    )
    
    print("  Training on lastname_cleaned blocks...")
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("lastname_cleaned"),
        estimate_without_term_frequencies=True
    )
    
    print("  Training on phonetic (soundex) blocks...")
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("firstname_soundex", "lastname_soundex"),
        estimate_without_term_frequencies=True
    )
    
    print("\nModel training complete!")


def generate_predictions(linker: Linker, threshold: float = 0.9):
    """
    Generate match predictions using the trained model.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Generating Predictions")
    print("=" * 60)
    
    print(f"Generating predictions with threshold: {threshold}")
    df_predictions = linker.inference.predict(threshold_match_probability=threshold)
    
    n_predictions = df_predictions.as_pandas_dataframe().shape[0]
    print(f"Generated {n_predictions} predicted matches")
    
    return df_predictions


def cluster_predictions(linker: Linker, df_predictions, threshold: float = 0.95):
    """
    Cluster predicted matches into entity groups.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Clustering Predictions")
    print("=" * 60)
    
    print(f"Clustering with threshold: {threshold}")
    df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        df_predictions,
        threshold_match_probability=threshold
    )
    
    df_clusters_pd = df_clusters.as_pandas_dataframe()
    n_clusters = df_clusters_pd['cluster_id'].nunique()
    print(f"Created {n_clusters} clusters from {len(df_clusters_pd)} records")
    
    return df_clusters


def save_outputs(linker: Linker, df_predictions, df_clusters, output_dir: Path):
    """
    Save model outputs, predictions, and visualizations.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Saving Outputs")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    predictions_path = output_dir / f"predictions_{timestamp}.parquet"
    df_predictions.as_pandas_dataframe().to_parquet(predictions_path)
    print(f"Saved predictions to {predictions_path}")
    
    # Save clusters
    clusters_path = output_dir / f"clusters_{timestamp}.parquet"
    df_clusters.as_pandas_dataframe().to_parquet(clusters_path)
    print(f"Saved clusters to {clusters_path}")
    
    # Save model settings
    model_path = output_dir / f"model_{timestamp}.json"
    linker.misc.save_model_to_json(str(model_path), overwrite=True)
    print(f"Saved model to {model_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    try:
        # Match weights chart
        chart = linker.visualisations.match_weights_chart()
        chart_path = output_dir / f"match_weights_{timestamp}.html"
        chart.save(str(chart_path))
        print(f"Saved match weights chart to {chart_path}")
    except Exception as e:
        print(f"Could not save match weights chart: {e}")
    
    try:
        # M-U parameters chart
        chart = linker.visualisations.m_u_parameters_chart()
        chart_path = output_dir / f"m_u_parameters_{timestamp}.html"
        chart.save(str(chart_path))
        print(f"Saved m-u parameters chart to {chart_path}")
    except Exception as e:
        print(f"Could not save m-u parameters chart: {e}")
    
    try:
        # Comparison viewer dashboard
        dashboard_path = output_dir / f"comparison_dashboard_{timestamp}.html"
        linker.visualisations.comparison_viewer_dashboard(
            df_predictions,
            str(dashboard_path),
            overwrite=True
        )
        print(f"Saved comparison dashboard to {dashboard_path}")
    except Exception as e:
        print(f"Could not save comparison dashboard: {e}")
    
    print("\nAll outputs saved successfully!")
    
    return {
        'predictions_path': predictions_path,
        'clusters_path': clusters_path,
        'model_path': model_path,
    }


def print_summary(df_original: pd.DataFrame, df_clusters):
    """
    Print summary statistics.
    """
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    df_clusters_pd = df_clusters.as_pandas_dataframe()
    
    n_records = len(df_original)
    n_clusters = df_clusters_pd['cluster_id'].nunique()
    
    # Cluster size distribution
    cluster_sizes = df_clusters_pd.groupby('cluster_id').size()
    
    print(f"Total records processed: {n_records}")
    print(f"Total clusters created: {n_clusters}")
    print(f"Reduction: {n_records - n_clusters} duplicate records identified")
    print(f"Deduplication rate: {(1 - n_clusters/n_records)*100:.2f}%")
    
    print("\nCluster size distribution:")
    size_counts = cluster_sizes.value_counts().sort_index()
    for size, count in size_counts.items():
        if size <= 10:
            print(f"  Size {size}: {count} clusters")
        elif size == size_counts.index.max():
            print(f"  Size {size}: {count} clusters (max)")


def main():
    """Main execution function."""
    print("=" * 60)
    print("SPM Splink + BlockingPy Model Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if input data exists
    if not TRAIN_DATA.exists():
        print(f"\nERROR: Training data not found at {TRAIN_DATA}")
        print("Please run data_preparation.py first.")
        return
    
    # Load data
    df = load_data(TRAIN_DATA)
    
    # Apply BlockingPy blocking
    df = apply_blockingpy(df)
    
    # Create Splink linker
    linker = create_linker(df)
    
    # Train model
    train_model(linker)
    
    # Generate predictions
    df_predictions = generate_predictions(linker, threshold=MATCH_PROBABILITY_THRESHOLD)
    
    # Cluster predictions
    df_clusters = cluster_predictions(linker, df_predictions, threshold=CLUSTER_THRESHOLD)
    
    # Save outputs
    output_paths = save_outputs(linker, df_predictions, df_clusters, OUTPUT_DIR)
    
    # Print summary
    print_summary(df, df_clusters)
    
    print("\n" + "=" * 60)
    print(f"Training complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return linker, df_predictions, df_clusters


if __name__ == "__main__":
    linker, predictions, clusters = main()

