"""
SPM Data Preparation Script for Splink + BlockingPy Model

This script handles:
1. Loading and splitting data (90% train, 10% test)
2. Splitting test set into batches (20%, 40%, 40%)
3. Creating lookup table with unique_id for Splink
4. Cleaning and standardizing all fields
5. Creating text blob for BlockingPy embeddings
6. Combining cleaned fields into arrays

Author: SPM Team
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import List, Optional, Tuple, Set
import warnings
from tqdm import tqdm

# Phone number parsing
import phonenumbers
from phonenumbers import NumberParseException

# NLP for name extraction
import spacy

# Phonetic encoding (Soundex, Metaphone)
import jellyfish

# Text normalization
from unidecode import unidecode

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

INPUT_FILE = DATA_DIR / "spm_data_latest.tsv"

# Train/Test split ratio
TRAIN_RATIO = 0.90
TEST_RATIO = 0.10

# Test batch split ratios
BATCH1_RATIO = 0.20
BATCH2_RATIO = 0.40
BATCH3_RATIO = 0.40

# LinkedIn URL prefix to validate
LINKEDIN_PREFIX = "https://linkedin.com/in/"
LINKEDIN_PREFIX_ALT = "https://www.linkedin.com/in/"

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_spacy_model():
    """Load spaCy model for NER, downloading if necessary."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model 'en_core_web_sm'...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def is_null_or_empty(value) -> bool:
    """Check if a value should be treated as null."""
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in ['nan', 'none', 'null', '']
    return False


def to_splink_null(value):
    """
    Convert empty strings and invalid values to None for Splink.
    Splink requires actual None/NULL values, not empty strings.
    """
    if is_null_or_empty(value):
        return None
    return value


def clean_text(text: str) -> Optional[str]:
    """Basic text cleaning: strip whitespace, normalize unicode."""
    if is_null_or_empty(text):
        return None
    text = str(text).strip()
    # Normalize unicode characters
    text = unidecode(text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text if text else None


def extract_alpha_only(text: str) -> Optional[str]:
    """Extract only alphabetic characters from text."""
    if is_null_or_empty(text):
        return None
    # Keep only letters
    alpha_only = re.sub(r'[^a-zA-Z]', '', str(text))
    return alpha_only.lower() if alpha_only else None


def extract_digits_only(text: str) -> Optional[str]:
    """Extract only digits from text."""
    if is_null_or_empty(text):
        return None
    digits = re.sub(r'[^\d]', '', str(text))
    return digits if digits else None


# =============================================================================
# NAME CLEANING FUNCTIONS
# =============================================================================

def extract_names_from_text(text: str, nlp) -> Set[str]:
    """
    Use spaCy NER to extract person names from text.
    Returns a set of unique names found.
    """
    if is_null_or_empty(text):
        return set()
    
    text = str(text).strip()
    doc = nlp(text)
    
    names = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Clean and normalize the name
            name = clean_text(ent.text)
            if name:
                # Split into parts and add each part
                for part in name.split():
                    cleaned_part = re.sub(r'[^a-zA-Z\s]', '', part).strip()
                    if cleaned_part and len(cleaned_part) > 1:
                        names.add(cleaned_part.lower())
    
    # Also try to extract names without NER (fallback for simple names)
    # Remove common prefixes/suffixes and quotes
    text_cleaned = re.sub(r'^["\']+|["\']+$', '', text)
    text_cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', text_cleaned)
    text_cleaned = re.sub(r'Phone:.*|Mobile:.*|Email:.*|http.*', '', text_cleaned, flags=re.IGNORECASE)
    
    # Split by common delimiters
    parts = re.split(r'[,;|\n\t]+', text_cleaned)
    for part in parts:
        part = part.strip()
        # Check if it looks like a name (2-3 words, mostly letters)
        words = part.split()
        if 1 <= len(words) <= 4:
            for word in words:
                cleaned_word = re.sub(r'[^a-zA-Z]', '', word).strip()
                if cleaned_word and len(cleaned_word) > 1:
                    names.add(cleaned_word.lower())
    
    return names


def clean_firstname(firstname: str) -> Optional[str]:
    """
    Clean FIRSTNAME column for exact match blocking.
    Returns cleaned lowercase first name.
    """
    if is_null_or_empty(firstname):
        return None
    
    fn = clean_text(firstname)
    if not fn:
        return None
    
    # Keep only alphabetic characters, normalize to lowercase
    fn_clean = re.sub(r'[^a-zA-Z]', '', fn).lower().strip()
    
    return fn_clean if fn_clean and len(fn_clean) > 1 else None


def clean_lastname(lastname: str) -> Optional[str]:
    """
    Clean LASTNAME column for exact match blocking.
    Returns cleaned lowercase last name.
    """
    if is_null_or_empty(lastname):
        return None
    
    ln = clean_text(lastname)
    if not ln:
        return None
    
    # Keep only alphabetic characters, normalize to lowercase
    ln_clean = re.sub(r'[^a-zA-Z]', '', ln).lower().strip()
    
    return ln_clean if ln_clean and len(ln_clean) > 1 else None


def get_soundex(name: Optional[str]) -> Optional[str]:
    """
    Get Soundex phonetic encoding of a name.
    Useful for blocking on names that sound similar but are spelled differently.
    Example: "Smith" and "Smyth" both encode to "S530"
    """
    if not name:
        return None
    
    try:
        # Soundex requires at least one letter
        if not any(c.isalpha() for c in name):
            return None
        return jellyfish.soundex(name)
    except Exception:
        return None


def get_metaphone(name: Optional[str]) -> Optional[str]:
    """
    Get Metaphone phonetic encoding of a name.
    More accurate than Soundex for English names.
    """
    if not name:
        return None
    
    try:
        if not any(c.isalpha() for c in name):
            return None
        return jellyfish.metaphone(name)
    except Exception:
        return None


# =============================================================================
# NUMERIC COLUMN CLEANING
# =============================================================================

def clean_numeric_column(value, allow_zero: bool = True) -> Optional[float]:
    """
    Clean and convert numeric columns to float.
    Returns None for invalid/empty values.
    """
    if is_null_or_empty(value):
        return None
    
    try:
        num = float(value)
        # Check for invalid numbers
        if pd.isna(num) or np.isinf(num):
            return None
        # Optionally exclude zeros
        if not allow_zero and num == 0:
            return None
        return num
    except (ValueError, TypeError):
        return None


def clean_months_experience(value) -> Optional[float]:
    """
    Clean SOV_MONTHS_OF_WORK_EXPERIENCE column.
    Returns months as float, None for invalid values.
    """
    num = clean_numeric_column(value, allow_zero=True)
    if num is not None and num < 0:
        return None  # Negative months doesn't make sense
    return num


def clean_avg_months_per_employer(value) -> Optional[float]:
    """
    Clean SOV_AVG_MONTHS_PER_EMPLOYER column.
    Returns average months as float, None for invalid values.
    """
    num = clean_numeric_column(value, allow_zero=True)
    if num is not None and num < 0:
        return None  # Negative months doesn't make sense
    return num


def build_names_array(row: pd.Series, nlp, linkedin_cleaned: Optional[str], 
                       firstname_cleaned: Optional[str], lastname_cleaned: Optional[str]) -> List[str]:
    """
    Build names_array from:
    - NAME_1, NAME_2, NAME_3 (extracted via spaCy)
    - linkedin_cleaned 
    - Combined FIRSTNAME + LASTNAME (for deduplication)
    
    Returns a list of unique name values.
    """
    all_names = set()
    
    # Extract names from NAME_1, NAME_2, NAME_3 using spaCy
    for col in ['NAME_1', 'NAME_2', 'NAME_3']:
        if col in row and not is_null_or_empty(row[col]):
            names = extract_names_from_text(row[col], nlp)
            all_names.update(names)
    
    # Add linkedin_cleaned if available
    if linkedin_cleaned:
        all_names.add(linkedin_cleaned)
    
    # Add combined FIRSTNAME + LASTNAME for deduplication
    # This helps catch when the same person appears with different spellings
    if firstname_cleaned and lastname_cleaned:
        combined_name = f"{firstname_cleaned}{lastname_cleaned}"
        all_names.add(combined_name)
        # Also add them individually (they might already be captured from NAME columns)
        all_names.add(firstname_cleaned)
        all_names.add(lastname_cleaned)
    elif firstname_cleaned:
        all_names.add(firstname_cleaned)
    elif lastname_cleaned:
        all_names.add(lastname_cleaned)
    
    return sorted(list(all_names)) if all_names else None


# =============================================================================
# LINKEDIN URL CLEANING
# =============================================================================

def clean_linkedin_url(url: str) -> Optional[str]:
    """
    Clean LinkedIn URL:
    1. Check if it starts with https://linkedin.com/in/ or https://www.linkedin.com/in/
    2. If not, return None (Splink null)
    3. If yes, extract the profile part and keep only alphabetic characters
    """
    if is_null_or_empty(url):
        return None
    
    url = str(url).strip().lower()
    
    # Check for valid LinkedIn URL prefixes
    profile_part = None
    if url.startswith(LINKEDIN_PREFIX.lower()):
        profile_part = url[len(LINKEDIN_PREFIX):]
    elif url.startswith(LINKEDIN_PREFIX_ALT.lower()):
        profile_part = url[len(LINKEDIN_PREFIX_ALT):]
    elif url.startswith("linkedin.com/in/"):
        profile_part = url[len("linkedin.com/in/"):]
    elif url.startswith("www.linkedin.com/in/"):
        profile_part = url[len("www.linkedin.com/in/"):]
    else:
        # Invalid LinkedIn URL
        return None
    
    if not profile_part:
        return None
    
    # Remove any trailing paths or query params
    profile_part = profile_part.split('/')[0].split('?')[0]
    
    # Extract only alphabetic characters
    alpha_only = re.sub(r'[^a-zA-Z]', '', profile_part)
    
    return alpha_only.lower() if alpha_only else None


# =============================================================================
# PHONE NUMBER CLEANING
# =============================================================================

def clean_phone_number(phone: str, default_region: str = "US") -> Optional[str]:
    """
    Clean and standardize phone number using phonenumbers library.
    Returns only the digits of a valid phone number.
    """
    if is_null_or_empty(phone):
        return None
    
    phone_str = str(phone).strip()
    
    # Remove common invalid placeholders
    invalid_patterns = ['0000000000', '1111111111', '1234567890', '9999999999']
    digits_only = re.sub(r'[^\d]', '', phone_str)
    
    if digits_only in invalid_patterns:
        return None
    
    # Skip very short or very long numbers
    if len(digits_only) < 7 or len(digits_only) > 15:
        return None
    
    try:
        # Try to parse with phonenumbers library
        parsed = phonenumbers.parse(phone_str, default_region)
        if phonenumbers.is_valid_number(parsed):
            # Return the national number as digits
            national_num = str(parsed.national_number)
            return national_num if national_num else None
    except NumberParseException:
        pass
    
    # Fallback: just return the digits if reasonable length
    if 7 <= len(digits_only) <= 11:
        return digits_only
    
    return None


def clean_phone_columns(row: pd.Series) -> List[str]:
    """
    Clean all phone columns and return unique phone numbers.
    """
    phone_columns = ['PHONE', 'MOBILEPHONE', 'HOMEPHONE', 'OTHERPHONE', 
                     'PHONE_1', 'PHONE_2', 'PHONE_3']
    
    unique_phones = set()
    
    for col in phone_columns:
        if col in row and not is_null_or_empty(row[col]):
            cleaned = clean_phone_number(row[col])
            if cleaned:
                unique_phones.add(cleaned)
    
    return sorted(list(unique_phones)) if unique_phones else None


# =============================================================================
# EMPLOYER/TITLE/LOCATION CLEANING
# =============================================================================

def clean_employer_name(employer: str) -> Optional[str]:
    """Clean and standardize employer name."""
    if is_null_or_empty(employer):
        return None
    
    employer = clean_text(employer)
    if not employer:
        return None
    
    # Remove common suffixes
    employer = re.sub(r'\s+(Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Co\.?)$', 
                      '', employer, flags=re.IGNORECASE)
    
    # Normalize to lowercase
    employer = employer.lower().strip()
    
    return employer if employer else None


def clean_title(title: str) -> Optional[str]:
    """Clean and standardize job title."""
    if is_null_or_empty(title):
        return None
    
    title = clean_text(title)
    if not title:
        return None
    
    # Normalize to lowercase
    title = title.lower().strip()
    
    return title if title else None


def clean_country(country: str) -> Optional[str]:
    """Clean and standardize country code/name."""
    if is_null_or_empty(country):
        return None
    
    country = clean_text(country)
    if not country:
        return None
    
    # Normalize common variations
    country = country.upper().strip()
    
    # Map common variations
    country_map = {
        'USA': 'US',
        'UNITED STATES': 'US',
        'UNITED STATES OF AMERICA': 'US',
        'CAN': 'CA',
        'CANADA': 'CA',
        'UK': 'GB',
        'UNITED KINGDOM': 'GB',
        'GREAT BRITAIN': 'GB',
    }
    
    return country_map.get(country, country)


def clean_region(region: str) -> Optional[str]:
    """Clean and standardize region/state."""
    if is_null_or_empty(region):
        return None
    
    region = clean_text(region)
    if not region:
        return None
    
    return region.upper().strip()


def clean_municipality(municipality: str) -> Optional[str]:
    """Clean and standardize municipality/city."""
    if is_null_or_empty(municipality):
        return None
    
    municipality = clean_text(municipality)
    if not municipality:
        return None
    
    return municipality.lower().strip()


def clean_degree(degree: str) -> Optional[str]:
    """Clean and standardize degree field."""
    if is_null_or_empty(degree):
        return None
    
    degree = clean_text(degree)
    if not degree:
        return None
    
    # Normalize common degree abbreviations
    degree = degree.lower().strip()
    
    # Standardize common degrees
    degree_map = {
        'bachelor of science': 'bs',
        'bachelor of arts': 'ba',
        'master of science': 'ms',
        'master of arts': 'ma',
        'master of business administration': 'mba',
        'doctor of philosophy': 'phd',
        'high school diploma': 'hs diploma',
        'high school': 'hs diploma',
        'ged': 'ged',
        'associate': 'associate',
        'associates': 'associate',
        "associate's": 'associate',
        'certificate': 'certificate',
        'diploma': 'diploma',
    }
    
    for pattern, replacement in degree_map.items():
        if pattern in degree:
            return replacement
    
    return degree


# =============================================================================
# ARRAY COLUMN BUILDERS
# =============================================================================

def build_employer_array(row: pd.Series) -> List[str]:
    """Build array of unique cleaned employer names."""
    employer_cols = ['EMPLOYER_1', 'EMPLOYER_2', 'EMPLOYER_3', 'EMPLOYER_4']
    
    unique_employers = set()
    for col in employer_cols:
        if col in row:
            cleaned = clean_employer_name(row[col])
            if cleaned:
                unique_employers.add(cleaned)
    
    return sorted(list(unique_employers)) if unique_employers else None


def build_title_array(row: pd.Series) -> List[str]:
    """Build array of unique cleaned job titles."""
    title_cols = ['TITLE_1_1', 'TITLE_1_2', 'TITLE_2_1', 'TITLE_2_2', 
                  'TITLE_3_1', 'TITLE_3_2', 'TITLE_4_1', 'TITLE_4_2']
    
    unique_titles = set()
    for col in title_cols:
        if col in row:
            cleaned = clean_title(row[col])
            if cleaned:
                unique_titles.add(cleaned)
    
    return sorted(list(unique_titles)) if unique_titles else None


def build_country_array(row: pd.Series) -> List[str]:
    """Build array of unique cleaned countries."""
    country_cols = ['COUNTRY_1_1', 'COUNTRY_1_2', 'COUNTRY_2_1', 'COUNTRY_2_2',
                    'COUNTRY_3_1', 'COUNTRY_3_2', 'COUNTRY_4_1', 'COUNTRY_4_2']
    
    unique_countries = set()
    for col in country_cols:
        if col in row:
            cleaned = clean_country(row[col])
            if cleaned:
                unique_countries.add(cleaned)
    
    return sorted(list(unique_countries)) if unique_countries else None


def build_region_array(row: pd.Series) -> List[str]:
    """Build array of unique cleaned regions/states."""
    region_cols = ['REGION_1_1', 'REGION_1_2', 'REGION_2_1', 'REGION_2_2',
                   'REGION_3_1', 'REGION_3_2', 'REGION_4_1', 'REGION_4_2']
    
    unique_regions = set()
    for col in region_cols:
        if col in row:
            cleaned = clean_region(row[col])
            if cleaned:
                unique_regions.add(cleaned)
    
    return sorted(list(unique_regions)) if unique_regions else None


def build_municipality_array(row: pd.Series) -> List[str]:
    """Build array of unique cleaned municipalities/cities."""
    municipality_cols = ['MUNICIPALITY_1_1', 'MUNICIPALITY_1_2', 'MUNICIPALITY_2_1', 'MUNICIPALITY_2_2',
                         'MUNICIPALITY_3_1', 'MUNICIPALITY_3_2', 'MUNICIPALITY_4_1', 'MUNICIPALITY_4_2']
    
    unique_municipalities = set()
    for col in municipality_cols:
        if col in row:
            cleaned = clean_municipality(row[col])
            if cleaned:
                unique_municipalities.add(cleaned)
    
    return sorted(list(unique_municipalities)) if unique_municipalities else None


def build_degree_array(row: pd.Series) -> List[str]:
    """Build array of unique cleaned degrees."""
    degree_cols = ['DEGREE_1_1', 'DEGREE_1_2', 'DEGREE_2_1', 'DEGREE_2_2',
                   'DEGREE_3_1', 'DEGREE_3_2']
    
    unique_degrees = set()
    for col in degree_cols:
        if col in row:
            cleaned = clean_degree(row[col])
            if cleaned:
                unique_degrees.add(cleaned)
    
    return sorted(list(unique_degrees)) if unique_degrees else None


def build_email_array(row: pd.Series) -> List[str]:
    """Build array of unique cleaned emails."""
    email_cols = ['EMAIL', 'EMAIL_1', 'EMAIL_2', 'EMAIL_3']
    
    unique_emails = set()
    for col in email_cols:
        if col in row and not is_null_or_empty(row[col]):
            email = str(row[col]).strip().lower()
            # Basic email validation
            if '@' in email and '.' in email:
                # Remove common invalid patterns
                if not email.startswith('http') and email not in ['nan', 'none', 'null']:
                    unique_emails.add(email)
    
    return sorted(list(unique_emails)) if unique_emails else None


# =============================================================================
# TEXT BLOB FOR BLOCKINGPY
# =============================================================================

def build_text_blob(row: pd.Series, cleaned_data: dict) -> str:
    """
    Build a text blob combining all cleaned and standardized fields.
    This will be used for BlockingPy embeddings.
    
    Format: "column_name: value; column_name: value; ..."
    
    Includes both text fields and numeric fields for comprehensive matching.
    """
    parts = []
    
    # Add cleaned first name and last name (from cleaned_data, not raw row)
    if cleaned_data.get('firstname_cleaned'):
        parts.append(f"firstname: {cleaned_data['firstname_cleaned']}")
    
    if cleaned_data.get('lastname_cleaned'):
        parts.append(f"lastname: {cleaned_data['lastname_cleaned']}")
    
    # Add cleaned names
    if cleaned_data.get('names_array'):
        parts.append(f"names: {' '.join(cleaned_data['names_array'])}")
    
    # Add LinkedIn
    if cleaned_data.get('linkedin_cleaned'):
        parts.append(f"linkedin: {cleaned_data['linkedin_cleaned']}")
    
    # Add phones
    if cleaned_data.get('phones_array'):
        parts.append(f"phones: {' '.join(cleaned_data['phones_array'])}")
    
    # Add emails
    if cleaned_data.get('emails_array'):
        parts.append(f"emails: {' '.join(cleaned_data['emails_array'])}")
    
    # Add employers
    if cleaned_data.get('employers_array'):
        parts.append(f"employers: {' '.join(cleaned_data['employers_array'])}")
    
    # Add titles
    if cleaned_data.get('titles_array'):
        parts.append(f"titles: {' '.join(cleaned_data['titles_array'])}")
    
    # Add countries
    if cleaned_data.get('countries_array'):
        parts.append(f"countries: {' '.join(cleaned_data['countries_array'])}")
    
    # Add regions
    if cleaned_data.get('regions_array'):
        parts.append(f"regions: {' '.join(cleaned_data['regions_array'])}")
    
    # Add municipalities
    if cleaned_data.get('municipalities_array'):
        parts.append(f"municipalities: {' '.join(cleaned_data['municipalities_array'])}")
    
    # Add degrees
    if cleaned_data.get('degrees_array'):
        parts.append(f"degrees: {' '.join(cleaned_data['degrees_array'])}")
    
    # Add location from CONTACT_LOCATION__C
    if 'CONTACT_LOCATION__C' in row and not is_null_or_empty(row['CONTACT_LOCATION__C']):
        loc = clean_text(row['CONTACT_LOCATION__C'])
        if loc:
            parts.append(f"location: {loc.lower()}")
    
    # Add numeric fields for work experience
    # These help BlockingPy find similar candidates by experience level
    if cleaned_data.get('months_experience') is not None:
        # Round to nearest year for text blob (more semantic grouping)
        years_exp = round(cleaned_data['months_experience'] / 12, 1)
        parts.append(f"experience_years: {years_exp}")
    
    if cleaned_data.get('avg_months_per_employer') is not None:
        # Round to nearest year for text blob
        avg_years = round(cleaned_data['avg_months_per_employer'] / 12, 1)
        parts.append(f"avg_tenure_years: {avg_years}")
    
    return "; ".join(parts) if parts else ""


# =============================================================================
# MAIN DATA PROCESSING
# =============================================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """Load the TSV data file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep='\t', dtype=str, low_memory=False)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def create_lookup_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create lookup table with RESUME_ID, FILTERED_TALENT_ID, BATCH_TYPE and unique_id.
    Returns (lookup_table, main_table_with_unique_id)
    """
    print("Creating lookup table...")
    
    # Create unique_id as integer (required by Splink)
    lookup_df = df[['RESUME_ID', 'FILTERED_TALENT_ID', 'BATCH_TYPE']].copy()
    lookup_df['unique_id'] = range(len(lookup_df))
    
    # Add unique_id to main dataframe
    main_df = df.copy()
    main_df['unique_id'] = range(len(main_df))
    
    # Remove the original ID columns from main table (they're in lookup now)
    main_df = main_df.drop(columns=['RESUME_ID', 'FILTERED_TALENT_ID', 'BATCH_TYPE'])
    
    print(f"Lookup table created with {len(lookup_df)} records")
    return lookup_df, main_df


def split_train_test(df: pd.DataFrame, lookup_df: pd.DataFrame, 
                     train_ratio: float = 0.9, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    Returns (train_df, test_df, train_lookup, test_lookup)
    """
    print(f"Splitting data into train ({train_ratio*100}%) and test ({(1-train_ratio)*100}%)...")
    
    np.random.seed(random_seed)
    
    n = len(df)
    indices = np.random.permutation(n)
    train_size = int(n * train_ratio)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    train_lookup = lookup_df.iloc[train_indices].reset_index(drop=True)
    test_lookup = lookup_df.iloc[test_indices].reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    return train_df, test_df, train_lookup, test_lookup


def split_test_into_batches(test_df: pd.DataFrame, test_lookup: pd.DataFrame,
                            batch_ratios: Tuple[float, float, float] = (0.2, 0.4, 0.4),
                            random_seed: int = 42) -> Tuple[dict, dict]:
    """
    Split test set into batches with specified ratios.
    Returns (batch_dfs, batch_lookups) dictionaries
    """
    print(f"Splitting test set into batches: {batch_ratios}...")
    
    np.random.seed(random_seed)
    
    n = len(test_df)
    indices = np.random.permutation(n)
    
    batch1_size = int(n * batch_ratios[0])
    batch2_size = int(n * batch_ratios[1])
    
    batch1_indices = indices[:batch1_size]
    batch2_indices = indices[batch1_size:batch1_size + batch2_size]
    batch3_indices = indices[batch1_size + batch2_size:]
    
    batch_dfs = {
        'batch1': test_df.iloc[batch1_indices].reset_index(drop=True),
        'batch2': test_df.iloc[batch2_indices].reset_index(drop=True),
        'batch3': test_df.iloc[batch3_indices].reset_index(drop=True),
    }
    
    batch_lookups = {
        'batch1': test_lookup.iloc[batch1_indices].reset_index(drop=True),
        'batch2': test_lookup.iloc[batch2_indices].reset_index(drop=True),
        'batch3': test_lookup.iloc[batch3_indices].reset_index(drop=True),
    }
    
    for name, batch in batch_dfs.items():
        print(f"  {name}: {len(batch)} records ({len(batch)/n*100:.1f}%)")
    
    return batch_dfs, batch_lookups


def process_record(row: pd.Series, nlp) -> dict:
    """Process a single record and return cleaned data."""
    cleaned = {}
    
    # Clean FIRSTNAME and LASTNAME separately (for exact match blocking in Splink)
    cleaned['firstname_cleaned'] = clean_firstname(row.get('FIRSTNAME'))
    cleaned['lastname_cleaned'] = clean_lastname(row.get('LASTNAME'))
    
    # Phonetic encoding for blocking (catches spelling variations like Smith/Smyth)
    cleaned['firstname_soundex'] = get_soundex(cleaned['firstname_cleaned'])
    cleaned['lastname_soundex'] = get_soundex(cleaned['lastname_cleaned'])
    cleaned['firstname_metaphone'] = get_metaphone(cleaned['firstname_cleaned'])
    cleaned['lastname_metaphone'] = get_metaphone(cleaned['lastname_cleaned'])
    
    # Clean LinkedIn URL
    linkedin_col = 'LINKEDIN_URL__C'
    if linkedin_col in row:
        cleaned['linkedin_cleaned'] = clean_linkedin_url(row[linkedin_col])
    else:
        cleaned['linkedin_cleaned'] = None
    
    # Build names_array from NAME_1,2,3 + linkedin_cleaned + combined firstname/lastname
    cleaned['names_array'] = build_names_array(
        row, nlp, 
        cleaned['linkedin_cleaned'],
        cleaned['firstname_cleaned'],
        cleaned['lastname_cleaned']
    )
    
    # Clean phone numbers
    cleaned['phones_array'] = clean_phone_columns(row)
    
    # Clean emails
    cleaned['emails_array'] = build_email_array(row)
    
    # Clean employer names
    cleaned['employers_array'] = build_employer_array(row)
    
    # Clean titles
    cleaned['titles_array'] = build_title_array(row)
    
    # Clean countries
    cleaned['countries_array'] = build_country_array(row)
    
    # Clean regions
    cleaned['regions_array'] = build_region_array(row)
    
    # Clean municipalities
    cleaned['municipalities_array'] = build_municipality_array(row)
    
    # Clean degrees
    cleaned['degrees_array'] = build_degree_array(row)
    
    # Clean numeric columns for work experience
    cleaned['months_experience'] = clean_months_experience(row.get('SOV_MONTHS_OF_WORK_EXPERIENCE'))
    cleaned['avg_months_per_employer'] = clean_avg_months_per_employer(row.get('SOV_AVG_MONTHS_PER_EMPLOYER'))
    
    # Build text blob for BlockingPy (includes numeric fields)
    cleaned['text_blob'] = build_text_blob(row, cleaned)
    
    # Keep description column for BlockingPy
    desc_col = 'SOV_DESCRIPTION'
    if desc_col in row and not is_null_or_empty(row[desc_col]):
        cleaned['description'] = clean_text(row[desc_col])
    else:
        cleaned['description'] = None
    
    return cleaned


def process_dataframe(df: pd.DataFrame, nlp, desc: str = "Processing") -> pd.DataFrame:
    """Process entire dataframe and add cleaned columns."""
    print(f"{desc}...")
    
    # Initialize new columns
    new_columns = {
        'unique_id': [],
        'firstname_cleaned': [],       # For exact match blocking
        'lastname_cleaned': [],        # For exact match blocking
        'firstname_soundex': [],       # Phonetic blocking (catches spelling variations)
        'lastname_soundex': [],        # Phonetic blocking
        'firstname_metaphone': [],     # Alternative phonetic encoding
        'lastname_metaphone': [],      # Alternative phonetic encoding
        'names_array': [],             # For array intersection/fuzzy matching
        'linkedin_cleaned': [],
        'phones_array': [],
        'emails_array': [],
        'employers_array': [],
        'titles_array': [],
        'countries_array': [],
        'regions_array': [],
        'municipalities_array': [],
        'degrees_array': [],
        'months_experience': [],       # Numeric: total months of work experience
        'avg_months_per_employer': [], # Numeric: average tenure per employer
        'text_blob': [],               # For BlockingPy embeddings
        'description': [],             # SOV_DESCRIPTION for BlockingPy
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        cleaned = process_record(row, nlp)
        
        new_columns['unique_id'].append(row['unique_id'])
        new_columns['firstname_cleaned'].append(cleaned['firstname_cleaned'])
        new_columns['lastname_cleaned'].append(cleaned['lastname_cleaned'])
        new_columns['firstname_soundex'].append(cleaned['firstname_soundex'])
        new_columns['lastname_soundex'].append(cleaned['lastname_soundex'])
        new_columns['firstname_metaphone'].append(cleaned['firstname_metaphone'])
        new_columns['lastname_metaphone'].append(cleaned['lastname_metaphone'])
        new_columns['names_array'].append(cleaned['names_array'])
        new_columns['linkedin_cleaned'].append(cleaned['linkedin_cleaned'])
        new_columns['phones_array'].append(cleaned['phones_array'])
        new_columns['emails_array'].append(cleaned['emails_array'])
        new_columns['employers_array'].append(cleaned['employers_array'])
        new_columns['titles_array'].append(cleaned['titles_array'])
        new_columns['countries_array'].append(cleaned['countries_array'])
        new_columns['regions_array'].append(cleaned['regions_array'])
        new_columns['municipalities_array'].append(cleaned['municipalities_array'])
        new_columns['degrees_array'].append(cleaned['degrees_array'])
        new_columns['months_experience'].append(cleaned['months_experience'])
        new_columns['avg_months_per_employer'].append(cleaned['avg_months_per_employer'])
        new_columns['text_blob'].append(cleaned['text_blob'])
        new_columns['description'].append(cleaned['description'])
    
    # Create result dataframe
    result_df = pd.DataFrame(new_columns)
    
    return result_df


def save_outputs(output_dir: Path, 
                 train_df: pd.DataFrame, 
                 train_lookup: pd.DataFrame,
                 test_batches: dict,
                 test_lookups: dict):
    """Save all processed data to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving outputs to {output_dir}...")
    
    # Save train data
    train_df.to_parquet(output_dir / "train_cleaned.parquet", index=False)
    train_lookup.to_parquet(output_dir / "train_lookup.parquet", index=False)
    print(f"  Saved train_cleaned.parquet ({len(train_df)} records)")
    print(f"  Saved train_lookup.parquet ({len(train_lookup)} records)")
    
    # Save test batches
    for batch_name, batch_df in test_batches.items():
        batch_df.to_parquet(output_dir / f"test_{batch_name}_cleaned.parquet", index=False)
        print(f"  Saved test_{batch_name}_cleaned.parquet ({len(batch_df)} records)")
    
    for batch_name, lookup_df in test_lookups.items():
        lookup_df.to_parquet(output_dir / f"test_{batch_name}_lookup.parquet", index=False)
        print(f"  Saved test_{batch_name}_lookup.parquet ({len(lookup_df)} records)")
    
    # Also save as CSV for inspection
    train_df.to_csv(output_dir / "train_cleaned.csv", index=False)
    train_lookup.to_csv(output_dir / "train_lookup.csv", index=False)
    
    print("\nAll outputs saved successfully!")


def main():
    """Main execution function."""
    print("=" * 60)
    print("SPM Data Preparation for Splink + BlockingPy")
    print("=" * 60)
    
    # Load spaCy model
    nlp = load_spacy_model()
    print("spaCy model loaded successfully")
    
    # Load data
    df = load_data(INPUT_FILE)
    
    # Create lookup table and add unique_id
    lookup_df, main_df = create_lookup_table(df)
    
    # Split into train/test
    train_df, test_df, train_lookup, test_lookup = split_train_test(
        main_df, lookup_df, 
        train_ratio=TRAIN_RATIO, 
        random_seed=RANDOM_SEED
    )
    
    # Split test into batches
    test_batches, test_lookups = split_test_into_batches(
        test_df, test_lookup,
        batch_ratios=(BATCH1_RATIO, BATCH2_RATIO, BATCH3_RATIO),
        random_seed=RANDOM_SEED
    )
    
    # Process train data
    train_cleaned = process_dataframe(train_df, nlp, desc="Processing train data")
    
    # Process test batches
    test_batches_cleaned = {}
    for batch_name, batch_df in test_batches.items():
        test_batches_cleaned[batch_name] = process_dataframe(
            batch_df, nlp, desc=f"Processing test {batch_name}"
        )
    
    # Save outputs
    save_outputs(
        OUTPUT_DIR,
        train_cleaned,
        train_lookup,
        test_batches_cleaned,
        test_lookups
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records processed: {len(df)}")
    print(f"Train set: {len(train_cleaned)} records")
    print(f"Test set total: {sum(len(b) for b in test_batches_cleaned.values())} records")
    for batch_name, batch_df in test_batches_cleaned.items():
        print(f"  - {batch_name}: {len(batch_df)} records")
    
    print("\nOutput columns for Splink/BlockingPy:")
    print("  - unique_id: Integer ID for Splink")
    print("  - firstname_cleaned: Cleaned first name (for exact match blocking)")
    print("  - lastname_cleaned: Cleaned last name (for exact match blocking)")
    print("  - firstname_soundex: Soundex encoding of first name (phonetic blocking)")
    print("  - lastname_soundex: Soundex encoding of last name (phonetic blocking)")
    print("  - firstname_metaphone: Metaphone encoding of first name")
    print("  - lastname_metaphone: Metaphone encoding of last name")
    print("  - names_array: Array of names from NAME_1,2,3 + linkedin + combined first/last")
    print("  - linkedin_cleaned: Cleaned LinkedIn handle (alpha only)")
    print("  - phones_array: Array of standardized phone numbers")
    print("  - emails_array: Array of cleaned emails")
    print("  - employers_array: Array of standardized employer names")
    print("  - titles_array: Array of standardized job titles")
    print("  - countries_array: Array of standardized country codes")
    print("  - regions_array: Array of standardized regions/states")
    print("  - municipalities_array: Array of standardized cities")
    print("  - degrees_array: Array of standardized degrees")
    print("  - months_experience: Total months of work experience (numeric)")
    print("  - avg_months_per_employer: Average months per employer (numeric)")
    print("  - text_blob: Combined text for BlockingPy embeddings (includes numeric fields)")
    print("  - description: SOV_DESCRIPTION for BlockingPy")
    
    print("\n--- Splink Usage Example ---")
    print("""
from splink import SettingsCreator, Linker, DuckDBAPI, block_on
import splink.comparison_library as cl

settings = SettingsCreator(
    unique_id_column_name="unique_id",
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("block"),                                    # BlockingPy blocks
        block_on("firstname_cleaned", "lastname_cleaned"),    # Exact name match
        block_on("firstname_soundex", "lastname_soundex"),    # Phonetic blocking (Smith/Smyth)
        block_on("firstname_cleaned"),                        # First name only
    ],
    comparisons=[
        cl.NameComparison("firstname_cleaned"),
        cl.NameComparison("lastname_cleaned"),
        cl.ExactMatch("firstname_soundex"),                   # Same soundex = similar sounding
        cl.ExactMatch("lastname_soundex"),
        cl.ArrayIntersectAtSizes("names_array", [2, 1]),
        cl.ArrayIntersectAtSizes("phones_array", [1]),
        cl.ExactMatch("linkedin_cleaned"),
        cl.ArrayIntersectAtSizes("emails_array", [1]),
        cl.ArrayIntersectAtSizes("employers_array", [1]),
        # Numeric comparisons for experience
        cl.AbsoluteDifferenceAtThresholds("months_experience", [1, 2, 3]),
        cl.AbsoluteDifferenceAtThresholds("avg_months_per_employer", [1, 2, 3]),
        # ... more comparisons
    ],
)
""")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

