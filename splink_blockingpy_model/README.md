# SPM Data Preparation for Splink + BlockingPy

This project provides data preparation scripts for building a probabilistic record linkage model combining **Splink** and **BlockingPy**.

## Overview

The data preparation pipeline:
1. Loads the SPM data from TSV format
2. Splits data into train (90%) and test (10%) sets
3. Further splits test set into batch1 (20%), batch2 (40%), batch3 (40%)
4. Creates a lookup table with original IDs and a new integer `unique_id` (required by Splink)
5. Cleans and standardizes all fields
6. Creates array columns for matching
7. Generates text blobs for BlockingPy embeddings

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Run the data preparation script
python data_preparation.py
```

## Output Files

The script generates the following files in the `output/` directory:

### Parquet Files (for Splink/BlockingPy)
- `train_cleaned.parquet` - Cleaned training data
- `train_lookup.parquet` - Lookup table for training data
- `test_batch1_cleaned.parquet` - Cleaned test batch 1 (20%)
- `test_batch1_lookup.parquet` - Lookup table for batch 1
- `test_batch2_cleaned.parquet` - Cleaned test batch 2 (40%)
- `test_batch2_lookup.parquet` - Lookup table for batch 2
- `test_batch3_cleaned.parquet` - Cleaned test batch 3 (40%)
- `test_batch3_lookup.parquet` - Lookup table for batch 3

### CSV Files (for inspection)
- `train_cleaned.csv`
- `train_lookup.csv`

## Output Schema

### Lookup Table
| Column | Type | Description |
|--------|------|-------------|
| RESUME_ID | string | Original resume ID |
| FILTERED_TALENT_ID | string | Original talent ID |
| BATCH_TYPE | string | Original batch type |
| unique_id | integer | New integer ID for Splink |

### Cleaned Data
| Column | Type | Description |
|--------|------|-------------|
| unique_id | integer | Integer ID for Splink (links to lookup table) |
| firstname_cleaned | string | Cleaned first name for exact match blocking |
| lastname_cleaned | string | Cleaned last name for exact match blocking |
| firstname_soundex | string | Soundex phonetic encoding (catches Smith/Smyth) |
| lastname_soundex | string | Soundex phonetic encoding |
| firstname_metaphone | string | Metaphone phonetic encoding (alternative) |
| lastname_metaphone | string | Metaphone phonetic encoding |
| names_array | array[string] | Unique names from NAME_1,2,3 + linkedin + combined first/last |
| linkedin_cleaned | string | Cleaned LinkedIn handle (alpha characters only) |
| phones_array | array[string] | Standardized phone numbers (digits only) |
| emails_array | array[string] | Cleaned email addresses |
| employers_array | array[string] | Standardized employer names |
| titles_array | array[string] | Standardized job titles |
| countries_array | array[string] | Standardized country codes (e.g., US, CA) |
| regions_array | array[string] | Standardized regions/states |
| municipalities_array | array[string] | Standardized city names |
| degrees_array | array[string] | Standardized degree names |
| months_experience | float | Total months of work experience (numeric) |
| avg_months_per_employer | float | Average months per employer (numeric) |
| text_blob | string | Combined text for BlockingPy embeddings (includes numeric) |
| description | string | SOV_DESCRIPTION for BlockingPy |

## Data Cleaning Details

### Name Cleaning (spaCy NER)
Three types of name outputs:

**1. `firstname_cleaned` and `lastname_cleaned`** (for exact match blocking):
- Cleaned versions of FIRSTNAME and LASTNAME columns
- Used for efficient `block_on("firstname_cleaned", "lastname_cleaned")` in Splink
- Removes special characters, normalizes to lowercase

**2. `firstname_soundex`, `lastname_soundex`, `firstname_metaphone`, `lastname_metaphone`** (phonetic blocking):
- Soundex and Metaphone phonetic encodings
- Catches spelling variations like "Smith" vs "Smyth" (both Soundex = "S530")
- Use for blocking: `block_on("firstname_soundex", "lastname_soundex")`
- Excellent for names with typos or alternative spellings

**3. `names_array`** (for fuzzy matching and deduplication):
- Uses spaCy's Named Entity Recognition to extract person names from NAME_1, NAME_2, NAME_3
- Includes `linkedin_cleaned` handle
- Includes combined `firstname + lastname` as single entry (helps catch spelling variations)
- All values deduplicated into unique array
- Used with `ArrayIntersectAtSizes` or `PairwiseStringDistanceFunctionAtThresholds`

### Numeric Fields (Work Experience)

**`months_experience`** (SOV_MONTHS_OF_WORK_EXPERIENCE):
- Total months of work experience as float
- Use with `AbsoluteDifferenceAtThresholds` in Splink
- Included in text_blob as "experience_years: X" for BlockingPy

**`avg_months_per_employer`** (SOV_AVG_MONTHS_PER_EMPLOYER):
- Average tenure per employer as float
- Useful for identifying job-hopping patterns
- Included in text_blob as "avg_tenure_years: X" for BlockingPy

### LinkedIn URL Cleaning
- Validates URL starts with `https://linkedin.com/in/` or `https://www.linkedin.com/in/`
- Invalid URLs become `None` (Splink null)
- Extracts profile ID and keeps only alphabetic characters
- Example: `https://linkedin.com/in/john-smith-123` → `johnsmith`

### Phone Number Cleaning
- Uses `phonenumbers` library for parsing and validation
- Removes common invalid placeholders (0000000000, 1234567890, etc.)
- Returns only digits of valid phone numbers
- Processes: PHONE, MOBILEPHONE, HOMEPHONE, OTHERPHONE, PHONE_1, PHONE_2, PHONE_3

### Employer/Title/Location Cleaning
- Removes common suffixes (Inc., LLC, Corp., etc.)
- Normalizes to lowercase
- Standardizes country codes (USA→US, CAN→CA, etc.)
- Standardizes common degree names

### Text Blob for BlockingPy
Format: `"column_name: value; column_name: value; ..."`

Example:
```
firstname: john; lastname: smith; names: john smith; linkedin: johnsmith; 
phones: 5551234567; emails: john@example.com; employers: acme corp; 
titles: software engineer; countries: US; regions: CA; municipalities: san francisco
```

## Null Handling

**Important for Splink:** Empty strings are converted to `None` (Python null) because Splink requires actual null values, not empty strings. The `is_null_level` in Splink comparisons expects SQL `NULL`, not empty strings.

## Configuration

Edit the constants at the top of `data_preparation.py`:

```python
# Train/Test split ratio
TRAIN_RATIO = 0.90
TEST_RATIO = 0.10

# Test batch split ratios
BATCH1_RATIO = 0.20
BATCH2_RATIO = 0.40
BATCH3_RATIO = 0.40

# Random seed for reproducibility
RANDOM_SEED = 42
```

## Next Steps

See **[SPLINK_USAGE_GUIDE.md](SPLINK_USAGE_GUIDE.md)** for detailed instructions on:
- BlockingPy blocking setup
- Splink configuration
- Understanding `ArrayIntersectAtSizes` thresholds
- Training the model
- Generating predictions and clusters
- Visualization

