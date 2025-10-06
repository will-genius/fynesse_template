"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None

# function to download files using urls 
def download_if_not_exists(url, filepath):
    """Download file if it doesn't exist locally"""
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
    else:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to: {filepath}")
    return filepath

import os
import glob
import re
from dateutil import parser
import pandas as pd

# ---------------------------
# Helpers
# ---------------------------
EXPECTED_HEADER = [
    "S/NO", "TIME 24 HOURS", "BASE/SUB BASE", "COUNTY", "ROAD", "PLACE",
    "MV INVOLVED", "BRIEF ACCIDENT DETAILS", "NAME OF VICTIM", "GENDER",
    "AGE", "CAUSE CODE", "VICTIM", "NO."
]

def _normalize_token(s):
    """Normalize a header/token for robust matching."""
    if pd.isna(s):
        return ""
    s = str(s).upper()
    s = s.replace(".", "")        # remove dots
    s = s.replace("/", " ")       # treat slashes as separators
    s = s.replace("_", " ")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)  # remove odd punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_date_from_filename(fname):
    """Robust date extraction from messy filenames (ordinal suffixes handled safely)."""
    base = os.path.basename(fname)
    name = base.upper()
    # try to locate a 4-digit year token and window backwards
    tokens = re.split(r'[\s,._\-\(\)]+', name)
    for i, t in enumerate(tokens):
        if re.fullmatch(r'20\d{2}', t):
            year_idx = i
            for window in range(1, 4):  # lookback window 1..3 tokens
                start = max(0, year_idx - window)
                candidate = " ".join(tokens[start:year_idx + 1])
                # only strip ordinal suffix if it follows a digit (keeps AUGUST intact)
                candidate_clean = re.sub(r'(\d)(ST|ND|RD|TH)\b', r'\1', candidate, flags=re.I)
                try:
                    dt = parser.parse(candidate_clean, dayfirst=True, fuzzy=True)
                    return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
    # fallback numeric patterns like 13/03/2017 or 13.03.2017
    m = re.search(r'(\d{1,2}[./-]\d{1,2}[./-](20\d{2}))', base)
    if m:
        try:
            dt = parser.parse(m.group(1), dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None

# ---------------------------
# Header detection
# ---------------------------
def detect_header_row_by_expected(path, max_preview_rows=40, expected_tokens=EXPECTED_HEADER, threshold_ratio=0.6):
    """
    Inspect the first max_preview_rows of the file to find the row
    that best matches the expected header tokens.
    Returns header_row (0-index) or None.
    """
    try:
        preview = pd.read_excel(path, header=None, nrows=max_preview_rows)
    except Exception:
        return None

    norm_expected = [_normalize_token(x) for x in expected_tokens]
    # remove empty tokens
    norm_expected = [t for t in norm_expected if t]

    best_idx = None
    best_score = -1
    for idx, row in preview.iterrows():
        cells = [str(x) for x in row.tolist() if pd.notna(x)]
        norm_cells = [_normalize_token(c) for c in cells]
        # count how many expected tokens appear in the row (allow partial containment)
        score = 0
        for exp in norm_expected:
            if any(exp in cell or cell in exp for cell in norm_cells):
                score += 1
        if score > best_score:
            best_score = score
            best_idx = int(idx)

    if best_score >= max(1, int(len(norm_expected) * threshold_ratio)):
        return best_idx
    return None

# ---------------------------
# Read one file with detected header
# ---------------------------
def read_report_file_with_header(path, preview_header_rows=40):
    """
    Detect header row and return cleaned DataFrame and detected header row index.
    Accident table stops before 'SUMMARY'.
    """
    header_row = detect_header_row_by_expected(path, max_preview_rows=preview_header_rows)
    try:
        if header_row is not None:
            df = pd.read_excel(path, header=header_row)
        else:
            df = pd.read_excel(path, header=0)
    except Exception as e:
        print(f"Error reading {os.path.basename(path)}: {e}")
        return None, header_row

    # --- cutoff before SUMMARY row ---
    if not df.empty:
        mask = df.astype(str).apply(lambda row: row.str.contains("SUMMARY", case=False, na=False)).any(axis=1)
        if mask.any():
            cutoff_row = mask.idxmax()   # first index where SUMMARY appears
            df = df.iloc[:cutoff_row, :]

    # drop completely empty columns and rows
    df = df.dropna(axis=1, how='all')
    df = df.dropna(how='all').reset_index(drop=True)

    # normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]

    return df, header_row

# ---------------------------
# Main combining function
# ---------------------------
def combine_reports_folder(folder_path, output_dir="combined_reports", preview=False, preview_header_rows=40):
    """
    Process all .xlsx/.xls files in folder_path:
      - detect header row containing the expected header
      - read data from that header
      - drop empty rows/cols
      - insert Date column **after** the NO. column (or append if not found)
      - combine per-year and save Accidents_<year>.xlsx in output_dir
    Returns dict year -> combined DataFrame (in-memory).
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")) + glob.glob(os.path.join(folder_path, "*.xls")))
    if not files:
        print("No Excel files found in the folder.")
        return {}

    yearly = {}
    for f in files:
        fname = os.path.basename(f)
        date = extract_date_from_filename(fname)
        if not date:
            print(f"Could not extract date from {fname}")
            continue

        df, header_row = read_report_file_with_header(f, preview_header_rows=preview_header_rows)
        if df is None:
            continue

        # insert Date column after 'NO.' column
        norm_cols = [_normalize_token(c).replace(" ", "") for c in df.columns]
        no_idx = None
        try:
            no_idx = norm_cols.index("NO")
        except ValueError:
            # fallback: find first column that starts with "NO" token
            for i, c in enumerate(norm_cols):
                if c.startswith("NO"):
                    no_idx = i
                    break

        if no_idx is not None:
            insert_pos = no_idx + 1
            # ensure we don't overwrite existing 'Date'
            if 'Date' in df.columns:
                df['Date'] = date
            else:
                df.insert(insert_pos, 'Date', date)
        else:
            # append if NO column not found
            df['Date'] = date

        # final cleanup (drop cols/rows completely empty again)
        df = df.dropna(axis=1, how='all')
        df = df.dropna(how='all').reset_index(drop=True)

        year = date.split("-")[0]
        yearly.setdefault(year, []).append((fname, header_row, df))

    # combine & save per year
    result = {}
    for year, items in yearly.items():
        dfs = [t[2] for t in items]
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        out_path = os.path.join(output_dir, f"Accidents_{year}.xlsx")
        combined.to_excel(out_path, index=False)
        print(f" Saved {out_path}  (files: {len(dfs)})")
        result[year] = combined

    return result
