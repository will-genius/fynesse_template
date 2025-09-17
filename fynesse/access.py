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


def extract_date_from_filename(filename):
    """
    Extract a date (YYYY-MM-DD) from messy accident report filenames.
    """

    # Normalize filename
    name = filename.upper()
    name = re.sub(r"[^A-Z0-9 ]", " ", name)  # remove commas, dots, brackets
    name = re.sub(r"\s+", " ", name) 

    # Look for year first
    year_match = re.search(r"(20\d{2})", name)
    if not year_match:
        return None
    year = year_match.group(1)

    # Try full patterns (day month year)
    full_match = re.search(r"(\d{1,2}(ST|ND|RD|TH)?\s+[A-Z]+\s+" + year + ")", name)
    
    if full_match:
        date_str = re.sub(r"(\d)(ST|ND|RD|TH)", r"\1", full_match.group(1))
        return parser.parse(date_str, dayfirst=True).strftime("%Y-%m-%d")

    numeric_match = re.search(r"(\d{1,2}[./-]\d{1,2}[./-]" + year + ")", filename)
    if numeric_match:
        return parser.parse(numeric_match.group(1), dayfirst=True).strftime("%Y-%m-%d")

    return None


def combine_accident_reports_from_folder(folder_path, output_dir="combined_reports"):
    """
    Combines daily accident reports from a folder into yearly Excel files.
    Extracts the date from messy filenames and adds it as a 'Date' column.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not file_paths:
        print("⚠️ No Excel files found in the folder.")
        return

    yearly_data = {}

    for file in file_paths:
        filename = os.path.basename(file)
        date = extract_date_from_filename(filename)

        if not date:
            print(f"⚠️ Could not extract date from {filename}")
            continue

        year = date.split("-")[0]

        try:
            df = pd.read_excel(file)
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            continue

        df["Date"] = date
        yearly_data.setdefault(year, []).append(df)

    for year, dfs in yearly_data.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(output_dir, f"Accidents_{year}.xlsx")
        combined_df.to_excel(output_path, index=False)
        print(f"✅ Saved {output_path}")
