# data_collection/metadata/metadata_collection.py
import os
import sys
from datetime import date
import pandas as pd

# Add parent directories to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config

def import_from_utils(module_name, script_is_in_metadata_collection=True):
    """
    Imports a module from the 'utils' directory.
    """
    try:
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        print(f"DEBUG: __file__ resolved to: {current_script_path}")
    except NameError:
        current_script_dir = os.getcwd()
        print(f"DEBUG: __file__ not defined. Using current working directory: {current_script_dir}")

    if script_is_in_metadata_collection:
        # Go up two levels to reach project root
        project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
    else:
        project_root_dir = current_script_dir

    # Construct the path to the 'utils' directory
    utils_dir = os.path.join(project_root_dir, 'utils')

    # Add the 'utils' directory to sys.path if it's not already there
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
        print(f"DEBUG: Added '{utils_dir}' to sys.path")

    # Import the module
    try:
        imported_module = __import__(module_name)
        print(f"Successfully imported '{module_name}' from '{utils_dir}'")
        return imported_module
    except ImportError as e:
        print(f"Error: Could not import module '{module_name}': {e}")
        return None

# Import arXiv scraper functions
arXiv_scraper_functions = import_from_utils('arXiv_scraper_functions')

if arXiv_scraper_functions is None:
    print("Failed to import arXiv_scraper_functions. Exiting.")
    sys.exit(1)

# Get today's date and calculate start date
today = date.today()
start = today.replace(year=today.year - Config.ARXIV_LOOKBACK_YEARS)

print(f"Fetching arXiv data from {start} to {today}...")

# Pull fetched arxiv metadata into a dataframe
metadata = arXiv_scraper_functions.fetch_arxiv_database(
    start_date=f"{start}", 
    end_date=None
)

# Remove title duplicates
metadata = metadata.drop_duplicates(subset=['title', 'doi'])

# Drop unnecessary columns
metadata = metadata.drop(columns=['doi', 'journal', 'title'], errors='ignore')

# Ensure date format is correct
metadata['date'] = pd.to_datetime(metadata['date'])

# Add month column for binning
metadata['month'] = metadata['date'].dt.to_period('M')

# Write csv file to be processed later
output_path = os.path.join(os.path.dirname(__file__), 'metadata.csv')
metadata.to_csv(output_path, index=False)
print(f"Metadata saved to: {output_path}")