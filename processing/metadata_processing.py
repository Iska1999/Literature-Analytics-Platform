# processing/metadata_processing.py
import os
import sys
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sqlalchemy import text

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.db_manager import db_manager

def import_from_utils(module_name: str):
    """
    Imports a module from the 'utils' directory, assuming 'utils' is a sibling
    to the parent directory of the script calling this function.
    """
    try:
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        project_root_dir = os.path.dirname(current_script_dir)
        utils_dir_path = os.path.join(project_root_dir, 'utils')
        
        if not os.path.isdir(utils_dir_path):
            print(f"Error: 'utils' directory not found at expected location: {utils_dir_path}")
            return None

        if utils_dir_path not in sys.path:
            sys.path.insert(0, utils_dir_path)

        imported_module = __import__(module_name)
        print(f"Successfully imported '{module_name}' from '{utils_dir_path}'.")
        return imported_module
    except Exception as e:
        print(f"An error occurred during import of '{module_name}': {e}")
        return None

def read_csv_from_metadata_collection(csv_filename="my_data.csv"):
    """
    Reads a CSV file from the 'data_collection/metadata_collection/' directory.
    """
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_script_dir)
        csv_file_path = os.path.join(
            project_root_dir,
            'data_collection',
            'metadata',
            csv_filename
        )
        print(f"DEBUG: Attempting to read CSV from: {csv_file_path}")

        df = pd.read_csv(csv_file_path, index_col=0)
        print(f"Successfully read '{csv_filename}' into a DataFrame.")
        return df
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None

# Validate configuration
try:
    Config.validate()
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Please ensure your .env file is properly configured with DB_HOST, DB_USER, and DB_PASSWORD")
    sys.exit(1)

# Import metadata processing functions modules
field_growth_processing_functions = import_from_utils('field_growth_processing_functions')
topic_growth_processing_functions = import_from_utils('topic_growth_processing_functions')

# Read metadata.csv
metadata = read_csv_from_metadata_collection('metadata.csv')

if metadata is None:
    print("Failed to read metadata.csv. Exiting.")
    sys.exit(1)

###############################################################################
# Topic Growth Metrics
###############################################################################

print("Processing topic growth metrics...")
topic_growth = metadata.copy()

# Apply preprocessing to abstracts
topic_growth['cleaned_abstract'] = topic_growth['abstract'].apply(topic_growth_processing_functions.preprocess)

# Model training with config parameters
dictionary = Dictionary(topic_growth['cleaned_abstract'])
dictionary.filter_extremes(no_below=Config.LDA_NO_BELOW, no_above=Config.LDA_NO_ABOVE)
bow_corpus = [dictionary.doc2bow(text) for text in topic_growth['cleaned_abstract']]

# Train the LDA model with parameters from config
ldamodel = LdaModel(
    bow_corpus, 
    num_topics=Config.LDA_NUM_TOPICS, 
    id2word=dictionary, 
    passes=Config.LDA_PASSES, 
    alpha=Config.LDA_ALPHA, 
    eta=Config.LDA_ETA
)

# Get topics and calculate monthly aggregates
topic_distributions = topic_growth_processing_functions.get_gensim_topic_distributions(
    ldamodel, bow_corpus, num_topics=Config.LDA_NUM_TOPICS
)
topic_df = topic_growth_processing_functions.topic_df_from_distributions(
    topic_distributions, num_topics=Config.LDA_NUM_TOPICS
)
month_col = metadata['month']
topic_df['month'] = month_col

# Compute monthly trends
monthly_topic, rolling_topic = topic_growth_processing_functions.compute_monthly_topic_trends(topic_df)

# Get topic words for wordcloud
topic_words_df = topic_growth_processing_functions.get_topic_words(ldamodel)

###############################################################################
# Field Growth Metrics
###############################################################################

print("Processing field growth metrics...")
field_growth = metadata.copy()

# Extract field from primary_category
field_growth['field'] = field_growth['primary_category'].str.split('.').str[0]

# Convert authors to list
field_growth['authors'] = field_growth['authors'].str.split(',\s*')

# Calculate diversity factor
field_growth['diversity_factor'] = field_growth.apply(
    field_growth_processing_functions.compute_diversity, axis=1
)

# Aggregate by month and field
monthly_cat_summary = (
    field_growth.groupby(["month", "field"])
    .agg(
        num_publications=("abstract", field_growth_processing_functions.count_publications),
        unique_authors=("authors", field_growth_processing_functions.count_unique_authors),
        avg_diversity_factor=("diversity_factor", "mean")
    )
    .reset_index()
)

# Calculate growth trends
value_columns = ["num_publications", "unique_authors", "avg_diversity_factor"]
monthly_growth, rolling_growth = field_growth_processing_functions.compute_monthly_growth_trends(
    monthly_cat_summary, value_columns
)

###############################################################################
# Upload to AWS Server
###############################################################################

print("Uploading to database...")

try:
    # Create database if it doesn't exist
    db_manager.create_database_if_not_exists()
    
    # Upload features to server
    db_manager.write_table(monthly_growth, 'monthly_metadata')
    db_manager.write_table(monthly_topic, 'monthly_topic')
    db_manager.write_table(rolling_growth, 'rolling_metadata')
    db_manager.write_table(rolling_topic, 'rolling_topic')
    db_manager.write_table(topic_words_df, 'topic_words')
    
    print("Data successfully uploaded to database.")
    
except Exception as e:
    print(f"Error uploading to database: {e}")
    
finally:
    # Dispose of database connections
    db_manager.dispose()
    print("Database connections closed.")