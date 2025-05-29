import os
import sys
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy import text

def import_from_utils(module_name: str):
    """
    Imports a module from the 'utils' directory, assuming 'utils' is a sibling
    to the parent directory of the script calling this function.

    Args:
        module_name (str): The name of the Python module (filename without .py)
                           to import from the 'utils' directory.

    Returns:
        module or None: The imported module object, or None if an error occurs.
    """
    try:
        # Get the directory of the script THAT IS CALLING THIS FUNCTION.
        # This assumes this function is defined within, or directly imported into,
        # the script that needs to perform the import (e.g., current_calling_script.py).
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        # Example: current_script_dir = ".../Your_Main_Project_Folder/processing/"
    except NameError:
        # Fallback if __file__ is not defined (e.g., running in an interactive interpreter or Jupyter)
        current_script_dir = os.getcwd()
        print(f"DEBUG: `__file__` not defined. Using current working directory: {current_script_dir}")
        print("WARNING: For correct relative paths, ensure interactive session's CWD is the calling script's directory (e.g., '.../processing/').")

    # Go up one level from the current script's directory to get the project root
    # Example: from ".../processing/" to ".../Your_Main_Project_Folder/"
    project_root_dir = os.path.dirname(current_script_dir)
    # print(f"DEBUG: Deduced project root: {project_root_dir}")

    # Construct the absolute path to the 'utils' directory
    utils_dir_path = os.path.join(project_root_dir, 'utils')
    # print(f"DEBUG: Target utils directory: {utils_dir_path}")

    if not os.path.isdir(utils_dir_path):
        print(f"Error: 'utils' directory not found at expected location: {utils_dir_path}")
        print(f"       Project root was determined as: {project_root_dir}")
        print(f"       Is your project structure as expected (e.g., 'processing/' and 'utils/' are siblings under a common project root)?")
        return None

    # Add the 'utils' directory to sys.path if it's not already there
    if utils_dir_path not in sys.path:
        sys.path.insert(0, utils_dir_path) # Insert at the beginning
        # print(f"DEBUG: Added '{utils_dir_path}' to sys.path.")

    # Now import the module
    try:
        imported_module = __import__(module_name)
        print(f"Successfully imported '{module_name}' from '{utils_dir_path}'.")
        return imported_module
    except ModuleNotFoundError:
        print(f"Error: Module '{module_name}.py' not found in '{utils_dir_path}' or other sys.path locations.")
        print(f"       Please ensure '{module_name}.py' exists directly inside '{utils_dir_path}'.")
        # print(f"  Current sys.path: {sys.path}") # Uncomment for deep debugging
        return None
    except ImportError as e:
        print(f"Error: Could not import module '{module_name}'. Details: {e}")
        # print(f"  Current sys.path: {sys.path}") # Uncomment for deep debugging
        return None
    except Exception as e_general:
        print(f"An unexpected error occurred during import of '{module_name}': {e_general}")
        return None

def read_csv_from_metadata_collection(csv_filename="my_data.csv"):
    """
    Reads a CSV file from the 'data_collection/metadata_collection/' directory.
    This function assumes it's being called from a script located in a sibling
    directory to 'data_collection' (e.g., 'processing/').

    Args:
        csv_filename (str): The name of the CSV file to read.

    Returns:
        pandas.DataFrame or None: The loaded DataFrame, or None if an error occurs.
    """
    try:
        # 1. Get the directory of the current script (data_processor.py)
        #    This will be .../Your_Main_Project_Folder/processing/
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback if __file__ is not defined (e.g., interactive session)
        # This assumes you run the interactive session from Your_Main_Project_Folder/processing/
        current_script_dir = os.getcwd()
        print(f"DEBUG: __file__ not defined. Using CWD: {current_script_dir}")
        print("WARNING: Ensure CWD is '.../processing/' for correct relative path.")


    # 2. Get the parent directory of the current script's directory
    #    This will be .../Your_Main_Project_Folder/
    project_root_dir = os.path.dirname(current_script_dir)
    # Alternatively, if you are certain current_script_dir IS processing:
    # project_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))


    # 3. Construct the full path to the CSV file
    #    Path: .../Your_Main_Project_Folder/data_collection/metadata_collection/your_file.csv
    csv_file_path = os.path.join(
        project_root_dir,
        'data_collection',
        'metadata',
        csv_filename
    )
    print(f"DEBUG: Attempting to read CSV from: {csv_file_path}")

    # 4. Read the CSV file
    try:
        df = pd.read_csv(csv_file_path,index_col=0)
        print(f"Successfully read '{csv_filename}' into a DataFrame.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_file_path}'.")
        print("Please check the following:")
        print(f"  1. Does '{csv_filename}' exist in '.../data_collection/metadata_collection/'?")
        print(f"  2. Is the script structure as expected (processing/ and data_collection/ as siblings)?")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_file_path}' is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while reading '{csv_file_path}': {e}")
        return None

# import metadata processing functions modules
field_growth_processing_functions = import_from_utils('field_growth_processing_functions')
topic_growth_processing_functions = import_from_utils('topic_growth_processing_functions')

# read metadata.csv (the scraped file through arXiv)
# MAKE SURE IT IS IN THE data_collection/metadata FOLDER!
metadata = read_csv_from_metadata_collection('metadata.csv')

###############################################################################
# Topic Growth Metrics
###############################################################################

topic_growth = metadata.copy()
#applying the function to the dataset
topic_growth['cleaned_abstract'] = topic_growth['abstract'].apply(topic_growth_processing_functions.preprocess)

# Model training
# create a dictionary from the preprocessed data
dictionary = Dictionary(topic_growth['cleaned_abstract'])
# filter out words that appear in fewer than 5 documents or more than 50% of the documents
dictionary.filter_extremes(no_below=5, no_above=0.5)
bow_corpus = [dictionary.doc2bow(text) for text in topic_growth['cleaned_abstract']]
# train the LDA model
num_topics = 3
ldamodel = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10, alpha='auto', eta='auto')
# get the topics per row then calculate monthly aggregates
topic_distributions = topic_growth_processing_functions.get_gensim_topic_distributions(ldamodel, bow_corpus, num_topics=3)
topic_df = topic_growth_processing_functions.topic_df_from_distributions(topic_distributions, num_topics=3)
month_col = metadata['month']
topic_df['month'] = month_col

#topic_df = pd.read_csv('topic_df.csv')

monthly_topic,rolling_topic = topic_growth_processing_functions.compute_monthly_topic_trends(topic_df)

#monthly_topic.to_csv('monthy_topic.csv',index = False)
#rolling_topic.to_csv('rolling_topic.csv',index = False)

# get topic words for wordcloud later on
topic_words_df = topic_growth_processing_functions.get_topic_words(ldamodel)

###############################################################################
# Field Growth Metrics
###############################################################################

#metadata = pd.read_csv('metadata.csv')

field_growth = metadata.copy()
#Maintain field (i.e., math, physics, etc.). I will keep primary_category because it will simplify diversity factor calculations later on.
field_growth['field'] = field_growth['primary_category'].str.split('.').str[0]
#Convert to list
field_growth['authors'] = field_growth['authors'].str.split(',\s*')
#Diversity is the number of subcategories in a paper that are distinct from the field.
#Example: A paper in the CS field with a subcategory of quant-ph has a diversity factor 1.
field_growth['diversity_factor'] = field_growth.apply(field_growth_processing_functions.compute_diversity, axis=1)

monthly_cat_summary = (
    field_growth.groupby(["month", "field"])
    .agg(
        num_publications=("abstract", field_growth_processing_functions.count_publications),
        unique_authors=("authors", field_growth_processing_functions.count_unique_authors),
        avg_diversity_factor=("diversity_factor", "mean")
    )
    .reset_index()
)

#Calculate % change
value_columns = ["num_publications", "unique_authors","avg_diversity_factor"]
monthly_growth,rolling_growth = field_growth_processing_functions.compute_monthly_growth_trends(monthly_cat_summary, value_columns)

#monthly_growth.to_csv('monthly_growth.csv',index = False)
#rolling_growth.to_csv('rolling_growth.csv',index = False)

###############################################################################
# Upload to AWS Server
###############################################################################

# The final step in this script is to upload the processed features to the AWS EC2 instance using pymysql

# parameters
host_ip = "3.148.234.227" # my server's IP address
id = "test1"
pw = "Test1234#"

# connect to mysql server
url = URL.create(
    drivername="mysql+pymysql",
    host=host_ip,
    port=3306,
    username= id,
    password=pw)

sqlEngine = create_engine(url)
sql_connection = sqlEngine.connect()

#sql_connection.execute(text('DROP DATABASE IF EXISTS literature_analytics_platform'))

sql_connection.execute(text("CREATE DATABASE IF NOT EXISTS literature_analytics_platform"))

db_url = URL.create(
    drivername="mysql+pymysql",
    host=host_ip,
    port=3306,
    username=id,
    password=pw,
    database="literature_analytics_platform"
)

db_engine = create_engine(db_url)

# Upload features to server
monthly_growth.to_sql(con= db_engine, name= 'monthly_metadata', if_exists = 'replace')
monthly_topic.to_sql(con= db_engine, name= 'monthly_topic', if_exists = 'replace')
rolling_growth.to_sql(con= db_engine, name= 'rolling_metadata', if_exists = 'replace')
rolling_topic.to_sql(con= db_engine, name= 'rolling_topic', if_exists = 'replace')
topic_words_df.to_sql(con= db_engine, name= 'topic_words', if_exists = 'replace')


# Close connection
sql_connection.close()
sqlEngine.dispose()
db_engine.dispose()

print ("Connection closed.")