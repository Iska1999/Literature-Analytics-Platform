# utils

This folder contains the contains Python scripts with functionality that is reused in scripts. The files are organized as follows:

- [`arXiv_scraper_functions.py`](arXiv_scraper_functions.py): Contains the functions to scrape arXiv for metadata. It is based on the [paperscraper](https://github.com/jannisborn/paperscraper) module but modified to also scrape for topic categories and dump into a dataframe. As opposed to the barebones arXiV API, this function accounts for the max fetch limit per request of 300,000 and repeats upon receiving errors or losing connection (which is unfortunately very frequent).
- [`field_growth_processing_functions.py`](field_growth_processing_functions.py): Contains the aggregation functions to calculate the field growth features.
- [`streamlit_functions.py`](streamlit_functions.py): Contains the functions used in the Streamlit pages.
- [`topic_growth_processing_functions.py`](topic_growth_processing_functions.py): Contains the functions to preprocess the abstracts and run the LdaModel to extract the topic features.
- [`useful_functions.py`](useful_functions.py): Contains the functions used in multiple scripts (e.g., scaling).
- [`var_analysis_functions.py`](var_analysis_functions.py): Contains the functions to preprocess the features, fit the VAR model, and forecasts.
- [`bsts_analysis_functions.py`](bsts_analysis_functions.py): Contains the function to fit the BSTS model using the Orbit module.
- [`visualization_functions.py`](visualization_functions.py): Contains the functions to visualize the the trends, create a Granger Causality network, and produce IRF response plots.

Note that these files are **NOT** meant to be run on their own.

