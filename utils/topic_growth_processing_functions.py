import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re

# Data preprocessing
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
#nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

#function to lemmatize and remove stopwords from the text data
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = word_tokenize(text)
    words = [
    lemmatizer.lemmatize(word)
    for word in words
    if word not in stop_words and len(word) > 2 and not word.isnumeric()
    ]
    return words

def get_gensim_topic_distributions(ldamodel, corpus, num_topics):
    """
    Extracts topic distributions for each document using a trained Gensim LDA model.

    Returns a list of dictionaries: one per document, mapping topic_id to weight.
    """
    topic_distributions = []
    for doc in corpus:
        topic_prob = dict(ldamodel.get_document_topics(doc, minimum_probability=0))
        topic_distributions.append(topic_prob)
    return topic_distributions

def topic_df_from_distributions(topic_distributions, num_topics):
    """
    Converts list of topic distributions into a DataFrame with one column per topic.
    """
    topic_data = []
    for dist in topic_distributions:
        row = [dist.get(i, 0.0) for i in range(num_topics)]
        topic_data.append(row)
    return pd.DataFrame(topic_data, columns=[f"topic_{i}" for i in range(num_topics)])

def compute_monthly_topic_trends(topic_metadata_df, date_col="month", topic_prefix="topic_",window=12):
    """
    Computes average monthly topic distributions, monthly % change, rolling average over 12 months, and growth over 12 months
    """
    # Get topic columns
    topic_cols = [col for col in topic_metadata_df.columns if col.startswith(topic_prefix)]

    # Step 1: Group by month and compute average topic distribution
    monthly_avg = topic_metadata_df.groupby(date_col)[topic_cols].mean().reset_index()

    # Step 2: Compute percentage change
    pct_change = monthly_avg[topic_cols].pct_change()
    
    avg_24m = monthly_avg[topic_cols].rolling(window=window).mean()
    growth_24m = monthly_avg[topic_cols].pct_change(periods=window)
    
    pct_change.columns = [f"{col.lower()}_pct_change" for col in topic_cols]
    avg_24m.columns = [f"{col.lower()}_avg" for col in topic_cols]
    growth_24m.columns = [f"{col.lower()}_growth" for col in topic_cols]

    # Step 3: Combine average and pct_change, drop NaNs from first row
    result_df = pd.concat([monthly_avg, pct_change], axis=1).dropna().reset_index(drop=True)
    
    rolling_df = pd.concat([monthly_avg, avg_24m,growth_24m], axis=1).dropna().reset_index(drop=True)

    # Step 4: Rename topic columns to lowercase (e.g., Topic_0 → topic_0)
    result_df.rename(columns={col: col.lower() for col in topic_cols}, inplace=True)

    return result_df,rolling_df

def get_topic_words(lda_model, topn=10):
    """
    Extracts top words for each topic from a Gensim LDA model and saves to CSV.

    Parameters:
        lda_model: Trained Gensim LdaModel
        topn: Number of top words per topic
        output_path: Path to save the CSV
    """

    topic_dict = {}

    for i in range(lda_model.num_topics):
        words = [word for word, _ in lda_model.show_topic(i, topn=topn)]
        topic_dict[f"topic_{i}"] = words

    # Convert to DataFrame (words as rows, topics as columns)
    df = pd.DataFrame.from_dict(topic_dict, orient='index').transpose()

    # Save to CSV
    return df
