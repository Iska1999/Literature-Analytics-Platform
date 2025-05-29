import logging
#import os
import sys
#import paperscraper
import arxiv
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
#import pkg_resources

global ARXIV_QUERIER
ARXIV_QUERIER = None

arxiv_field_mapper = {
    "published": "date",
    "journal_ref": "journal",
    "summary": "abstract",
    "entry_id": "doi",
}

# Authors, date, and journal fields need specific processing
process_fields = {
    "authors": lambda authors: ", ".join([a.name for a in authors]),
    "date": lambda date: date.strftime("%Y-%m-%d"),
    "journal": lambda j: j if j is not None else "",
    "doi": lambda entry_id: f"10.48550/arXiv.{entry_id.split('/')[-1].split('v')[0]}",
}

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

#dump_root = pkg_resources.resource_filename("paperscraper", "server_dumps")

today = datetime.today().strftime("%Y-%m-%d")

#fetch_arxiv_papers is the base function for the fetch_arxiv_database function
def fetch_arxiv_papers(
    query: str,
    fields: List = ["title", "primary_category","categories","authors", "date", "abstract", "journal", "doi"],
    max_results: int = 9999,
    client_options: Dict = {"num_retries": 10},
    search_options: Dict = dict(),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Performs arxiv API request of a given query and returns list of papers with
    fields as desired.

    Args:
        query Query to arxiv API. Needs to match the arxiv API notation.
        fields: List of strings with fields to keep in output.
        max_results: Maximal number of results, defaults to 99999.
        client_options: Optional arguments for `arxiv.Client`. E.g.:
            page_size (int), delay_seconds (int), num_retries (int).
            NOTE: Decreasing 'num_retries' will speed up processing but might
            result in more frequent 'UnexpectedEmptyPageErrors'.
        search_options: Optional arguments for `arxiv.Search`. E.g.:
            id_list (List), sort_by, or sort_order.

    Returns:
        pd.DataFrame: One row per paper.

    """
    client = arxiv.Client(**client_options)
    search = arxiv.Search(query=query, max_results=max_results, **search_options)
    results = client.results(search)

    processed = pd.DataFrame(
        [
            {
                arxiv_field_mapper.get(key, key): process_fields.get(
                    arxiv_field_mapper.get(key, key), lambda x: x
                )(value)
                for key, value in vars(paper).items()
                if arxiv_field_mapper.get(key, key) in fields and key != "doi"
            }
            for paper in tqdm(results, desc=f"Processing {query}", disable=not verbose)
        ]
    )
    return processed

#fetch_arxiv_database fetches paper metadata between a date range.
def fetch_arxiv_database(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches papers from arXiv based on a time range and returns them as a DataFrame.

    Args:
        start_date (str, optional): Start date in 'YYYY-MM-DD'. Defaults to '1991-01-01'.
        end_date (str, optional): End date in 'YYYY-MM-DD'. Defaults to today.

    Returns:
        pd.DataFrame: All papers collected in the date range.
    """
    EARLIEST_START = "1991-01-01"
    if start_date is None:
        start_date = EARLIEST_START
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date > end_date:
        raise ValueError(f"start_date {start_date} cannot be later than end_date {end_date}")

    all_papers = []
    progress_bar = tqdm(total=(end_date - start_date).days + 1)

    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)
        progress_bar.set_description(f"Fetching {current_date.strftime('%Y-%m-%d')}")

        query = f"submittedDate:[{current_date.strftime('%Y%m%d0000')} TO {next_date.strftime('%Y%m%d0000')}]"
        try:
            papers = fetch_arxiv_papers(
                query=query,
                fields=["title", "primary_category", "categories", "authors", "date", "abstract", "journal", "doi"],
                verbose=False,
            )
            #print(papers)
            if not papers.empty:
                all_papers.append(papers)
        except Exception as e:
            print(f"Arxiv scraping error: {current_date.strftime('%Y-%m-%d')}: {e}")
        current_date = next_date
        progress_bar.update(1)

    if all_papers:
        return pd.concat(all_papers, ignore_index=True)
    else:
        return pd.DataFrame(columns=["title", "primary_category", "categories", "authors", "date", "abstract", "journal", "doi"])
