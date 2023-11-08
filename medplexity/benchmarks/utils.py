from io import StringIO

import requests
import pandas as pd


def parse_csv_from_url(url: str) -> pd.DataFrame:
    """Parses a CSV file from the given URL and returns it as a pandas DataFrame."""

    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.content.decode("utf-8")))
    else:
        raise Exception(
            f"Failed to retrieve CSV file from {url}. HTTP status code: {response.status_code}"
        )
