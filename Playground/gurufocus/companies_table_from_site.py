import requests
from bs4 import BeautifulSoup
import pandas as pd

# https://stockanalysis.com/list/
# https://stockanalysis.com/list/sp-500-stocks/
# https://stockanalysis.com/list/nasdaq-100-stocks/
# https://stockanalysis.com/list/nasdaq-100-stocks/
# https://stockanalysis.com/list/nyse-stocks/
# https://stockanalysis.com/list/nyse-stocks/

# Custom headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# URL of the webpage containing the S&P 500 stocks list
url = 'https://stockanalysis.com/list/sp-500-stocks/'


# Function to fetch the HTML content of the page
def fetch_html(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None


# Function to parse the table data
def parse_table_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table
    table = soup.find('table')

    # Initialize lists to store table data
    table_data = []
    headers = []

    # Get the headers of the table
    for th in table.find_all('th'):
        headers.append(th.text.strip())

    # Get the rows of the table
    rows = table.find_all('tr')[1:]  # Skip the header row
    for row in rows:
        row_data = []
        for td in row.find_all('td'):
            row_data.append(td.text.strip())
        table_data.append(row_data)

    return headers, table_data


# Main function to get the S&P 500 table data
def get_sp500_table_data(url):
    # Fetch the page content
    html_content = fetch_html(url)

    if html_content:
        # Parse the table data
        headers, table_data = parse_table_data(html_content)

        # Convert the table data to a pandas DataFrame
        df = pd.DataFrame(table_data, columns=headers)

        return df
    else:
        return None


# Example usage
sp500_df = get_sp500_table_data(url)

if sp500_df is not None:
    print(sp500_df)
else:
    print("Failed to retrieve the S&P 500 table data.")