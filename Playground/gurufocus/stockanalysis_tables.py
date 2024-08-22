import requests
from bs4 import BeautifulSoup
import pandas as pd

# https://stockanalysis.com/list/
sp500_url = 'https://stockanalysis.com/list/sp-500-stocks/'
nasdaq100_rul = 'https://stockanalysis.com/list/nasdaq-100-stocks/'
nasdaq_url = 'https://stockanalysis.com/list/nasdaq-stocks/'
nyse_url = 'https://stockanalysis.com/list/nyse-stocks/'
israeli_us_url= 'https://stockanalysis.com/list/israeli-stocks-us/'

# Custom headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Function to fetch the HTML content of the page
def fetch_html(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None


# Function to parse the table data from HTML content
def parse_table_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table in the page content
    table = soup.find('table')

    if not table:
        print("Table not found on the page.")
        return None, None

    # Extract table headers
    headers = []
    for header in table.find_all('th'):
        headers.append(header.get_text().strip())

    # Extract table rows
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cells = row.find_all('td')
        row_data = [cell.get_text().strip() for cell in cells]
        rows.append(row_data)

    return headers, rows


# Main function to get the S&P 500 table data
def get_data_table(url):
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


def get_company_financials(ticker='dov'):
    company_financials = {
        'income': get_data_table(f"https://stockanalysis.com/stocks/{ticker}/financials/"),
        'balance_sheet': get_data_table(f"https://stockanalysis.com/stocks/{ticker}/financials/balance-sheet/"),
        'cash_flow': get_data_table(f"https://stockanalysis.com/stocks/{ticker}/financials/cash-flow-statement/"),
        'ratios': get_data_table(f"https://stockanalysis.com/stocks/{ticker}/financials/ratios/"),

    }
    return company_financials

sp500_df = get_data_table(sp500_url)


