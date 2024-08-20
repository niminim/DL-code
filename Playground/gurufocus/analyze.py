import requests
from bs4 import BeautifulSoup

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

# General function to extract rank scores
def extract_rank_score(soup, rank_name, identifier):
    rank_section = soup.find('a', href=lambda href: href and identifier in href)

    if rank_section:
        score_div = rank_section.find_next('div', class_='indicator-progress-bar-header')

        if score_div:
            style_attr = score_div.div.get('style')
            if style_attr:
                score_percentage = style_attr.split('width:')[1].split('%')[0].strip()
                score = int(float(score_percentage) / 10)
                return f"{score}/10"

    return f"{rank_name} score not found."


# Function to parse and extract other financial data from tables
def extract_financial_data(soup):
    financial_data = {}

    # Extracting data from rows with financial metrics
    rows = soup.find_all('tr', class_='stock-indicators-table-row')
    for row in rows:
        metric_name_tag = row.find('td', class_='t-caption p-v-sm semi-bold')
        metric_value_tag = row.find('span', class_='p-l-sm')

        if metric_name_tag and metric_value_tag:
            metric_name = metric_name_tag.get_text(strip=True)
            metric_value = metric_value_tag.get_text(strip=True)
            financial_data[metric_name] = metric_value

    return financial_data

# Main function to fetch and print financial data for any stock ticker
def get_financial_data_for_ticker(ticker):
    url = f'https://www.gurufocus.com/stock/{ticker}/summary'

    # Fetch the page content
    html_content = fetch_html(url)

    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract various rank scores
        financial_strength_score = extract_rank_score(soup, "Financial Strength", 'rank-balancesheet')
        profitability_score = extract_rank_score(soup, "Profitability Rank", 'rank-profitability')
        growth_score = extract_rank_score(soup, "Growth Rank", 'rank-growth')
        gf_value_score = extract_rank_score(soup, "GF Value Rank", 'rank-gf-value')
        momentum_score = extract_rank_score(soup, "Momentum Rank", 'rank-momentum')

        # Print the extracted scores and metrics
        print(f"Financial Strength Score for {ticker.upper()}: {financial_strength_score}")
        print(f"Profitability Rank Score for {ticker.upper()}: {profitability_score}")
        print(f"Growth Rank Score for {ticker.upper()}: {growth_score}")
        print(f"GF Value Rank Score for {ticker.upper()}: {gf_value_score}")
        print(f"Momentum Rank Score for {ticker.upper()}: {momentum_score}")

        # Extract financial data from the table
        all_data = extract_financial_data(soup)

        # Print the extracted financial data
        if all_data:
            print(f"\nOther Financial Data for {ticker.upper()}:")
            for key, value in all_data.items():
                print(f"{key}: {value}")
        else:
            print("No additional financial data found.")

# Example usage
ticker = 'NVDA'  # You can change this ticker symbol to fetch data for another company
get_financial_data_for_ticker(ticker)
