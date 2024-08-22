import pandas as pd
import numpy as np
# pip install lxml

# https://stackoverflow.com/questions/44232578/get-the-sp-500-tickers-list
def list_wikipedia_sp500() -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/75845569/
    url = 'https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url, attrs={'id': 'constituents'}, index_col='Symbol')[0]

df = list_wikipedia_sp500()
df.head()

for i, row in enumerate(df.iterrows()):
    print(f"{i+1}. Ticker: {row[0]}, Company: {row[1]['Security']}")

mmm_row = df.loc['MMM']
ticker_list = list(df.index)

df.to_csv("/home/nim/output_1.csv", index=True) # for option 1
#######


# ###### Symbol is now a column rather than the index (option 2)
# df = df.reset_index()  # This will convert the current index into a column
# mmm_row = df[df['Symbol'] == 'MMM']

# for i, row in enumerate(df.iterrows()):
#     print(f"{i+1}. Ticker: {row[1]['Symbol']}, Company: {row[1]['Security']}")
#
# for i, row in df.iterrows():
#     print(f"{i+1}. Ticker: {row['Symbol']}, Company: {row['Security']}")
#
# ticker_list = list(df['Symbol'])
# df.to_csv("/home/nim/output_1.csv", index=False) # for option 2
# ######



from Playground.gurufocus.gf_analyze_ticker import get_financial_data_for_ticker

###### Make a dataframe of all best companies


# Define a function to extract and handle the score
def extract_scores(main_scores, key):
    try:
        return int(main_scores[key].split('/')[0])
    except (ValueError, KeyError):
        return np.nan


def get_best_companies(ticker_list, df):
    # go over all companies in Gurufocus, extracts scores

    # Create a blank DataFrame with specified columns
    df_best_companies = pd.DataFrame(
        columns=['ticker', 'company_name', 'financial_str', 'profit', 'growth', 'gf_value', 'momentum']
    )

    # find best companies
    for i, ticker in enumerate(ticker_list):
        print(f'i: {i+1} - ticker: {ticker}')
        main_scores, all_data = get_financial_data_for_ticker(ticker, print_all_data=False)

        # for key in score_keys:
        #     scores[key] = extract_score(main_scores, key)
        scores = {key: extract_scores(main_scores, key) for key in main_scores.keys()}

        if (scores['financial_str']>=8) & (scores['profit']>=8):

            new_row = pd.DataFrame({
                'ticker': [ticker],
                'company_name': [df.loc[ticker]['Security']], # for option 1 - ticker is index
                # 'company_name': [df[df['Symbol'] == ticker]['Security'].values[0]], # option2 after index reset
                'financial_str': [scores['financial_str']],
                'profit': [scores['profit']],
                'growth': [scores['growth']],
                'gf_value': [scores['gf_value']],
                'momentum': [scores['momentum']],
                'GF_score': [scores['GF_score']],

            })
            df_best_companies = pd.concat([df_best_companies, new_row], ignore_index=True)

    df_best_companies = df_best_companies.sort_values(by='GF_score', ascending=False)
    # df_best_companies.to_csv("/home/nim/best_companies.csv", index=False)

    return df_best_companies

df_best_companies = get_best_companies(ticker_list, df)

# to rename a column
# df_best_companies.rename(columns={'financial_str_score': 'financial_str'}, inplace=True)




