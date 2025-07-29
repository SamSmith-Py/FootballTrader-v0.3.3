from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import pandas as pd
import betfairlightweight
import matplotlib.pyplot as plt
import sqlite3
import numpy as np



def get_sports_iq_stats():
    # Initialize Selenium WebDriver with Edge options
    edge_options = Options()
    # edge_options.add_argument("--headless=new")  # or "--headless" if "new" doesn't work
    # edge_options.add_argument("--disable-gpu")   # (optional) improves compatibility
    # edge_options.add_argument("--window-size=1920,1080")  # needed if elements aren't visible

    
    # Delete file download if it already exists
    data_file_path = r"C:\Users\Sam\FootballTrader v0.3.2\Football Data Fixtures.xlsx"
    if os.path.exists(data_file_path):
        os.remove(data_file_path)


    prefs = {
            "download.default_directory": r"C:\Users\Sam\FootballTrader v0.3.2",  # Change to your desired directory
            "download.prompt_for_download": False,  # Disable download prompt
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True  # Enable safe browsing
            }
    edge_options.add_experimental_option("prefs", prefs)

    service = EdgeService(executable_path=r'C:/Program Files (x86)/msedgedriver.exe')
    


    # Wait for the login form to be loaded and enter login details
    while True:
        try:
            driver = webdriver.Edge(service=service, options=edge_options)
            # Load the login page
            url = 'https://sports-iq.co.uk/login/'
            driver.get(url)
            # Make instances of input boxes for login
            username_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'login_email')))  # Adjust selector as needed
            password_input = driver.find_element(By.ID, 'password')  # Adjust selector as needed

            # Enter your login credentials
            username_input.send_keys('samcsmith17@gmail.com')
            password_input.send_keys('Dexyboy17!')
            password_input.send_keys(Keys.RETURN)

            

            # Wait for the next page to load
            tool_dropdown = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'menu-text') and text()='Tools']"))
            )
            tool_dropdown.click()

            custom_tables = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'menu-text') and text()='Custom Tables']"))
            )
            custom_tables.click()

            football_table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'table_name') and text()='Football Data']"))
            )
            football_table.click()

            excel_export = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'btn btn-secondary buttons-excel buttons-html5 ms-3 btn-sm btn-outline-default')]"))
            )
            excel_export.click()

            

            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")
            driver.quit()
            time.sleep(10)
            print('Re-attempting connection with driver.')  
     
    time.sleep(2)
    driver.quit()  

"""# Connect to Betfair API
api = betfairlightweight.APIClient('smudge2049', 'Dex17@£141117', '4oAYsDJiYA7P5Wej')
api.login_interactive()
cleared = api.betting.list_cleared_orders(market_ids=[1.245350965], group_by='MARKET',
                                                      lightweight=True)
print(cleared)
c = cleared.orders[0].profit
print(c)"""


autotrader_db_path = r'C:\Users\Sam\FootballTrader v0.3.2\database\autotrader_data.db'
con = sqlite3.connect(autotrader_db_path, check_same_thread=False)
df = pd.read_sql_query("SELECT * from archive_v2", con)

# Set up columns required for backtesting
df['home_ht_score'] = [int(x[0]) for x in df['ht_score'].values]
df['away_ht_score'] = [int(x[-1]) for x in df['ht_score'].values]

df['home_ft_score'] = [int(x[0]) for x in df['ft_score'].values]
df['away_ft_score'] = [int(x[-1]) for x in df['ft_score'].values]

df['total_goals'] = df['home_ft_score'] + df['away_ft_score']

df['ht_result'] = np.where(df['home_ht_score'] == df['away_ht_score'], 'draw', 0)
df['ht_result'] = np.where(df['home_ht_score'] > df['away_ht_score'], 'home', df['ht_result'])
df['ht_result'] = np.where(df['home_ht_score'] < df['away_ht_score'], 'away', df['ht_result'])

df['ft_result'] = np.where(df['home_ft_score'] == df['away_ft_score'], 'draw', 0)
df['ft_result'] = np.where(df['home_ft_score'] > df['away_ft_score'], 'home', df['ft_result'])
df['ft_result'] = np.where(df['home_ft_score'] < df['away_ft_score'], 'away', df['ft_result'])

df.copy()['Odds Betfair Draw'] = np.where(df['Odds Betfair Draw'] == 'nan', np.nan, df['Odds Betfair Draw'])
df['Odds Betfair Draw'] = np.where(df['Odds Betfair Draw'] == 'None', np.nan, df['Odds Betfair Draw'])
df.replace('None', np.nan, inplace=True)
df.replace('nan', np.nan, inplace=True)
df.replace('-', np.nan, inplace=True)
df['Odds Betfair Draw'].bfill(inplace=True)
df['Odds Betfair Draw'].ffill(inplace=True)


df['Goals 2.5+ L8 Avg'] = df['Goals 2.5+ L8 Avg'].apply(
    lambda x: x * 100 if x < 1 else x
)

league_groups = df.groupby('League')['ft_result'].value_counts()

# Step 1: Turn the grouped series into a DataFrame
league_df = league_groups.reset_index(name='count')

# Step 2: Pivot the data to get ft_result categories as columns
pivot_df = league_df.pivot(index='League', columns='ft_result', values='count').fillna(0)

# Step 3: Rename columns to lowercase for easier access (optional)
pivot_df.columns = [col.lower() for col in pivot_df.columns]

# Step 4: Create new columns: GP, Won, Draw
pivot_df['GP'] = pivot_df.sum(axis=1)
pivot_df['Won'] = pivot_df.get('home', 0) + pivot_df.get('away', 0)
pivot_df['Draw'] = pivot_df.get('draw', 0)

# Step 5: Keep only the desired columns and reset index
final_df = pivot_df[['GP', 'Won', 'Draw']].reset_index()
final_df['strike'] = round(final_df['Won'] / final_df['GP'] * 100, 2)
final_df['pnl 4'] = round(final_df['Won'] * 98 - (final_df['Draw'] * 300), 2)
final_df['pnl 3.5'] = round(final_df['Won'] * 98 - (final_df['Draw'] * 250), 2)
final_df = final_df.loc[(final_df['GP'] > 10) & (final_df['strike'] > 76)]

final_df.to_excel('league_strike_rate.xlsx')

selected_leagues = final_df['League'].to_list()
print(selected_leagues)

filtered_df = df[df['League'].isin(selected_leagues)]  # Filter data to only include selected leagues
print(len(df))
print(len(filtered_df))
print(filtered_df)


df_LTD_strat = pd.read_sql_query("SELECT * from LTD_strategy_criteria", con)
print(df_LTD_strat)

strat_df = filtered_df[(filtered_df['GP Avg'].astype(float) >= 8) &
                                    ((filtered_df['Form H v A'].astype(float) >= df_LTD_strat.loc[0, 'hva_pos']) | (filtered_df['Form H v A'].astype(float) <= float(df_LTD_strat.loc[0, 'hva_neg']))) &  # Rel2 pos and neg min limits
                                    ((filtered_df['Form Goal Edge'].astype(float) <= float(df_LTD_strat.loc[0, 'goal_edge_pos']))) & 
                                    ((filtered_df['Form Goal Edge'] >= float(df_LTD_strat.loc[0, 'goal_edge_neg']))) &
                                      (filtered_df['Goals 2.5+ L8 Avg'] >=  float(df_LTD_strat.loc[0, 'last8_25']))  # Magic number range
                                    ] 


print(strat_df[['League', 'GP Avg', 'Form H v A', 'Form Goal Edge', 'Goals 2.5+ L8 Avg']])

# print(strat_df[['League', 'GP Avg', 'Form H v A']])
league_groups = strat_df.groupby('League')['ft_result'].value_counts()
print(league_groups)

# Step 1: Turn the grouped series into a DataFrame
league_df = league_groups.reset_index(name='count')

# Step 2: Pivot the data to get ft_result categories as columns
pivot_df = league_df.pivot(index='League', columns='ft_result', values='count').fillna(0)

# Step 3: Rename columns to lowercase for easier access (optional)
pivot_df.columns = [col.lower() for col in pivot_df.columns]

# Step 4: Create new columns: GP, Won, Draw
pivot_df['GP'] = pivot_df.sum(axis=1)
pivot_df['Won'] = pivot_df.get('home', 0) + pivot_df.get('away', 0)
pivot_df['Draw'] = pivot_df.get('draw', 0)

# Step 5: Keep only the desired columns and reset index
final_df = pivot_df[['GP', 'Won', 'Draw']].reset_index()
final_df['strike'] = round(final_df['Won'] / final_df['GP'] * 100, 2)
final_df['pnl 4'] = round(final_df['Won'] * 98 - (final_df['Draw'] * 300), 2)
final_df['pnl 3.5'] = round(final_df['Won'] * 98 - (final_df['Draw'] * 250), 2)

print(final_df.sort_values(by='strike', ascending=False))
final_df.to_excel('strat_league_strike_rate.xlsx')
