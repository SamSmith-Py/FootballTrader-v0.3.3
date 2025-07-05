# imports
import sqlite3
import time
import logging
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from inputimeout import inputimeout, TimeoutOccurred
from itertools import product
from progress.bar import Bar
import matplotlib.pyplot as plt

import betfairlightweight
import numpy as np
import pandas as pd
from betfairlightweight import filters
import pythoncom
import win32com.client

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# Connect to Betfair API
api = betfairlightweight.APIClient('smudge2049', 'Dex17@£141117', '4oAYsDJiYA7P5Wej')
api.login_interactive()

# Paths to databases
autotrader_db_path = r'C:\Users\Sam\FootballTrader v0.3.2\database\autotrader_data.db'

class MatchFinder:
    def __init__(self, hours=24, continuos='on'):
        print('\nMatchFinder')
        # Market projection list to grab data required.
        self.market_projection_list = ['COMPETITION', 'EVENT', 'MARKET_START_TIME',
                                       'RUNNER_DESCRIPTION']
        # Variable placeholders.
        self.df = None
        self.df_sheets = None

        # Dictionary to select market depending on strategy
        self.strategy_market = {'LTD': 'MATCH_ODDS', 'O45': 'OVER_UNDER_45'}

        # Dictionary to select if the strategy should be live or paper
        self.live_select = None

        # Set data types for data frames.
        self.col_dtypes = {}

        if continuos == 'off':
            # Set hours market search time range.
            self.hours = int(input('Select hours to search upto: '))
        if continuos == 'on':
            self.hours = hours

    def get_betfair_details(self):
        """
        Query Betfair's market catalogue api for all football matches within the time range given. Saves all data to a
        df and sqlite database. This data will then later be used to grab additional required stats. The data includes 
        team information, market and selection id's and market start times.
        :return:
        """
        print('\nGetting matches from Betfair...')
        # Time range variables.
        MarketStartTime = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        MarketEndTime = (datetime.now() + timedelta(hours=self.hours))
        MarketEndTime = MarketEndTime.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Filter for markets and time range required.
        cat_filter = betfairlightweight.filters.market_filter(event_type_ids=[1],
                                                              market_type_codes=['MATCH_ODDS', 'OVER_UNDER_45'],
                                                              in_play_only=False,
                                                              market_start_time={"from": MarketStartTime,
                                                                                 "to": MarketEndTime}
                                                              )
        # API call to Betfair Market Catalogue.
        print(cat_filter)
        catalogue = api.betting.list_market_catalogue(filter=cat_filter,
                                                    market_projection=self.market_projection_list,
                                                    max_results=200,
                                                    lightweight=True)
        
        # Save data to data frame.
        self.df = pd.DataFrame(catalogue)

        # Extract required data
        
        self.df['home_team'] = [d[0].get('runnerName') for d in self.df['runners']]
        # self.df['home_odds_id'] = [d[0].get('selectionId') for d in self.df['runners']]
        self.df['away_team'] = [d[1].get('runnerName') for d in self.df['runners']]
        # self.df['away_odds_id'] = [d[1].get('selectionId') for d in self.df['runners']]
        self.df['event_name'] = [d.get('name') for d in self.df['event']]
        self.df['event_id'] = [d.get('id') for d in self.df['event']]
        self.df['marketStartTime'] = pd.to_datetime(self.df['marketStartTime'])
        self.df['start_date'] = self.df['marketStartTime'].dt.strftime('%d.%m.%Y')
        self.df['start_time'] = self.df['marketStartTime'].dt.strftime('%H:%M')
        self.df['comp'] = [d.get('name') for d in self.df['competition']]

        print(self.df)

        market_df = self.df[['event_id', 'marketName', 'marketId']]

        # Pivot the market IDs into separate columns based on market name
        pivot_df = market_df.pivot(index='event_id', columns='marketName', values='marketId').reset_index()
        pivot_df.columns.name = None  # remove hierarchy on columns

        # Rename the columns for clarity
        pivot_df = pivot_df.rename(columns={
            'Match Odds': 'marketID_match_odds',
            'Over/Under 4.5 Goals': 'marketID_overunder45'
        })
        print(pivot_df)

        # Drop duplicate event_ids in the original dataframe (keep first occurrence)
        merged_df = self.df.sort_values('marketName', ascending=True).drop_duplicates(subset='event_id', keep='first')

        # Merge with the pivoted market ID data
        self.df = pd.merge(merged_df, pivot_df, on='event_id', how='left')

        # Clean data
        self.df.dropna(inplace=True)
        self.df.drop(columns=['runners', 'event', 'competition', 'marketName', 'marketId'], inplace=True)

        # Save data to sqlite database.
        cnx = sqlite3.connect(autotrader_db_path, check_same_thread=False)
        self.df.to_sql(name='betfair_matches', con=cnx, if_exists='replace')
        cnx.close()

        # Placeholder for merged dataframes
        self.df_merge_data_and_stats = None
        
        print(self.df)

    def get_sports_iq_stats(self):
        print('\nConnecting to Sports-IQ to download stats...')
        # Initialize Selenium WebDriver with Edge options
        edge_options = Options()
        edge_options.add_argument("--headless=new")  # or "--headless" if "new" doesn't work
        edge_options.add_argument("--window-size=1920,1080")  # needed if elements aren't visible

        # Delete file download if it already exists
        data_file_path = r"C:\Users\Sam\FootballTrader v0.3.2\sports-iq\Football Data Fixtures.xlsx"
        if os.path.exists(data_file_path):
            os.remove(data_file_path)


        prefs = {
                "download.default_directory": r"C:\Users\Sam\FootballTrader v0.3.2\sports-iq",  # Change to your desired directory
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

    def merge_data(self):
        """
        Merge data found from Betfair Market Catalogue with the stats found from Sports-IQ for selected strategies.
        This is done by matching the event_name from self.df to Fixtures from Sports-IQ then merging the data.
        Returns merged df for selected strategies.
        :return: merge_data_match_odds
        """

        # Get Sports-IQ stats as dataframe
        df_sportsiq = pd.read_excel(r'C:\Users\Sam\FootballTrader v0.3.2\sports-iq\Football Data Fixtures.xlsx', engine='openpyxl', header=1)

        # Dataframe for Match Odds market only.
        df_match_odds = self.df.copy()
        # Create list of all event names to add to temp df.
        event_name_list = df_match_odds['event_name'].to_list()
        # Temp df from daily sheets of event.
        df_temp = df_sportsiq[['Kickoff', 'Fixture']].copy()

        # Empty temp df where the matching will take place.
        df_temp_2 = pd.DataFrame(columns=['Fixture', 'seq_score', 'event_name'])
        # Iterate through event name list and match to temp df event.
        while True:
            for event in event_name_list:
                df_temp['seq_score'] = df_temp['Fixture'].apply(lambda e: SequenceMatcher(None, event, e).ratio())
                # Sort so best match is at index 0.
                df_temp.sort_values('seq_score', inplace=True, ascending=False, ignore_index=True)
                # If score below threshold then remove from list as match not found.
                if df_temp.loc[0, 'seq_score'] < 0.8:
                    event_name_list.remove(event)
                # Score above threshold event and sequence match score to temp df.
                else:
                    df_temp.loc[0, 'event_name'] = event
                    data_df = pd.DataFrame({
                                            'Fixture': [df_temp.loc[0, 'Fixture']],
                                            'seq_score': [df_temp.loc[0, 'seq_score']],
                                            'event_name': [df_temp.loc[0, 'event_name']]})
                    # Continually add matched events to empty df.
                    df_temp_2 = pd.concat([df_temp_2, data_df], ignore_index=True, axis=0)
                    # Event can then be removed from list as match has been found.
                    event_name_list.remove(event)
                break
            if len(event_name_list) == 0:
                break
        # Merge matched events to df_sportsiq. Now df_sportsiq will have a column exactly matching event_name
        # from self.df.
        merge_data = pd.merge(df_sportsiq, df_temp_2, on='Fixture', how='inner')
        # Merge self.df and df_sportsiq at event_name.
        self.df_merge_data_and_stats = pd.merge(df_match_odds, merge_data, on='event_name', how='inner')

        # Remove any unnecessary columns
        self.df_merge_data_and_stats.drop(columns=['Kickoff', 'Fixture', 'seq_score'], inplace=True)

        # Find favourite
        self.df_merge_data_and_stats['favourite'] = np.where(self.df_merge_data_and_stats['Odds Betfair Home'] < self.df_merge_data_and_stats['Odds Betfair Away'], 1, 2)

        print(self.df_merge_data_and_stats.columns)

    def add_matches_to_db(self):
        """
        This method will add alll the information obtained from Betfair and SPorts-IQ to the AutoTrader SQLite database. Firstly it will add all the required columns needed for 
        live trading and betting. The user wil alos determine whether the events found will be traded using a live or paper method, pending criteria being matched to a strategy."""
        
        # Create df with the required columns for the AutoTrader
        cols = ['live/paper', 'strategy', 'market', 'inplay_state', 'market_state', 'time_elapsed',
                'home_score', 'away_score', 'score', 'lay_price', 'back_price', 'entry_condition', 'entry_ordered',
                'entry_status', 'entry_price_avg', 'entry_amount_matched', 'entry_amount_remaining', 'exit_condition',
                'exit_ordered', 'exit_status', 'exit_price_avg', 'exit_amount_matched', 'exit_amount_remaining',
                'current_order_status', 'current_order_side', 'current_order_betid', 'potential_pnl', 'cleared_pnl',
                'ht_score', 'ft_score', 'goals_15', 'goals_30', 'goals_45', 'goals_60', 'goals_75', 'goals_90', ]
        df_autotrader = pd.DataFrame(columns=cols)

        # Add new columns to the df created from Betfair details and Sports-IQ
        df_autotrader = pd.concat([df_autotrader, self.df_merge_data_and_stats], axis=1)

        df_autotrader['entry_condition'] = 0
        df_autotrader['entry_ordered'] = 0
        df_autotrader['entry_amount_matched'] = 0
        df_autotrader['exit_condition'] = 0
        df_autotrader['exit_ordered'] = 0
        df_autotrader.dropna(subset=['event_id'], inplace=True)
        values = {'goals_15': 0, 'goals_30': 0, 'goals_45': 0, 'goals_60': 0, 'goals_75': 0, 'goals_90': 0,
                  'home_score': 0, 'away_score': 0}
        df_autotrader.fillna(value=values, inplace=True)
        
        # Save the dataframe to AutoTrader database
        cnx = sqlite3.connect(autotrader_db_path, check_same_thread=False)
        df_autotrader.to_sql(name='autotrader_matches_v3', con=cnx, if_exists='append', index=False) # Change to append once testing is complete
        cnx.close()
        self.remove_duplicates()

    def remove_duplicates(self):
        cnx = sqlite3.connect(autotrader_db_path, check_same_thread=False)
        df = pd.read_sql_query(f"SELECT * from autotrader_matches_v3", cnx, dtype=self.col_dtypes)
        df.drop_duplicates(ignore_index=True, subset=['event_name', 'start_date'], inplace=True)
        df.to_sql(name='autotrader_matches_v3', con=cnx, if_exists='replace', index=False,
                    dtype=self.col_dtypes)
        cnx.close()

class AutoTrader:
    def __init__(self):
        # balance = api.account.get_account_funds()

        # Variable placeholders
        self.cnx = None
        self.df = None
        self.run = None      

        self.col_dtypes = {}   
        self.max_lay_the_draw_price = 5

        # Time required to wait for next run_autotrader run through
        self.wait_time = 10  # Seconds

    def initialise_data(self):
        # Connect to autotrader database if not conected.
        if not self.is_database_connected():
            self.connect_autotrader_db()
        self.df = pd.read_sql_query("SELECT * from autotrader_matches_v2", self.cnx, dtype=self.col_dtypes) 

    def connect_autotrader_db(self):
        """
        Creates an instance for the connection to the autotrader database.
        :return:
        """
        self.cnx = sqlite3.connect(autotrader_db_path, check_same_thread=False)
    
    def close_connection_db(self):
        """
        Close database connections if open.
        :return:
        """
        if self.cnx:
            self.cnx.close()

    def is_database_connected(self):
        try:
            # Attempt to execute a simple query
            self.cnx.execute("SELECT * from autotrader_matches_v2")
            return True
        except sqlite3.ProgrammingError:
            return False
        except AttributeError:
            return False

    def assign_strategy(self, betting='paper'):
        # Connect to autotrader database if not conected.
        if not self.is_database_connected():
            self.connect_autotrader_db()
        # Get LTD strategy criteria
        df_LTD_strat = pd.read_sql_query("SELECT * from LTD_strategy_criteria", self.cnx, dtype=self.col_dtypes)

        # Assign strategy to any events applicable
        self.df['strategy'].loc[
                                (self.df['GP Avg'] >= 8) & 
                                (self.df['Form H v A'] >= df_LTD_strat.loc[0, 'hva_pos'] | self.df['Form H v A'] <= df_LTD_strat.loc[0, 'hva_neg']) &
                                (self.df['Form Goal Edge'] <= df_LTD_strat.loc[0, 'goal_edge_pos']) &
                                (self.df['Form Goal Edge'] >= df_LTD_strat.loc[0, 'goal_edge_neg']) & 
                                (self.df['Goals 2.5+ L8 Avg'] >= df_LTD_strat.loc[0, 'last8_25']) 
                                ] = 'LTD'

        # Decide if live or paper betting for strategy
        self.df['live/paper'].loc[(self.df['strategy'] == 'LTD')] = betting


    def run_autotrader(self, continuous='off'):
        pass

    def stop_autotrader(self):
        """
        Stops the loop that monitors for trade-able matches.
        :return:
        """
        # Check if there is an open connection, if not connect to database.
        if not self.is_database_connected():
            self.connect_autotrader_db()
        self.run = False  # Stop while loop.
        # Save all match updates to database.
        self.df.to_sql(name='autotrader_matches_v3', con=self.cnx, if_exists='replace', index=False)
        self.close_connection_db()  # Close any connection to db.

    def continuos_match_finder(self, activate='off'):
        # This is the Continuos MAtchFinder feature. When this is activated it will stop the AutoTrader tool, run the MatchFinder tool then restart the AutoTrader.
        self.df['marketStartTime'] = pd.to_datetime(self.df['marketStartTime'])
        latest_kickoff  = self.df['marketStartTime'].max()
        if activate == 'on':
            if datetime.now(timezone.utc) > latest_kickoff:
                self.stop_autotrader()
                
                mf = MatchFinder(continuos=activate)
                mf.get_match_details()
                mf.get_daily_sheets()
                df = mf.merge_data()
                if len(df) > 0:
                    mf.add_matches_to_db(df)
                    self.initialise_data()
                    self.assign_strategy()
                self.run_autotrader(continuos=activate)

class BackTester:
    def __init__(self):
        self.con = sqlite3.connect(autotrader_db_path, check_same_thread=False)
        df = pd.read_sql_query("SELECT * from archive_v2", self.con)

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
        df['Odds Betfair Draw'].bfill(inplace=True)
        df['Odds Betfair Draw'].ffill(inplace=True)

        train_data, validate_data, test_data = self.split_data(df)

        # Optimise LTD strategy with Training data
        filt, opt = self.optimise_ltd_strategy(train_data)

        # Validate LTD strategy
        self.validate_ltd_strategy(validate_data, filt, opt)

    def split_data(self, df):
        split_60_percent = round(len(df)*0.6)
        split_80_percent = round(len(df)*0.8)
        
        train_data = df.iloc[:split_60_percent]
        validate_data = df.iloc[split_60_percent:split_80_percent]
        test_data = df.iloc[split_80_percent:]
        return train_data, validate_data, test_data
    
    def optimise_ltd_strategy(self, train_data, df=None):
        
        # Set strategy criteria to be optimised 
        hva_pos = [10, 11, 12, 13, 14, 15]
        hva_neg = [-10, -11, -12, -13, -14, -15]
        goal_edge_pos = [1, 2, 3, 4, 5]
        goal_edge_neg = [-1, -2, -3, -4, -5]
        last8_25 = [0.3, 0.35, 0.4, 0.45, 0.5]
        draw_odds = [5]

        # Create list of combination lists
        combinations = list(product(hva_pos, hva_neg, goal_edge_pos, goal_edge_neg, last8_25, draw_odds))

        # Empty list to append results to display in a df 
        hva_pos_col = []
        hva_neg_col = []
        goal_edge_pos_col = []
        goal_edge_neg_col = []
        last8_25_col = []
        draw_perc_col = []
        total_matches_col = []
        total_no_draw_col = []
        total_draw_col = []
        avg_loss_draw_price = []

        # Iterate through combinations for strategy criteria
        with Bar('Processing...', max=len(combinations), fill="\u26BD") as bar:
            # bar.max(len(train_data))
            for comb in combinations:
                df_opt = train_data[(train_data['GP Avg'].astype(float) >= 8) &
                                    ((train_data['Form H v A'].astype(float) >= comb[0]) | (train_data['Form H v A'].astype(float) <= comb[1])) &  # Rel2 pos and neg min limits
                                    ((train_data['Form Goal Edge'] <= comb[2]) & (train_data['Form Goal Edge'] >= comb[3])) &  # Magic number range
                                    (train_data['Goals 2.5+ L8 Avg'] >= comb[4]) & # Percentage of 2.5 goals
                                    (train_data['Odds Betfair Draw'].astype(float) <= comb[5])]  
                # Add data to relevant column lists
                hva_pos_col.append(comb[0])
                hva_neg_col.append(comb[1])
                goal_edge_pos_col.append(comb[2])
                goal_edge_neg_col.append(comb[3])
                last8_25_col.append(comb[4])

                draw_perc_col.append(df_opt['ft_result'].value_counts(normalize=True).get('draw'))
                total_matches_col.append(len(df_opt))
                total_no_draw_col.append(len(df_opt[(df_opt['ft_result'] != 'draw')]))
                total_draw_col.append(len(df_opt[(df_opt['ft_result'] == 'draw')]))
                draw_result_df = df_opt[(df_opt['ft_result'] == 'draw')]
                draw_result_df.copy().loc[draw_result_df['Odds Betfair Draw'] == 'nan'] = np.nan
                draw_result_df.copy()['Odds Betfair Draw'] = draw_result_df['Odds Betfair Draw'].bfill()
                avg_loss_draw_price.append(draw_result_df['Odds Betfair Draw'].astype(float).mean(skipna=True))
                bar.next()

        # Create df of all optimised results
        optimised_df = pd.DataFrame({'hva_pos': hva_pos_col, 'hva_neg': hva_neg_col, 
                                     'goal_edge_pos': goal_edge_pos_col,'goal_edge_neg': goal_edge_neg_col, 'last8_25': last8_25_col,
                                    'draw_perc': draw_perc_col,
                                    'total_matches': total_matches_col,
                                    'no_draw': total_no_draw_col,
                                    'draw': total_draw_col,
                                    'avg_draw_loss_price': avg_loss_draw_price}).sort_values(by=['draw_perc'])
        
        # Calculate the pnl for all results
        optimised_df['profit'] = optimised_df['no_draw'] * 98
        optimised_df['loss'] = ((optimised_df['avg_draw_loss_price'] - 1) * 100) * optimised_df['draw']
        optimised_df['loss'].loc[optimised_df['draw'] == 0] = 0
        optimised_df['pnl'] = optimised_df['profit'] - optimised_df['loss'] 
        optimised_df = optimised_df.loc[(optimised_df['draw_perc'] <= 0.17) & (optimised_df['total_matches'] > 200)]
        optimised_df.sort_values(by=['pnl', 'total_matches', 'draw_perc'], ascending=[False, False, True], inplace=True)  # sort df by highest pnl value

        # Save the top 1000 LTD strategy optimised results
        optimised_df.head(1000).to_html(r"C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Optimised_Strategy_Results\optimised_LTD_strategy.html")
        optimised_df.reset_index(inplace=True)

        # Apply general strategy to train data    
        idx = 0   
        df_train = train_data[(train_data['GP Avg'].astype(float) >= 8) &
                              ((train_data['Form H v A'].astype(float) >= optimised_df.loc[idx, 'hva_pos']) | (train_data['Form H v A'].astype(float) <= optimised_df.loc[idx, 'hva_neg'])) &  # Rel2 pos and neg min limits
                                ((train_data['Form Goal Edge'] <= optimised_df.loc[idx, 'goal_edge_pos']) & (train_data['Form Goal Edge'] >= optimised_df.loc[idx, 'goal_edge_neg'])) &  # Magic number range
                                (train_data['Goals 2.5+ L8 Avg'] >= optimised_df.loc[idx, 'last8_25']) &
                                    (train_data['Odds Betfair Draw'].astype(float) <= 5)]
        df_train['pnl'] = 98
        df_train['pnl'] = np.where(df_train['ft_result'] == 'draw', -100 * (df_train['Odds Betfair Draw'].astype(float) - 1), df_train['pnl'])
        df_train['cumsum'] = df_train.copy()['pnl'].cumsum()
        df_train.reset_index(inplace=True)
        df_train.to_html(r"C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Optimised_Strategy_Results\all_trades.html")
        df_train['cumsum'].plot(title='LTD Strategy Training Data - P&L').legend(['Optimised GS'])
        plt.savefig(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Plots\optimised_LTD_strategy_train_data_pnl.png')

        # create league performance results
        leagues_pnl = round(df_train.groupby(['League'])['pnl'].sum().reset_index(), 2)
        leagues_lose = df_train.groupby('League')['ft_result'].apply(lambda x: (x=='draw').sum()).reset_index()
        leagues_lose.rename(columns={'ft_result': 'loss'}, inplace=True)
        leagues_win = df_train.groupby('League')['ft_result'].apply(lambda x: (x!='draw').sum()).reset_index()
        leagues_win.rename(columns={'ft_result': 'win'}, inplace=True)
        league_data = leagues_pnl.merge(leagues_win, on='League', how='inner').merge(leagues_lose, on='League', how='inner')
        league_data['win_rate'] = round(league_data['win']/(league_data['win'] + league_data['loss'])*100, 2)
        league_data.sort_values('pnl', ascending=False).to_html(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Optimised_Strategy_Results\optimised_LTD_league_performance.html')
        
        poor_result_league = league_data.loc[(league_data['win_rate'] < 80) & (league_data['win'] + league_data['loss'] > 3)]
        print(poor_result_league.sort_values(by='pnl'))
        print(poor_result_league['pnl'].sum())

        # Create a list of underpeforming leagues to remove from validation and testing. 
        ## These leagues are identified to be unprofitable using LTD strategy.
        # under_performing = ['England, National League', 'Denmark, Superliga', 'England, League Two', 'Italy, Serie B', 'Austria, 2. Liga', 'Germany, Bundesliga']
        under_performing = ['Denmark, Superliga', 'Italy, Serie B', 'Germany, Bundesliga']
        all_leagues = league_data['League'].values
        filtered_leagues = [league for league in all_leagues if league not in under_performing]
        remove_league_train_data = train_data[train_data['League'].isin(filtered_leagues)]

        remove_league_train_data = remove_league_train_data[(remove_league_train_data['GP Avg'].astype(float) >= 8) &
                              ((remove_league_train_data['Form H v A'].astype(float) >= optimised_df.loc[idx, 'hva_pos']) | (remove_league_train_data['Form H v A'].astype(float) <= optimised_df.loc[idx, 'hva_neg'])) &  # Rel2 pos and neg min limits
                                ((remove_league_train_data['Form Goal Edge'] <= optimised_df.loc[idx, 'goal_edge_pos']) & (remove_league_train_data['Form Goal Edge'] >= optimised_df.loc[idx, 'goal_edge_neg'])) &  # Magic number range
                                (remove_league_train_data['Goals 2.5+ L8 Avg'] >= optimised_df.loc[idx, 'last8_25']) &
                                    (remove_league_train_data['Odds Betfair Draw'].astype(float) <= 5)]
        remove_league_train_data['pnl'] = 98
        remove_league_train_data['pnl'] = np.where(remove_league_train_data['ft_result'] == 'draw', -100 * (remove_league_train_data['Odds Betfair Draw'].astype(float) - 1), remove_league_train_data['pnl'])
        remove_league_train_data['cumsum'] = remove_league_train_data.copy()['pnl'].cumsum()
        remove_league_train_data.reset_index(inplace=True)
        remove_league_train_data.to_html(r"C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Optimised_Strategy_Results\all_trades_remove_league.html")
        remove_league_train_data['cumsum'].plot(title='LTD Strategy Training Data - P&L').legend(['Optimised LTD', 'Removed Leagues LTD'])
        plt.savefig(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Plots\remove_league_train_data_pnl.png')

        return filtered_leagues, optimised_df

    def validate_ltd_strategy(self, validate_data, filtered_leagues, optimised_df):
        idx=0

        # Apply validated general strategy criteria to validate data
        df = validate_data[(validate_data['GP Avg'].astype(float) >= 8) &
                              ((validate_data['Form H v A'].astype(float) >= optimised_df.loc[idx, 'hva_pos']) | (validate_data['Form H v A'].astype(float) <= optimised_df.loc[idx, 'hva_neg'])) &  # Rel2 pos and neg min limits
                                ((validate_data['Form Goal Edge'] <= optimised_df.loc[idx, 'goal_edge_pos']) & (validate_data['Form Goal Edge'] >= optimised_df.loc[idx, 'goal_edge_neg'])) &  # Magic number range
                                (validate_data['Goals 2.5+ L8 Avg'] >= optimised_df.loc[idx, 'last8_25']) &
                                    (validate_data['Odds Betfair Draw'].astype(float) <= 5)]
        
        # Add pnl and cumsum 
        df['pnl'] = 98
        df['pnl'] = np.where(df['ft_result'] == 'draw', -100 * (df['Odds Betfair Draw'].astype(float) - 1), df['pnl'])
        df['cumsum'] = df['pnl'].cumsum()
        df.reset_index(inplace=True)
        df.to_html(r"C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Validated_Strategy_Results\all_trades.html")
        df['cumsum'].plot(title='Validated General Strategy - Validate Data - P&L').legend(['Optimised LTD', 'Removed Leagues LTD', 'Validated LTD'])
        plt.savefig(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Plots\validated_LTD_strategy_validate_data_pnl.png')

        # create league perormance results
        leagues_pnl = round(df.groupby(['League'])['pnl'].sum().reset_index(), 2)
        leagues_lose = df.groupby('League')['ft_result'].apply(lambda x: (x=='draw').sum()).reset_index()
        leagues_lose.rename(columns={'ft_result': 'loss'}, inplace=True)
        leagues_win = df.groupby('League')['ft_result'].apply(lambda x: (x!='draw').sum()).reset_index()
        leagues_win.rename(columns={'ft_result': 'win'}, inplace=True)
        league_data = leagues_pnl.merge(leagues_win, on='League', how='inner').merge(leagues_lose, on='League', how='inner')
        league_data['win_rate'] = round(league_data['win']/(league_data['win'] + league_data['loss'])*100, 2)
        league_data.sort_values('pnl', ascending=False).to_html(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Validated_Strategy_Results\validated_LTD_league_performance.html')

        print(df)
        print(df['pnl'].sum())

        validate_data = validate_data[validate_data['League'].isin(filtered_leagues)]
        # Apply validated general strategy criteria to validate data
        df = validate_data[(validate_data['GP Avg'].astype(float) >= 8) &
                              ((validate_data['Form H v A'].astype(float) >= optimised_df.loc[idx, 'hva_pos']) | (validate_data['Form H v A'].astype(float) <= optimised_df.loc[idx, 'hva_neg'])) &  # Rel2 pos and neg min limits
                                ((validate_data['Form Goal Edge'] <= optimised_df.loc[idx, 'goal_edge_pos']) & (validate_data['Form Goal Edge'] >= optimised_df.loc[idx, 'goal_edge_neg'])) &  # Magic number range
                                (validate_data['Goals 2.5+ L8 Avg'] >= optimised_df.loc[idx, 'last8_25']) &
                                    (validate_data['Odds Betfair Draw'].astype(float) <= 5)]
        
        # Add pnl and cumsum 
        df['pnl'] = 98
        df['pnl'] = np.where(df['ft_result'] == 'draw', -100 * (df['Odds Betfair Draw'].astype(float) - 1), df['pnl'])
        df['cumsum'] = df['pnl'].cumsum()
        df.reset_index(inplace=True)
        df.to_html(r"C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Validated_Strategy_Results\all_trades_remove_leagues.html")
        df['cumsum'].plot(title='Validated General Strategy - Validate Data - P&L').legend(['Optimised LTD', 'Removed Leagues LTD', 'Validated LTD', 'Validated Removed Leagues LTD'])
        plt.savefig(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Plots\removed_leagues_validated_LTD_strategy_validate_data_pnl.png')

        print(df)
        print(df['pnl'].sum())

        # create league perormance results
        leagues_pnl = round(df.groupby(['League'])['pnl'].sum().reset_index(), 2)
        leagues_lose = df.groupby('League')['ft_result'].apply(lambda x: (x=='draw').sum()).reset_index()
        leagues_lose.rename(columns={'ft_result': 'loss'}, inplace=True)
        leagues_win = df.groupby('League')['ft_result'].apply(lambda x: (x!='draw').sum()).reset_index()
        leagues_win.rename(columns={'ft_result': 'win'}, inplace=True)
        league_data = leagues_pnl.merge(leagues_win, on='League', how='inner').merge(leagues_lose, on='League', how='inner')
        league_data['win_rate'] = round(league_data['win']/(league_data['win'] + league_data['loss'])*100, 2)
        league_data.sort_values('pnl', ascending=False).to_html(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Validated_Strategy_Results\removed_leagues_validated_LTD_league_performance.html')


if __name__ == '__main__':
    #pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    #pd.set_option('expand_frame_repr', False)
    #mf = MatchFinder(continuos='on')
    #mf.get_betfair_details()
    #mf.get_sports_iq_stats()
    #mf.merge_data()
    #mf.add_matches_to_db()

    bt = BackTester()

