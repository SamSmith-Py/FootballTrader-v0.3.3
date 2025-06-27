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

# Connect to Betfair API
api = betfairlightweight.APIClient('smudge2049', 'Dex17@£141117', '4oAYsDJiYA7P5Wej')
api.login_interactive()

# Paths to databases
autotrader_db_path = r'C:\Users\Sam\FootballTrader v0.3.2\database\autotrader_data.db'

class MatchFinder:
    def __init__(self, hours=12, continuos='off'):
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

    def get_match_details(self):
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
        # self.df['home_team'] = [d[0].get('runnerName') for d in self.df['runners']]
        # self.df['home_odds_id'] = [d[0].get('selectionId') for d in self.df['runners']]
        # self.df['away_team'] = [d[1].get('runnerName') for d in self.df['runners']]
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
        merged_df = self.df.drop_duplicates(subset='event_id', keep='first')

        # Merge with the pivoted market ID data
        self.df = pd.merge(merged_df, pivot_df, on='event_id', how='left')

        # Clean data
        self.df.dropna(inplace=True)
        self.df.drop(columns=['runners', 'event', 'competition'], inplace=True)

        # Save data to sqlite database.
        cnx = sqlite3.connect(autotrader_db_path, check_same_thread=False)
        self.df.to_sql(name='betfair_matches', con=cnx, if_exists='replace')
        cnx.close()
        
        print(self.df)

    def get_daily_sheets(self):
        print('\nQuerying Daily Sheets for stats...')
        # Path to your .xlsm file
        file_path = r'C:\Users\Sam\FootballTrader v0.3.1\DailySheets\TTM Football Selection Tool V4.xlsm'

        # Macro name (ensure it includes the module name if it's within a module)
        macro_name = 'Module1.Refresh_Data'

        # Initialize COM
        pythoncom.CoInitialize()

        # Open Excel application
        excel_app = win32com.client.Dispatch('Excel.Application')
        excel_app.Application.DisplayAlerts = False

        try:
            # Open the .xlsm file
            workbook = excel_app.Workbooks.Open(file_path)

            # Run the macro
            excel_app.Application.Run(macro_name)

            # Save and close the workbook
            workbook.Save()
            workbook.Close()
        finally:
            # Quit the Excel application
            excel_app.Quit()

            # Ensure COM is uninitialized
            pythoncom.CoUninitialize()

        self.df_sheets = pd.read_excel(
            r'C:\Users\Sam\FootballTrader v0.3.2\DailySheets\TTM Football Selection Tool V4.xlsm',
            sheet_name='Sheet1').dropna()
        self.df_sheets.columns = self.df_sheets.iloc[0]

        self.df_sheets = self.df_sheets.drop(self.df_sheets.index[0])

        self.df_sheets.rename(columns={'Home Team': 'home_team'}, inplace=True)
        self.df_sheets.rename(columns={'Away Team': 'away_team'}, inplace=True)

        self.df_sheets['event'] = self.df_sheets['home_team'] + ' v ' + self.df_sheets['away_team']

        self.df_sheets.to_excel('daily_sheet_stats.xlsx', engine='openpyxl')
        cnx = sqlite3.connect(autotrader_db_path, check_same_thread=False)
        self.df_sheets.to_sql(name='daily_sheet_stats', con=cnx, if_exists='replace')
        cnx.close()


class AutoTrader:
    def __init__(self):
        pass

class BackTester:
    def __init__(self):
        pass

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # pd.set_option('expand_frame_repr', False)
    mf = MatchFinder(continuos='off')
    mf.get_match_details()