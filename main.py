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


api = betfairlightweight.APIClient(self.username, self.password, self.app_key)
class AutoTrader:
    def __init__(self):
        # balance = api.account.get_account_funds()

        # Variable placeholders
        self.cnx = None
        self.df = None
        self.run = None      

        self.col_dtypes = {}   
        self.max_lay_the_draw_price = 5
        self.ltd_paper_stake_size = 100
        self.ltd_live_stake_size = 3

        # Time required to wait for next run_autotrader run through
        self.wait_time = 10  # Seconds

        self.initialise_data()
        self.assign_strategy()

    def initialise_data(self):
        # Connect to autotrader database if not conected.
        if not self.is_database_connected():
            self.connect_autotrader_db()
        self.df = pd.read_sql_query("SELECT * from autotrader_matches_v3", self.cnx, dtype=self.col_dtypes) 


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
            self.cnx.execute("SELECT * from autotrader_matches_v3")
            return True
        except sqlite3.ProgrammingError:
            return False
        except AttributeError:
            return False

    def assign_strategy(self, betting='live'):
        """
        Assign's strategies to events and gives the option to bet on them live or paper.
        """
        print('testing assign strategy')
        # Connect to autotrader database if not conected.
        if not self.is_database_connected():
            self.connect_autotrader_db()

        # Get list of leagues that are eligible to assign. These leagues are filtered first from the league strike rate and then from the optimised league performance results, then filtered again to get leagues with more than 75% strike rate.
        # This leaves only leagues that have a track record of low drawing matches and proven to work with the current strategy criteria.
        leagues = pd.read_html(r'C:\Users\Sam\FootballTrader v0.3.2\backtest\strategy\LTD\Q4 2025\Optimised_Strategy_Results\optimised_LTD_league_performance.html', index_col=0)[0]
        leagues = leagues.loc[leagues['win_rate']>=75, 'League'].to_list()

        # Get LTD strategy criteria
        df_LTD_strat = pd.read_sql_query("SELECT * from LTD_strategy_criteria", self.cnx, dtype=self.col_dtypes)
        # Assign strategy to any events applicable
        for row in self.df.index:
            if int(self.df.loc[row, 'GP Avg']) >= 0 and \
                (int(self.df.loc[row, 'Form H v A']) >= df_LTD_strat.loc[0, 'hva_pos'] or int(self.df.loc[row, 'Form H v A']) <= df_LTD_strat.loc[0, 'hva_neg']) and \
                float(self.df.loc[row, 'Form Goal Edge']) <= df_LTD_strat.loc[0, 'goal_edge_pos'] and \
                float(self.df.loc[row, 'Form Goal Edge']) >= float(df_LTD_strat.loc[0, 'goal_edge_neg']) and \
                self.df.loc[row, 'League'] in leagues:
                    self.df.loc[row, 'strategy'] = 'LTD'
            else:
                self.df.loc[row, 'strategy'] = None
                self.df['live/paper'].loc[(self.df['strategy'] == 'LTD')] = None
        
        # Decide if live or paper betting for strategy
        self.df['live/paper'].loc[(self.df['strategy'] == 'LTD')] = betting

        # Save all updates to database
        self.df.to_sql(name='autotrader_matches_v3', con=self.cnx, if_exists='replace', index=False,
                        dtype=self.col_dtypes)
        self.close_connection_db()

    def run_autotrader(self, continuous='off'):
        """
        Starts the loop to monitor for trade-able matches. Loop will stop if no matches are found.
        :return:
        """
        # Ensure while loop starts if run_autotrader is called after a stop_autotrader.
        self.run = True

        while self.run:
            # Connect to autotrader database and set to data frame.
            self.initialise_data()
            if len(self.df) > 0:  # Check there is data.
                for row in self.df.index:
                    # Get event id
                    event_id = self.df.loc[row, 'event_id']
                    # Run queries for selected match.
                    try:
                        # Check if matches have been removed.
                        if not self.check_inplay_state(event_id=event_id, idx=row):
                            self.check_market_state(idx=row)
                            self.check_time_elapsed(event_id=event_id, idx=row)
                            self.check_score(event_id=event_id, idx=row)
                            self.check_lay_price(idx=row)
                            self.check_back_price(idx=row)
                            self.check_current_orders(idx=row, check=1)
                            self.check_potential_uncleared_pnl(idx=row)
                        else:
                            break
                    except betfairlightweight.exceptions.APIError:
                        print('Connection lost. Attempting reconnect...')
                        logging.exception('API ERROR')
                        time.sleep(10)
                        continue
                    # Check for match strategy conditions.
                    try:
                        self.strategy_ltd(idx=row)
                    except betfairlightweight.exceptions.APIError:
                        print('Connection lost. Attempting reconnect...')
                        logging.exception('API ERROR')
                        time.sleep(10)
                        continue
            elif len(self.df) == 0:  
                # If no data, wait until next hour for continous MatchFinder
                if continuous == 'on': 
                    # Get the current time
                    now = datetime.now()
                    # Calculate the next hour
                    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    print(next_hour)
                    # Calculate the time difference in seconds
                    time_to_sleep = (next_hour - now).total_seconds()
                    print(time_to_sleep)
                    # Sleep until the next hour
                    print('Waiting.')
                    time.sleep(time_to_sleep)
                    print('Running MatchFinder after wait.')
                    self.continuos_match_finder(activate=continuous)
                    print('MatchFinder complete after wait.')
                # If continuous MatchFinder is off then stop autotrader
                else:
                    self.stop_autotrader()
                    print('Stopping 1')
            
            # Connect to autotrader database and set to data frame.
            if not self.is_database_connected():
                self.connect_autotrader_db()
            # Save all updates to database
            self.df.to_sql(name='autotrader_matches_v3', con=self.cnx, if_exists='replace', index=False,
                           dtype=self.col_dtypes)
            self.close_connection_db()

            # TESTING
            if len(self.df.loc[self.df['live/paper'] == 'live']) > 0:
                print(self.df.loc[self.df['live/paper'] == 'live', ['event_name', 'League', 'live/paper', 'strategy', 'start_date', 'start_time', 'inplay_state', 'time_elapsed', 'market_state',
                           'score', 'entry_ordered', 'entry_amount_matched', 'GP Avg', 'Form Goal Edge', 'favourite']].sort_values(by=['start_date', 'start_time']))
            else: 
                 print(self.df[['event_name', 'League', 'live/paper', 'strategy', 'start_date', 'start_time', 'inplay_state', 'time_elapsed', 'market_state',
                           'score', 'entry_ordered', 'entry_amount_matched', 'GP Avg', 'Form Goal Edge', 'favourite']].sort_values(by=['start_date', 'start_time']))

            self.continuos_match_finder(activate=continuous)

            try:
                quit_running = inputimeout(prompt='Stop runnning? y/n ', timeout=5)
                if quit_running == 'y':
                    self.stop_autotrader()
            except TimeoutOccurred:
                pass

            # Wait
            time.sleep(self.wait_time)

    def check_inplay_state(self, event_id, idx):
        """
        Checks the in play state of the selected match. Updates database. If match has finished and traded, archive and
        remove from databases as necessary.
        :return:
        """
        try:
            get_scores = api.in_play_service.get_scores(event_ids=[event_id])
            # Get inplay state if available.
            
            if len(get_scores) > 0:
                for x in get_scores:
                    
                    self.df.loc[idx, 'inplay_state'] = x.match_status

            # If time is over 3 hours past kick off then mark as finished
            self.df['marketStartTime'] = pd.to_datetime(self.df['marketStartTime'])
            if datetime.now(timezone.utc) - self.df.loc[idx, 'marketStartTime'] > timedelta(hours=4):
                self.df.loc[idx, 'ft_score'] = self.df.loc[idx, 'score']
                self.df.loc[idx, 'inplay_state'] = 'Finished'
                self.check_paper_bet_result(idx)

            # If match finished check if traded. If traded then update trading log. Archive all and remove from trading log
            # database if un-traded.
            if self.df.loc[idx, 'inplay_state'] == 'Finished':
                self.df.loc[idx, 'ft_score'] = self.df.loc[idx, 'score']
                self.check_paper_bet_result(idx)
                self.check_cleared_orders_pnl(idx=idx)
                self.archive_autotrader_match(idx)
                self.remove_finished_matches_autotrader(idx)
                return True
            return False
        except betfairlightweight.exceptions.StatusCodeError:
            print('GetScores Status code error')
            logging.exception('STATUS CODE ERROR')
            
    
    def check_market_state(self, idx):
        """
        Checks market state of selected match. Updates database.
        :return:
        """
        if self.df.loc[idx, 'strategy'] == 'LTD':
            market_book = api.betting.list_market_book(market_ids=[str(self.df.loc[idx, 'marketID_match_odds'])])
            if len(market_book) > 0:
                self.df.loc[idx, 'market_state'] = market_book[0].status
        if self.df.loc[idx, 'strategy'] == 'LO45':
            market_book = api.betting.list_market_book(market_ids=[str(self.df.loc[idx, 'marketID_overunder45'])])
            if len(market_book) > 0:
                self.df.loc[idx, 'market_state'] = market_book[0].status

    def check_time_elapsed(self, event_id, idx):
        """
        Checks the time elapsed for selected match. Shows negative number to indicate minutes before match starts.
        Updates database.
        :return:
        """
        try:
            time_elap = api.in_play_service.get_scores(event_ids=[event_id])
            if len(time_elap) > 0:
                self.df.loc[idx, 'time_elapsed'] = time_elap[0].time_elapsed
        except betfairlightweight.exceptions.StatusCodeError:
            print('GetScores Status code error')
            pass

    def check_score(self, event_id, idx):
        """
        Checks score for selected match. Updates database.
        :return:
        """
        try:
            scores = api.in_play_service.get_scores(event_ids=[event_id])
        
            if len(scores) > 0 and self.df.loc[idx, 'time_elapsed'] != None:
                for x in scores:
                    self.df.loc[idx, 'home_score'] = int(x.score.home.score)
                    self.df.loc[idx, 'away_score'] = int(x.score.away.score)
                    self.df.loc[idx, 'score'] = f'{int(x.score.home.score)} - {int(x.score.away.score)}'
                try:
                    
                    # Get score for first 15 minutes
                    if self.df.loc[idx, 'time_elapsed'] <= 15:
                        self.df.loc[idx, 'goals_15'] = f"{self.df.loc[idx, 'home_score']} - {self.df.loc[idx, 'away_score']}"            
                    # Get score within 15 - 30 minutes
                    if 15 < self.df.loc[idx, 'time_elapsed'] <= 30 and self.df.loc[idx, 'goals_15'] is not None:
                        self.df.loc[idx, 'goals_30'] = f"{int(self.df.loc[idx, 'home_score'])} - {int(self.df.loc[idx, 'away_score'])}"
                    # Get score within 30 - 45 minutes    
                    if 30 < self.df.loc[idx, 'time_elapsed'] <= 60 and \
                            self.df.loc[idx, 'inplay_state'] == 'KickOff' and \
                            self.df.loc[idx, 'goals_15'] is not None and \
                            self.df.loc[idx, 'goals_30'] is not None:
                        self.df.loc[idx, 'goals_45'] = f"{int(self.df.loc[idx, 'home_score'])} - {int(self.df.loc[idx, 'away_score'])}"
                    # Get score within 45 - 60 minutes     
                    if 45 <= self.df.loc[idx, 'time_elapsed'] <= 60 and \
                            self.df.loc[idx, 'inplay_state'] == 'SecondHalfKickOff' and \
                            self.df.loc[idx, 'goals_15'] is not None and \
                            self.df.loc[idx, 'goals_30'] is not None and \
                            self.df.loc[idx, 'goals_45'] is not None:
                        self.df.loc[idx, 'goals_60'] = f"{int(self.df.loc[idx, 'home_score'])} - {int(self.df.loc[idx, 'away_score'])}"
                    # Get score within 60 - 75 minutes   
                    if 60 < self.df.loc[idx, 'time_elapsed'] <= 75 and \
                            self.df.loc[idx, 'goals_15'] is not None and \
                            self.df.loc[idx, 'goals_30'] is not None and \
                            self.df.loc[idx, 'goals_45'] is not None and \
                            self.df.loc[idx, 'goals_60'] is not None:
                        self.df.loc[idx, 'goals_75'] = f"{int(self.df.loc[idx, 'home_score'])} - {int(self.df.loc[idx, 'away_score'])}"
                    # Get score within 75 - 90 minutes
                    if 75 < self.df.loc[idx, 'time_elapsed'] <= 120 and \
                            self.df.loc[idx, 'goals_15'] is not None and \
                            self.df.loc[idx, 'goals_30'] is not None and \
                            self.df.loc[idx, 'goals_45'] is not None and \
                            self.df.loc[idx, 'goals_60'] is not None and \
                            self.df.loc[idx, 'goals_75'] is not None:
                        self.df.loc[idx, 'goals_90'] = f"{int(self.df.loc[idx, 'home_score'])} - {int(self.df.loc[idx, 'away_score'])}"

                    if self.df.loc[idx, 'inplay_state'] == 'FirstHalfEnd':
                        self.df.loc[idx, 'ht_score'] = self.df.loc[idx, 'score']    

                except TypeError:
                    logging.exception()
        except betfairlightweight.exceptions.StatusCodeError:
            print('GetScores Status code error')
            logging.exception('STATUS CODE ERROR')

    def check_lay_price(self, idx):
        """
        Checks the lay price of the market for selected match. Updates database.
        :return:
        """
        if self.df.loc[idx, 'strategy'] == 'LTD':
            try:
                # Get current lay price.
                market_books = api.betting.list_market_book(market_ids=[market_id])
                market_books[0].runners[0].sp.actual_sp
                lay_price = api.betting.list_runner_book(market_id=str(self.df.loc[idx, 'marketID_match_odds']),
                                                        selection_id=58805,
                                                        price_projection={'priceData': ['EX_ALL_OFFERS']})
                lay_the_draw_price = lay_price[0].runners[0].ex.available_to_lay[0].price

                self.df.loc[idx, 'lay_price'] = lay_the_draw_price
            except IndexError:
                self.df.loc[idx, 'lay_price'] = -1
            except betfairlightweight.exceptions.StatusCodeError:
                self.df.loc[idx, 'lay_price'] = -1

        if self.df.loc[idx, 'strategy'] == 'LO45':
            try:
                # Get current lay price.
                lay_price = api.betting.list_runner_book(market_id=str(self.df.loc[idx, 'marketID_overunder45']),
                                                        selection_id=1222346,
                                                        price_projection={'priceData': ['EX_ALL_OFFERS']})
                lay_the_draw_price = lay_price[0].runners[0].ex.available_to_lay[0].price

                self.df.loc[idx, 'lay_price'] = lay_the_draw_price
            except IndexError:
                self.df.loc[idx, 'lay_price'] = -1
            except betfairlightweight.exceptions.StatusCodeError:
                self.df.loc[idx, 'lay_price'] = -1

    def check_back_price(self, idx):
        """
        Checks the back price of the market for selected match. Updates database.
        :return:
        """
        if self.df.loc[idx, 'strategy'] == 'LTD':
            try:
                # Get current back price.
                back_price = api.betting.list_runner_book(market_id=str(self.df.loc[idx, 'marketID_match_odds']),
                                                        selection_id=58805,
                                                        price_projection={'priceData': ['EX_ALL_OFFERS']})
                back_price[0].runners[0].sp.actual_sp
                back_the_draw_price = back_price[0].runners[0].ex.available_to_back[0].price
                self.df.loc[idx, 'back_price'] = back_the_draw_price
            except IndexError:
                self.df.loc[idx, 'back_price'] = -1
            except betfairlightweight.exceptions.StatusCodeError:
                    self.df.loc[idx, 'back_price'] = -1
        if self.df.loc[idx, 'strategy'] == 'LO45':
            try:
                # Get current back price.
                back_price = api.betting.list_runner_book(market_id=str(self.df.loc[idx, 'marketID_overunder45']),
                                                        selection_id=1222346,
                                                        price_projection={'priceData': ['EX_ALL_OFFERS']})
                back_the_draw_price = back_price[0].runners[0].ex.available_to_back[0].price
                self.df.loc[idx, 'back_price'] = back_the_draw_price
            except IndexError:
                self.df.loc[idx, 'back_price'] = -1
            except betfairlightweight.exceptions.StatusCodeError:
                    self.df.loc[idx, 'back_price'] = -1

    def check_current_orders(self, idx, check=0):
        """
        Check all current orders.
        If there are Lay/Back orders then an entry/exit has been ordered. Update order status, if there are multiple
        orders of the same side then get average price. Get size matched and remaining for entry and exits.
        If size remains unmatched then order should be updated with current price and the remaining size that requires
        matching.
        :return:
        """
        if self.df.loc[idx, 'live/paper'] == 'live' and self.df.loc[idx, 'strategy'] == 'LTD':
            orders = api.betting.list_current_orders(market_ids=[self.df.loc[idx, 'marketID_match_odds']],
                                                     sort_dir='LATEST_TO_EARLIEST')
            
            if len(orders.orders) > 0:
                self.df.loc[idx, 'current_order_side'] = orders.orders[0].side
                self.df.loc[idx, 'current_order_status'] = orders.orders[0].status
                self.df.loc[idx, 'current_order_betid'] = orders.orders[0].bet_id
                for order in orders['currentOrders']:
                    # Check if any Lay orders.
                    if order['side'] == 'LAY':
                        # Update if entry ordered.
                        self.df.loc[idx, 'entry_ordered'] = 1
                        # Update entry order status.
                        self.df.loc[idx, 'entry_status'] = order['status']
                        # get average price of entry's.
                        self.df.loc[idx, 'entry_price_avg'] = round(order['averagePriceMatched'], 2)
                        # Update entry size matched and remaining.
                        self.df.loc[idx, 'entry_amount_matched'] = order['sizeMatched']
                        self.df.loc[idx, 'entry_amount_remaining'] = order['sizeRemaining']
                    # Check if any Back orders. - No current need for BACK ORDERS this will need to be revised in the future.
                    """if order['side'] == 'BACK':
                        # Update if exit ordered.
                        self.df.loc[idx, 'exit_ordered'] = 1
                        # Update exit order status.
                        self.df.loc[idx, 'exit_status'] = order['status']
                        # get average price of exit's.
                        self.df.loc[idx, 'exit_price_avg'] = round(order['averagePriceMatched'], 2)
                        # Update exit size matched and remaining.
                        self.df.loc[idx, 'exit_amount_matched'] = order['sizeMatched']
                        self.df.loc[idx, 'exit_amount_remaining'] = order['sizeRemaining']"""
                # Check if no orders.
                if len(orders['currentOrders']) == 0:
                    self.df.loc[idx, 'entry_ordered'] = 0
                    self.df.loc[idx, 'exit_ordered'] = 0

                # Check if orders partially matched and update.
                if check == 1:
                    # E
                    if self.df.loc[idx, 'entry_amount_remaining'] >= 1.0 and self.df.loc[idx, 'entry_amount_remaining'] <= self.ltd_live_stake_size * 0.25:
                        self.update_current_orders(idx, self.df.loc[idx, 'lay_price'],
                                                self.df.loc[idx, 'current_order_betid'])
                    # No current need for BACK ORDERS this will need to be revised in the future.
                    """if self.df.loc[idx, 'exit_amount_remaining'] >= 1.0:
                        self.update_current_orders(idx, self.df.loc[idx, 'lay_price'],
                                                self.df.loc[idx, 'current_order_betid'])"""
                    
    def update_current_orders(self, idx, price, betid):
        if self.df.loc[idx, 'strategy'] == 'LTD':
            instruction = filters.replace_instruction(bet_id=betid, new_price=price)
            api.betting.replace_orders(market_id=str(self.df.loc[idx, 'marketID_match_odds']), instructions=[instruction])

    def check_potential_uncleared_pnl(self, idx):
        if self.df.loc[idx, 'entry_ordered'] == 1:
            back_stake = round(self.df.loc[idx, 'entry_price_avg'] *
                               self.df.loc[idx, 'entry_amount_matched'] /
                               self.df.loc[idx, 'back_price'], 2)
            self.df.loc[idx, 'potential_pnl'] = round(self.df.loc[idx, 'entry_amount_matched'] - back_stake, 2)
        # No current need for BACK ORDERS this will need to be revised in the future.
        """if self.df.loc[idx, 'exit_ordered'] == 1:
            back_stake = round(self.df.loc[idx, 'entry_price_avg'] *
                               self.df.loc[idx, 'entry_amount_matched'] /
                               self.df.loc[idx, 'exit_price_avg'], 2)
            self.df.loc[idx, 'potential_pnl'] = round(self.df.loc[idx, 'entry_amount_matched'] - back_stake, 2)
        """

    def check_cleared_orders_pnl(self, idx):
        if self.df.loc[idx, 'strategy'] == 'LTD' and float(self.df.loc[idx, 'entry_amount_matched']) > 0 and self.df.loc[idx, 'live/paper'] == 'live':
            cleared = api.betting.list_cleared_orders(market_ids=[self.df.loc[idx, 'marketID_match_odds']], group_by='MARKET')
            print(cleared)
            if len(cleared['clearedOrders']) > 0:
                self.df.loc[idx, 'cleared_pnl'] = cleared.orders[0].profit

    def place_lay_order(self, size, price, ptype, idx, side):
        selection_id = {'LTD': 58805}
        if self.df.loc[idx, 'live/paper'] == 'live':
            limit_order = filters.limit_order(size=size, price=price, persistence_type=ptype)
            instruction = filters.place_instruction(
                order_type="LIMIT",
                selection_id=selection_id[self.df.loc[idx, 'strategy']],
                side=side,
                limit_order=limit_order)
            place_orders = api.betting.place_orders(
                market_id=str(self.df.loc[idx, 'marketID_match_odds']), instructions=[instruction])
            if place_orders.place_instruction_reports[0].status == 'SUCCESS':
                self.check_current_orders(idx=idx)
                self.df.loc[idx, 'entry_status'] = place_orders.place_instruction_reports[0].order_status
            if place_orders.place_instruction_reports[0].status == 'TIMEOUT':
                time.sleep(15)
                self.check_current_orders(idx=idx)
            if place_orders.place_instruction_reports[0].status == 'FAILURE':
                self.df.loc[idx, 'entry_ordered'] = 0
        if self.df.loc[idx, 'strategy'] == 'LTD' and self.df.loc[idx, 'live/paper'] == 'paper':
            self.df.loc[idx, 'entry_ordered'] = 1
            self.df.loc[idx, 'entry_status'] = 'EXECUTION_COMPLETE'
            self.df.loc[idx, 'entry_price_avg'] = self.df.loc[idx, 'lay_price']
            self.df.loc[idx, 'entry_amount_matched'] = self.ltd_paper_stake_size
            self.df.loc[idx, 'entry_amount_remaining'] = 0
            self.df.loc[idx, 'current_order_side'] = side
            self.df.loc[idx, 'current_order_status'] = 'EXECUTION_COMPLETE'
            self.df.loc[idx, 'current_order_betid'] = 0

            self.adjust_paper_account(amount=self.ltd_paper_stake_size * (self.df.loc[idx, 'lay_price'] - 1), adjustment='decrease')

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

    def remove_finished_matches_autotrader(self, idx):
        while True:
            if self.is_database_connected():
                print('test remove')

                # Remove finished matches
                cur = self.cnx.cursor()
                cur.execute("DELETE FROM autotrader_matches_v3 WHERE inplay_state = ?", ('Finished',))
                self.cnx.commit()
                self.close_connection_db()
                self.df.drop(index=idx, inplace=True)  # Remove finished match from df
                break
            else:
                self.connect_autotrader_db()

    def archive_autotrader_match(self, idx):
        while True:
            if self.is_database_connected():

                # Archive match.
                df_temp = pd.DataFrame(self.df.loc[idx, self.df.columns.to_list()]).T.apply(lambda x: x.astype(str))
                df_temp.to_sql(name='archive_v2', con=self.cnx,
                               if_exists='append', index=False,
                               dtype=self.col_dtypes)  # Change if_exists to append after initial submit

                # Remove any duplicates after submitting to archive
                # self.drop_duplicates('autotrader_archive')
                break
            else:
                self.connect_autotrader_db()

    def drop_duplicates(self, table):
        while True:
            if self.is_database_connected():
                df = pd.read_sql_query(f"SELECT * from {table}", self.cnx, dtype=self.col_dtypes)
                df.drop_duplicates(ignore_index=True, subset=['event_name', 'start_date'])
                df.to_sql(name=table, con=self.cnx, if_exists='replace', index=False,
                          dtype=self.col_dtypes)
                break
            else:
                self.connect_autotrader_db()

    def continuos_match_finder(self, activate='off'):
        
            # This is the Continuos MAtchFinder feature. When this is activated it will stop the AutoTrader tool, run the MatchFinder tool then restart the AutoTrader.
            # self.df['marketStartTime'] = pd.to_datetime(self.df['marketStartTime'])
            # latest_kickoff  = self.df['marketStartTime'].max()
            if activate == 'on':
                # Check if time is half past or on the hour
                if pd.Timestamp.now().minute == 0 or pd.Timestamp.now().minute == 30:
                    print('*'*200)
                    print(pd.Timestamp.now())
                    print('*'*200)
                    # if datetime.now(timezone.utc) > latest_kickoff:
                    self.stop_autotrader()
                    
                    mf = MatchFinder(continuous=activate)
                    mf.get_betfair_details()
                    mf.get_sports_iq_stats()
                    df = mf.merge_data()
                    if len(df) > 0:
                        mf.add_matches_to_db()
                        self.initialise_data()
                        self.assign_strategy()
                    self.run_autotrader(continuous=activate)

    def adjust_paper_account(self, amount, adjustment):
        if not self.is_database_connected():
            self.connect_autotrader_db()
        paper_account_df = pd.read_sql_query("SELECT * from paper_account", self.cnx)
        print(paper_account_df, amount, adjustment)
        if adjustment == 'decrease':
            print('test', 'decrease')
            paper_account_df['balance'] = paper_account_df['balance'].astype(int) - amount
        if adjustment == 'increase':
            print('test', 'increase')
            paper_account_df['balance'] = paper_account_df['balance'].astype(int) + amount
        print(paper_account_df)
        paper_account_df.to_sql(name='paper_account', con=self.cnx, if_exists='replace', index=False)

    def check_paper_bet_result(self, idx):
        if self.df.loc[idx, 'strategy'] == 'LTD' and self.df.loc[idx, 'live/paper'] == 'paper':
            if int(self.df.loc[idx, 'entry_ordered']) == 1:
                if self.df.loc[idx, 'home_score'] != self.df.loc[idx, 'away_score']:
                    paper_profit = (self.ltd_paper_stake_size * (self.df.loc[idx, 'entry_price_avg'] - 1)) + (self.ltd_paper_stake_size - (self.ltd_paper_stake_size * 0.02))
                    self.adjust_paper_account(amount=paper_profit , adjustment='increase')

    def offset_tick_to_lay(self, cur_price, idx):
        offset_lay_price = cur_price
        if cur_price > 2 and cur_price <= 3:  # 0.02 increments
            offset_lay_price = cur_price - 0.12
        if cur_price > 3 and cur_price <= 3.30:
            offset_lay_price = 2.9
        if cur_price > 3.30 and cur_price <= 4:  # 0.05 increments
            offset_lay_price = cur_price - 0.30  
        if cur_price > 4 and cur_price <= 4.5:  # 0.1 increments
            offset_lay_price = cur_price - 0.6
        if cur_price > 4.5 and cur_price <= 5:
            offset_lay_price = 3.5
        if float(self.df.loc[idx, 'back_price']) > 5 or float(self.df.loc[idx, 'lay_price']) > 5:
            offset_lay_price = 4.0
        print('Offset Lay Price: ', round(offset_lay_price, 2))
        return round(offset_lay_price, 2)
    
    def strategy_ltd(self, idx):
        if self.df.loc[idx, 'strategy'] == 'LTD' and int(self.df.loc[idx, 'entry_ordered']) == 0:
            self.df['marketStartTime'] = pd.to_datetime(self.df['marketStartTime'])

            # Enter a lay order 10 minutes out from kick off
            if datetime.now(timezone.utc) > self.df.loc[idx, 'marketStartTime'] - timedelta(minutes=10):
                # Place lay order 1 tick below current price
                offset_lay_price = self.offset_tick_to_lay(self.df.loc[idx, 'lay_price'], idx)
                self.place_lay_order(size=self.ltd_live_stake_size, 
                                     price=offset_lay_price, 
                                     ptype='PERSIST',
                                     idx=idx,
                                     side='LAY')
                print(f"ENTRY ORDER PLACED for LTD: {self.df.loc[idx, 'event_name']}")
    


if __name__ == '__main__':
    pass