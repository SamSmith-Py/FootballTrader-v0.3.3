"""
AutoTrader v2 — strategy-only loop.
MatchFinder is now called ONLY by scheduler.py
"""

from __future__ import annotations
import time
import logging
from typing import List, Type
from datetime import datetime, timezone

from core.settings import (
    DB_PATH,
    TABLE_CURRENT,
    PAPER_MODE,
    BOT_VERSION,
)

from core.db_helper import DBHelper
from core.betfair_session import BetfairSession

# Strategy registry
from autotrader.strategies.base_strategy import BaseStrategy
from autotrader.strategies.ltd60 import LTD60

logger = logging.getLogger("autotrader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)


class AutoTrader:
    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        self._install_strategies([LTD60])
        logger.info("AutoTrader initialised. Paper=%s Bot=%s", PAPER_MODE, BOT_VERSION)

    def _install_strategies(self, strategy_types: List[Type[BaseStrategy]]):
        for cls in strategy_types:
            self.strategies.append(cls())

    def start(self):
        """Main trading loop."""
        logger.info("AutoTrader run started. Waiting for matches...")

        while True:
            with DBHelper(DB_PATH) as db:
                rows = db.fetch_all(
                    f"""
                    SELECT * FROM {TABLE_CURRENT}
                    ORDER BY datetime(kickoff) ASC
                    """
                )

                if not rows:
                    time.sleep(10)
                    continue

                needs_api = any(s.requires_api for s in self.strategies)
                api = None
                if needs_api:
                    try:
                        api = BetfairSession().connect()
                    except Exception as e:
                        logger.warning(f"Betfair session unavailable: {e}")

                for row in rows:
                    ev = dict(row)
                    for strat in self.strategies:
                        try:
                            strat.assign_if_applicable(db, ev)
                            strat.on_tick(db, ev, api=api)
                        except Exception as e:
                            logger.error("[%s] error on %s: %s", strat.name, ev.get("event_id"), e)

                if api:
                    try:
                        api.logout()
                    except:
                        pass

            time.sleep(5)
