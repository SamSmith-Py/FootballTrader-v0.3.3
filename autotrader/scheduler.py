import logging
import threading
import time
from core.settings import SCHEDULE_MATCHFINDER_MIN
from match_finder import MatchFinder
import os

logger = logging.getLogger("scheduler")

def _run_matchfinder_job():
    try:
        logger.info("MatchFinder scheduled run starting...")
        mf = MatchFinder()
        updated = mf.run()   # should return inserted/updated row count later
        logger.info(f"MatchFinder completed. ({updated} rows refreshed)")
    except Exception as e:
        logger.warning(f"MatchFinder failed: {e}")

def start_scheduler():
    """
    Runs MatchFinder immediately, then every interval.
    """
    def loop():
        print("CWD =", os.getcwd())
        print("config.ini exists =", os.path.exists("config//config.ini"))
        _run_matchfinder_job()  # immediate first run
        while True:
            time.sleep(SCHEDULE_MATCHFINDER_MIN * 60)
            _run_matchfinder_job()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    logger.info(f"Scheduler started: MatchFinder every {SCHEDULE_MATCHFINDER_MIN} minutes.")
