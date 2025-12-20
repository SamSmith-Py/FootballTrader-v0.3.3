"""
run_bot.py — Main entrypoint for FootballTrader v0.3.3

Responsibilities:
  • Load config
  • Start MatchFinder scheduler in background
  • Run AutoTrader in main thread
  • Handle graceful shutdown (Ctrl+C)
"""

import os
import logging
import time
import threading

from core.config_loader import load_betfair_credentials
from core.settings import BOT_VERSION, SCHEDULE_MATCHFINDER_MIN
from autotrader.scheduler import start_scheduler
from autotrader.autotrader import AutoTrader

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("run_bot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)


# -----------------------------------------------------------------------------
# Guard to ensure scheduler only starts once
# -----------------------------------------------------------------------------
_scheduler_started = threading.Event()


def safe_start_scheduler():
    """Ensure scheduler starts only once globally."""
    if not _scheduler_started.is_set():
        start_scheduler()
        _scheduler_started.set()
    else:
        logger.warning("Scheduler already running, skipping duplicate start.")


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
def main():
    logger.info(f"Bot starting... (FootballTrader {BOT_VERSION})")
    cwd = os.getcwd()
    logger.info(f"CWD = {cwd}")

    # Config
    config_path = os.path.join(cwd, "config", "config.ini")
    logger.info(f"config.ini exists = {os.path.exists(config_path)}")
    if not os.path.exists(config_path):
        logger.error("Missing config.ini! Bot cannot start.")
        return

    # Load config (ensures credentials + constants available)
    load_betfair_credentials(config_path)

    # Start scheduler thread (MatchFinder every N minutes)
    safe_start_scheduler()
    # logger.info(f"Scheduler started: MatchFinder every {SCHEDULE_MATCHFINDER_MIN} minutes.")
    # Keep main thread alive indefinitely
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Bot stopped manually. Exiting...")

    # Start AutoTrader (main loop)
    """trader = AutoTrader()
    try:
        trader.start()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user (CTRL+C).")
    except Exception as e:
        logger.exception(f"❌ AutoTrader crashed: {e}")
    finally:
        logger.info("⚙️ FootballTrader shutting down gracefully...")"""


# -----------------------------------------------------------------------------
# Entrypoint guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
