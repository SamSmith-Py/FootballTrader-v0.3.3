"""
Simple helper if you want a dedicated runner module.
AutoTrader currently calls MatchFinder directly, so this is optional.
Keeping this file here for future scheduler centralisation.
"""

from match_finder import MatchFinder
from core.settings import BETFAIR_HOURS_LOOKAHEAD

def run_once():
    mf = MatchFinder(hours=BETFAIR_HOURS_LOOKAHEAD, continuous='on')
    mf.run()
