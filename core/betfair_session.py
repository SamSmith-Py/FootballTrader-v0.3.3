import logging
import betfairlightweight

logger = logging.getLogger(__name__)

class BetfairSession:
    """
    Simple context manager for a single API session per run.
    Uses interactive login. If you switch to cert-based login later,
    change __enter__/connect accordingly.
    """
    def __init__(self, username: str, password: str, app_key: str):
        self.username = username
        self.password = password
        self.app_key = app_key
        self.client = betfairlightweight.APIClient(self.username, self.password, self.app_key)

    def __enter__(self):
        logger.info("Opening Betfair API session...")
        self.client.login_interactive()
        logger.info("Betfair API session established.")
        return self.client

    def __exit__(self, exc_type, exc, tb):
        try:
            self.client.logout()
            logger.info("Betfair API session closed.")
        except Exception as e:
            logger.warning(f"Error during Betfair logout: {e}")

    # Optional: manual use without `with`
    def connect(self):
        self.client.login_interactive()
        return self.client
