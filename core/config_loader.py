from configparser import ConfigParser
from pathlib import Path
from typing import Tuple

def load_betfair_credentials(config_path: Path) -> Tuple[str, str, str]:
    """
    Reads username, password, app_key from config.ini
    [betfair]
    username=...
    password=...
    app_key=...
    """
    parser = ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(f"Config missing or unreadable: {config_path}")
    if "betfair" not in parser:
        raise KeyError("Section [betfair] not found in config.ini")

    section = parser["betfair"]
    username = section.get("username")
    password = section.get("password")
    app_key = section.get("app_key")
    if not all([username, password, app_key]):
        raise ValueError("Missing one or more Betfair credentials (username/password/app_key).")
    return username, password, app_key
