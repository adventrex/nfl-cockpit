"""notify.py — the one place that texts Andrew (scan_injuries, autopicker, anything else)."""
import os, subprocess, tomllib

_SECRETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml")


def phone() -> str:
    """Andrew's iMessage number from secrets.toml (ANDREW_PHONE) or the env. Not in the repo: it is public."""
    try:
        with open(_SECRETS, "rb") as f:
            return tomllib.load(f).get("ANDREW_PHONE") or os.environ["ANDREW_PHONE"]
    except (FileNotFoundError, KeyError):
        raise RuntimeError("ANDREW_PHONE missing: add it to .streamlit/secrets.toml")


def imessage(text: str, to: str = None) -> None:
    to = to or phone()
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "Messages" to send "{safe}" to buddy "{to}" of (service 1 whose service type is iMessage)'
    subprocess.run(["osascript", "-e", script], check=False, timeout=30)
