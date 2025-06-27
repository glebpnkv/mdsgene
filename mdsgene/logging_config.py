# logging_config.py
import logging
import sys


def configure_logging(level: int = logging.INFO):
    """Call this once, anywhere, to set up the root logger."""
    if not logging.root.handlers:            # guard so basicConfig only runs once
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stderr
        )
