import logging
import logging.handlers
import os

# Third-party libraries log a lot of noise at DEBUG (HTTP request/response
# bodies, connection pool internals) that would drown out our own log lines.
NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3")


def setup_logging(logs_dir: str, level: str):
    """Configure logging once, for every module that does logging.getLogger(__name__).

    Takes its settings as plain arguments rather than importing config.py —
    config.py is what calls this, so importing it back here would be circular.

    Two handlers: console (so uvicorn's terminal shows live activity) and a
    rotating file (so history survives after the terminal scrolls away or
    the process restarts).
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "app.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Rotates once a file hits 1MB, keeping 3 old copies — bounded disk use,
    # recent history still survives a restart.
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for noisy_logger in NOISY_LOGGERS:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
