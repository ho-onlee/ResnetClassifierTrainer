import logging
import logging.config

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        filename="app.log",
    )

