import logging
import sys


def setup_logger():
    # # Set up logging configurations
    # logger = logging.getLogger(__name__)

    # if not logger.handlers:  # Check if handlers are already added
    #     logger.setLevel(logging.INFO)

    #     unbuffered_stdout = open(
    #         sys.stdout.fileno(), "w", 1, closefd=False
    #     )  # setting line buffering
    #     stdout_handler = logging.StreamHandler(unbuffered_stdout)
    #     stdout_handler.setLevel(logging.INFO)
    #     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    #     stdout_handler.setFormatter(formatter)

    #     logger.addHandler(stdout_handler)

    # Notebook debug
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.debug("test")

    return logger