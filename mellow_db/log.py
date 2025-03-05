import logging
import os

from dotenv import load_dotenv

load_dotenv(override=True)


class ServerLogFormatter(logging.Formatter):
    def formatTime(self, record, datefmt="%m-%d,%H:%M"):
        return super().formatTime(record, datefmt)

    def format(self, record):
        msg = record.msg  # Expecting record.msg to be a dict
        log_message = (
            f"{self.formatTime(record)} "
            f"{record.name}:{record.funcName}:{msg.get('step', 'N/A')} "
            f"{msg.get('client_id', 'N/A')}:{msg.get('collection', 'N/A')} "
            f"{', '.join([f'{k}: {v}' for k, v in msg.get('extra', {}).items()])}"
        )
        return log_message


def setup_server_logging():
    """
    Configures server-side logging for the application.

    This function sets up logging to both a file and the console. The file logs are written
    to `mellow_db.log` in append mode, while console logs use a custom formatter that
    processes log records as dictionaries. The log levels for the console are determined by
    the `MELLOW_SERVER_LOG_LEVEL` environment variable, defaulting to `INFO` if not set.
    """

    # Set up the file handler for logging to a file
    logging.basicConfig(
        level=logging.DEBUG,  # Default level for file logging
        format=('%(asctime)s\t%(levelname)s\t%(threadName)s:%(process)d\t%(filename)s:%(lineno)d\t'
                '%(name)s:%(funcName)s %(message)s'),
        filename='mellow_db.log',
        filemode='a',
    )

    # Set up the console handler with a custom formatter
    stdout_handler = logging.StreamHandler()
    log_level = os.getenv('MELLOW_SERVER_LOG_LEVEL', 'INFO').upper()
    stdout_handler.setLevel(getattr(logging, log_level, logging.DEBUG))
    stdout_handler.setFormatter(ServerLogFormatter())

    # Add the console handler to the root logger
    logging.getLogger().addHandler(stdout_handler)


def setup_client_logging():
    """
    Configures client-side logging for the application.

    This function sets up logging to the console with a basic formatter. The log levels for
    the console are determined by the `MELLOW_CLIENT_LOG_LEVEL` environment variable,
    defaulting to `WARNING` if not set.
    """

    stdout_handler = logging.StreamHandler()

    # Define the logging format
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    stdout_handler.setFormatter(formatter)

    # Get the root logger and set its level
    logger = logging.getLogger()
    log_level = os.getenv('MELLOW_CLIENT_LOG_LEVEL', 'WARNING').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.addHandler(stdout_handler)
