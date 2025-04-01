import logging


def pytest_configure(config):
    # Enable debug logging for the projec itself,
    # so that in case of errors during tests we have
    # additional debug information.
    specific_logger = logging.getLogger("mustela")
    specific_logger.setLevel(logging.DEBUG)
