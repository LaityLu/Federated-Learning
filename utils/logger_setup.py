import logging


def setup_logger(log_file=None, name='main_logger'):
    """create and set shared logger"""
    logger = logging.getLogger(name)

    # set the minimum recording level
    logger.setLevel(logging.DEBUG)

    # make sure that main_fed can set the saving path of log
    if log_file is None:
        return logger

    # prevent duplicate addition of handlers
    if logger.handlers:
        return logger

    # create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.DEBUG)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # set logger formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # add handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
