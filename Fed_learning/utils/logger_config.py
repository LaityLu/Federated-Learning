import logging

log_format = "%(message)s"
logging.basicConfig(level=logging.INFO, filename='./save/logs/my.log', format=log_format)
logger = logging.getLogger()
