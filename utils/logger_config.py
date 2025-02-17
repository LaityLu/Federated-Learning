import logging
from datetime import datetime

current_time = datetime.now()
formatted_time = current_time.strftime("%m-%d_%H-%M")

log_format = "%(message)s"
logging.basicConfig(level=logging.INFO, filename=f'./save/logs/{formatted_time}.log', format=log_format)
logger = logging.getLogger()
logger.info(f'save time:{formatted_time}')
