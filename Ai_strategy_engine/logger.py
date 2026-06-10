# logger.py
import logging

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG if needed
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),   # save logs to file
        logging.StreamHandler()           # show logs in terminal
    ]
)

logger = logging.getLogger()
