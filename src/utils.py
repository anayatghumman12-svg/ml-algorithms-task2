import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_folders_exist(paths: list) -> None:
    """
    Diye gaye folder paths agar exist nahi karte to bana deta hai.
    (Taake save karte waqt "folder not found" error na aaye.)

    Args:
        paths (list): folder paths ki list.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Checked/created folder: {path}")
