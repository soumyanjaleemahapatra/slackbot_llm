import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.info)

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

# Get the folder this file is in:
path_of_folder = os.path.dirname(os.path.realpath(__file__))
load_dotenv(Path(path_of_folder) / ".env")

from src.slackbot import slack_bot

async def start():
    await slack_bot.start()

if __name__ == "__main__":
    asyncio.run(start())