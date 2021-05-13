import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser

from loguru import logger


logger.configure(
    handlers=[
        dict(
            sink=sys.stderr,
            format="§ <level>tez: {time:YYYY-MM-DD HH:mm:ss} {level:<7} </level> <cyan>| {message}</cyan>",
        )
    ]
)


class BaseTezCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
