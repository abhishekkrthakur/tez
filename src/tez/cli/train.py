import sys
from argparse import ArgumentParser

from loguru import logger

from . import BaseTezCommand


def _command_factory(args):
    return TrainCommand(args.config)


class TrainCommand(BaseTezCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("train", description="the training command")
        _parser.add_argument(
            "--config", type=str, default=None, required=True, help="config file to be used for training"
        )
        _parser.set_defaults(func=_command_factory)

    def __init__(self, config: str):
        self._config = config

    def run(self):
        from ..tez import Tez

        _tez = Tez(self._config)
        _tez.start()
