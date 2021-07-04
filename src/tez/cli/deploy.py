from argparse import ArgumentParser

from . import BaseTezCommand


def _command_factory(args):
    return DeployCommand(args.config)


class DeployCommand(BaseTezCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("deploy", description="the deploy command")
        _parser.add_argument(
            "--config", type=str, default=None, required=True, help="config file to be used for deployment"
        )
        _parser.set_defaults(func=_command_factory)

    def __init__(self, config: str):
        self._config = config

    def run(self):
        from ..tez import Tez

        _tez = Tez(self._config)
        _tez.deploy()
