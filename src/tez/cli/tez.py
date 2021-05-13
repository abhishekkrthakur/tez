import argparse

from .. import __version__
from .train import TrainCommand


def main():
    parser = argparse.ArgumentParser("tez CLI", usage="tez <command> [<args>]")
    parser.add_argument("--version", "-v", help="Display tez version", action="store_true")
    commands_parser = parser.add_subparsers(help="commands")

    TrainCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
