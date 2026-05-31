"""Neural surrogate POC for discrete-event supply-chain simulation."""

__version__ = "0.1.0"


def main() -> None:
    from neural_simulator.cli import main as cli_main

    cli_main()
