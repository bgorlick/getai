# getai/__main__.py
import asyncio
from getai.cli import cli_main


def run():
    asyncio.run(cli_main())


if __name__ == "__main__":
    run()
