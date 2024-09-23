#!/usr/bin/env python3
"""
Example of python script

@author: Gianmarco Perantoni
"""
import logging
import typing

import scripting


def main(
    arg1: typing.Any,
    arg2: typing.Any,
    kwarg1: typing.Any | None = None,
    **kwargs,
):
    logger = logging.getLogger("main")
    logger.info(f"{arg1=}")
    print(f"{arg1=}")
    logger.info(f"{arg2=}")
    print(f"{arg2=}")
    if kwarg1 is not None:
        logger.info(f"{kwarg1=}")
        print(f"{kwarg1=}")


if __name__ == "__main__":
    scripting.logged_main(
        "Example of python script",
        main,
    )
