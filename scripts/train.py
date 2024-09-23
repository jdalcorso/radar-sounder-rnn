#!/usr/bin/env python3
"""
Training script

@author: Jordy Dal Corso
"""
import logging

import scripting


def main(
    input_size,
    hidden_size,
    seq_len,
    epochs,
    batch_size,
    lr,
    out_dir,
    **kwargs,
):
    logger = logging.getLogger("Train")
    


if __name__ == "__main__":
    scripting.logged_main(
        "Training",
        main,
    )
