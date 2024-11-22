# Recurrent Convolutional models for Radar Sounder Data
### Author: Jordy Dal Corso, @ RSLab University of Trento, Trento, Italy

This repository contains the codebase to perform supervised and weakly-supervised training with custom convolutional recurrent models
on the task of radargram segmentation.
Among all the scripts, we have:
* Custom implementations of convolutional recurrent blocks (which may be useful to users also outside the radargram segmentation field)
* Custom implementation of vertical non-local operations (i.e. possibly self-attention)
* Custom modifications of U-Net architectures

We suggest users to build the Docker container provided in the `Dockerfile` and work within it with the codebase. Otherwise, one could simply
pick codes from `/src` for models and blocks, and from `/scripts` for train/test and experiments (experiments are in `.sh` files).

One should launch

    bash launch_docker.sh <chosen_container_name> <chosen_container_tag>

to create a Docker image to train, test and run experiments with this codebase.

## Supervised training
Users can choose the supervised model to train using the keyword `model` in the `train.yaml` file. One can choose between:
* `u`, U-Net
* `nlu`, U-Net with non-local operations instead of the second convolution within U-Net DoubleConv blocks.
* `ur`, U-Net with ConvLSTM layer at the bottleneck
* `nlur`, U-Net with ConvLSTM layer at the bottleneck and non-local operations as `nlu` (**Best model** among the others)
* `aspp`, U-Net with ASPP bottleneck

After being attached to the container, modify the `train.yaml` file with the desired configuration and, to **train** a model, run:

    python scripts/train.py -c scripts/config_files/train.yaml

After training, the latest model weights are saved into `log/latest.pt`. In order to test the saved model, modify
the `test.py` file to match the training yaml `train.yaml`, then run:

    python scripts/train.py -c scripts/config_files/train.yaml

## Weakly-supervised training
The weakly-supervised training scripts are:
* `cycle.py`
* `cycle_mod.py`

In order to perform weakly-supervised training, modify the correspondent yaml file in the config folder and run:

    python scripts/cycle.py -c scripts/config_files/cycle.yaml

or:

    python scripts/cycle.py -c scripts/config_files/cycle.yaml

And to test the trained model, as above, run:

    python scripts/test_weak.py -c scripts/config_files/test_weak.yaml

The latest training step overwrite the file `latest.pt` irregardless of whether the training is supervised or weakly-supervised.

## Notes

* Be sure to check config files before launching any `.py` script from `/scripts/`, especially the paths to folders.
* Be sure to download a dataset and format it such that the `dataset.py` class can process it.
* When using `nlur` and choosing a `patch_len` of 1, a custom model is created, which operates on sequences of
rangelines and employs (fast) 1D convolutions.

For inquiries, comments and in case you need a radargram dataset to try the code, contact jordy.dalcorso@unitn.it