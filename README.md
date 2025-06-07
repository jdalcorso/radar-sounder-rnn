# Recurrent Convolutional models for Radar Sounder Data
### Author: Jordy Dal Corso, @ RSLab University of Trento, Trento, Italy

This repository contains the codebase to perform supervised and weakly-supervised training with custom convolutional recurrent models
on the task of radargram segmentation.
Among all the scripts, we have:
* Custom implementations of convolutional recurrent blocks (which may be useful to users also outside the radargram segmentation field)
* Custom implementation of vertical non-local operations (i.e. possibly self-attention)
* Custom modifications of U-Net architectures

We suggest users to build the Docker container provided in the `Dockerfile` and work within it with the codebase. 
One should launch:

    bash launch_docker.sh <chosen_container_name> <chosen_container_tag>

to create a Docker image to train, test and run experiments with this codebase. In this way, not only a container is created, but also the package is installed within it 
with all the correct dependencies.

Another way to run the code is to install it as a pyproject, i.e. via pip by entering the project folder and run:

    pip install .

This has not been tested outside the Docker container but should work if a compatible Python version is used, and if the machine has a compatible cuda version.
Otherwise, one could simply pick codes from `/src` for models and blocks, and from `/scripts` for train/test and experiments (multi-seed run and averaging).

## Supervised training
Users can choose the supervised model to train using the keyword `model` in the `train.yaml` file. One can choose between:
* `u`, U-Net
* `nlu`, U-Net with non-local operations instead of the second convolution within U-Net DoubleConv blocks.
* `ur`, U-Net with ConvLSTM layer at the bottleneck
* `nlur`, U-Net with ConvLSTM layer at the bottleneck and non-local operations as `nlu` (**Best model** among the others)
* `aspp`, U-Net with ASPP bottleneck

After being attached to the container, modify the `train.yaml` file with the desired configuration and, to **train** a model, run:

    python scripts/train.py -c scripts/config_files/train.yaml

After training, the latest model weights are saved into `log/epoch_xxxxx.pt` where `xxxxx` is the validation loss without decimal point. In order to test the saved model, run:

    python scripts/test.py -c scripts/config_files/test.yaml

This will pick the best model among the saved epochs (by looking at the validation loss in the filename), remove all the others, rename it as `best.pt` an run the test with it.
Notice that running `test.py` deletes all the checkpoints except the best one (in terms of validation loss).

## Weakly-supervised training
The weakly-supervised training scripts are:
* `cycle.py`
* `cycle_mod.py`
* `cycle_double.py`

In order to perform weakly-supervised training, modify the correspondent yaml file in the config folder and run:

    python scripts/cycle.py -c scripts/config_files/cycle.yaml

And to test the trained model, as above, run:

    python scripts/test_weak.py -c scripts/config_files/test_weak.yaml

Again, the latest training step overwrite the files named `epoch_xxxxx.pt` irregardless of whether the training is supervised or weakly-supervised
Also in this case, when testing, the file `best.pt` is created and all the non-best checkpoints are deleted from the log folder.

## Notes

* Be sure to check config files before launching any `.py` script from `/scripts/`, especially the paths to folders.
* Be sure to download a dataset and format it such that the `dataset.py` class can process it.
* When using `nlur` and choosing a `patch_len` of 1, a custom model is created, which operates on sequences of
rangelines and employs (fast) 1D convolutions.

For inquiries, comments and in case you need a radargram dataset to try the code, contact jordy.dalcorso@unitn.it