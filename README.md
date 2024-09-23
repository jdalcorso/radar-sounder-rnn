# README #

This repository contains the template for a python project (pyproject). The code is expected to be structured into modules/packages and scripts/notebooks. The modules/packages should be put in the src folder and are installed with the editable installation of pip, which allows to use the modules wherever you are as long as the repo is installed in this way in the current python environment (see `pyproject.toml`). Scripts and notebooks should be stored in dedicated folders. The whole thing is then supposed to be run into a docker container, allowing to isolate the development environment and to have more flexibility. Therefore, edit the `Dockerfile` as you wish.

## Table of Contents ##
1. [How do I get set up?](#markdown-header-how-do-i-get-set-up)
1. [How is this repo organized?](#markdown-header-how-is-this-repo-organized)
    * [Example of project folder structure](#markdown-header-example-of-project-folder-structure)
1. [How to run the scripts?](#markdown-header-how-to-run-the-scripts)


## How do I get set up? ##

1. Copy the template file in your empty repository.
1. Edit the `Dockerfile` to satisfy system-wide requirements (_e.g._, apt packages) and with correct starting Docker image (the `FROM` directive)
1. Edit the `pyproject.toml` with the correct python version and with requirements. Here you can add some metadata info about the project.
1. To run the development docker container follow these steps:
    1. Enter the repo folder and execute the command `bash launch_docker.sh <custom-image-name> <custom-tag-name>` (_e.g._, `bash launch_docker.sh cool-transformer v1.0.0`)
        * This will build and run a docker container with all the required dependencies installed.
        * Your user account will be mapped to the user `containeruser` inside the container.
        * Your `/home/<yourname>` folder will be mapped to `/home/containeruser`.
        * The path `/media` will also be mounted inside the container.
        * `NOTE`: By default, the container will run in the background with bash shell. This is set up as a Dev Container, _i.e._, a container that you can attach to using _e.g._ VSCode to develop, debug and run the code. Any change to the code inside the container will be reflected outside, and it is also possible to use git with any remote.
    1. Attach to the container using `docker attach <custom-image-name>_<user-name>` (or using _e.g._ VSCode with Dev Container extension) and navigate to the repo location.

## How is this repo organized? ##

The repository is organized as a Python project with src layout, and it is used as an editable installation. What does it mean? The packages and modules inside the `src` folder will be accessible everywhere as they were installed with pip. The repository is structured as follows:

* `notebooks` folder: it contains all the python notebooks
* `scripts` folder: it contains all the runnable bash and python scripts
    * `config_files` folder: it contains the default YAML configuration files with the parameters used to run the scripts.
* `src` folder: it contains the developed custom packages and modules used by the python scripts.
* `Dockerfile`: used to create the docker container.
* `entrypoint.sh`: the first script run by the container upon creation, it installs the repository as a pip editable installation.
* `launch_docker.sh`: it builds the docker image and run the dev container.
* `README.md`: this file.


### Example of project folder structure ###

An example of project tree structure:
```
.vscode
└── launch.json
Dockerfile
entrypoint.sh
launch_docker.sh
notebooks
├── HRLC-sar-data-availability-PALSAR.ipynb
└── HRLC-sar-data-availability-Sentinel-1.ipynb
pyproject.toml
README.md
scripts
├── config_filess
│   ├── ...
│   ├── step2_retrieve_products.yaml
│   └── step3_meta_analysis.yaml
├── step1_create_access_key.sh
├── step2_retrieve_products.py
└── step3_meta_analysis.py
src
├── data_availability
│   ├── _data_availability.py
│   ├── __init__.py
│   └── _meta_analysis.py
├── google_session.py
└── scripting
    ├── __init__.py
    └── _scripting.py
```

## How to run the scripts? ##
1. Edit the YAML configuration files in `scripts/config_files` as needed specifying the desired parameters and paths to the folders where the data is stored.
2. Enter the `scripts` folder
3. Run a python script as follows: `python <script-basename>.py -c config_files/<script-basename>.yaml`.

If you follow this template, also the debugging becomes much easier. The file `vscode/launch.json` contains the configuration for running a script with the correct configuration file automatically by the press of a button in Visual Studio Code. Just open the script in the editor and press `F5` to debug it with the corresponding configuration file.