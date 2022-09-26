# *Brassica napus* image analysis

Segmentation of images of *Brassica napus* plants in the field using a deep-learning network based on *densenet*:
- "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>

## Installation / Upgrading

### Installation

__Warning__: When installing the GPU enabled pytorch version, over a gigabyte of disk space might be used.
If your device doesn't have a GPU, installing the CPU only version will save a lot of disk space.

__Note__: Below are instructions to install the `hooloovoo` package as a regular
package and this package in development mode.
The advantage of development mode is that all the files of the repository are
accessible and easy to edit.
You can install both packages in development mode using `pip install -e`.
   
#### Conda (recommended)
1) Create / use a conda python 3 environment with at least pip version 19:

   ```bash
   conda create -n $MYENV python=3.7
   source activate $MYENV
   ```
   
   __Note__: pip can upgrade itself in the current env:

   ```bash
   pip install --upgrade pip
   ```
   
2) Install the correct pytorch version for your system, see the instructions here: <https://pytorch.org/get-started/locally/>.

3) Install all conda requirements:

   ```bash
   conda install cytoolz future matplotlib numpy pandas pillow pyyaml scikit-image scipy tensorboard tensorflow
   ```
   
4) Install this package with pip:
   ```bash
   # The hooloovoo dependency
   pip install 'git+https://github.com/MMichaelVdV/hooloovoo.git' or pip install git+https://github.com/MMichaelVdV/hooloovoo.git (Windows)
   
   # This repo
   git clone 'git+https://github.com/MMichaelVdV/Brassica_segmentation.git' or git clone https://github.com/MMichaelVdV/Brassica_segmentation.git (Windows)
   cd Brassica_segmentation
   pip install -e .
   ```
   
   After installing a package with pip in conda (if not using `-e`), package files can be found here:

   ```
   $HOME/.conda/envs/$MYENV/lib/python3.$X/site-packages/$SOME_PACKAGE
   ```

   Where `$X` is the python version of `$MYENV`.

#### Pip

1) (optional) Create a venv for pip. You will at need at least pip version 19.

   __Note__: pip can upgrade itself in the current env:

   ```bash
   pip install --upgrade pip
   ```

2) Install the correct pytorch version for your system, see the instructions here: <https://pytorch.org/get-started/locally/>.

3) Install this package with pip:

   ```bash
   # The hooloovoo dependency
   pip install 'git+https://github.com/MMichaelVdV/hooloovoo.git'
   
   # This repo
   git clone 'git+https://github.com/MMichaelVdV/Brassica_segmentation.git'
   cd Brassica_segmentation
   pip install -e .
   ```

### Upgrading

```bash
pip install --upgrade 'git+https://github.com/MMichaelVdV/hooloovoo.git'

cd Brassica_segmentation
git pull
pip install -e .
```

## Usage

Once installed, try the following:

```bash
Brassica_segmentation -h
```

If you get a help message, the installation worked.

The `cnn_segmentation` command takes an application and a settings file as argument.
Example settings files can be found here:

* <https://github.com/MMichaelVdV/Brassica_segmentation/blob/main/cnn_segmentation/applications/brassomics/settings/settings_devel.yaml>

Settings file can be either in `yaml` or `json` format.

### Running on SGE:

Below is a template of a qsub script to run the program on an SGE cluster such as midas.
This assumes you have installed the package in a conda env called `$MYENV`.

```
#!/bin/bash
#$ -l h_vmem=24G
#$ -l h_stack=24M

module load anaconda
source activate $MYENV
Brassica_segmentation $APP $SETTINGS_FILE
```

Where `$SETTINGS_FILE` contains the path to the `.json` or `.yaml` settings file,
and `$APP` is `brassomics`.

### Tensorboard

If you are training a model, you can follow the training progress remotely by starting a tensorboard server.
Inside the python environment, issue the following command:

```bash
tensorboard --logdir=$LOGDIR/tensorboard
```

Where `$LOGDIR` is the logging path declared in the settings file.

__Warning__: Tensorboard is not real-time, usually it lags a few minutes behind.
As such it is possible that not all metrics are visible right after starting training.

__Note__: Tensorboard starts scanning the most recent files first, as such, to
see the most recent results faster, restart tensorboard and refresh the page.
