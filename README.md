Effect of data balance in Machine Learning
==========================================

#### Manifesto

Because *Human* is **perfectible** and **error-prone**, because *Science* should be **open** and **flow** and because *cogito ergo sum*.

## Research target output

1. ACPR 2015 - Apply the different techniques for melanoma detection with imbalanced dataset.
1. CVIU - Comparison of the different techniques on some benchmark datasets.

# Package requirements

In order to execute the scripts and pipelines designed for the experiments, you need to install the following python package:

* [protoclass](https://github.com/glemaitre/protoclass),
* [UnbalancedDataset](https://github.com/fmfn/UnbalancedDataset).

These packages can be found as submodule in this repository.

## Dataset

The dataset can be found in the following [link](http://grid.cs.gsu.edu/~zding/research/benchmark-data.php).
In our experiment, refer to the file `data/README.md` in order to fetch the data and be able to reproduce the experiments.

Project folder structure
------------------------

### Structure Description
```
    project
    |- doc/                  # documentation for the study
    |  |- paper/             # manuscript(s), whether generated or not
    |  +- source/            # sphinx source
    |
    |- data                  # raw and primary data, are not changed once created
    |  |- raw/               # raw data, will not be altered
    |  +- clean/             # cleaned data, will not be altered once created
    |
    |- pipeline/             # The different pipeline used for the study
    |  +- feature-classification  # pipeline to perform the classification
    |
    |- results               # all output from workflows and analyses
    |  |- figures/           # graphs, likely designated for manuscript figures
    |  +- pictures/          # diagrams, images, and other non-graph graphics
    |
    |- scratch/              # temporary files that can be safely deleted or lost
    |
    |- script/               # scripts used to run on the cluster
    |
    |- src/                  # any programmatic code
    |
    |- notebook/	     # workflow notebook
    |
    |- Makefile              # executable Makefile for this study, if applicable
    |- datapackage.json      # metadata for the (input and output) data files
    |- requirements.txt      # list of the required packages (see virtualenv)
    |
    |- LICENSE.md
    |- README.md             # the top level description of content
```

### Recomendations

#### Use a virtual environment (Virtualenv + VirtualenvWrapper)

Virtual-environments are not **virtual machines**.
Virtual-environments are used to avoid library classing between the libraries of a project and those fom the system.
Find more information in this [virtual environment post] describing how to use virtual environment for a [mozilla marketplace testing].

Use the following to create a `data_balancing` environment based on the `./requirements.txt` associated with the source directory `./src`:

```
mkvirtualenv data_balancing -a src -r ../requirements.txt
```

Notice that `mkvirtualenv` activates such environment.
The command `deactivate` is used to exit the virtual environment.
Once the virtual environment exist on the system, the command `workon data_balancing` is rather convenient since it jumps into the working directory and activates the virtual enviroment.

**Remember** to keep `requirements.txt` up to date.
For more details regarding the usage of the virtual enviroment, please look at the [command reference].

Todo
----

- [ ]

