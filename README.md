Effect of data balance in Machine Learning
==========================================

#### Manifesto

Because *Human* is **perfectible** and **error-prone**, because *Science* should be **open** and **flow** and because *cogito ergo sum*.

## Research target output

This study targets ACPR2015 to be scientifically disseminated

## Datasets

### Some thoughts:

- [List of highly imbalanced data sets](http://www.cs.gsu.edu/~zding/research/benchmark-data.php)
- [UCI dataset](http://archive.ics.uci.edu/ml/datasets.html?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=attup&view=list) described as imalanced (137)
- Scikit-learn has some [dataset loading utilities](http://scikit-learn.org/stable/datasets/) which also refair to [svmlight-loader](https://github.com/mblondel/svmlight-loader) as example.
- [SciKit-data](https://github.com/jaberg/skdata) is a package for loading/creating datasets for scikit-learn

### Datasets susceptible to be used in this study

| Dataset    | Size  | imbalance | liu2008 | UCI | stalog | svmlight-loader | skdata |
|------------|-------|:---------:|:-------:|:---:|:------:|:---------------:|:------:|
| abalone    | 4177  | 1: 9.7    | x       | x   |        | x               |        |
| balance    | 625   | 1: 11.8   | x       |     |        |                 |        |
| car        | 1728  | 1: 3.5    | x       |     |        |                 |        |
| cmc        | 1473  | 1: 3.4    | x       |     |        |                 |        |
| haberman   | 306   | 1: 2.8    | x       |     |        |                 |        |
| housing    | 506   | 1: 3.8    | x       | x   |        | x               |        |
| ionosphere | 351   | 1: 1.8    | x       | x   |        | x               |        |
| letter     | 20000 | 1: 24.3   | x       |     |        |                 |        |
| mf-morph   | 2000  | 1: 9.0    | x       |     |        |                 |        |
| mf-zernike | 2000  | 1: 9.0    | x       |     |        |                 |        |
| phoneme    | 5404  | 1: 2.4    | x       |     |        |                 |        |
| pima       | 768   | 1: 1.9    | x       | x   |        | x               |        |
| satimage   | 6435  | 1: 9.3    | x       |     | x      | x               |        |
| vehicle    | 846   | 1: 3.0    | x       |     | x      | x               |        |
| wdbc       | 569   | 1: 1.7    | x       |     |        |                 |        |
| wpbc       | 198   | 1: 3.2    | x       |     |        |                 |        |


Project folder structure
------------------------

This project has been structured using this [rr-init repository] as a template.
In order to keep an upstream to such project, add it as remote like this:

```
git remote add rr-initUPS git@github.com:massich/rr-init.git
```

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
    |- results               # all output from workflows and analyses
    |  |- figures/           # graphs, likely designated for manuscript figures
    |  +- pictures/          # diagrams, images, and other non-graph graphics
    |
    |- scratch/              # temporary files that can be safely deleted or lost
    |- src/                  # any programmatic code
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

Use the following to create a `diabetes` environment based on the `./requirements.txt` associated with the source directory `./src`:

```
mkvirtualenv diabetes -a src -r ../requirements.txt
```

Notice that `mkvirtualenv` activates such environment.
The command `deactivate` is used to exit the virtual environment.
Once the virtual environment exist on the system, the command `workon diabetes` is rather convenient since it jumps into the working directory and activates the virtual enviroment.

**Remember** to keep `requirements.txt` up to date.
For more details regarding the usage of the virtual enviroment, please look at the [command reference].

Todo
----

- [x] Add virtual-env behaviour (at /src)
- [?] Add sphinx documentation as project.io website
- [x] Add paper submodule
  - [ ] apply ACPR2015 template
- [ ] Add code modules
- [ ]


[rr-init repository]: https://github.com/massich/rr-init

[virtual environment post]: http://www.silverwareconsulting.com/index.cfm/2012/7/24/Getting-Started-with-virtualenv-and-virtualenvwrapper-in-Python
[mozilla marketplace testing]: https://github.com/mozilla/marketplace-tests
[command reference]:http://virtualenvwrapper.readthedocs.org/en/latest/command_ref.html
