Effect of data balance in Machine Learning
==========================================

#### Manifesto

Because *Human* is **perfectible** and **error-prone**, because *Science* should be **open** and **flow** and because *cogito ergo sum*.

## Research target output

This study targets ACPR2015 to be scientifically disseminated

## Datasets
Datasets would be taken from [mldata.org]
[mldata.org]:mldata.org

### Some thoughts:

- checkout [mldata.org]. I’ve troubles to connect
- [List of highly imbalanced data sets](http://www.cs.gsu.edu/~zding/research/benchmark-data.php)
- [UCI dataset](http://archive.ics.uci.edu/ml/datasets.html?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=attup&view=list) described as imalanced (137)
- Scikit-learn has some [dataset loading utilities](http://scikit-learn.org/stable/datasets/) which also refair to [svmlight-loader](https://github.com/mblondel/svmlight-loader) as example.
- [SciKit-data](https://github.com/jaberg/skdata) is a package for loading/creating datasets for scikit-learn

### Datasets susceptible to be used in this study

| x | Dataset       | Size   | imbalance  | UCI     | Stalog  | svmlight-loader | skdata  | used at                              |
|:-:|:--------------| ------ | :--------: | :-----: | :-----: | :-------------: | :-----: | :-------------:                      |
|   |               |        |            |         |         |                 |         |                                      |
| x | Abalone       | 4177   | 1: 9.7     | x       |         | x               |         | [liu2008] [akbani] [sobhani]* [guo]* |
|   | Adult         | 48842  | 1: 3.2     | ---     | ---     | ---             |         | [chawla02]                           |
|   | Anneal        |        | 1:         | ---     | ---     | ---             |         | [akbani]                             |
|   | Arrhythmia    | 452    | 1: 17      | ---     | ---     | ---             |         | [sobhani]                            |
|   | Balance       | 625    | 1: 11.8    |         |         |                 |         | [liu2008] [sobhani] [akbani]         |
|   | Breast-cancer | 286    |            | x?      |         | x?              |         | [guo]                                |
|   | Breast-w      | 699    |            | ---     | ---     | ---             |         | [guo]                                |
|   | Can           | 443872 | 1: 52.1    | ---     | ---     | ---             |         | [chawla02]                           |
|   | Car           | 1728   | 1: 3.5     |         |         |                 |         | [liu2008] [akbani] [sobhani]         |
|   | Chess         |        | 1:         | ---     | ---     | ---             |         | [akbani]                             |
|   | Cmc           | 1473   | 1: 3.4     |         |         |                 |         | [liu2008]                            |
|   | E-state       | 53220  | 1: 7.4     | ---     | ---     | ---             |         | [chawla02]                           |
|   | Ecoli         | 336    | 1: 9       | ---     | ---     | ---             |         | [sobhani]                            |
|   | Forest-cover  | 38501  | 1: 13      | ---     | ---     | ---             |         | [chawla02]                           |
|   | Glass         | 214    | 1:         | ---     | ---     | ---             |         | [akbani] [guo]                       |
|   | Haberman      | 306    | 1: 2.8     |         |         |                 |         | [liu2008]                            |
|   | Hepatitis     | 155    | 1:         | ---     | ---     | ---             |         | [akbani] [guo]                       |
|   | Housing       | 506    | 1: 3.8     | x       |         | x               |         | [liu2008]                            |
|   | Hypothyroid   |        | 1:         | ---     | ---     | ---             |         | [akbani]                             |
|   | Imbalance     |        | 1:         | ---     | ---     | ---             |         | [akbani]                             |
|   | Ionosphere    | 351    | 1: 1.8     | x       |         | x               |         | [liu2008] [akbani] [guo]             |
| x | Letter        | 20000  | 1: 24.3    |         | x       | x               |         | [liu2008] [akbani]                   |
|   | Libras Move   | 360    | 1: 14      | ---     | ---     | ---             |         | [sobhani]                            |
|   | Mammography   | 11183  | 1: 42      | ---     | ---     | ---             |         | [chawla02]                           |
|   | Mf-morph      | 2000   | 1: 9.0     |         |         |                 |         | [liu2008]                            |
|   | Mf-zernike    | 2000   | 1: 9.0     |         |         |                 |         | [liu2008]                            |
|   | Monk2         | 169    |            | ---     | ---     | ---             |         | [guo]                                |
|   | Mushroom      | 8124?  | 1:         | x       |         | x?              |         | [akbani]                             |
|   | Oil           | 937    |            | ---     | ---     | ---             |         | [guo] [chawla02]                     |
|   | Phoneme       | 5404   | 1: 2.4     |         |         |                 |         | [liu2008] [guo] [chawla02]           |
| x | Pima          | 768    | 1: 1.9     | x       |         | x               | x       | [liu2008] [chawla02]                 |
|   | Primary-tumor | 339    |            | ---     | ---     | ---             |         | [guo]                                |
|   | Sat           |        | 1:         |         |         |                 |         | [liu2008] [guo]                      |
|   | Satimage      | 6435   | 1: 9.3     |         | x       | x               |         | [liu2008] [chawla02]                 |
|   | Segment       | 2310   | 1:         |         | x       | x               |         | [akbani] [guo]                       |
|   | Sick          | 3772   | 1:         | ---     | ---     | ---             |         | [skbani] [guo]                       |
|   | Sonar         | 208    | 1:         | x       |         | x               |         | [akbani] [guo]                       |
|   | Soybean       |        | 1:         | ---     | ---     | ---             |         | [akbani]                             |
|   | Spectrometer  | 531    | 1: 11      | ---     | ---     | ---             |         | [sobhani]                            |
|   | Vehicle       | 846    | 1: 3.0     |         | x       | x               |         | [liu2008] [guo]                      |
|   | Vowel         | 990    |            | ---     | ---     | ---             |         | [guo]                                |
|   | Wdbc          | 569    | 1: 1.7     |         |         |                 |         | [liu2008]                            |
|   | Wpbc          | 198    | 1: 3.2     |         |         |                 |         | [liu2008]                            |
|   | Yeast         | 1484   | 1: 28      | ---     | ---     | ---             |         | [sobhani] [guo]                      |

* sobhani claims that using abalone as ring=19 the imbalance turns 1:130? 
* guo and liu sizes disagree at phoneme and abalone. 
* guo and sobhani yeast size difers
* car sobhani says 1:25
* breast-cancer UCI is 683 elements and can be found at svmlitle
* `---` is stuff to be checked




[liu2008]:http://cse.seu.edu.cn/people/xyliu/publication/tsmcb09.pdf
[akbani]:http://link.springer.com/chapter/10.1007%2F978-3-540-30115-8_7
[sobhani]:http://www.di.uniba.it/~ceci/micFiles/NFMCP2014Proceedings/nfmcp2014_submission_6.pdf
[guo]:https://www.site.uottawa.ca/~hguo028/papers/KDDExplorations2004.pdf
[chawla02]:https://www.jair.org/media/953/live-953-2037-jair.pdf

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

