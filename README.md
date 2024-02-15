# Coffea4bees

This is the repository for the 4b analyses at CMU based in coffea. 
This repository in based on the coffea implementation of the ZZ/ZH to 4b analysis for Run2. More information about that repository [here](https://github.com/patrickbryant/ZZ4b).

The package has a python component, where most of the analysis is made, and a c++ component meant to be run inside CMSSW.

Information about the analysis steps can be found in the [README](python/analysis/README.md) of the analysis folder.

## Installation

### How to run the python files

This repository assumes that you are running in a machine that has access to [cvmfs](https://cernvm.cern.ch/fs/). Then you can clone this repository as:

```
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/coffea4bees.git
```

This code has been tested at the cmslpc, and to simplify the setup, it can be used with the container needed to run on lpc condor computers. To set this container:
```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```
This creates two new files in this directory: `shell` and `.bashrc`. _Additionally, this package contains a `set_shell.sh file`_ which runs the `./shell` executable with the coffea4bees container. This container is based on the `coffeateam/coffea-dask:latest` container including some additional python packages. 
```
source set_shell.sh
```

Remember to run this previous command (aka set your environment) *every time you want to run something*.

To know more about the container, you can see the [Dockerfile](Dockerfile). To know more information about the lpcjobqueue package click [here](https://github.com/CoffeaTeam/lpcjobqueue).


In addition, dont forget to run your voms-proxy to have access to remote files:

```
voms-proxy-init -rfc -voms cms --valid 168:00
```

### How to run the src files

To be used inside CMSSW. More info later.


## How to contribute 

If you want to submit your changes to the code to the **main repository** (aka cms-cmu gitlab user) it is highly recommended to first fork this repository to your user. 
Then, in your local machine you can add your own fork to your working directory:
```
git remote add myRepo ssh://git@gitlab.cern.ch:7999/USER/coffea4bees.git      ### change USER for your gitlab username
```
Then, if you want to push your changes to your own gitlab repository:
```
git add FILE1 FILE2 
git commit -m "add a message"
git push myRepo BRANCH        #### change BRANCH with the name of your branch
```
Once you are happy with your changes, you can make a merge request in the gitlab website to the main repository.

## Information for continuos integration (CI)

By default only the **master branch** runs the gitlab CI workflow. If you want to push incomplete or buggy code, without running the CI workflow, create a new branch. 

The workflow for the CI can be found in the [gitlab-ci.yml](.gitlab-ci.yml) file.

The CI runs on remote files and therefore it needs your grid certificate. If you want to run the gitlab CI workflow in your private fork, you need first to create some variables to set up your voms-proxy. You can follow [these steps](https://awesome-workshop.github.io/gitlab-cms/03-vomsproxy/index.html) (except the last part, Using the grid proxy).

If you did this step correctly, then you can check in your pipelines and see that the stage `build`, job `voms-proxy` ran succesfully.

## Information about the container

This packages uses its own container. It is based on `coffeateam/coffea-dask:latest` including some additional python packages. This container is created automatically in the gitlab CI step **IF** the name of the branch (and the merging branch in the case of a pull request to the master) starts with `container_`. Additionally, one can take a look at the file [.dockerfiles/Dockerfile_analysis](.dockerfiles/Dockerfile_analysis) which is the one used to create the container.

## Python sytle tips:

PEP8

https://peps.python.org/pep-0008/
https://pypi.org/project/pycodestyle/


```
> pycodestyle  --show-source base_class/plots.py

> pycodestyle --show-pep8 --show-source base_class/plots.py 
```
