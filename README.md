# Coffea4bees


[![pipeline status](https://gitlab.cern.ch/cms-cmu/coffea4bees/badges/master/pipeline.svg)](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commits/master)


This is the repository for the 4b analyses at CMU based in coffea. 

The package has a python component, where most of the analysis is made, and a c++ component meant to be run inside CMSSW.

Information about the analysis steps can be found in the [README](python/analysis/README.md) of the analysis folder.

## Installation

### How to run the python files

This repository assumes that you are running in a machine that has access to [cvmfs](https://cernvm.cern.ch/fs/). Then you can clone this repository as:

```
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/coffea4bees.git --recursive
```

#### To run at the CMSLPC

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

#### To run on lxplus

The script `lxplus_shell.sh` can be run to run the code on lxplus. This script contains two options `--coffea4bees` and `--combine`, each one will setup the dedicated container need, as:

```
source lxplus_shell.sh --coffea4bees
```


#### Conda environment

In case you want to run the package using a conda environmnent, you can use the [environment.yml](environment.yml) file. Notice however that there are some libraries missing in case you want to run the full framework.


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

## REANA

[![Launch with Snakemake on REANA](https://www.reana.io/static/img/badges/launch-on-reana.svg)]($https://reana.cern.ch/launch?name=Coffea4bees&specification=reana.yml&url=https%3A%2F%2Fgitlab.cern.ch%2Fcms-cmu%2Fcoffea4bees)

This package runs a workflow in [REANA](https://reana.cern.ch/) for every commit to the master. The output of the reana workflow can be found here:

Website with plots and output files are in [https://plotsalgomez.webtest.cern.ch/HH4b/reana/](https://plotsalgomez.webtest.cern.ch/HH4b/reana/)

The folders there should contain the date the reana job was launched and the git hash of the commit. 
Differnt than before, the folders are copied to this folder only if the reana job sucessfully finished. 

## Information for continuos integration (CI)

By default only the **master branch** runs the gitlab CI workflow. If you want to push incomplete or buggy code, without running the CI workflow, create a new branch. 

The workflow for the CI can be found in the [gitlab-ci.yml](.gitlab-ci.yml) file.

The CI runs on remote files and therefore it needs your grid certificate. If you want to run the gitlab CI workflow in your private fork, you need first to create some variables to set up your voms-proxy. You can follow [these steps](https://awesome-workshop.github.io/gitlab-cms/03-vomsproxy/index.html) (except the last part, Using the grid proxy).

If you did this step correctly, then you can check in your pipelines and see that the stage `build`, job `voms-proxy` ran succesfully.

### To run the ci workflow locally in your machine

We are using [Snakemake](https://snakemake.readthedocs.io/en/stable/) to recreate the workflow run in the GitLab CI. Snakemake is the workflow management package that REANA uses to submit the jobs. 

Inside the [.ci-workflows](.ci-workflows) folder there are two files: `run_local_ci.sh` and `Snakefile_testCI`. The `run_local_ci.sh` file is a convenient way to run the part of the CI workflow that needs to be run locally, while the `Snakefile_testCI` defines the workflow. If you need to run the CI locally, you can just run **from the root folder coffea4bees/**:

```
source .ci-workflows/run_local_ci.sh NAME_OF_CI_JOB
```
where `NAME_OF_CI_JOB` corresponds to the job's name in the GitLab CI workflow. This command will automatically run the part of the CI to which the job belongs. All the output files will be located inside the `python/output/` folder, and each step will create a separate folder with the job's name. 

Remember, that is a **feature** of `Snakemake` to first check if the output files of each job exist. If the files exist, the job will be skipped to the next part of the workflow. Therefore, if you are debugging and need to rerun the workflow, remember to manually remove the folder containing the output files. 

If you are interested in `Snakemake`, the file `Snakefile_testCI` defines a "rule" (job) similar to the job defined for gitlab CI. Also, the way of including rules in the workflow depends on the input to `rule all`. Rules can be defined anywhere after `rule all` but will only be run IF the output files are listed in `rule all`. Finally, unlike gitlab CI, where the output files **should** be listed, in snakemake, the output files need to define the subsequent rule to follow. 


## Information about the container

This packages uses its own container. It is based on `coffeateam/coffea-dask:latest` including some additional python packages. This container is created automatically in the gitlab CI step **IF** the name of the branch (and the merging branch in the case of a pull request to the master) starts with `container_`. Additionally, one can take a look at the file [.dockerfiles/Dockerfile_analysis](.dockerfiles/Dockerfile_analysis) which is the one used to create the container.

## Python sytle tips:

If you want to test your code against the PEP8 style, you can use this:

```
> pycodestyle  --show-source base_class/plots.py

> pycodestyle --show-pep8 --show-source base_class/plots.py 

> pycodestyle --ignore E501,E222,E241,E202,E221,E201     --show-source analysis/processors/processor_HH4b.py
```
