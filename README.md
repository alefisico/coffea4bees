# Coffea4bees

This is the repository for the 4b analyses at CMU based in coffea. 
This repository in based on the coffea implementation of the ZZ/ZH to 4b analysis for Run2. More information about that repository [here](https://github.com/patrickbryant/ZZ4b).

## Installation

This repository assumes that you are running in a machine that has access to [cvmfs](https://cernvm.cern.ch/fs/). Then you can clone this repository as:

```
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/coffea4bees.git
```

Then you can set the environment for the package to work:
```
cd coffea4bees/
source set_env.sh
```

Remember to run this command (aka set your environment) *every time you want to run something*. In addition, dont forget to run your voms-proxy to have access to remote files:

```
voms-proxy-init -rfc -voms cms --valid 168:00
```

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

By default only the **master branch** of the **main repository** (cms-cmu user) runs the gitlab CI workflow. 

If you want to run the gitlab CI workflow in your private fork, you need first to create some variables to set up your voms-proxy. You can follow [these steps](https://awesome-workshop.github.io/gitlab-cms/03-vomsproxy/index.html) (except the last part, Using the grid proxy).

After you need to run the pipeline manually.
