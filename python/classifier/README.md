# Classifier

## Getting Started

### Setup Environment

#### Use [Singularity(Apptainer)](https://apptainer.org/docs/user/latest/) (Recommended)

```bash
export SINGULARITY_IMAGE="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/chuyuanliu/heptools:ml"
singularity shell -B .:/srv -B /cvmfs -B $(readlink ${HOME}/nobackup) --nv --pwd /srv ${SINGULARITY_IMAGE}
```

where:

- `-B .:/srv` mount the current directory to `/srv`
- `-B /cvmfs` mount the CVMFS
- `-B $(readlink ${HOME}/nobackup)` mount the `nobackup` directory when using LPC
- `--nv` enable GPU support
- `--pwd /srv` set the working directory to `/srv` when entering the container

If `/cvmfs` is not available, consider use the container directly from [Docker Hub](https://hub.docker.com/repository/docker/)

```bash
export SINGULARITY_IMAGE="docker://chuyuanliu/heptools:ml"
```

which may take a few minutes and ~20GB disk space to download, unpack and convert the image for the first time.

On LPC, to avoid run out of quota, change the temp and cache directory to a different location:

```bash
export APPTAINER_CACHEDIR="${HOME}/nobackup/.apptainer/"
export APPTAINER_TMPDIR="${HOME}/nobackup/.apptainer/"
```

#### Use [LCG](https://lcgdocs.web.cern.ch/lcgdocs/lcgreleases/introduction/)

Select a release with `torch`, `cuda`, `awkward>=2.0.0` and `uproot>=5.0.0`, e.g. [LCG-dev3cuda](https://lcginfo.cern.ch/release_packages/dev3cuda/x86_64-centos7-gcc11-opt/)

```bash
source /cvmfs/sft.cern.ch/lcg/views/dev3cuda/x86_64-centos7-gcc11-opt/setup.sh
```

### Use [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

Create conda environment from `env.yml`:

```bash
conda env create -f env.yml
conda activate ml
```

or use [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for faster dependency resolution:

```bash
mamba env create -f env.yml
```

#### Server Specific

##### falcon/rogue01

`singularity` is now available in rogue01.

1. change the cache directory for singularity:
    1. `mkdir -p /mnt/scratch/${USER}/.apptainer`
    2. add the following to `~/.bashrc`

        ```bash
        export SINGULARITY_CACHEDIR="/mnt/scratch/${USER}/.apptainer/"
        export SINGULARITY_TMPDIR="/mnt/scratch/${USER}/.apptainer/"
        ```

2. install grid certificate:
    1. `mkdir -p ~/.globus/`
    2. upload the certificate file e.g. `mycert.p12` to `~/.globus/`
    3. follow the instructions in this [twiki](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookStartingGrid#ObtainingCert) page until `voms-proxy-init` step.
  
3. start a container:
    1. `cd` into the the working directory (e.g. `/mnt/scratch/${USER}/coffea4bees/python/`)
    2. create a container from the docker image and setup the proxy:

        ```bash
        singularity shell -B .:/srv --nv --pwd /srv docker://chuyuanliu/heptools:ml
        # this may take a while for the first time
        # inside the container (when you see Singularity>)
        voms-proxy-init --rfc --voms cms -valid 192:00
        # setup the autocomplete
        source ./classifier/install.sh
        ```

Notes:

- use e.g. "localhost:10200" when connecting to the monitor started in the same rogue node

### Run Command Line Interface by examples

#### Setup

Setup the environment for

- command line autocomplete

```bash
source classifier/install.sh
```

Uninstall the environment:

```bash
source classifier/uninstall.sh
```

#### Autocomplete

Some examples:

complete main task

```bash
./pyml.py <tab><tab>
./pyml.py h<tab><tab>
```

complete path

```bash
./pyml.py from <tab><tab>
./pyml.py train -template "user: "${USER} <tab><tab>
```

complete model

```bash
./pyml.py help -s<tab><tab>
./pyml.py train -setting <tab><tab>
./pyml.py debug -dataset HCR<tab><tab>
```

complete setting

```bash
./pyml.py train -setting IO "<tab><tab>
```

```bash
./pyml.py train -setting IO "o<tab><tab>
```

#### Help

To list all available commands and options:

```bash
./pyml.py help --all
```

#### Monitor

Start a monitor using port 10200 and save the logs to a local directory named by the current timestamp:

```bash
./pyml.py monitor -setting Monitor "address: 10200" -setting IO "output: ./logs-{timestamp}/"
```

It will print the IP address and port number that the monitor is listening to. e.g.

```console
[04/18/24 13:36:18] [ main] INFO     Started Monitor at 127.0.1.1:10200
```

Which port to use:

- LPC: 10000-10200 is usally safe to use

Connect to the monitor:

```bash
./pyml.py ... -setting Monitor "{address: 127.0.1.1:10200, connect: true}"
```

#### Quick start

##### SvB

Setup some environment variables:

```bash
export WFS="classifier/config/workflows/HCR/SvB"
```

###### Cache the datasets

Cache the `HCR.SvB.Background` and `HCR.SvB.Signal` datasets:

```bash
./pyml.py template "{user: ${USER}, norm: 6, dataset: default}" ${WFS}/cache_dataset.yml 
```

where the `norm` is the total signal normalization and `dataset` is the name of the dataset. 

> **_NOTE:_** By using `-template` with a mapping followed by files, it will replace the keys in the files with Python's [`str.format`](https://docs.python.org/3/library/string.html#format-string-syntax) (use double curlies `{{KEY}}` to escape). e.g. replace `{user}` by current `${USER}`. The definitions of the keys can be found (and should be included) in the comments at the beginning of the template files.

By default, it will write to `root://cmseos.fnal.gov//store/user/{user}/HH4b/classifier/SvB/dataset/{dataset}/`, which can be overwritten by appending `-setting IO "output: /new/path/to/save/"`.

> **_Optional:_** Check the definitions of the datasets used above (use `-setting IO "output: null"` to disable output):

```bash
./pyml.py debug -dataset HCR.SvB.Background -dataset HCR.SvB.Signal -setting IO "output: null"
```

> **_NOTE:_** Inspect the command defined by the workflow:

```bash
./pyml.py expand "{user: ${USER}, norm: 6, dataset: default}" ${WFS}/cache_dataset.yml --as-template
```

###### Train the classifier

Load the dataset from cache and train the classifier using the example workflow:

```bash
./pyml.py template "{tag: test}" ${WFS}/train_baseline.yml -template "{user: ${USER}, dataset: default}" ${WFS}/load_cache.yml -setting torch.DataLoader "{num_workers: 4, batch_eval: 100000}"
```

where `-setting torch.DataLoader` is used to set the number of workers used by dataloaders and evaluation (used in validation step) batch size. In general, the larger these numbers are, the faster the training will be but the more memory (`num_workers`) and GPU memeory (`batch_eval`) will be used.

The result will be stored in `root://cmseos.fnal.gov//store/user/${USER}/HH4b/classifier/SvB/train/{tag}/result.json` including the benchmarks and the path to the pickled model.

###### Make benchmark plots

Publish the roc and loss plots to cernbox (grid certification required):

```bash
./pyml.py template "{tag: test, user: ${USER}, cernuser: CERN_USERNAME}" ${WFS}/plot_roc.yml
```

The result will be stored in `root://eosuser.cern.ch//eos/user/{cernuser}/www/HH4b/classifier/SvB/{tag}/` and is accessible through the url.

> **_NOTE:_** The `cernuser` is required to be the format of e.g. `c/chuyuan` to complete the path.

###### Evaluate the classifier

Evaluate the classifier and merge the k-folds:

```bash
./pyml.py template "{tag: test, user: ${USER}}" ${WFS}/evaluate.yml -setting torch.DataLoader "{num_workers: 4, batch_eval: 100000}"
```

##### FvT

Similar to [`SvB`](#svb),

```bash
export WFS="classifier/config/workflows/HCR/FvT"
# cache the datasets
./pyml.py template "{user: ${USER}, dataset: default}" ${WFS}/cache_dataset.yml
# train the classifier
./pyml.py template "{tag: test, offset: 0-2}" ${WFS}/train_random.yml -template "{user: ${USER}, dataset: default}" ${WFS}/load_cache.yml -setting torch.DataLoader "{num_workers: 4}"
# make benchmark plots
./pyml.py template "{tag: test, user: ${USER}, cernuser: CERN_USERNAME}" ${WFS}/plot_roc.yml
# evaluate the classifier (without merging k-folds)
./pyml.py template "{tag: test, user: ${USER}}" ${WFS}/evaluate.yml -setting torch.DataLoader "{num_workers: 4, batch_eval: 100000}"
```

where the `tag` and `dataset` can be any string. The `offset` is used to make different random k-foldings, e.g.

- `offset: 0-2` will give offset 0, 1, 2
- `offset: "0 2 4"` will give offset 0, 2, 4
- `offset: "0-2 4-6 8"` will give offset 0, 1, 2, 4, 5, 6, 8

To work with cluster, each job can have a different `offset` and `tag`.

```bash
# job1
./pyml.py template "{tag: offset1, offset: 1-3}" ${WFS}/train_random.yml -setting ...
# job2
./pyml.py template "{tag: offset2, offset: 4-6}" ${WFS}/train_random.yml -setting ...
# job3
...
```

To merge k-folds from multiple results:

```bash
./pyml.py analyze /path/to/offset1/result.json /path/to/offset2/result.json ... -analysis kfold.Merge --name FvT --workers 8 -setting IO "output: /path/to/output/directory/"
```

##### Tips on performance

- Training set caching:
  - in `-dataset HCR.*`, consider increase `--max-workers` (mainly CPU bounded, require extra memory)
- Training:
  - in main task `train`, consider parallel multiple models by increasing `--max-trainers` (CPU, GPU, memory bounded)
  - in `-setting torch.DataLoader`, consider increase `num_workers` to speed up batch generation (mainly CPU bounded, require extra memory)
  - in `-setting torch.DataLoader`, consider increase `batch_eval`  to speed up evaluation (mainly GPU bounded)
  - in `-setting torch.Training`, the `disable_benchmark` can be enabled to skip all benchmarking steps.
- Evaluation:
  - in main task `evaluate`, consider parallel multiple models by increasing `--max-evaluators` (CPU, GPU, memory bounded)
  - in `-setting torch.DataLoader`, consider increase `num_workers` and `batch_eval`. (see above)
- Merging k-folds:
  - in `-analysis kfold.Merge`, consider increase `--workers` (mainly CPU bounded, require extra memory)
  - in `-analysis kfold.Merge`, consider use a finite `--step` to split root files into smaller chunks (only useful when the total number of files is smaller than the number of cores)
- **WARNING**:
  - NEVER turn on `-setting monitor.Usage "enable: true"` in production. It is designed for quick test jobs only.

## Advanced

> **_WARNING:_** Do NOT import any large modules or modules that depend on those (e.g. `torch`, `numpy`, `pandas`, `base_class.root`, `classifier.ml`, etc) at the top-level in any files located under `classifier/config/`. Instead, import them inside the scope that use them. Otherwise, the `autocomplete`, `help` functions will be dramatically slowed down.

> **_NOTE:_** If the code exits unexpectedly without showing any error message, try to run with the option `-setting monitor.Log "forward_exception: false"`.

TODO

## TODO

- main:
  - Add simple torch histogram
  - Add train/plot from saved model
- Add DvT
