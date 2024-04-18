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

which may take a huge amount of time (~30min) and disk space to download, unpack and convert the image for the first time.

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

### Run Command Line Interface by examples

#### Help

To list all available commands and options:

```bash
python run_classifier.py help --all
```

#### Monitor

Start a monitor using port 10200 and save the logs to a local directory named by the current timestamp:

```bash
python run_classifier.py monitor --setting Monitor "port: 10200" --setting IO "output: ./logs-{timestamp}/"
```

It will print the IP address and port number that the monitor is listening to. e.g.
  
```console
[04/18/24 13:36:18] [ main] INFO     Started Monitor at 127.0.1.1:10200
```

Which port to use:

- LPC: 10000-10200 is usally safe to use
- PSC: TBD
- LXPLUS: TBD

#### Training in two steps

##### Cache preprocessed datasets to speed up loading (Optional)

Cache the `HCR.FvT_picoAOD` dataset using a pre-defined workflow and connect to the monitor we started above:

```bash
export WFS="classifier/config/workflows/examples"
python run_classifier.py from ${WFS}/cache_training_set.yml --setting Monitor "{address: 127.0.1.1, port: 10200}"
```

By default, it will write to `root://cmseos.fnal.gov//store/user/{user}/HH4b/classifier/cache/` in LPC, which can be changed by appending `--setting IO "output: /path/to/save/"`.

> **_NOTE:_** Check what files are cached:

```bash
python run_classifier.py debug --dataset HCR.FvT_picoAOD
```

> **_NOTE:_** Use `expand` to recover the command line arguments from workflows:

```bash
python run_classifier.py expand ${WFS}/cache_training_set.yml
```

> **_NOTE:_** Use `workflow` to generate the workflow file from command line:

```bash
python run_classifier.py workflow ${WFS}/test.yml train --max-loaders 4 --max-trainers 1  --dataset ... --model ... --setting torch.DataLoader "yaml:##{batch_skim: 65536, num_workers: 2}"
```

##### Train FvT classifier with HCR architecture using cached datasets

Load the dataset from cache and train the classifier using the example workflow:

```bash
python run_classifier.py from ${WFS}/train_hcr_fvt.yml --template "user: "${USER} ${WFS}/template/load_cached_dataset.yml --setting Monitor ... --setting IO ...
```

> **_NOTE:_** By using `--template` with a mapping followed by files, it will replace the keys in the files with Python's [`str.format`](https://docs.python.org/3/library/string.html#format-string-syntax) (escaped by `{{` and `}}`). e.g. replace `{user}` by current `${USER}`.

#### Evaluate

TODO

## Advanced

TODO

## TODO

- HCR:
  - Add die loss?
- main:
  - Add evaluation
  - Add plotting
- Add SvB
- Add DvT
