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

### Run Command Line Interface

#### Help

To list all available commands and options:

```bash
python run_classifier.py help --all
```

#### Train

##### Simple HCR FvT

Check what files will be used:

```bash
python run_classifier.py debug --dataset HCR.FvT_picoAOD
```

Train with default datasets and settings:

```bash
python run_classifier.py train --max-trainers 3 \
--dataset HCR.FvT_picoAOD \
--friends data,friend path/to/friend.metadata.json@@friend.data \
--friends ttbar,friend path/to/friend.metadata.json@@friend.ttbar \
--model HCR.FvT
```

#### Evaluate

TODO

## Advanced

TODO

## TODO

- HCR:
  - Add fine-tuning
  - Add die loss?
  - Add help --filter FILTER regex
- main:
  - Add evaluation
  - Add logging, benchmark, plotting
- FvT
  - SR ?
- Add SvB
- Add DvT
