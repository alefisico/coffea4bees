# Coffea4bees skimmer

To run the skimmer, remember first to set the coffea environment and your grid certificate. If you followed the instructions in the [README.md](../../README.md), the `set_shell.sh` file must be located right after the package, and then:
```
voms-proxy-init -rfc -voms cms --valid 168:00
cd coffea4bees/ ## Or go to this directory
source set_shell.sh
```

## In this folder

Here you find the code to create `picoAODs` (skims from nanoAOD)

Each folder contains:
 - [metadata](./metadata/): yml files to run the processors
 - [processors](./processors/): python files with the processors for each skimms.

Then, the run-all script is called `runner.py` and it is one directory below (in [python/](../../python/)). This script will run local or condor depending on the flag used. To learn all the options of the script, just run:
```
# (inside /coffea4bees/python/)
python runner.py --help
```

## Run Skimmer

### Example to run the skimmer 

For example, to run a processor you can do:
```
#  (inside /coffea4bees/python/)
python runner.py -s -p skimmer/processor/skimmer_4b.py -m skimmer/metadata/HH4b.yml -y UL18 -d TTTo2L2Nu -op skimmer/metadata/ -o picoaod_datasets.yml -t
```

This processor creates two files: 
 - One yaml file which is explicitely define in the commands above. This file contains the information of the picoAOD file created and some other variables
 - One root file, which its location is define in the HH4b.yml file. 
