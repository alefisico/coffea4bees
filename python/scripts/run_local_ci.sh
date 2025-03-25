#!/bin/bash

# Check if the job name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <job_name>"
fi

JOB_NAME=$1
SNAKEFILE="workflows/Snakefile_testCI"

# Check if JOB_NAME contains '-'
if [[ "$JOB_NAME" == *-* ]]; then
  # Replace '-' with '_'
  JOB_NAME=${JOB_NAME//-/_}
fi

# Check if the folder named 'output' exists
if [ -d "CI_output" ]; then
  echo "The folder 'CI_output' exists. Remember that snakemake will not run a step if the output files already exist."
else
  echo "Output files will be created in the 'CI_output' folder."
fi

# Check if the file ~/x509up* exists
if ls ~/x509up* 1> /dev/null 2>&1; then
  echo "Copying ~/x509up* to /proxy/x509_proxy."
  /bin/cp ~/x509up* proxy/x509_proxy
else
  echo "File ~/x509up* does not exist. Run voms-proxy-init -voms cms before running this script."
  return 0
fi

# Search for the job name and assign the output list it belongs to a variable
OUTPUT_LIST=$(awk -v job="$JOB_NAME" '
  BEGIN { found=0; }
  /^output_/ { in_list=1; output_list=$1; next; }
  in_list && /^\]/ { in_list=0; }
  in_list && $0 ~ job { found=1; print output_list; exit; }
  END { if (!found) print ""; }
' "$SNAKEFILE")

if [ -z "$OUTPUT_LIST" ]; then
  echo "Job name not found in any output list."
else
  echo "Changing the output list to $OUTPUT_LIST"

  sed -e "s/input: outputs/input: $OUTPUT_LIST/" "$SNAKEFILE" > /tmp/Snakefile_testCI

  pwd -LP
  ./run_container snakemake --snakefile /tmp/Snakefile_testCI --use-apptainer --printshellcmds --keep-incomplete --until $JOB_NAME --cores 1

fi

