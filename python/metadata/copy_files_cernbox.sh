#!/bin/bash

# Define source and destination paths
CERNBOX_PATH="/eos/user/m/mkolosov/Run3_HHTo4B_NTuples/2022/Main_PNet_MinDiag_w4j35_w2b30_JetTypeAK4PFPuppiPNetRegrNeutrino_noJER_wJets10_woSyst_19Sep2024_2022_0L/

"  # Replace with your CERNbox path
LPC_PATH="/store/user/algomez/XX4b/Florida_Run3/2022_postEE/"  # Replace with your LPC path

# Loop through files in CERNbox folder
for file in $(xrdfs root://eosuser.cern.ch/ ls $CERNBOX_PATH); do
  # Construct full source and destination file paths
  source_file="root://eosuser.cern.ch/$file"
  file_name=$(basename "$file")  # Extract file name
  destination_file="root://cmseos.fnal.gov/$LPC_PATH/$file_name"

  # Copy the file using xrdcp
  echo "Copying $source_file to $destination_file"
  xrdcp -rf $source_file $destination_file

  # Check for errors
  if [[ $? -ne 0 ]]; then
    echo "Error copying $source_file"
  fi
done

echo "Copying complete!"
