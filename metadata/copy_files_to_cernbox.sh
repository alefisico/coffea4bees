#!/bin/bash

# Default values
source_folder="/store/user/algomez/XX4b/2024_v1p1/"
destination_folder="/eos/user/a/algomez/tmpFiles/XX4b/2024_v1p1/"
folder_filter="Glu"
file_filter="SvB"

# Parse command line arguments
while getopts s:d:f:F: flag
do
    case "${flag}" in
        s) source_folder=${OPTARG};;
        d) destination_folder=${OPTARG};;
        f) folder_filter=${OPTARG};;
        F) file_filter=${OPTARG};;
    esac
done

for ifolder in `eosls $source_folder`;
do
    if [[ $ifolder =~ $folder_filter ]];
    then
        for ifile in `eosls ${source_folder}/${ifolder}`;
        do
            if [[ -z $file_filter || $ifile =~ $file_filter ]];
            then
                echo "Transferring ${source_folder}/${ifolder}/${ifile}"
                f=${source_folder}/${ifolder}/${ifile}
                echo xrdcp -f root://cmseos.fnal.gov/${f} root://eosuser.cern.ch/${destination_folder}/${ifolder}/${ifile}
                xrdcp -f root://cmseos.fnal.gov/${f} root://eosuser.cern.ch/${destination_folder}/${ifolder}/${ifile}
            fi
        done
    fi
done