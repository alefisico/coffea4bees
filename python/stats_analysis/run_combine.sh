#!/bin/bash

# Function to handle argument parsing
parse_arguments() {

  # Check if folder argument is provided
  if [ -z "$1" ]; then
    echo "Missing folder argument"
  fi
  datacard_folder="$1"

  # Set defaults for flags
  impacts=false
  postfit=false

  # Process arguments
  while [[ $# -gt 1 ]]; do
    case "$2" in
      --impacts)
        impacts=true
        shift
        ;;
      --postfit)
        postfit=true
        shift
        ;;
      *)
        echo "Invalid argument: '$2'"
        ;;
    esac
  done
}

# Parse arguments
parse_arguments "$@"

currentDir=$PWD

for iclass in SvB_MA;
do
    cd ${datacard_folder}/
    text2workspace.py combine_${iclass}.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/ggHH4b_kl1_kt1_c20:rggHH4b_kl1_kt1_c20[1,-10,10]' #--PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
    combine -M AsymptoticLimits combine_${iclass}.root --redefineSignalPOIs rggHH4b_kl1_kt1_c20 -n _${iclass} > limits.txt
    cat limits.txt
    combineTool.py -M CollectLimits higgsCombine_${iclass}.AsymptoticLimits.mH120.root -o limits.json

    if [ "$impacts" = true ]; then

        combineTool.py -M Impacts -d combine_${iclass}.root --doInitialFit --setParameterRanges rggHH4b_kl1_kt1_c20=-10,10 --setParameters rggHH4b_kl1_kt1_c20=1 --robustFit 1 -m 125 -n ${iclass} -t -1 ## expected -t -1

        combineTool.py -M Impacts -d combine_${iclass}.root --doFits --setParameterRanges rggHH4b_kl1_kt1_c20=-10,10 --setParameters rggHH4b_kl1_kt1_c20=1 --robustFit 1 -m 125 --parallel 4 -n ${iclass} -t -1

        combineTool.py -M Impacts -d combine_${iclass}.root -o impacts_combine_${iclass}_exp.json -m 125 -n ${iclass}

        plotImpacts.py -i impacts_combine_${iclass}_exp.json -o impacts_combine_${iclass}_exp_HH --POI rggHH4b_kl1_kt1_c20 --per-page 20 --left-margin 0.3 --height 400 --label-size 0.04 --translate nuisance_names.json

    elif [ "$postfit" = true ]; then

        combine -M MultiDimFit --setParameters rZZ=1,rZH=1,rggHH4b_kl1_kt1_c20=1 --robustFit 1 -n _${iclass}_fit_s --saveWorkspace --saveFitResult -d combine_${iclass}.root

        PostFitShapesFromWorkspace -w higgsCombine_${iclass}_fit_s.MultiDimFit.mH120.root -f multidimfit_${iclass}_fit_s.root:fit_mdf --total-shapes --postfit --output postfit_s.root

    fi

    cd $currentDir

done
