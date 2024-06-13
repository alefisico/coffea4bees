#!/bin/bash

# Function to handle arument parsing
parse_aruments() {

  # Check if folder arument is provided
  if [ -z "$1" ]; then
    echo "Missing folder arument"
  fi
  datacard_folder="$1"

  # Set defaults for flags
  impacts=false
  postfit=false

  # Process aruments
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
        echo "Invalid arument: '$2'"
        ;;
    esac
  done
}

# Parse aruments
parse_aruments "$@"

currentDir=$PWD
signallabel="HH"
# signallabel="ggHH_hbbhbb"

for iclass in SvB_MA;
do
    cd ${datacard_folder}/
    text2workspace.py combine_${iclass}.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO "map=.*/${signallabel}:r${signallabel}[1,-10,10]" #--PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
    combine -M AsymptoticLimits combine_${iclass}.root --redefineSignalPOIs r${signallabel} -n _${iclass} --run blind > limits.txt
    cat limits.txt
    combineTool.py -M CollectLimits higgsCombine_${iclass}.AsymptoticLimits.mH120.root -o limits.json

    if [ "$impacts" = true ]; then

        combineTool.py -M Impacts -d combine_${iclass}.root --doInitialFit --setParameterRanges r${signallabel}=-10,10 --setParameters r${signallabel}=1 --robustFit 1 -m 125 -n ${iclass} -t -1 ## expected -t -1

        combineTool.py -M Impacts -d combine_${iclass}.root --doFits --setParameterRanges r${signallabel}=-10,10 --setParameters r${signallabel}=1 --robustFit 1 -m 125 --parallel 4 -n ${iclass} -t -1

        combineTool.py -M Impacts -d combine_${iclass}.root -o impacts_combine_${iclass}_exp.json -m 125 -n ${iclass}

        plotImpacts.py -i impacts_combine_${iclass}_exp.json -o impacts_combine_${iclass}_exp_HH --POI r${signallabel} --per-page 20 --left-marin 0.3 --height 400 --label-size 0.04 --translate nuisance_names.json

    elif [ "$postfit" = true ]; then

        combine -M MultiDimFit --setParameters rZZ=1,rZH=1,r${signallabel}=1 --robustFit 1 -n _${iclass}_fit_s --saveWorkspace --saveFitResult -d combine_${iclass}.root

        PostFitShapesFromWorkspace -w higgsCombine_${iclass}_fit_s.MultiDimFit.mH120.root -f multidimfit_${iclass}_fit_s.root:fit_mdf --total-shapes --postfit --output postfit_s.root

    fi

    cd $currentDir

done
