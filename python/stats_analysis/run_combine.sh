#!/bin/bash

# Function to handle arument parsing
parse_aruments() {

  # Check if folder arument is provided
  if [ -z "$1" ]; then
    echo "Missing folder arument"
  fi
  datacard_folder="$1"

  # Set defaults for flags
  limits=false
  impacts=false
  postfit=false

  # Process aruments
  while [[ $# -gt 1 ]]; do
    case "$2" in
      --limits)
        limits=true
        shift
        ;;
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
signallabel="ggHH_kl_1_kt_1_hbbhbb"
# signallabel="ggHH"

run_limits() {
  local datacard=$1
  local signallabel=$2
  local iclass=$3

  text2workspace.py ${datacard}.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO "map=.*/${signallabel}:r${signallabel}[1,-20,20]" \
    --PO "map=.*/ggHH_kl_0_kt_1_hbbhbb:rggHH_kl_0_kt_1_hbbhbb[0,0,0]" \
    --PO "map=.*/ggHH_kl_2p45_kt_1_hbbhbb:rggHH_kl_2p45_kt_1_hbbhbb[0,0,0]" \
    --PO "map=.*/ggHH_kl_5_kt_1_hbbhbb:rggHH_kl_5_kt_1_hbbhbb[0,0,0]" 
    # --PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
  combine -M AsymptoticLimits ${datacard}.root --redefineSignalPOIs r${signallabel} \
    -n _${iclass} --run blind \
    --setParameters rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \
    > limits_${datacard}.txt
  cat limits_${datacard}.txt
  combineTool.py -M CollectLimits higgsCombine_${iclass}.AsymptoticLimits.mH120.root -o limits_${datacard}.json
}

for iclass in SvB_MA;
do
    datacard="datacard" #_stat_only"
    # datacard="combine_"${iclass}
    cd ${datacard_folder}/
    
    if [ "$limits" = true ]; then

        combineCards.py HHbb_2016=datacard_HHbb_2016.txt \
            HHbb_2017=datacard_HHbb_2017.txt \
            HHbb_2018=datacard_HHbb_2018.txt > ${datacard}.txt
        run_limits $datacard $signallabel $iclass

        run_limits datacard_HHbb_2016 $signallabel $iclass
        run_limits datacard_HHbb_2017 $signallabel $iclass
        run_limits datacard_HHbb_2018 $signallabel $iclass

    elif [ "$impacts" = true ]; then

        if [ -f "${datacard}.root" ]; then

            # datacard="datacard_HHbb_2016"
            # iclass="SvB_MA_2016"

            combineTool.py -M Impacts -d ${datacard}.root --doInitialFit \
            --setParameterRanges r${signallabel}=-10,10:rggHH_kl_0_kt_1_hbbhbb=0,0:rggHH_kl_2p45_kt_1_hbbhbb=0,0:rggHH_kl_5_kt_1_hbbhbb=0,0 \
            --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \
            --robustFit 1 -m 125 -n ${iclass} -t -1 ## expected -t -1

            combineTool.py -M Impacts -d ${datacard}.root --doFits \
            --setParameterRanges r${signallabel}=-10,10:rggHH_kl_0_kt_1_hbbhbb=0,0:rggHH_kl_2p45_kt_1_hbbhbb=0,0:rggHH_kl_5_kt_1_hbbhbb=0,0 \
            --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \
            --robustFit 1 -m 125 --parallel 4 -n ${iclass} -t -1

            combineTool.py -M Impacts -d ${datacard}.root -o impacts_combine_${iclass}_exp.json -m 125 -n ${iclass}

            plotImpacts.py -i impacts_combine_${iclass}_exp.json -o impacts_combine_${iclass}_exp_HH --POI r${signallabel} --per-page 20 --left-margin 0.3 --height 400 --label-size 0.04 --translate ${currentDir}/stats_analysis/nuisance_names.json
            mkdir -p impacts/
            mv higgsCombine*Fit* impacts/
        else
            echo "File ${datacard}.root does not exist."
        fi

    elif [ "$postfit" = true ]; then

        if [ -f "${datacard}.root" ]; then
            combine -M MultiDimFit --robustFit 1 -n _${iclass}_fit_s \
            --saveWorkspace --saveFitResult -d ${datacard}.root \
            --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0

            PostFitShapesFromWorkspace -w higgsCombine_${iclass}_fit_s.MultiDimFit.mH120.root -f multidimfit_${iclass}_fit_s.root:fit_mdf --total-shapes --output postfit_s.root --freeze r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 #--postfit

        else
            echo "File ${datacard}.root does not exist."
        fi
    fi

    cd $currentDir

done
