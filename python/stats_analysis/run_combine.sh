#!/bin/bash

# Function to handle arument parsing
parse_arguments() {

  # Check if folder arument is provided
  if [ -z "$1" ]; then
    echo "Missing folder arument"
  fi
  datacard_folder="$1"

  # Set defaults for flags
  limits=false
  impacts=false
  postfit=false
  likelihoodscan=false
  unblind=false
  gof=false

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
      --likelihoodscan)
        likelihoodscan=true
        shift
        ;;
      --unblind)
        unblind=true
        shift
        ;;
      --gof)
        gof=true
        shift
        ;;
      *)
        echo "Invalid argument: '$2'"
        ;;
    esac
  done
}
# Parse aruments
parse_arguments "$@"

echo "Running combine script with arguments: $@"

if [ "$unblind" = true ]; then
    echo "Running in unblind mode"
    blind_label="_unblinded"
    limit_blind=""
    asymov_data=""
else
    echo "Running in blind mode"
    blind_label=""
    limit_blind="--run blind"
    asymov_data="-t -1"
fi

currentDir=$PWD
signallabel="ggHH_kl_1_kt_1_hbbhbb"
# signallabel="ZZ4b"
# signallabel="ZH4b"

run_limits() {
  local datacard=$1
  local signallabel=$2
  local iclass=$3

    text2workspace.py ${datacard}.txt \
        -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
        --PO "map=.*/${signallabel}:r${signallabel}[1,-10,10]" \
        --PO "map=.*/ggHH_kl_0_kt_1_hbbhbb:rggHH_kl_0_kt_1_hbbhbb[1,-10,10]" \
        --PO "map=.*/ggHH_kl_2p45_kt_1_hbbhbb:rggHH_kl_2p45_kt_1_hbbhbb[1,-10,10]" \
        --PO "map=.*/ggHH_kl_5_kt_1_hbbhbb:rggHH_kl_5_kt_1_hbbhbb[1,-10,10]" \
        # --PO 'map=.*/ZH4b:rZH4b[1,-10,10]' \
        # --PO 'map=.*/ZZ4b:rZZ4b[1,-10,10]'
        
    combine -M AsymptoticLimits ${datacard}.root --redefineSignalPOIs r${signallabel} \
        -n _${iclass}${blind_label} ${limit_blind} \
        --setParameters rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0,rZZ4b=0,rZH4b=0 \
        --freezeParameters rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb,rZZ4b,rZH4b \
        > limits_${datacard}_${iclass}${blind_label}.txt
    cat limits_${datacard}_${iclass}${blind_label}.txt
    combineTool.py -M CollectLimits higgsCombine_${iclass}${blind_label}.AsymptoticLimits.mH120.root -o limits_${datacard}_${iclass}${blind_label}.json

    combine -M Significance ${datacard}.root --redefineSignalPOIs r${signallabel} \
        -n _${iclass}${blind_label} ${asymov_data} \
        --setParameters rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0,rZZ4b=0,rZH4b=0 \
        --freezeParameters rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb,rZZ4b,rZH4b \
        > significance_${datacard}_${iclass}${blind_label}.txt
    cat significance_${datacard}_${iclass}${blind_label}.txt

}

for iclass in SvB_MA;
do
    datacard="datacard" 
    cd ${datacard_folder}/
    
    if [ "$limits" = true ]; then

        datacard_label="HHbb"
        # datacard_label="ZZbb"
        # datacard_label="ZHbb"
        combineCards.py ${datacard_label}_2016=datacard_${datacard_label}_2016.txt \
            ${datacard_label}_2017=datacard_${datacard_label}_2017.txt \
            ${datacard_label}_2018=datacard_${datacard_label}_2018.txt > ${datacard}.txt
        run_limits $datacard $signallabel $iclass

        # run_limits datacard_${datacard_label}_2016 $signallabel $iclass
        # run_limits datacard_${datacard_label}_2017 $signallabel $iclass
        # run_limits datacard_${datacard_label}_2018 $signallabel $iclass

    elif [ "$impacts" = true ]; then

        if [ -f "${datacard}.root" ]; then

            # for year in _2016 _2017 _2018; do
            #     datacard="datacard_HHbb"${year}
            #     iclass="SvB_MA"${year}

                combineTool.py -M Impacts -d ${datacard}.root --doInitialFit \
                --robustFit 1 -n ${iclass} -m 125 ${asymov_data} \
                --setParameterRanges r${signallabel}=-10,10:rggHH_kl_0_kt_1_hbbhbb=0,0:rggHH_kl_2p45_kt_1_hbbhbb=0,0:rggHH_kl_5_kt_1_hbbhbb=0,0 \
                --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \

                combineTool.py -M Impacts -d ${datacard}.root --doFits \
                --robustFit 1 -m 125 --parallel 4 -n ${iclass} ${asymov_data} \
                --setParameterRanges r${signallabel}=-10,10:rggHH_kl_0_kt_1_hbbhbb=0,0:rggHH_kl_2p45_kt_1_hbbhbb=0,0:rggHH_kl_5_kt_1_hbbhbb=0,0 \
                --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \

                combineTool.py -M Impacts -d ${datacard}.root -o impacts_combine_${iclass}_exp.json -m 125 -n ${iclass}

                if [[ ! -d "${currentDir}/stats_analysis" ]]; then
                    tmpDir=${currentDir}/python/stats_analysis
                else
                    tmpDir=${currentDir}/stats_analysis
                fi
                plotImpacts.py -i impacts_combine_${iclass}_exp.json -o impacts_combine_${iclass}_exp_HH --POI r${signallabel} --per-page 20 --left-margin 0.3 --height 400 --label-size 0.04 --translate ${tmpDir}/nuisance_names.json
                mkdir -p impacts/
                mv higgsCombine*Fit* impacts/
            # done
        else
            echo "File ${datacard}.root does not exist."
        fi

    elif [ "$gof" = true ]; then

        if [ -f "${datacard}.root" ]; then

            echo "Running goodness of fit test on data"
            combine -M GoodnessOfFit ${datacard}.root --algo saturated  \
                -n _${iclass}${blind_label}_gof_data \
                --setParameters r${signallabel}=1 \
                > gof_data_${datacard}_${iclass}${blind_label}.txt
            cat gof_data_${datacard}_${iclass}${blind_label}.txt
                # --freezeParameters rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb \
                # --setParameters rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \
            
            # echo "Running goodness of fit test on toys"
            combine -M GoodnessOfFit ${datacard}.root --algo saturated  \
                -n _${iclass}${blind_label}_gof_toys --toysFrequentist -t 1000 \
                > gof_toys_${datacard}_${iclass}${blind_label}.txt
            cat gof_toys_${datacard}_${iclass}${blind_label}.txt
            #     --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \
            #     --freezeParameters r${signallabel},rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb \

            combineTool.py -M CollectGoodnessOfFit \
                --input higgsCombine_${iclass}${blind_label}_gof_data.GoodnessOfFit.mH120.root \
                higgsCombine_${iclass}${blind_label}_gof_toys.GoodnessOfFit.mH120.123456.root \
                -o gof_${datacard}_${iclass}${blind_label}.json
            
            plotGof.py gof_${datacard}_${iclass}${blind_label}.json \
                --statistic staturated --mass 120.0 \
                --output gof_${datacard}_${iclass}${blind_label}

        else
            echo "File ${datacard}.root does not exist."
        fi

    elif [ "$postfit" = true ]; then

        if [ -f "${datacard}.root" ]; then
    
            #### Need to revise this part
            echo "Running postfit b-only"
            combine -M FitDiagnostics ${datacard}.root --redefineSignalPOIs r${signallabel} \
                -n _${iclass}${blind_label}_prefit_bonly ${asymov_data} \
                --setParameters r${signallabel}=0,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0,rZZ4b=0,rZH4b=0 \
                --freezeParameters rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb,rZZ4b,rZH4b \
                > fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_bonly.txt
            cat fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_bonly.txt
            mkdir -p fitDiagnostics_bonly/
            mv *th1x* fitDiagnostics_bonly/
            mv covariance* fitDiagnostics_bonly/

            python /home/cmsusr/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
                -p r${signallabel} \
                -a fitDiagnostics_${iclass}${blind_label}_prefit_bonly.root \
                -g diffNuisances_${datacard}_${iclass}${blind_label}_prefit_bonly.root

            # combine -M MultiDimFit --robustFit 1 -n _${iclass}_fit_b \
            # --saveWorkspace --saveFitResult -d ${datacard}.root \
            # --setParameters r${signallabel}=0,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \
            # --freezeParameters r${signallabel},rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb

            # PostFitShapesFromWorkspace -w higgsCombine_${iclass}_fit_b.MultiDimFit.mH120.root -f multidimfit_${iclass}_fit_b.root:fit_mdf --total-shapes --output postfit_b.root --freeze r${signallabel}=0,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 --postfit
            
            echo "Running postfit s+b"
            combine -M FitDiagnostics ${datacard}.root --redefineSignalPOIs r${signallabel} \
                -n _${iclass}${blind_label}_prefit_sb ${asymov_data} --saveShapes --plots \
                --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0,rZZ4b=0,rZH4b=0 \
                --freezeParameters rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb,rZZ4b,rZH4b \
                > fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_sb.txt
            cat fitDiagnostics_${datacard}_${iclass}${blind_label}_prefit_sb.txt
            mkdir -p fitDiagnostics_sb/
            mv *th1x* fitDiagnostics_sb/
            mv covariance* fitDiagnostics_sb/

            python /home/cmsusr/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
                -p r${signallabel} \
                -a fitDiagnostics_${iclass}${blind_label}_prefit_sb.root \
                -g diffNuisances_${datacard}_${iclass}${blind_label}_prefit_sb.root

            # combine -M MultiDimFit --robustFit 1 -n _${iclass}_fit_s \
            #     --saveWorkspace --saveFitResult -d ${datacard}.root \
            #     --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 \
            #     --freezeParameters rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb 

            # PostFitShapesFromWorkspace -w higgsCombine_${iclass}_fit_s.MultiDimFit.mH120.root \
            #     -f multidimfit_${iclass}_fit_s.root:fit_mdf \
            #     --total-shapes --output postfit_s.root \
            #     --freeze r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0 --postfit

        else
            echo "File ${datacard}.root does not exist."
        fi

    elif [ "$likelihoodscan" = true ]; then

        if [ -f "${datacard}.root" ]; then

            combine -M MultiDimFit -n _${iclass}_likelihoodscan_postfit \
                --saveWorkspace -d ${datacard}.root --robustFit 1 ${asymov_data} \
                --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0,rZZ4b=0,rZH4b=0 \
                --freezeParameters  r${signallabel},rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb,rZZ4b,rZH4b

            combine -M MultiDimFit -n _${iclass}_likelihoodscan_freeze_all \
                -P r${signallabel} ${asymov_data} --snapshotName MultiDimFit \
                --rMin -10 --rMax 10 --algo grid --points 50 --alignEdges 1 \
                --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0,rZZ4b=0,rZH4b=0 \
                --freezeParameters r${signallabel},rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb,rZZ4b,rZH4b,allConstrainedNuisances \
                -d higgsCombine_${iclass}_likelihoodscan_postfit.MultiDimFit.mH120.root

            scan_cmd="combine -M MultiDimFit \
                -P r${signallabel} ${asymov_data} --snapshotName MultiDimFit \
                --rMin -10 --rMax 10 --algo grid --points 50 --alignEdges 1 \
                --setParameters r${signallabel}=1,rggHH_kl_0_kt_1_hbbhbb=0,rggHH_kl_2p45_kt_1_hbbhbb=0,rggHH_kl_5_kt_1_hbbhbb=0,rZZ4b=0,rZH4b=0 \
                --freezeParameters r${signallabel},rggHH_kl_0_kt_1_hbbhbb,rggHH_kl_2p45_kt_1_hbbhbb,rggHH_kl_5_kt_1_hbbhbb,rZZ4b,rZH4b \
                -d higgsCombine_${iclass}_likelihoodscan_postfit.MultiDimFit.mH120.root"

            ${scan_cmd} -n _${iclass}_likelihoodscan_total
            ${scan_cmd} -n _${iclass}_likelihoodscan_freeze_multijet --freezeNuisanceGroups multijet
            ${scan_cmd} -n _${iclass}_likelihoodscan_freeze_btag --freezeNuisanceGroups multijet,btag

            plot1DScan.py higgsCombine_${iclass}_likelihoodscan_total.MultiDimFit.mH120.root \
                --main-label "Total uncert." --others \
                higgsCombine_${iclass}_likelihoodscan_freeze_multijet.MultiDimFit.mH120.root:"Multijet":2 \
                higgsCombine_${iclass}_likelihoodscan_freeze_btag.MultiDimFit.mH120.root:"b-tagging":4 \
                higgsCombine_${iclass}_likelihoodscan_freeze_all.MultiDimFit.mH120.root:"Stat. only":3 \
                --breakdown "multijet,btag,others,stat" \
                --POI r${signallabel} -o likelihoodscan_${iclass}_breakdown

        else
            echo "File ${datacard}.root does not exist."
        fi
    fi

    cd $currentDir

done
