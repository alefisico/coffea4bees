datacard_folder=$1

for iclass in SvB_MA;
do
    text2workspace.py ${datacard_folder}/combine_${iclass}.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/HH:rHH[1,-10,10]' #--PO 'map=.*/ZH:rZH[1,-10,10]' --PO 'map=.*/ZZ:rZZ[1,-10,10]'
    combine -M AsymptoticLimits ${datacard_folder}/combine_${iclass}.root --redefineSignalPOIs rHH -n _${iclass} > ${datacard_folder}/limits.txt

    if [ $# -eq 2 ]
      then
        combineTool.py -M Impacts -d ${datacard_folder}/combine_${iclass}.root --doInitialFit --setParameterRanges rHH=-10,10 --setParameters rHH=1 --robustFit 1 -m 125 -n ${iclass} -t -1 ## expected -t -1

        combineTool.py -M Impacts -d ${datacard_folder}/combine_${iclass}.root --doFits --setParameterRanges rHH=-10,10 --setParameters rHH=1 --robustFit 1 -m 125 --parallel 4 -n ${iclass} -t -1

        combineTool.py -M Impacts -d ${datacard_folder}/combine_${iclass}.root -o ${datacard_folder}/impacts_combine_${iclass}_exp.json -m 125 -n ${iclass}

        plotImpacts.py -i ${datacard_folder}/impacts_combine_${iclass}_exp.json -o ${datacard_folder}/impacts_combine_${iclass}_exp_HH --POI rHH --per-page 20 --left-margin 0.3 --height 400 --label-size 0.04 --translate nuisance_names.json

    fi
    mv higgsCombine_*  ${datacard_folder}/
done
