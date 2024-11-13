##### Run 2
if [ -z "$1" ]; then
    echo "No folder provided. Please provide the folder containing the datacards."
else
    cd $1
fi

run_combination() {
    local datacard=$1
    local NAME=$2

    text2workspace.py ${datacard}.txt 

    combine -M AsymptoticLimits ${datacard}.root -n _run2${NAME} --run blind
    combineTool.py -M CollectLimits higgsCombine_run2${NAME}.AsymptoticLimits.mH120.root -o limits_run2${NAME}.json
}

echo $PWD
echo "THIS ASSUMES THAT THE DATACARDS HAVE ONLY THE SM HIGGS SIGNALS (REMOVED BY INFERENCE TOOL)"

combineCards.py -s resolved=datacard_resolved.txt > datacard_resolved_stat_only.txt
run_combination datacard_resolved_stat_only _resolved_stat_only
run_combination datacard_resolved _resolved

combineCards.py -s boosted=datacard_boosted.txt > datacard_boosted_stat_only.txt
run_combination datacard_boosted_stat_only _boosted_stat_only
run_combination datacard_boosted _boosted

combineCards.py -s boosted=datacard_boosted.txt resolved=datacard_resolved.txt > datacard_combination_stat_only.txt 
run_combination datacard_combination_stat_only _stat_only

combineCards.py boosted=datacard_boosted.txt resolved=datacard_resolved.txt > datacard_combination.txt 
run_combination datacard_combination  

cd -