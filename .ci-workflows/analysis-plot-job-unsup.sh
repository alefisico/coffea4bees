echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python analysis/makePlots.py analysis/hists/test_unsup.coffea --doTest   -o analysis/testCoffeaPlots -m analysis/metadata/plotsAll_unsup.yml
echo "############### Checking if pdf files exist"
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/v4j_mass.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR_vs_SB/mixeddata/v4j_mass.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR_vs_SB/data_3b_for_mixed/v4j_mass.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/mixeddata/quadJet_selected_lead_vs_subl_m.pdf 
ls analysis/testCoffeaPlots/RunII/passPreSel/threeTag/SR/Multijet/quadJet_selected_lead_vs_subl_m.pdf 
cd ../
