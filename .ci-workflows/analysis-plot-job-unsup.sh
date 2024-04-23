echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python analysis/makePlots_unsup.py analysis/hists/test_unsup.coffea --doTest   -o analysis/testCoffeaPlots_unsup/ -m analysis/metadata/plotsAll_unsup.yml 
echo "############### Checking if pdf files exist"
ls analysis/testCoffeaPlots_unsup/RunII/passPreSel/fourTag/SR/v4j_mass.pdf
ls analysis/testCoffeaPlots_unsup/RunII/passPreSel/fourTag/SR_vs_SB/mix_v0/v4j_mass.pdf
ls analysis/testCoffeaPlots_unsup/RunII/passPreSel/fourTag/SR/mix_v0/quadJet_selected_lead_vs_subl_m.pdf 
ls analysis/testCoffeaPlots_unsup/RunII/passPreSel/threeTag/SR/data_3b_for_mixed/quadJet_selected_lead_vs_subl_m.pdf 
cd ../
