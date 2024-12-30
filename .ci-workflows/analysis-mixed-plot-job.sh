echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python plots/makePlotsMixed.py analysis/hists/testMixedBkg_master.coffea analysis/hists/testMixedData_master.coffea --combine_input_files -m plots/metadata/plotsMixed.yml   -o plots/testMixedPlots
#python plots/makePlots.py analysis/hists/test.coffea    -o plots/testCoffeaPlots -m plots/metadata/plotsAll.yml
echo "############### Checking if pdf files exist"
ls plots/testMixedPlots/RunII/passPreSel/fourTag/SR/
#ls plots/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.pdf
#ls plots/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
#ls plots/testCoffeaPlots/RunII/passPreSel/fourTag/SR_vs_SB/data/SvB_MA_ps.pdf
#ls plots/testCoffeaPlots/RunII/passPreSel/fourTag/SR_vs_SB/HH4b/SvB_MA_ps.pdf
#ls plots/testCoffeaPlots/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/data/v4j_mass.pdf
#ls plots/testCoffeaPlots/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/HH4b/v4j_mass.pdf 
#ls plots/testCoffeaPlots/RunII/passPreSel/fourTag/SR/data/quadJet_min_dr_close_vs_other_m.pdf 
#ls plots/testCoffeaPlots/RunII/passPreSel/fourTag/SR/HH4b/quadJet_min_dr_close_vs_other_m.pdf
#ls plots/testCoffeaPlots/RunII/passPreSel/threeTag/SR/Multijet/quadJet_min_dr_close_vs_other_m.pdf 
cd ../
