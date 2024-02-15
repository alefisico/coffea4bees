echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python analysis/makePlots.py analysis/hists/test.coffea --doTest   -o analysis/testCoffeaPlots -m analysis/metadata/plotsAll.yml
echo "############### Checking if pdf files exist"
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zz.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR_vs_SB/data/SvB_MA_ps.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR_vs_SB/HH4b/SvB_MA_ps.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/data/v4j_mass.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/HH4b/v4j_mass.pdf 
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/data/quadJet_min_dr_close_vs_other_m.pdf 
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/HH4b/quadJet_min_dr_close_vs_other_m.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/threeTag/SR/Multijet/quadJet_min_dr_close_vs_other_m.pdf 
cd ../
