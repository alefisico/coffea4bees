echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python analysis/makePlots.py --doTest -i analysis/hists/test.coffea  -o analysis/testCoffeaPlots -m analysis/metadata/plotsAll.yml
echo "############### Checking if pdf files exist"
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zz.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.pdf
ls analysis/testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
cd ../
