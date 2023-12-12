echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python analysis/makePlots.py --doTest -i hists/test.coffea  -o testCoffeaPlots -m analysis/metadata/plotTest.yml
echo "############### Checking if pdf files exist"
ls testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zz.pdf
ls testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.pdf
ls testCoffeaPlots/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
cd ../
