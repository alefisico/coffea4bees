
# In coffea envi
echo "############### Moving to python folder"
cd python/

## In root envirornment
#
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/histmixeddata_mixeddata-UL16_preVFP.yml --output analysis/hists/
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/histmixeddata_mixeddata-UL17.yml        --output analysis/hists/
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/histmixeddata_mixeddata-UL18.yml        --output analysis/hists/
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/histmixeddata_TTbar_3b_for_mixed.yml    --output analysis/hists/
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/histmixeddata_data_3b_for_mixed.yml     --output analysis/hists/
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/histAll.yml                             --output analysis/hists/



#
# Make the input with 
#
#  python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/tests --skip_closure

#
# Test it with
#
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/tests --reuse_inputs --do_CI
python3 stats_analysis/tests/test_runTwoStageClosure.py --knownCounts stats_analysis/tests/twoStageClosure_counts_SvB_MA_ps_hh_rebin20.yml --output_path stats_analysis/tests/


python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zz  --rebin 8 --outputPath stats_analysis/tests --reuse_inputs --do_CI
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zh  --rebin 5 --outputPath stats_analysis/tests --reuse_inputs --do_CI

cd ../
