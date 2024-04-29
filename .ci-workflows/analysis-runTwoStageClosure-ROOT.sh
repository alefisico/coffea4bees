
# In coffea envi
echo "############### Moving to python folder"
cd python/


## In root envirornment
#
python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/testMixedData.yml                  --output analysis/hists/
python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/testMixedBkg_TT.yml                --output analysis/hists/
python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/testMixedBkg_data_3b_for_mixed.yml --output analysis/hists/
python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/testSignal.yml                     --output analysis/hists/



#
# Test it with the 
#
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/testsLocal  --do_CI \
    --input_file_data3b analysis/hists/testMixedBkg_data_3b_for_mixed.root \
    --input_file_TT     analysis/hists/testMixedBkg_TT.root \
    --input_file_mix    analysis/hists/testMixedData.root \
    --input_file_sig    analysis/hists/testSignal.root



#
# Test it with full inputs
#
#

#
# Make the input with 
#
#  python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/tests --skip_closure

python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/tests --reuse_inputs --do_CI
python3 stats_analysis/tests/test_runTwoStageClosure.py --knownCounts stats_analysis/tests/known_twoStageClosure_counts_SvB_MA_ps_hh_rebin20.yml --output_path stats_analysis/tests/

python3 stats_analysis/tests/dumpTwoStageInputs.py --input stats_analysis/tests/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin20.root  --output stats_analysis/tests/test_dump_twoStageClosureInputsCounts.yml



cd ../
