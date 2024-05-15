
# In coffea envi
echo "############### Moving to python folder"
cd python/


## In root envirornment
#
echo "############### Convert json to root"
python3 stats_analysis/convert_json_to_root.py -f analysis/hists/testMixedData.json                  --output analysis/hists/
python3 stats_analysis/convert_json_to_root.py -f analysis/hists/testMixedBkg_TT.json                --output analysis/hists/
python3 stats_analysis/convert_json_to_root.py -f analysis/hists/testMixedBkg_data_3b_for_mixed.json --output analysis/hists/
python3 stats_analysis/convert_json_to_root.py -f analysis/hists/testSignal_UL.json                  --output analysis/hists/
#python3 stats_analysis/convert_json_to_root.py -f analysis/hists/testSignal_preUL.json               --output analysis/hists/



#
# Test it with the
#
echo "############### Run test runTwoStageClosure"
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/testsLocal  --do_CI \
    --input_file_data3b analysis/hists/testMixedBkg_data_3b_for_mixed.root \
    --input_file_TT     analysis/hists/testMixedBkg_TT.root \
    --input_file_mix    analysis/hists/testMixedData.root \
    --input_file_sig    analysis/hists/testSignal_UL.root \
#    --input_file_sig_preUL    analysis/hists/testSignal_preUL.root


#python old_make_combine_hists.py -i ./files_HIG-20-011/hists_closure_3bDvTMix4bDvT_SR_weights_newSBDef.root -o HIG-20-011/hist_closure_SvB_MA.root --TDirectory 3bDvTMix4bDvT_v0/hh2018 --var multijet --channel hh2018 -n mj --rebin 10 --systematics ./files_HIG-20-011/closureResults_SvB_MA_hh.pkl
#python old_make_combine_hists.py -i ./files_HIG-20-011/data2018/hists_j_r.root -o HIG-20-011/hist_SvB_MA.root  -r SR --var SvB_MA_ps_hh --channel hh2018 -n mj --tag three --cut passPreSel --rebin 10 --systematics ./files_HIG-20-011/closureResults_SvB_MA_hh.pkl

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

python3 stats_analysis/tests/dumpTwoStageInputs.py --input stats_analysis/tests/3bDvTMix4bDvT/SvB_MA/rebin20/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin20.root   --output stats_analysis/tests/test_dump_twoStageClosureInputsCounts.yml


cd ../
