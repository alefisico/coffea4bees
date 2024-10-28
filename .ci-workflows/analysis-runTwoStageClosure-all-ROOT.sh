# In coffea envi
echo "############### Moving to python folder"
cd python/

#
# In root envirornment
#
python3 stats_analysis/convert_json_to_root.py -f hists/histMixedData.json                         --output hists/
python3 stats_analysis/convert_json_to_root.py -f hists/histMixedBkg_TT.json                       --output hists/
python3 stats_analysis/convert_json_to_root.py -f hists/histMixedBkg_data_3b_for_mixed.json        --output hists/
python3 stats_analysis/convert_json_to_root.py -f hists/histMixedBkg_data_3b_for_mixed_kfold.json  --output hists/
python3 stats_analysis/convert_json_to_root.py -f hists/histMixedBkg_data_3b_for_mixed_ZZinSB.json --output hists/
python3 stats_analysis/convert_json_to_root.py -f hists/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.json --output hists/
python3 stats_analysis/convert_json_to_root.py -f hists/histSignal.json                            --output hists/



#
# Make the input with
#
#  python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/tests --skip_closure

#
# Test it with
#
# python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/closureFits/closureFixTrig
# python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zh  --rebin 10 --outputPath stats_analysis/closureFits/closureFixTrig
# python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zz  --rebin  8 --outputPath stats_analysis/closureFits/closureFixTrig

python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/closureFits/ULHH
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 10 --outputPath stats_analysis/closureFits/ULHH
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 8 --outputPath stats_analysis/closureFits/ULHH
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 5 --outputPath stats_analysis/closureFits/ULHH
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 4 --outputPath stats_analysis/closureFits/ULHH
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 2 --outputPath stats_analysis/closureFits/ULHH
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 1 --outputPath stats_analysis/closureFits/ULHH


python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/closureFits/ULHH_kfold --use_kfold --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_kfold.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 10 --outputPath stats_analysis/closureFits/ULHH_kfold --use_kfold --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_kfold.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 8  --outputPath stats_analysis/closureFits/ULHH_kfold --use_kfold --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_kfold.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 5  --outputPath stats_analysis/closureFits/ULHH_kfold --use_kfold --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_kfold.root 
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 4  --outputPath stats_analysis/closureFits/ULHH_kfold --use_kfold --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_kfold.root 
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 2  --outputPath stats_analysis/closureFits/ULHH_kfold --use_kfold --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_kfold.root 


python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/closureFits/ULHH_ZZinSB --use_ZZinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 10 --outputPath stats_analysis/closureFits/ULHH_ZZinSB --use_ZZinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 8  --outputPath stats_analysis/closureFits/ULHH_ZZinSB --use_ZZinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 5  --outputPath stats_analysis/closureFits/ULHH_ZZinSB --use_ZZinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 4  --outputPath stats_analysis/closureFits/ULHH_ZZinSB --use_ZZinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 2  --outputPath stats_analysis/closureFits/ULHH_ZZinSB --use_ZZinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZinSB.root

python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/closureFits/ULHH_ZZandZHinSB --use_ZZandZHinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 10 --outputPath stats_analysis/closureFits/ULHH_ZZandZHinSB --use_ZZandZHinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 8  --outputPath stats_analysis/closureFits/ULHH_ZZandZHinSB --use_ZZandZHinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 5  --outputPath stats_analysis/closureFits/ULHH_ZZandZHinSB --use_ZZandZHinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 4  --outputPath stats_analysis/closureFits/ULHH_ZZandZHinSB --use_ZZandZHinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root
python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 2  --outputPath stats_analysis/closureFits/ULHH_ZZandZHinSB --use_ZZandZHinSB --input_file_data3b hists/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root



#python3 stats_analysis/tests/test_runTwoStageClosure.py --knownCounts stats_analysis/tests/twoStageClosure_counts_SvB_MA_ps_hh_rebin20.yml --output_path stats_analysis/tests/


#python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zz  --rebin 8 --outputPath stats_analysis/tests --reuse_inputs --do_CI
#python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zh  --rebin 5 --outputPath stats_analysis/tests --reuse_inputs --do_CI

cd ../
