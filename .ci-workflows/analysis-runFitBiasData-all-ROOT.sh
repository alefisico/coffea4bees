# In coffea envi
echo "############### Moving to python folder"
cd python/


#  In hist environment
# 

python stats_analysis/convert_hist_to_json.py --input analysis/hists/histAll.coffea --output analysis/hists/histAll.json

## In root envirornment
#

python3 stats_analysis/convert_json_to_root.py -f analysis/hists/histAll.json                  --output analysis/hists/

#
# Make the input with
#
#  python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/tests --skip_closure

#
# Test it with
#

python3 stats_analysis/runFitBiasData.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/fitBiasData/ULHH --bkg_syst_file stats_analysis/closureFits/ULHH/3bDvTMix4bDvT/SvB_MA/rebin8/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin8.pk


cd ../
