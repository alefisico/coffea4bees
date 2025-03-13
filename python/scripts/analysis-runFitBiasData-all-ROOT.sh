#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} 

OUTPUT_DIR="${DEFAULT_DIR}/analysis_runFitBiasData_all_ROOT"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

#  In hist environment
# 

python stats_analysis/convert_hist_to_json.py --input analysis/hists/histAll.coffea --output ${OUTPUT_DIR}/histAll.json

## In root envirornment
#

python3 stats_analysis/convert_json_to_root.py -f ${OUTPUT_DIR}/histAll.json                  --output analysis/${OUTPUT_DIR}

#
# Make the input with
#
#  python3 stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/tests --skip_closure

#
# Test it with
#

#python3 stats_analysis/runFitBiasData.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath stats_analysis/fitBiasData/ULHH --bkg_syst_file stats_analysis/closureFits/ULHH/3bDvTMix4bDvT/SvB_MA/rebin8/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin8.pk
python3 stats_analysis/runFitBiasData.py  --var SvB_MA_ps_zz   --rebin 8 --outputPath stats_analysis/fitBiasData/ULHH --bkg_syst_file stats_analysis/closureFits/ULHH/3bDvTMix4bDvT/SvB_MA/rebin8/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin8.pkl  

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi