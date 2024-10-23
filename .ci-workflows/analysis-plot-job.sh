echo "############### Moving to python folder"
cd python/

INPUT_DIR="output/analysis_test_job"
OUTPUT_DIR="output/analysis_plot_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
python analysis/makePlots.py $INPUT_DIR/test.coffea --doTest -o $OUTPUT_DIR -m analysis/metadata/plotsAll.yml

echo "############### Checking if pdf files exist"
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zz.pdf
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.pdf
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR_vs_SB/data/SvB_MA_ps.pdf
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR_vs_SB/HH4b/SvB_MA_ps.pdf
ls $OUTPUT_DIR/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/data/v4j_mass.pdf
ls $OUTPUT_DIR/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/HH4b/v4j_mass.pdf 
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/data/quadJet_min_dr_close_vs_other_m.pdf 
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/HH4b/quadJet_min_dr_close_vs_other_m.pdf
ls $OUTPUT_DIR/RunII/passPreSel/threeTag/SR/Multijet/quadJet_min_dr_close_vs_other_m.pdf 
cd ../
