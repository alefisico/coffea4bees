echo "############### Moving to python folder"
cd python/

INPUT_DIR="output/analysis_test_job_unsup"
OUTPUT_DIR="output/analysis_plot_job_unsup"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
python analysis/makePlots_unsup.py $INPUT_DIR/test_unsup.coffea --doTest   -o $OUTPUT_DIR/ -m analysis/metadata/plotsAll_unsup.yml 
echo "############### Checking if pdf files exist"
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/mix_v0/v4j_mass.pdf
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR_vs_SB/mix_v0/v4j_mass.pdf
ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/mix_v0/quadJet_selected_lead_vs_subl_m.pdf 
ls $OUTPUT_DIR/RunII/passPreSel/threeTag/SR/data_3b_for_mixed/quadJet_selected_lead_vs_subl_m.pdf 
cd ../
