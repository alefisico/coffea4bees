#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=false ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}analysis_test_job"
OUTPUT_DIR="${DEFAULT_DIR}analysis_plot_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
python plots/makePlots.py $INPUT_DIR/test.coffea --doTest -o $OUTPUT_DIR -m plots/metadata/plotsAll.yml

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


echo "############### check making the plots from yaml "
python plots/plot_from_yaml.py --input_yaml $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zz.yaml \
   $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.yaml \
   $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.yaml \
   $OUTPUT_DIR/RunII/passPreSel/fourTag/SR_vs_SB/data/SvB_MA_ps.yaml \
   $OUTPUT_DIR/RunII/passPreSel/fourTag/SR_vs_SB/HH4b/SvB_MA_ps.yaml \
   $OUTPUT_DIR/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/data/v4j_mass.yaml \
   $OUTPUT_DIR/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/HH4b/v4j_mass.yaml \
   $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/data/quadJet_min_dr_close_vs_other_m.yaml \
   $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/HH4b/quadJet_min_dr_close_vs_other_m.yaml \
   $OUTPUT_DIR/RunII/passPreSel/threeTag/SR/Multijet/quadJet_min_dr_close_vs_other_m.yaml \
   --out $OUTPUT_DIR/test_plots_from_yaml 

echo "############### Checking if pdf files exist"
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zz.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR_vs_SB/data/SvB_MA_ps.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR_vs_SB/HH4b/SvB_MA_ps.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/data/v4j_mass.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/HH4b/v4j_mass.pdf 
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/data/quadJet_min_dr_close_vs_other_m.pdf 
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/HH4b/quadJet_min_dr_close_vs_other_m.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/threeTag/SR/Multijet/quadJet_min_dr_close_vs_other_m.pdf 

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi