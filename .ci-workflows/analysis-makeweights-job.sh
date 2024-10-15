echo "############### Moving to python folder"
cd python/

INPUT_DIR="output/analysis_test_job"
OUTPUT_DIR="output/analysis_makeweights_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running makeweights test"
python analysis/make_weights.py -o $OUTPUT_DIR/testJCM_ROOT   -c passPreSel -r SB --ROOTInputs --i analysis/tests/HistsFromROOTFile.coffea
python analysis/make_weights.py -o $OUTPUT_DIR/testJCM_Coffea -c passPreSel -r SB -i $INPUT_DIR/test.coffea
python analysis/tests/make_weights_test.py
cd ../

