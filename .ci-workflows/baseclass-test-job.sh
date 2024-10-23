echo "############### Moving to python folder"
cd python/

INPUT_DIR="output/analysis_test_job"
OUTPUT_DIR="output/baseclass_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
echo "############### Running base class test"
python base_class/tests/plots_test.py --inputFile $INPUT_DIR/test.coffea --known base_class/tests/known_PlotCounts.yml
python base_class/tests/dumpPlotCounts.py --input $INPUT_DIR/test.coffea --output $OUTPUT_DIR/test_dumpPlotCounts.yml
ls $OUTPUT_DIR/test_dumpPlotCounts.yml
cd ../

