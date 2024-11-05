echo "############### Moving to python folder"
cd python/

INPUT_DIR="output/analysis_test_job"

echo "############### Running iPlot test"
python analysis/tests/iPlot_test.py --inputFile $INPUT_DIR/test.coffea
cd ../
