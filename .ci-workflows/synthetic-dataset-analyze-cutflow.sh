echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
INPUT_DIR="output/synthetic_dataset_analyze"
OUTPUT_DIR="output/synthetic_dataset_analyze_cutflow"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test_synthetic_datasets.coffea --knownCounts analysis/tests/known_counts_test_synthetic_datasets.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test_synthetic_datasets.coffea -o $OUTPUT_DIR/test_dump_cutflow_synthetic_datasets.yml
ls $OUTPUT_DIR/test_dump_cutflow_synthetic_datasets.yml
cd ../

