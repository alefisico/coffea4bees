echo "############### Moving to python folder"
cd python/


OUTPUT_DIR="output/memory_test"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
echo "############### Running memory test"
python base_class/tests/memory_test.py --threshold 3689.422 -o $OUTPUT_DIR/mprofile_ci_test --script runner.py -o test
.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op local_outputs/analysis/
ls $OUTPUT_DIR/mprofile_ci_test.png
cd ../

