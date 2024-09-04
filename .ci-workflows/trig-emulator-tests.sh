echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running trigger emulator test"
python base_analysis/tests/test_trigger_emulator.py
cd ../

