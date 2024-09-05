echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running trigger emulator test"
python -m unittest base_class.tests.test_trigger_emulator
cd ../

