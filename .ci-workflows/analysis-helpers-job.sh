echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running makeweights test"
python analysis/tests/topCand_test.py
cd ../

