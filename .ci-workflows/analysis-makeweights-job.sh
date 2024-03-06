echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running makeweights test"
python analysis/tests/make_weights_test.py 
cd ../

