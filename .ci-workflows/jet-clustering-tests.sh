echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running jet clustering test"
python jet_clustering/tests/test_clustering.py 
cd ../

