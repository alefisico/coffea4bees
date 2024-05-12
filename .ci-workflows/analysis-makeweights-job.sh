echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running makeweights test"
python analysis/make_weights.py -o analysis/testJCM_ROOT   -c passPreSel -r SB --ROOTInputs --i analysis/tests/HistsFromROOTFile.coffea
python analysis/make_weights.py -o analysis/testJCM_Coffea -c passPreSel -r SB -i analysis/hists/test.coffea
python analysis/tests/make_weights_test.py
cd ../

