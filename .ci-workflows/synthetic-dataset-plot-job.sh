echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python  analysis/makePlotsSyntheticDatasets.py analysis/hists/test_synthetic_datasets.coffea  --out analysis/plots_synthetic_datasets
echo "############### Checking if pdf files exist"
#ls analysis/TruthStudy/RunII/pass4GenBJets00/fourTag/SR/otherGenJet00_pt.pdf
cd ../
