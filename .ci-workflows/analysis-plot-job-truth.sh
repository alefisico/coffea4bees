echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python  plots/makePlotsTruthStudy.py analysis/hists/testTruth.coffea -m plots/metadata/plotsSignal.yml --out plots/TruthStudy
echo "############### Checking if pdf files exist"
ls plots/TruthStudy/RunII/pass4GenBJets00/fourTag/SR/otherGenJet00_pt.pdf
cd ../
