echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python  analysis/makePlotsTruthStudy.py analysis/hists/testTruth.coffea -m analysis/metadata/plotsSignal.yml --out analysis/TruthStudy
echo "############### Checking if pdf files exist"
ls analysis/TruthStudy/RunII/pass4GenBJets00/fourTag/SR/otherGenJet00_pt.pdf
cd ../
