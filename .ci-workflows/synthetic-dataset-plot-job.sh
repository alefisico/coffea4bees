echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python  jet_clustering/make_jet_splitting_PDFs.py analysis/hists/test_synthetic_datasets.coffea  --out jet_clustering/jet-splitting-PDFs-test
echo "############### Checking if pdf files exist"
ls jet_clustering/jet-splitting-PDFs-test/clustering_pdfs_vs_pT.yml 
ls jet_clustering/jet-splitting-PDFs-test/test_sampling_pt_1b0j_1b0j_mA.pdf 
cd ../
