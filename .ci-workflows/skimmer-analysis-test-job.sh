echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Modifying dataset file with skimmer ci output"
python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f skimmer/metadata/picoaod_datasets_TTToSemiLeptonic_UL18.yml -o metadata/datasets_ci.yml
echo "############### Changing metadata"
sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/\#run_SvB.*/run_SvB: false/"  analysis/metadata/HH4b.yml > analysis/metadata/tmp.yml
cat analysis/metadata/tmp.yml
echo "############### Running test processor"
python runner.py -o test_skimmer.coffea -d TTToSemiLeptonic -p analysis/processors/processor_HH4b.py -y UL18 -op analysis/hists/ -m analysis/metadata/tmp.yml
cd ../
