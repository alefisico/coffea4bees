echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=analysis/metadata/unsup4b.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
fi
echo "############### Running datasets from " $DATASETS
echo "############### Running unsup test processor"
python runner.py -t -o test_unsup.coffea -d mixeddata data_3b_for_mixed TTToHadronic TTToSemiLeptonic TTTo2L2Nu -p analysis/processors/processor_unsup.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/unsup4b.yml -m $DATASETS
ls
cd ../
