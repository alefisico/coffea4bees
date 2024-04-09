echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
else
    echo "############### Modifying corrections for ci"
    sed -e "/Absolute_/d" -e "/BBEC1/d" -e "/EC2/d" -e "/- HF/d" -e "/- Relative/d" -e "/- hf/d" -e "/- lf/d" analysis/metadata/corrections.yml > analysis/metadata/corrections_ci.yml
    cat analysis/metadata/corrections_ci.yml
    echo "############### Modifying config"
    sed -e "s/corrections\.yml/corrections_ci\.yml/" -i analysis/metadata/HH4b_systematics.yml
    cat analysis/metadata/HH4b.yml
    DATASETS=metadata/datasets_HH4b_cernbox.yml
fi
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o test_systematics.coffea -d GluGluToHHTo4B_cHHH1 ZH4b ZZ4b -p analysis/processors/processor_HH4b.py -y UL18 -op analysis/hists/ -m $DATASETS -c analysis/metadata/HH4b_systematics.yml
ls
cd ../
