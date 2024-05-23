echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Modifying config"
sed -e "s/condor_cores.*/condor_cores: 6/" -e "s/condor_memory.*/condor_memory: 8GB/" -i analysis/metadata/HH4b.yml
cat analysis/metadata/HH4b.yml
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
fi
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py  -o histMixedBkg_TT.coffea -d   TTTo2L2Nu_for_mixed TTToHadronic_for_mixed TTToSemiLeptonic_for_mixed   -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op analysis/hists/ -m $DATASETS
python runner.py  -o histMixedBkg_data_3b_for_mixed.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op analysis/hists/ -m $DATASETS
python runner.py  -o histMixedData.coffea -d    mixeddata  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op analysis/hists/ -m $DATASETS
python runner.py  -o histSignal.coffea -d    GluGluToHHTo4B_cHHH1 ZH4b ZZ4b  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op analysis/hists/ -m $DATASETS
ls

echo "############### Hist --> JSON"

python stats_analysis/convert_hist_to_json_closure.py --input analysis/hists/histMixedData.coffea
python stats_analysis/convert_hist_to_json_closure.py --input analysis/hists/histMixedBkg_TT.coffea
python stats_analysis/convert_hist_to_json_closure.py --input analysis/hists/histMixedBkg_data_3b_for_mixed.coffea
python stats_analysis/convert_hist_to_json_closure.py --input analysis/hists/histSignal.coffea


cd ../
