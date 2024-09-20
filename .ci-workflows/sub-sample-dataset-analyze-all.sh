echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/

# echo "############### Changing metadata"
# sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/"  analysis/metadata/HH4b.yml > analysis/metadata/tmp.yml
# cat analysis/metadata/tmp.yml
#echo "############### Running test processor"
#time python runner.py -o test_synthetic_data_test.coffea -d data -p analysis/processors/processor_HH4b.py -y UL18  -op analysis/hists/ -c analysis/metadata/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_seed17.yml


echo "############### Running test processor "
# python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f skimmer/metadata/picoaod_datasets_TT_pseudodata_Run2.yml  -o metadata/datasets_TT_pseudodata_Run2.yml
#cat metadata/datasets_synthetic_test.yml
time python runner.py -o TT_pseudodata_datasets.coffea -d ps_data_TTToSemiLeptonic ps_data_TTTo2L2Nu ps_data_TTToHadronic -p analysis/processors/processor_HH4b.py -y UL18 UL17 UL16_preVFP UL16_postVFP  -op analysis/hists/ -c analysis/metadata/HH4b_rerun_SvB.yml -m metadata/datasets_TT_pseudodata_Run2.yml
#time python runner.py -o histAll_TT.coffea            -d TTToHadronic TTToSemiLeptonic TTTo2L2Nu                         -p analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op analysis/hists/ -c analysis/metadata/HH4b_rerun_SvB.yml 
cd ../
