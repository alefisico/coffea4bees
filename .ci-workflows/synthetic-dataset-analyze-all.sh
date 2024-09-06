echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
# if [[ $(hostname) = *fnal* ]]; then
#     echo "No changing files"
# else
#     echo "############### Modifying previous dataset file (to read local files)"
#     ls -lR skimmer/
#     sed -i "s/\/builds\/algomez\/coffea4bees\/python\///" skimmer/metadata/picoaod_datasets_TTToSemiLeptonic_UL18.yml
#     cat skimmer/metadata/picoaod_datasets_TTToSemiLeptonic_UL18.yml
# fi
# echo "############### Modifying dataset file with skimmer ci output"
# cat metadata/datasets_ci.yml
#python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f skimmer/metadata/picoaod_datasets_declustered_data_Run2_seed17.yml -o metadata/datasets_synthetic_seed17.yml

# echo "############### Changing metadata"
# sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/"  analysis/metadata/HH4b.yml > analysis/metadata/tmp.yml
# cat analysis/metadata/tmp.yml
echo "############### Running test processor"
time python runner.py -o test_synthetic_data_seed17.coffea -d data -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_seed17.yml

time python runner.py -o nominal.coffea -d data -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b_subtract_tt.yml

echo "############### Running test processor HHSignal"
python metadata/merge_yaml_datasets.py -m metadata/datasets_synthetic_seed17.yml -f skimmer/metadata/picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_Run2_seed17.yml -o metadata/datasets_synthetic_seed17.yml

time python runner.py -o test_synthetic_GluGluToHHTo4B_cHHH1_seed17.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_seed17.yml
cd ../
