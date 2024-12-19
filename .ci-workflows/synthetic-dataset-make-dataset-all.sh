echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"

new_seed=0

# sed -e "s/declustering_rand_seed: [0-9]/declustering_rand_seed: $new_seed/" skimmer/metadata/declustering.yml > skimmer/metadata/declustering_seed_${new_seed}.yml
# cat skimmer/metadata/declustering_seed_${new_seed}.yml
# time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustering_seed_${new_seed}.yml -y UL17 UL18 UL16_preVFP UL16_postVFP -d data -op skimmer/metadata/ -o picoaod_datasets_declustered_data_Run2_seed${new_seed}.yml -m metadata/datasets_HH4b.yml   # --dask


sed -e "s/declustering_rand_seed: [0-9]/declustering_rand_seed: $new_seed/" skimmer/metadata/declustering_noTT_subtraction.yml > skimmer/metadata/declustering_noTT_subtraction_seed_${new_seed}.yml
cat skimmer/metadata/declustering_noTT_subtraction_seed_${new_seed}.yml
time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustering_noTT_subtraction_seed_${new_seed}.yml -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix -d data -op skimmer/metadata/ -o picoaod_datasets_declustered_data_Run3_v3_seed${new_seed}.yml -m metadata/datasets_HH4b_Run3_fourTag_v3.yml --condor    # --dask

# time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustering_signal.yml -y UL17 UL18 UL16_preVFP UL16_postVFP -d GluGluToHHTo4B_cHHH1 -op skimmer/metadata/ -o picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_Run2_seed17.yml -m metadata/datasets_HH4b.yml

#ls -R skimmer/
cd ../
