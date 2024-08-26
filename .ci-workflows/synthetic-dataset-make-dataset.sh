echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
#echo "############### Changing metadata"
#if [[ $(hostname) = *fnal* ]]; then
#    sed -e "s#base_path.*#base_path: \/srv\/python\/skimmer\/test\/#" -e "s/\#max.*/maxchunks: 5/" -e "s/\#test.*/test_files: 1/" -e "s/2024_.*/tmp\//" skimmer/metadata/HH4b.yml > skimmer/metadata/tmp.yml
#else
#    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/skimmer\/#" -e "s/\#max.*/maxchunks: 5/" -e "s/\#test.*/test_files: 1/" -e "s/2024_.*/tmp\//" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/HH4b.yml > skimmer/metadata/tmp.yml
#fi
#cat skimmer/metadata/tmp.yml
#sed -e "s#/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM#metadata/nano_TTToSemiLeptonic_ci.txt#" metadata/datasets_HH4b.yml > metadata/datasets_ci.yml
echo "############### Running test processor"
python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustereing.yml -y UL18 -d data -op skimmer/metadata/ -o picoaod_datasets_declustered_data_UL18.yml -m metadata/datasets_HH4b.yml  -t # --dask
ls -R skimmer/
cd ../
