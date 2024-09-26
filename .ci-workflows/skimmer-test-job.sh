echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/python\/skimmer\/test\/#" -e "s/\#max.*/maxchunks: 5/" -e "s/\#test.*/test_files: 1/" -e "s/2024_.*/tmp\//" skimmer/metadata/HH4b.yml > skimmer/metadata/tmp.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/skimmer\/#" -e "s/\#max.*/maxchunks: 5/" -e "s/\#test.*/test_files: 1/" -e "s/2024_.*/tmp\//" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/HH4b.yml > skimmer/metadata/tmp.yml
fi
cat skimmer/metadata/tmp.yml
#nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/92D0BDF3-91AE-514F-88B5-8F591450B8AD.root"
#sed -e "s#/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" metadata/datasets_HH4b.yml > metadata/datasets_ci.yml
#python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/tmp.yml -y UL18 -d TTToSemiLeptonic -op skimmer/metadata/ -o picoaod_datasets_TTToSemiLeptonic_UL18.yml -m metadata/datasets_ci.yml  -t 
nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/3F95108D-84D2-CD4D-A0D2-324A7D15E691.root"
sed -e "s#/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" metadata/datasets_HH4b.yml > metadata/datasets_ci.yml
echo "############### Running test processor"
python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/tmp.yml -y UL18 -d   GluGluToHHTo4B_cHHH0 -op skimmer/metadata/ -o picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml -m metadata/datasets_ci.yml  -t 
ls -R skimmer/
cd ../
