echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata file"
sed -i 's/base_path.*/base_path: python\/skimmer\/rootfiles\//g' skimmer/metadata/HH4b.yml
echo "############### Running test processor"
python runner.py -s -p skimmer/processor/skimmer_4b.py -m skimmer/metadata/HH4b.yml -y UL18 -d TTTo2L2Nu -op skimmer/metadata/ -o picoaod_datasets.yml -t
ls -R skimmer/
cd ../
