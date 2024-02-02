echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
sed -i 's/base_path.*/base_path: $CI_PROJECT_DIR\/python\/skimmer\/rootfiles\//g' skimmer/metadata/HH4b.yml
python runner.py -s -p skimmer/processor/skimmer_4b.py -m skimmer/metadata/HH4b.yml -y UL18 -d TTTo2L2Nu -op skimmer/metadata/ -t
ls
cd ../
