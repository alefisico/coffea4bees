echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"

time python runner.py -o synthetic_data_RunII_seedXXX.coffea -d synthetic_data data -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op hists/ -c analysis/metadata/HH4b_run_fastTopReco.yml -m metadata/datasets_HH4b_fourTag.yml
#time python runner.py -o test_synthetic_data_seedXXX_hTRW.coffea -d synthetic_data  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b_run_fastTopReco.yml -m metadata/datasets_HH4b_fourTag.yml

time python runner.py -o synthetic_data_Run3_seedXXX.coffea -d synthetic_data data -p analysis/processors/processor_HH4b.py -y 2022_preEE 2022_EE 2023_preBPix 2023_BPix  -op hists -c analysis/metadata/HH4b_run_fastTopReco.yml -m metadata/datasets_HH4b_Run3_fourTag.yml



#time python runner.py -o histData.coffea  -d data  -p analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op analysis/hists/ -c analysis/metadata/HH4b_run_fastTopReco.yml -m metadata/datasets_HH4b_fourTag.yml

# time python runner.py -o histAll_bkg.coffea            -d TTToHadronic TTToSemiLeptonic TTTo2L2Nu data                         -p analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op analysis/hists/ -c analysis/metadata/HH4b_run_fastTopReco.yml
#time python runner.py -o histAll_bkg.coffea            -d TTToHadronic TTToSemiLeptonic TTTo2L2Nu data                         -p analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op analysis/hists/ -c analysis/metadata/HH4b_run_slowTopReco.yml


#time python runner.py -o test_synthetic_data_seedXXX_noPSData.coffea -d synthetic_data  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b_run_fastTopReco.yml -m metadata/datasets_HH4b_fourTag.yml
#time python runner.py -o nominal_noTT.coffea -d data -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b_subtract_tt.yml 



## echo "############### Running test processor HHSignal"
## 
## time python runner.py -o test_synthetic_GluGluToHHTo4B_cHHH1.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_seed17.yml
## 
## time python runner.py -o nominal_GluGluToHHTo4B_cHHH1.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -c analysis/metadata/HH4b.yml
cd ../
