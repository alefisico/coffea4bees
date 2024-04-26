
# In coffea envi
echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/

echo "############### Merging"
python analysis/merge_coffea_files.py -f \
    analysis/hists/histmixeddata_data_3b_for_mixed-UL16_preVFP.coffea \
    analysis/hists/histmixeddata_data_3b_for_mixed-UL17.coffea \
    analysis/hists/histmixeddata_data_3b_for_mixed-UL18.coffea \
    -o analysis/hists/histmixeddata_data_3b_for_mixed.coffea

python analysis/merge_coffea_files.py -f \
    analysis/hists/histmixed_TTTo2L2Nu_for_mixed-UL16_preVFP.coffea \
    analysis/hists/histmixed_TTTo2L2Nu_for_mixed-UL16_postVFP.coffea \
    analysis/hists/histmixed_TTTo2L2Nu_for_mixed-UL17.coffea \
    analysis/hists/histmixed_TTTo2L2Nu_for_mixed-UL18.coffea \
    analysis/hists/histmixed_TTToSemiLeptonic_for_mixed-UL16_preVFP.coffea \
    analysis/hists/histmixed_TTToSemiLeptonic_for_mixed-UL16_postVFP.coffea \
    analysis/hists/histmixed_TTToSemiLeptonic_for_mixed-UL17.coffea \
    analysis/hists/histmixed_TTToSemiLeptonic_for_mixed-UL18.coffea \
    analysis/hists/histmixed_TTToHadronic_for_mixed-UL16_preVFP.coffea \
    analysis/hists/histmixed_TTToHadronic_for_mixed-UL16_postVFP.coffea \
    analysis/hists/histmixed_TTToHadronic_for_mixed-UL17.coffea \
    analysis/hists/histmixed_TTToHadronic_for_mixed-UL18.coffea \
    -o analysis/hists/histmixeddata_TTbar_3b_for_mixed.coffea

echo "############### Hist --> YML"

python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/histmixeddata_mixeddata-UL16_preVFP.coffea
python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/histmixeddata_mixeddata-UL17.coffea
python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/histmixeddata_mixeddata-UL18.coffea
python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/histmixeddata_TTbar_3b_for_mixed.coffea
python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/histmixeddata_data_3b_for_mixed.coffea
python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/histAll.coffea

