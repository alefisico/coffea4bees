export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/FvT_ttbar_study"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/FvT/nott"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/FvT_ttbar_study/"

# train mixed and make plots
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, offset: 0, user: ${LPCUSER}, source: 'mixed detector', option: no-detector-4b, tag: mixed}" $WFS/train.yml
    ./pyml.py analyze --results ${MODEL}/mixed-${i}/result.json -analysis HCR.LossROC -setting IO "output: ${WEB}" -setting IO "report: nott-${i}"
done
# evaluate
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, tag: mixed, user: ${LPCUSER}}" $WFS/evaluate.yml
done