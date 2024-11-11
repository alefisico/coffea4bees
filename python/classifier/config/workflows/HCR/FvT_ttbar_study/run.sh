export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/FvT_ttbar_study"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/FvT/nott"
export FRIEND="${BASE}/friend/FvT/nott"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/FvT_ttbar_study/"

# train and make plots
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, offset: 0, user: ${LPCUSER}}" $WFS/train_mixed.yml
    ./pyml.py analyze --results ${MODEL}/baseline-${i}/result.json -analysis HCR.LossROC -setting IO "output: ${WEB}" -setting IO "report: baseline-${i}"
    ./pyml.py template "{mixed: ${i}, offset: 0, user: ${LPCUSER}}" $WFS/train_mixed_nott.yml
    ./pyml.py analyze --results ${MODEL}/nott-${i}/result.json -analysis HCR.LossROC -setting IO "output: ${WEB}" -setting IO "report: nott-${i}"
done
# evaluate
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, tag: baseline, user: ${LPCUSER}}" $WFS/evaluate.yml
    ./pyml.py template "{mixed: ${i}, tag: nott, user: ${LPCUSER}}" $WFS/evaluate.yml
done
# merge friend tree metafiles
python -m analysis.tools.collect_friend_meta -i ${FRIEND}/baseline-{0..14}/result.json@@analysis.0.merged -o ${FRIEND}/baseline.json
python -m analysis.tools.collect_friend_meta -i ${FRIEND}/nott-{0..14}/result.json@@analysis.0.merged -o ${FRIEND}/nott.json