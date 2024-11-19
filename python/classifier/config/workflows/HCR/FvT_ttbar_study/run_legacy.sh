export LPCUSER="chuyuanl"
export CERNUSER="c/chuyuan"
export WFS="classifier/config/workflows/HCR/FvT_ttbar_study"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b"
export MODEL="${BASE}/classifier/FvT/nott/legacy/"
export WEB="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/FvT_ttbar_study/legacy/"

# check first argument
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi


# train mixed and make plots
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, offset: 0, user: ${LPCUSER}, tag: mixed}" $WFS/train_legacy.yml -setting Monitor "address: :${port}"
    ./pyml.py analyze --results ${MODEL}/mixed-${i}/result.json -analysis HCR.LossROC -setting IO "output: ${WEB}" -setting IO "report: mixed-${i}" -setting Monitor "address: :${port}"
done
# evaluate
for i in {0..14}
do
    ./pyml.py template "{mixed: ${i}, tag: mixed, user: ${LPCUSER}}" $WFS/evaluate_legacy.yml -setting Monitor "address: :${port}"
done