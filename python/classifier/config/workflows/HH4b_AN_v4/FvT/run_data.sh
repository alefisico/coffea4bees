export LPCUSER="chuyuanl"
export WFS="classifier/config/workflows/HH4b_AN_v4/FvT"
export GMAIL=~/gmail.yml

# check port
if [ -z "$2" ]; then
    port=10200
else
    port=$2
fi

# train
./pyml.py from $WFS/train.yml \
    -template "user: ${LPCUSER}" $WFS/train_data.yml \
    -setting Monitor "address: :${port}"

# TODO
# evaluate
# ./pyml.py template "{mixed: ${i}, tag: mixed, user: ${LPCUSER}}" $WFS/evaluate_data.yml

if [ -e "$GMAIL" ]; then
    ./pyml.py analyze \
        -analysis notify.Gmail \
        --title "FvT done" \
        --body "All jobs done at $(date)" \
        --labels HH4b AN_v4 \
        -from $GMAIL \
        -setting Monitor "address: :${port}"
fi